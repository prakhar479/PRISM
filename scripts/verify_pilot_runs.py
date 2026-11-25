#!/usr/bin/env python
"""Are the finished runs of a queue real, complete and internally consistent? (no model, no GPU; run it on the box after a queue)

    .venv-harness/bin/python scripts/verify_pilot_runs.py --queue queue/pilot-g3-server.yaml            # runs/ in the repo
    .venv-harness/bin/python scripts/verify_pilot_runs.py --queue queue/pilot-g3-server.yaml --only P12

For every experiment config of the queue and every COMPLETE run of it under <runs>/<EXP>/<hash>/ it checks:
  provenance   status complete; not smoke, not tiny, not allow_dirty, git tree clean; the run's config equals the committed config
  coverage     every configured cell present in the run's cell checks and in per_pair.parquet; none skipped; every configured
               condition present; random-subspace nulls have the configured number of seeds at every cell
  data         per_pair.parquet passes harness.schema; no NaN / inf in logits; usable test pairs per cell (f_pilot) and train pairs
  machinery    full_A reproduces clean_A and full_B reproduces clean_B (the write-back path is the identity when everything is
               patched): median / max |delta logit-difference| per cell; the clean baselines predict their own answers
  directions   every cached DAS / diffmean / ridge direction file loads, finite training loss; v2 DAS keeps its held-out split
and, when results_server/<tag>/PUBLISHED.json exists, that it lists exactly the complete runs found (and no smoke run).
Prints OK / WARN / FAIL lines; exit code 1 when anything FAILs. Partial run directories (a stage-3 attempt that never finalized,
e.g. after a re-tag) are listed as INFO: the publisher and scripts/make_pilot.py ignore them.
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from harness.analysis import CELL
from harness.config import ExpConfig
from harness.queue import Queue
from harness.schema import validate_per_pair

MIN_TEST = 60            # usable test pairs per cell below which a cell is flagged (n_test is 150 in these pilots)
IDENTITY_TOL = 1.0       # |delta(logit_A - logit_B)| of full_A vs clean_A above which the write-back path is suspect (bf16 emulation)


class Rep:
    def __init__(self):
        self.n = {"OK": 0, "WARN": 0, "FAIL": 0, "INFO": 0}

    def __call__(self, level, text):
        self.n[level] += 1
        print(f"  [{level}] {text}")


def classify_skips(cells):
    """(capability skips, other skips). A capability skip is the harness's own gate: too few TRAIN pairs pass the behavioural
    filter to fit directions on ('only N train pairs pass f_pilot ...') or too few test pairs: the model does not do that task
    well enough, which is a result. Anything else (a dictionary that could not load or fit in memory, a crash) is a failure."""
    skips = [c for c in cells if c.get("skipped")]
    cap = [c for c in skips if re.search(r"train pairs pass|\d+ test pairs", str(c["skipped"]))]
    return cap, [c for c in skips if c not in cap]


def cell_task(name: str) -> str:
    return name.split("_L")[0]


def jsonable(x):
    return json.loads(json.dumps(x, default=str))


def cell_df(pp):
    return pp.groupby(CELL, sort=False)


def check_run(rep, cfg: ExpConfig, d: Path, m: dict, allow_smoke: bool):
    g = m.get("git", {})
    tags = ",".join(g.get("tags") or []) or "no tag"
    print(f"- {cfg.exp_id} run {m['run_hash']}  commit {str(g.get('commit'))[:10]} ({tags})  {m.get('created', '')[:16]}  "
          f"wall {m.get('wall_time_s', 0) / 3600:.1f} h (final stage only)")
    bad = [k for k in ("smoke", "tiny", "allow_dirty") if m.get(k)] + (["dirty tree"] if g.get("dirty") else [])
    if bad:
        rep("INFO" if allow_smoke else "FAIL", f"not a real run: {bad}" + (" (allowed by --allow-smoke)" if allow_smoke else ""))
    else:
        rep("OK", "real run: not smoke / tiny / allow-dirty; tree clean")
    cf = Path(m["config_file"])
    if cf.exists():
        want = jsonable(ExpConfig.load(cf).to_dict())
        diff = sorted(k for k in set(want) | set(m["config"]) if want.get(k) != m["config"].get(k))
        rep("OK" if not diff else "FAIL", "config equals the committed " + str(cf) if not diff else f"config differs from {cf}: {diff}")
    # ---- coverage
    summ = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {"cells": []}
    cells = summ["cells"]
    cap, other = classify_skips(cells)
    n_skip = len(cap) + len(other)
    rep("OK" if len(cells) == len(cfg.cells) else "FAIL", f"{len(cells)} cells checked of {len(cfg.cells)} configured")
    if cap:
        by_task = collections.Counter(cell_task(c["cell"]) for c in cap)
        done_task = collections.Counter(cell_task(c["cell"]) for c in cells if not c.get("skipped"))
        rep("WARN", f"{len(cap)} cells skipped by the capability gate (too few usable pairs: the model does not do the task well "
                    f"enough): {dict(by_task)}; cells that ran: {dict(done_task)}")
        for reason in list(dict.fromkeys(str(c["skipped"]).split(": ", 1)[-1] for c in cap))[:3]:
            rep("INFO", f"skip reason: {reason[:230]}")
    if other:
        rep("FAIL", f"{len(other)} cells skipped for another reason: " + "; ".join(sorted({str(c['skipped'])[:160] for c in other})[:3]))
    expected = len(cfg.cells) - n_skip
    hooks = summ.get("hook_checks") or []
    badh = [h for h in hooks if isinstance(h, dict) and (h.get("status") == "FAIL" or h.get("ok") is False)]
    rep("OK" if not badh else "FAIL", f"{len(hooks)} hook / model checks" + (f", failing: {badh[:2]}" if badh else ", none failing"))
    pq = d / "per_pair.parquet"
    if not pq.exists():
        rep("FAIL", "per_pair.parquet missing")
        return
    pp = pd.read_parquet(pq)
    cells_pp = pp[CELL].drop_duplicates()
    rep("OK" if len(cells_pp) == expected else "FAIL",
        f"{len(cells_pp)} cells in per_pair.parquet ({len(pp):,} rows); expected {expected} = {len(cfg.cells)} configured - {n_skip} skipped")
    errs = validate_per_pair(pp)
    rep("OK" if not errs else "FAIL", "per_pair schema ok" if not errs else f"schema: {errs[:3]}")
    nonfin = int((~np.isfinite(pp[["logit_A", "logit_B", "lse"]].to_numpy(float))).any(axis=1).sum())
    rep("OK" if nonfin == 0 else "FAIL", f"non-finite logits: {nonfin} rows")
    want_c = set(cfg.conditions) | {"clean_A", "clean_B", "full_A", "full_B"} | set((cfg.directions.conditions if cfg.directions else []))
    have_c = set(pp.condition)
    miss = sorted(want_c - have_c - {"repair_mean"})
    rep("OK" if not miss else "FAIL", f"all {len(want_c)} configured conditions present" if not miss else f"conditions missing: {miss}")
    # ---- usable pairs
    ran = [c for c in cells if not c.get("skipped")]
    te = [c["n_test_union"] for c in ran if c.get("n_test_union") is not None]
    tr = [c["n_train"] for c in ran if c.get("n_train") is not None]
    keep = [(c.get("keep_rates") or {}).get("test", {}).get("f_pilot") for c in ran]
    keep = [k for k in keep if k is not None]
    low = [c["cell"] for c in ran if (c.get("n_test_union") or 0) < MIN_TEST]
    if te:
        rep("OK" if not low else ("FAIL" if min(te) < 10 else "WARN"),
            f"usable test pairs per cell that ran: min {min(te)}, median {int(np.median(te))}, max {max(te)}; train min {min(tr)}"
            f"; test keep rate (f_pilot) median {np.median(keep):.2f}" + (f"; below {MIN_TEST}: {len(low)} cells, e.g. {low[:3]}" if low else ""))
    # ---- random nulls
    if cfg.directions and cfg.directions.random_seeds:
        rnd = pp[pp.estimator == "random"]
        per = cell_df(rnd).seed.nunique() if len(rnd) else pd.Series(dtype=int)
        short = per[per != cfg.directions.random_seeds]
        rep("OK" if len(per) == expected and short.empty else "FAIL",
            f"random-subspace nulls: {cfg.directions.random_seeds} seeds at every one of the {expected} cells that ran"
            if short.empty and len(per) == expected
            else f"random nulls wrong at {len(short) + abs(expected - len(per))} cells (want {cfg.directions.random_seeds} seeds)")
    # ---- machinery: full patch = the other run's own output; baselines predict their own answers
    f = pp[pp.f_pilot] if "f_pilot" in pp else pp
    dl = lambda df: (df.logit_A - df.logit_B)
    worst, med = [], []
    for cond_a, cond_b in (("full_A", "clean_A"), ("full_B", "clean_B")):
        a = f[f.condition == cond_a].drop_duplicates(CELL + ["pair_id"]).set_index(CELL + ["pair_id"])
        b = f[f.condition == cond_b].drop_duplicates(CELL + ["pair_id"]).set_index(CELL + ["pair_id"])
        idx = a.index.intersection(b.index)
        if len(idx):
            e = (dl(a.loc[idx]) - dl(b.loc[idx])).abs()
            worst.append((cond_a, float(e.max()))); med.append(float(e.median()))
    if worst:
        mx = max(w for _, w in worst)
        rep("OK" if mx <= IDENTITY_TOL else "WARN",
            f"write-back identity: |delta(logit_A - logit_B)| full_A vs clean_A / full_B vs clean_B: median {max(med):.4f}, max {mx:.3f}"
            + ("" if mx <= IDENTITY_TOL else f" (> {IDENTITY_TOL}: bf16 emulation noise, or a broken patch path: look at the worst cells)"))
    ca = f[f.condition == "clean_A"].drop_duplicates(CELL + ["pair_id"])
    if len(ca):
        acc = float((ca.top1_id == ca.ans_A_id).mean())
        rep("OK" if acc >= 0.9 else "WARN", f"clean_A predicts its own answer on {acc:.1%} of the usable test pairs (the filter requires it)")
    # ---- directions
    dfiles = sorted((d / "directions").glob("*.json"))
    n_est = n_bad = 0
    worst_loss = 0.0
    for fpath in dfiles:
        try:
            dd = json.loads(fpath.read_text())
        except ValueError:
            n_bad += 1
            continue
        for est, v in dd.items():
            if not isinstance(v, dict):
                continue
            n_est += 1
            meta = v.get("meta") or {}
            ll = meta.get("loss_last_epoch")
            if est.startswith("das") and ll is not None and not np.isfinite(ll):
                n_bad += 1
            if est.startswith("das") and meta.get("version") == 2 and not meta.get("n_val"):
                n_bad += 1
            worst_loss = max(worst_loss, float(ll)) if ll is not None and np.isfinite(ll) else worst_loss
    rep("OK" if dfiles and not n_bad else "FAIL",
        f"{n_est} direction entries in {len(dfiles)} files" + (f", {n_bad} unreadable / non-finite / without a held-out split" if n_bad else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--only", default=None, help="comma list of experiment ids")
    ap.add_argument("--allow-smoke", action="store_true", help="plumbing test on tiny / smoke runs: downgrade those flags")
    a = ap.parse_args()
    q = Queue.load(a.queue)
    runs = Path(a.runs)
    want = set(a.only.split(",")) if a.only else None
    rep = Rep()
    seen = {}
    for cpath in dict.fromkeys(q.configs()):
        cfg = ExpConfig.load(cpath)
        if want and cfg.exp_id not in want:
            continue
        print(f"== {cfg.exp_id} ({cpath})")
        complete, partial = [], []
        for md in sorted((runs / cfg.exp_id).glob("*/")):
            mf = md / "manifest.json"
            m = json.loads(mf.read_text()) if mf.exists() else None
            (complete if m and m.get("status") == "complete" else partial).append((md, m))
        if not complete:
            rep("FAIL", f"no complete run under {runs / cfg.exp_id}")
        for md, m in complete:
            check_run(rep, cfg, md, m, a.allow_smoke)
            seen[(cfg.exp_id, m["run_hash"])] = m
        for md, _ in partial:
            rep("INFO", f"partial run directory {md.name}: not finalized (ignored by publish / make_pilot; safe to delete)")
        if len(complete) > 1:
            rep("INFO", f"{len(complete)} complete runs of {cfg.exp_id} (different commits): the latest by time is the one make_pilot reads")
    pub = Path("results_server") / q.tag / "PUBLISHED.json"
    print(f"== published ({pub})")
    if pub.exists():
        p = json.loads(pub.read_text())
        listed = {(r["exp_id"], r["run_hash"]): r for r in p["runs"]}
        smoke = [k for k, r in listed.items() if r.get("smoke")]
        missing = sorted(set(seen) - set(listed))
        extra = sorted(k for k in set(listed) - set(seen) if not want or k[0] in want)
        rep("OK" if not (smoke or missing or extra) else "FAIL",
            f"published runs {sorted(listed)}" if not (smoke or missing or extra)
            else f"published vs complete mismatch: smoke {smoke}, complete but not published {missing}, published but not complete {extra}")
    else:
        rep("WARN", "no PUBLISHED.json yet (publish has not run)")
    print(f"\nsummary: {rep.n['OK']} OK, {rep.n['WARN']} WARN, {rep.n['FAIL']} FAIL, {rep.n['INFO']} INFO")
    return 1 if rep.n["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
