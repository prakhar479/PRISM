#!/usr/bin/env python
"""DEV: per-pilot tables and the go / no-go verdicts of docs/PILOT_BREADTH_PLAN.md, from finished breadth-pilot runs.

    python scripts/make_pilot.py --runs results_server/pilot-breadth-server/runs        # published runs (after ingest)
    python scripts/make_pilot.py --runs runs                                            # local runs root
    python scripts/make_pilot.py --runs tests/_tiny_runs --out /tmp/pilot --boot 50     # plumbing check (numbers meaningless)

Writes <out>/{P0,P1,P11,P2,P4,P5,P7,P8}_*.csv and <out>/VERDICTS.md. Every number is exploratory (post hoc, uncorrected).
Definitions (all on the f_pilot test pairs of a cell; clean_A / clean_B baselines; NLDR = harness.metrics):
  capture(U)       NLDR(feat_A_along_U) / NLDR(full_along_U): share of the subspace's causal effect carried by the dictionary's
                   reconstruction (P0 also with feat_codes_along_U: the code-level write-back); 95% CI = pair bootstrap of the ratio
  gap              NLDR(full_A) - NLDR(feat_A_errB) (the task state lost when only features are copied)
  random p95       95th percentile of the condition over the random-U seeds at the same rank
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from harness import metrics as M
from harness.analysis import CELL, with_defaults

FILT = "f_pilot"


# --------------------------------------------------------------------------- loading
def load_runs(root: Path) -> dict:
    """Latest complete run per pilot exp_id (P0..P11) -> {exp_id: {dir, manifest, pp, summary, checks}}; PF / PD records too."""
    out = {}
    for man in sorted(root.glob("P*/*/manifest.json"), key=lambda p: p.stat().st_mtime):
        m = json.loads(man.read_text())
        if m.get("status") != "complete":
            continue
        d = man.parent
        e = m["exp_id"]
        rec = {"dir": d, "manifest": m}
        if (d / "per_pair.parquet").exists():
            rec["pp"] = backfill_donor_status(with_defaults(pd.read_parquet(d / "per_pair.parquet")))
            rec["summary"] = pd.read_csv(d / "summary.csv") if (d / "summary.csv").exists() else None
            rec["checks"] = json.loads((d / "summary.json").read_text()).get("cells", []) if (d / "summary.json").exists() else []
        if e in ("PF", "PD"):
            out.setdefault(e, []).append(rec)
        else:
            out[e] = rec
    return out


def backfill_donor_status(pp: pd.DataFrame) -> pd.DataFrame:
    """Runs from commit 2df949e (tag run-20261122-pilot) record resampU_* rows with donor_status 'n/a', so a pair with no
    donor (the condition then keeps B's own error) would not be dropped. The cell's feat_A_resampC_<same|diff>Ans rows use
    the SAME donor draw (harness.run: one `donors` dict per cell) and do record it: copy it across by (cell, pair_id)."""
    if "donor_status" not in pp or not pp.condition.astype(str).str.startswith("resampU_").any():
        return pp
    key = ["task", "layer", "sae", "fit_families", "pair_id"]
    pp = pp.copy()
    for kind in ("sameAns", "diffAns"):
        src = pp[pp.condition == f"feat_A_resampC_{kind}"].drop_duplicates(key).set_index(key)["donor_status"].to_dict()
        m = (pp.condition == f"resampU_{kind}") & (pp.donor_status == "n/a")
        if m.any() and src:
            pp.loc[m, "donor_status"] = [src.get(t, "n/a") for t in pp.loc[m, key].itertuples(index=False, name=None)]
    return pp


def outs(df):
    return M.Out(df["logit_A"].to_numpy(float), df["logit_B"].to_numpy(float), df["lse"].to_numpy(float),
                 df["top1_id"].to_numpy(), np.stack(df["cand_logits"].to_numpy()) if "cand_logits" in df else None)


class Cell:
    """One cell's per-pair rows (f_pilot pairs), with metric and ratio helpers."""

    def __init__(self, g: pd.DataFrame):
        g = g[g[FILT]] if FILT in g else g
        self.g = g
        a = g[g.condition == "clean_A"].set_index("pair_id")
        b = g[g.condition == "clean_B"].set_index("pair_id")
        self.pairs = a.index.intersection(b.index)
        self.A, self.B = outs(a.loc[self.pairs]), outs(b.loc[self.pairs])
        self.ansA = a.loc[self.pairs, "ans_A_id"].to_numpy()
        self.ansB = a.loc[self.pairs, "ans_B_id"].to_numpy()

    def rows(self, condition, estimator="", r=-1, seed=-1, source=None):
        c = self.g[(self.g.condition == condition) & (self.g.estimator == estimator) & (self.g.r == r) & (self.g.seed == seed)]
        if source is not None:
            c = c[c.source == source]
        if "donor_status" in c:                          # no-donor placeholders (B's own error), as harness.analysis.summarize
            c = c[c["donor_status"].isin(["ok", "relaxed", "n/a"])]
        return c.drop_duplicates("pair_id").set_index("pair_id")

    def metric(self, c, idx=None, key="nldr"):
        common = self.pairs.intersection(c.index)
        pos = self.pairs.get_indexer(common)
        P = outs(c.loc[common])
        if idx is not None:
            P, pos = P.take(idx), pos[idx]
        return M.per_condition(P, self.A.take(pos), self.B.take(pos), self.ansA[pos], self.ansB[pos])[key]

    def aligned(self, *cs):
        common = self.pairs
        for c in cs:
            common = common.intersection(c.index)
        return [c.loc[common] for c in cs], common

    def value(self, cond, key="nldr", **kw):
        c = self.rows(cond, **kw)
        return self.metric(c, key=key) if len(c) else np.nan

    def ratio(self, num, den, n_boot=1000, seed=0, key="nldr"):
        """(point, lo, hi) of metric(num) / metric(den) over the pairs both have; pair bootstrap."""
        (n_, d_), common = self.aligned(num, den)
        if len(common) < 2:
            return (np.nan, np.nan, np.nan)
        pos = self.pairs.get_indexer(common)
        Pn, Pd = outs(n_), outs(d_)

        def r(idx):
            a = M.per_condition(Pn.take(idx), self.A.take(pos[idx]), self.B.take(pos[idx]), self.ansA[pos[idx]], self.ansB[pos[idx]])[key]
            b = M.per_condition(Pd.take(idx), self.A.take(pos[idx]), self.B.take(pos[idx]), self.ansA[pos[idx]], self.ansB[pos[idx]])[key]
            return a / b if abs(b) > 1e-9 else np.nan
        pt = r(np.arange(len(common)))
        rng = np.random.default_rng(seed)
        bs = np.array([r(rng.integers(0, len(common), len(common))) for _ in range(n_boot)]) if n_boot else np.array([])
        bs = bs[np.isfinite(bs)]
        return (pt, *(np.percentile(bs, [2.5, 97.5]) if len(bs) else (np.nan, np.nan)))

    def random_p95(self, cond, r, key="nldr"):
        vals = [self.metric(c.drop_duplicates("pair_id").set_index("pair_id"), key=key)
                for _, c in self.g[(self.g.condition == cond) & (self.g.estimator == "random") & (self.g.r == r)].groupby("seed")]
        return float(np.percentile(vals, 95)) if vals else np.nan


def cells(rec):
    for key, g in rec["pp"].groupby(CELL, sort=False):
        yield dict(zip(CELL, key)), Cell(g)


def das_info(rec) -> dict:
    """{(task, layer, fit_tag, estimator): DAS training info (val / fit IIA, NLDR, share above full)} from directions/*.json."""
    out = {}
    for f in (rec["dir"] / "directions").glob("*.json"):
        d = json.loads(f.read_text())
        stem = f.stem
        for est, v in d.items():
            if est.startswith("das") and isinstance(v, dict):
                out[(stem, est)] = v.get("meta") or {}
    return out


def capture_table(rec, ests, conds=("feat_A_along_U",), n_boot=1000):
    rows = []
    for meta, c in cells(rec):
        for e in ests:
            from harness.config import das_rank
            r = das_rank(e) if e.startswith("das") else None
            for rr in ([r] if r else sorted(set(c.g[c.g.estimator == e].r))):
                full = c.rows("full_along_U", e, rr)
                if not len(full):
                    continue
                row = {**meta, "estimator": e, "r": int(rr), "full_nldr": c.metric(full),
                       "full_flip": c.metric(full, key="flip_rate"),
                       "random_p95_full_nldr": c.random_p95("full_along_U", rr),
                       "random_p95_full_flip": c.random_p95("full_along_U", rr, "flip_rate")}
                for cond in conds:
                    num = c.rows(cond, e, rr)
                    if len(num):
                        pt, lo, hi = c.ratio(num, full, n_boot)
                        row.update({f"capture_{cond}": pt, f"capture_{cond}_lo": lo, f"capture_{cond}_hi": hi})
                rows.append(row)
    return pd.DataFrame(rows)


def gap_table(rec, conds=("feat_A_errB", "feat_A_meanPos", "feat_B_errA", "feat_B_noise", "feat_A_noise")):
    rows = []
    for meta, c in cells(rec):
        full = c.value("full_A")
        fflip = c.value("full_A", key="flip_rate")
        row = {**meta, "n_pairs": len(c.pairs), "full_A_nldr": full}
        for cond in conds:
            v = c.value(cond)
            if np.isfinite(v):
                row[f"{cond}_nldr"] = v
                row[f"{cond}_flip"] = c.value(cond, key="flip_rate")
                row[f"gap_{cond}_nldr"] = full - v
                row[f"gap_{cond}_flip"] = fflip - row[f"{cond}_flip"]
        rows.append(row)
    return pd.DataFrame(rows)


def keep_rates(rec) -> pd.DataFrame:
    return pd.DataFrame([{"cell": k.get("cell"), "keep_test": (k.get("keep_rates") or {}).get("test", {}).get(FILT),
                          "n_test": k.get("n_test_union"), "n_train": k.get("n_train"), "skipped": k.get("skipped")}
                         for k in rec.get("checks", [])])


# --------------------------------------------------------------------------- pilots
def p0(rec, n_boot):
    das = sorted({e for e in rec["pp"].estimator.unique() if e.startswith("das")})
    t = capture_table(rec, das, ("feat_A_along_U", "feat_codes_along_U"), n_boot)
    info = das_info(rec)
    for i, row in t.iterrows():
        m = next((v for (stem, e), v in info.items() if e == row.estimator and stem.startswith(f"{row.task}_L{row.layer}")), {})
        t.loc[i, "val_iia"] = (m.get("val") or {}).get("iia", m.get("val_interchange_acc"))
        t.loc[i, "val_nldr"] = (m.get("val") or {}).get("nldr_mean")
        t.loc[i, "val_share_above_full"] = (m.get("val") or {}).get("share_above_full")
        t.loc[i, "objective"] = m.get("objective", "ce" if "v2" not in row.estimator else None)
    lines = []
    ok_all = True
    for (task, r), g in t[t.estimator.str.contains("v2")].groupby(["task", "r"]):
        spread = g["capture_feat_A_along_U"].max() - g["capture_feat_A_along_U"].min()
        code_dev = (g["capture_feat_codes_along_U"] - g["capture_feat_A_along_U"]).abs().max() if "capture_feat_codes_along_U" in g else np.nan
        val_ok = (g["val_iia"].astype(float) >= 0.5).mean() if g["val_iia"].notna().any() else np.nan
        ok = spread <= 0.10 and (not np.isfinite(code_dev) or code_dev <= 0.10) and (not np.isfinite(val_ok) or val_ok >= 0.5)
        ok_all &= bool(ok)
        lines.append(f"- {task} rank {r}: capture spread over objective x seed = {spread:.3f}; max |code - subspace| = {code_dev:.3f}; "
                     f"share of fits with held-out IIA >= 0.5 = {val_ok:.2f} -> {'robust' if ok else 'NOT robust'}")
    verdict = "GO (protocol robust: keep it as the default)" if ok_all else "NO-GO as is: adopt kl + held-out validity; report the sensitivity"
    return {"P0_capture": t}, verdict, lines


def p1(rec, n_boot):
    t = capture_table(rec, ["das4v2_kl", "das8v2_kl", "das16v2_kl"], ("feat_A_along_U",), n_boot)
    t["family"] = t.fit_families
    lines, hits = [], {}
    cap = "capture_feat_A_along_U"
    for (L, sae, e), g in t.groupby(["layer", "sae", "estimator"]):
        g = g.set_index("family")
        if "pc_en" not in g.index:
            continue
        for other in ("pc_en_swap", "pc_en_query"):
            if other in g.index:
                d = g.loc["pc_en", cap] - g.loc[other, cap]
                sep = g.loc["pc_en", f"{cap}_lo"] > g.loc[other, f"{cap}_hi"]
                hits.setdefault((other, sae.split("_")[1] if "_" in sae else sae), {}).setdefault(L, []).append(bool(d >= 0.15 and sep))
                lines.append(f"- L{L} {sae} {e}: content {g.loc['pc_en', cap]:.3f} vs {other} {g.loc[other, cap]:.3f} (diff {d:+.3f}, CIs separate: {sep})")
    go = any(sum(any(v) for v in byL.values()) >= 2 for (o, w), byL in hits.items() if w == "16k")
    return {"P1_capture": t}, ("GO (relational capture lower than content)" if go else "NO-GO / unclear (no layer-robust content-relational gap)"), lines


def p11(rec, n_boot, prompt_rows=None):
    g = gap_table(rec)
    t = capture_table(rec, ["das4v2_kl", "das8v2_kl"], ("feat_A_along_U",), n_boot)
    k = keep_rates(rec)
    fb = []
    if prompt_rows is not None:
        fall = {r["pair_id"]: r["cand_ids"][r["meta"]["fallback_index"]] for r in prompt_rows if "fallback_index" in r.get("meta", {})}
        pp = rec["pp"]
        pp = pp[pp[FILT]] if FILT in pp else pp
        for (L, cond), c in pp.groupby(["layer", "condition"]):
            f = c.pair_id.map(fall)
            fb.append({"layer": L, "condition": cond, "fallback_rate": float((c.top1_id == f).mean())})
    lines = [f"- keep rate (test, {FILT}): " + ", ".join(f"{r.cell}: {r.keep_test}" for r in k.itertuples())]
    writes = [(r.layer, r.estimator, r.full_flip, r.random_p95_full_flip) for r in t.itertuples()]
    ok_keep = (k.keep_test.dropna() >= 0.30).any() if len(k) else False
    ok_write = any(np.isfinite(f) and np.isfinite(p) and f >= 3 * max(p, 1e-3) for _, _, f, p in writes)
    lines += [f"- L{L} {e}: DAS full-patch flip {f:.3f} vs random p95 {p:.3f}" for L, e, f, p in writes]
    out = {"P11_gaps": g, "P11_capture": t, "P11_keep": k}
    if fb:
        out["P11_fallback"] = pd.DataFrame(fb)
    return out, ("GO (OT-v2 usable)" if ok_keep and ok_write else "NO-GO (OT stays out)"), lines


def p2(rec, n_boot, pf=None):
    t = capture_table(rec, ["das8v2_kl"], ("feat_A_along_U",), n_boot)
    g = gap_table(rec, ("feat_A_errB", "feat_A_meanPos"))
    gen = {c.get("cell"): c.get("generic") or {} for c in rec.get("checks", [])}
    t["generic_l0"] = [next((v.get("mean_l0") for k, v in gen.items() if k and k.startswith(f"{r.task}_{r.sae}")), np.nan) for r in t.itertuples()]
    t["generic_err_energy"] = [next((v.get("err_energy_ratio") for k, v in gen.items() if k and k.startswith(f"{r.task}_{r.sae}")), np.nan)
                               for r in t.itertuples()]
    if pf is not None:
        t = t.merge(pf[["dict", "recovered_mean", "ce_delta", "fvu"]].rename(columns={"dict": "sae"}), on="sae", how="left")
    lines = []
    cap = "capture_feat_A_along_U"
    go_a = go_b = False
    for task, gt in t[t.layer == 12].groupby("task"):
        gt = gt.copy()
        gt["width"] = gt.sae.str.extract(r"_(\d+k|\d+m)_")[0]
        for (w, b), gg in gt.assign(l0bin=pd.cut(gt.generic_l0, [0, 30, 55, 120, 1e9])).groupby(["width", "l0bin"], observed=True):
            if len(gg) >= 3:
                spread = gg[cap].max() - gg[cap].min()
                sep = gg[f"{cap}_lo"].max() > gg[f"{cap}_hi"].min()
                go_a |= bool(spread >= 0.10 and sep)
                lines.append(f"- {task} {w} L0 {b}: {len(gg)} dictionaries, capture spread {spread:.3f} (CIs separate: {sep})")
        if "recovered_mean" in gt and gt.recovered_mean.notna().sum() >= 10:
            rho = gt[["recovered_mean", cap]].corr(method="spearman").iloc[0, 1]
            go_b |= bool(gt.recovered_mean.notna().sum() >= 30 and abs(rho) <= 0.3)
            lines.append(f"- {task}: Spearman(CE recovered, capture) = {rho:.3f} over {gt.recovered_mean.notna().sum()} dictionaries")
    v = "GO (" + ", ".join(x for x, y in (("architecture spread", go_a), ("fidelity does not predict capture", go_b)) if y) + ")" \
        if (go_a or go_b) else "NO-GO as a standalone paper (section of the winner)"
    return {"P2_capture": t, "P2_gaps": g}, v, lines


def p4(rec, n_boot, pf=None):
    g = gap_table(rec, ("feat_A_errB", "feat_A_meanPos"))
    t = capture_table(rec, ["das8v2_kl"], ("feat_A_along_U",), n_boot)
    lines, hits = [], 0
    for task, gg in g.groupby("task"):
        base = gg[~gg.sae.str.contains("__") & ~gg.sae.str.contains("klft")]
        if base.empty:
            continue
        b = base.iloc[0]
        for r in gg.itertuples():
            if r.sae == b.sae:
                continue
            task_closed = (b.gap_feat_A_errB_nldr - r.gap_feat_A_errB_nldr) / b.gap_feat_A_errB_nldr if abs(b.gap_feat_A_errB_nldr) > 1e-6 else np.nan
            ce_closed = np.nan
            if pf is not None:
                cb = pf[pf["dict"] == b.sae]
                cv = pf[pf["dict"] == r.sae]
                if len(cb) and len(cv) and abs(float(cb.ce_delta.iloc[0])) > 1e-9:
                    ce_closed = (float(cb.ce_delta.iloc[0]) - float(cv.ce_delta.iloc[0])) / float(cb.ce_delta.iloc[0])
            ratio = task_closed / ce_closed if np.isfinite(ce_closed) and abs(ce_closed) > 1e-6 else np.nan
            hit = np.isfinite(ratio) and (ratio <= 0.5 or ratio >= 2)
            hits += bool(hit)
            lines.append(f"- {task} {r.sae}: task gap closed {task_closed:+.3f}, CE gap closed {ce_closed:+.3f}, ratio {ratio:.2f}"
                         + ("  (P6: share of the task-relevant error in the linearly predictable part)" if r.sae.endswith("__lin") else ""))
    return {"P4_gaps": g, "P4_capture": t}, ("GO (decoupled in >= 2 cells)" if hits >= 2 else "NO-GO / mixed"), lines


def p5(rec, n_boot):
    ests = ["ridge_value", "fourier_T10", "fourier_T5", "fourier_T2", "das4v2_kl", "das8v2_kl"]
    t = capture_table(rec, ests, ("feat_A_along_U",), n_boot)
    rows, lines, go = [], [], False
    for meta, c in cells(rec):
        full_eff = c.value("feat_A_resampC_sameAns") - c.value("feat_A_resampC_diffAns")
        for e in ests:
            for r in sorted(set(c.g[(c.g.estimator == e) & (c.g.condition == "resampU_sameAns")].r)):
                eff = c.value("resampU_sameAns", estimator=e, r=r) - c.value("resampU_diffAns", estimator=e, r=r)
                rows.append({**meta, "estimator": e, "r": r, "resampU_effect": eff, "full_resample_effect": full_eff,
                             "share_of_full": eff / full_eff if abs(full_eff) > 1e-9 else np.nan})
        cap = t[(t.task == meta["task"]) & (t.layer == meta["layer"]) & (t.sae == meta["sae"])].set_index(["estimator", "r"])
        if ("ridge_value", 1) in cap.index and ("fourier_T10", 2) in cap.index:
            m_, p_ = cap.loc[("ridge_value", 1), "capture_feat_A_along_U"], cap.loc[("fourier_T10", 2), "capture_feat_A_along_U"]
            sh = next((x["share_of_full"] for x in rows if x["estimator"] == "ridge_value" and x["sae"] == meta["sae"] and x["layer"] == meta["layer"]), np.nan)
            ok = (p_ - m_ >= 0.20) and np.isfinite(sh) and sh >= 0.5
            go |= bool(ok)
            lines.append(f"- L{meta['layer']} {meta['sae']}: capture magnitude {m_:.3f} vs period-10 plane {p_:.3f}; "
                         f"magnitude carries {sh:.2f} of the resampled-error effect -> {'yes' if ok else 'no'}")
    return {"P5_capture": t, "P5_resample": pd.DataFrame(rows)}, ("GO (magnitude goes dark)" if go else "NO-GO / unclear"), lines


def p7(rec, n_boot):
    rows, lines, go = [], [], False
    for meta, c in cells(rec):
        base = c.value("feat_A_errB")
        full = c.value("full_A")
        for e in ("grad_err", "grad_pooled", "pca_pooled", "answer_sep", "das8v2_kl"):
            for r in sorted(set(c.g[(c.g.estimator == e) & (c.g.condition == "repair")].r)):
                v = c.value("repair", estimator=e, r=r)
                rows.append({**meta, "estimator": e, "r": r, "repair_nldr": v,
                             "share_closed": (v - base) / (full - base) if abs(full - base) > 1e-9 else np.nan,
                             "random_p95": c.random_p95("repair", r)})
    t = pd.DataFrame(rows)
    for (task, L), g in t.groupby(["task", "layer"]) if len(t) else []:
        for r in (1, 2, 4, 8):
            ge = g[(g.estimator == "grad_err") & (g.r == r)]
            pc = g[(g.estimator == "pca_pooled") & (g.r == r)]
            if len(ge) and len(pc):
                a, b, p95 = ge.share_closed.iloc[0], pc.share_closed.iloc[0], ge.random_p95.iloc[0]
                ok = a >= 2 * max(b, 1e-3) and ge.repair_nldr.iloc[0] > p95
                go |= bool(ok)
                lines.append(f"- {task} L{L} r={r}: grad_err closes {a:.3f}, pca_pooled {b:.3f} (repair above random p95: "
                             f"{ge.repair_nldr.iloc[0] > p95}) -> {'yes' if ok else 'no'}")
    return {"P7_repair": t}, ("GO (sensitivity beats variance)" if go else "NO-GO / unclear"), lines


def p8(rec, n_boot):
    g = gap_table(rec, ("feat_A_errB", "feat_A_meanPos", "feat_B_errA"))
    t = capture_table(rec, ["das4v2_kl", "das8v2_kl"], ("feat_A_along_U",), n_boot)
    k = keep_rates(rec)
    tasks_ok = {c.split("_L")[0] for c, v in zip(k.cell, k.keep_test) if v is not None and v >= 0.30} if len(k) else set()
    gap_pos = (g.gap_feat_A_errB_nldr > 0).any() if "gap_feat_A_errB_nldr" in g else False
    lines = [f"- tasks with keep >= 0.30: {sorted(tasks_ok)}", f"- any positive feature-only gap: {bool(gap_pos)}"]
    return {"P8_gaps": g, "P8_capture": t, "P8_keep": k}, ("GO (second model usable)" if len(tasks_ok) >= 2 and gap_pos else "NO-GO / fallback Llama-3.2-1B"), lines


def family_of(sae: str) -> str:
    """P12 / P13 dictionary family from the cell's SAE key (registry names of scripts/build_dict_registry.py)."""
    name = sae.split("_", 1)[1] if sae.startswith("L") else sae
    if not name.startswith(("gs2_1bc_", "gs2_1bitc_")):                     # residual SAE (Gemma Scope 2 resid_post)
        return "resid_seed2" if name.endswith("seed_1") else "resid"
    k = name.split("_L", 1)[1].split("_", 1)[1]
    fam = "xc" if k.startswith("xc") else "clt" if k.startswith("clt") else "mlpsae_all" if k.startswith("mlpsae") \
        else "tc_single" if k.endswith("_single") else "tc_all"
    return fam + ("+attn" if "+attn" in k else "")


def p12(rec, n_boot):
    t = capture_table(rec, ["das8v2_kl"], ("feat_A_along_U",), n_boot)
    g = gap_table(rec, ("feat_A_errB", "feat_A_meanPos", "feat_B_errA"))
    for df in (t, g):
        if len(df):
            df["family"] = df.sae.map(family_of)
    cap, lines = "capture_feat_A_along_U", []
    go_a = go_b = False
    for (task, L), gt in (t.groupby(["task", "layer"]) if cap in t else []):
        fam = gt.groupby("family")[cap].median().sort_values()
        lines.append(f"- {task} L{L}: median capture by family " + ", ".join(f"{k} {v:.3f}" for k, v in fam.items()))
        lo, hi = gt.loc[gt[cap].idxmin()], gt.loc[gt[cap].idxmax()]
        sep = bool(hi[f"{cap}_lo"] > lo[f"{cap}_hi"])
        spread = float(hi[cap] - lo[cap])
        go_a |= bool(L == 13 and spread >= 0.10 and sep and hi.family != lo.family)
        lines.append(f"  spread {spread:.3f} ({lo.sae} {lo[cap]:.3f} .. {hi.sae} {hi[cap]:.3f}; CIs separate: {sep})")
        s1 = gt[gt.family == "resid_seed2"]
        s0 = gt[gt.sae.str.contains("_262k_medium") & (gt.family == "resid")]
        if len(s1) and len(s0):
            lines.append(f"  seed floor (262k medium, two seeds): |delta capture| = {abs(float(s1[cap].iloc[0] - s0[cap].iloc[0])):.3f}")
    for (task, L), gg in (g.groupby(["task", "layer"]) if len(g) else []):
        rep = gg[gg.family.isin(["tc_all", "clt", "tc_all+attn", "clt+attn", "mlpsae_all"])]
        for r in rep.itertuples():
            gap, err = getattr(r, "gap_feat_A_errB_nldr", np.nan), getattr(r, "feat_B_errA_nldr", np.nan)
            go_b |= bool(L == 13 and gap >= 0.10 and err >= 0.10)
        if len(rep):
            best = rep.sort_values("gap_feat_A_errB_nldr").iloc[-1]
            lines.append(f"- {task} L{L}: replacement models: largest feature-only gap {best.gap_feat_A_errB_nldr:.3f} ({best.sae}; "
                         f"error-only NLDR {best.get('feat_B_errA_nldr', np.nan):.3f})")
    v = "GO (" + ", ".join(x for x, y in (("dictionary family matters", go_a), ("replacement-model error nodes carry task state", go_b)) if y) + ")" \
        if (go_a or go_b) else "NO-GO (families agree; error nodes carry little task state)"
    return {"P12_capture": t, "P12_gaps": g}, v, lines


def p13(rec, n_boot):
    t = capture_table(rec, ["diffmean", "das1v2_kl", "das4v2_kl", "das8v2_kl", "pca_pooled"], ("feat_A_along_U",), n_boot)
    g = gap_table(rec, ("feat_A_errB", "feat_A_meanPos", "feat_B_errA", "feat_B_noise"))
    k = keep_rates(rec)
    for df in (t, g):
        if len(df):
            df["family"] = df.sae.map(family_of)
    lines, cap = [], "capture_feat_A_along_U"
    usable = float(k.keep_test.dropna().median()) if len(k) and k.keep_test.notna().any() else np.nan
    lines.append(f"- usable test pairs (median keep rate over cells): {usable:.2f}")
    dark = light = False
    for r in g.sort_values(["layer", "sae"]).itertuples():
        gap, err = getattr(r, "gap_feat_A_errB_nldr", np.nan), getattr(r, "feat_B_errA_nldr", np.nan)
        lines.append(f"- L{r.layer} {r.sae}: full_A {r.full_A_nldr:.3f}, feature-only gap {gap:.3f}, error-only {err:.3f}, n {r.n_pairs}")
        dark |= bool(gap >= 0.10 or err >= 0.10)
    for r in (t[t.estimator.isin(["diffmean", "das1v2_kl"])].sort_values(["layer", "sae", "estimator"]).itertuples() if cap in t else []):
        c, lo, hi = getattr(r, cap), getattr(r, f"{cap}_lo"), getattr(r, f"{cap}_hi")
        lines.append(f"  L{r.layer} {r.sae} {r.estimator} r{r.r}: capture {c:.3f} [{lo:.3f}, {hi:.3f}] (full NLDR {r.full_nldr:.3f}, "
                     f"random p95 {r.random_p95_full_nldr:.3f})")
        dark |= bool(r.estimator == "diffmean" and hi < 0.9 and r.full_nldr > r.random_p95_full_nldr)
        light |= bool(r.estimator == "diffmean" and lo >= 0.9)
    gate = np.isfinite(usable) and usable >= 0.3
    v = ("GO (refusal state partly in the SAE error)" if dark else "GO as a negative result (features carry refusal)" if light
         else "UNCLEAR") if gate else "NO-GO (too few pairs refuse / comply as built)"
    return {"P13_capture": t, "P13_gaps": g}, v, lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default="analysis/pilot")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--data-root", default="data")
    a = ap.parse_args()
    runs = load_runs(Path(a.runs))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pf = {}
    for rec in runs.get("PF", []):
        f = rec["dir"] / "fidelity.csv"
        if f.exists():
            pf[rec["manifest"]["config"].get("identity", {}).get("config")] = pd.read_csv(f)
    report = ["# Breadth pilot verdicts (exploratory, post hoc; scripts/make_pilot.py)", ""]
    jobs = {"P0": lambda r: p0(r, a.boot), "P1": lambda r: p1(r, a.boot), "P2": lambda r: p2(r, a.boot, pf.get("P2")),
            "P4": lambda r: p4(r, a.boot, pf.get("P4")), "P5": lambda r: p5(r, a.boot), "P7": lambda r: p7(r, a.boot),
            "P8": lambda r: p8(r, a.boot), "P12": lambda r: p12(r, a.boot), "P13": lambda r: p13(r, a.boot)}
    if "P11" in runs:
        from harness import tasks as T
        ps = runs["P11"]["manifest"]["prompt_sets"].get("entity_track", {}).get("file")
        rows = T.load_set(Path(a.data_root) / ps)[0] if ps and (Path(a.data_root) / ps).exists() else None
        jobs["P11"] = lambda r, rows=rows: p11(r, a.boot, rows)
    for e in ("P0", "P11", "P1", "P2", "P4", "P5", "P7", "P8", "P12", "P13"):
        if e not in runs and e in ("P12", "P13"):
            continue                                       # second set (branch pilot-g3): only when its results are ingested
        if e not in runs:
            report += [f"## {e}: no complete run", ""]
            continue
        try:
            tables, verdict, lines = jobs[e](runs[e])
        except Exception as ex:  # noqa: BLE001 - one pilot's failure must not hide the others
            report += [f"## {e}: analysis failed ({type(ex).__name__}: {ex})", ""]
            continue
        for name, df in tables.items():
            df.to_csv(out / f"{name}.csv", index=False)
        report += [f"## {e}: {verdict}", f"run {runs[e]['manifest']['run_hash']} ({runs[e]['manifest'].get('git', {}).get('commit', '')[:10]})", *lines, ""]
    for rec in runs.get("PD", []):
        i = json.loads((rec["dir"] / "info.json").read_text()) if (rec["dir"] / "info.json").exists() else {}
        report.append(f"- trained {i.get('name')}: held-out CE delta {i.get('eval_before', {}).get('ce_delta')} -> {i.get('eval_after', {}).get('ce_delta')}")
    (out / "VERDICTS.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
