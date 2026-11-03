#!/usr/bin/env python
"""DEV: E8 all-cell table from ingested runs (runs/ after scripts/ingest_results.py). Never reads smoke or tiny runs.

    python scripts/make_e8.py [--n-boot 10000] [--only E1,E2,E6] [--filters f_pilot,f_A_correct]

Writes
  tables/E8_gaps.csv         gap of every baseline per (run, cell, filter, metric) with 95% CIs
  tables/E8_shares.csv       share of the feat_A_errB gap closed by every repair condition
  tables/E8_allcells.csv     same-task repair vs best control (other task, generic, random p95) at r = 1, 2, the
                             preregistered margin test (diff > 0.30), Holm (confirmatory) / BH (exploratory), to-half r
  tables/E8_allcells.tex     the primary-metric rows of E8_allcells.csv (canonical runs only)
Numbers are copied from run outputs only; nothing here interprets them (harness/e8.py has the definitions).
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from harness import e8
from harness.profile import REPO_ROOT, load_profile


def load_runs(runs_root: Path, only=None):
    out = []
    for man in sorted(runs_root.glob("E*/*/manifest.json")):
        m = json.loads(man.read_text())
        if m.get("smoke") or m.get("tiny") or m.get("status") != "complete":
            continue
        if only and m["exp_id"] not in only:
            continue
        pp = pd.read_parquet(man.parent / "per_pair.parquet")
        if not {"full_A", "feat_A_errB"} <= set(pp.condition):
            continue                                              # gaps need full_A and feat_A_errB
        out.append((m, pp))                                       # runs without repair still give gap rows
    return out


def latex(spec: pd.DataFrame) -> str:
    d = spec[spec.canonical & (spec.metric == e8.PRIMARY)].sort_values(["role", "task", "layer", "filter", "estimator", "r"])
    f = lambda x: "--" if pd.isna(x) else f"{x:.2f}"
    lines = [r"\begin{tabular}{llrllrrrlrrr}", r"\toprule",
             r"role & task & L & filter & estimator & $r$ & share (same) & best control & control & diff & $p_{adj}$ & to-half $r$ \\",
             r"\midrule"]
    for _, x in d.iterrows():
        padj = x.p_holm if x.role == "confirmatory" else x.p_bh
        lines.append(" & ".join([x.role[:4], x.task.replace("_", r"\_"), str(int(x.layer)), x["filter"].replace("_", r"\_"),
                                 x.estimator.replace("_", r"\_"), str(int(x.r)),
                                 f"{f(x.share_same)} [{f(x.share_same_lo)}, {f(x.share_same_hi)}]",
                                 f(x.get("best_control_share")), str(x.get("best_control", "")).replace("_", r"\_"),
                                 f(x.get("diff")), f(padj), f(x.to_half_r)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--only", default=None)
    ap.add_argument("--filters", default=None, help="default: each run's configured filters")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--out", default=None, help="default: <repo>/tables")
    a = ap.parse_args()
    prof = load_profile(a.profile or "dev")
    runs = load_runs(prof.root("runs_root"), set(a.only.split(",")) if a.only else None)
    if not runs:
        print("no complete runs with full_A and feat_A_errB under", prof.root("runs_root"))
        return 1
    frames = e8.analyse_runs(runs, a.filters.split(",") if a.filters else None, n_boot=a.n_boot)
    out = Path(a.out) if a.out else REPO_ROOT / "tables"
    out.mkdir(parents=True, exist_ok=True)
    names = {"gaps": "E8_gaps", "shares": "E8_shares", "spec": "E8_allcells"}
    for k, df in frames.items():
        df.to_csv(out / f"{names[k]}.csv", index=False)
        print(f"{names[k]}: {len(df)} rows from {len(runs)} runs")
    if not frames["spec"].empty:
        (out / "E8_allcells.tex").write_text(latex(frames["spec"]))
        sp = frames["spec"]
        print(f"canonical rows {int(sp.canonical.sum())}; underpowered random controls "
              f"{int((sp.canonical & sp.underpowered_random).sum())}; rows missing a control "
              f"{int((sp.canonical & sp.missing_controls.fillna('').ne('')).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
