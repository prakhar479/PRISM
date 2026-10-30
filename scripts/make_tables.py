#!/usr/bin/env python
"""DEV: tables/ from ingested runs only (runs/ after scripts/ingest_results.py). Never reads smoke or tiny runs.

    python scripts/make_tables.py [--n-boot 10000] [--only E0,E2]

Writes per experiment tables/<exp>_<name>.csv (schema: tables/SCHEMA.md) recomputed from per_pair.parquet with a
10 000-resample paired bootstrap, and tables/E0_regression.csv comparing E0 with the pilot CSVs.
Numbers are copied from run outputs only; nothing here interprets them.
"""
import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from harness.analysis import add_gaps, random_distribution, summarize
from harness.profile import REPO_ROOT, load_profile

PILOT_EXP1 = {"feat_A_errB": "sae_only", "feat_B_errA": "error_only", "feat_B_noise": "noise_control",
              "feat_A_zero": "sae_splice", "full_A": "clean_full"}
PILOT_EST = {"pca_pooled": "unsupervised_svd", "answer_sep": "supervised_centroid_svd", "random": "random_subspace"}
TABLE_NAME = {"E0": "E0_pilot_regression", "E1": "E1_rescored", "E2": "E2_baselines", "E6": "E6_layers", "E7": "E7_estimators"}


def runs(runs_root: Path, only=None):
    out = []
    for man in sorted(runs_root.glob("E*/*/manifest.json")):
        m = json.loads(man.read_text())
        if m.get("smoke") or m.get("tiny") or m.get("status") != "complete":
            continue
        if only and m["exp_id"] not in only:
            continue
        out.append((m, man.parent))
    return out


def e0_regression(summ: pd.DataFrame, m: dict) -> pd.DataFrame:
    ref = REPO_ROOT / m["config"]["pilot_reference"] / "metrics"
    e1, e3 = pd.read_csv(ref / "exp1_results.csv"), pd.read_csv(ref / "exp3_results.csv")
    s = summ[summ["filter"] == "f_pilot"]
    rows = []
    for _, r in s.iterrows():
        base = dict(task=r.task, layer=r.layer, condition=r.condition, estimator=r.estimator, r=r.r, n_pairs=r.n_pairs,
                    nldr=r.nldr, nldr_lo=r.nldr_lo, nldr_hi=r.nldr_hi)
        if r.condition in PILOT_EXP1:
            q = e1[(e1.task == r.task) & (e1.layer == r.layer) & (e1.width == "16k") & (e1.positions == "from_diff")
                   & (e1.intervention == PILOT_EXP1[r.condition])]
        elif r.condition == "repair" and r.estimator in PILOT_EST:
            q = e3[(e3.task == r.task) & (e3.layer == r.layer) & (e3.width == "16k") & (e3.positions == "from_diff")
                   & (e3.kind == "hybrid_patch") & (e3.method == PILOT_EST[r.estimator]) & (e3["rank"] == r.r)]
        else:
            continue
        if r.estimator == "random" or r.condition == "feat_B_noise":
            continue            # aggregated over seeds below
        if len(q):
            base.update(pilot_nldr=q.iloc[0].nldr, pilot_lo=q.iloc[0].nldr_lo, pilot_hi=q.iloc[0].nldr_hi)
        rows.append(base)
    for (task, layer, cond, est, rr), g in s[(s.estimator == "random") | (s.condition == "feat_B_noise")].groupby(
            ["task", "layer", "condition", "estimator", "r"], dropna=False):
        row = dict(task=task, layer=layer, condition=cond, estimator=est, r=rr, n_pairs=int(g.n_pairs.iloc[0]),
                   nldr=g.nldr.mean(), nldr_lo=g.nldr_lo.mean(), nldr_hi=g.nldr_hi.mean(), n_seeds=len(g))
        if cond == "feat_B_noise":
            q = e1[(e1.task == task) & (e1.layer == layer) & (e1.width == "16k") & (e1.positions == "from_diff")
                   & (e1.intervention == "noise_control")]
        else:
            q = e3[(e3.task == task) & (e3.layer == layer) & (e3.width == "16k") & (e3.positions == "from_diff")
                   & (e3.kind == "hybrid_patch") & (e3.method == "random_subspace") & (e3["rank"] == rr)]
        if len(q):
            row.update(pilot_nldr=q.iloc[0].nldr, pilot_lo=q.iloc[0].nldr_lo, pilot_hi=q.iloc[0].nldr_hi)
        rows.append(row)
    out = pd.DataFrame(rows)
    if "pilot_nldr" in out:
        out["abs_diff_points"] = 100 * (out.nldr - out.pilot_nldr).abs()
        out["within_1_point"] = out.abs_diff_points <= 1.0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--only", default=None)
    ap.add_argument("--profile", default="dev")
    a = ap.parse_args()
    prof = load_profile(a.profile)
    out = REPO_ROOT / "tables"
    out.mkdir(exist_ok=True)
    by_exp = {}
    for m, d in runs(prof.root("runs_root"), set(a.only.split(",")) if a.only else None):
        pp = pd.read_parquet(d / "per_pair.parquet")
        s = add_gaps(summarize(pp, m["config"]["filters"], n_boot=a.n_boot, seed=m["config"]["seed"]))
        s.insert(0, "run_hash", m["run_hash"])
        s.insert(1, "config_name", m["config"]["name"])
        s.insert(2, "commit", m["git"]["commit"][:10])
        s["role"] = s.apply(lambda r: next((c["role"] for c in m["config"]["cells"]
                                            if c["task"] == r.task and c["layer"] == r.layer), "exploratory"), axis=1)
        by_exp.setdefault(m["exp_id"], []).append(s)
        if m["exp_id"] == "E0":
            e0_regression(s, m).to_csv(out / "E0_regression.csv", index=False)
        rd = random_distribution(s)
        if len(rd):
            rd.insert(0, "run_hash", m["run_hash"])
            rd.to_csv(out / f"{m['exp_id']}_{m['config']['name']}_random.csv", index=False)
    for exp, parts in by_exp.items():
        pd.concat(parts, ignore_index=True).to_csv(out / f"{TABLE_NAME.get(exp, exp)}.csv", index=False)
        print(f"{exp}: {sum(len(p) for p in parts)} rows -> tables/{TABLE_NAME.get(exp, exp)}.csv")
    if not by_exp:
        print("no complete, non-smoke runs ingested")
    return 0


if __name__ == "__main__":
    sys.exit(main())
