"""Shared helpers for the STANDARD-METHODS readout (role `standard`) of E17-E22.

Reads the standard tables written by harness/final.py + scripts/make_final.py ($SP/r5/tables, $SP/hw/tables/final) and the
per-pair parquet files ($SP/ana_runs) only for paired exact tests. Decision rules are the addendum's (harness/final.py constants):
  effective        flip >= random p95 + 0.10 AND flip_lo > random p95 (>= 20 random seeds)
  illusion         full_along_U effective AND full - feat >= 0.10 AND paired CI lo > 0
  feature-mediated feat_A_along_U effective
  DAS fit valid    train interchange acc >= 0.5 (invalid -> rule outcome NA, "not evidence")
  Q2               steer flip >= 0.50 AND random p95 <= 0.05
Added here (post hoc, for multiplicity accounting only; NOT part of the frozen rules):
  p_eff   one-sided exact binomial test H0: flip rate <= random p95 + 0.10 (k = flips among eligible pairs)
  p_ill   one-sided exact McNemar (discordant pairs) H0: P(full flips, feat not) <= P(feat flips, full not)
  BH (q = 0.05) within each experiment over every test the experiment's script reports.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd
from scipy import stats

SP = Path("/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad")
R5 = SP / "r5" / "tables"
FINAL = SP / "hw" / "tables" / "final"
RUNS = SP / "ana_runs"
OUT = SP / "team" / "standard_out"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SP / "hw"))

EFFECT_PTS, DAS_VALID_ACC, ALIGNED = 0.10, 0.5, 0.5
FILTERS = ["f_pilot", "f_A_correct", "f_none"]                 # f_pilot (fit filter) first, as instructed

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 2000)
pd.set_option("display.max_columns", 60)


def bh(p):
    """Benjamini-Hochberg adjusted p-values (NaN-safe)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    n = len(q)
    if n == 0:
        return out
    o = np.argsort(q)
    adj = q[o] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(n)
    res[o] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def p_effective(flip, n_elig, p95):
    """One-sided exact binomial p for H0: rate <= p95 + 0.10."""
    if not (np.isfinite(flip) and np.isfinite(p95) and n_elig > 0):
        return np.nan
    k = int(round(flip * n_elig))
    p0 = min(p95 + EFFECT_PTS, 1.0)
    if p0 >= 1.0:
        return 1.0
    return float(stats.binomtest(k, int(n_elig), p0, alternative="greater").pvalue)


def add_p_eff(t):
    t = t.copy()
    t["p_eff"] = [p_effective(f, n, q) for f, n, q in zip(t.flip_rate, t.flip_n_eligible, t.random_p95)]
    return t


def recheck_rules(t):
    """Recompute 'effective' from the table's own columns and count disagreements with the shipped flag."""
    ev = t.random_p95.notna() & t.flip_rate_lo.notna() & (t.get("random_n", 20) >= 20)
    mine = (t.flip_rate >= t.random_p95 + EFFECT_PTS) & (t.flip_rate_lo > t.random_p95)
    if "das_train_acc" in t:
        bad = t.estimator.astype(str).str.startswith("das") & (pd.to_numeric(t.das_train_acc, errors="coerce") < DAS_VALID_ACC)
        ev = ev & ~bad
    shipped = t.effective.astype("boolean")
    m = ev & shipped.notna()
    return int((mine[m] != shipped[m].astype(bool)).sum()), int(m.sum())


def eval_family(fam: pd.Series) -> pd.Series:
    f = fam.fillna("pilot").astype(str)
    return pd.Series(np.where(f.str.startswith("pilot"), "pilot", f), index=fam.index)


def load_pp(exp, conds, extra_filter=None):
    d = next((RUNS / exp).glob("*/per_pair.parquet"))
    cols = ["pair_id", "family", "task", "layer", "sae", "condition", "estimator", "r", "seed", "ans_A_id", "ans_B_id",
            "fit_families", "f_none", "f_A_correct", "f_pilot", "top1_id"]
    import pyarrow.parquet as pq
    names = pq.ParquetFile(d).schema_arrow.names
    cols = [c for c in cols if c in names]
    pp = pd.read_parquet(d, columns=cols, filters=[("condition", "in", list(conds))])
    if "fit_families" not in pp:
        pp["fit_families"] = "pilot"
    pp["family"] = eval_family(pp["family"])
    return pp


def mcnemar_rows(pp, cells_keys=("task", "layer", "sae", "fit_families", "family")):
    """Per (cell, filter, estimator, r): exact one-sided McNemar for full_along_U vs feat_A_along_U flips on eligible pairs,
    plus my own recomputed flip rates (to cross-check the shipped tables)."""
    cb = load_pp_base = None
    out = []
    keys = list(cells_keys)
    for cell, g in pp.groupby(keys, sort=False):
        base = g[g.condition == "clean_B"].drop_duplicates("pair_id").set_index("pair_id")
        elig_all = base.top1_id == base.ans_B_id
        sub = g[g.condition.isin(["full_along_U", "feat_A_along_U"]) & (g.estimator != "random") & (g.r > 0)]
        for (est, r), h in sub.groupby(["estimator", "r"], sort=False):
            fu = h[h.condition == "full_along_U"].drop_duplicates("pair_id").set_index("pair_id")
            fe = h[h.condition == "feat_A_along_U"].drop_duplicates("pair_id").set_index("pair_id")
            for flt in FILTERS:
                ids = base.index[base[flt].astype(bool) & elig_all].intersection(fu.index).intersection(fe.index)
                a = (fu.loc[ids].top1_id == fu.loc[ids].ans_A_id).to_numpy()
                b = (fe.loc[ids].top1_id == fe.loc[ids].ans_A_id).to_numpy()
                n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
                p = float(stats.binomtest(n10, n10 + n01, 0.5, alternative="greater").pvalue) if n10 + n01 else 1.0
                out.append({**dict(zip(keys, cell)), "filter": flt, "estimator": est, "r": int(r), "n_elig_mine": len(ids),
                            "flip_full_mine": a.mean() if len(ids) else np.nan, "flip_feat_mine": b.mean() if len(ids) else np.nan,
                            "n10": n10, "n01": n01, "p_ill": p})
    return pd.DataFrame(out)


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    if isinstance(x, (bool, np.bool_)):
        return "T" if x else "F"
    if isinstance(x, (float, np.floating)):
        return f"{x:.{nd}f}"
    return str(x)


def flag(v):
    if v is pd.NA or v is None or (isinstance(v, float) and np.isnan(v)):
        return "NA"
    return "T" if bool(v) else "F"


def md_table(df, cols=None):
    df = df if cols is None else df[cols]
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) if not isinstance(v, str) else v for v in r.values) + " |")
    return "\n".join(lines)
