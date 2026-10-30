"""Summaries derived from per_pair.parquet (pure; the same code runs on RUN for a quick summary and on DEV for the
tables). Never interprets: it computes metrics, intervals and flags."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from harness import metrics as M
from harness import stats as St

KEY = ["task", "layer", "sae", "condition", "estimator", "source", "r", "seed"]
CELL = ["task", "layer", "sae"]
BOOT_METRICS = ("nldr", "nldr_pair_mean", "prob_recovery", "flip_rate", "top1_is_A", "cand_entropy")


def _out(df: pd.DataFrame) -> M.Out:
    return M.Out(df["logit_A"].to_numpy(float), df["logit_B"].to_numpy(float), df["lse"].to_numpy(float),
                 df["top1_id"].to_numpy(), np.stack(df["cand_logits"].to_numpy()) if "cand_logits" in df else None)


def summarize(pp: pd.DataFrame, filters: List[str], n_boot: int = 1000, seed: int = 0, cluster_col: str = None) -> pd.DataFrame:
    """One row per (cell, filter, condition key): metrics with paired-bootstrap 95% intervals over pairs.
    Baselines: clean_A from stage 1 ('clean_A'), clean B from the same stage-3 path ('clean_B')."""
    rows = []
    for cell, g in pp.groupby(CELL, sort=False):
        for flt in filters:
            sel = g[g[flt]] if flt in g else g
            base = sel[sel.condition == "clean_A"].set_index("pair_id")
            baseB = sel[sel.condition == "clean_B"].set_index("pair_id")
            pairs = base.index.intersection(baseB.index)
            if len(pairs) == 0:
                continue
            A, B = _out(base.loc[pairs]), _out(baseB.loc[pairs])
            ansA = base.loc[pairs, "ans_A_id"].to_numpy()
            ansB = base.loc[pairs, "ans_B_id"].to_numpy()
            clusters = base.loc[pairs, cluster_col].to_numpy() if cluster_col else None
            for key, c in sel[~sel.condition.isin(["clean_A"])].groupby(KEY[3:], sort=False, dropna=False):
                c = c.set_index("pair_id")
                if "donor_status" in c:
                    c = c[c["donor_status"].isin(["ok", "relaxed", "n/a"])]
                common = pairs.intersection(c.index)
                if len(common) == 0:
                    continue
                pos = pairs.get_indexer(common)
                P = _out(c.loc[common])

                def stat(idx, pos=pos, P=P):
                    return {k: v for k, v in M.per_condition(P.take(idx), A.take(pos[idx]), B.take(pos[idx]),
                                                             ansA[pos[idx]], ansB[pos[idx]]).items()}
                pt = stat(np.arange(len(common)))
                bs = St.bootstrap(lambda idx: {k: pt_ for k, pt_ in stat(idx).items() if k in BOOT_METRICS},
                                  len(common), n_boot, seed, None if clusters is None else clusters[pos]) if n_boot else {}
                row = dict(zip(CELL, cell), filter=flt, **dict(zip(KEY[3:], key)), n_pairs=len(common), **pt)
                for k, v in bs.items():
                    row[f"{k}_lo"], row[f"{k}_hi"] = v["lo"], v["hi"]
                rows.append(row)
    return pd.DataFrame(rows)


def add_gaps(summary: pd.DataFrame) -> pd.DataFrame:
    """Gap of each condition against full_A within its (cell, filter); share closed for repair-like conditions
    against their base (repair <-> feat_A_errB, repair_mean <-> feat_A_meanPos)."""
    out = []
    for _, g in summary.groupby(CELL + ["filter"], sort=False):
        full = g[g.condition == "full_A"]
        if full.empty:
            out.append(g)
            continue
        fA = full.iloc[0].to_dict()
        g = g.copy()
        for k in M.GAP_METRICS:
            if k in g:
                g[f"gap_{k}"] = fA[k] - g[k]
        for cond, base in (("repair", "feat_A_errB"), ("repair_mean", "feat_A_meanPos")):
            b = g[g.condition == base]
            if b.empty:
                continue
            bd = b.iloc[0].to_dict()
            m = g.condition == cond
            for k in M.GAP_METRICS:
                if k in g:
                    gap = fA[k] - bd[k]
                    g.loc[m, f"share_closed_{k}"] = (g.loc[m, k] - bd[k]) / gap if abs(gap) > 1e-8 else np.nan
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else summary


def random_distribution(summary: pd.DataFrame, metric: str = "nldr") -> pd.DataFrame:
    """Per (cell, filter, condition, r): mean / 95th percentile of `metric` over random-direction seeds."""
    r = summary[summary.estimator == "random"]
    rows = []
    for key, g in r.groupby(CELL + ["filter", "condition", "r"], sort=False):
        rows.append(dict(zip(CELL + ["filter", "condition", "r"], key), metric=metric, **St.distribution(g[metric])))
    return pd.DataFrame(rows)
