"""Independent re-implementation of the per-cell metric bootstrap (no harness imports), used by stats_tests.py and
stats_ci_audit.py. Definitions copied from the BRIEF / harness docstrings (not the code): flip = share of eligible pairs (clean B
top-1 = B's answer) whose top-1 is A's answer; NLDR = ratio of means (mean D_p - mean D_B)/(mean D_A - mean D_B); gap = M(full_A)
- M(cond). Pair bootstrap: resample pairs with replacement (rng = np.random.default_rng(seed).integers(0, n, (B, n)), the same
generator scheme as harness/stats.boot_indices so that the audit can match exactly)."""
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

COLS = ["pair_id", "split", "family", "task", "layer", "sae", "condition", "estimator", "source", "r", "seed", "ans_A_id",
        "ans_B_id", "fit_families", "f_none", "f_A_correct", "f_pilot", "logit_A", "logit_B", "top1_id", "lse"]
KEY = ["condition", "estimator", "source", "r", "seed"]


def load(path, extra_filter=None):
    t = ds.dataset(path).to_table(columns=COLS, filter=extra_filter).to_pandas()
    t = t[t.split == "test"].copy()
    fam = t.family.fillna("pilot").astype(str)
    t["family"] = np.where(fam.str.startswith("pilot"), "pilot", fam)
    t["fit_families"] = t.fit_families.fillna("pilot")
    return t


class Cell:
    """One (run, task, layer, sae, fit, eval family, filter)."""

    def __init__(self, g, flt):
        sel = g[g[flt]] if flt in g else g
        self.frames = {}
        for k, f in sel.groupby(KEY, sort=False, dropna=False):
            self.frames[tuple(k)] = f.drop_duplicates("pair_id").set_index("pair_id")
        base = {k[0]: f for k, f in self.frames.items() if k[0] in ("clean_A", "clean_B", "full_A")}
        self.ok = set(base) == {"clean_A", "clean_B", "full_A"}
        if not self.ok:
            return
        self.pairs = base["clean_A"].index.intersection(base["clean_B"].index).intersection(base["full_A"].index).sort_values()
        ca, cb = base["clean_A"].loc[self.pairs], base["clean_B"].loc[self.pairs]
        self.ansA = ca.ans_A_id.to_numpy()
        self.DA = (ca.logit_A - ca.logit_B).to_numpy(float)
        self.DB = (cb.logit_A - cb.logit_B).to_numpy(float)
        self.elig = (cb.top1_id.to_numpy() == cb.ans_B_id.to_numpy()).astype(float)
        self.n = len(self.pairs)

    def arrays(self, key, pairs=None):
        f = self.frames.get(key)
        if f is None:
            return None
        f = f.reindex(self.pairs if pairs is None else pairs)
        if f.top1_id.isna().any():
            return None
        return (f.logit_A - f.logit_B).to_numpy(float), (f.top1_id.to_numpy() == self.ansA).astype(float)

    def keys(self):
        return list(self.frames)


def boot_W(n, B=2000, seed=0):
    idx = np.random.default_rng(seed).integers(0, n, size=(B, n))
    W = np.zeros((B + 1, n))
    W[0] = 1
    for b in range(B):
        W[b + 1] = np.bincount(idx[b], minlength=n)
    return W


def samples(W, D, hit, DA, DB, elig):
    tot = W.sum(1)
    m = lambda x: (W @ x) / tot
    nldr = (m(D) - m(DB)) / (m(DA) - m(DB) + 1e-8)
    ne = W @ elig
    with np.errstate(invalid="ignore", divide="ignore"):
        fl = np.where(ne > 0, (W @ (hit * elig)) / np.maximum(ne, 1e-12), np.nan)
    return {"flip_rate": fl, "nldr": nldr}


def ci(s, level=0.95):
    s = s[np.isfinite(s)]
    a = (1 - level) / 2
    return (float(np.quantile(s, a)), float(np.quantile(s, 1 - a))) if len(s) else (np.nan, np.nan)
