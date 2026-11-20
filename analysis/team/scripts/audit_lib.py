"""Claims-audit helpers (independent re-implementation; no imports from team/deep/hw-analysis scripts or harness).
Metric definitions follow harness/metrics.py + harness/e8.py as documented in the brief:
  flip = share of eligible pairs (clean_B top1 == ans_B) whose top1 == ans_A
  NLDR = (mean D_p - mean D_B) / (mean D_A - mean D_B), D = logit_A - logit_B (ratio of means)
Bootstrap: pair resampling, percentile CI."""
import glob
import json
import os

import numpy as np
import pandas as pd

SP = "/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad"
RUNS = f"{SP}/ana_runs"
OUT = f"{SP}/team/audit_out"
os.makedirs(OUT, exist_ok=True)
COLS = ["pair_id", "task", "layer", "sae", "condition", "estimator", "source", "r", "seed", "ans_A_id", "ans_B_id",
        "fit_families", "family", "f_none", "f_A_correct", "f_both_correct", "f_pilot", "logit_A", "logit_B", "lse", "top1_id"]


def run_dir(exp, h=None):
    d = sorted(glob.glob(f"{RUNS}/{exp}/*/"))
    if h:
        d = [x for x in d if h in x]
    return d[0]


def load(exp, h=None, extra=(), filters=None):
    f = run_dir(exp, h) + "per_pair.parquet"
    import pyarrow.parquet as pq
    have = set(pq.read_schema(f).names)
    cols = [c for c in list(COLS) + list(extra) if c in have]
    d = pd.read_parquet(f, columns=cols, filters=filters)
    for c in ("estimator", "source"):
        if c in d:
            d[c] = d[c].fillna("")
    return d


class Cell:
    """One (task, layer, sae, fit_families[, family]) cell under one filter: aligned per-pair arrays per condition key."""

    def __init__(self, g, flt="f_pilot"):
        g = g[g[flt]] if flt else g
        self.g = g
        ca = g[g.condition == "clean_A"].drop_duplicates("pair_id").set_index("pair_id").sort_index()
        cb = g[g.condition == "clean_B"].drop_duplicates("pair_id").set_index("pair_id").loc[ca.index]
        self.pairs = ca.index
        self.ansA = ca.ans_A_id.to_numpy()
        self.ansB = ca.ans_B_id.to_numpy()
        self.DA = (ca.logit_A - ca.logit_B).to_numpy()
        self.DB = (cb.logit_A - cb.logit_B).to_numpy()
        self.elig = cb.top1_id.to_numpy() == self.ansB
        self.ca, self.cb = ca, cb

    def get(self, condition, estimator="", r=None, seed=None, source=None):
        q = self.g[(self.g.condition == condition)]
        if estimator is not None:
            q = q[q.estimator == estimator]
        if r is not None:
            q = q[q.r == r]
        if seed is not None:
            q = q[q.seed == seed]
        if source is not None:
            q = q[q.source == source]
        q = q.drop_duplicates("pair_id").set_index("pair_id")
        return q.reindex(self.pairs)

    def flips(self, f):
        """per-pair flip indicator (float) on eligible pairs (NaN elsewhere)."""
        x = (f.top1_id.to_numpy() == self.ansA).astype(float)
        return np.where(self.elig, x, np.nan)

    def D(self, f):
        return (f.logit_A - f.logit_B).to_numpy()


def flip_rate(c, f, idx=None):
    x = c.flips(f)
    if idx is not None:
        x = x[idx]
    return np.nanmean(x)


def nldr(c, f, idx=None):
    D = c.D(f)
    i = slice(None) if idx is None else idx
    return (D[i].mean() - c.DB[i].mean()) / (c.DA[i].mean() - c.DB[i].mean())


def boot(stat, n, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    pt = stat(np.arange(n))
    s = np.array([stat(rng.integers(0, n, n)) for _ in range(B)])
    s = s[np.isfinite(s)]
    return pt, np.percentile(s, 2.5), np.percentile(s, 97.5), s


def bh(p):
    p = np.asarray(p, float)
    m = len(p)
    o = np.argsort(p)
    q = np.empty(m)
    run = 1.0
    for k in range(m - 1, -1, -1):
        run = min(run, p[o[k]] * m / (k + 1))
        q[o[k]] = run
    return q


def orth(U):
    return np.linalg.qr(np.asarray(U, np.float64))[0]


def pcos(A, B):
    return np.linalg.svd(orth(A).T @ orth(B), compute_uv=False)


_INIT = {}


def das_init(d, k, seed):
    """Independent re-implementation of harness Rotation(d,k,seed) initial weight: QR of randn(d,k) from a seeded torch
    Generator, then torch's orthogonal parametrization (its initial value is checked against the plain QR in audit_das.py)."""
    import torch
    from torch import nn
    if (d, k, seed) not in _INIT:
        g = torch.Generator().manual_seed(seed)
        lin = nn.Linear(k, d, bias=False)
        with torch.no_grad():
            lin.weight.copy_(torch.linalg.qr(torch.randn(d, k, generator=g))[0])
        lin = nn.utils.parametrizations.orthogonal(lin)
        _INIT[(d, k, seed)] = lin.weight.detach().double().numpy()
    return _INIT[(d, k, seed)]


def parse_das(name):
    p = {"k": None, "epochs": 3, "lr": 1e-2, "seed": 0}
    parts = name.split("_")
    p["k"] = int(parts[0][3:])
    for t in parts[1:]:
        if t.startswith("e") and t[1:].isdigit():
            p["epochs"] = int(t[1:])
        elif t.startswith("lr"):
            p["lr"] = float(t[2:])
        elif t.startswith("s") and t[1:].isdigit():
            p["seed"] = int(t[1:])
    return p


def learned(U, name):
    p = parse_das(name)
    I = das_init(U.shape[0], p["k"], p["seed"])
    U = U.astype(np.float64)
    cap = float(((I.T @ U) ** 2).sum() / U.shape[1])
    R = U - I @ (I.T @ U)
    q, s, _ = np.linalg.svd(R, full_matrices=False)
    return q[:, s > 1e-6 * s.max()], cap


def capture(A, B):
    Qa, Qb = orth(A), orth(B)
    return float(((Qb.T @ Qa) ** 2).sum() / Qa.shape[1])


def directions(exp, stem_glob="*"):
    out = {}
    for f in glob.glob(f"{run_dir(exp)}directions/{stem_glob}.npz"):
        z = np.load(f)
        j = json.load(open(f[:-4] + ".json"))
        out[os.path.basename(f)[:-4]] = (z, j)
    return out
