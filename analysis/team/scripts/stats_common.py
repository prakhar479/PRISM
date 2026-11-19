"""Shared helpers for the statistician's scripts (stats_*.py). Pure numpy/pandas (no statsmodels on this machine)."""
import glob
import json
import os

import numpy as np
import pandas as pd

SP = "/tmp/claude-1002/-home-prakhar-dev-legend479-MILP-prism-sae-dark-matter/218361b4-64de-4a4e-b6e1-5d9977ad0da5/scratchpad"
RUNS = f"{SP}/ana_runs"
OUT = f"{SP}/team/stats_out"
os.makedirs(OUT, exist_ok=True)

NATIVE = ["hi_deva_dakshina", "ta_taml_dakshina", "bn_beng_dakshina", "hi_deva_web", "ta_web", "bn_web"]
LATIN = ["hi_latn_dakshina", "ta_latn_dakshina", "bn_latn_dakshina", "en_pile", "fr_web", "hinglish_phinc"]
DAK = {"hi": ("hi_deva_dakshina", "hi_latn_dakshina"), "ta": ("ta_taml_dakshina", "ta_latn_dakshina"),
       "bn": ("bn_beng_dakshina", "bn_latn_dakshina")}


def e12_windows():
    """All E12/E21 per-window arrays -> long DataFrame (one row per set, layer, width, window)."""
    rows = []
    for d in sorted(glob.glob(f"{RUNS}/E12/*/")):
        m = json.load(open(d + "manifest.json"))
        width = m["config"]["width"]
        host = m["machine"]["hostname"]
        for f in glob.glob(d + "per_window_*.npz"):
            b = os.path.basename(f)[len("per_window_"):-4]
            s, L = b.rsplit("_L", 1)
            if s == "en_pilot100":
                continue
            z = np.load(f)
            df = pd.DataFrame({k: z[k] for k in z.files})
            df["set"], df["layer"], df["width"], df["host"], df["run"] = s, int(L), width, host, m["run_hash"]
            df["win"] = np.arange(len(df))
            rows.append(df)
    w = pd.concat(rows, ignore_index=True)
    w["cost"] = (w.nll_splice - w.nll_clean) / w.pred
    w["fvu"] = w.fvu_num / w.fvu_den
    w["eps"] = w.eps_ratio_sum / w.pos
    w["nllc"] = w.nll_clean / w.pred
    w["native"] = w.set.isin(NATIVE).astype(int)
    return w


def ols(X, y, cluster=None):
    """OLS with HC1 (cluster=None) or CR1 cluster-robust covariance. Returns beta, se."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    if cluster is None:
        meat = (X * e[:, None]).T @ (X * e[:, None])
        V = XtXi @ meat @ XtXi * n / (n - k)
    else:
        cl = pd.factorize(np.asarray(cluster))[0]
        G = cl.max() + 1
        S = np.zeros((G, k))
        np.add.at(S, cl, X * e[:, None])
        V = XtXi @ (S.T @ S) @ XtXi * (G / (G - 1)) * ((n - 1) / (n - k))
    return b, np.sqrt(np.diag(V))


def wild_cluster_p(X, y, cluster, j, B=None, seed=0):
    """Wild cluster bootstrap (Rademacher, restricted under H0: beta_j = 0) p-value for coefficient j, using CR1 t.
    With G <= 12 clusters all 2^G sign vectors are enumerated."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    cl = pd.factorize(np.asarray(cluster))[0]
    G = cl.max() + 1
    b, se = ols(X, y, cl)
    t0 = b[j] / se[j]
    Xr = np.delete(X, j, axis=1)
    br = np.linalg.lstsq(Xr, y, rcond=None)[0]
    fit_r, er = Xr @ br, y - Xr @ br
    if G <= 12:
        signs = ((np.arange(2 ** G)[:, None] >> np.arange(G)) & 1) * 2 - 1
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1, 1], size=(B or 999, G))
    ts = []
    for s in signs:
        ys = fit_r + er * s[cl]
        bb, ss = ols(X, ys, cl)
        ts.append(bb[j] / ss[j])
    ts = np.array(ts)
    return float(np.mean(np.abs(ts) >= abs(t0))), float(t0)


def bh(p):
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(p, float)
    n = np.sum(np.isfinite(p))
    q = np.full_like(p, np.nan)
    idx = np.where(np.isfinite(p))[0]
    o = idx[np.argsort(p[idx])]
    ranked = p[o] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q[o] = np.minimum(ranked, 1)
    return q
