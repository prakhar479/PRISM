"""Shared loaders for the per-pair mechanism analyses (exploratory, post hoc, DEV only)."""
import glob
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

SP = __import__("layout_shim").root()
RUNS = SP / "ana_runs"
PS = SP / "hw" / "data" / "prompt_sets"


def run_dirs(exp):
    out = []
    for m in sorted(glob.glob(str(RUNS / exp / "*" / "manifest.json"))):
        j = json.load(open(m))
        if j.get("status") == "complete" and not j.get("smoke") and Path(m).with_name("per_pair.parquet").exists():
            out.append(Path(m).parent)
    return out


def load(exp, cols=None):
    ds = run_dirs(exp)
    return pd.concat([pd.read_parquet(d / "per_pair.parquet", columns=cols).assign(run=d.name) for d in ds], ignore_index=True)


def ckey(df):
    return (df.condition + "|" + df.estimator.astype(str) + "|" + df.source.astype(str) + "|" + df.r.astype(str)
            + "|" + df.seed.astype(str))


def prompts(name):
    rows = [json.loads(l) for l in gzip.open(PS / f"{name}.jsonl.gz", "rt")]
    df = pd.DataFrame(rows)
    df["meta"] = df["meta"].apply(lambda s: eval(s) if isinstance(s, str) else s)
    return df


def wide(pp, group=("run", "task", "layer", "sae", "fit_families", "family")):
    """Per pair: D = logit_A - logit_B, top1 per condition key; plus references clean_B, full_A."""
    pp = pp.copy()
    if {"source", "r", "seed"} <= set(pp.columns):
        pp["key"] = ckey(pp)
    pp["D"] = pp.logit_A - pp.logit_B
    pp["isA"] = pp.top1_id == pp.ans_A_id
    pp["isB"] = pp.top1_id == pp.ans_B_id
    return pp


def boot_ci(x, f=np.mean, n=2000, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    v = np.array([f(x[i]) for i in idx])
    return float(f(x)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def boot_ratio(num, den, n=2000, seed=0):
    num, den = np.asarray(num), np.asarray(den)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (n, len(num)))
    v = num[idx].mean(1) / den[idx].mean(1)
    return float(num.mean() / den.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def logit_irls(X: pd.DataFrame, y, iters=50, ridge=1e-6):
    """Plain logistic regression by IRLS; returns a table of coef, se, z, p (Wald) and odds ratio. X gets an intercept."""
    from scipy.stats import norm
    Xm = np.column_stack([np.ones(len(X)), X.values.astype(float)])
    names = ["const"] + list(X.columns)
    b = np.zeros(Xm.shape[1]); y = np.asarray(y, float)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xm @ b)); W = p * (1 - p)
        H = Xm.T @ (Xm * W[:, None]) + ridge * np.eye(len(b))
        step = np.linalg.solve(H, Xm.T @ (y - p)); b += step
        if np.abs(step).max() < 1e-8: break
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    z = b / se
    return pd.DataFrame({"coef": b, "se": se, "z": z, "p": 2 * norm.sf(np.abs(z)), "OR": np.exp(b)}, index=names)


def ols(X: pd.DataFrame, y):
    from scipy.stats import t as tdist
    Xm = np.column_stack([np.ones(len(X)), X.values.astype(float)]); y = np.asarray(y, float)
    b, *_ = np.linalg.lstsq(Xm, y, rcond=None); res = y - Xm @ b
    dof = len(y) - Xm.shape[1]; s2 = res @ res / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(Xm.T @ Xm)))
    r2 = 1 - res @ res / ((y - y.mean()) @ (y - y.mean()))
    return pd.DataFrame({"coef": b, "se": se, "p": 2 * tdist.sf(np.abs(b / se), dof)}, index=["const"] + list(X.columns)), r2
