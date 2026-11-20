"""Shared loaders for the steering / dose-response analysis (role: steering). Exploratory, post hoc, DEV only.

Per-pair rows from E19 / E10 / E10b (steering) and E9 / E17 / E18 (patching along U) are turned into a tidy frame with:
  D = logit_A - logit_B, p_A, p_B, p_cand (mass on all candidate answers), top-1 outcome (A / B / other candidate / non-cand),
  eligibility (clean B top-1 == B's answer), and per-task roles of the top-1 candidate.
Metric conventions follow harness/metrics.py (flip on eligible pairs; NLDR = ratio of means vs clean_A / clean_B).
"""
import glob
import gzip
import json
import os
import re
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd

SP = __import__("layout_shim").root()
RUNS = SP / "ana_runs"
PS = SP / "hw" / "data" / "prompt_sets"
OUT = SP / "team" / "steering_out"
OUT.mkdir(parents=True, exist_ok=True)
PROMPTS = {"add_chain": "add_chain_pilot", "entity_track": "entity_track_pilot", "people_cities": "people_cities_en"}
COLS = ["pair_id", "task", "layer", "family", "condition", "estimator", "source", "r", "seed", "ans_A_id", "ans_B_id",
        "f_none", "f_A_correct", "f_pilot", "logit_A", "logit_B", "lse", "top1_id", "cand_logits"]


def run_dir(exp):
    ds = [Path(m).parent for m in sorted(glob.glob(str(RUNS / exp / "*" / "manifest.json")))
          if json.load(open(m)).get("status") == "complete"]
    assert len(ds) == 1, (exp, ds)
    return ds[0]


def prompts(task):
    rows = [json.loads(l) for l in gzip.open(PS / f"{PROMPTS[task]}.jsonl.gz", "rt")]
    df = pd.DataFrame(rows)
    df["meta"] = df["meta"].apply(lambda s: eval(s) if isinstance(s, str) else s)
    return df.set_index("pair_id")


def scale_of(cond):
    m = re.match(r"^steer_x(\d+(?:\.\d+)?)$", str(cond))
    return float(m.group(1)) if m else np.nan


def load(exp, tasks=None, conditions=None):
    pp = pd.read_parquet(run_dir(exp) / "per_pair.parquet", columns=COLS)
    pp = pp[pp.layer.isin([12, 18])] if exp in ("E18",) else pp
    if tasks is not None:
        pp = pp[pp.task.isin(tasks)]
    if conditions is not None:
        pp = pp[pp.condition.isin(conditions)]
    pp = pp.copy()
    pp["estimator"] = pp.estimator.fillna("").astype(str)
    pp["s"] = pp.condition.map(scale_of)
    pp["D"] = pp.logit_A - pp.logit_B
    pp["p_A"] = np.exp(pp.logit_A - pp.lse)
    pp["p_B"] = np.exp(pp.logit_B - pp.lse)
    C = np.stack(pp.cand_logits.to_numpy()) if pp.cand_logits.map(len).nunique() == 1 else None
    if C is None:                                       # pad to the max number of candidates
        k = pp.cand_logits.map(len).max()
        C = np.full((len(pp), k), np.nan)
        for i, v in enumerate(pp.cand_logits.to_numpy()):
            C[i, :len(v)] = v
    c = np.where(np.isfinite(C), C, -np.inf)
    mx = c.max(1, keepdims=True)
    pp["p_cand"] = np.exp(mx[:, 0] + np.log(np.exp(c - mx).sum(1)) - pp.lse.to_numpy())
    pp["p_other_cand"] = pp.p_cand - pp.p_A - pp.p_B
    pp["isA"] = pp.top1_id == pp.ans_A_id
    pp["isB"] = pp.top1_id == pp.ans_B_id
    pp = pp.drop(columns=["cand_logits"])
    return pp


def attach_refs(pp, keycols=("task", "layer")):
    """Add clean_A / clean_B references per pair: D_A, D_B, pA_A, pA_B, elig (clean_B top-1 == B's answer)."""
    ref = pp[pp.condition.isin(["clean_A", "clean_B"])]
    ref = ref.drop_duplicates(list(keycols) + ["pair_id", "condition"])
    w = ref.pivot_table(index=list(keycols) + ["pair_id"], columns="condition", values=["D", "p_A", "top1_id"], aggfunc="first")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    out = pp.merge(w, on=list(keycols) + ["pair_id"], how="left")
    out["elig"] = out["top1_id_clean_B"] == out["ans_B_id"]
    return out


def flip(g):
    e = g[g.elig]
    return float(e.isA.mean()) if len(e) else np.nan


def nldr(g):
    return float((g.D.mean() - g.D_clean_B.mean()) / (g.D_clean_A.mean() - g.D_clean_B.mean()))


def prob_rec(g):
    return float((g.p_A.mean() - g.p_A_clean_B.mean()) / (g.p_A_clean_A.mean() - g.p_A_clean_B.mean()))


def wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)
