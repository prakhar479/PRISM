"""Per-pair loader shared by stats_variance.py / stats_power.py / stats_multiplicity.py / stats_ci_audit.py."""
import glob
import json

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from stats_common import RUNS

COLS = ["pair_id", "split", "family", "task", "layer", "sae", "condition", "estimator", "source", "r", "seed", "ans_A_id",
        "ans_B_id", "fit_families", "f_none", "f_A_correct", "f_pilot", "logit_A", "logit_B", "lse", "top1_id"]


def run_dirs(exp):
    return sorted(d for d in glob.glob(f"{RUNS}/{exp}/*/") if glob.glob(d + "per_pair.parquet"))


def host(d):
    return json.load(open(d + "manifest.json"))["machine"]["hostname"]


def load_cell(d, task, layer, sae, fit=None, family=None):
    """Per-pair rows of one run/cell, test split, with per-pair D, hitA and the clean A/B baselines attached.
    Returns long frame: key cols + pair_id, family, D, hitA, DA, DB, elig, nldr_i (per-pair NLDR), filters."""
    f = (ds.field("sae") == sae) & (ds.field("task") == task) & (ds.field("layer") == layer)
    t = ds.dataset(d + "per_pair.parquet").to_table(columns=COLS, filter=f).to_pandas()
    t = t[t.split == "test"]
    if fit is not None:
        t = t[t.fit_families.fillna("pilot") == fit]
    if family is not None:
        t = t[t.family.fillna("pilot").astype(str).str.startswith(family)]
    t = t.copy()
    t["estimator"] = t.estimator.fillna("")
    t["source"] = t.source.fillna("")
    t["D"] = t.logit_A - t.logit_B
    t["hitA"] = (t.top1_id == t.ans_A_id).astype(float)
    t = t.drop_duplicates(["condition", "estimator", "source", "r", "seed", "pair_id"])
    ca = t[t.condition == "clean_A"].set_index("pair_id")
    cb = t[t.condition == "clean_B"].set_index("pair_id")
    base = pd.DataFrame({"DA": ca.D, "DB": cb.D, "elig": (cb.top1_id == cb.ans_B_id).astype(float)}).dropna()
    t = t.join(base, on="pair_id", how="inner")
    t["nldr_i"] = (t.D - t.DB) / (t.DA - t.DB)
    t["key"] = (t.condition + "|" + t.estimator + "|" + t.source + "|" + t.r.astype(str) + "|" + t.seed.astype(str))
    return t.drop(columns=["split", "logit_A", "logit_B", "lse"])


def flip(t):
    e = t.elig > 0
    return float(t.hitA[e].mean())


def nldr_rom(t):
    return float((t.D.mean() - t.DB.mean()) / (t.DA.mean() - t.DB.mean() + 1e-8))
