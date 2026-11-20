"""Schemas of run outputs (documented in tables/SCHEMA.md). Used by ingest and the tests."""
from __future__ import annotations

from typing import List

import pandas as pd

PER_PAIR = {
    "pair_id": "str", "split": "str", "family": "str", "task": "str", "layer": "int", "sae": "str",
    "condition": "str", "estimator": "str", "source": "str", "r": "int", "seed": "int",
    "ans_A_id": "int", "ans_B_id": "int", "donor_status": "str",
    "f_none": "bool", "f_A_correct": "bool", "f_both_correct": "bool", "f_pilot": "bool",
    "logit_A": "float", "logit_B": "float", "lse": "float", "top1_id": "int", "cand_logits": "list",
}
# optional columns (absent from release-1 outputs; readers fill the default)
OPTIONAL_PER_PAIR = {"fit_families": ("str", "pilot")}
MANIFEST_KEYS = ("exp_id", "run_hash", "status", "config", "profile", "git", "packages", "machine", "model",
                 "sae_revision", "prompt_sets", "seeds", "timings", "smoke", "tiny", "wall_time_s")


def validate_per_pair(df: pd.DataFrame) -> List[str]:
    errs = []
    for col, kind in PER_PAIR.items():
        if col not in df:
            errs.append(f"missing column {col}")
            continue
        s = df[col]
        ok = {"str": lambda s: s.map(lambda v: isinstance(v, str)).all(),
              "int": lambda s: pd.api.types.is_integer_dtype(s),
              "bool": lambda s: pd.api.types.is_bool_dtype(s),
              "float": lambda s: pd.api.types.is_float_dtype(s),
              "list": lambda s: s.map(lambda v: hasattr(v, "__len__")).all()}[kind](s)
        if not ok:
            errs.append(f"column {col} is not {kind}")
    for col, (kind, _) in OPTIONAL_PER_PAIR.items():
        if col in df and not df[col].map(lambda v: isinstance(v, str)).all():
            errs.append(f"column {col} is not {kind}")
    key = ["pair_id", "task", "layer", "sae", "condition", "estimator", "source", "r", "seed"]
    key += [c for c in OPTIONAL_PER_PAIR if c in df]
    if not errs and df.duplicated(key).any():
        errs.append("duplicate (pair, condition key) rows")
    return errs


def validate_manifest(m: dict) -> List[str]:
    errs = [f"manifest missing {k}" for k in MANIFEST_KEYS if k not in m]
    if m.get("tiny"):
        errs.append("tiny-model run (test plumbing only; never ingested)")
    return errs
