"""Shared loaders for the PATTERN HUNTER analyses (exploratory, post hoc). Read-only on $SP/ana_runs and $SP/hw."""
import glob
import gzip
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SP = __import__("layout_shim").root()
RUNS = SP / "ana_runs"
PS = SP / "hw" / "data" / "prompt_sets"
OUT = SP / "team" / "patterns_out"
OUT.mkdir(parents=True, exist_ok=True)

BASE_COLS = ["pair_id", "family", "task", "layer", "sae", "condition", "estimator", "source", "r", "seed",
             "ans_A_id", "ans_B_id", "f_A_correct", "f_both_correct", "f_pilot", "logit_A", "logit_B", "lse", "top1_id"]
CELL = ["run", "task", "layer", "sae", "family", "pair_id"]


def run_dirs(exp):
    return sorted(Path(m).parent for m in glob.glob(str(RUNS / exp / "*" / "per_pair.parquet")))


def load(exp, cols=BASE_COLS, cand=False, filt=None):
    cols = list(cols) + (["cand_logits"] if cand else [])
    out = []
    for d in run_dirs(exp):
        df = pd.read_parquet(d / "per_pair.parquet", columns=cols)
        if filt is not None:
            df = df[filt(df)]
        out.append(df.assign(run=d.name, exp=exp))
    df = pd.concat(out, ignore_index=True)
    df["D"] = df.logit_A - df.logit_B
    df["isA"] = df.top1_id == df.ans_A_id
    df["isB"] = df.top1_id == df.ans_B_id
    return df


def add_refs(df):
    """Attach per-pair clean_A / clean_B D and top1 and compute per-pair NLDR = (D - D_B)/(D_A - D_B)."""
    key = ["run", "task", "layer", "sae", "pair_id"]
    a = df[df.condition == "clean_A"][key + ["D", "isA"]].rename(columns={"D": "D_A", "isA": "A_ok"})
    b = df[df.condition == "clean_B"][key + ["D", "isB"]].rename(columns={"D": "D_B", "isB": "B_ok"})
    a = a.drop_duplicates(key); b = b.drop_duplicates(key)
    df = df.merge(a, on=key, how="left").merge(b, on=key, how="left")
    df["den"] = df.D_A - df.D_B
    df["nldr"] = (df.D - df.D_B) / df.den
    df["eligible"] = df.B_ok
    return df


def ckey(df):
    return (df.condition + "|" + df.estimator.astype(str) + "|r" + df.r.astype(str))


# ------------------------------------------------------------------ prompt features
_prompt_cache = {}


def prompts(name):
    if name not in _prompt_cache:
        rows = [json.loads(l) for l in gzip.open(PS / f"{name}.jsonl.gz", "rt")]
        _prompt_cache[name] = {r["pair_id"]: r for r in rows}
    return _prompt_cache[name]


def _last_line(p):
    return p.rstrip("\n").split("\n")[-1]


def features(task, pair_ids, set_name):
    """Per-pair prompt features from the prompt file used by the run."""
    P = prompts(set_name)
    rows = []
    for pid in pair_ids:
        r = P[pid]
        m = r["meta"] if isinstance(r["meta"], dict) else eval(r["meta"])
        f = {"pair_id": pid, "corrupt_step": m.get("corrupt_step"), "n_patched": r["T"] - r["first_diff"],
             "T": r["T"], "ans_A_tok": r["clean_target"].strip(), "ans_B_tok": r["corrupt_target"].strip(),
             "template": m.get("template", r.get("family"))}
        lc, lb = _last_line(r["clean_prompt"]), _last_line(r["corrupt_prompt"])
        if task == "add_chain":
            nc = [int(x) for x in re.findall(r"\d+", lc)]
            nb = [int(x) for x in re.findall(r"\d+", lb)]
            f.update(start=nc[0], incs=tuple(nc[1:]), inc_A=nc[1 + f["corrupt_step"]] if len(nc) > 1 + f["corrupt_step"] else np.nan,
                     inc_B=nb[1 + f["corrupt_step"]] if len(nb) > 1 + f["corrupt_step"] else np.nan,
                     ans_A=int(r["state_clean"]), ans_B=int(r["state_corrupt"]))
            f["delta"] = f["ans_A"] - f["ans_B"]
        elif task == "entity_track":
            # init: "The X is in box L." x3 ; moves: "The X moves to box L." x2 ; query "The X is in box"
            inits = re.findall(r"The (\w+) is in box (\w)\.", lc)
            moves_c = re.findall(r"The (\w+) moves to box (\w)", lc)
            moves_b = re.findall(r"The (\w+) moves to box (\w)", lb)
            q = re.findall(r"The (\w+) is in box$", lc.strip())
            q = q[0] if q else None
            f.update(query=q, n_moves=len(moves_c),
                     q_moved_in_A=sum(o == q for o, _ in moves_c), q_moved_in_B=sum(o == q for o, _ in moves_b),
                     q_init_pos=[o for o, _ in inits].index(q) if q in [o for o, _ in inits] else -1,
                     A_is_init=dict(inits).get(q) == f["ans_A_tok"])
        elif task == "people_cities":
            stm = re.findall(r"(\w+)'s home is in (\w+)", lc)
            q = re.findall(r"(\w+)'s home is in$", lc.strip())
            f.update(query=q[0] if q else None, q_pos=m.get("corrupt_step"), cityA=f["ans_A_tok"], cityB=f["ans_B_tok"])
        rows.append(f)
    return pd.DataFrame(rows)


def set_name_for(exp_dir, task):
    m = json.load(open(Path(exp_dir) / "manifest.json"))
    p = m["config"]["prompt_sets"][task]
    return Path(p).name.replace(".jsonl.gz", "")


def boot_ci(x, f=np.mean, n=2000, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    v = np.array([f(x[i]) for i in idx])
    return float(f(x)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    q = p[o] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1)
    return out
