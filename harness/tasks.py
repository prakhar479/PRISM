"""Prompt sets (Stage 0): the pilot's candidate pools rebuilt with the pilot generators, kept UNFILTERED, plus
template families, token-level slot labels, candidate answer ids and filter flags.

The pilot (src/data/synthetic_tasks.build_task_splits) generated n * candidate_multiplier candidates per split,
tokenized them, kept those passing the base-model filter and took the first n. Here every tokenized candidate is kept
in that same order (`rank`); the pilot's pairs are recovered exactly as the first n candidates with f_pilot = True.
Filters are flags computed from clean-run outputs (Stage 1), never deletions:
    f_none          every tokenized candidate
    f_A_correct     top-1 on A is A's answer
    f_both_correct  top-1 on A is A's answer and top-1 on B is B's answer
    f_pilot         the pilot's rule for that task (strict or clean_only, min_logit_diff 2)
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.data import synthetic_tasks as st

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_CONFIG = "configs/paper_residue.yaml"
PILOT_PREFLIGHT = "results/paper_20261026_001341/preflight/preflight.json"
PILOT_TASKS = ["entity_track", "add_chain", "var_binding", "ioi_control"]
CANDIDATE_STRINGS = {"entity_track": [f" {b}" for b in st.BOXES], "add_chain": [f" {d}" for d in range(10)],
                     "var_binding": [f" {d}" for d in range(10)], "ioi_control": [f" {n}" for n in st.NAMES]}
FILTERS = ("f_none", "f_A_correct", "f_both_correct", "f_pilot")


# --------------------------------------------------------------------------- pilot settings
def pilot_cfg() -> dict:
    """configs/paper_residue.yaml + the paper run's preflight task_overrides (what run_full_suite.py merged)."""
    from src.experiments.common import merge_task_overrides
    from src.utils import load_config

    cfg = load_config(str(REPO_ROOT / PILOT_CONFIG))
    ov = json.loads((REPO_ROOT / PILOT_PREFLIGHT).read_text())
    cfg["task_overrides"] = merge_task_overrides(cfg.get("task_overrides"), ov.get("task_overrides"))
    return cfg


def task_settings(cfg: dict, task: str) -> dict:
    from src.experiments.common import task_cfg

    c = task_cfg(cfg, task)
    return {k: c.get(k) for k in ("n_train", "n_test", "seed", "candidate_multiplier", "n_shots", "answer_format",
                                  "filter_mode", "min_logit_diff", "gen_kwargs")}


# --------------------------------------------------------------------------- slots
def _last_line_offset(prompt: str) -> int:
    return prompt.rfind("\n") + 1


def slot_spans(task: str, prompt: str) -> List[Tuple[str, int, int]]:
    """(slot, char_start, char_end) for the semantic slots of the query (last) line of `prompt`."""
    base = _last_line_offset(prompt)
    line = prompt[base:]
    out: List[Tuple[str, int, int]] = []

    def add(name, m, g):
        out.append((name, base + m.start(g), base + m.end(g)))

    if task == "entity_track":
        for k, m in enumerate(re.finditer(r"The (\w+) is in box (\w)\.", line), 1):
            add(f"obj_{k}", m, 1), add(f"box_{k}", m, 2)
        for k, m in enumerate(re.finditer(r"The (\w+) moves to box (\w)\.", line), 1):
            add(f"move_{k}_obj", m, 1), add(f"move_{k}_box", m, 2)
        m = re.search(r"The (\w+) is in box\s*$", line)
        if m:
            add("query_obj", m, 1)
    elif task == "add_chain":
        m = re.match(r"(Start with|Begin with|Start at) (\d)\.", line) or re.match(r"(\w+) has (\d) \w+\.", line)
        if m:
            add("start_num", m, 2)
        for k, m in enumerate(re.finditer(r"(?:Add|gets) (\d)(?: more)?\.", line), 1):
            add(f"add_{k}", m, 1)
        m = re.search(r"((?:Total|Sum|Result) =|now has)\s*$", line)
        if m:
            add("query", m, 1)
    elif task == "var_binding":
        for k, m in enumerate(re.finditer(r"([a-z]) = (\w)\.", line), 1):
            add(f"stmt_{k}_var", m, 1), add(f"stmt_{k}_val", m, 2)
        m = re.search(r"([a-z]) =\s*$", line)
        if m:
            add("query_var", m, 1)
    elif task == "ioi_control":
        m = re.match(r"When (\w+) and (\w+) went to the (\w+), (\w+) gave a (\w+) to", line)
        if m:
            for g, name in enumerate(("name_1", "name_2", "place", "subj", "object"), 1):
                add(name, m, g)
    return out


def token_offsets(tok, text: str) -> List[Tuple[int, int]]:
    """Character span of every token of tok.encode(text) (BOS -> (0, 0))."""
    if isinstance(tok, st.FallbackTokenizer):
        spans = [(0, 0)] + [(m.start(), m.end()) for m in tok._PAT.finditer(text)]
        return spans
    enc = tok(text, return_offsets_mapping=True)
    return [tuple(o) for o in enc["offset_mapping"]]


def token_slots(task: str, tok, prompt: str, n_tokens: int) -> List[str]:
    """Slot label per token: semantic slot of the query line if the token overlaps one, 'final' for the last token,
    else 'off:<k>' (offset from the final token)."""
    offs = token_offsets(tok, prompt)
    if len(offs) != n_tokens:
        raise ValueError(f"offset mapping has {len(offs)} tokens, encode() gave {n_tokens}")
    spans = slot_spans(task, prompt)
    labels = []
    for p, (a, b) in enumerate(offs):
        lab = next((name for name, s, e in spans if b > s and a < e and b > a), None)
        labels.append(lab or f"off:{n_tokens - 1 - p}")
    labels[-1] = "final"
    return labels


def family_of(task: str, pair: st.CounterfactualPair) -> str:
    if task == "add_chain":
        return f"pilot_{pair.meta.get('template', 'calc')}"
    return "pilot"


# --------------------------------------------------------------------------- building
def candidate_ids(tok, prompt: str, target_str: str, task: str) -> List[int]:
    lead = target_str.startswith(" ")
    out = []
    for c in CANDIDATE_STRINGS[task]:
        i = st.answer_token_id(tok, prompt, c if lead else c.lstrip())
        out.append(-1 if i is None else int(i))
    return out


def build_pilot_candidates(task: str, tok, cfg: Optional[dict] = None, n_train: Optional[int] = None,
                           n_test: Optional[int] = None, with_slots: bool = True) -> List[dict]:
    """Every tokenized candidate of the pilot's train and test pools, in pilot order (no model needed)."""
    cfg = cfg or pilot_cfg()
    s = task_settings(cfg, task)
    n_tr, n_te = n_train or s["n_train"], n_test or s["n_test"]
    mult = s["candidate_multiplier"] or 3
    shots = st.draw_shots(task, s["n_shots"] or 0, s["seed"] + 77, gen_kwargs=s["gen_kwargs"])
    exclude = st.shot_prompts(shots)
    raw = {}
    for split, n, seed in (("train", n_tr, s["seed"]), ("test", n_te, s["seed"] + 10_000)):
        raw[split] = st.generate_pairs(task, int(n * mult), seed, exclude=exclude, split=split, gen_kwargs=s["gen_kwargs"])
        exclude |= st.shot_prompts(raw[split])
    rows = []
    for split in ("train", "test"):
        toks = st.tokenize_pairs(tok, st.apply_prefix(raw[split], shots), s["answer_format"] or "auto")
        for rank, it in enumerate(toks):
            p = it.pair
            r = {"pair_id": p.pair_id, "task": task, "split": split, "rank": rank, "family": family_of(task, p),
                 "clean_prompt": p.clean_prompt, "corrupt_prompt": p.corrupt_prompt,
                 "clean_target": p.clean_target_str, "corrupt_target": p.corrupt_target_str,
                 "state_clean": int(p.clean_state_id), "state_corrupt": int(p.corrupt_state_id),
                 "clean_ids": list(map(int, it.clean_ids)), "corrupt_ids": list(map(int, it.corrupt_ids)),
                 "ans_A": int(it.clean_ans), "ans_B": int(it.corrupt_ans), "first_diff": int(it.diff_pos),
                 "T": len(it.clean_ids), "cand_ids": candidate_ids(tok, p.clean_prompt, p.clean_target_str, task),
                 "meta": {k: v for k, v in p.meta.items() if k != "heur"}}
            if with_slots:
                r["slots"] = token_slots(task, tok, p.clean_prompt, len(it.clean_ids))
            rows.append(r)
    return rows


def check_invariants(rows: Sequence[dict]) -> List[str]:
    """Generation-time invariants; returns a list of violations (empty = ok)."""
    bad = []
    prefixes = {}
    for r in rows:
        a, b = r["clean_ids"], r["corrupt_ids"]
        if len(a) != len(b):
            bad.append(f"{r['pair_id']}: lengths {len(a)} != {len(b)}")
            continue
        diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if not diff or diff[0] != r["first_diff"]:
            bad.append(f"{r['pair_id']}: first_diff {r['first_diff']} vs {diff[:1]}")
        if r["ans_A"] == r["ans_B"]:
            bad.append(f"{r['pair_id']}: identical answers")
        if r["ans_A"] not in r["cand_ids"] or r["ans_B"] not in r["cand_ids"]:
            bad.append(f"{r['pair_id']}: answer not among candidates")
        pre = r["clean_prompt"][:_last_line_offset(r["clean_prompt"])]
        if pre != r["corrupt_prompt"][:_last_line_offset(r["corrupt_prompt"])]:
            bad.append(f"{r['pair_id']}: few-shot prefixes differ")
        prefixes.setdefault((r["task"], r["family"]), set()).add(pre)
    for k, v in prefixes.items():
        if len(v) != 1:
            bad.append(f"{k}: {len(v)} different few-shot prefixes")
    return bad


# --------------------------------------------------------------------------- storage
def set_hash(rows: Sequence[dict]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps([r["pair_id"], r["clean_ids"], r["corrupt_ids"], r["ans_A"], r["ans_B"]]).encode())
    return h.hexdigest()[:16]


def save_set(rows: Sequence[dict], path: Path, info: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    meta = {**info, "n": len(rows), "hash": set_hash(rows), "file": path.name}
    path.with_suffix("").with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    return path


def load_set(path: Path, verify: bool = True) -> Tuple[List[dict], dict]:
    path = Path(path)
    with gzip.open(path, "rt") as f:
        rows = [json.loads(l) for l in f]
    meta = json.loads(path.with_suffix("").with_suffix(".meta.json").read_text())
    if verify and set_hash(rows) != meta["hash"]:
        raise ValueError(f"{path}: content hash mismatch (file edited?)")
    return rows, meta


# --------------------------------------------------------------------------- filters
def filter_flags(rows: Sequence[dict], top1_A: np.ndarray, top1_B: np.ndarray, D_A: np.ndarray, D_B: np.ndarray,
                 pilot_mode: str, min_logit_diff: float = 2.0) -> Dict[str, np.ndarray]:
    ansA = np.array([r["ans_A"] for r in rows])
    ansB = np.array([r["ans_B"] for r in rows])
    a_ok, b_ok = top1_A == ansA, top1_B == ansB
    sc = {"clean_top1": a_ok, "corrupt_top1": b_ok, "dl_clean": D_A, "dl_corrupt": D_B}
    return {"f_none": np.ones(len(rows), bool), "f_A_correct": a_ok, "f_both_correct": a_ok & b_ok,
            "f_pilot": st.valid_mask(sc, pilot_mode, min_logit_diff)}


def select(rows: Sequence[dict], flags: Dict[str, np.ndarray], split: str, flt: str, n: int) -> List[int]:
    """Indices of the first n candidates of `split` (in pilot rank order) that pass filter `flt`."""
    idx = [i for i, r in enumerate(rows) if r["split"] == split and flags[flt][i]]
    idx.sort(key=lambda i: rows[i]["rank"])
    return idx[:n]
