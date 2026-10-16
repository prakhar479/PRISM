"""Synthetic counterfactual tasks, tokenizers and token-alignment utilities.

Tasks: mod_sum, dyck_depth, graph_reach (algorithmic) and ioi_control (natural-language control).
Every pair (clean, corrupt) tokenizes to the same length, so patching at pos=-1 is aligned.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

log = logging.getLogger("prism")

TASKS = ["mod_sum", "dyck_depth", "graph_reach", "ioi_control"]
MOD = 7
N_NODES = 6
NAMES = ["Mary", "John", "Alice", "Bob", "Tom", "Anna", "Mike", "Lucy"]
PLACES = ["park", "store", "school", "beach", "market", "garden"]
VERBS = ["drink", "book", "gift", "letter"]
# set of distinct state values per task (used for classification probes / mock readouts)
TASK_STATES: Dict[str, List[int]] = {
    "mod_sum": list(range(MOD)),
    "dyck_depth": [1, 2, 3, 4, 5, 6],
    "graph_reach": [1, 2, 3, 4, 5, 6],
    "ioi_control": list(range(len(NAMES))),
}


@dataclass
class CounterfactualPair:
    task_name: str
    pair_id: str
    clean_prompt: str
    corrupt_prompt: str
    clean_target_str: str
    corrupt_target_str: str
    clean_state_id: int
    corrupt_state_id: int
    step_states_clean: List[int] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenizedPair:
    pair: CounterfactualPair
    clean_ids: List[int]
    corrupt_ids: List[int]
    clean_ans: int
    corrupt_ans: int

    @property
    def diff_pos(self) -> int:
        """First token index where clean and corrupt differ (positions before it are identical)."""
        for i, (a, b) in enumerate(zip(self.clean_ids, self.corrupt_ids)):
            if a != b:
                return i
        return len(self.clean_ids) - 1


# --------------------------------------------------------------------------- solvers
def solve_mod_sum(start: int, incs: Sequence[int]) -> Tuple[int, List[int]]:
    s, states = start, [start]
    for a in incs:
        s = (s + a) % MOD
        states.append(s)
    return s, states


def solve_dyck(start: int, ops: Sequence[int]) -> Tuple[int, List[int]]:
    d, states = start, [start]
    for o in ops:
        d += o
        states.append(d)
    return d, states


def solve_graph(edges: Sequence[Tuple[int, int]], n_nodes: int = N_NODES) -> Tuple[int, List[int]]:
    """Size of the component of node 1 (union-find), also after each added edge."""
    parent = list(range(n_nodes + 1))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def size1() -> int:
        r = find(1)
        return sum(1 for v in range(1, n_nodes + 1) if find(v) == r)

    states = [size1()]
    for a, b in edges:
        parent[find(a)] = find(b)
        states.append(size1())
    return states[-1], states


# --------------------------------------------------------------------------- prompt builders
def _mod_prompt(start: int, incs: Sequence[int]) -> str:
    return f"State = {start}. " + " ".join(f"Add {a}." for a in incs) + f" Modulo {MOD}, State ="


def _dyck_prompt(start: int, ops: Sequence[int]) -> str:
    return f"Depth = {start}. " + " ".join(f"Op: {'+' if o > 0 else '-'}1." for o in ops) + " Current Depth ="


def _graph_prompt(edges: Sequence[Tuple[int, int]]) -> str:
    return (f"Graph nodes 1 to {N_NODES}. " + " ".join(f"Edge {a}-{b}." for a, b in edges)
            + " Size of component with node 1 is")


def _ioi_prompt(a: str, b: str, place: str, verb: str, subj: str) -> str:
    return f"When {a} and {b} went to the {place}, {subj} gave a {verb} to"


# --------------------------------------------------------------------------- generators
def _gen_mod_sum(rng: np.random.Generator, n_steps: int = 4) -> CounterfactualPair:
    start = int(rng.integers(0, MOD))
    incs = [int(x) for x in rng.integers(1, MOD, size=n_steps)]
    k = int(rng.integers(0, n_steps - 1))  # intermediate (non-final) increment
    new = int(rng.choice([v for v in range(1, MOD) if v != incs[k]]))
    cinc = list(incs)
    cinc[k] = new
    s, states = solve_mod_sum(start, incs)
    cs, _ = solve_mod_sum(start, cinc)
    return CounterfactualPair("mod_sum", "", _mod_prompt(start, incs), _mod_prompt(start, cinc),
                              f" {s}", f" {cs}", s, cs, states, {"corrupt_step": k})


def _walk_dyck(rng: np.random.Generator, n_ops: int) -> Tuple[int, List[int]]:
    start = int(rng.integers(1, 7))
    d, ops = start, []
    for _ in range(n_ops):
        choices = [o for o in (1, -1) if 1 <= d + o <= 6]
        o = int(rng.choice(choices))
        ops.append(o)
        d += o
    return start, ops


def _gen_dyck(rng: np.random.Generator, n_ops: int = 8) -> Optional[CounterfactualPair]:
    start, ops = _walk_dyck(rng, n_ops)
    k = int(rng.integers(0, n_ops // 2))
    cops = list(ops)
    cops[k] = -cops[k]
    _, cstates = solve_dyck(start, cops)
    if not all(1 <= d <= 6 for d in cstates):
        return None
    s, states = solve_dyck(start, ops)
    cs = cstates[-1]
    return CounterfactualPair("dyck_depth", "", _dyck_prompt(start, ops), _dyck_prompt(start, cops),
                              f" {s}", f" {cs}", s, cs, states, {"corrupt_step": k})


def _gen_graph(rng: np.random.Generator, n_edges: int = 4) -> Optional[CounterfactualPair]:
    def rand_edge() -> Tuple[int, int]:
        a, b = rng.choice(np.arange(1, N_NODES + 1), size=2, replace=False)
        return int(a), int(b)

    edges = [rand_edge() for _ in range(n_edges)]
    s, states = solve_graph(edges)
    k = int(rng.integers(0, n_edges))
    for _ in range(20):
        cedges = list(edges)
        cedges[k] = rand_edge()
        cs, _ = solve_graph(cedges)
        if cs != s:
            return CounterfactualPair("graph_reach", "", _graph_prompt(edges), _graph_prompt(cedges),
                                      f" {s}", f" {cs}", s, cs, states, {"corrupt_edge": k})
    return None


def _gen_ioi(rng: np.random.Generator) -> CounterfactualPair:
    ia, ib = rng.choice(len(NAMES), size=2, replace=False)
    a, b = NAMES[int(ia)], NAMES[int(ib)]
    place, verb = str(rng.choice(PLACES)), str(rng.choice(VERBS))
    # clean: subject B -> indirect object A ; corrupt: subject A -> indirect object B
    return CounterfactualPair("ioi_control", "", _ioi_prompt(a, b, place, verb, b),
                              _ioi_prompt(a, b, place, verb, a), f" {a}", f" {b}", int(ia), int(ib), [], {})


_GENERATORS = {"mod_sum": _gen_mod_sum, "dyck_depth": _gen_dyck, "graph_reach": _gen_graph, "ioi_control": _gen_ioi}


def generate_pairs(task: str, n: int, seed: int, exclude: Optional[Set[str]] = None,
                   split: str = "train") -> List[CounterfactualPair]:
    """Deterministic unique pairs. Pairs whose clean/corrupt prompts occur in `exclude` are rejected,
    so splits sharing an `exclude` set are prompt-disjoint (zero train/test leakage)."""
    rng = np.random.default_rng(seed)
    exclude = exclude if exclude is not None else set()
    seen: Set[Tuple[str, str]] = set()
    out: List[CounterfactualPair] = []
    for _ in range(n * 200):
        if len(out) >= n:
            break
        p = _GENERATORS[task](rng)
        if p is None or p.clean_state_id == p.corrupt_state_id:
            continue
        key = (p.clean_prompt, p.corrupt_prompt)
        if key in seen or p.clean_prompt in exclude or p.corrupt_prompt in exclude:
            continue
        seen.add(key)
        p.pair_id = f"{task}-{split}-{len(out):05d}"
        out.append(p)
    if len(out) < n:
        log.warning("%s/%s: only %d/%d unique pairs available", task, split, len(out), n)
    return out


def add_few_shot_prefix(pairs: List[CounterfactualPair], n_shots: int, seed: int) -> List[CounterfactualPair]:
    """Prepend the same n solved examples (one per line) to clean and corrupt prompts."""
    if n_shots <= 0 or not pairs:
        return pairs
    task = pairs[0].task_name
    avoid = {p.clean_prompt for p in pairs} | {p.corrupt_prompt for p in pairs}
    shots = generate_pairs(task, n_shots, seed, exclude=avoid, split="shot")
    prefix = "".join(f"{s.clean_prompt}{s.clean_target_str}\n" for s in shots)
    for p in pairs:
        p.clean_prompt, p.corrupt_prompt = prefix + p.clean_prompt, prefix + p.corrupt_prompt
    return pairs


def parse_state(prompt: str) -> Tuple[str, int]:
    """Recover (task, ground-truth state) from a prompt string (the last line if few-shot)."""
    prompt = prompt.strip("\n").split("\n")[-1].strip()
    if prompt.startswith("State ="):
        start = int(re.match(r"State = (\d+)\.", prompt).group(1))
        incs = [int(x) for x in re.findall(r"Add (\d+)\.", prompt)]
        return "mod_sum", solve_mod_sum(start, incs)[0]
    if prompt.startswith("Depth ="):
        start = int(re.match(r"Depth = (\d+)\.", prompt).group(1))
        ops = [1 if s == "+" else -1 for s in re.findall(r"Op: ([+-])1\.", prompt)]
        return "dyck_depth", solve_dyck(start, ops)[0]
    if prompt.startswith("Graph nodes"):
        edges = [(int(a), int(b)) for a, b in re.findall(r"Edge (\d+)-(\d+)\.", prompt)]
        return "graph_reach", solve_graph(edges)[0]
    m = re.match(r"When (\w+) and (\w+) went to the \w+, (\w+) gave a \w+ to", prompt)
    if m:
        a, b, subj = m.groups()
        io = a if subj == b else b
        return "ioi_control", NAMES.index(io)
    raise ValueError(f"unrecognised prompt: {prompt!r}")


# --------------------------------------------------------------------------- tokenizers
class FallbackTokenizer:
    """Deterministic offline tokenizer: regex word/punct split over a closed, pre-registered vocab.

    Tokens carry their leading space (e.g. " 3", " Mary"), id 0 is BOS. Whitespace-only runs
    (including newlines) are dropped.
    """

    BOS = 0
    _PAT = re.compile(r" ?\w+| ?[^\w\s]")

    def __init__(self) -> None:
        words = ["State", "Add", "Modulo", "Depth", "Op", "Current", "Graph", "nodes", "to", "Edge", "Size", "of",
                 "component", "with", "node", "is", "When", "and", "went", "the", "gave", "a"]
        words += NAMES + PLACES + VERBS + [str(d) for d in range(10)]
        puncts = list(".,-=:+")
        toks: Set[str] = set()
        for w in words:
            toks.add(w)
            toks.add(" " + w)
        for p in puncts:
            toks.add(p)
            toks.add(" " + p)
        self.itos = ["<bos>"] + sorted(toks)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

    def _split(self, text: str) -> List[str]:
        return self._PAT.findall(text)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = [self.BOS] if add_special_tokens else []
        for t in self._split(text):
            if t not in self.stoi:
                raise KeyError(f"token {t!r} outside closed fallback vocab")
            ids.append(self.stoi[t])
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids if int(i) != self.BOS)

    def __call__(self, text: str, **_: Any) -> Dict[str, List[int]]:
        return {"input_ids": self.encode(text)}


def get_tokenizer(mock: bool, hf_name: str = "google/gemma-2-2b"):
    """HF tokenizer in production; deterministic fallback if mock or no HF_TOKEN."""
    import os

    if mock:
        return FallbackTokenizer()
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        log.warning("HF_TOKEN not set; using FallbackTokenizer (offline testing only)")
        return FallbackTokenizer()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(hf_name)


# --------------------------------------------------------------------------- alignment
def assert_token_alignment(clean_ids: Sequence[int], corrupt_ids: Sequence[int]) -> None:
    assert len(clean_ids) == len(corrupt_ids), (
        f"token length mismatch: clean={len(clean_ids)} corrupt={len(corrupt_ids)}")


def answer_token_id(tok, prompt: str, target: str) -> Optional[int]:
    """Id of the single token that `target` adds after `prompt`; None if it is not exactly one token."""
    base = tok.encode(prompt)
    full = tok.encode(prompt + target)
    if len(full) == len(base) + 1 and full[: len(base)] == base:
        return int(full[-1])
    return None


ANSWER_FORMATS = ("auto", "space_target", "space_prompt")


def _with_format(p: CounterfactualPair, fmt: str) -> CounterfactualPair:
    """space_target: prompt '... =' + target ' 3' (default). space_prompt: prompt '... = ' + target '3'
    (needed for tokenizers such as Gemma's that split ' 3' into '▁' + '3')."""
    if fmt == "space_target":
        return p
    from copy import copy

    q = copy(p)
    q.clean_prompt, q.corrupt_prompt = p.clean_prompt + " ", p.corrupt_prompt + " "
    q.clean_target_str, q.corrupt_target_str = p.clean_target_str.lstrip(), p.corrupt_target_str.lstrip()
    q.meta = dict(p.meta, answer_format="space_prompt")
    return q


def tokenize_pairs(tok, pairs: Sequence[CounterfactualPair], answer_format: str = "auto") -> List[TokenizedPair]:
    assert answer_format in ANSWER_FORMATS, answer_format
    out, bad_len, bad_ans = [], 0, 0
    fmts = ["space_target", "space_prompt"] if answer_format == "auto" else [answer_format]
    for p0 in pairs:
        got = None
        for fmt in fmts:
            p = _with_format(p0, fmt)
            c, k = tok.encode(p.clean_prompt), tok.encode(p.corrupt_prompt)
            if len(c) != len(k):
                continue
            ca = answer_token_id(tok, p.clean_prompt, p.clean_target_str)
            ka = answer_token_id(tok, p.corrupt_prompt, p.corrupt_target_str)
            if ca is None or ka is None or ca == ka:
                continue
            got = TokenizedPair(p, list(c), list(k), ca, ka)
            break
        if got is None:
            p = _with_format(p0, fmts[0])
            if len(tok.encode(p.clean_prompt)) != len(tok.encode(p.corrupt_prompt)):
                bad_len += 1
            else:
                bad_ans += 1
            continue
        assert_token_alignment(got.clean_ids, got.corrupt_ids)
        out.append(got)
    if bad_len or bad_ans:
        log.warning("tokenize_pairs dropped %d misaligned and %d non-single-token-answer pairs (of %d)",
                    bad_len, bad_ans, len(pairs))
    return out


# --------------------------------------------------------------------------- filtering / batching
def batch_by_length(items: Sequence[TokenizedPair], batch_size: int) -> List[List[TokenizedPair]]:
    """Group by sequence length so no padding is ever needed (pos=-1 is always the real last token)."""
    groups: Dict[int, List[TokenizedPair]] = {}
    for it in items:
        groups.setdefault(len(it.clean_ids), []).append(it)
    batches = []
    for L in sorted(groups):
        g = groups[L]
        batches += [g[i:i + batch_size] for i in range(0, len(g), batch_size)]
    return batches


def make_tensors(batch: Sequence[TokenizedPair], device) -> Dict[str, torch.Tensor]:
    return {
        "clean": torch.tensor([b.clean_ids for b in batch], dtype=torch.long, device=device),
        "corrupt": torch.tensor([b.corrupt_ids for b in batch], dtype=torch.long, device=device),
        "clean_ans": torch.tensor([b.clean_ans for b in batch], dtype=torch.long, device=device),
        "corrupt_ans": torch.tensor([b.corrupt_ans for b in batch], dtype=torch.long, device=device),
    }


FILTER_MODES = ("strict", "clean_only", "none")


@torch.inference_mode()
def score_pairs(model, items: Sequence[TokenizedPair], batch_size: int = 32, device=None) -> Dict[str, np.ndarray]:
    """Per-pair base-model diagnostics (in the order of `items`)."""
    from src.utils import model_device

    device = device if device is not None else model_device(model)
    pos = {id(it): i for i, it in enumerate(items)}
    n = len(items)
    out = {k: np.zeros(n) for k in ("dl_clean", "dl_corrupt")}
    out.update({k: np.zeros(n, bool) for k in ("clean_top1", "corrupt_top1")})
    for batch in batch_by_length(items, batch_size):
        t = make_tensors(batch, device)
        lc = model.run_with_hooks(t["clean"], fwd_hooks=[])[:, -1].float()
        lk = model.run_with_hooks(t["corrupt"], fwd_hooks=[])[:, -1].float()
        idx = torch.arange(len(batch), device=lc.device)
        ca, ka = t["clean_ans"].to(lc.device), t["corrupt_ans"].to(lc.device)
        rows = [pos[id(b)] for b in batch]
        out["dl_clean"][rows] = (lc[idx, ca] - lc[idx, ka]).cpu().numpy()
        out["dl_corrupt"][rows] = (lk[idx, ca] - lk[idx, ka]).cpu().numpy()
        out["clean_top1"][rows] = (lc.argmax(-1) == ca).cpu().numpy()
        out["corrupt_top1"][rows] = (lk.argmax(-1) == ka).cpu().numpy()
    return out


def valid_mask(sc: Dict[str, np.ndarray], mode: str = "strict", min_logit_diff: float = 2.0) -> np.ndarray:
    """strict: both runs predict their own target and dl_clean > margin.
    clean_only: clean run predicts its target and dl_clean - dl_corrupt > margin (enough for a well-posed NLDR)."""
    assert mode in FILTER_MODES, mode
    if mode == "none":
        return np.ones(len(sc["dl_clean"]), bool)
    if mode == "strict":
        return sc["clean_top1"] & sc["corrupt_top1"] & (sc["dl_clean"] > min_logit_diff)
    return sc["clean_top1"] & ((sc["dl_clean"] - sc["dl_corrupt"]) > min_logit_diff)


def filter_valid_pairs(model, items: Sequence[TokenizedPair], min_logit_diff: float = 2.0,
                       batch_size: int = 32, device=None, mode: str = "strict") -> List[TokenizedPair]:
    """Keep pairs the base model actually solves (see valid_mask)."""
    items = list(items)
    if not items:
        return []
    ok = valid_mask(score_pairs(model, items, batch_size, device), mode, min_logit_diff)
    keep = [it for it, o in zip(items, ok) if o]
    log.info("filter_valid_pairs[%s]: kept %d/%d", mode, len(keep), len(items))
    return keep


def build_task_splits(task: str, tok, n_train: int, n_test: int, seed: int, model=None, mock: bool = False,
                      candidate_multiplier: int = 3, min_logit_diff: float = 2.0, n_shots: int = 0,
                      batch_size: int = 32, answer_format: str = "auto", filter_mode: str = "strict",
                      ) -> Dict[str, List[TokenizedPair]]:
    """Deterministic, prompt-disjoint train/test splits (filtered by the base model in production)."""
    do_filter = (model is not None) and not mock and filter_mode != "none"
    mult = candidate_multiplier if do_filter else 1
    exclude: Set[str] = set()
    out: Dict[str, List[TokenizedPair]] = {}
    for split, n, s in (("train", n_train, seed), ("test", n_test, seed + 10_000)):
        pairs = generate_pairs(task, int(n * mult), s, exclude=exclude, split=split)
        for p in pairs:
            exclude.add(p.clean_prompt)
            exclude.add(p.corrupt_prompt)
        add_few_shot_prefix(pairs, n_shots, seed + 77)
        toks = tokenize_pairs(tok, pairs, answer_format)
        if do_filter:
            toks = filter_valid_pairs(model, toks, min_logit_diff, batch_size, mode=filter_mode)
        out[split] = toks[:n]
    return out
