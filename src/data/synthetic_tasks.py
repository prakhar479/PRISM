"""Synthetic counterfactual tasks, tokenizers and token-alignment utilities.

State-tracking tasks (gemma-2-2b solves them; see scripts/analyze_tasks.py): add_chain (running sum),
entity_track (object locations after moves), var_binding (value copied through variables), plus ioi_control
(natural-language control). The legacy tasks mod_sum, dyck_depth and graph_reach are kept for the mock world
and regression tests: gemma-2-2b is at chance on all three at every difficulty, so the configs no longer use them.
Every pair (clean, corrupt) tokenizes to the same length, so patching at pos=-1 is aligned.
"""
from __future__ import annotations

import functools
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

log = logging.getLogger("prism")

TASKS = ["add_chain", "entity_track", "var_binding", "ioi_control", "mod_sum", "dyck_depth", "graph_reach"]
STATE_TASKS = ["add_chain", "entity_track", "var_binding"]
LEGACY_TASKS = ["mod_sum", "dyck_depth", "graph_reach"]
MOD = 7
N_NODES = 6
NAMES = ["Mary", "John", "Alice", "Bob", "Tom", "Anna", "Mike", "Lucy",
         "David", "Sarah", "James", "Emma", "Paul", "Kate", "Mark", "Laura"]
PLACES = ["park", "store", "school", "beach", "market", "garden", "office", "library", "station", "river"]
VERBS = ["drink", "book", "gift", "letter", "pen", "ball", "note", "cake"]
VARS = ["x", "y", "z", "p", "q", "r", "s", "t"]
OBJS = ["coin", "key", "ring", "book", "cup", "pen", "ball", "card"]
BOXES = list("ABCDEFG")
ADD_OPEN = ["Start with", "Begin with", "Start at"]
ADD_CLOSE = ["Total =", "Sum =", "Result ="]
ADD_ITEMS = ["apples", "coins", "books", "cards", "pens", "stones", "shells", "stamps"]
# set of distinct state values per task (used for classification probes / mock readouts)
TASK_STATES: Dict[str, List[int]] = {
    "add_chain": list(range(10)),
    "entity_track": list(range(len(BOXES))),
    "var_binding": list(range(1, 10)),
    "ioi_control": list(range(len(NAMES))),
    "mod_sum": list(range(MOD)),
    "dyck_depth": [1, 2, 3, 4, 5, 6],
    "graph_reach": [1, 2, 3, 4, 5, 6],
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


def solve_entity(loc0: Dict[str, str], moves: Sequence[Tuple[str, str]], q: str) -> str:
    loc = dict(loc0)
    for o, b in moves:
        loc[o] = b
    return loc[q]


def solve_var(program: str) -> int:
    """Value of the queried variable for '<v> = <digit|var>. ... <query> =' (copy semantics)."""
    *body, query = [x.strip() for x in program.split(".") if x.strip()]
    env: Dict[str, int] = {}
    for stmt in body:
        k, v = (t.strip() for t in stmt.split("="))
        env[k] = int(v) if v.isdigit() else env[v]
    return env[query.rstrip("=").strip()]


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


# ---- state-tracking tasks (difficulty kwargs default to the settings validated by analyze_tasks.py)
def _add_prompt(tpl: Tuple, start: int, incs: Sequence[int]) -> str:
    if tpl[0] == "calc":
        _, opener, closer = tpl
        return f"{opener} {start}. " + " ".join(f"Add {a}." for a in incs) + f" {closer}"
    _, name, item = tpl
    return f"{name} has {start} {item}. " + " ".join(f"{name} gets {a} more." for a in incs) + f" {name} now has"


def _gen_add(rng: np.random.Generator, depth: int = 3, templates: Sequence[str] = ("calc", "story")
             ) -> Optional[CounterfactualPair]:
    """Running sum without modulo (totals <= 9, so the answer is one digit token). The corrupt run changes
    one increment. 'calc' = 'Start with 2. Add 3. Add 1. Total =', 'story' = 'Tom has 2 apples. Tom gets 3 more.
    ... Tom now has'."""
    kind = str(rng.choice(list(templates)))
    tpl = (("calc", str(rng.choice(ADD_OPEN)), str(rng.choice(ADD_CLOSE))) if kind == "calc"
           else ("story", str(rng.choice(NAMES)), str(rng.choice(ADD_ITEMS))))
    start = int(rng.integers(0, 7))
    incs = [int(x) for x in rng.integers(1, 5, size=depth)]
    total = start + sum(incs)
    k = int(rng.integers(0, depth))
    opts = [v for v in range(1, 5) if v != incs[k] and total - incs[k] + v <= 9]
    if total > 9 or not opts:
        return None
    cinc = list(incs)
    cinc[k] = int(rng.choice(opts))
    ctotal = start + sum(cinc)
    states = list(np.cumsum([start] + incs))
    heur = {"start": str(start), "last_add": str(incs[-1]), "start_plus_last": str(start + incs[-1])}
    return CounterfactualPair("add_chain", "", _add_prompt(tpl, start, incs), _add_prompt(tpl, start, cinc),
                              f" {total}", f" {ctotal}", total, ctotal, [int(v) for v in states],
                              {"heur": heur, "corrupt_step": k, "template": kind})


def _gen_entity(rng: np.random.Generator, depth: int = 2) -> Optional[CounterfactualPair]:
    """depth+1 objects in boxes, depth moves; the corrupt run moves the queried object instead of another one."""
    n_obj = depth + 1
    objs = [OBJS[i] for i in rng.permutation(len(OBJS))[:n_obj]]
    loc0 = dict(zip(objs, (BOXES[i] for i in rng.permutation(len(BOXES))[:n_obj])))
    q = objs[int(rng.integers(0, n_obj))]
    loc, moves = dict(loc0), []
    for _ in range(depth):
        o = objs[int(rng.integers(0, n_obj))]
        b = str(rng.choice([x for x in BOXES if x != loc[o]]))
        moves.append((o, b))
        loc[o] = b
    ks = [k for k, (o, _) in enumerate(moves) if o != q and all(o2 != q for o2, _ in moves[k + 1:])]
    if not ks:
        return None
    k = int(rng.choice(ks))
    cmoves = list(moves)
    cmoves[k] = (q, moves[k][1])
    a, ca = solve_entity(loc0, moves, q), solve_entity(loc0, cmoves, q)
    if a == ca:
        return None

    def prompt(ms):
        return (" ".join(f"The {o} is in box {loc0[o]}." for o in objs) + " "
                + " ".join(f"The {o} moves to box {b}." for o, b in ms) + f" The {q} is in box")

    heur = {"initial_box": loc0[q], "last_box_mentioned": moves[-1][1], "first_box": loc0[objs[0]]}
    return CounterfactualPair("entity_track", "", prompt(moves), prompt(cmoves), f" {a}", f" {ca}",
                              BOXES.index(a), BOXES.index(ca), [], {"heur": heur, "corrupt_step": k})


def _gen_var(rng: np.random.Generator, depth: int = 1) -> Optional[CounterfactualPair]:
    """Variable binding with copy semantics: 't = 7. q = 4. x = q. q = 2. x =' -> 4. depth = copy hops =
    distractor assignments; a distractor always opens the program so 'first number' is not a shortcut."""
    names = [VARS[i] for i in rng.permutation(len(VARS))[: 2 * depth + 1]]
    chain, distract = names[: depth + 1], names[depth + 1:]
    n, nc, r = (int(v) for v in rng.choice(np.arange(1, 10), size=3, replace=False))
    seq = ["{N}"] + [f"{chain[i]} = {chain[i - 1]}." for i in range(1, depth + 1)]
    for j, d in enumerate(distract):
        pos = 0 if j == 0 else int(rng.integers(0, len(seq) + 1))
        seq.insert(pos, f"{d} = {int(rng.integers(1, 10))}.")
    seq.append(f"{chain[0]} = {r}.")                     # the source is overwritten after the copies

    def prompt(v):
        return " ".join(f"{chain[0]} = {v}." if x == "{N}" else x for x in seq) + f" {chain[-1]} ="

    nums = re.findall(r"= (\d)\.", prompt(n))
    heur = {"last_number": nums[-1], "first_number": nums[0]}
    return CounterfactualPair("var_binding", "", prompt(n), prompt(nc), f" {n}", f" {nc}", n, nc, [n],
                              {"heur": heur, "corrupt_step": seq.index("{N}")})


_GENERATORS = {"add_chain": _gen_add, "entity_track": _gen_entity, "var_binding": _gen_var, "ioi_control": _gen_ioi,
               "mod_sum": _gen_mod_sum, "dyck_depth": _gen_dyck, "graph_reach": _gen_graph}


def generate_pairs(task: str, n: int, seed: int, exclude: Optional[Set[str]] = None,
                   split: str = "train", gen_kwargs: Optional[Dict[str, Any]] = None) -> List[CounterfactualPair]:
    """Deterministic unique pairs. Pairs whose clean/corrupt prompts occur in `exclude` are rejected,
    so splits sharing an `exclude` set are prompt-disjoint (zero train/test leakage). gen_kwargs sets the
    generator's difficulty (e.g. {"depth": 3}). Stops early once the prompt space is clearly exhausted."""
    rng = np.random.default_rng(seed)
    gen = functools.partial(_GENERATORS[task], **(gen_kwargs or {}))
    exclude = exclude if exclude is not None else set()
    seen: Set[Tuple[str, str]] = set()
    out: List[CounterfactualPair] = []
    misses = 0
    for _ in range(n * 200):
        if len(out) >= n or misses > 5000 + 2 * n:
            break
        p = gen(rng)
        misses += 1
        if p is None or p.clean_state_id == p.corrupt_state_id:
            continue
        key = (p.clean_prompt, p.corrupt_prompt)
        if key in seen or p.clean_prompt in exclude or p.corrupt_prompt in exclude:
            continue
        seen.add(key)
        p.pair_id = f"{task}-{split}-{len(out):05d}"
        out.append(p)
        misses = 0
    if len(out) < n:
        log.warning("%s/%s: only %d/%d unique pairs available", task, split, len(out), n)
    return out


def add_few_shot_prefix(pairs: List[CounterfactualPair], n_shots: int, seed: int,
                        gen_kwargs: Optional[Dict[str, Any]] = None) -> List[CounterfactualPair]:
    """Prepend the same n solved examples (one per line) to clean and corrupt prompts. The shots must avoid
    `pairs`, so in a small prompt space prefer draw_shots() BEFORE generating the pairs."""
    if n_shots <= 0 or not pairs:
        return pairs
    avoid = {p.clean_prompt for p in pairs} | {p.corrupt_prompt for p in pairs}
    return apply_prefix(pairs, draw_shots(pairs[0].task_name, n_shots, seed, avoid, gen_kwargs))


def draw_shots(task: str, n_shots: int, seed: int, exclude: Optional[Set[str]] = None,
               gen_kwargs: Optional[Dict[str, Any]] = None) -> List[CounterfactualPair]:
    if n_shots <= 0:
        return []
    shots = generate_pairs(task, n_shots, seed, exclude=exclude, split="shot", gen_kwargs=gen_kwargs)
    if len(shots) < n_shots:
        log.warning("%s: only %d/%d few-shot examples available", task, len(shots), n_shots)
    return shots


def shot_prompts(shots: Sequence[CounterfactualPair]) -> Set[str]:
    return {q.clean_prompt for q in shots} | {q.corrupt_prompt for q in shots}


def apply_prefix(pairs: List[CounterfactualPair], shots: Sequence[CounterfactualPair]) -> List[CounterfactualPair]:
    prefix = "".join(f"{q.clean_prompt}{q.clean_target_str}\n" for q in shots)
    for p in pairs:
        p.clean_prompt, p.corrupt_prompt = prefix + p.clean_prompt, prefix + p.corrupt_prompt
        p.meta["n_shots"] = len(shots)
    return pairs


def generate_with_shots(task: str, n: int, n_shots: int, seed: int, shot_seed: int, split: str = "probe",
                        gen_kwargs: Optional[Dict[str, Any]] = None) -> List[CounterfactualPair]:
    """n pairs with an n_shots prefix; the shots are drawn first and excluded from the pairs."""
    shots = draw_shots(task, n_shots, shot_seed, gen_kwargs=gen_kwargs)
    pairs = generate_pairs(task, n, seed, exclude=shot_prompts(shots), split=split, gen_kwargs=gen_kwargs)
    return apply_prefix(pairs, shots)


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
    m = re.match(r"(?:Start with|Begin with|Start at) (\d)\.", prompt)
    if m:
        return "add_chain", int(m.group(1)) + sum(int(x) for x in re.findall(r"Add (\d)\.", prompt))
    m = re.match(r"(\w+) has (\d) \w+\.", prompt)
    if m and prompt.endswith("now has"):
        return "add_chain", int(m.group(2)) + sum(int(x) for x in re.findall(r"gets (\d) more\.", prompt))
    if prompt.startswith("The ") and prompt.endswith("is in box"):
        loc0 = dict(re.findall(r"The (\w+) is in box (\w)\.", prompt))
        moves = re.findall(r"The (\w+) moves to box (\w)\.", prompt)
        q = re.search(r"The (\w+) is in box$", prompt).group(1)
        return "entity_track", BOXES.index(solve_entity(loc0, moves, q))
    if re.match(r"[a-z] = ", prompt):
        return "var_binding", solve_var(prompt)
    m = re.match(r"When (\w+) and (\w+) went to the \w+, (\w+) gave a \w+ to", prompt)
    if m:
        a, b, subj = m.groups()
        io = a if subj == b else b
        return "ioi_control", NAMES.index(io)
    raise ValueError(f"unrecognised prompt: {prompt!r}")


# --------------------------------------------------------------------------- tokenizers
class FallbackTokenizer:
    """Deterministic offline tokenizer: regex word/punct split over a closed, pre-registered vocab.

    Tokens carry their leading space (e.g. " 3", " Mary"), id 0 is BOS. Newlines are tokens (so few-shot
    prompts decode back to separate lines); other whitespace-only runs are dropped.
    """

    BOS = 0
    _PAT = re.compile(r"\n| ?\w+| ?[^\w\s]")

    def __init__(self) -> None:
        words = ["State", "Add", "Modulo", "Depth", "Op", "Current", "Graph", "nodes", "to", "Edge", "Size", "of",
                 "component", "with", "node", "is", "When", "and", "went", "the", "gave", "a",
                 "Start", "Begin", "at", "Total", "Sum", "Result", "has", "gets", "more", "now",
                 "The", "in", "box", "moves"]
        words += NAMES + PLACES + VERBS + VARS + OBJS + BOXES + ADD_ITEMS + [str(d) for d in range(10)]
        puncts = list(".,-=:+")
        toks: Set[str] = set()
        for w in words:
            toks.add(w)
            toks.add(" " + w)
        for p in puncts:
            toks.add(p)
            toks.add(" " + p)
        toks.add("\n")
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
MAX_BATCH_TOKENS = 4096   # B*L cap per forward; set from free VRAM by runtime.setup_runtime


def batch_by_length(items: Sequence[TokenizedPair], batch_size: int,
                    max_tokens: Optional[int] = None) -> List[List[TokenizedPair]]:
    """Group by sequence length so no padding is ever needed (pos=-1 is always the real last token).
    Batches also respect a token budget so long few-shot prompts cannot run out of activation memory."""
    max_tokens = max_tokens or MAX_BATCH_TOKENS
    groups: Dict[int, List[TokenizedPair]] = {}
    for it in items:
        groups.setdefault(len(it.clean_ids), []).append(it)
    batches = []
    for L in sorted(groups):
        g = groups[L]
        bs = max(1, min(batch_size, max_tokens // max(L, 1)))
        batches += [g[i:i + bs] for i in range(0, len(g), bs)]
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
    from src.models.runtime import last_logits
    from src.utils import model_device

    device = device if device is not None else model_device(model)
    pos = {id(it): i for i, it in enumerate(items)}
    n = len(items)
    out = {k: np.zeros(n) for k in ("dl_clean", "dl_corrupt")}
    out.update({k: np.zeros(n, bool) for k in ("clean_top1", "corrupt_top1")})
    for batch in batch_by_length(items, batch_size):
        t = make_tensors(batch, device)
        lc, lk = last_logits(model, t["clean"]), last_logits(model, t["corrupt"])
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
                      gen_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, List[TokenizedPair]]:
    """Deterministic, prompt-disjoint train/test splits (filtered by the base model in production).

    The shared few-shot prefix is drawn FIRST and its prompts are excluded from both splits, so it can never be
    starved by a small prompt space (the 2026-10-19 run silently ran 2-shot settings as 0-shot)."""
    do_filter = (model is not None) and not mock and filter_mode != "none"
    mult = candidate_multiplier if do_filter else 1
    shots = draw_shots(task, n_shots, seed + 77, gen_kwargs=gen_kwargs)
    exclude: Set[str] = shot_prompts(shots)
    raw: Dict[str, List[CounterfactualPair]] = {}
    for split, n, s in (("train", n_train, seed), ("test", n_test, seed + 10_000)):
        raw[split] = generate_pairs(task, int(n * mult), s, exclude=exclude, split=split, gen_kwargs=gen_kwargs)
        if len(raw[split]) < n:
            log.warning("%s/%s: the prompt space holds only %d pairs for %d requested; widen the generator or lower "
                        "n_%s", task, split, len(raw[split]), n, split)
        exclude |= shot_prompts(raw[split])
    out: Dict[str, List[TokenizedPair]] = {}
    for split, n in (("train", n_train), ("test", n_test)):
        pairs = apply_prefix(raw[split], shots)
        toks = tokenize_pairs(tok, pairs, answer_format)
        if do_filter:
            toks = filter_valid_pairs(model, toks, min_logit_diff, batch_size, mode=filter_mode)
        out[split] = toks[:n]
    return out
