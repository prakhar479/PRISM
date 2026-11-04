"""Fresh template families for E3 (handoff 7.3). Pure Python, no model.

Each family changes the surface of a pilot task while keeping its semantics, difficulty (gen_kwargs), answer type,
pair construction (which step the corrupt prompt changes) and slot names, so that per-slot quantities fitted on one
family (mean errors, directions) can be applied to another:

  entity_track: new object nouns (none from the pilot's OBJS), new frames for placement / moves / query, and new
                box-container words; box labels are single letters (Gemma splits ' 3' into two tokens).
  add_chain:    new phrasings of the running sum ('Begin at .. Plus ..', symbolic '3 + 2 + 1 =', a new story frame
                with names and items not used by the pilot).

Every family draws its own few-shot examples from its own generator, so the prefixes differ from the pilot's.
The pilot generators in src/ are not touched; the generation loop below mirrors src.data.synthetic_tasks.generate_pairs
(same uniqueness and prompt-disjointness rules).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from src.data import synthetic_tasks as st


@dataclass(frozen=True)
class OTFamily:
    id: str
    objects: Tuple[str, ...]
    labels: Tuple[str, ...]
    init: str            # format with o, b; one sentence ending in '.'
    move: str
    query: str           # format with o; no trailing space, no answer
    # regexes over the query line, groups (object, label); query regex group (object)
    init_re: str
    move_re: str
    query_re: str


@dataclass(frozen=True)
class RSFamily:
    id: str
    kind: str            # "calc" | "symbolic" | "story"
    openers: Tuple[str, ...] = ()   # calc: 'Begin at {s}.'; symbolic: a lead-in ('' = none)
    step: str = ""                  # calc: 'Plus {a}.'
    closers: Tuple[str, ...] = ()   # calc: 'Answer:'; symbolic: '=' / 'equals'
    names: Tuple[str, ...] = ()
    items: Tuple[str, ...] = ()
    start_re: str = ""
    add_re: str = ""
    query_re: str = ""


OT_FAMILIES: Dict[str, OTFamily] = {f.id: f for f in [
    OTFamily("ot_placed", ("hat", "shoe", "bell", "lamp", "fork", "doll", "drum", "vase"), tuple("HJKLMNP"),
             "The {o} is placed in box {b}.", "The {o} goes to box {b}.", "The {o} is in box",
             r"The (\w+) is placed in box (\w)\.", r"The (\w+) goes to box (\w)\.", r"The (\w+) is in box\s*$"),
    OTFamily("ot_bin", ("spoon", "sock", "comb", "bowl", "mug", "toy", "brush", "clock"), tuple("QRSTUVW"),
             "The {o} sits in bin {b}.", "The {o} is moved into bin {b}.", "The {o} is now in bin",
             r"The (\w+) sits in bin (\w)\.", r"The (\w+) is moved into bin (\w)\.", r"The (\w+) is now in bin\s*$"),
    OTFamily("ot_drawer", ("shell", "stone", "bead", "badge", "flag", "mask", "plate", "bag"), tuple("ABCDEFG"),
             "There is a {o} in drawer {b}.", "Someone puts the {o} into drawer {b}.", "So the {o} is in drawer",
             r"There is a (\w+) in drawer (\w)\.", r"Someone puts the (\w+) into drawer (\w)\.",
             r"So the (\w+) is in drawer\s*$"),
]}

RS_FAMILIES: Dict[str, RSFamily] = {f.id: f for f in [
    # one fixed phrasing gives only a few hundred distinct prompts (too few for prompt-disjoint 900/450 pools), so,
    # like the pilot's calc template (3 openers x 3 closers), each family varies its opener and closer
    RSFamily("rs_plus", "calc", openers=("Begin at {s}.", "Starting value {s}.", "Initially {s}."), step="Plus {a}.",
             closers=("Answer:", "Final value:", "We get:"),
             start_re=r"^(?:Begin at|Starting value|Initially) (\d)\.", add_re=r"Plus (\d)\.",
             query_re=r"(Answer:|Final value:|We get:)\s*$"),
    RSFamily("rs_symbolic", "symbolic", openers=("", "Compute:", "Evaluate:"), closers=("=", "equals", "gives"),
             start_re=r"(\d) \+", add_re=r"\+ (\d)", query_re=r"(=|equals|gives)\s*$"),
    RSFamily("rs_story", "story", names=("Ravi", "Asha", "Leo", "Nina", "Omar", "Priya", "Arjun", "Maya"),
             items=("marbles", "buttons", "tickets", "beads", "rocks", "cookies", "flowers", "stickers"),
             start_re=r"^\w+ had (\d) \w+\.", add_re=r"Then \w+ found (\d) more\.", query_re=r"(In the end \w+ had)\s*$"),
]}

FAMILIES = {"entity_track": OT_FAMILIES, "add_chain": RS_FAMILIES}


def is_pilot(family: str) -> bool:
    return family == "pilot" or family.startswith("pilot_")


def group_of(family: str) -> str:
    """Selection unit: all pilot families form one group (keeps the pilot's own rank order); fresh ones are their own."""
    return "pilot" if is_pilot(family) else family


def families_for(task: str) -> List[str]:
    return list(FAMILIES.get(task, {}))


def expand(task: str, names: Sequence[str]) -> List[str]:
    """Family-group names in a config ('pilot', 'fresh', or explicit ids) -> group ids."""
    out: List[str] = []
    for n in names:
        if n == "fresh":
            out += families_for(task)
        elif n == "pilot" or n in FAMILIES.get(task, {}):
            out.append(n)
        else:
            raise ValueError(f"{task}: unknown family '{n}' (known: pilot, fresh, {families_for(task)})")
    return list(dict.fromkeys(out))


def candidates(task: str, family: str) -> List[str]:
    """Candidate answer strings (with a leading space) of a family."""
    if is_pilot(family):
        from harness.tasks import CANDIDATE_STRINGS
        return CANDIDATE_STRINGS[task]
    if task == "entity_track":
        return [f" {b}" for b in OT_FAMILIES[family].labels]
    return [f" {d}" for d in range(10)]


# --------------------------------------------------------------------------- generators
def _gen_ot(f: OTFamily, rng: np.random.Generator, depth: int = 2) -> Optional[st.CounterfactualPair]:
    """Same construction as the pilot's _gen_entity: depth+1 objects, depth moves, the corrupt run moves the
    queried object instead of another one at step k (and no later move touches the queried object)."""
    n_obj = depth + 1
    objs = [f.objects[i] for i in rng.permutation(len(f.objects))[:n_obj]]
    loc0 = dict(zip(objs, (f.labels[i] for i in rng.permutation(len(f.labels))[:n_obj])))
    q = objs[int(rng.integers(0, n_obj))]
    loc, moves = dict(loc0), []
    for _ in range(depth):
        o = objs[int(rng.integers(0, n_obj))]
        b = str(rng.choice([x for x in f.labels if x != loc[o]]))
        moves.append((o, b))
        loc[o] = b
    ks = [k for k, (o, _) in enumerate(moves) if o != q and all(o2 != q for o2, _ in moves[k + 1:])]
    if not ks:
        return None
    k = int(rng.choice(ks))
    cmoves = list(moves)
    cmoves[k] = (q, moves[k][1])
    a, ca = st.solve_entity(loc0, moves, q), st.solve_entity(loc0, cmoves, q)
    if a == ca:
        return None

    def prompt(ms):
        return (" ".join(f.init.format(o=o, b=loc0[o]) for o in objs) + " "
                + " ".join(f.move.format(o=o, b=b) for o, b in ms) + " " + f.query.format(o=q))

    heur = {"initial_box": loc0[q], "last_box_mentioned": moves[-1][1], "first_box": loc0[objs[0]]}
    return st.CounterfactualPair("entity_track", "", prompt(moves), prompt(cmoves), f" {a}", f" {ca}",
                                 f.labels.index(a), f.labels.index(ca), [],
                                 {"heur": heur, "corrupt_step": k, "family": f.id})


def _rs_prompt(f: RSFamily, start: int, incs: Sequence[int], name: str = "", item: str = "", opener: str = "",
               closer: str = "") -> str:
    if f.kind == "calc":
        return " ".join([opener.format(s=start)] + [f.step.format(a=a) for a in incs] + [closer])
    if f.kind == "symbolic":
        return " ".join(([opener] if opener else []) + [" + ".join(str(v) for v in [start, *incs]), closer])
    return (f"{name} had {start} {item}. " + " ".join(f"Then {name} found {a} more." for a in incs)
            + f" In the end {name} had")


def _gen_rs(f: RSFamily, rng: np.random.Generator, depth: int = 3) -> Optional[st.CounterfactualPair]:
    """Same construction as the pilot's _gen_add: start 0-6, depth increments 1-4, total <= 9, the corrupt run
    changes increment k."""
    name = str(rng.choice(list(f.names))) if f.kind == "story" else ""
    item = str(rng.choice(list(f.items))) if f.kind == "story" else ""
    opener = str(rng.choice(list(f.openers))) if f.openers else ""
    closer = str(rng.choice(list(f.closers))) if f.closers else ""
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
    return st.CounterfactualPair("add_chain", "", _rs_prompt(f, start, incs, name, item, opener, closer),
                                 _rs_prompt(f, start, cinc, name, item, opener, closer), f" {total}", f" {ctotal}", total, ctotal,
                                 [int(v) for v in states],
                                 {"heur": heur, "corrupt_step": k, "template": f.kind, "family": f.id})


def generator(task: str, family: str, **gen_kwargs) -> Callable[[np.random.Generator], Optional[st.CounterfactualPair]]:
    if task == "entity_track":
        f = OT_FAMILIES[family]
        return lambda rng: _gen_ot(f, rng, **gen_kwargs)
    if task == "add_chain":
        g = RS_FAMILIES[family]
        return lambda rng: _gen_rs(g, rng, **gen_kwargs)
    raise KeyError(f"no fresh families for task {task}")


def generate_pairs(task: str, family: str, n: int, seed: int, exclude: Optional[Set[str]] = None,
                   split: str = "train", gen_kwargs: Optional[dict] = None) -> List[st.CounterfactualPair]:
    """Deterministic unique pairs of one family; mirrors src.data.synthetic_tasks.generate_pairs."""
    rng = np.random.default_rng(seed)
    gen = generator(task, family, **(gen_kwargs or {}))
    exclude = exclude if exclude is not None else set()
    seen: Set[Tuple[str, str]] = set()
    out: List[st.CounterfactualPair] = []
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
        p.pair_id = f"{task}-{family}-{split}-{len(out):05d}"
        out.append(p)
        misses = 0
    return out


def family_seed(base_seed: int, family: str) -> int:
    """Stable per-family seed offset (so families do not share random streams)."""
    import zlib
    return int(base_seed) + 1_000 * (zlib.crc32(family.encode()) % 997 + 1)


# --------------------------------------------------------------------------- slots
def slot_spans(task: str, family: str, line: str) -> List[Tuple[str, int, int]]:
    """(slot, start, end) within the query line, with the pilot's slot names."""
    out: List[Tuple[str, int, int]] = []
    if task == "entity_track":
        f = OT_FAMILIES[family]
        for k, m in enumerate(re.finditer(f.init_re, line), 1):
            out += [(f"obj_{k}", m.start(1), m.end(1)), (f"box_{k}", m.start(2), m.end(2))]
        for k, m in enumerate(re.finditer(f.move_re, line), 1):
            out += [(f"move_{k}_obj", m.start(1), m.end(1)), (f"move_{k}_box", m.start(2), m.end(2))]
        m = re.search(f.query_re, line)
        if m:
            out.append(("query_obj", m.start(1), m.end(1)))
    elif task == "add_chain":
        g = RS_FAMILIES[family]
        m = re.search(g.start_re, line)
        if m:
            out.append(("start_num", m.start(1), m.end(1)))
        adds = list(re.finditer(g.add_re, line))
        for k, m in enumerate(adds, 1):
            out.append((f"add_{k}", m.start(1), m.end(1)))
        m = re.search(g.query_re, line)
        if m:
            out.append(("query", m.start(1), m.end(1)))
    return out
