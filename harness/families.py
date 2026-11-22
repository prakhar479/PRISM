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
    cf: str = "default"  # counterfactual: default (corrupt moves the queried object instead of another one; A's answer is
                         # fixed by the prefix, PAPER_FINDINGS P2) | dest (pilot P11: both runs move the queried object at the
                         # same step, to different boxes; a patch that only erases B's update yields neither A nor B)


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


@dataclass(frozen=True)
class PCFamily:
    """People-cities binding: 'Asha lives in Pune. Ravi lives in Delhi. ... Asha lives in' (handoff 7.3). The corrupt prompt
    differs in ONE binding: the queried person's city is replaced by a city nobody else has. Answers are single-token cities."""
    id: str
    people: Tuple[str, ...]
    cities: Tuple[str, ...]
    init: str            # format with p, c; one sentence
    query: str           # format with p; no answer
    init_re: str         # groups (person, city)
    query_re: str        # group (person)
    cf: str = "content"  # counterfactual (pilot P1): content (the queried person's city -> a city nobody has) | swap (the
                         # queried person's city is exchanged with another person's: same tokens, different pairing) | query
                         # (another person is asked about)


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

FAMILIES = {"entity_track": OT_FAMILIES, "add_chain": RS_FAMILIES}      # E3's "fresh" = exactly these (unchanged)


def _fam(task: str, family: str):
    """Family object of any task: E3's fresh families, the language families of E13 (harness/langfamilies.py), then the
    breadth pilot's counterfactual variants (harness/pilotfamilies.py)."""
    from harness import langfamilies as LF
    from harness import pilotfamilies as PF
    for table in (FAMILIES.get(task, {}), LF.FAMILIES.get(task, {}), PF.families().get(task, {})):
        if family in table:
            return table[family]
    raise KeyError(f"unknown family {family!r} for task {task}")


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
        elif n in _lang_families(task):                  # E14 / E11: language families by explicit id
            out.append(n)
        else:
            raise ValueError(f"{task}: unknown family '{n}' (known: pilot, fresh, {families_for(task)})")
    return list(dict.fromkeys(out))


def _lang_families(task: str):
    """Families accepted by explicit id: E13's language families and the breadth pilot's variants."""
    from harness import langfamilies as LF
    from harness import pilotfamilies as PF
    return {**LF.FAMILIES.get(task, {}), **PF.families().get(task, {})}


def candidates(task: str, family: str) -> List[str]:
    """Candidate answer strings (with a leading space) of a family."""
    if is_pilot(family):
        from harness.tasks import CANDIDATE_STRINGS
        return CANDIDATE_STRINGS[task]
    if task == "entity_track":
        return [f" {b}" for b in _fam(task, family).labels]
    if task == "people_cities":
        return [f" {c}" for c in _fam(task, family).cities]
    return [f" {d}" for d in range(10)]


# --------------------------------------------------------------------------- generators
def _gen_ot(f: OTFamily, rng: np.random.Generator, depth: int = 2) -> Optional[st.CounterfactualPair]:
    """Same construction as the pilot's _gen_entity: depth+1 objects, depth moves, the corrupt run moves the
    queried object instead of another one at step k (and no later move touches the queried object)."""
    if f.cf == "dest":
        return _gen_ot_dest(f, rng, depth)
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


def _gen_ot_dest(f: OTFamily, rng: np.random.Generator, depth: int = 2) -> Optional[st.CounterfactualPair]:
    """Pilot P11 (no default answer): depth+1 objects, depth moves; at step k BOTH runs move the queried object q, the clean
    run to box X and the corrupt run to box Y (X != Y, both != q's box before step k); no later move touches q. Erasing B's
    update leaves q's earlier box Z (meta.fallback_label), which is neither answer."""
    n_obj = depth + 1
    objs = [f.objects[i] for i in rng.permutation(len(f.objects))[:n_obj]]
    loc0 = dict(zip(objs, (f.labels[i] for i in rng.permutation(len(f.labels))[:n_obj])))
    q = objs[int(rng.integers(0, n_obj))]
    k = int(rng.integers(0, depth))
    others = [o for o in objs if o != q]
    loc, moves = dict(loc0), []
    for s_ in range(depth):
        o = q if s_ == k else (str(rng.choice(objs)) if s_ < k else str(rng.choice(others)))
        if s_ == k:
            z = loc[q]
            opts = [x for x in f.labels if x != z]
            if len(opts) < 2:
                return None
            xa, xb = (str(v) for v in rng.choice(opts, size=2, replace=False))
            moves.append((q, xa))
            loc[q] = xa
            continue
        b = str(rng.choice([x for x in f.labels if x != loc[o]]))
        moves.append((o, b))
        loc[o] = b
    cmoves = list(moves)
    cmoves[k] = (q, xb)
    a, ca = st.solve_entity(loc0, moves, q), st.solve_entity(loc0, cmoves, q)
    if a == ca:
        return None

    def prompt(ms):
        return (" ".join(f.init.format(o=o, b=loc0[o]) for o in objs) + " "
                + " ".join(f.move.format(o=o, b=b) for o, b in ms) + " " + f.query.format(o=q))

    return st.CounterfactualPair("entity_track", "", prompt(moves), prompt(cmoves), f" {a}", f" {ca}",
                                 f.labels.index(a), f.labels.index(ca), [],
                                 {"heur": {"fallback_box": z}, "corrupt_step": k, "family": f.id, "fallback_label": z,
                                  "fallback_index": f.labels.index(z)})


def _gen_pc(f: PCFamily, rng: np.random.Generator, depth: int = 3) -> Optional[st.CounterfactualPair]:
    if f.cf in ("swap", "query"):
        return _gen_pc_variant(f, rng, depth)
    n = depth
    people = [f.people[i] for i in rng.permutation(len(f.people))[:n]]
    cities = [f.cities[i] for i in rng.permutation(len(f.cities))[:n + 1]]
    own, spare = cities[:n], cities[n]
    qi = int(rng.integers(0, n))
    q = people[qi]

    def prompt(cs):
        return " ".join(f.init.format(p=p, c=c) for p, c in zip(people, cs)) + " " + f.query.format(p=q)

    ccs = list(own)
    ccs[qi] = spare
    a, ca = own[qi], spare
    heur = {"first_city": own[0], "last_city": own[-1], "query_position": str(qi)}
    return st.CounterfactualPair("people_cities", "", prompt(own), prompt(ccs), f" {a}", f" {ca}",
                                 f.cities.index(a), f.cities.index(ca), [],
                                 {"heur": heur, "corrupt_step": qi, "family": f.id})


def _gen_pc_variant(f: PCFamily, rng: np.random.Generator, depth: int = 3) -> Optional[st.CounterfactualPair]:
    """Pilot P1. swap: the corrupt prompt exchanges the cities of the queried person and person j (identical tokens, different
    pairing: a pure binding variable). query: the corrupt prompt asks about person j (the pointer to the entity retrieved)."""
    n = depth
    if n < 2:
        return None
    people = [f.people[i] for i in rng.permutation(len(f.people))[:n]]
    own = [f.cities[i] for i in rng.permutation(len(f.cities))[:n]]
    qi = int(rng.integers(0, n))
    j = int(rng.choice([x for x in range(n) if x != qi]))

    def prompt(cs, who):
        return " ".join(f.init.format(p=p, c=c) for p, c in zip(people, cs)) + " " + f.query.format(p=people[who])

    if f.cf == "swap":
        ccs = list(own)
        ccs[qi], ccs[j] = own[j], own[qi]
        clean, corrupt, step = prompt(own, qi), prompt(ccs, qi), min(qi, j)
    else:
        clean, corrupt, step = prompt(own, qi), prompt(own, j), n
    a, ca = own[qi], own[j]
    return st.CounterfactualPair("people_cities", "", clean, corrupt, f" {a}", f" {ca}", f.cities.index(a), f.cities.index(ca), [],
                                 {"heur": {"query_position": str(qi), "other_position": str(j)}, "corrupt_step": step,
                                  "family": f.id, "cf": f.cf, "other_position": j})


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
        f = _fam(task, family)
        return lambda rng: _gen_ot(f, rng, **gen_kwargs)
    if task == "add_chain":
        g = _fam(task, family)
        return lambda rng: _gen_rs(g, rng, **gen_kwargs)
    if task == "people_cities":
        c = _fam(task, family)
        return lambda rng: _gen_pc(c, rng, **gen_kwargs)
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
        f = _fam(task, family)
        for k, m in enumerate(re.finditer(f.init_re, line), 1):
            out += [(f"obj_{k}", m.start(1), m.end(1)), (f"box_{k}", m.start(2), m.end(2))]
        for k, m in enumerate(re.finditer(f.move_re, line), 1):
            out += [(f"move_{k}_obj", m.start(1), m.end(1)), (f"move_{k}_box", m.start(2), m.end(2))]
        m = re.search(f.query_re, line)
        if m:
            out.append(("query_obj", m.start(1), m.end(1)))
    elif task == "people_cities":
        c = _fam(task, family)
        for k, m in enumerate(re.finditer(c.init_re, line), 1):
            out += [(f"person_{k}", m.start(1), m.end(1)), (f"city_{k}", m.start(2), m.end(2))]
        m = re.search(c.query_re, line)
        if m:
            out.append(("query_person", m.start(1), m.end(1)))
    elif task == "add_chain":
        g = _fam(task, family)
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
