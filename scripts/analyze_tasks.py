#!/usr/bin/env python
"""Task-capability study: is gemma-2-2b unable to do mod_sum / dyck_depth / graph_reach, and what can it do?

check_forward.py showed the forward pass matches Hugging Face and IOI works (100% with 2 shots), while the
three algorithmic tasks keep 0% of pairs: changing one step of the prompt barely moves the answer. This
script tests that explanation against its alternatives, each with an explicit decision rule:

    C0  control     the pipeline is sound: IOI (2-shot) top-1 >= 0.9 and >= 5/7 sanity prompts correct
    H1  incapable   at the configured difficulty every algorithmic task is non-viable
                    (clean_only keep < 0.1 and accuracy not reliably above the best trivial baseline)
    H2  difficulty  the failure is difficulty-limited: some easier level / more shots IS viable
                    (otherwise the prompt format itself is the problem)
    H3  shortcut    when the model is wrong it follows a surface heuristic (last number, start value, ...)
    H4  latent      the state is linearly decodable from the residual even where the output is wrong, beyond
                    what a probe on the prompt's surface tokens gets (residual probe >= max(shuffled-label,
                    surface-token probe) + 0.15): represented but not read out
    H5  replacement simple state-tracking tasks (variable binding, entity tracking, short addition)
                    are viable for the PRISM experiments

A cell (task x difficulty x shots) is VIABLE when: clean top-1 >= 0.6, clean_only keep >= 0.3 (the share
of pairs the pipeline's filter would keep), and the Wilson 95% lower bound of accuracy beats the best
trivial baseline (most frequent answer, or the best surface heuristic).

    python scripts/analyze_tasks.py                 # full grid (~15-25 min on an 8 GB GPU)
    python scripts/analyze_tasks.py --quick         # n=48, shots 0/4, no probes (~4 min)
    python scripts/analyze_tasks.py --tasks var_binding,entity_track --shots 0,2

Writes <out>/report.md, cells.csv, cells.json, probes.json, recommendations.json and figures/*.png.
"""
import argparse
import contextlib
import functools
import json
import logging
import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import torch

from src.data import synthetic_tasks as st
from src.data.synthetic_tasks import CounterfactualPair
from src.models import runtime
from src.models.sae_loader import hook_name, load_model
from src.utils import free_memory, get_device, get_dtype, load_config, model_device, save_json, setup_logging

log = logging.getLogger("prism")
MARGIN = 2.0                      # logit-gap margin used by the pipeline's filter (min_logit_diff)
VIABLE = {"acc": 0.6, "keep": 0.3}
SANITY = [("The capital of France is", " Paris"),
          ("When Mary and John went to the store, John gave a drink to", " Mary"),
          ("When Tom and Anna went to the park, Anna gave a book to", " Tom"),
          ("Monday, Tuesday, Wednesday,", " Thursday"),
          ("The opposite of hot is", " cold"),
          ("1, 2, 3, 4, ", "5"),
          ("A, B, C, D,", " E")]


# --------------------------------------------------------------------------- candidate replacement tasks
VARS = ["x", "y", "z", "p", "q", "r", "s", "t"]
OBJS = ["coin", "key", "ring", "book", "cup", "pen", "ball", "card"]
BOXES = list("ABCDEFG")


def _gen_var(rng, depth: int = 2):
    """Variable binding with copy semantics; depth = copy hops = distractor assignments.
    depth 1: 't = 7. q = 4. x = q. q = 2. x =' -> 4 (x copied q before q was overwritten)."""
    names = [VARS[i] for i in rng.permutation(len(VARS))[: 2 * depth + 1]]
    chain, distract = names[: depth + 1], names[depth + 1:]
    n, nc, r = (int(v) for v in rng.choice(np.arange(1, 10), size=3, replace=False))
    seq = ["{N}"] + [f"{chain[i]} = {chain[i - 1]}." for i in range(1, depth + 1)]
    for j, d in enumerate(distract):    # independent assignments; the first one opens the program so that
        pos = 0 if j == 0 else int(rng.integers(0, len(seq) + 1))   # "first number" is not a free shortcut
        seq.insert(pos, f"{d} = {int(rng.integers(1, 10))}.")
    seq.append(f"{chain[0]} = {r}.")                     # the source is overwritten after the copies

    def prompt(v):
        return " ".join(f"{chain[0]} = {v}." if s == "{N}" else s for s in seq) + f" {chain[-1]} ="

    nums = re.findall(r"= (\d)\.", prompt(n))
    heur = {"last_number": nums[-1], "first_number": nums[0]}   # last_number = the overwrite of the source
    return CounterfactualPair("var_binding", "", prompt(n), prompt(nc), f" {n}", f" {nc}", n, nc, [n],
                              {"heur": heur, "corrupt_step": seq.index("{N}")})


def _gen_entity(rng, depth: int = 2):
    """Entity tracking: depth+1 objects in boxes, depth moves; the corrupt run moves the queried object instead."""
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

    def final(ms):
        lc = dict(loc0)
        for o, b in ms:
            lc[o] = b
        return lc[q]

    a, ca = final(moves), final(cmoves)
    if a == ca:
        return None

    def prompt(ms):
        return (" ".join(f"The {o} is in box {loc0[o]}." for o in objs) + " "
                + " ".join(f"The {o} moves to box {b}." for o, b in ms) + f" The {q} is in box")

    heur = {"initial_box": loc0[q], "last_box_mentioned": moves[-1][1], "first_box": loc0[objs[0]]}
    return CounterfactualPair("entity_track", "", prompt(moves), prompt(cmoves), f" {a}", f" {ca}",
                              BOXES.index(a), BOXES.index(ca), [], {"heur": heur, "corrupt_step": k})


def _gen_add(rng, depth: int = 2):
    """Short addition without modulo (totals <= 9 so the answer is one digit token)."""
    for _ in range(50):
        start = int(rng.integers(0, 5))
        incs = [int(x) for x in rng.integers(1, 4, size=depth)]
        total = start + sum(incs)
        k = int(rng.integers(0, depth))
        opts = [v for v in range(1, 5) if v != incs[k] and total - incs[k] + v <= 9]
        if total > 9 or not opts:
            continue
        cinc = list(incs)
        cinc[k] = int(rng.choice(opts))
        ctotal = start + sum(cinc)

        def prompt(xs):
            return f"Start with {start}. " + " ".join(f"Add {a}." for a in xs) + " Total ="

        heur = {"start": str(start), "last_add": str(incs[-1]), "start_plus_last": str(start + incs[-1])}
        return CounterfactualPair("add_chain", "", prompt(incs), prompt(cinc), f" {total}", f" {ctotal}",
                                  total, ctotal, [], {"heur": heur, "corrupt_step": k})
    return None


def _last_line(p: str) -> str:
    return p.split("\n")[-1]


def heuristics(task: str, pair: CounterfactualPair) -> dict:
    """Surface-level answers a model could give without tracking the state."""
    if "heur" in pair.meta:
        return pair.meta["heur"]
    s = _last_line(pair.clean_prompt)
    if task == "mod_sum":
        start, incs = int(re.search(r"State = (\d+)", s).group(1)), [int(x) for x in re.findall(r"Add (\d+)\.", s)]
        return {"start": str(start), "last_add": str(incs[-1]), "first_add": str(incs[0]),
                "start_plus_last_mod": str((start + incs[-1]) % st.MOD), "sum_no_mod_last_digit": str(start + sum(incs))[-1]}
    if task == "dyck_depth":
        start, ops = int(re.search(r"Depth = (\d+)", s).group(1)), re.findall(r"Op: ([+-])1", s)
        return {"start": str(start), "start_plus_last_op": str(start + (1 if ops[-1] == "+" else -1)),
                "count_plus": str(ops.count("+")), "count_minus": str(ops.count("-"))}
    if task == "graph_reach":
        edges = [(int(a), int(b)) for a, b in re.findall(r"Edge (\d+)-(\d+)", s)]
        nb = {b if a == 1 else a for a, b in edges if 1 in (a, b)}
        return {"direct_neighbours_plus_1": str(len(nb) + 1), "n_edges": str(len(edges)),
                "n_nodes_mentioned": str(len({x for e in edges for x in e}))}
    return {}


# task -> generator, difficulty kwarg, levels, configured level (the one the overnight run used)
SPECS = {
    "mod_sum": dict(gen=st._gen_mod_sum, kw="n_steps", levels=[2, 3, 4], configured=4, kind="original"),
    "dyck_depth": dict(gen=st._gen_dyck, kw="n_ops", levels=[2, 4, 6, 8], configured=8, kind="original"),
    "graph_reach": dict(gen=st._gen_graph, kw="n_edges", levels=[1, 2, 3, 4], configured=4, kind="original"),
    "ioi_control": dict(gen=st._gen_ioi, kw=None, levels=[0], configured=0, kind="control"),
    "var_binding": dict(gen=_gen_var, kw="depth", levels=[1, 2, 3], configured=None, kind="new"),
    "entity_track": dict(gen=_gen_entity, kw="depth", levels=[1, 2, 3], configured=None, kind="new"),
    "add_chain": dict(gen=_gen_add, kw="depth", levels=[1, 2, 3], configured=None, kind="new"),
}


@contextlib.contextmanager
def task_level(task: str, level: int):
    """Route st.generate_pairs / add_few_shot_prefix for `task` to the generator at this difficulty."""
    spec, old = SPECS[task], st._GENERATORS.get(task)
    st._GENERATORS[task] = functools.partial(spec["gen"], **{spec["kw"]: level}) if spec["kw"] else spec["gen"]
    try:
        yield
    finally:
        if old is None:
            st._GENERATORS.pop(task, None)
        else:
            st._GENERATORS[task] = old


def make_items(tok, task: str, level: int, shots: int, n: int, seed: int):
    """Shots are drawn FIRST and excluded from the pairs, so small prompt spaces cannot starve the prefix."""
    with task_level(task, level):
        ex = st.generate_pairs(task, shots, seed + 77, split="shot") if shots else []
        exclude = {p.clean_prompt for p in ex} | {p.corrupt_prompt for p in ex}
        pairs = st.generate_pairs(task, n, seed, exclude=exclude, split="study")
    prefix = "".join(f"{q.clean_prompt}{q.clean_target_str}\n" for q in ex)
    for p in pairs:
        p.clean_prompt, p.corrupt_prompt = prefix + p.clean_prompt, prefix + p.corrupt_prompt
        p.meta["shots_got"] = len(ex)
    return pairs, st.tokenize_pairs(tok, pairs, "auto")


# --------------------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# --------------------------------------------------------------------------- scoring
@torch.inference_mode()
def score(model, tok, items):
    """Per item: clean/corrupt argmax, logits at both answers, answer-set argmax and probability mass."""
    ans = sorted({i.clean_ans for i in items} | {i.corrupt_ans for i in items})
    ans_t = torch.tensor(ans)
    pos = {id(it): j for j, it in enumerate(items)}
    n = len(items)
    out = {k: np.zeros(n) for k in ("lc_c", "lc_k", "lk_c", "lk_k", "mass")}
    out.update({k: np.zeros(n, dtype=np.int64) for k in ("pred_c", "pred_k", "set_pred_c")})
    dev = model_device(model)
    for batch in st.batch_by_length(items, 128):
        t = st.make_tensors(batch, dev)
        lc, lk = runtime.last_logits(model, t["clean"]).cpu(), runtime.last_logits(model, t["corrupt"]).cpu()
        rows = [pos[id(b)] for b in batch]
        ca, ka = t["clean_ans"].cpu(), t["corrupt_ans"].cpu()
        idx = torch.arange(len(batch))
        out["lc_c"][rows], out["lc_k"][rows] = lc[idx, ca].numpy(), lc[idx, ka].numpy()
        out["lk_c"][rows], out["lk_k"][rows] = lk[idx, ca].numpy(), lk[idx, ka].numpy()
        out["pred_c"][rows], out["pred_k"][rows] = lc.argmax(-1).numpy(), lk.argmax(-1).numpy()
        out["set_pred_c"][rows] = ans_t[lc[:, ans_t].argmax(-1)].numpy()
        out["mass"][rows] = torch.softmax(lc, -1)[:, ans_t].sum(-1).numpy()
    return out


def cell_metrics(task, level, shots, pairs, items, sc, tok) -> dict:
    n = len(items)
    ca = np.array([i.clean_ans for i in items])
    ka = np.array([i.corrupt_ans for i in items])
    correct = sc["pred_c"] == ca
    dl_c, dl_k = sc["lc_c"] - sc["lc_k"], sc["lk_c"] - sc["lk_k"]
    delta = dl_c - dl_k
    keep = correct & (delta > MARGIN)
    strict = correct & (sc["pred_k"] == ka) & (dl_c > MARGIN)
    targets = [i.pair.clean_target_str.strip() for i in items]
    preds = [tok.decode([int(p)]).strip() for p in sc["pred_c"]]
    mode_acc = Counter(targets).most_common(1)[0][1] / n
    heur_acc, heur_agree = {}, {}
    for name in heuristics(task, items[0].pair):
        hs = [heuristics(task, i.pair)[name] for i in items]
        heur_acc[name] = float(np.mean([h == t for h, t in zip(hs, targets)]))
        heur_agree[name] = float(np.mean([h == p for h, p in zip(hs, preds)]))
    wrong = ~correct
    heur_agree_wrong = {k: float(np.mean([heuristics(task, i.pair)[k] == p for i, p, w in zip(items, preds, wrong) if w]))
                        if wrong.any() else 0.0 for k in heur_acc}
    by_step = defaultdict(list)
    for i, d in zip(items, delta):
        by_step[i.pair.meta.get("corrupt_step", -1)].append(float(d))
    k = int(correct.sum())
    lo, hi = wilson(k, n)
    # a heuristic that is (almost) always right on clean prompts IS the solution at this level, not a shortcut;
    # whether the model really uses the corrupted step is what keep_clean_only measures
    shortcuts = {k: v for k, v in heur_acc.items() if v < 0.95}
    baseline = max([mode_acc] + list(shortcuts.values()))
    row = dict(task=task, kind=SPECS[task]["kind"], level=level, shots=shots, shots_got=items[0].pair.meta.get("shots_got", 0), n=n, tokenized=n / max(len(pairs), 1),
               acc=k / n, acc_lo=lo, acc_hi=hi, acc_answer_set=float(np.mean(sc["set_pred_c"] == ca)),
               acc_corrupt=float(np.mean(sc["pred_k"] == ka)), chance=1 / len(set(targets)), mode_acc=mode_acc,
               best_heuristic=max(shortcuts, key=shortcuts.get) if shortcuts else None,
               best_heuristic_acc=max(shortcuts.values()) if shortcuts else 0.0, baseline=baseline,
               solution_equivalent=sorted(set(heur_acc) - set(shortcuts)),
               heuristic_acc=heur_acc, model_agrees_heuristic=heur_agree, wrong_agrees_heuristic=heur_agree_wrong,
               delta_mean=float(delta.mean()), delta_median=float(np.median(delta)), frac_delta_gt_margin=float(np.mean(delta > MARGIN)),
               keep_clean_only=float(keep.mean()), keep_strict=float(strict.mean()), answer_mass=float(sc["mass"].mean()),
               top_predictions=Counter(preds).most_common(4),
               delta_by_corrupt_step={str(s): round(float(np.mean(v)), 3) for s, v in sorted(by_step.items())},
               configured=(level == SPECS[task]["configured"]))
    row["viable"] = bool(row["acc"] >= VIABLE["acc"] and row["keep_clean_only"] >= VIABLE["keep"] and lo > baseline)
    return row


# --------------------------------------------------------------------------- probes (H4)
@torch.inference_mode()
def residuals(model, items, layers):
    """Last-position resid_post at `layers` for the clean prompts -> {layer: [n, d] float32}."""
    store = {l: [] for l in layers}
    pos = {id(it): j for j, it in enumerate(items)}
    order = []
    for batch in st.batch_by_length(items, 64):
        t = st.make_tensors(batch, model_device(model))["clean"]
        hooks = [(hook_name(l), functools.partial(lambda act, hook, l: store[l].append(act[:, -1].float().cpu()), l=l))
                 for l in layers]
        inp, kw = runtime._input(model, t)
        model.run_with_hooks(inp, fwd_hooks=hooks, stop_at_layer=max(layers) + 1, tokens=t, **kw)
        order += [pos[id(b)] for b in batch]
    inv = np.argsort(order)
    return {l: torch.cat(v).numpy()[inv] for l, v in store.items()}


def probe(X, y, seed=0, C=0.1):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    counts = Counter(y)
    keep = np.array([counts[v] >= 5 for v in y])
    X, y = X[keep], y[keep]
    if len(set(y)) < 2:
        return None
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=C))
    acc = float(cross_val_score(clf, X, y, cv=cv).mean())
    ctrl = float(cross_val_score(clf, X, np.random.default_rng(seed).permutation(y), cv=cv).mean())
    return {"acc": acc, "shuffled": ctrl, "majority": max(Counter(y).values()) / len(y), "n": int(len(y))}


def surface_features(tok, items):
    """Token counts + (position, token) one-hots of the query line: what a probe can read without any
    computation by the model (e.g. dyck depth = start + #'+' - #'-' is linear in token counts).
    Returns (counts, counts + position one-hots); the probe baseline is the better of the two."""
    seqs = [tok.encode(_last_line(i.pair.clean_prompt), add_special_tokens=False) for i in items]
    vocab = {t: j for j, t in enumerate(sorted({t for q in seqs for t in q}))}
    slots = {pt: j for j, pt in enumerate(sorted({(k, t) for q in seqs for k, t in enumerate(q)}))}
    X = np.zeros((len(seqs), len(vocab) + len(slots)), np.float32)
    for r, q in enumerate(seqs):
        for k, t in enumerate(q):
            X[r, vocab[t]] += 1
            X[r, len(vocab) + slots[(k, t)]] = 1
    return X[:, :len(vocab)], X


def surface_probe(tok, items, y, seed=0):
    # best over feature sets and regularisation: a deliberately strong baseline (conservative for H4)
    res = [p for X in surface_features(tok, items) for C in (0.1, 1.0, 10.0) for p in [probe(X, y, seed, C)] if p]
    return max(res, key=lambda p: p["acc"]) if res else None


# --------------------------------------------------------------------------- figures
def figures(rows, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in SPECS if any(r["task"] == t and SPECS[t]["kw"] for r in rows)]
    for metric, ylabel, fname in (("acc", "clean top-1 accuracy", "accuracy_vs_difficulty.png"),
                                  ("keep_clean_only", "share of pairs kept (clean_only)", "keep_vs_difficulty.png")):
        fig, axes = plt.subplots(1, len(tasks), figsize=(3.2 * len(tasks), 3.2), sharey=True, squeeze=False)
        for ax, task in zip(axes[0], tasks):
            rs = [r for r in rows if r["task"] == task]
            xs = sorted({r["level"] for r in rs})
            for shots in sorted({r["shots"] for r in rs}):
                pts = sorted((r["level"], r[metric]) for r in rs if r["shots"] == shots)   # skipped cells are absent
                ax.plot([x for x, _ in pts], [v for _, v in pts], marker="o", lw=2, label=f"{shots} shot")
            if metric == "acc":
                ax.plot(xs, [max(r["baseline"] for r in rs if r["level"] == x) for x in xs], "k--", lw=1, label="best baseline")
            else:
                ax.axhline(VIABLE["keep"], color="k", ls="--", lw=1, label="viable")
            if SPECS[task]["configured"] is not None:
                ax.axvline(SPECS[task]["configured"], color="0.6", ls=":", lw=1)
            ax.set_xticks(xs)
            ax.set_title(task, fontsize=10)
            ax.set_xlabel(SPECS[task]["kw"])
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.25)
        axes[0][0].set_ylabel(ylabel)
        axes[0][-1].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=130)
        plt.close(fig)


# --------------------------------------------------------------------------- verdicts & report
def verdicts(rows, sanity_ok, probes_res):
    by = defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    V = []
    ioi = [r for r in by.get("ioi_control", []) if r["shots"] >= 2]
    c0 = bool(ioi) and max(r["acc"] for r in ioi) >= 0.9 and sanity_ok[0] >= 5
    V.append(("C0", "pipeline is sound (positive control)", f"IOI >=2-shot best top-1 "
              f"{max((r['acc'] for r in ioi), default=0):.2f}; sanity {sanity_ok[0]}/{sanity_ok[1]}", "PASS" if c0 else "FAIL"))
    orig = [t for t in by if SPECS[t]["kind"] == "original"]
    detail, all_bad, per_task = [], True, {}
    for t in orig:
        conf = [r for r in by[t] if r["configured"]]
        bad = all(r["keep_clean_only"] < 0.1 and r["acc_lo"] <= r["baseline"] for r in conf)
        all_bad &= bad
        best = max(conf, key=lambda r: r["keep_clean_only"])
        detail.append(f"{t}: keep<={best['keep_clean_only']:.2f}, acc {best['acc']:.2f} vs baseline {best['baseline']:.2f}")
        easier = [r for r in by[t] if r["viable"]]
        per_task[t] = {"non_viable_at_configured": bad, "viable_cells": len(easier),
                       "easiest_viable": None if not easier else min((r["level"], r["shots"]) for r in easier)}
    h1 = "CONFIRMED" if (c0 and all_bad and orig) else ("INCONCLUSIVE (control failed)" if not c0 else "REJECTED")
    V.append(("H1", "original tasks non-viable at the configured difficulty", "; ".join(detail), h1))
    h2 = []
    for t in orig:
        lv = per_task[t]
        accs = {lvl: max(r["acc"] for r in by[t] if r["level"] == lvl) for lvl in sorted({r["level"] for r in by[t]})}
        trend = list(accs.values())
        mono = len(trend) > 1 and trend[0] - trend[-1] >= 0.15
        state = ("difficulty-limited" if lv["viable_cells"] else
                 "difficulty-sensitive but never viable" if mono else "flat: prompt/format-limited")
        h2.append(f"{t}: {state} (best acc by level {', '.join(f'{k}:{v:.2f}' for k, v in accs.items())})")
    V.append(("H2", "failure is difficulty-limited (easier levels become viable)", "; ".join(h2),
              "CONFIRMED" if all(per_task[t]["viable_cells"] for t in orig) else
              "PARTIAL" if any(per_task[t]["viable_cells"] for t in orig) else "REJECTED"))
    h3 = []
    shortcut_tasks = 0
    for t in orig:
        conf = [r for r in by[t] if r["configured"]]
        agg = defaultdict(list)
        for r in conf:
            for k, v in r["wrong_agrees_heuristic"].items():
                agg[k].append(v)
        if agg:
            k, v = max(((k, float(np.mean(v))) for k, v in agg.items()), key=lambda kv: kv[1])
            chance = float(np.mean([r["chance"] for r in conf]))
            shortcut_tasks += v >= 2 * chance
            h3.append(f"{t}: wrong answers match '{k}' {v:.2f} (chance {chance:.2f})")
    V.append(("H3", "wrong answers follow a surface heuristic", "; ".join(h3),
              "CONFIRMED" if orig and shortcut_tasks == len(orig) else "PARTIAL" if shortcut_tasks else "REJECTED"))
    if probes_res:
        h4, latent = [], 0
        for t, layers in probes_res.items():
            if SPECS[t]["kind"] != "original":
                continue
            surf = layers["surface"]["acc"] if layers.get("surface") else 0.0
            best_l, best = max(((l, p) for l, p in layers["layers"].items() if p), key=lambda lp: lp[1]["acc"],
                               default=(None, None))
            if best is None:
                continue
            floor = max(best["shuffled"], surf)
            ok = best["acc"] - floor >= 0.15
            latent += ok
            note = "" if ok else " -> confounded: surface tokens already decode it" if surf >= best["acc"] - 0.05 else ""
            h4.append(f"{t}: best layer {best_l} probe {best['acc']:.2f} vs shuffled {best['shuffled']:.2f}, "
                      f"surface-token probe {surf:.2f}{note}")
        V.append(("H4", "state is represented in the residual beyond what the prompt tokens give", "; ".join(h4),
                  "CONFIRMED" if latent and latent == len(h4) else "PARTIAL" if latent else "REJECTED"))
    new = [t for t in by if SPECS[t]["kind"] == "new"]
    if new:
        ok = {t: [r for r in by[t] if r["viable"]] for t in new}
        V.append(("H5", "replacement tasks are viable", "; ".join(
            f"{t}: {len(v)} viable cells" + (f", hardest {max((r['level'], -r['shots']) for r in v)[0]}" if v else "")
            for t, v in ok.items()), "CONFIRMED" if all(ok.values()) else "PARTIAL" if any(ok.values()) else "REJECTED"))
    return V, per_task


def recommendations(rows):
    """Per task: the hardest viable level, then the fewest shots, then the highest keep rate."""
    rec = {}
    for t in SPECS:
        v = [r for r in rows if r["task"] == t and r["viable"]]
        if not v:
            rec[t] = {"viable": False}
            continue
        b = max(v, key=lambda r: (r["level"], -r["shots"], r["keep_clean_only"]))
        rec[t] = {"viable": True, SPECS[t]["kw"] or "level": b["level"], "n_shots": b["shots"], "filter_mode": "clean_only",
                  "expected_retention": round(b["keep_clean_only"], 3),
                  "candidate_multiplier": int(math.ceil(1.5 / max(b["keep_clean_only"], 0.05))),
                  "acc": round(b["acc"], 3), "baseline": round(b["baseline"], 3)}
    return rec


def write_report(out, rows, V, sanity, probes_res, rec, meta):
    L = [f"# Task capability analysis ({time.strftime('%Y-%m-%d %H:%M')})", "",
         f"model `{meta['model']}` on `{meta['device']}`, n={meta['n']} pairs per cell, shots {meta['shots']}, "
         f"viable = top-1 >= {VIABLE['acc']}, keep >= {VIABLE['keep']}, Wilson lower bound > best baseline.", "",
         "## Verdicts", "", "| id | hypothesis | evidence | verdict |", "|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | **{d}** |" for a, b, c, d in V]
    L += ["", "## Sanity prompts", "", "| prompt | target | top-1 | ok |", "|---|---|---|---|"]
    L += [f"| {s['prompt']!r} | {s['target']!r} | {s['pred']!r} | {'yes' if s['ok'] else 'no'} |" for s in sanity]
    L += ["", "## All cells", "", "Configured = the difficulty the overnight run used. delta = (clean gap) - (corrupt gap), "
          "the quantity the pipeline's filter thresholds at 2.", "",
          "| task | level | shots | n | top-1 [95% CI] | answer-set top-1 | baseline (which) | keep | strict | delta median "
          "| answer mass | top predictions | viable |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        which = r["best_heuristic"] if r["best_heuristic_acc"] >= r["mode_acc"] else "mode"
        shots = str(r["shots"]) if r["shots_got"] == r["shots"] else f"{r['shots']} (got {r['shots_got']})"
        L.append(f"| {r['task']}{' (configured)' if r['configured'] and r['kind'] == 'original' else ''} | {r['level']} | "
                 f"{shots} | {r['n']} | {r['acc']:.2f} [{r['acc_lo']:.2f}, {r['acc_hi']:.2f}] | {r['acc_answer_set']:.2f} | "
                 f"{r['baseline']:.2f} ({which}) | {r['keep_clean_only']:.2f} | {r['keep_strict']:.2f} | {r['delta_median']:.2f} | "
                 f"{r['answer_mass']:.2f} | {', '.join(f'{p!r}:{c}' for p, c in r['top_predictions'])} | "
                 f"{'**yes**' if r['viable'] else 'no'} |")
    L += ["", "## Which shortcut do wrong answers follow? (configured level, mean over shots)", "",
          "Heuristics with accuracy >= 0.95 equal the true solution at that level and are not used as baselines.", "",
          "| task | heuristic | heuristic accuracy | share of wrong answers matching it |", "|---|---|---|---|"]
    for t in SPECS:
        conf = [r for r in rows if r["task"] == t and (r["configured"] or SPECS[t]["kind"] == "new")]
        if not conf or not conf[0]["heuristic_acc"]:
            continue
        for h in conf[0]["heuristic_acc"]:
            L.append(f"| {t} | {h} | {np.mean([r['heuristic_acc'][h] for r in conf]):.2f} | "
                     f"{np.mean([r['wrong_agrees_heuristic'][h] for r in conf]):.2f} |")
    L += ["", "## Does the corrupted step matter? (median delta by corrupted position, configured level, most shots)", ""]
    for t in SPECS:
        conf = [r for r in rows if r["task"] == t and r["configured"]]
        if conf and SPECS[t]["kind"] == "original":
            r = max(conf, key=lambda r: r["shots"])
            L.append(f"- **{t}** ({r['shots']} shot): {r['delta_by_corrupt_step']}")
    if probes_res:
        L += ["", "## Linear probes on the residual (last position, clean state, 5-fold CV)", "",
              "A residual probe only shows computation if it beats the surface-token probe (same classifier on token "
              "counts + position one-hots of the prompt).", "",
              "| task | level | shots | layer | probe acc | shuffled | surface tokens | majority | n |",
              "|---|---|---|---|---|---|---|---|---|"]
        for t, layers in probes_res.items():
            m = meta["probe_cells"][t]
            surf = layers["surface"]["acc"] if layers.get("surface") else float("nan")
            for l, p in layers["layers"].items():
                if p:
                    L.append(f"| {t} | {m[0]} | {m[1]} | {l} | {p['acc']:.2f} | {p['shuffled']:.2f} | {surf:.2f} | "
                             f"{p['majority']:.2f} | {p['n']} |")
    L += ["", "## Recommended settings (hardest viable level, fewest shots)", "", "```json", json.dumps(rec, indent=1), "```",
          "", "Figures: `figures/accuracy_vs_difficulty.png`, `figures/keep_vs_difficulty.png` (dotted line = configured level)."]
    (out / "report.md").write_text("\n".join(L) + "\n")


@torch.inference_mode()
def sanity_prompts(model, tok):
    res = []
    dev = model_device(model)
    for p, tgt in SANITY:
        ids = torch.tensor([tok.encode(p)], device=dev)
        pred = int(runtime.last_logits(model, ids)[0].argmax())
        tid = st.answer_token_id(tok, p, tgt)
        res.append({"prompt": p, "target": tgt, "pred": tok.decode([pred]), "ok": tid is not None and pred == tid})
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="results/task_analysis")
    ap.add_argument("--tasks", default=",".join(SPECS))
    ap.add_argument("--shots", default="0,2,4,8")
    ap.add_argument("--n", type=int, default=128, help="pairs per cell")
    ap.add_argument("--probe-n", type=int, default=400, help="prompts per probed cell")
    ap.add_argument("--probe-layers", default="2,6,10,14,18,22")
    ap.add_argument("--no-probes", action="store_true")
    ap.add_argument("--quick", action="store_true", help="n=48, shots 0,4, no probes")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.quick:
        a.n, a.shots, a.no_probes = 48, "0,4", True
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    setup_logging(out / "analysis.log")
    cfg = load_config(a.config)
    device = get_device()
    tok = st.get_tokenizer(False, cfg.get("hf_tokenizer", "google/gemma-2-2b"))
    model = load_model(device=device, dtype=get_dtype(device), model_name=cfg["model_name"], tokenizer=tok,
                       runtime_cfg=cfg.get("runtime"))
    runtime.setup_runtime(cfg, model, device)
    tasks = [t for t in a.tasks.split(",") if t]
    shots_list = [int(s) for s in a.shots.split(",")]
    t0 = time.time()

    sanity = sanity_prompts(model, tok)
    sanity_ok = (sum(s["ok"] for s in sanity), len(sanity))
    log.info("sanity prompts %d/%d correct", *sanity_ok)

    rows = []
    cells = [(t, lvl, s) for t in tasks for lvl in SPECS[t]["levels"] for s in (shots_list if SPECS[t]["kind"] != "control"
                                                                                 else sorted({0, 2} | set(shots_list)))]
    for j, (task, level, shots) in enumerate(cells):
        pairs, items = make_items(tok, task, level, shots, a.n, a.seed + 1000 * level + shots)
        if len(items) < 8:
            log.warning("%s level %s %d-shot: only %d tokenizable pairs, skipped", task, level, shots, len(items))
            continue
        r = cell_metrics(task, level, shots, pairs, items, score(model, tok, items), tok)
        rows.append(r)
        log.info("[%d/%d] %s L%s %ds: acc %.2f [%.2f,%.2f] baseline %.2f keep %.2f delta~%.2f %s (%.0fs)", j + 1, len(cells),
                 task, level, shots, r["acc"], r["acc_lo"], r["acc_hi"], r["baseline"], r["keep_clean_only"],
                 r["delta_median"], "VIABLE" if r["viable"] else "", time.time() - t0)
    save_json(rows, out / "cells.json")
    try:
        import pandas as pd

        pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rows]).to_csv(out / "cells.csv", index=False)
    except ImportError:
        pass

    probes_res, probe_cells = {}, {}
    if not a.no_probes:
        layers = [l for l in (int(x) for x in a.probe_layers.split(",")) if l < model.cfg.n_layers]
        for task in tasks:
            spec = SPECS[task]
            if spec["kind"] == "control":
                continue
            level = spec["configured"] if spec["configured"] is not None else spec["levels"][len(spec["levels"]) // 2]
            shots = 2 if 2 in shots_list else shots_list[0]
            _, items = make_items(tok, task, level, shots, a.probe_n, a.seed + 555)
            if len(items) < 50:
                continue
            R = residuals(model, items, layers)
            y = np.array([i.pair.clean_state_id for i in items])
            probes_res[task] = {"layers": {l: probe(R[l], y, a.seed) for l in layers},
                                "surface": surface_probe(tok, items, y, a.seed)}
            probe_cells[task] = (level, shots)
            log.info("probe %s L%s %ds: layers %s, surface tokens %s", task, level, shots,
                     {l: (round(p["acc"], 2), round(p["shuffled"], 2)) for l, p in probes_res[task]["layers"].items() if p},
                     probes_res[task]["surface"] and round(probes_res[task]["surface"]["acc"], 2))
            del R
            free_memory()
        save_json({t: {"layers": {str(l): p for l, p in v["layers"].items()}, "surface": v["surface"]}
                   for t, v in probes_res.items()}, out / "probes.json")

    V, per_task = verdicts(rows, sanity_ok, probes_res)
    rec = recommendations(rows)
    save_json({"verdicts": [dict(zip(("id", "hypothesis", "evidence", "verdict"), v)) for v in V], "per_task": per_task,
               "sanity": sanity}, out / "verdicts.json")
    save_json(rec, out / "recommendations.json")
    figures(rows, out / "figures")
    write_report(out, rows, V, sanity, probes_res, rec,
                 {"model": cfg["model_name"], "device": str(device), "n": a.n, "shots": shots_list, "probe_cells": probe_cells})
    for v in V:
        log.info("%s %-55s %s", v[0], v[1], v[3])
    log.info("done in %.1f min -> %s", (time.time() - t0) / 60, out / "report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
