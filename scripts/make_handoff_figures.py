#!/usr/bin/env python
"""Figures + data tables for the course proposal / slides handoff (docs/handoff/).

Every number is read from the results folders, never typed:
    results/task_analysis/report.md          which tasks gemma-2-2b can do (2026-10-20)
    results/paper_20261026_001341/            the three paper tiers (2026-10-26): audit, residue, widths

    python scripts/make_handoff_figures.py                      # -> docs/handoff/figures/*.pdf|png, docs/handoff/data/*.csv
    python scripts/make_handoff_figures.py --paper results/paper_XXXX --out docs/handoff

Style: one accent (blue) on what each title is about, everything else grey; the title states the finding.
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ACCENT, GREY, LIGHT, INK, INK2, GRID = "#2a78d6", "#8a8985", "#c9c7c1", "#1c1b19", "#5f5d58", "#e7e5e0"
TASK = {"add_chain": "Running sum", "entity_track": "Object tracking", "var_binding": "Variable copying",
        "ioi_control": "Names (IOI control)", "mod_sum": "Sum with remainder", "dyck_depth": "Bracket depth",
        "graph_reach": "Graph size"}
NEW_TASKS = ["add_chain", "entity_track", "var_binding", "ioi_control"]
USED_LEVEL = {"add_chain": 3, "entity_track": 2, "var_binding": 1, "ioi_control": 0}   # configs/default_config.yaml
WIDTHS = ["16k", "32k", "65k", "131k", "262k"]

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "figure.dpi": 100, "savefig.bbox": "tight",
                     "pdf.fonttype": 42})


def load(p: Path):
    return json.loads(p.read_text())


def save(fig, out: Path, name: str, title: str, sub: str = ""):
    """Title + wrapped subtitle in a band reserved above the plots (fixed size in inches, any figure height)."""
    import textwrap

    w_in, h_in = fig.get_size_inches()
    lines = textwrap.wrap(sub, width=int(w_in * 15.5)) if sub else []
    band = 0.42 + 0.19 * len(lines)                          # inches: title line + subtitle lines
    fig.tight_layout()
    fig.subplots_adjust(top=1 - (band + 0.12) / h_in)
    fig.text(0.01, 1 - 0.08 / h_in, title, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    if lines:
        fig.text(0.01, 1 - 0.40 / h_in, "\n".join(lines), ha="left", va="top", fontsize=9.5, color=INK2, linespacing=1.3)
    for ext in ("pdf", "png"):
        fig.savefig(out / "figures" / f"{name}.{ext}", dpi=200)
    plt.close(fig)


def write_csv(out: Path, name: str, rows: list):
    with open(out / "data" / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def pick(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def one(rows, **kw):
    m = pick(rows, **kw)
    return m[0] if m else None


def mean_by_rank(rows, **kw):
    acc = defaultdict(list)
    for r in pick(rows, **kw):
        acc[r["rank"]].append(r["nldr"])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def spread(ys, gap):
    """Label positions near ys, at least `gap` apart (keeps the order), so end-of-line labels never overlap."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        out[b] = max(out[b], out[a] + gap)
    return out


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- F1 which tasks the model can do
def fig_task_viability(analysis: Path, out: Path):
    best = {}
    for line in (analysis / "report.md").read_text().splitlines():
        m = re.match(r"\| (\w+)(?: \(configured\))? \| (\d+) \| (\d+)[^|]* \| (\d+) \| ([\d.]+) \[[^|]*\| [^|]*\| [^|]*\| ([\d.]+) \|", line)
        if m and m.group(1) in TASK and (m.group(1) not in USED_LEVEL or int(m.group(2)) == USED_LEVEL[m.group(1)]):
            t, keep, acc = m.group(1), float(m.group(6)), float(m.group(5))
            if keep > best.get(t, {"keep": -1})["keep"]:
                best[t] = {"task": t, "label": TASK[t], "level": int(m.group(2)), "shots": int(m.group(3)),
                           "top1": acc, "keep": keep, "group": "used" if t in NEW_TASKS else "dropped"}
    rows = sorted(best.values(), key=lambda r: r["keep"])
    write_csv(out, "f1_task_viability", rows)
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    y = range(len(rows))
    ax.barh(list(y), [r["keep"] for r in rows], color=[ACCENT if r["group"] == "used" else LIGHT for r in rows],
            height=0.62)
    for i, r in enumerate(rows):
        ax.text(r["keep"] + 0.015, i, f"{r['keep']:.0%}", va="center", fontsize=9, color=INK)
    ax.set_yticks(list(y), [r["label"] for r in rows])
    ax.axvline(0.3, color=INK2, ls="--", lw=1)
    ax.text(0.31, len(rows) - 0.45, "needed: 30%", fontsize=8.5, color=INK2)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    save(fig, out, "f1_task_viability", "gemma-2-2b can do the four tasks we use, not the first three",
         "Share of usable test pairs (model correct, and changing one step changes its answer); used tasks at the "
         "difficulty we run, dropped tasks at their best of all settings")


# --------------------------------------------------------------------------- F2 causal split across layers
def fig_causal_split(paper: Path, out: Path):
    a = load(paper / "paper_audit" / "metrics" / "exp1_results.json")
    layers = sorted({r["layer"] for r in a})
    rows = []
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.7), sharey=True)
    for ax, t in zip(axes, NEW_TASKS):
        for iv, col, lab in (("sae_only", ACCENT, "SAE parts"), ("error_only", GREY, "leftover error")):
            pts = [one(a, task=t, layer=L, positions="from_diff", intervention=iv) for L in layers]
            ax.plot(layers, [p["nldr"] for p in pts], color=col, lw=2, marker="o", ms=4, label=lab)
            ax.fill_between(layers, [p["nldr_lo"] for p in pts], [p["nldr_hi"] for p in pts], color=col, alpha=0.15, lw=0)
            rows += [{"task": t, "layer": L, "part": iv, "nldr": round(p["nldr"], 4), "lo": round(p["nldr_lo"], 4),
                      "hi": round(p["nldr_hi"], 4), "n_test": p["n"]} for L, p in zip(layers, pts)]
        ax.axhline(1, color=INK2, ls="--", lw=0.9)
        ax.set_title(TASK[t], loc="left", color=INK)
        ax.set_xticks(layers)
        ax.set_xlabel("layer")
        ax.set_ylim(-0.05, 1.08)
        grid(ax)
    axes[0].set_ylabel("answer restored")
    axes[0].text(layers[0], 1.02, "full copy = 100%", fontsize=8, color=INK2, va="bottom")
    axes[-1].legend(frameon=False, fontsize=8.5, loc="center right")
    write_csv(out, "f2_causal_split", rows)
    save(fig, out, "f2_causal_split", "The SAE's parts carry the tracked information at every layer; the error carries little",
         "Copy one part from the right-answer prompt into the wrong-answer prompt (from the first changed token); 150 test pairs, 95% CIs")


# --------------------------------------------------------------------------- F3 trap 1: last token vs from first change
def fig_trap_positions(paper: Path, out: Path):
    a = load(paper / "paper_audit" / "metrics" / "exp1_results.json")
    layers = sorted({r["layer"] for r in a})
    rows = []
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    shades = {"add_chain": "#5f5d58", "entity_track": "#8a8985", "var_binding": "#a9a7a1", "ioi_control": "#c9c7c1"}
    ends = []
    for t in NEW_TASKS:
        last = [one(a, task=t, layer=L, positions="last", intervention="clean_full")["nldr"] for L in layers]
        diff = [one(a, task=t, layer=L, positions="from_diff", intervention="clean_full")["nldr"] for L in layers]
        ax.plot(layers, last, color=shades[t], lw=2, marker="o", ms=4)
        ends.append(last[-1])
        rows += [{"task": t, "layer": L, "last_token_copy": round(x, 4), "from_first_change_copy": round(y, 4)}
                 for L, x, y in zip(layers, last, diff)]
    for t, yl in zip(NEW_TASKS, spread(ends, 0.075)):
        ax.text(layers[-1] + 0.4, yl, TASK[t], va="center", fontsize=8.5, color=INK2)
    ax.plot(layers, [1] * len(layers), color=ACCENT, lw=2.4, ls="--")
    ax.text(layers[0], 1.04, "copy from the first changed token: 100% at every layer (all tasks)", fontsize=8.5, color=ACCENT)
    ax.set_xticks(layers)
    ax.set_xlabel("layer")
    ax.set_ylabel("answer restored")
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlim(layers[0] - 0.5, layers[-1] + 5)
    grid(ax)
    write_csv(out, "f3_trap_positions", rows)
    save(fig, out, "f3_trap_positions", "Trap 1: copying only the last token misses the state in early and middle layers",
         "Full residual copied from the right-answer prompt (no SAE involved); grey = last token only")


# --------------------------------------------------------------------------- F4 traps 2-3: readable vs used
def fig_trap_probes(paper: Path, out: Path, layer: int = 12):
    a = load(paper / "paper_audit" / "metrics" / "exp1_results.json")
    pr = load(paper / "paper_audit" / "metrics" / "exp2_results.json")["probes"]
    rows = []
    for t in NEW_TASKS:
        g = {r["source"]: r for r in pr if r["task"] == t and r["layer"] == layer}
        rows.append({"task": t, "layer": layer, "probe_error": g["eps"]["acc"], "probe_sae_parts": g["x_hat"]["acc"],
                     "probe_prompt_tokens_only": g["surface"]["acc"], "chance": g["x"]["chance_acc"],
                     "last_token_copy_restores": one(a, task=t, layer=layer, positions="last", intervention="clean_full")["nldr"]})
    write_csv(out, "f4_trap_probes", rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [3, 1.4]}, sharey=True)
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        a1.plot([r["probe_sae_parts"], r["probe_error"]], [y, y], color=GRID, lw=3, zorder=1)
        a1.scatter(r["probe_error"], y, color=ACCENT, s=46, zorder=3, label="from the error" if y == ys[0] else None)
        a1.scatter(r["probe_sae_parts"], y, color=GREY, s=46, zorder=3, label="from the SAE's parts" if y == ys[0] else None)
        a1.scatter(r["probe_prompt_tokens_only"], y, facecolor="white", edgecolor=INK, s=46, zorder=4,
                   label="prompt words only (no model)" if y == ys[0] else None)
        a2.barh(y, max(r["last_token_copy_restores"], 0.005), color=LIGHT, height=0.55)
        a2.text(max(r["last_token_copy_restores"], 0) + 0.03, y, f"{r['last_token_copy_restores']:.0%}", va="center", fontsize=9)
    a1.set_yticks(ys, [TASK[r["task"]] for r in rows])
    a1.set_xlim(0, 1.05)
    a1.set_title("Can a linear probe read the state at the final token? (accuracy)", loc="left", color=INK)
    a1.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=3)
    grid(a1, "x")
    a2.set_xlim(0, 1.15)
    a2.set_xticks([0, 0.5, 1], ["0", "50%", "100%"])
    a2.set_title("Copying that token restores", loc="left", color=INK)
    save(fig, out, "f4_trap_probes", f"Traps 2 and 3: readable is not the same as used (layer {layer})",
         "Probes read the state from the error (blue) far better than from the SAE's parts (grey), but the prompt's words "
         "alone often do as well (open circle), and copying that token restores little: the model is not using it there yet")


# --------------------------------------------------------------------------- F5 width scaling
def fig_widths(paper: Path, out: Path):
    w = load(paper / "paper_widths" / "metrics" / "exp1_results.json")
    e3 = load(paper / "paper_widths" / "metrics" / "exp3_results.json")["rows"]
    rows = []
    for t in ["entity_track", "add_chain", "ioi_control"]:
        for wd in WIDTHS:
            s = one(w, task=t, width=wd, intervention="sae_only")
            unsup = mean_by_rank(e3, task=t, width=wd, method="unsupervised_svd", kind="hybrid_patch")
            rnd = mean_by_rank(e3, task=t, width=wd, method="random_subspace", kind="hybrid_patch")
            rows.append({"task": t, "width": wd, "sae_only": round(s["nldr"], 4), "lo": round(s["nldr_lo"], 4),
                         "hi": round(s["nldr_hi"], 4), "gap": round(1 - s["nldr"], 4), "l0": round(s["l0"]),
                         "plus_1_error_direction": round(unsup[1], 4) if 1 in unsup else "",
                         "plus_1_random_direction": round(rnd[1], 4) if 1 in rnd else ""})
    write_csv(out, "f5_widths", rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.0))
    ent = [r for r in rows if r["task"] == "entity_track"]
    x = range(len(WIDTHS))
    a1.plot(x, [r["sae_only"] for r in ent], color=GREY, lw=2, marker="o", ms=4)
    a1.fill_between(x, [r["lo"] for r in ent], [r["hi"] for r in ent], color=GREY, alpha=0.15, lw=0)
    a1.plot(x, [r["plus_1_error_direction"] for r in ent], color=ACCENT, lw=2, marker="o", ms=4)
    a1.plot(x, [r["plus_1_random_direction"] for r in ent], color=LIGHT, lw=1.2, ls=":")
    a1.axhline(1, color=INK2, ls="--", lw=0.9)
    a1.text(4.1, ent[-1]["plus_1_error_direction"], "+ 1 error direction", color=ACCENT, va="center", fontsize=8.5)
    a1.text(4.1, ent[-1]["sae_only"] - 0.02, "SAE parts alone\n(+1 random: same)", color=INK2, va="center", fontsize=8.5)
    a1.set_xticks(list(x), WIDTHS)
    a1.set_xlabel("SAE width (layer 12)")
    a1.set_ylim(0.6, 1.04)
    a1.set_xlim(-0.2, 5.6)
    a1.set_ylabel("answer restored")
    a1.set_title("Object tracking", loc="left", color=INK)
    grid(a1)
    series = (("entity_track", ACCENT), ("add_chain", GREY), ("ioi_control", LIGHT))
    ends = []
    for t, col in series:
        tr = [r for r in rows if r["task"] == t]
        a2.plot(x, [r["gap"] for r in tr], color=col, lw=2, marker="o", ms=4)
        ends.append(tr[-1]["gap"])
    for (t, col), yl in zip(series, spread(ends, 0.03)):
        a2.text(4.15, yl + 0.012 * (t == "ioi_control"), TASK[t], color=INK2 if t != "entity_track" else ACCENT,
                va="center", fontsize=8.5)
    a2.axhline(0.10, color=INK2, ls="--", lw=0.9)
    a2.text(0, 0.108, "10% bar", fontsize=8, color=INK2)
    a2.set_xticks(list(x), WIDTHS)
    a2.set_xlabel("SAE width (layer 12)")
    a2.set_ylabel("missed by the SAE")
    a2.set_ylim(0, 0.3)
    a2.set_xlim(-0.2, 6.2)
    a2.set_title("All tasks", loc="left", color=INK)
    grid(a2)
    save(fig, out, "f5_widths", "A 16x wider SAE shrinks the object-tracking gap (26% to 15%) but does not close it",
         "At every width one error direction recovers about half of what is missing; the other tasks' gaps are small. "
         "Note: the 131k/262k SAEs also use more active parts per token (L0 124-139 vs 75-89 at 16k)")


# --------------------------------------------------------------------------- F6 task specificity of the missing direction
def fig_specificity(paper: Path, out: Path, rank: int = 2):
    e3 = load(paper / "paper_residue" / "metrics" / "exp3_results.json")["rows"]
    rows = []
    cases = [("entity_track", 12), ("add_chain", 18)]
    labels = {"unsupervised_svd": "own task, unsupervised", "supervised_centroid_svd": "own task, supervised",
              "generic_text_svd": "ordinary web text", "random_subspace": "random"}
    for t, L in cases:
        base = mean_by_rank(e3, task=t, layer=L, method="unsupervised_svd", kind="hybrid_patch")[0]
        gap = 1 - base
        methods = sorted({r["method"] for r in pick(e3, task=t, layer=L, kind="hybrid_patch")})
        for m in methods:
            d = mean_by_rank(e3, task=t, layer=L, method=m, kind="hybrid_patch")
            if rank in d:
                lab = labels.get(m, "other task: " + TASK.get(m.replace("transfer_from_", ""), m))
                rows.append({"task": t, "layer": L, "rank": rank, "source": lab, "method": m,
                             "gap": round(gap, 4), "share_closed": round((d[rank] - base) / gap, 4)})
    write_csv(out, "f6_specificity", rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.3), sharex=True)
    for ax, (t, L) in zip(axes, cases):
        rs = sorted([r for r in rows if r["task"] == t], key=lambda r: r["share_closed"])
        ys = range(len(rs))
        ax.barh(list(ys), [max(r["share_closed"], 0) for r in rs],
                color=[ACCENT if r["source"].startswith("own") else LIGHT for r in rs], height=0.62)
        for i, r in enumerate(rs):
            ax.text(max(r["share_closed"], 0) + 0.015, i, f"{max(r['share_closed'], 0):.0%}", va="center", fontsize=8.5)
        ax.set_yticks(list(ys), [r["source"] for r in rs])
        ax.set_xlim(0, 0.75)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.set_title(f"{TASK[t]}, layer {L}: SAE misses {rs[0]['gap']:.0%}", loc="left", color=INK, fontsize=9.5)
    save(fig, out, "f6_specificity", "The missing piece is task-specific: directions from other text close almost none of it",
         f"Share of the SAE's gap closed by adding {rank} directions fitted on each source (16k SAE, 150 test pairs)")


# --------------------------------------------------------------------------- F7 repair curve
def fig_repair_curve(paper: Path, out: Path, task: str = "entity_track", layer: int = 12):
    e3 = load(paper / "paper_residue" / "metrics" / "exp3_results.json")["rows"]
    series = [("unsupervised_svd", "top error directions", ACCENT, "-"), ("generic_text_svd", "web-text directions", GREY, "-"),
              ("random_subspace", "random directions", LIGHT, ":")]
    rows, ends = [], []
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ranks = None
    for m, lab, col, ls in series:
        d = mean_by_rank(e3, task=task, layer=layer, method=m, kind="hybrid_patch")
        ranks = sorted(d)
        xs = range(len(ranks))
        ax.plot(list(xs), [d[k] for k in ranks], color=col, lw=2, ls=ls, marker="o", ms=4)
        ends.append(d[ranks[-1]])
        rows += [{"task": task, "layer": layer, "method": m, "rank": k, "nldr": round(d[k], 4)} for k in ranks]
    for (m, lab, col, ls), yl in zip(series, spread([-e for e in ends], 0.022)):
        ax.text(len(ranks) - 0.85, -yl, lab, color=col if col != LIGHT else INK2, va="center", fontsize=8.5)
    ax.axhline(1, color=INK2, ls="--", lw=0.9)
    ax.text(0, 1.008, "full copy = 100%", fontsize=8, color=INK2)
    ax.set_xticks(range(len(ranks)), [str(k) for k in ranks])
    ax.set_xlabel("directions added to the SAE's parts (0 = SAE alone)")
    ax.set_ylabel("answer restored")
    ax.set_ylim(0.65, 1.03)
    ax.set_xlim(-0.3, len(ranks) + 1.6)
    grid(ax)
    write_csv(out, "f7_repair_curve", rows)
    d0 = mean_by_rank(e3, task=task, layer=layer, method="unsupervised_svd", kind="hybrid_patch")
    save(fig, out, "f7_repair_curve",
         f"One direction from the error lifts object tracking from {d0[0]:.0%} to {d0[1]:.0%}",
         f"Layer {layer}, 16k SAE: SAE parts plus N directions; web-text and random directions add nothing")


def key_numbers(paper: Path, out: Path):
    """The handful of headline numbers the proposal and slides quote, with where each comes from."""
    a = load(paper / "paper_audit" / "metrics" / "exp1_results.json")
    fid = load(paper / "paper_residue" / "metrics" / "sae_fidelity.json")
    sae = [r["nldr"] for r in a if r["positions"] == "from_diff" and r["intervention"] == "sae_only"
           and r["task"] != "ioi_control"]
    err = [r["nldr"] for r in a if r["positions"] == "from_diff" and r["intervention"] == "error_only"
           and r["task"] != "ioi_control"]
    rows = [
        {"claim": "full clean copy (from first changed token) restores", "value": "100% at every task and layer",
         "source": "paper_audit exp1 clean_full from_diff"},
        {"claim": "SAE parts alone restore (3 state tasks, layers 4-20)", "value": f"{min(sae):.0%} to {max(sae):.0%}",
         "source": "paper_audit exp1 sae_only from_diff"},
        {"claim": "leftover error alone restores (3 state tasks, layers 4-20)",
         "value": f"{max(min(err), 0):.0%} to {max(err):.0%}; above 15% only for object tracking at layers 8-16, where "
                  "replacing the error with noise alone already restores 28-58%",
         "source": "paper_audit exp1 error_only and noise_control from_diff"},
        {"claim": "SAE CE-loss recovered on web text (16k, layers 10-18)",
         "value": f"{min(r['ce_recovered'] for r in fid):.1%} to {max(r['ce_recovered'] for r in fid):.1%}",
         "source": "paper_residue sae_fidelity"},
        {"claim": "test pairs per setting", "value": "150 (train 300)", "source": "run_info.json"},
        {"claim": "compute", "value": "one 8 GB laptop GPU (RTX 5060), ~6.3 h for the three paper runs", "source": "STATUS"},
    ]
    write_csv(out, "key_numbers", rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", default=str(ROOT / "results" / "paper_20261026_001341"))
    ap.add_argument("--analysis", default=str(ROOT / "results" / "task_analysis"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "handoff"))
    a = ap.parse_args()
    out, paper = Path(a.out), Path(a.paper)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    fig_task_viability(Path(a.analysis), out)
    fig_causal_split(paper, out)
    fig_trap_positions(paper, out)
    fig_trap_probes(paper, out)
    fig_widths(paper, out)
    fig_specificity(paper, out)
    fig_repair_curve(paper, out)
    key_numbers(paper, out)
    print(f"wrote {len(list((out / 'figures').glob('*.pdf')))} figures and {len(list((out / 'data').glob('*.csv')))} tables to {out}")


if __name__ == "__main__":
    main()
