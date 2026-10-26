#!/usr/bin/env python
"""Print-size figures for the proposal (A4, 17 cm text width), in the same style and terms as the slides.

Reads the main pilot run's metrics (results/paper_20261026_001341) and docs/handoff/data; no numbers are typed here.

    python scripts/make_proposal_figures.py [--font-dir DIR]     # -> proposal/figures/p_*.pdf

Terms: features x̂, SAE error ε, effect recovered (%), generic text (the Pile). Blue = the point of the figure.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "paper_20261026_001341"
DATA = ROOT / "docs" / "handoff" / "data"
ACCENT, ORANGE, GREY, LIGHT, INK, INK2, GRID = "#2a78d6", "#c9761e", "#8a8985", "#c9c7c1", "#1c1b19", "#5f5d58", "#e7e5e0"
TASK = {"add_chain": "Running sum", "entity_track": "Object tracking", "var_binding": "Variable copying",
        "ioi_control": "Names (IOI)"}
TASKS = ["add_chain", "entity_track", "var_binding", "ioi_control"]
W = 6.6   # inches: the proposal's text width


def setup_style(font_dir):
    if font_dir:
        for f in Path(font_dir).glob("*.[ot]tf"):
            font_manager.fontManager.addfont(str(f))
    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({"font.family": "Fira Sans" if "Fira Sans" in names else "DejaVu Sans", "font.size": 8.5,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
                         "axes.titlesize": 9, "axes.titlecolor": INK, "axes.titleweight": "medium",
                         "savefig.bbox": "tight", "savefig.transparent": True, "pdf.fonttype": 42})


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def pct_axis(ax, ticks=(0, 25, 50, 75, 100), axis="y"):
    labels = [f"{t}%" for t in ticks]
    if axis == "y":
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    else:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=200)
    plt.close(fig)


def fig_split(out):
    """Patch only the features or only the SAE error, per task and layer (95% CI bands), with the noise control."""
    e1 = [r for r in rows(RUN / "paper_audit" / "metrics" / "exp1_results.csv")
          if r["positions"] == "from_diff" and r["width"] == "16k"]
    get = lambda t, kind: sorted((int(r["layer"]), 100 * float(r["nldr"]), 100 * float(r["nldr_lo"] or r["nldr"]),
                                  100 * float(r["nldr_hi"] or r["nldr"])) for r in e1
                                 if r["task"] == t and r["intervention"] == kind)
    fig, axes = plt.subplots(1, 4, figsize=(W, 2.0), sharey=True)
    for ax, t in zip(axes, TASKS):
        for kind, color, lw, ls in (("sae_only", ACCENT, 1.8, "-"), ("error_only", GREY, 1.8, "-"),
                                    ("noise_control", GREY, 1.0, (0, (1, 1.4)))):
            pts = get(t, kind)
            xs = [p[0] for p in pts]
            if kind != "noise_control":
                ax.fill_between(xs, [p[2] for p in pts], [p[3] for p in pts], color=color, alpha=0.18, lw=0)
            ax.plot(xs, [p[1] for p in pts], color=color, lw=lw, ls=ls, marker="o" if ls == "-" else None, ms=3)
        ax.set_title(TASK[t], loc="left", color=ACCENT if t == "entity_track" else INK)
        ax.set_xticks([4, 8, 12, 16, 20])
        ax.set_ylim(-5, 108)
        pct_axis(ax)
        ax.set_xlabel("layer")
    axes[0].set_ylabel("effect recovered")
    handles = [Line2D([], [], color=ACCENT, lw=1.8, marker="o", ms=3), Line2D([], [], color=GREY, lw=1.8, marker="o", ms=3),
               Line2D([], [], color=GREY, lw=1.0, ls=(0, (1, 1.4)))]
    fig.legend(handles, ["patch features $\\hat{x}$ only", "patch SAE error $\\varepsilon$ only",
                         "noise in place of $\\varepsilon$ (control)"], loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=0.4, rect=(0, 0, 1, 0.92))
    save(fig, out, "p_split")


def curve(e3, task, layer, method):
    acc = defaultdict(list)
    for r in e3:
        if (r["task"], r["layer"], r["method"], r["kind"], r["width"]) == (task, layer, method, "hybrid_patch", "16k"):
            acc[int(r["rank"])].append(100 * float(r["nldr"]))
    return sorted((k, sum(v) / len(v)) for k, v in acc.items())


def fig_repair(out):
    """Features plus N error directions: two concentrated gaps and one diffuse gap."""
    e3 = rows(RUN / "paper_residue" / "metrics" / "exp3_results.csv")
    panels = [("entity_track", "12", "concentrated"), ("add_chain", "18", "concentrated"), ("ioi_control", "18", "diffuse")]
    methods = [("unsupervised_svd", ACCENT, "-", "same task (top principal directions)"),
               ("supervised_centroid_svd", ORANGE, "-", "same task (answer-separating directions)"),
               ("generic_text_svd", GREY, "-", "generic text (the Pile)"),
               ("random_subspace", LIGHT, (0, (2, 1.5)), "random")]
    fig, axes = plt.subplots(1, 3, figsize=(W, 1.85), sharey=True)
    ranks = [0, 1, 2, 4, 8, 16, 32]
    for ax, (t, layer, kind) in zip(axes, panels):
        for m, color, ls, _ in methods:
            pts = curve(e3, t, layer, m)
            ax.plot([ranks.index(r) for r, _ in pts], [v for _, v in pts], color=color, ls=ls, lw=1.6, marker="o", ms=2.8)
        ax.set_title(f"{TASK[t]}, layer {layer}", loc="left")
        ax.text(0.98, 0.04, f"{kind} gap", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
                color=ACCENT if kind == "concentrated" else INK2, fontweight="bold")
        ax.axhline(100, color=INK2, lw=0.7, ls=(0, (3, 2)))
        ax.set_xticks(range(len(ranks)))
        ax.set_xticklabels([str(r) for r in ranks])
        ax.set_xlabel("error directions added")
        ax.set_ylim(68, 103)
        pct_axis(ax, ticks=(70, 80, 90, 100))
    axes[0].set_ylabel("effect recovered")
    handles = [Line2D([], [], color=c, ls=ls, lw=1.6, marker="o", ms=2.8) for _, c, ls, _ in methods]
    fig.legend(handles, [lab for *_, lab in methods], loc="upper center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.02), handlelength=1.8, columnspacing=1.0)
    fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.93))
    save(fig, out, "p_repair")


def fig_width(out):
    """Gap against SAE width at layer 12: object tracking with and without one error direction; all tasks."""
    f5 = rows(DATA / "f5_widths.csv")
    widths = ["16k", "32k", "65k", "131k", "262k"]
    x = list(range(len(widths)))
    end = len(widths) - 1
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.1))
    ot = [r for r in f5 if r["task"] == "entity_track"]
    gap = [100 * float(r["gap"]) for r in ot]
    left = [100 * (1 - float(r["plus_1_error_direction"])) for r in ot]
    a.fill_between(x, left, gap, color=ACCENT, alpha=0.12, lw=0)
    a.plot(x, gap, color=ACCENT, lw=1.8, marker="o", ms=3.5)
    a.plot(x, left, color=GREY, lw=1.8, marker="o", ms=3.5)
    a.text(end + 0.15, gap[-1], "features only", color=ACCENT, va="center", fontsize=8)
    a.text(end + 0.15, left[-1], "+ 1 error\ndirection", color=INK2, va="center", fontsize=8)
    a.set_title("Object tracking: gap", loc="left")
    for task, color, dy in (("entity_track", ACCENT, 0), ("add_chain", GREY, 1.2), ("ioi_control", LIGHT, -1.2)):
        pts = [(widths.index(r["width"]), 100 * float(r["gap"])) for r in f5 if r["task"] == task]
        b.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=1.8, marker="o", ms=3.5)
        b.text(end + 0.15, pts[-1][1] + dy, TASK[task], color=INK2 if color != ACCENT else ACCENT, va="center",
               fontsize=8)
    b.set_title("Gap by task", loc="left")
    for ax in (a, b):
        ax.set_xticks(x)
        ax.set_xticklabels([f"{w}\n({2 ** i}×)" for i, w in enumerate(widths)])
        ax.set_xlabel("SAE width (layer 12)")
        ax.set_xlim(-0.3, end + 1.6)
        ax.spines["bottom"].set_bounds(0, end)
        ax.set_ylim(0, 32)
        pct_axis(ax, ticks=(0, 10, 20, 30))
    a.set_ylabel("gap (effect not recovered)")
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "p_width")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "proposal" / "figures"))
    ap.add_argument("--font-dir", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    setup_style(a.font_dir)
    fig_split(out)
    fig_repair(out)
    fig_width(out)
    print(f"wrote {sorted(p.name for p in out.glob('*.pdf'))} to {out}")


if __name__ == "__main__":
    main()
