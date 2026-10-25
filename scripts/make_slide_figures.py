#!/usr/bin/env python
"""Simple, plain-language figures for the slides and the proposal's core pages.

Reads docs/handoff/data/*.csv (made by scripts/make_handoff_figures.py) and, for the noise-only control, the main
run's exp1 metrics; no numbers are typed here.
No titles inside the images: the slide or caption states the message.

    python scripts/make_slide_figures.py                  # -> slides/figures/*.pdf|png
    python scripts/make_slide_figures.py --font-dir DIR   # register extra fonts (e.g. Fira Sans) first

Style: blue = what the message is about, grey = everything else.
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "handoff" / "data"
EXP1 = ROOT / "results" / "paper_20261026_001341" / "paper_audit" / "metrics" / "exp1_results.csv"
ACCENT, GREY, LIGHT, INK, INK2, GRID = "#2a78d6", "#8a8985", "#c9c7c1", "#1c1b19", "#5f5d58", "#e7e5e0"
TASK = {"add_chain": "Running sum", "entity_track": "Object tracking", "var_binding": "Variable copying",
        "ioi_control": "Names"}
TASKS = ["add_chain", "entity_track", "var_binding", "ioi_control"]


def setup_style(font_dir):
    if font_dir:
        for f in Path(font_dir).glob("*.[ot]tf"):
            font_manager.fontManager.addfont(str(f))
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = "Fira Sans" if "Fira Sans" in names else "DejaVu Sans"
    plt.rcParams.update({"font.family": family, "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.8, "xtick.major.width": 0.8,
                         "ytick.major.width": 0.8, "axes.titlesize": 10.5, "axes.titlecolor": INK,
                         "axes.titleweight": "medium", "savefig.bbox": "tight", "savefig.transparent": True,
                         "pdf.fonttype": 42})


def read(name):
    with open(DATA / f"{name}.csv") as f:
        return list(csv.DictReader(f))


def pct(x):
    return f"{round(100 * float(x))}%"


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=220)
    plt.close(fig)


def hgrid(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_split(out):
    """Readable features vs leftovers, per task, across layers."""
    rows = read("f2_causal_split")
    noise = {}    # object tracking, swap from the first changed token: leftovers replaced by noise of the same size
    if EXP1.exists():
        with open(EXP1) as f:
            for r in csv.DictReader(f):
                if (r["task"], r["width"], r["positions"], r["intervention"]) == \
                        ("entity_track", "16k", "from_diff", "noise_control"):
                    noise[int(r["layer"])] = 100 * float(r["nldr"])
    fig, axes = plt.subplots(1, 4, figsize=(5.6, 2.25), sharey=True)   # drawn at slide size
    for ax, t in zip(axes, TASKS):
        for part, color, label in (("sae_only", ACCENT, "readable features"), ("error_only", GREY, "leftovers")):
            pts = sorted((int(r["layer"]), 100 * float(r["nldr"])) for r in rows if r["task"] == t and r["part"] == part)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=2, marker="o", ms=4.5,
                    markeredgecolor="white", markeredgewidth=0.8, label=label)
        if t == "entity_track" and noise:
            pts = sorted(noise.items())
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=LIGHT, lw=1.6, ls=(0, (2, 2)))
            x16 = dict(pts).get(16)
            if x16 is not None:     # label sits just above the dashed line at layer 16, clear of both solid lines
                ax.text(20.6, x16 + 4, "noise\nonly", color=INK2, fontsize=8.5, ha="right", va="bottom",
                        linespacing=1.0)
        ax.set_title(TASK[t], loc="left")
        ax.set_xticks([4, 12, 20])
        ax.set_ylim(-5, 108)
        ax.set_yticks([0, 50, 100])
        ax.set_yticklabels(["0%", "50%", "100%"])
        ax.set_xlabel("layer", fontsize=10)
        hgrid(ax)
    axes[0].set_ylabel("answer brought back", fontsize=10)
    axes[0].text(12, 80, "readable\nfeatures", color=ACCENT, fontsize=9, va="top", ha="center", fontweight="medium")
    axes[0].text(12, 16, "leftovers", color=GREY, fontsize=9, va="bottom", ha="center", fontweight="medium")
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "s_split")


def fig_gap(out):
    """Object tracking, layer 12: readable features alone vs + one hidden signal."""
    rows = read("f7_repair_curve")
    get = lambda m, r: float(next(x["nldr"] for x in rows if x["method"] == m and x["rank"] == str(r)))
    bars = [("Readable features only", get("unsupervised_svd", 0), GREY),
            ("+ 1 random signal", get("random_subspace", 1), LIGHT),
            ("+ 1 hidden signal from the leftovers", get("unsupervised_svd", 1), ACCENT),
            ("Everything (full swap)", 1.0, None)]
    fig, ax = plt.subplots(figsize=(5.2, 2.0))
    ys = range(len(bars))[::-1]
    for y, (label, v, c) in zip(ys, bars):
        if c is None:     # reference bar: outline only, so blue stays the focus
            ax.barh(y, 100 * v, height=0.62, fill=False, edgecolor=GREY, linewidth=1)
        else:
            ax.barh(y, 100 * v, color=c, height=0.62)
        ax.text(100 * v + 1.5, y, pct(v), va="center", fontsize=10, color=INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([b[0] for b in bars], fontsize=10)
    ax.set_xlim(0, 114)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("answer brought back (object tracking, layer 12)", fontsize=10)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save(fig, out, "s_gap")


def fig_specific(out):
    """Share of the gap recovered by 2 signals from each source. Same task = the unsupervised fit (the method of the
    gap chart); other tasks = the better of the two transfers."""
    rows = read("f6_specificity")
    panels = [("entity_track", "Object tracking", "12"), ("add_chain", "Running sum", "18")]
    groups = [("same task", lambda s: s == "own task, unsupervised"), ("other tasks", lambda s: s.startswith("other task")),
              ("ordinary text", lambda s: s == "ordinary web text"), ("random", lambda s: s == "random")]
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 1.95), sharex=True, sharey=True)
    for ax, (t, name, layer) in zip(axes, panels):
        mine = [r for r in rows if r["task"] == t]
        title = f"{name}, layer {layer} (gap {pct(mine[0]['gap'])})"
        ys = range(len(groups))[::-1]
        for y, (label, match) in zip(ys, groups):
            v = max(float(r["share_closed"]) for r in mine if match(r["source"]))
            ax.barh(y, 100 * max(v, 0), color=ACCENT if label == "same task" else LIGHT, height=0.62)
            ax.text(100 * max(v, 0) + 1.5, y, pct(max(v, 0)), va="center", fontsize=10, color=INK)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([g[0] for g in groups], fontsize=10)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlim(0, 72)
        ax.set_xticks([0, 25, 50])
        ax.set_xticklabels(["0%", "25%", "50%"])
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("share of gap recovered by 2 signals", fontsize=10)
    axes[1].tick_params(axis="y", labelleft=False)
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "s_specific")


def fig_width(out):
    """Object tracking: what the SAE misses as it gets 16x bigger, with and without one hidden signal."""
    rows = [r for r in read("f5_widths") if r["task"] == "entity_track"]
    widths = [r["width"] for r in rows]
    gap = [100 * float(r["gap"]) for r in rows]
    left = [100 * (1 - float(r["plus_1_error_direction"])) for r in rows]
    x = range(len(widths))
    fig, ax = plt.subplots(figsize=(5.2, 2.2))
    ax.plot(x, gap, color=ACCENT, lw=2, marker="o", ms=5, markeredgecolor="white", markeredgewidth=0.8)
    ax.plot(x, left, color=GREY, lw=2, marker="o", ms=5, markeredgecolor="white", markeredgewidth=0.8)
    ax.text(len(widths) - 0.65, gap[-1], "missed by the SAE", color=ACCENT, va="center", fontsize=10,
            fontweight="medium")
    ax.text(len(widths) - 0.65, left[-1], "still missed after\nadding 1 hidden signal", color=GREY, va="center",
            fontsize=10, fontweight="medium")
    for i in (0, len(widths) - 1):
        ax.annotate(f"{round(gap[i])}%", (i, gap[i]), textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=10, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{w}\n({2 ** i}×)" for i, w in enumerate(widths)])   # widths double each step
    ax.set_xlim(-0.3, len(widths) + 1.0)
    ax.set_ylim(0, 32)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_yticklabels(["0%", "10%", "20%", "30%"])
    ax.set_ylabel("answer missed", fontsize=10)
    ax.set_xlabel("SAE size (number of features)", fontsize=10)
    ax.spines["bottom"].set_bounds(0, len(widths) - 1)
    hgrid(ax)
    fig.tight_layout()
    save(fig, out, "s_width")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "slides" / "figures"))
    ap.add_argument("--font-dir", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    setup_style(a.font_dir)
    fig_split(out)
    fig_gap(out)
    fig_specific(out)
    fig_width(out)
    print(f"wrote {sorted(p.name for p in out.glob('*.pdf'))} to {out}")


if __name__ == "__main__":
    main()
