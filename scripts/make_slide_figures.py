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


W, H = 5.5, 2.3   # inches: the usable area of a 16:9 beamer slide below the title, so charts are included at 100%


def fig_split(out):
    """SAE features vs SAE error, per task, across layers; the object-tracking dip is annotated."""
    rows = read("f2_causal_split")
    noise = {}    # object tracking, patch from the first changed token: SAE error replaced by noise of the same size
    if EXP1.exists():
        with open(EXP1) as f:
            for r in csv.DictReader(f):
                if (r["task"], r["width"], r["positions"], r["intervention"]) == \
                        ("entity_track", "16k", "from_diff", "noise_control"):
                    noise[int(r["layer"])] = 100 * float(r["nldr"])
    fig, axes = plt.subplots(1, 4, figsize=(W, H), sharey=True)
    for ax, t in zip(axes, TASKS):
        series = {}
        for part, color in (("sae_only", ACCENT), ("error_only", GREY)):
            pts = sorted((int(r["layer"]), 100 * float(r["nldr"])) for r in rows if r["task"] == t and r["part"] == part)
            series[part] = dict(pts)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=2.2, marker="o", ms=5,
                    markeredgecolor="white", markeredgewidth=0.9, zorder=3)
        if t == "entity_track":
            if noise:
                pts = sorted(noise.items())
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=LIGHT, lw=1.6, ls=(0, (2, 2)), zorder=2)
                ax.text(20.6, noise.get(16, 28) + 4, "noise\nonly", color=INK2, fontsize=8.5, ha="right",
                        va="bottom", linespacing=1.0)
            dip = series["sae_only"][12]
            ax.annotate(f"{round(dip)}%", (12, dip), xytext=(15.2, 70), ha="left", va="center", fontsize=10,
                        fontweight="bold", color=ACCENT, arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.8))
        ax.set_title(TASK[t], loc="left", color=ACCENT if t == "entity_track" else INK)
        ax.set_xticks([4, 12, 20])
        ax.set_ylim(-5, 110)
        ax.set_yticks([0, 50, 100])
        ax.set_yticklabels(["0%", "50%", "100%"])
        ax.set_xlabel("layer", fontsize=10)
        hgrid(ax)
    axes[0].set_ylabel("effect recovered", fontsize=10)
    axes[0].text(12, 82, "SAE\nfeatures", color=ACCENT, fontsize=9.5, va="top", ha="center", fontweight="bold")
    axes[0].text(12, 16, "SAE error", color=GREY, fontsize=9.5, va="bottom", ha="center", fontweight="bold")
    fig.tight_layout(w_pad=0.5)
    save(fig, out, "s_split")


def fig_gap(out):
    """Object tracking, layer 12: SAE features alone vs + one error direction, drawn as an increment on the 26% gap."""
    rows = read("f7_repair_curve")
    get = lambda m, r: 100 * float(next(x["nldr"] for x in rows if x["method"] == m and x["rank"] == str(r)))
    base, rand, err = get("unsupervised_svd", 0), get("random_subspace", 1), get("unsupervised_svd", 1)
    fig, ax = plt.subplots(figsize=(W, 2.2))
    labels = ["SAE features only", "+ 1 random direction", "+ 1 error direction", "full activation (control)"]
    ys = [3, 2, 1, 0]
    ax.barh(3, base, color=GREY, height=0.6)
    ax.barh(2, base, color=GREY, height=0.6)
    ax.barh(2, rand - base, left=base, color=LIGHT, height=0.6)
    ax.barh(1, base, color=GREY, height=0.6)
    ax.barh(1, err - base, left=base, color=ACCENT, height=0.6)
    ax.barh(0, 100, height=0.6, fill=False, edgecolor=GREY, linewidth=1)
    for y, v in zip(ys, (base, rand, err, 100)):
        ax.text(v + 1.5, y, f"{round(v)}%", va="center", fontsize=10.5, color=INK)
    # the gap, and the share of it that one error direction closes
    ax.annotate("", xy=(base, 3.55), xytext=(100, 3.55), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.9))
    ax.text((base + 100) / 2, 3.65, f"gap {round(100 - base)} points", ha="center", va="bottom", fontsize=9.5,
            color=INK2)
    ax.text((base + err) / 2, 1, f"+{round(err - base)}", ha="center", va="center", fontsize=9.5, color="white",
            fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, 112)
    ax.set_ylim(-0.5, 4.1)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("effect recovered (object tracking, layer 12)", fontsize=10)
    ax.spines["bottom"].set_bounds(0, 100)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    save(fig, out, "s_gap")


def fig_specific(out):
    """Share of the gap closed by 2 directions from each source. Same task = the unsupervised fit (the method of the
    gap chart); other tasks = the better of the two transfers."""
    rows = read("f6_specificity")
    panels = [("entity_track", "Object tracking", "12"), ("add_chain", "Running sum", "18")]
    groups = [("same task", lambda s: s == "own task, unsupervised"), ("other tasks", lambda s: s.startswith("other task")),
              ("generic text", lambda s: s == "ordinary web text"), ("random", lambda s: s == "random")]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.2), sharex=True, sharey=True)
    for ax, (t, name, layer) in zip(axes, panels):
        mine = [r for r in rows if r["task"] == t]
        ys = range(len(groups))[::-1]
        for y, (label, match) in zip(ys, groups):
            v = 100 * max(0.0, max(float(r["share_closed"]) for r in mine if match(r["source"])))
            ax.barh(y, v, color=ACCENT if label == "same task" else LIGHT, height=0.6)
            ax.text(v + 1.5, y, f"{round(v)}%", va="center", fontsize=10.5,
                    color=ACCENT if label == "same task" else INK2, fontweight="bold" if label == "same task" else None)
        ax.set_yticks(list(ys))
        ax.set_yticklabels([g[0] for g in groups], fontsize=10.5)
        ax.set_title(f"{name}, layer {layer}\n", loc="left", fontsize=10.5)
        ax.text(0, 1.02, f"gap {pct(mine[0]['gap'])}", transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")
        ax.set_xlim(0, 70)
        ax.set_xticks([0, 25, 50])
        ax.set_xticklabels(["0%", "25%", "50%"])
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[1].tick_params(axis="y", labelleft=False)
    fig.supxlabel("share of the gap closed by 2 directions fitted on each source", fontsize=10, color=INK2, y=0.02)
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "s_specific")


def fig_width(out):
    """Object tracking: the gap as the SAE gets 16x wider; the band is what one error direction closes."""
    rows = [r for r in read("f5_widths") if r["task"] == "entity_track"]
    widths = [r["width"] for r in rows]
    gap = [100 * float(r["gap"]) for r in rows]
    left = [100 * (1 - float(r["plus_1_error_direction"])) for r in rows]
    x = list(range(len(widths)))
    fig, ax = plt.subplots(figsize=(W, H))
    ax.fill_between(x, left, gap, color=ACCENT, alpha=0.12, lw=0)
    ax.plot(x, gap, color=ACCENT, lw=2.2, marker="o", ms=5.5, markeredgecolor="white", markeredgewidth=0.9)
    ax.plot(x, left, color=GREY, lw=2.2, marker="o", ms=5.5, markeredgecolor="white", markeredgewidth=0.9)
    end = len(widths) - 1
    ax.text(end + 0.25, gap[-1] + 1.5, "gap: SAE features only", color=ACCENT, va="center", fontsize=10,
            fontweight="bold")
    ax.text(end + 0.25, left[-1] - 1.5, "gap: + 1 error direction", color=GREY, va="center", fontsize=10,
            fontweight="bold")
    ax.text(1.5, (gap[1] + left[1]) / 2 + 2.5, "closed by one error direction", color=ACCENT, ha="center",
            va="center", fontsize=9, alpha=0.9)
    for i in (0, end):
        ax.annotate(f"{round(gap[i])}%", (i, gap[i]), textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=10.5, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{w}\n({2 ** i}×)" for i, w in enumerate(widths)])   # widths double each step
    ax.set_xlim(-0.3, end + 2.3)
    ax.set_ylim(0, 32)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_yticklabels(["0%", "10%", "20%", "30%"])
    ax.set_ylabel("gap (effect not recovered)", fontsize=10)
    ax.set_xlabel("SAE width (number of features), object tracking, layer 12", fontsize=10)
    ax.spines["bottom"].set_bounds(0, end)
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
