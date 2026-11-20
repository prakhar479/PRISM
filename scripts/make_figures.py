#!/usr/bin/env python
"""DEV: figures/ from tables/*.csv only (never from runs/ or tiny/smoke outputs). Each figure states its source table.

    python scripts/make_figures.py [--only gaps,layers,e0,spec] [--filter f_pilot]

  figures/E0_regression.{pdf,png}            harness vs pilot NLDR per condition (tables/E0_regression.csv)
  figures/gaps_<task>_L<layer>_<filter>.*    gap vs full_A of each baseline, per metric (tables/E8_gaps.csv, E2 run)
  figures/E6_layers_<filter>.*               gap vs full_A by layer, OT (tables/E8_gaps.csv, E6 run)
  figures/E8_spec_<filter>.*                 share of the gap closed: same task vs best control, r = 1, 2
                                             (tables/E8_allcells.csv, canonical rows, primary metric)
Color identifies the condition (fixed order, validated palette: dataviz reference palette slots 1-5); every condition
is also named on the axis or in the legend and has its own marker, so identity is never color-alone.
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from harness.profile import REPO_ROOT  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]        # validated (light): fixed order, never cycled
MARKERS = ["o", "s", "D", "^", "v"]
BASELINES = ["feat_A_errB", "feat_A_meanPos", "feat_A_zero", "feat_B_noise", "feat_A_noise"]   # fixed slot per condition
LABEL = {"feat_A_errB": "features-only  x̂_A + ε_B", "feat_A_meanPos": "x̂_A + ε̄_pos", "feat_A_zero": "x̂_A (no error)",
         "feat_B_noise": "noise  x̂_B + η", "feat_A_noise": "x̂_A + η",
         "flip_rate": "flip rate", "prob_recovery": "prob. recovery", "nldr": "NLDR"}
METRICS = ["flip_rate", "prob_recovery", "nldr"]


def style():
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                         "axes.titlesize": 9.5, "axes.titleweight": "bold", "legend.frameon": False,
                         "lines.linewidth": 2})


def save(fig, out: Path, stem: str, source: str):
    fig.text(0.01, 0.005, f"source: {source}", fontsize=6.5, color=INK2, ha="left", va="bottom")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("figure", stem)


def color(cond):
    i = BASELINES.index(cond)
    return SERIES[i], MARKERS[i]


# --------------------------------------------------------------------------- E0
def fig_e0(tables: Path, out: Path):
    d = pd.read_csv(tables / "E0_regression.csv")
    d["label"] = d.apply(lambda x: x.condition + (f" {x.estimator} r={int(x.r)}" if isinstance(x.estimator, str) else ""), axis=1)
    cells = list(d.groupby(["task", "layer"], sort=False))
    same = len({tuple(g["label"]) for _, g in cells}) == 1        # identical condition lists: share one y axis
    fig, axes = plt.subplots(1, len(cells), figsize=(3.6 * len(cells) + 1.6, 0.22 * max(len(g) for _, g in cells) + 1.2),
                             squeeze=False, sharey=same)
    for ax, ((task, layer), g) in zip(axes[0], cells):
        y = np.arange(len(g))[::-1]
        for (col, lo, hi, c, mk, name, dy) in (("pilot_nldr", "pilot_lo", "pilot_hi", SERIES[1], "s", "pilot", 0.15),
                                               ("nldr", "nldr_lo", "nldr_hi", SERIES[0], "o", "harness (E0)", -0.15)):
            ax.errorbar(g[col], y + dy, xerr=[g[col] - g[lo], g[hi] - g[col]], fmt=mk, ms=4, color=c, ecolor=c,
                        elinewidth=1.2, capsize=0, label=name)
        if ax is axes[0][0] or not same:
            ax.set_yticks(y, g["label"], fontsize=7)
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlabel("NLDR (95% CI)")
        ax.set_title(f"{task} L{layer}: max |diff| {g.abs_diff_points.max():.2f} pts")
    axes[0][0].legend(loc="lower left", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, out, "E0_regression", "tables/E0_regression.csv")


# --------------------------------------------------------------------------- gaps (E2) and layers (E6)
def _gap_panel(ax, g, metric):
    g = g[(g.metric == metric) & (g.seed <= 0)]
    conds = [c for c in BASELINES if c in set(g.condition)]
    y = np.arange(len(conds))[::-1]
    for yi, cnd in zip(y, conds):
        x = g[g.condition == cnd].iloc[0]
        c, mk = color(cnd)
        ax.plot([100 * x.gap_lo, 100 * x.gap_hi], [yi, yi], color=c, lw=2, solid_capstyle="round")
        ax.plot(100 * x.gap, yi, mk, color=c, ms=8, mec=SURFACE, mew=1.5)
        ax.annotate(f"{100 * x.gap:.1f}", (100 * x.gap_hi, yi), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7, color=INK2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, [LABEL[c] for c in conds])
    ax.set_title(LABEL[metric])
    ax.set_xlabel("gap vs full_A (points, 95% CI)")


def fig_gaps(tables: Path, out: Path, flt: str):
    d = pd.read_csv(tables / "E8_gaps.csv")
    d = d[(d.exp_id == "E2") & (d["filter"] == flt)]
    for (task, layer), g in d.groupby(["task", "layer"]):
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.6), sharey=True)
        for ax, met in zip(axes, METRICS):
            _gap_panel(ax, g, met)
        x = g.iloc[0]
        fig.suptitle(f"{task} L{layer} ({x.sae}), filter {flt}: n = {x.n_pairs} pairs, flip-rate eligible "
                     f"{x.flip_n_eligible}; run {x.run_hash}", fontsize=8.5, color=INK2, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.03, 1, 0.93))
        save(fig, out, f"gaps_{task}_L{layer}_{flt}", "tables/E8_gaps.csv (E2 run)")


def fig_layers(tables: Path, out: Path, flt: str):
    d = pd.read_csv(tables / "E8_gaps.csv")
    d = d[(d.exp_id == "E6") & (d["filter"] == flt) & (d.seed <= 0)]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9))
    for ax, met in zip(axes, METRICS):
        g = d[d.metric == met]
        for cnd in [c for c in BASELINES if c in set(g.condition)]:
            s = g[g.condition == cnd].sort_values("layer")
            c, mk = color(cnd)
            ax.fill_between(s.layer, 100 * s.gap_lo, 100 * s.gap_hi, color=c, alpha=0.15, lw=0)
            ax.plot(s.layer, 100 * s.gap, color=c, marker=mk, ms=6, mec=SURFACE, mew=1.2, label=LABEL[cnd])
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(LABEL[met])
        ax.set_xlabel("layer")
        ax.set_ylabel("gap vs full_A (points)")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"entity_track, 16k, filter {flt} (bands: 95% CI)", fontsize=8.5, color=INK2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    save(fig, out, f"E6_layers_{flt}", "tables/E8_gaps.csv (E6 run)")


# --------------------------------------------------------------------------- E8 specificity
def fig_spec(tables: Path, out: Path, flt: str):
    p = tables / "E8_allcells.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d = d[d.canonical & (d.metric == "flip_rate") & (d["filter"] == flt) & d.r.isin([1, 2])]
    if d.empty:
        return
    d = d.sort_values(["role", "task", "layer"])
    for est, g in d.groupby("estimator"):
        cells = g.drop_duplicates(["task", "layer"])[["task", "layer", "role"]].values.tolist()
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 0.28 * len(cells) + 1.4), sharey=True)
        for ax, r in zip(axes, (1, 2)):
            s = g[g.r == r].set_index(["task", "layer"])
            y = np.arange(len(cells))[::-1]
            for yi, (t, L, role) in zip(y, cells):
                if (t, L) not in s.index:
                    continue
                x = s.loc[(t, L)]
                if not x.base_gap_resolved:
                    ax.annotate("no gap", (0, yi), fontsize=6.5, color=INK2, va="center")
                    continue
                ax.plot([100 * x.share_same_lo, 100 * x.share_same_hi], [yi, yi], color=SERIES[0], lw=2)
                ax.plot(100 * x.share_same, yi, "o", color=SERIES[0], ms=7, mec=SURFACE, mew=1.2,
                        label="same task" if yi == y[0] else None)
                if pd.notna(x.get("best_control_share")):
                    ax.plot(100 * x.best_control_share, yi, "s", color=SERIES[1], ms=6, mec=SURFACE, mew=1.2,
                            label="best control" if yi == y[0] else None)
            ax.axvline(0, color=INK2, lw=0.8)
            ax.set_yticks(y, [f"{t} L{L}{' *' if role == 'confirmatory' else ''}" for t, L, role in cells], fontsize=7)
            ax.set_title(f"r = {r}")
            ax.set_xlabel("share of the features-only flip-rate gap closed (%)")
        axes[0].legend(fontsize=7, loc="lower right")
        fig.suptitle(f"{est}, filter {flt}; * confirmatory; best control = max(other task, generic, random p95)",
                     fontsize=8.5, color=INK2, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.03, 1, 0.93))
        save(fig, out, f"E8_spec_{est}_{flt}", "tables/E8_allcells.csv (canonical rows)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="e0,gaps,layers,spec")
    ap.add_argument("--filter", default="f_pilot")
    ap.add_argument("--tables", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tables = Path(a.tables) if a.tables else REPO_ROOT / "tables"
    out = Path(a.out) if a.out else REPO_ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    style()
    for name in a.only.split(","):
        {"e0": lambda: fig_e0(tables, out), "gaps": lambda: fig_gaps(tables, out, a.filter),
         "layers": lambda: fig_layers(tables, out, a.filter), "spec": lambda: fig_spec(tables, out, a.filter)}[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
