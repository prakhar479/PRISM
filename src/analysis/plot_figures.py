"""Publication figures (PDF + PNG), built only from saved experiment outputs. Missing inputs -> figure skipped.

fig1 causal split vs layer | fig2 PCA manifolds | fig3 spectra + probes | fig4 hybrid repair vs rank
fig5 width scaling         | fig6 last-token vs from-diff patching     | fig7 H3 subspace methods vs null
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils import load_json, load_npz

log = logging.getLogger("prism")

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e3e2de"
SERIES = {"sae_only": "#2a78d6", "error_only": "#eb6834", "sae_splice": "#1baf7a", "noise_control": "#8a8985",
          "error_resample_same_state": "#c4508f", "error_resample_diff_state": "#7a5bd0", "clean_full": "#0b0b0b",
          "mean_error": "#b38f2a"}
WIDTH_LS = {"16k": "-", "65k": "--", "32k": "-.", "131k": ":", "262k": (0, (5, 1)), "524k": (0, (1, 2)), "1m": (0, (3, 1, 1, 1))}
METHOD_COLOR = {"supervised_centroid_svd": "#eb6834", "unsupervised_svd": "#2a78d6", "span_svd": "#1baf7a",
                "generic_text_svd": "#c4508f", "random_subspace": "#8a8985", "top_xhat_svd": "#b38f2a"}
SRC_COLOR = {"eps": "#eb6834", "x_hat": "#2a78d6", "x": "#8a8985"}
SRC_LABEL = {"eps": r"error $\epsilon$", "x_hat": r"SAE recon $\hat{x}$", "x": "full $x$"}
TASK_TITLE = {"add_chain": "Running sum", "entity_track": "Entity tracking", "var_binding": "Variable binding",
              "mod_sum": "Modular sum", "dyck_depth": "Stack depth", "graph_reach": "Graph reach", "ioi_control": "IOI (control)"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8, "legend.frameon": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.6, "lines.markersize": 4,
    "figure.facecolor": "white", "axes.facecolor": "white", "pdf.fonttype": 42,
})


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _tasks(rows: List[dict]) -> List[str]:
    order = list(TASK_TITLE)
    return [t for t in order if any(r["task"] == t for r in rows)]


def _pos(rows: List[dict], positions: str = "last") -> List[dict]:
    return [r for r in rows if r.get("positions", "last") == positions]


def _width_key(w: str) -> int:
    from src.models.sae_loader import parse_width
    return parse_width(w)


def fig1_causal_split(rows: List[dict], out_dir: Path) -> None:
    # from_diff (every token from the first differing one) is the valid test at every layer: patching only the last
    # token at early/middle layers misses the state, which still sits at earlier positions (clean_full ~ 0 there)
    mode = "from_diff" if _pos(rows, "from_diff") else "last"
    rows = _pos(rows, mode)
    if not rows:
        return
    tasks = _tasks(rows)
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.0 * len(tasks), 3.0), sharey=True, squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for iv in ("clean_full", "sae_only", "error_only", "noise_control"):
            for w in sorted({r["width"] for r in rows}, key=_width_key):
                sel = sorted([r for r in rows if r["task"] == task and r["intervention"] == iv and r["width"] == w],
                             key=lambda r: r["layer"])
                if sel:
                    xs = [r["layer"] for r in sel]
                    ax.plot(xs, [r["nldr"] for r in sel], color=SERIES[iv], ls=WIDTH_LS.get(w, "-"), marker="o",
                            lw=1.0 if iv == "clean_full" else 2.0, ms=3 if iv == "clean_full" else 6)
                    if "nldr_lo" in sel[0]:
                        ax.fill_between(xs, [r["nldr_lo"] for r in sel], [r["nldr_hi"] for r in sel],
                                        color=SERIES[iv], alpha=0.12, lw=0)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.axhline(1, color=INK2, lw=0.6, ls=":")
        ax.set_title(TASK_TITLE[task])
        ax.set_xlabel("layer")
    axes[0][0].set_ylabel("NLDR")
    handles = [plt.Line2D([], [], color=SERIES[k], marker="o", label=k)
               for k in ("clean_full", "sae_only", "error_only", "noise_control")]
    handles += [plt.Line2D([], [], color=INK2, ls=WIDTH_LS.get(w, "-"), label=f"width {w}") for w in sorted({r["width"] for r in rows})]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.14))
    fig.suptitle(f"Causal split: which part of the residual carries the state? (patch: {mode})", y=1.02)
    _save(fig, out_dir, "fig1_causal_split")


def fig2_error_manifolds(tensor_dir: Path, rows: List[dict], out_dir: Path, layer=None, width=None) -> None:
    files = sorted(tensor_dir.glob("exp2_*_L*_*.npz"))
    parsed = []
    for f in files:
        task, l, w = f.stem[len("exp2_"):].rsplit("_", 2)
        parsed.append((task, int(l[1:]), w, f))
    if not parsed:
        return
    layers = sorted({p[1] for p in parsed})
    layer = layer if layer is not None else layers[len(layers) // 2]
    width = width if width is not None else sorted({p[2] for p in parsed})[0]
    sel = {p[0]: p[3] for p in parsed if p[1] == layer and p[2] == width}
    tasks = [t for t in TASK_TITLE if t in sel]
    fig, axes = plt.subplots(len(tasks), 2, figsize=(6.2, 2.9 * len(tasks)), squeeze=False)
    fig.subplots_adjust(hspace=0.55, wspace=0.35)
    for i, task in enumerate(tasks):
        z = load_npz(sel[task])
        y = z["labels"]
        classes = np.unique(y)
        cmap = plt.get_cmap("viridis")
        for j, src in enumerate(("eps", "x_hat")):
            ax = axes[i][j]
            P = z[f"pca_{src}"][:, :2]
            sc = ax.scatter(P[:, 0], P[:, 1], c=y, cmap=cmap, s=9, alpha=0.55, linewidths=0)
            cents = np.stack([P[y == c].mean(0) for c in classes])
            ax.plot(cents[:, 0], cents[:, 1], color=INK, lw=0.9, alpha=0.8)
            ax.scatter(cents[:, 0], cents[:, 1], c=classes, cmap=cmap, s=38, edgecolors="white", linewidths=1.2,
                       vmin=y.min(), vmax=y.max(), zorder=3)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"{TASK_TITLE[task]} - {SRC_LABEL[src]}", fontsize=9)
            ax.set_aspect("equal", adjustable="datalim")
        fig.colorbar(sc, ax=axes[i].tolist(), fraction=0.025, pad=0.02, label="state")
    fig.suptitle(f"Test-set PCA of state-conditioned activations (layer {layer}, width {width})", y=0.94)
    _save(fig, out_dir, "fig2_error_manifolds")


def fig3_spectrum_and_probes(tensor_dir: Path, exp2: Dict, out_dir: Path, layer=None, width=None) -> None:
    probes, spectra = exp2["probes"], exp2["spectra"]
    layers = sorted({r["layer"] for r in probes})
    layer = layer if layer is not None else layers[len(layers) // 2]
    width = width if width is not None else sorted({r["width"] for r in probes})[0]
    tasks = [t for t in TASK_TITLE if any(r["task"] == t for r in probes)]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    # (a) cumulative explained variance of the error spectrum
    tcol = {t: c for t, c in zip(TASK_TITLE, ["#2a78d6", "#eb6834", "#1baf7a", "#a35ccf", "#d4a72c", "#3fb8c4", "#8a8985"])}
    for task in tasks:
        f = tensor_dir / f"exp2_{task}_L{layer}_{width}.npz"
        if not f.exists():
            continue
        z = load_npz(f)
        k = np.arange(1, len(z["cumevr_eps"]) + 1)
        a1.plot(k, z["cumevr_eps"], color=tcol[task], label=TASK_TITLE[task])
        a1.plot(k, z["cumevr_x_hat"], color=tcol[task], ls=":", lw=1.1)
    a1.set_xscale("log")
    a1.axvline(16, color=INK2, lw=0.6, ls="--")
    a1.set_xlabel("number of singular directions")
    a1.set_ylabel("cumulative explained variance")
    a1.set_title(r"Spectrum: $\epsilon$ (solid) vs $\hat{x}$ (dotted)")
    a1.legend(loc="lower right")
    # (b) classification accuracy, (c) R^2 (circular for mod_sum, ordinal otherwise)
    w = 0.26
    xs = np.arange(len(tasks))
    for ax, key, title in ((a2, "acc", "State decoding accuracy (test)"), (a3, "r2", "Probe $R^2$ (test)")):
        for k, src in enumerate(("x", "x_hat", "eps")):
            vals = []
            for task in tasks:
                r = [q for q in probes if q["task"] == task and q["layer"] == layer and q["width"] == width and q["source"] == src]
                if not r:
                    vals.append(np.nan)
                elif key == "acc":
                    vals.append(r[0]["acc"])
                else:
                    vals.append(r[0].get("r2_circular", r[0]["r2_ordinal"]) if task == "mod_sum" else r[0]["r2_ordinal"])
            ax.bar(xs + (k - 1) * w, vals, w * 0.9, color=SRC_COLOR[src], label=SRC_LABEL[src])
        surf = []      # prompt-token baseline: a black tick across each task's bars
        for task in tasks:
            r = [q for q in probes if (q["task"], q["layer"], q["width"], q["source"]) == (task, layer, width, "surface")]
            surf.append(np.nan if not r else r[0]["acc"] if key == "acc" else
                        (r[0].get("r2_circular", r[0]["r2_ordinal"]) if task == "mod_sum" else r[0]["r2_ordinal"]))
        if not np.all(np.isnan(surf)):
            ax.hlines(surf, xs - 1.5 * w, xs + 1.5 * w, colors=INK, lw=1.4, label="prompt tokens only")
        ax.set_xticks(xs)
        ax.set_xticklabels([TASK_TITLE[t].replace(" (control)", "") for t in tasks], rotation=30, ha="right")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    a3.set_ylabel(r"$R^2$ (circular for mod. sum)")
    fig.legend(*a2.get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=7, frameon=False)
    fig.suptitle(f"Error geometry and state probes (layer {layer}, width {width})", y=1.03)
    fig.tight_layout(rect=(0, 0.07, 1, 1))     # bottom strip reserved for the shared legend
    _save(fig, out_dir, "fig3_spectrum_and_probes")


def fig4_hybrid_pareto(rows: List[dict], out_dir: Path, overhead: Dict = None, layer=None) -> None:
    rows = _pos(rows, "last")
    if not rows:
        return
    tasks = _tasks(rows)
    layers = sorted({r["layer"] for r in rows})
    layer = layer if layer is not None else layers[len(layers) // 2]
    widths = sorted({r["width"] for r in rows}, key=_width_key)
    mcol = {"unsupervised_svd": "#2a78d6", "supervised_centroid_svd": "#eb6834"}
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.0 * len(tasks), 3.1), sharey=True, squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for method, col in mcol.items():
            for w in widths:
                pts = sorted([(r["rank"], r["nldr"]) for r in rows if r["task"] == task and r["layer"] == layer
                              and r["width"] == w and r["method"] == method and r["kind"] == "hybrid_patch"])
                if pts:
                    ax.plot(*zip(*pts), color=col, ls=WIDTH_LS.get(w, "-"), marker="o")
        ref = [r["nldr"] for r in rows if r["task"] == task and r["layer"] == layer and r["method"] == "reference"]
        if ref:
            ax.axhline(float(np.mean(ref)), color=INK2, ls=":", lw=1)
        rnd = sorted([(r["rank"], r["nldr"]) for r in rows if r["task"] == task and r["layer"] == layer
                      and r["width"] == widths[0] and r["method"] == "random_subspace" and r["kind"] == "hybrid_patch"])
        if rnd:
            ax.plot(*zip(*rnd), color="#8a8985", lw=1.0, marker=".", ls="-")
        ax.axhline(0.9, color="#1baf7a", lw=0.8, ls="--")
        ax.set_xscale("symlog", linthresh=1, base=2)
        ax.set_xticks([0, 1, 2, 4, 8, 16, 32, 64])
        ax.set_xticklabels(["0", "1", "2", "4", "8", "16", "32", "64"])
        ax.set_title(TASK_TITLE[task])
        ax.set_xlabel("subspace rank $r$")
    axes[0][0].set_ylabel("NLDR (hybrid patch)")
    handles = [plt.Line2D([], [], color=c, marker="o", label=m.replace("_", " ")) for m, c in mcol.items()]
    handles += [plt.Line2D([], [], color=INK2, ls=WIDTH_LS.get(w, "-"), label=f"SAE {w}") for w in widths]
    handles += [plt.Line2D([], [], color="#8a8985", marker=".", label="random subspace"),
                plt.Line2D([], [], color=INK2, ls=":", label="full error_only"),
                plt.Line2D([], [], color="#1baf7a", ls="--", lw=0.8, label="0.9 target")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.16))
    sub = ""
    if overhead and "width_scaling_extra_params_16k_to_65k" in overhead:
        sub = f" (widening 16k->65k adds {overhead['width_scaling_extra_params_16k_to_65k'] / 1e6:.0f}M params; r=64 adds {64 * overhead['d_model'] / 1e3:.0f}K)"
    fig.suptitle(f"Hybrid low-rank repair, layer {layer}{sub}", y=1.03)
    _save(fig, out_dir, "fig4_hybrid_pareto")


def fig5_width_scaling(exp1: List[dict], exp2: Dict, exp3: List[dict], out_dir: Path) -> None:
    """At the layer with the most widths: recon error, causal split and eps-probe quality vs SAE width."""
    rows = _pos(exp1, "last")
    if not rows:
        return
    by_layer = {}
    for r in rows:
        by_layer.setdefault(r["layer"], set()).add(r["width"])
    layer = max(by_layer, key=lambda l: (len(by_layer[l]), -abs(l - 12)))
    widths = sorted(by_layer[layer], key=_width_key)
    if len(widths) < 3:
        return
    tasks = _tasks(rows)
    tcol = dict(zip(TASK_TITLE, ["#2a78d6", "#eb6834", "#1baf7a", "#8a8985"]))
    xs = [_width_key(w) for w in widths]
    fig, (a1, a2, a3, a4) = plt.subplots(1, 4, figsize=(13, 3.1))
    for task in tasks:
        g = lambda iv, w: next((r for r in rows if (r["task"], r["layer"], r["width"], r["intervention"]) == (task, layer, w, iv)), None)
        rel = [g("error_only", w)["rel_recon_err"] if g("error_only", w) else np.nan for w in widths]
        a1.plot(xs, rel, color=tcol[task], marker="o", label=TASK_TITLE[task])
        for iv, ls in (("error_only", "-"), ("sae_only", "--")):
            a2.plot(xs, [g(iv, w)["nldr"] if g(iv, w) else np.nan for w in widths], color=tcol[task], ls=ls, marker="o")
        if exp2:
            pv = lambda src, w: next((p.get("r2_circular", p["r2_ordinal"]) if task == "mod_sum" else p["acc"]
                                      for p in exp2["probes"] if (p["task"], p["layer"], p["width"], p["source"]) == (task, layer, w, src)), np.nan)
            a3.plot(xs, [pv("eps", w) for w in widths], color=tcol[task], marker="o")
            a3.plot(xs, [pv("x_hat", w) for w in widths], color=tcol[task], marker="o", ls="--")
        if exp3:
            def best(w):
                ref = [r["nldr"] for r in exp3 if (r["task"], r["layer"], r["width"], r["method"]) == (task, layer, w, "reference")
                       and r.get("positions", "last") == "last"]
                sel = sorted([(r["rank"], r["nldr"]) for r in exp3 if (r["task"], r["layer"], r["width"], r["method"], r["kind"])
                              == (task, layer, w, "supervised_centroid_svd", "hybrid_patch") and r.get("positions", "last") == "last"])
                if not ref or not sel:
                    return np.nan
                hit = [k for k, v in sel if v >= 0.9 * ref[0]]
                return hit[0] if hit else np.nan
            a4.plot(xs, [best(w) for w in widths], color=tcol[task], marker="o")
    for ax in (a1, a2, a3, a4):
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels(widths, rotation=30)
        ax.set_xlabel("SAE width")
    a1.set_title(r"$\|\epsilon\|^2/\|x\|^2$")
    a2.set_title("NLDR: error_only (solid) / sae_only (dashed)")
    a3.set_title(r"probe: $\epsilon$ (solid) / $\hat{x}$ (dashed)")
    a3.set_ylabel("acc (circ. $R^2$ for mod sum)")
    a4.set_title("rank for 90% of error_only")
    a1.legend(fontsize=7)
    fig.suptitle(f"Width scaling at layer {layer}: does the state stay in the error as the dictionary grows?", y=1.04)
    fig.tight_layout()
    _save(fig, out_dir, "fig5_width_scaling")


def fig6_positions(rows: List[dict], out_dir: Path) -> None:
    """error_only / sae_only / clean_full NLDR vs layer, last-token vs from-diff patching (first width)."""
    if not _pos(rows, "from_diff") or not _pos(rows, "last"):
        return
    tasks = _tasks(rows)
    w0 = sorted({r["width"] for r in rows}, key=_width_key)[0]
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.0 * len(tasks), 3.0), sharey=True, squeeze=False)
    for ax, task in zip(axes[0], tasks):
        for iv in ("sae_only", "error_only", "clean_full"):
            for pos, ls in (("last", "--"), ("from_diff", "-")):
                sel = sorted([r for r in rows if (r["task"], r["width"], r["intervention"], r.get("positions", "last")) == (task, w0, iv, pos)],
                             key=lambda r: r["layer"])
                if sel:
                    ax.plot([r["layer"] for r in sel], [r["nldr"] for r in sel], color=SERIES[iv], ls=ls, marker="o", ms=3)
        ax.axhline(0, color=INK2, lw=0.6)
        ax.axhline(1, color=INK2, lw=0.6, ls=":")
        ax.set_title(TASK_TITLE[task])
        ax.set_xlabel("layer")
    axes[0][0].set_ylabel("NLDR")
    handles = [plt.Line2D([], [], color=SERIES[k], marker="o", label=k) for k in ("sae_only", "error_only", "clean_full")]
    handles += [plt.Line2D([], [], color=INK2, ls="--", label="last token"), plt.Line2D([], [], color=INK2, ls="-", label="from first diff token")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.14))
    fig.suptitle(f"Where is the state? last-token vs multi-position patching (SAE {w0})", y=1.02)
    _save(fig, out_dir, "fig6_positions")


def fig7_subspace_methods(rows: List[dict], out_dir: Path, layer=None) -> None:
    """H3 against nulls: NLDR vs rank for every subspace method (hybrid patch, first width)."""
    methods = sorted({r["method"] for r in rows if r["method"] != "reference"})
    if len(methods) <= 2:
        return
    tasks = _tasks(rows)
    layers = sorted({r["layer"] for r in rows})
    layer = layer if layer is not None else layers[len(layers) // 2]
    w0 = sorted({r["width"] for r in rows}, key=_width_key)[0]
    modes = [p for p in ("last", "from_diff") if _pos(rows, p)]
    fig, axes = plt.subplots(len(modes), len(tasks), figsize=(3.0 * len(tasks), 2.8 * len(modes)), sharey=True, squeeze=False)
    tcol = {}
    for i, pos in enumerate(modes):
        for ax, task in zip(axes[i], tasks):
            sel = [r for r in rows if (r["task"], r["layer"], r["width"], r.get("positions", "last"), r["kind"]) == (task, layer, w0, pos, "hybrid_patch")]
            for m in methods:
                pts = sorted([(r["rank"], r["nldr"]) for r in sel if r["method"] == m])
                if pts:
                    c = METHOD_COLOR.get(m) or tcol.setdefault(m, plt.get_cmap("tab10")(len(tcol) % 10))
                    ax.plot(*zip(*pts), color=c, marker="o", ms=3, lw=1.2, ls="-" if not m.startswith("transfer") else "--")
            ref = [r["nldr"] for r in rows if (r["task"], r["layer"], r["width"], r.get("positions", "last"), r["method"]) == (task, layer, w0, pos, "reference")]
            if ref:
                ax.axhline(ref[0], color=INK2, ls=":", lw=1)
            ax.set_xscale("symlog", linthresh=1, base=2)
            rk = sorted({r["rank"] for r in sel})
            ax.set_xlim(0, max(rk) if rk else 1)
            ax.set_xticks(rk)
            ax.set_xticklabels([str(k) for k in rk], fontsize=7)
            ax.set_title(f"{TASK_TITLE[task]} [{pos}]", fontsize=9)
            ax.set_xlabel("rank $r$")
        axes[i][0].set_ylabel("NLDR (hybrid patch)")
    handles = [plt.Line2D([], [], color=METHOD_COLOR.get(m) or tcol.get(m, INK2), marker="o", ms=3,
                          ls="--" if m.startswith("transfer") else "-", label=m) for m in methods]
    handles.append(plt.Line2D([], [], color=INK2, ls=":", label="full error_only"))
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), bbox_to_anchor=(0.5, -0.04 - 0.03 * (len(handles) // 4)))
    fig.suptitle(f"Which error subspace repairs the state? (layer {layer}, SAE {w0})", y=1.01)
    fig.tight_layout()
    _save(fig, out_dir, "fig7_subspace_methods")


def make_all_figures(results_dir: Path, out_dir: Path = None) -> List[Path]:
    results_dir = Path(results_dir)
    out_dir = Path(out_dir) if out_dir else results_dir / "figures"
    m = results_dir / "metrics"
    load = lambda n: load_json(m / n) if (m / n).exists() else None
    exp1, exp2, e3 = load("exp1_results.json"), load("exp2_results.json"), load("exp3_results.json")
    if exp2 is not None and not exp2.get("probes"):
        exp2 = None
    jobs = []
    if exp1:
        jobs += [lambda: fig1_causal_split(exp1, out_dir), lambda: fig6_positions(exp1, out_dir),
                 lambda: fig5_width_scaling(exp1, exp2, e3["rows"] if e3 else [], out_dir)]
    if exp2:
        jobs += [lambda: fig2_error_manifolds(results_dir / "tensors", exp2["probes"], out_dir),
                 lambda: fig3_spectrum_and_probes(results_dir / "tensors", exp2, out_dir)]
    if e3 and e3.get("rows"):
        jobs += [lambda: fig4_hybrid_pareto(e3["rows"], out_dir, e3.get("overhead")),
                 lambda: fig7_subspace_methods(e3["rows"], out_dir)]
    for j in jobs:
        try:
            j()
        except Exception as e:  # noqa: BLE001 - one broken figure must not block the others
            log.error("figure failed: %s: %s", type(e).__name__, e)
            plt.close("all")
    return sorted(out_dir.glob("fig*.pdf"))
