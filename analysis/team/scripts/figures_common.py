"""Shared style + I/O for the paper figures (fig_*.py). Exploratory/post hoc figures of PRISM results.

Palette: dataviz reference categorical slots 1-4 in fixed order (blue, orange, aqua, violet), validated with
validate_palette.js (light surface, adjacent CVD dE >= 9.2, normal-vision >= 27.6). Identity never rests on colour alone:
every series also has its own marker and/or line style so the figures survive grayscale printing.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SP = __import__("layout_shim").root()
FINAL = SP / "hw" / "tables" / "final"
TABLES = SP / "hw" / "tables"
TEAM = SP / "team"
FIG = TEAM / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SINGLE = 3.3   # in, single column
DOUBLE = 6.9   # in, double column

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
RAND = "#b9b8b1"       # random-subspace envelope fill (neutral, not a series colour)
RAND_LINE = "#7a7974"  # random mean / p95 line


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 7.5,
        "axes.labelcolor": INK,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": 6.6,
        "legend.frameon": False,
        "legend.handlelength": 2.2,
        "legend.borderaxespad": 0.3,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "errorbar.capsize": 0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 200,
        "figure.dpi": 100,
        "text.color": INK,
    })


def save(fig, name, data, note=None):
    """Write <name>.pdf, <name>.png (200 dpi) and <name>_data.csv (exactly the plotted numbers)."""
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    data.to_csv(FIG / f"{name}_data.csv", index=False)
    if note:
        (FIG / f"{name}_notes.txt").write_text(note)
    plt.close(fig)
    print("wrote", FIG / name, "(.pdf/.png/_data.csv)", len(data), "rows")


def panel_label(ax, s, x=-0.02, y=1.02):
    ax.set_title(s, loc="left", pad=3)
