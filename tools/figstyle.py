"""House style for the paper's figures: vector PDF, sans-serif, colour-blind-safe,
grayscale-distinguishable, sized for the ICML column grid.

Widths: single column 3.25 in, full width 6.75 in. Text inside figures is 8 pt at
placement size (7.5 pt for tick labels), the smallest the style allows. Colours are
the validated reference palette (tools/PALETTE.md): categorical slots in fixed order,
one-hue ramps for ordered series, line style as the grayscale channel.
"""

from __future__ import annotations

import matplotlib as mpl

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]     # ordinal, light->dark
ORANGE_RAMP = ["#f5b08f", "#f08a5c", "#eb6834", "#c24d1f", "#8f3612"]
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8a8985"; GRID = "#e6e5e1"
LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D"]

COL_W = 3.25
FULL_W = 6.75


def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5,
        "lines.linewidth": 1.4, "lines.markersize": 3.5,
        "legend.frameon": False, "legend.handlelength": 2.2,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def panel_label(ax, text, x=-0.18, y=1.02):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left", color=INK)


def ygrid(ax):
    ax.grid(True, axis="y", color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
