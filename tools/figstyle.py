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


def halo(*artists, lw=2.2, color="white"):
    """Put a white outline behind text so it stays legible where it must sit near a curve.

    A backstop, not a licence: the first fix for a label on a line is to move the label.
    Use this for labels that are deliberately close to their own series (a curve's own
    name, a reference line's caption), never to excuse a collision that could be avoided.
    """
    import matplotlib.patheffects as pe
    fx = [pe.withStroke(linewidth=lw, foreground=color)]
    for a in artists:
        if a is not None:
            a.set_path_effects(fx)
    return artists[0] if len(artists) == 1 else artists


def repel(values, min_gap, lo=None, hi=None, iters=80):
    """Nudge label positions apart so consecutive ones differ by at least `min_gap`.

    Works in whatever coordinates it is given (use log10 of the data for a log axis).
    Returns positions in the input order; ties are resolved by moving both sides of a
    clash outward, so a label never jumps past its neighbour.
    """
    import numpy as np
    v = np.asarray(values, dtype=float).copy()
    order = np.argsort(v)
    for _ in range(iters):
        moved = False
        for a, b in zip(order[:-1], order[1:]):
            d = v[b] - v[a]
            if d < min_gap:
                shift = (min_gap - d) / 2
                v[a] -= shift; v[b] += shift; moved = True
        if lo is not None:
            v = np.maximum(v, lo)
        if hi is not None:
            v = np.minimum(v, hi)
        if not moved:
            break
    return v


def label_on_curve(ax, xs, ys, text, at_y=None, at_x=None, color=None, fontsize=7,
                   offset=3.0, halo_lw=2.4, **kw):
    """Write a label along a curve, rotated to its slope in DISPLAY space, with a halo.

    For a fan of curves too tightly packed for horizontal labels (five series across a
    two-inch panel), this is the placement that fits: each label rides its own curve at a
    height where the others are elsewhere, so no two can collide. The rotation is computed
    after the transform, so it is correct on log axes and survives a change of limits.

    Give `at_y` (the label sits where the curve crosses that y) or `at_x`. `offset` is in
    points, perpendicular to the curve, positive = above/left of it.
    """
    import numpy as np
    from matplotlib.transforms import ScaledTranslation
    xs = np.asarray(xs, dtype=float); ys = np.asarray(ys, dtype=float)
    if at_y is not None:
        k = int(np.argmin(np.abs(ys - at_y)))
    elif at_x is not None:
        k = int(np.argmin(np.abs(xs - at_x)))
    else:
        k = len(xs) // 2
    i0, i1 = max(0, k - 1), min(len(xs) - 1, k + 1)
    p0 = ax.transData.transform((xs[i0], ys[i0]))
    p1 = ax.transData.transform((xs[i1], ys[i1]))
    dx, dy = p1 - p0
    ang = float(np.degrees(np.arctan2(dy, dx)))
    if ang > 90:
        ang -= 180
    elif ang < -90:
        ang += 180
    n = np.hypot(dx, dy) or 1.0
    px, py = -dy / n, dx / n                        # unit normal, in display space
    dpi = ax.figure.dpi
    tr = ax.transData + ScaledTranslation(px * offset / 72, py * offset / 72,
                                          ax.figure.dpi_scale_trans)
    t = ax.text(xs[k], ys[k], text, transform=tr, rotation=ang, rotation_mode="anchor",
                ha="center", va="center", fontsize=fontsize, color=color, **kw)
    halo(t, lw=halo_lw)
    return t
