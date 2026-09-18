"""House style for the paper's figures: vector PDF, sans-serif, colour-blind-safe,
grayscale-distinguishable, sized for the ICML column grid.

Widths: single column 3.25 in, full width 6.75 in. Text inside figures is 8 pt at
placement size (7.5 pt for tick labels), the smallest the style allows. Colours are
the validated reference palette (tools/PALETTE.md): categorical slots in fixed order,
one-hue ramps for ordered series, line style as the grayscale channel.
"""

from __future__ import annotations

import re

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
    r"""The paper's own typography, inside matplotlib.

    The ICML class sets the body in Times (\RequirePackage{times}) and leaves mathematics
    in Computer Modern. Figures therefore use a Times-family serif for text and the "cm"
    mathtext set for anything in $...$, which is the closest rendering available without a
    LaTeX binary (there is none on this machine: only tectonic, which matplotlib's usetex
    and PGF back ends cannot drive). A generator never overrides font.family or
    mathtext.fontset; every symbol goes in $...$ so it is set by the math engine and not
    as a Unicode glyph in the text font -- docs/FIGURE_TABLE_STANDARDS.md, F2.
    """
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "mathtext.fontset": "cm", "mathtext.rm": "serif", "mathtext.it": "serif:italic",
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


# ----------------------------------------------------------------------------- the save gate

_GREEK = re.compile(r"[\u0370-\u03FF]")                 # any Unicode Greek letter
_SENTENCE = re.compile(r"[:;.]\s|\n")                     # a clause boundary or a line break
_MAX_WORDS = 3


def _outside_math(s: str) -> str:
    """The part of a label that mathtext will NOT set: everything outside $...$."""
    return re.sub(r"\$[^$]*\$", " ", s)


def _violations(fig):
    """Every text artist that would be prose in a chart, or a symbol set in the text font.

    Axis labels and tick labels are exempt from the word limit (an axis is allowed a
    phrase); nothing is exempt from the symbol rule. docs/FIGURE_TABLE_STANDARDS.md F1, F2.
    """
    bad = []
    for ax in fig.get_axes():
        exempt = {id(ax.xaxis.label), id(ax.yaxis.label)}
        exempt |= {id(t) for t in ax.get_xticklabels() + ax.get_yticklabels()
                   + ax.get_xticklabels(minor=True) + ax.get_yticklabels(minor=True)}
        texts = list(ax.texts) + [ax.xaxis.label, ax.yaxis.label]
        leg = ax.get_legend()
        if leg is not None:
            texts += list(leg.get_texts())
        for t in texts:
            s = t.get_text()
            if not s.strip():
                continue
            plain = _outside_math(s)
            if _GREEK.search(plain):
                bad.append((s, "Greek letter outside $...$ -- set it as math"))
                continue
            if id(t) in exempt:
                continue
            if _SENTENCE.search(s) or len(plain.split()) > _MAX_WORDS:
                bad.append((s, f"prose inside the chart (> {_MAX_WORDS} words or a clause): move it to the caption"))
    for t in fig.texts:
        s = t.get_text()
        if s.strip() and (_SENTENCE.search(s) or len(_outside_math(s).split()) > _MAX_WORDS):
            bad.append((s, "prose inside the figure: move it to the caption"))
    return bad


def save(fig, path, png_dpi=220, **kw):
    """Write a figure as PDF (and PNG) only if it passes the prose and symbol gate.

    Refuses, with every offending label listed, rather than writing a figure that the
    standards forbid. This is the one function generators call to write; `fig.savefig` is
    not used directly anywhere in tools/.
    """
    from pathlib import Path
    bad = _violations(fig)
    if bad:
        msg = "\n".join(f"  {label!r}: {why}" for label, why in bad)
        raise SystemExit(f"figstyle.save refused {Path(path).name}:\n{msg}\n"
                         "(docs/FIGURE_TABLE_STANDARDS.md F1/F2)")
    path = Path(path)
    fig.savefig(path, **kw)
    fig.savefig(path.with_suffix(".png"), dpi=png_dpi, **kw)
    return path
