#!/usr/bin/env python
"""Figure: commitment of the exact denoiser on the phi^4 chain (E50, 'speciation').

    python tools/make_fig_speciation.py                        # paper/figures/fig_speciation.pdf (+png)
    python tools/make_fig_speciation.py --out outputs/scratch/spec    # dry run, nothing in paper/

Three panels, full width, from outputs/E50/commitment.csv only (the two replicates are averaged level by level
first, as tools/make_extension_numbers.e50_crossings does):

(a) site-averaged commitment |2p - 1| (commit_all) against the noise level t, at Theta = 0.2, 0.5, 2 on the
    fixed-spacing cells T0.2_h1_L32, T0.5_h1_L32, T2_h1_L32: solid = the exact posterior, dashed = its Gaussian twin
    (twin_commit_all), one colour per Theta; the dotted line is the level 1/2.
(b) the level t_c where each curve first falls through 1/2 (src/markov_score/commitment.crossing_time: linear in
    log t between the bracketing levels) against Theta on the eight fixed-spacing cells (h = 1, L = 32): exact
    (filled) and twin (open).
(c) at Theta = 0.2, the central-site t_c (commit_bulk) against the chain length L = 4 ... 256, with (1/2) ln(1 +
    lambda_max(L)) (lambda_max: largest eigenvalue of the L x L covariance, the column lambda_max) as a second series
    (open markers): the registered prediction P5 compares the two (E50.json).

House rules: tools/figstyle (no prose in a figure, symbols in math mode, one width per placement).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
import figstyle as fs  # noqa: E402
from markov_score.commitment import crossing_time  # noqa: E402

THETAS_A = (0.2, 0.5, 2.0)                                   # panel (a)
THETAS_B = (0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0)          # panel (b): E32's fixed-spacing axis
LENGTHS = (4, 8, 16, 32, 64, 128, 256)                       # panel (c)
THETA_C = 0.2


def cell_fixed(theta: float) -> str:
    return f"T{theta:g}_h1_L32"


def cell_length(n_sites: int) -> str:
    return f"T{THETA_C:g}_h1_L{n_sites}"


# --------------------------------------------------------------------------- data
def pooled(d: pd.DataFrame) -> pd.DataFrame:
    """Level-wise mean over the replicates, per cell (the readings of E50 are made on these curves)."""
    return d.groupby(["cell", "level"]).mean(numeric_only=True).reset_index().sort_values(["cell", "level"])


def curve(g: pd.DataFrame, cell: str, col: str) -> tuple[np.ndarray, np.ndarray]:
    h = g[g.cell == cell].sort_values("level")
    if h.empty:
        raise KeyError(cell)
    return h.level.to_numpy(float), h[col].to_numpy(float)


def crossing(g: pd.DataFrame, cell: str, col: str) -> float:
    return crossing_time(*curve(g, cell, col))


def temperature_axis(g: pd.DataFrame) -> pd.DataFrame:
    """Per fixed-spacing cell: Theta, t_c of the exact posterior and of its Gaussian twin (all sites)."""
    rows = []
    for th in THETAS_B:
        c = cell_fixed(th)
        rows.append(dict(theta=th, exact=crossing(g, c, "commit_all"), twin=crossing(g, c, "twin_commit_all")))
    return pd.DataFrame(rows)


def length_axis(g: pd.DataFrame) -> pd.DataFrame:
    """Per length at Theta = 0.2: central-site t_c and (1/2) ln(1 + lambda_max)."""
    rows = []
    for n in LENGTHS:
        c = cell_length(n)
        lam = float(g[g.cell == c].lambda_max.iloc[0])
        rows.append(dict(L=n, tc=crossing(g, c, "commit_bulk"), predicted=0.5 * np.log1p(lam), lam=lam))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- drawing
def _ticks(axis, values, fmt="{:g}"):
    axis.set_major_locator(FixedLocator(values))
    axis.set_major_formatter(FixedFormatter([fmt.format(v) for v in values]))
    axis.set_minor_formatter(NullFormatter())


def panel_commitment(ax, g: pd.DataFrame) -> None:
    cols = (fs.BLUE_RAMP[4], fs.BLUE_RAMP[2], fs.BLUE_RAMP[0])        # one hue, ordered in Theta; line style is the other channel
    for th, col in zip(THETAS_A, cols):
        c = cell_fixed(th)
        t, y = curve(g, c, "commit_all")
        ax.plot(t, y, color=col, lw=1.4, ls="-", zorder=3)
        t, y = curve(g, c, "twin_commit_all")
        ax.plot(t, y, color=col, lw=1.2, ls="--", zorder=3)
    ax.axhline(0.5, color=fs.MUTED, lw=0.7, ls=":", zorder=1)
    ax.set_xscale("log"); ax.set_xlim(0.02, 4.0); ax.set_ylim(0.0, 1.04)
    _ticks(ax.xaxis, [0.03, 0.1, 0.3, 1, 3])
    ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0", "1/2", "1"])
    ax.set_xlabel(r"noise level $t$"); ax.set_ylabel(r"commitment $|2p-1|$")
    fs.ygrid(ax)
    handles = [plt.Line2D([], [], color=c, lw=1.4) for c in cols]
    labels = [rf"$\Theta={th:g}$" for th in THETAS_A]
    handles += [plt.Line2D([], [], color=fs.INK2, lw=1.4, ls="-"), plt.Line2D([], [], color=fs.INK2, lw=1.2, ls="--")]
    labels += ["exact", "twin"]
    ax.legend(handles, labels, loc="lower left", fontsize=7, handlelength=1.8, labelspacing=0.22,
              borderaxespad=0.3, borderpad=0.2, frameon=False, ncol=1)


def panel_temperature(ax, tab: pd.DataFrame) -> None:
    ax.plot(tab.theta, tab.exact, color=fs.INK, lw=1.2, marker="o", ms=4, mfc=fs.INK, mec=fs.INK, zorder=3)
    ax.plot(tab.theta, tab.twin, color=fs.INK2, lw=1.0, ls="--", marker="o", ms=4, mfc="white", mec=fs.INK2, mew=1.0, zorder=3)
    ax.set_xscale("log"); ax.set_xlim(0.17, 2.4); ax.set_ylim(0.2, 1.75)
    _ticks(ax.xaxis, [0.2, 0.5, 1, 2])
    ax.set_xlabel(r"temperature $\Theta$"); ax.set_ylabel(r"commitment time $t_{\mathrm{c}}$")
    fs.ygrid(ax)


def panel_length(ax, tab: pd.DataFrame) -> None:
    ax.plot(tab.L, tab.tc, color=fs.BLUE_RAMP[4], lw=1.3, marker="o", ms=4, mfc=fs.BLUE_RAMP[4], mec=fs.BLUE_RAMP[4], zorder=4,
            label=r"$t_{\mathrm{c}}$")
    ax.plot(tab.L, tab.predicted, color=fs.CAT[1], lw=1.0, ls="--", marker="s", ms=4, mfc="white", mec=fs.CAT[1], mew=1.0, zorder=3,
            label=r"$\frac{1}{2}\ln(1+\lambda_{\max})$")
    ax.set_xscale("log", base=2); ax.set_xlim(3.2, 330); ax.set_ylim(0.6, 2.05)
    _ticks(ax.xaxis, list(LENGTHS))
    ax.set_xlabel(r"chain length $L$"); ax.set_ylabel(r"level $t$")
    fs.ygrid(ax)
    ax.legend(loc="upper left", fontsize=7, handlelength=2.0, labelspacing=0.3, borderaxespad=0.3, borderpad=0.2, frameon=False)


def make_figure(d: pd.DataFrame, out: Path) -> dict:
    g = pooled(d)
    tab_t, tab_l = temperature_axis(g), length_axis(g)
    fs.apply()
    fig, axes = plt.subplots(1, 3, figsize=(fs.FULL_W, 2.2), gridspec_kw=dict(wspace=0.42))
    panel_commitment(axes[0], g); panel_temperature(axes[1], tab_t); panel_length(axes[2], tab_l)
    for ax, lab in zip(axes, ("(a)", "(b)", "(c)")):
        fs.panel_label(ax, lab, x=-0.24)
    out.parent.mkdir(parents=True, exist_ok=True)
    fs.save(fig, out)
    return dict(fig=fig, axes=axes, pooled=g, temperature=tab_t, length=tab_l)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None, help="dry-run directory (no paper file written)")
    a = ap.parse_args()
    d = pd.read_csv(ROOT / "outputs/E50/commitment.csv")
    out = (ROOT / a.out / "fig_speciation.pdf") if a.out else (ROOT / "paper/figures/fig_speciation.pdf")
    r = make_figure(d, out)
    print("t_c by Theta (exact / twin):", ", ".join(f"{r_.theta:g}: {r_.exact:.3f} / {r_.twin:.3f}" for r_ in r["temperature"].itertuples()))
    print("central-site t_c by L (t_c / (1/2) ln(1+lambda_max)):",
          ", ".join(f"{r_.L}: {r_.tc:.3f} / {r_.predicted:.3f}" for r_ in r["length"].itertuples()))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
