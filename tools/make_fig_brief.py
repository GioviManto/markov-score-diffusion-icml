#!/usr/bin/env python
"""The one figure for the short brief: the whole result in three panels, for a reader who
does not know the project.

Usage: make_fig_brief.py [--dest reports/figures]

Every number is read from a FROZEN output; nothing here is recomputed or hand-entered.

  (a) what the noise destroys        outputs/E01/shape_information.csv   (frozen)
  (b) structure against networks     outputs/E14/regimes.csv             (frozen)
  (c) what the margin is made of     outputs/E02a/capacity.csv           (frozen)

The three panels are the three sentences of the talk: noise removes fine structure far
faster than it removes correlation (a); an estimator that knows the chain beats a network
given the same information, in every regime tested (b); and the margin is fourth-order
structure, because a model that carries only second-order structure sits exactly on the
second-order floor (c).

Panel (b) plots the EVIDENCE-selected EM-BP arm, not the oracle-selected one: neither side
is allowed the true score to choose when to stop, which is the comparison the paper claims.
Panel (c)'s C = 1 arm is the same estimator with a single Gaussian innovation, so the gap
between the two curves is the fourth-order structure and nothing else.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
import numpy as np
import pandas as pd

import figstyle as fs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
WIDTH = 6.4        # the report's \textwidth at 2.3 cm margins on A4, not the ICML column

# rho and L as the generated macros spell them, so a plotted value can be checked against the
# number the manuscript prints for the same cell
_RHO = {0.5: "five", 0.85: "eightfive", 0.95: "ninefive"}
_L = {32: "thirtytwo", 128: "onetwentyeight"}


def _macros():
    """The manuscript's own numbers for the regime comparison, as {name: value}."""
    src = ROOT / "paper" / "sections" / "regime-baselines-numbers.tex"
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", src.read_text()))


def _agrees(macros, name, value, where):
    """A plotted value must round to the macro the manuscript prints, or this refuses.

    The figure recomputes from the frozen output rather than reading the table, so this is a
    genuine second derivation -- and a second derivation that is never compared is not a check.
    It caught the network arm being averaged over both architectures where the manuscript
    reports the better one, which drew the gap wider than the paper claims.
    """
    m = macros.get(name)
    if m is None:
        raise SystemExit(f"make_fig_brief: no macro {name} to check {where} against")
    if abs(float(m) - value) > 0.5 * 10 ** -(len(m.split(".")[1]) if "." in m else 0) + 1e-9:
        raise SystemExit(f"make_fig_brief: {where} recomputes to {value:.4f} but the manuscript "
                         f"prints {name} = {m}. One of them is wrong; the figure is not written.")


def panel_a(ax):
    """Efficient information per sequence, correlation against shape, Laplace chain."""
    s = pd.read_csv(OUT / "E01" / "shape_information.csv")
    d = s[s.beta == 1.0]
    m = d.groupby("t").agg(rho=("eff_rho", "mean"), beta=("eff_beta", "mean")).reset_index()
    m = m[m.t <= 3.0]   # past t = 3 the outer-product estimate of the shape coordinate floors
    ax.plot(m.t, m.rho, color=fs.CAT[0], lw=1.6)
    ax.plot(m.t, m.beta, color=fs.CAT[1], lw=1.6)
    tt = np.array([1.2, 3.0])
    ax.plot(tt, 3.0e1 * np.exp(-4 * (tt - 1.2)) * 1.1, color=fs.INK2, lw=0.8, ls=":")
    ax.plot(tt, 2.0e-2 * np.exp(-8 * (tt - 1.2)) * 1.6, color=fs.INK2, lw=0.8, ls=":")
    ax.text(2.2, 3.0e1 * np.exp(-4 * 0.95) * 3.0, "$e^{-4t}$", fontsize=7.5, color=fs.INK2)
    ax.text(2.2, 2.0e-2 * np.exp(-8 * 0.95) * 4.0, "$e^{-8t}$", fontsize=7.5, color=fs.INK2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(0.045, 3.4); ax.set_ylim(1e-9, 2e3)
    ax.set_xticks([0.05, 0.1, 0.3, 1, 3]); ax.set_xticklabels(["0.05", "0.1", "0.3", "1", "3"])
    ax.set_xlabel("noise level  $t$")
    ax.set_ylabel("information per sequence")
    fs.label_on_curve(ax, m.t.to_numpy(), m.rho.to_numpy(), r"$\rho$ (correlation)",
                      at_x=0.16, color=fs.CAT[0], fontsize=7.5, offset=5.0, zorder=6)
    fs.label_on_curve(ax, m.t.to_numpy(), m.beta.to_numpy(), r"$\beta$ (shape)",
                      at_x=0.16, color=fs.CAT[1], fontsize=7.5, offset=5.0, zorder=6)
    fs.panel_label(ax, "(a)")


def panel_b(ax):
    """EM-BP (evidence-selected) against the network given the SAME information, per cell."""
    d = pd.read_csv(OUT / "E14" / "regimes.csv")
    mac = _macros()
    cells = sorted({(r, L) for r, L in zip(d.rho, d.n_sites)})
    est, net, labs = [], [], []
    for rho, L in cells:
        c = d[(d.rho == rho) & (d.n_sites == L)]
        # interface C on both sides: one noised view per sequence, no clean chain, and no
        # true score used to choose when to stop. The interface-B network is a DIFFERENT
        # and easier problem (it is given the clean sequence), so it does not belong here.
        # the network arm holds SEVERAL architectures; the paper reports the BEST of them in
        # each cell, so averaging over them here would have plotted a worse network than the
        # paper claims and flattered us by the difference (0.183 against 0.137 at the first cell)
        e = c[c.arm == "em_bp_evidence"].risk.mean()
        n = c[c.arm == "ism_one_view"].groupby("arch").risk.mean().min()
        tag = _RHO[rho] + _L[L]
        _agrees(mac, "rbEmEv" + tag, e, f"EM-BP at ({rho:g}, {L})")
        _agrees(mac, "rbIsmBest" + tag, n, f"best network at ({rho:g}, {L})")
        est.append(e); net.append(n)
        labs.append(rf"${rho:g},\;{L:d}$")
    y = np.arange(len(cells))[::-1]
    est, net = np.array(est), np.array(net)
    for yi, a, b in zip(y, est, net):
        ax.plot([a, b], [yi, yi], color=fs.MUTED, lw=1.0, zorder=1)
    ax.scatter(net, y, s=26, color=fs.CAT[1], zorder=3)
    ax.scatter(est, y, s=26, color=fs.CAT[0], zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(labs)
    # the two series are named directly over the top row instead of in a legend box: with six
    # short rows there is no corner a box does not land in (F3)
    ax.text(est[0], y.max() + 0.62, "EM\u2013BP", fontsize=7.5, color=fs.CAT[0],
            ha="center", va="bottom")
    ax.text(net[0], y.max() + 0.62, "network", fontsize=7.5, color=fs.CAT[1],
            ha="center", va="bottom")
    ax.set_ylim(y.min() - 0.7, y.max() + 1.35)
    ax.set_xlim(0.010, 0.40)
    ax.xaxis.set_major_locator(FixedLocator([0.01, 0.03, 0.1, 0.3]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.01", "0.03", "0.1", "0.3"]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("relative score error")
    ax.set_ylabel(r"$\rho$,  $L$", labelpad=1)
    fs.panel_label(ax, "(b)", x=-0.30)


def panel_c(ax):
    """Error against sample size: the second-order floor, and what goes below it."""
    d = pd.read_csv(OUT / "E02a" / "capacity.csv")
    g = d.groupby(["arm", "components", "n_chains"]).risk.mean().reset_index()
    M = np.array(sorted(d.n_chains.unique()))
    floor = g[g.arm == "lmmse_true"].groupby("n_chains").risk.mean().reindex(M).to_numpy()
    ys = {c: g[(g.arm == "em_bp") & (g.components == c)].set_index("n_chains").risk.reindex(M).to_numpy()
          for c in (1, 8)}
    ax.plot(M, ys[1], color=fs.CAT[1], lw=1.6, marker="o", ms=3.5, zorder=3)
    ax.plot(M, ys[8], color=fs.CAT[0], lw=1.6, marker="o", ms=3.5, zorder=3)
    # the floor is drawn ON TOP: C = 1 sits on it, which is the point of the panel, and a
    # line hidden underneath the series it coincides with shows the reader nothing
    ax.plot(M, floor, color=fs.INK, lw=1.0, ls=(0, (4, 2.5)), zorder=5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(M))
    ax.xaxis.set_major_formatter(FixedFormatter([str(m) for m in M]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1, 0.2]))
    ax.yaxis.set_major_formatter(FixedFormatter(["0.02", "0.05", "0.1", "0.2"]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(M.min() * 0.78, M.max() * 2.6)
    ax.set_ylim(0.015, 0.26)
    ax.set_xlabel("sequences  $M$")
    ax.set_ylabel("relative score error", labelpad=1)
    ax.text(M[-1] * 1.25, ys[1][-1], "$C=1$", fontsize=7.5, color=fs.CAT[1], va="center")
    ax.text(M[-1] * 1.25, ys[8][-1], "$C=8$", fontsize=7.5, color=fs.CAT[0], va="center")
    ax.text(M[0] * 0.92, floor[0] * 1.14, "second-order floor", fontsize=7.5, color=fs.INK)
    fs.panel_label(ax, "(c)", x=-0.26)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=ROOT / "reports" / "figures")
    a = ap.parse_args()
    a.dest.mkdir(parents=True, exist_ok=True)
    fs.apply()
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.02))
    panel_a(axes[0]); panel_b(axes[1]); panel_c(axes[2])
    fig.subplots_adjust(wspace=0.52, left=0.088, right=0.965, bottom=0.20, top=0.94)
    p = fs.save(fig, a.dest / "fig-brief.pdf")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
