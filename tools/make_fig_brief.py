#!/usr/bin/env python
"""The one figure for the short brief: the whole result in three panels, for a reader who
does not know the project.

Usage: make_fig_brief.py [--dest reports/figures]

Every number is read from a FROZEN output; nothing here is recomputed or hand-entered.

  (a) structure against networks     outputs/E14/regimes.csv             (frozen)
  (b) what the margin is made of     outputs/E02a/capacity.csv           (frozen)

The two panels are the two sentences of the result: an estimator that knows the chain beats a
network given the same information, in every regime tested (a); and the margin is fourth-order
structure, because the same estimator restricted to second-order structure sits exactly on the
second-order floor and stays there (b).

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


def _agrees(macros, name, value, where, tol_digits=None):
    """A plotted value must round to the macro the manuscript prints, or this refuses.

    The figure recomputes from the frozen output rather than reading the table, so this is a
    genuine second derivation -- and a second derivation that is never compared is not a check.
    It caught the network arm being averaged over both architectures where the manuscript
    reports the better one, which drew the gap wider than the paper claims.
    """
    m = macros.get(name)
    if m is None:
        raise SystemExit(f"make_fig_brief: no macro {name} to check {where} against")
    dig = tol_digits if tol_digits is not None else (len(m.split(".")[1]) if "." in m else 0)
    if abs(float(m) - value) > 0.5 * 10 ** -dig + 1e-9:
        raise SystemExit(f"make_fig_brief: {where} recomputes to {value:.4f} but the manuscript "
                         f"prints {name} = {m}. One of them is wrong; the figure is not written.")


def panel_comparison(ax):
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
    fs.panel_label(ax, "(a)", x=-0.22)


def panel_capacity(ax):
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
    fs.panel_label(ax, "(b)", x=-0.20)


def _mac(x):
    """The spelling make_misspec_numbers.py uses for a number inside a macro name."""
    return "".join({"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
                    "6": "six", "7": "seven", "8": "eight", "9": "nine"}.get(c, c)
                   for c in str(x)).replace(".", "p")


# The four departures, in the order that matters: the two that cost us the advantage first.
# Named, not symbolised -- the paper calls the rank-one loading beta too, and in a page that has
# just spent a paragraph on beta as the innovation shape that collision is not worth risking.
MECHANISMS = [("ar2", "second order of memory"), ("sitevar", "correlation varies along chain"),
              ("longrange", "long-range coupling"), ("global", "rank-one global latent")]
SIZE_TAG = {512: "C", 2048: "D"}


def fig_boundary(dest):
    """How far the first-order assumption may be wrong before the advantage goes."""
    d = pd.read_csv(OUT / "E08" / "misspecification.csv")
    d = d[d.region == "all"]
    src = ROOT / "paper" / "sections" / "misspec-numbers.tex"
    mac = dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}", src.read_text()))
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 1.85), sharey=True)
    for ax, (mech, title) in zip(axes, MECHANISMS):
        for j, M in enumerate((512, 2048)):
            g = d[(d.mechanism == mech) & (d.n_chains == M)]
            xs, ys = [], []
            for eps, c in g.groupby("eps"):
                w = c[c.arm == "window"].set_index("seed").risk
                e = c[c.arm == "em_bp"].set_index("seed").risk
                r = float((w / e).mean())          # mean of per-seed ratios, as the paper defines it
                _agrees(mac, f"misRatio{_mac(mech).capitalize()}E{_mac(eps)}{SIZE_TAG[M]}", r,
                        f"{mech} ratio at eps={eps:g}, M={M}", tol_digits=2)
                xs.append(eps); ys.append(r)
            ax.plot(xs, ys, color=fs.CAT[j], marker=fs.MARKERS[j], ms=3, lw=1.3,
                    ls=fs.LINESTYLES[j], label=f"$M={M}$")
        ax.axhline(1.0, color=fs.INK2, lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.set_xlabel("strength", labelpad=1)
    axes[0].set_ylabel("window / EM\u2013BP error")
    axes[0].legend(loc="upper right", fontsize=7, borderpad=0.2, handletextpad=0.4,
                   borderaxespad=0.2, labelspacing=0.2)
    fig.subplots_adjust(wspace=0.16, left=0.093, right=0.985, bottom=0.25, top=0.84)
    return fs.save(fig, dest / "fig-boundary.pdf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=ROOT / "reports" / "figures")
    a = ap.parse_args()
    a.dest.mkdir(parents=True, exist_ok=True)
    fs.apply()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.35))
    panel_comparison(axes[0]); panel_capacity(axes[1])
    fig.subplots_adjust(wspace=0.30, left=0.105, right=0.965, bottom=0.175, top=0.95)
    p = fs.save(fig, a.dest / "fig-brief.pdf")
    print(f"wrote {p}")
    print(f"wrote {fig_boundary(a.dest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
