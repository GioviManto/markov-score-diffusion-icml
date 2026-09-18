#!/usr/bin/env python
"""Figure 2 — what survives diffusion noise, order by order (from outputs/E01).

(a) Exact Gaussian-chain information about rho, scaled by its predicted asymptote
    (1/2)||d_rho Sigma_0||_F^2 e^{-4t}, for five rho: the curves reach 1, and the
    crossover t_* = (1/2) log lambda_max(Sigma_0) (ticks) sets when.
(b) Information about the correlation coordinate rho (degree 2) and the shape
    coordinate beta (degree 4) of the generalised-Gaussian chain, efficient
    information per noised sequence, mean over replicates with the replicate band;
    guide lines e^{-4t} and e^{-8t}. Laplace (beta = 1) and the platykurtic member
    (beta = 3).
(c) The price of second-order closure: E||S_t - S_t^G||^2 measured by tree-exact BP,
    divided by the predicted C_4 e^{-8t} with C_4 = ||kappa_4||_F^2 / 6. Four
    families; the two with finite eighth moments converge to 1, Student-t(5) departs.

Every number drawn here comes from the CSVs; the fits quoted in the text come from
tools/make_theory_numbers.py on the same files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "E01"
# heights at which each rho curve in panel (a) carries its own label; staggered so that no
# two labels can share a neighbourhood whatever the curves do between them
LABEL_HEIGHTS = (None, 0.88, 0.64, 0.50, 0.13)
# alternate the side of the curve each label sits on: the rho = 0.85 and 0.95 curves
# run close together everywhere, so opposite offsets buy the clearance heights cannot
LABEL_SIDES = (None, 4.2, 4.2, -4.6, -4.6)
OUT = ROOT / "paper" / "figures"


def main():
    fs.apply()
    g = pd.read_csv(SRC / "gaussian_information.csv")
    s = pd.read_csv(SRC / "shape_information.csv")
    c = pd.read_csv(SRC / "closure_gap.csv")

    fig, axes = plt.subplots(1, 3, figsize=(fs.FULL_W, 2.15), gridspec_kw=dict(wspace=0.42))

    # ---- (a) exact Gaussian chain, scaled correlation information
    ax = axes[0]
    rhos = sorted(g.rho.unique())
    curves = []
    for i, rho in enumerate(rhos):
        d = g[g.rho == rho].sort_values("t")
        col = fs.BLUE_RAMP[i]
        ax.plot(d.t, d.I_rho_scaled, color=col, lw=1.4, zorder=3)
        ts = float(d.t_star.iloc[0])
        if ts > 0:
            y = np.interp(ts, d.t, d.I_rho_scaled)
            ax.plot([ts], [y], marker="|", ms=7, mew=1.2, color=col, zorder=4)
        curves.append((rho, d, col, ts))
    ax.axhline(1.0, color=fs.MUTED, lw=0.6, ls=":", zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 6.5)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("diffusion time  $t$")
    ax.set_ylabel(r"$I_t(\rho)\,e^{4t}\,/\,\frac{1}{2}\|\partial_\rho\Sigma_0\|_F^2$")
    ax.set_xticks([0.03, 0.1, 0.3, 1, 3])
    ax.set_xticklabels(["0.03", "0.1", "0.3", "1", "3"])
    # Five labels will not fit side by side across a two-inch panel, and stacking them at the
    # crossover ticks put rho = 0.6 on top of rho = 0.85. Each rides its own curve instead, at
    # a height the others have already left, rotated to the local slope and haloed.
    for (rho, d, col, ts), at, off in zip(curves, LABEL_HEIGHTS, LABEL_SIDES):
        if ts > 0:
            fs.label_on_curve(ax, d.t.to_numpy(), d.I_rho_scaled.to_numpy(), f"ρ = {rho:g}",
                              at_y=at, color=col, fontsize=7, offset=off, zorder=6)
        else:
            t = ax.text(0.024, 0.965, f"ρ = {rho:g}", color=col, fontsize=7, va="top", ha="left", zorder=6)
            fs.halo(t)
    fs.panel_label(ax, "(a)")

    # ---- (b) shape vs correlation information, generalised-Gaussian chain
    ax = axes[1]
    picks = [(1.0, "-", "Laplace, β = 1"), (3.0, "--", "β = 3")]
    for beta, ls, lab in picks:
        d = s[s.beta == beta]
        m = d.groupby("t").agg(rho_m=("eff_rho", "mean"), rho_lo=("eff_rho", "min"), rho_hi=("eff_rho", "max"),
                               b_m=("eff_beta", "mean"), b_lo=("eff_beta", "min"), b_hi=("eff_beta", "max")).reset_index()
        m = m[m.t <= 3.0]  # beyond t = 3 the outer-product estimate of the shape coordinate is at its numerical floor
        ax.fill_between(m.t, m.rho_lo, m.rho_hi, color=fs.CAT[0], alpha=0.18, lw=0)
        ax.fill_between(m.t, m.b_lo, m.b_hi, color=fs.CAT[1], alpha=0.18, lw=0)
        ax.plot(m.t, m.rho_m, color=fs.CAT[0], ls=ls, lw=1.4)
        ax.plot(m.t, m.b_m, color=fs.CAT[1], ls=ls, lw=1.4)
        if beta == 1.0:
            m_lab = {"rho": (m.t.to_numpy(), m.rho_m.to_numpy()),
                     "beta": (m.t.to_numpy(), m.b_m.to_numpy())}
    tt = np.array([1.2, 3.0])
    ax.plot(tt, 3.0e1 * np.exp(-4 * (tt - 1.2)) * 1.1, color=fs.INK2, lw=0.8, ls=":")
    ax.plot(tt, 2.0e-2 * np.exp(-8 * (tt - 1.2)) * 1.6, color=fs.INK2, lw=0.8, ls=":")
    ax.text(2.15, 3.0e1 * np.exp(-4 * 0.95) * 2.6, "$e^{-4t}$", fontsize=7.5, color=fs.INK2, ha="left")
    ax.text(2.15, 2.0e-2 * np.exp(-8 * 0.95) * 3.5, "$e^{-8t}$", fontsize=7.5, color=fs.INK2, ha="left")
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xlim(0.045, 3.4)
    ax.set_ylim(1e-9, 2e3)
    ax.set_xticks([0.05, 0.1, 0.3, 1, 3]); ax.set_xticklabels(["0.05", "0.1", "0.3", "1", "3"])
    ax.set_xlabel("diffusion time  $t$")
    ax.set_ylabel("efficient information per sequence")
    # These two labels used to be placed by hand above their curves and landed on them. Ride
    # the curve instead, on the near-flat small-t section where the two series are three
    # decades apart; the guide lines carry the degree, so the names can stay short.
    lab_rho = m_lab["rho"]; lab_beta = m_lab["beta"]
    fs.label_on_curve(ax, lab_rho[0], lab_rho[1], "ρ  (correlation)", at_x=0.16,
                      color=fs.CAT[0], fontsize=7.5, offset=5.0, zorder=6)
    fs.label_on_curve(ax, lab_beta[0], lab_beta[1], "β  (shape)", at_x=0.16,
                      color=fs.CAT[1], fontsize=7.5, offset=5.0, zorder=6)
    ax.text(0.05, 6e-8, "solid: Laplace (β = 1)\ndashed: β = 3", fontsize=7, color=fs.INK2, va="bottom")
    fs.panel_label(ax, "(b)")

    # ---- (c) closure gap, measured / predicted
    ax = axes[2]
    fams = [("laplace", "Laplace", fs.CAT[0], "-", "o"), ("bimodal", "bimodal", fs.CAT[1], "--", "s"),
            ("uniform", "uniform", fs.CAT[2], "-.", "^"), ("student", "Student-$t_5$", fs.CAT[3], ":", "D")]
    ends = []
    for key, lab, col, ls, mk in fams:
        d = c[c.family == key]
        m = d.groupby("t").apply(lambda x: (x.sq_gap_per_chain / x.pred_sq_gap_per_chain).mean(), include_groups=False).reset_index(name="ratio")
        m = m[m.t >= 0.3]
        ax.plot(m.t, m.ratio, color=col, ls=ls, marker=mk, ms=3, lw=1.3, mec=col, mfc="white" if mk in ("s", "D") else col, zorder=3)
        ends.append((float(m.t.iloc[-1]), float(m.ratio.iloc[-1]), lab, col))
    ax.axhline(1.0, color=fs.MUTED, lw=0.6, ls=":", zorder=1)
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xlim(0.28, 14.0); ax.set_ylim(3e-3, 60)
    from matplotlib.ticker import NullFormatter
    ax.set_xticks([0.3, 1, 3]); ax.set_xticklabels(["0.3", "1", "3"]); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("diffusion time  $t$")
    ax.set_ylabel(r"$\mathbb{E}\|S_t - S_t^{G}\|^2\;/\;C_4 e^{-8t}$")
    # Right-margin family labels, repelled in log space and kept inside the panel. Three of
    # the four curves finish within a factor of two of each other, so hand-tuned offsets put
    # "uniform" and "Laplace" on the curves and pushed "Student" off the top of the axes.
    ylo, yhi = ax.get_ylim()
    # the minimum gap is the text height converted to decades, not a guessed constant: a
    # 7.5 pt label needs about 0.104 in, and the panel gives (log yhi - log ylo) decades over
    # its own height, so anything smaller is a collision by construction
    h_in = ax.get_window_extent().height / ax.figure.dpi
    gap = 0.104 / (h_in / (np.log10(yhi) - np.log10(ylo)))
    ys = fs.repel([np.log10(v) for _, v, _, _ in ends], gap,
                  lo=np.log10(ylo) + gap, hi=np.log10(yhi) - gap)
    xr = max(t for t, _, _, _ in ends)
    for (t0, v, lab, col), yl in zip(ends, ys):
        a = ax.annotate(lab, xy=(t0 * 1.04, v), xytext=(xr * 1.55, 10 ** yl), color=col,
                        fontsize=7.5, va="center", ha="left",
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.5, alpha=0.6, shrinkA=0, shrinkB=1)
                        if abs(yl - np.log10(v)) > 0.04 else None)
        fs.halo(a, lw=1.8)
    fs.panel_label(ax, "(c)")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_theory.pdf")
    fig.savefig(OUT / "fig_theory.png", dpi=220)
    print("wrote", OUT / "fig_theory.pdf")


if __name__ == "__main__":
    main()
