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
    for i, rho in enumerate(rhos):
        d = g[g.rho == rho].sort_values("t")
        col = fs.BLUE_RAMP[i]
        ax.plot(d.t, d.I_rho_scaled, color=col, lw=1.4, zorder=3)
        ts = float(d.t_star.iloc[0])
        if ts > 0:
            y = np.interp(ts, d.t, d.I_rho_scaled)
            ax.plot([ts], [y], marker="|", ms=7, mew=1.2, color=col, zorder=4)
            dy = {0.3: -0.02, 0.6: -0.02, 0.85: -0.02, 0.95: -0.11}.get(rho, -0.02)
            ax.text(ts * 1.12, y + dy, f"ρ = {rho:g}", color=col, fontsize=7.5, va="top", ha="left")
        else:
            ax.text(0.024, 1.0 - 0.03, f"ρ = {rho:g}", color=col, fontsize=7.5, va="top", ha="left")
    ax.axhline(1.0, color=fs.MUTED, lw=0.6, ls=":", zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 6.5)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("diffusion time  $t$")
    ax.set_ylabel(r"$I_t(\rho)\,e^{4t}\,/\,\frac{1}{2}\|\partial_\rho\Sigma_0\|_F^2$")
    ax.set_xticks([0.03, 0.1, 0.3, 1, 3])
    ax.set_xticklabels(["0.03", "0.1", "0.3", "1", "3"])
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
    ax.text(0.16, 2.2e2, "ρ  (correlation, degree 2)", color=fs.CAT[0], fontsize=7.5, va="bottom")
    ax.text(0.16, 6.0, "β  (shape, degree 4)", color=fs.CAT[1], fontsize=7.5, va="bottom")
    ax.text(0.05, 6e-8, "solid: Laplace (β = 1)\ndashed: β = 3", fontsize=7, color=fs.INK2, va="bottom")
    fs.panel_label(ax, "(b)")

    # ---- (c) closure gap, measured / predicted
    ax = axes[2]
    fams = [("laplace", "Laplace", fs.CAT[0], "-", "o"), ("bimodal", "bimodal", fs.CAT[1], "--", "s"),
            ("uniform", "uniform", fs.CAT[2], "-.", "^"), ("student", "Student-$t_5$", fs.CAT[3], ":", "D")]
    for key, lab, col, ls, mk in fams:
        d = c[c.family == key]
        m = d.groupby("t").apply(lambda x: (x.sq_gap_per_chain / x.pred_sq_gap_per_chain).mean(), include_groups=False).reset_index(name="ratio")
        m = m[m.t >= 0.3]
        ax.plot(m.t, m.ratio, color=col, ls=ls, marker=mk, ms=3, lw=1.3, mec=col, mfc="white" if mk in ("s", "D") else col, zorder=3)
        offs = {"laplace": 1.35, "bimodal": 0.72, "uniform": 1.0, "student": 1.0}[key]
        ax.text(m.t.iloc[-1] * 1.08, m.ratio.iloc[-1] * offs, lab, color=col, fontsize=7.5, va="center", ha="left")
    ax.axhline(1.0, color=fs.MUTED, lw=0.6, ls=":", zorder=1)
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xlim(0.28, 9.0); ax.set_ylim(3e-3, 60)
    from matplotlib.ticker import NullFormatter
    ax.set_xticks([0.3, 1, 3]); ax.set_xticklabels(["0.3", "1", "3"]); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("diffusion time  $t$")
    ax.set_ylabel(r"$\mathbb{E}\|S_t - S_t^{G}\|^2\;/\;C_4 e^{-8t}$")
    fs.panel_label(ax, "(c)")

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_theory.pdf")
    fig.savefig(OUT / "fig_theory.png", dpi=220)
    print("wrote", OUT / "fig_theory.pdf")


if __name__ == "__main__":
    main()
