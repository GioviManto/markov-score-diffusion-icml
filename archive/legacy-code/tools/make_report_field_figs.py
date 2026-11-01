"""Figures for reports/field-laboratory-2026-10-01.tex from the frozen E32 outputs.

fig-field-spectrum   (a) lambda_1/lambda_0 against T for each spacing, with the iso-correlation
                     line at the laboratory's 0.85; (b) marginal excess kurtosis against T.
fig-field-chain      (a) the standardised stationary law at the four temperature cells;
                     (b) the arrival-point tilt log g at the base cell against the Gaussian
                     embedding's quadratic; (c) one exact sample at the base cell.
fig-field-ladder     (a) the pooled linear floor and the isotropic map across the six cells;
                     (b) the floor per level at the four temperature cells.
fig-field-theory     the measured closure gap over Theorem 2's prediction against t, four cells.

Reads outputs/E32/{spectrum,iso,ladder,theory}.csv and outputs/E32/cells.json; the chain
itself is rebuilt from markov_score.field for the densities. Writes reports/figures/.
House rules: tools/figstyle (no prose in a figure, every symbol in math mode, fs.save).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import figstyle as fs  # noqa: E402
from markov_score.field import Phi4Chain  # noqa: E402
from markov_score.utils import rng_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
E32 = ROOT / "outputs/E32"
OUT = ROOT / "reports/figures"
TCELLS = ["T0.3_h1_L32", "T0.5_iso_L32", "T1_iso_L32", "T2_iso_L32"]
REFINE = ["T0.3_h1_L32", "T0.3_h0.5_L64", "T0.3_h0.25_L128"]
SHORT = {"T0.3_h1_L32": r"$\Theta{=}0.3$", "T0.5_iso_L32": r"$\Theta{=}0.5$", "T1_iso_L32": r"$\Theta{=}1$", "T2_iso_L32": r"$\Theta{=}2$",
         "T0.3_h0.5_L64": "$h{=}0.5$", "T0.3_h0.25_L128": "$h{=}0.25$"}


def spectrum():
    sp = pd.read_csv(E32 / "spectrum.csv"); iso = pd.read_csv(E32 / "iso.csv")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(fs.FULL_W, 2.2), gridspec_kw=dict(wspace=0.32))
    hs = sorted(sp.h.unique())
    for j, h in enumerate(hs):
        s = sp[sp.h == h].sort_values("T")
        c = fs.BLUE_RAMP[j % len(fs.BLUE_RAMP)]
        ax.plot(s["T"], s.rho1, color=c, marker="o", ms=2.5, lw=1.1, label=f"$h={h:g}$")
        bx.plot(s["T"], s.excess_kurtosis, color=c, marker="o", ms=2.5, lw=1.1, label=f"$h={h:g}$")
    ax.axhline(0.85, color=fs.INK2, lw=0.8, ls="--")
    ax.plot(iso["T"], iso.rho1, color=fs.CAT[1], marker="s", ms=3.2, lw=0, label=r"$h^*(\Theta)$")
    ax.set_xscale("log"); ax.set_xlabel(r"$\Theta$"); ax.set_ylabel(r"$\lambda_1/\lambda_0$")
    ax.set_xticks([0.2, 0.3, 0.5, 1, 2]); ax.set_xticklabels(["0.2", "0.3", "0.5", "1", "2"]); ax.minorticks_off()
    ax.set_ylim(0.3, 1.0); fs.ygrid(ax); fs.panel_label(ax, "(a)")
    ax.legend(fontsize=6.2, ncol=2, loc="lower left", handlelength=1.6, columnspacing=0.8, labelspacing=0.25)
    bx.plot(iso["T"], iso.excess_kurtosis, color=fs.CAT[1], marker="s", ms=3.2, lw=0, label=r"$h^*(\Theta)$")
    bx.set_xscale("log"); bx.set_xlabel(r"$\Theta$"); bx.set_ylabel(r"$\kappa_4$ of $\pi$")
    bx.set_xticks([0.2, 0.3, 0.5, 1, 2]); bx.set_xticklabels(["0.2", "0.3", "0.5", "1", "2"]); bx.minorticks_off()
    fs.ygrid(bx); fs.panel_label(bx, "(b)")
    fs.save(fig, OUT / "fig-field-spectrum.pdf")


def chain():
    cells = {c["cell"]: c for c in json.load(open(E32 / "cells.json"))}
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(fs.FULL_W, 2.0), gridspec_kw=dict(wspace=0.38))
    g = np.linspace(-3.2, 3.2, 801)
    for j, cid in enumerate(TCELLS):
        ch = Phi4Chain(cells[cid]["T"], cells[cid]["h"])
        ax.plot(g, np.exp(ch.log_stationary(g)), color=fs.BLUE_RAMP[4 - j], lw=1.2, label=SHORT[cid])
    ax.set_xlabel("$a$"); ax.set_ylabel(r"$\pi(a)$"); fs.ygrid(ax); fs.panel_label(ax, "(a)")
    # the trough between the wells is the one region every curve leaves empty
    ax.legend(fontsize=6.0, loc="lower center", bbox_to_anchor=(0.5, 0.0), handlelength=1.2, labelspacing=0.15, ncol=1, borderpad=0.2)
    base = Phi4Chain(cells["T0.3_h1_L32"]["T"], cells["T0.3_h1_L32"]["h"])
    lt = base.log_tilt(g); lt = lt - lt.max()
    bx.plot(g, lt, color=fs.CAT[0], lw=1.3, label=r"$\log g$")
    rho, q = base.rho, 1 - base.rho**2
    quad = -(1 - rho) * g**2 / (2 * q); quad = quad - quad.max()
    bx.plot(g, quad, color=fs.INK2, lw=0.9, ls="--", label="Gaussian twin")
    bx.set_ylim(-12, 1); bx.set_xlabel("$a'$"); bx.set_ylabel(r"$\log g(a')$"); fs.ygrid(bx); fs.panel_label(bx, "(b)")
    bx.legend(fontsize=6.4, loc="lower center", handlelength=1.6, labelspacing=0.25)
    A = base.sample_batch(rng_for("fig-field-chain"), 2, 128)
    cx.plot(np.arange(128), A[0], color=fs.CAT[0], lw=1.0)
    cx.axhline(1 / base.scale, color=fs.MUTED, lw=0.6, ls=":"); cx.axhline(-1 / base.scale, color=fs.MUTED, lw=0.6, ls=":")
    cx.set_xlabel("$i$"); cx.set_ylabel("$a_i$"); fs.panel_label(cx, "(c)")
    fs.save(fig, OUT / "fig-field-chain.pdf")


def ladder():
    lad = pd.read_csv(E32 / "ladder.csv")
    order = TCELLS + REFINE[1:]
    pooled = lad[lad.level < 0].groupby(["cell", "arm"]).risk.mean().unstack().reindex(order)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(fs.FULL_W, 2.2), gridspec_kw=dict(wspace=0.32, width_ratios=[1.1, 1]))
    x = np.arange(len(order))
    ax.plot(x, pooled["isotropic"], color=fs.MUTED, marker="s", ms=3.2, lw=1.0, label="isotropic")
    ax.plot(x, pooled["lmmse_true"], color=fs.CAT[0], marker="o", ms=3.4, lw=1.2, label="linear floor")
    ax.axhline(0.168, color=fs.CAT[1], lw=0.8, ls="--")
    fs.halo(ax.text(len(order) - 0.55, 0.168 + 0.02, r"Laplace, $\rho{=}0.85$", fontsize=6.2, color=fs.CAT[1], ha="right", va="bottom"))
    ax.axhline(0.082, color=fs.CAT[2], lw=0.8, ls="--")
    fs.halo(ax.text(len(order) - 0.55, 0.082 - 0.02, "bimodal, E14b", fontsize=6.2, color=fs.CAT[2], ha="right", va="top"))
    ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in order], fontsize=6.6)
    ax.set_ylabel("pooled relative score error"); ax.set_ylim(0, 0.8); fs.ygrid(ax); fs.panel_label(ax, "(a)")
    ax.legend(fontsize=6.4, loc="upper left", handlelength=1.6, labelspacing=0.25)
    for j, cid in enumerate(TCELLS):
        s = lad[(lad.cell == cid) & (lad.arm == "lmmse_true") & (lad.level > 0)].groupby("level").risk.mean()
        bx.plot(s.index, s.values, color=fs.BLUE_RAMP[4 - j], marker="o", ms=2.4, lw=1.1, label=SHORT[cid])
    bx.set_xscale("log"); bx.set_yscale("log"); bx.set_xlabel("$t$"); bx.set_ylabel("floor per level")
    fs.ygrid(bx); fs.panel_label(bx, "(b)")
    bx.legend(fontsize=6.4, loc="lower left", handlelength=1.6, labelspacing=0.25)
    fs.save(fig, OUT / "fig-field-ladder.pdf")


def theory():
    th = pd.read_csv(E32 / "theory.csv")
    fig, ax = plt.subplots(figsize=(fs.COL_W, 2.1))
    for j, cid in enumerate(TCELLS):
        s = th[th.cell == cid].sort_values("level")
        ax.plot(s.level, s.ratio, color=fs.BLUE_RAMP[4 - j], marker="o", ms=2.6, lw=1.1, label=SHORT[cid])
        if j == 0:
            ax.axvline(float(s.t_star.iloc[0]), color=fs.INK2, lw=0.7, ls=":")
            ax.text(float(s.t_star.iloc[0]) * 1.05, 0.03, "$t_*$", fontsize=7, color=fs.INK2)
    ax.axhline(1.0, color=fs.INK2, lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(5e-4, 2)
    ax.set_xlabel("$t$"); ax.set_ylabel("measured / predicted gap"); fs.ygrid(ax)
    ax.legend(fontsize=6.4, loc="lower right", handlelength=1.6, labelspacing=0.25)
    fs.save(fig, OUT / "fig-field-theory.pdf")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fs.apply()
    spectrum(); chain(); ladder(); theory()
    print("wrote", sorted(p.name for p in OUT.glob("fig-field-*.pdf")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
