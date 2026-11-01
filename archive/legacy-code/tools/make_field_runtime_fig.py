#!/usr/bin/env python
"""The consolidated note's figure 6: refinement and runtime -- accuracy against cost across the lattice spacing h.

Accuracy: E33's risk over the linear floor at M = 2048 on the three refinement cells (h = 1, 0.5, 0.25; L = 32, 64,
128), sixteen seeds (hpc_E33k_s*, hpc_E33q_s*, hpc_E33n_s*, hpc_E33o). Inference cost: E24x's exclusive-node sweep in
the number of sites (outputs/hpc_E24x/scaling.csv, factor n_sites, B = 512, N_g = 401, cnode07, 8 BLAS threads), per
sequence; every BP-based arm shares one cost (grid BP's cost depends on L, N_g and B, not on the kernel). The network
latencies are E24's LARGEST screened architectures, upper bounds for the smaller ones E33's screen selected. Training
cost: the wall clock E33 recorded per fit at M = 2048 (shared nodes; indicative).

Writes reports/figures/fig-field-runtime.pdf and reports/numbers/field_runtime.csv.
House rules: tools/figstyle (no prose in a figure, every symbol in math mode, fs.save).
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import figstyle as fs  # noqa: E402

CELLS = [("T0.3_h1_L32", 32, r"$h{=}1$"), ("T0.3_h0.5_L64", 64, r"$h{=}0.5$"), ("T0.3_h0.25_L128", 128, r"$h{=}0.25$")]
ARMS = [("em_pair_quartic", "bp_grid", "site energy", fs.CAT[0]), ("em_tilted_spline", "bp_grid", "spline tilt", fs.CAT[2]),
        ("em_mdn", "bp_grid", "MDN", fs.CAT[6]), ("em_mixture", "bp_grid", "mixture", fs.CAT[1]),
        ("dsm:window", "window", "window", fs.CAT[3]), ("dsm:transformer", "transformer", "transformer", fs.CAT[7])]


def _load(pattern):
    return pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(ROOT / pattern)))], ignore_index=True)


def table():
    K = pd.concat([_load("outputs/hpc_E33k_s*/kernels.csv"), _load("outputs/hpc_E33q_s*/kernels.csv")], ignore_index=True)
    K = K[(K.selection == "risk") & (K.n_chains == 2048)][["cell", "arm", "risk", "wall"]]
    N = _load("outputs/hpc_E33n_s*/nets.csv")
    N = N[N.n_chains == 2048].assign(arm=lambda d: d.arm.str.replace("ism_one_view", "ism").str.replace("dsm_one_corruption", "dsm") + ":" + d.arch)
    A = pd.concat([K, N[["cell", "arm", "risk", "wall"]]], ignore_index=True)
    floor = _load("outputs/hpc_E33o/oracles.csv").query("arm == 'lmmse_true'").groupby("cell").risk.mean()
    S = pd.read_csv(ROOT / "outputs/hpc_E24x/scaling.csv")
    S = S[S.factor == "n_sites"]
    lat = (S.groupby(["arm", "value"]).wall_median.median() / S.batch.iloc[0] * 1e3)   # ms per sequence
    rows = []
    for cell, L, _ in CELLS:
        for arm, cost_arm, lab, _ in ARMS:
            g = A[(A.cell == cell) & (A.arm == arm)]
            rows.append(dict(cell=cell, L=L, arm=arm, label=lab, risk_over_floor=float(g.risk.mean() / floor[cell]),
                             infer_ms_per_seq=float(lat[(cost_arm, L)]), train_sec=float(g.wall.mean()), n=int(g.risk.notna().sum())))
    return pd.DataFrame(rows)


def figure(T, out):
    fig, axes = plt.subplots(1, 2, figsize=(fs.FULL_W, 2.45), gridspec_kw=dict(wspace=0.30))
    for ax, col, xl in ((axes[0], "infer_ms_per_seq", r"inference, ms per sequence"), (axes[1], "train_sec", r"training, s per fit")):
        for arm, _, lab, c in ARMS:
            g = T[T.arm == arm].sort_values("L")
            ax.plot(g[col], g.risk_over_floor, color=c, lw=0.9, alpha=0.8, label=lab)
            for (_, r), mk in zip(g.iterrows(), fs.MARKERS):
                ax.scatter(r[col], r.risk_over_floor, color=c, marker=mk, s=16, zorder=3, edgecolor="white", linewidth=0.4)
        ax.axhline(1.0, color=fs.INK2, lw=0.8, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(xl); fs.ygrid(ax)
    axes[0].set_ylabel(r"risk / linear floor")
    fs.panel_label(axes[0], "(a)"); fs.panel_label(axes[1], "(b)")
    handles = [plt.Line2D([], [], color=fs.INK2, marker=mk, ls="none", ms=4, label=lab) for (_, _, lab), mk in zip(CELLS, fs.MARKERS)]
    # both legends inside the axes, in regions no point reaches: outside the axes the PDF clipped them
    axes[0].legend(loc="lower left", fontsize=6.2, handlelength=1.6, labelspacing=0.25, borderpad=0.3, ncol=2, columnspacing=0.9)
    axes[1].legend(handles=handles, loc="lower right", fontsize=6.2, handlelength=1.0, labelspacing=0.25, borderpad=0.3)
    fs.save(fig, out)


def main() -> int:
    fs.apply()
    T = table()
    (ROOT / "reports/numbers").mkdir(parents=True, exist_ok=True)
    T.to_csv(ROOT / "reports/numbers/field_runtime.csv", index=False, float_format="%.5g")
    with pd.option_context("display.width", 200):
        print(T.round(4).to_string(index=False))
    figure(T, ROOT / "reports/figures/fig-field-runtime.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
