#!/usr/bin/env python
"""Figure 4 — the ladder: what each estimator recovers, against the number of sequences.

All-site schedule-pooled relative score error against M for every arm on identical test
bundles: the isotropic score and the LMMSE floor (horizontal, from E02a oracle rows),
EM-BP at C in (1, 8) (E02a), the networks (E03 confirm winners), the archival window
head where its bundle hash matches, and the exact score at zero (not drawn). Mean over
seeds with light per-seed points. The message is read off the ordering with respect to
the floor: an arm above it has learned at most second-order structure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import figstyle as fs  # noqa: E402

ARCH_NAME = {"mlp": "MLP", "window": "window head", "tcn": "dilated TCN", "bigru": "biGRU", "transformer": "transformer", "unet": "U-Net"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--em", default="outputs/E02a/capacity.csv"); ap.add_argument("--net", default="outputs/E03/confirm.csv")
    ap.add_argument("--out", default="paper/figures/fig_ladder")
    args = ap.parse_args()
    e = pd.read_csv(ROOT / args.em); e = e[e.region == "all"]
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.COL_W, 2.4))
    sizes = sorted(e.n_chains.unique())
    # oracles
    for arm, lab, col in (("isotropic", "isotropic score", fs.MUTED), ("lmmse_true", "LMMSE floor (second order)", fs.INK2)):
        m = e[e.arm == arm].groupby("n_chains").risk.mean()
        ax.plot(sizes, [m.loc[n] for n in sizes], color=col, ls=":", lw=1.2)
        ax.text(sizes[-1] * 1.12, float(m.loc[sizes[-1]]), lab, color=col, fontsize=7, va="center")
    # EM-BP
    for C, col, ls, mk in ((1, fs.CAT[3], "-.", "^"), (8, fs.CAT[0], "-", "o")):
        d = e[(e.arm == "em_bp") & (e.components == C)]
        if len(d) == 0:
            continue
        m = d.groupby("n_chains").risk.mean()
        ax.plot(sizes, [m.loc[n] for n in sizes], color=col, ls=ls, marker=mk, ms=3.5, lw=1.4, zorder=4)
        for n in sizes:
            pts = d[d.n_chains == n].risk
            ax.plot([n] * len(pts), pts, ls="none", marker=".", ms=2.5, color=col, alpha=0.3, zorder=3)
        ax.text(sizes[-1] * 1.12, float(m.loc[sizes[-1]]), f"EM–BP, $C={C}$", color=col, fontsize=7, va="center")
    # networks
    net = ROOT / args.net
    if net.exists():
        c = pd.read_csv(net); c = c[c.region == "all"]
        styles = {"mlp": (fs.CAT[1], "--", "s"), "window": (fs.CAT[2], "-", "D"), "tcn": (fs.CAT[4], "--", "v"), "bigru": (fs.CAT[5], "-.", "<"), "transformer": (fs.CAT[6], ":", ">"), "unet": (fs.CAT[7], "--", "p")}
        for arch, (col, ls, mk) in styles.items():
            d = c[c.arch == arch]
            if len(d) == 0:
                continue
            m = d.groupby("n_chains").risk.mean()
            xs = [n for n in sizes if n in m.index]
            ax.plot(xs, [m.loc[n] for n in xs], color=col, ls=ls, marker=mk, ms=3, lw=1.2, mfc="white")
            ax.text(xs[-1] * 1.12, float(m.loc[xs[-1]]), ARCH_NAME.get(arch, arch), color=col, fontsize=7, va="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(sizes); ax.set_xticklabels([str(n) for n in sizes])
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(sizes[0] / 1.3, sizes[-1] * 3.2)
    ax.set_xlabel("sequences  $M$"); ax.set_ylabel("pooled relative score error")
    fig.savefig(ROOT / (args.out + ".pdf")); fig.savefig(ROOT / (args.out + ".png"), dpi=220)
    print("wrote", args.out + ".pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
