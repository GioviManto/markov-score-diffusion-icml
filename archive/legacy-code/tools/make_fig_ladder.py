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


def _repel(entries, min_sep_log=0.075):
    """Push right-margin label positions apart in log10(y) so no two overlap; returns the
    entries with a `y_lab` field. Every series keeps its own direct label (identity is never
    colour-alone); the caption names the arms."""
    entries = sorted(entries, key=lambda d: d["y"])
    ys = [np.log10(d["y"]) for d in entries]
    for _ in range(50):
        moved = False
        for i in range(1, len(ys)):
            if ys[i] - ys[i - 1] < min_sep_log:
                mid = 0.5 * (ys[i] + ys[i - 1]); ys[i - 1] = mid - min_sep_log / 2; ys[i] = mid + min_sep_log / 2; moved = True
        if not moved:
            break
    for d, y in zip(entries, ys):
        d["y_lab"] = 10 ** y
    return entries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--em", default="outputs/E02a/capacity.csv"); ap.add_argument("--net", default="outputs/E03/confirm.csv")
    ap.add_argument("--out", default="paper/figures/fig_ladder")
    args = ap.parse_args()
    if args.out.endswith(".pdf"):
        args.out = args.out[:-4]
    e = pd.read_csv(ROOT / args.em); e = e[e.region == "all"]
    fs.apply()
    fig, ax = plt.subplots(figsize=(fs.COL_W, 2.4))
    sizes = sorted(e.n_chains.unique())
    labels = []
    # oracles
    for arm, lab, col in (("isotropic", "isotropic score", fs.MUTED), ("lmmse_true", "LMMSE floor (second order)", fs.INK2)):
        m = e[e.arm == arm].groupby("n_chains").risk.mean()
        ax.plot(sizes, [m.loc[n] for n in sizes], color=col, ls=":", lw=1.2)
        labels.append(dict(y=float(m.loc[sizes[-1]]), text=lab, color=col))
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
        labels.append(dict(y=float(m.loc[sizes[-1]]), text=f"EM–BP, $C={C}$", color=col))
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
            labels.append(dict(y=float(m.loc[xs[-1]]), text=ARCH_NAME.get(arch, arch), color=col))
    for d in _repel(labels):
        ax.annotate(d["text"], xy=(sizes[-1], d["y"]), xytext=(sizes[-1] * 1.12, d["y_lab"]), color=d["color"], fontsize=7, va="center",
                    arrowprops=dict(arrowstyle="-", color=d["color"], lw=0.5, alpha=0.6, shrinkA=0, shrinkB=2) if abs(np.log10(d["y_lab"] / d["y"])) > 0.02 else None)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(sizes); ax.set_xticklabels([str(n) for n in sizes])
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(sizes[0] / 1.3, sizes[-1] * 3.2)
    ax.set_xlabel("sequences  $M$"); ax.set_ylabel("pooled relative score error")
    fs.save(fig, ROOT / (args.out + ".pdf"))
    print("wrote", args.out + ".pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
