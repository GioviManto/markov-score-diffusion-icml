"""The temperature sweep of the phi^4 field on both axes: fixed spacing and matched correlation.

The consolidated note's figure 5 ("SPDE temperature sweep": domains, kink density, correlation
length and score error across Theta). Two axes are drawn, because they answer different
questions: at FIXED spacing h = 1 (E32 amendment 1, outputs/E32/fixed_h.csv) lowering Theta
lengthens the domains and sharpens the bimodality together, which is the physics reading; on
the ISO-CORRELATION line (the registered cells, outputs/E32/ladder.csv and outputs/E34/physics.csv)
the correlation length in sites is held at the AR(1) laboratory's value, which isolates the
non-Gaussian part.

fig-field-sweep.pdf, four panels against Theta (log axis):
    (a) correlation length in sites, from the spectrum (exact);
    (b) coarse mean domain length in physical units (runs shorter than the kink width merged);
    (c) coarse kink density per unit length;
    (d) the pooled relative score error of the Gaussian closure (the linear floor) and of the
        isotropic map, i.e. what structure is worth.
Fixed spacing: filled circles, solid; iso-correlation: open squares, dashed. Two reps per cell,
both drawn where they differ visibly. House rules: tools/figstyle.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

ISO = ["T0.3_h1_L32", "T0.5_iso_L32", "T1_iso_L32", "T2_iso_L32"]


def load(fixed_h="outputs/E32/fixed_h.csv"):
    fh = pd.read_csv(ROOT / fixed_h)
    lad = pd.read_csv(ROOT / "outputs/E32/ladder.csv")
    ph = pd.read_csv(ROOT / "outputs/E34/physics.csv")
    # fixed spacing: one row per (cell, rep) with the physics, and the pooled floor from the ladder rows
    phys = fh[fh.kind == "physics"].copy()
    pooled = fh[(fh.kind == "ladder") & (fh.level < 0)].pivot_table(index=["cell", "rep"], columns="arm", values="risk").reset_index()
    fixed = phys.merge(pooled, on=["cell", "rep"], suffixes=("", "_risk"))
    # iso line: E34's exact-sample physics and E32's pooled ladder
    iph = ph[(ph.source == "exact_samples") & ph.cell.isin(ISO)].copy()
    ipool = lad[(lad.level < 0) & lad.cell.isin(ISO)].pivot_table(index=["cell", "rep"], columns="arm", values="risk").reset_index()
    iso = iph.merge(ipool, on=["cell", "rep"], suffixes=("", "_risk"))
    return fixed.sort_values(["T", "rep"]), iso.sort_values(["T", "rep"])


def _curve(ax, df, col, color, filled, label=None, ls="-", marker="o"):
    g = df.groupby("T")[col]
    m = g.mean()
    ax.plot(m.index, m.values, color=color, ls=ls, lw=1.1, marker=marker, ms=3.4,
            mfc=color if filled else "white", mec=color, label=label)
    ax.scatter(df["T"], df[col], color=color, s=5, alpha=0.5, zorder=3)


def figure(fixed, iso, out):
    fig, axes = plt.subplots(2, 2, figsize=(fs.FULL_W, 3.7), gridspec_kw=dict(wspace=0.32, hspace=0.42))
    (ax, bx), (cx, dx) = axes
    c_fix, c_iso = fs.CAT[0], fs.CAT[1]
    _curve(ax, fixed, "corr_length_sites", c_fix, True, label="$h{=}1$")
    _curve(ax, iso, "corr_length_sites", c_iso, False, label=r"$\lambda_1/\lambda_0{=}0.85$", ls="--", marker="s")
    ax.set_ylabel(r"$\xi$ (sites)"); ax.set_yscale("log"); fs.panel_label(ax, "(a)")
    ax.legend(fontsize=6.2, loc="upper right", handlelength=1.8, labelspacing=0.25)
    _curve(bx, fixed, "coarse_mean_domain_physical", c_fix, True)
    _curve(bx, iso, "coarse_mean_domain_physical", c_iso, False, ls="--", marker="s")
    bx.set_ylabel(r"$\bar\ell_{\rm dom}$ (physical)"); bx.set_yscale("log"); fs.panel_label(bx, "(b)")
    _curve(cx, fixed, "coarse_kink_density_physical", c_fix, True)
    _curve(cx, iso, "coarse_kink_density_physical", c_iso, False, ls="--", marker="s")
    cx.set_ylabel(r"$\rho_{\rm kink}$ (per unit length)"); cx.set_yscale("log"); fs.panel_label(cx, "(c)")
    _curve(dx, fixed, "lmmse_true", c_fix, True, label="closure, $h{=}1$")
    _curve(dx, iso, "lmmse_true", c_iso, False, label="closure, iso", ls="--", marker="s")
    _curve(dx, fixed, "isotropic", fs.MUTED, True, label="isotropic, $h{=}1$", marker="^")
    _curve(dx, iso, "isotropic", fs.MUTED, False, label="isotropic, iso", ls="--", marker="v")
    dx.set_ylabel("pooled relative score error"); dx.set_yscale("log"); fs.panel_label(dx, "(d)")
    dx.legend(fontsize=5.8, loc="lower left", ncol=2, handlelength=1.8, labelspacing=0.2, columnspacing=0.8)
    for a in (ax, bx, cx, dx):
        a.set_xscale("log"); a.set_xlabel(r"$\Theta$")
        a.set_xticks([0.2, 0.3, 0.5, 1, 2]); a.set_xticklabels(["0.2", "0.3", "0.5", "1", "2"]); a.minorticks_off()
        fs.ygrid(a)
    fs.save(fig, out)


def table(fixed, iso, out):
    rows = []
    for name, df in (("fixed_h", fixed), ("iso", iso)):
        g = df.groupby("T").agg(xi=("corr_length_sites", "mean"), domain=("coarse_mean_domain_physical", "mean"),
                                kinks=("coarse_kink_density_physical", "mean"), bimodality=("bimodality_coefficient", "mean"),
                                kurtosis=("excess_kurtosis", "mean"), floor=("lmmse_true", "mean"), isotropic=("isotropic", "mean"),
                                n=("rep", "count")).reset_index()
        g.insert(0, "axis", name); rows.append(g)
    t = pd.concat(rows, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False, float_format="%.4f")
    return t


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fixed-h", default="outputs/E32/fixed_h.csv")
    ap.add_argument("--fig", default="reports/figures/fig-field-sweep.pdf")
    ap.add_argument("--table", default="reports/numbers/field_sweep.csv")
    args = ap.parse_args()
    fs.apply()
    fixed, iso = load(args.fixed_h)
    t = table(fixed, iso, ROOT / args.table)
    with pd.option_context("display.width", 200):
        print(t.to_string(index=False))
    figure(fixed, iso, ROOT / args.fig)
    print("wrote", args.fig, "and", args.table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
