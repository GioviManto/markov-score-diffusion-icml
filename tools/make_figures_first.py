#!/usr/bin/env python
"""The consolidated note's figures-first plan (section 28): the figures it lists that did not exist yet, and the summary.

    python tools/make_figures_first.py

Every panel reads frozen outputs only:
  fig 3  ff-nongaussian-gain   per-level relative score error against the exact score: EM-BP, the Gaussian score
                               (LMMSE), the six networks, isotropic -- E28 (outputs/hpc_E28/per_level.csv), M = 128, 2048
  fig 4  ff-sample-efficiency  pooled risk against the number of training sequences M: (a) the Laplace chain, E03 networks
                               and E02a EM-BP (C = 8) with the LMMSE floor (the paper's Table "baselines"); (b) the phi^4
                               field at Theta = 0.3, h = 1, E33 kernels (risk selection) and networks, with E33's floor
  fig 7  ff-misspec-repair     E35: the first-order kernel and the same kernel on the augmented state against the linear
                               floor, over the latent strength b, at M = 512 and 2048 (two seeds; E35x adds sixteen)
  summary ff-summary           four panels: non-Gaussian gain per level, sample efficiency, the field scoreboard across
                               Theta, accuracy against inference cost across h
Figures 2, 5 and 6 are paper/figures/fig_gaussian_sanity.pdf, reports/figures/fig-field-sweep.pdf and
fig-field-runtime.pdf; figure 1 is the TikZ schematic reports/figures/ff-method.tex.
House rules: tools/figstyle (no prose in a figure, symbols in math mode, legends inside the axes).
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

OUT = ROOT / "reports/figures"
NETS = ["mlp", "window", "tcn", "bigru", "transformer", "unet"]
C_EM, C_GAUSS, C_NET, C_ISO = fs.CAT[0], fs.CAT[1], fs.MUTED, fs.INK2


def _plain(ax, x=None, y=None, fmt="{:g}"):
    """Explicit major ticks with plain labels and no minor labels (log axes otherwise print 6x10^-1 clutter)."""
    from matplotlib.ticker import FixedLocator, NullFormatter, FixedFormatter
    for axis, vals in ((ax.xaxis, x), (ax.yaxis, y)):
        if vals is None:
            continue
        axis.set_major_locator(FixedLocator(vals)); axis.set_major_formatter(FixedFormatter([fmt.format(v) for v in vals]))
        axis.set_minor_formatter(NullFormatter())


def _load(pattern):
    return pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(ROOT / pattern)))], ignore_index=True)


# --------------------------------------------------------------------------- data
def gain_table():
    d = pd.read_csv(ROOT / "outputs/hpc_E28/per_level.csv")
    d = d[d.region == "all"]
    return d.groupby(["n_chains", "arm", "t"]).risk_level.mean().reset_index()


def efficiency_laplace():
    c = pd.read_csv(ROOT / "outputs/E03/confirm.csv")
    c = c[c.region == "all"].groupby(["arch", "n_chains"]).risk.mean().reset_index().rename(columns={"arch": "arm"})
    e = pd.read_csv(ROOT / "outputs/E02a/capacity.csv")
    e = e[e.region == "all"]
    em = e[(e.arm == "em_bp") & (e.components == 8)].groupby("n_chains").risk.mean().reset_index().assign(arm="em_bp")
    orc = e[e.arm.isin(["lmmse_true", "isotropic"])].groupby(["arm", "n_chains"]).risk.mean().reset_index()
    return pd.concat([c, em, orc], ignore_index=True)


def efficiency_field(cell="T0.3_h1_L32"):
    k = pd.concat([_load("outputs/hpc_E33k_s*/kernels.csv"), _load("outputs/hpc_E33q_s*/kernels.csv")], ignore_index=True)
    k = k[(k.cell == cell) & (k.selection == "risk")].groupby(["arm", "n_chains"]).risk.mean().reset_index()
    n = _load("outputs/hpc_E33n_s*/nets.csv")
    n = n[n.cell == cell].assign(arm=lambda x: x.arm.str.replace("ism_one_view", "ism").str.replace("dsm_one_corruption", "dsm") + ":" + x.arch)
    n = n.groupby(["arm", "n_chains"]).risk.mean().reset_index()
    floor = _load("outputs/hpc_E33o/oracles.csv")
    floor = float(floor[(floor.cell == cell) & (floor.arm == "lmmse_true")].risk.mean())
    return pd.concat([k, n], ignore_index=True), floor


def scoreboard():
    k = pd.concat([_load("outputs/hpc_E33k_s*/kernels.csv"), _load("outputs/hpc_E33q_s*/kernels.csv")], ignore_index=True)
    k = k[(k.selection == "risk") & (k.n_chains == 2048)]
    n = _load("outputs/hpc_E33n_s*/nets.csv")
    n = n[n.n_chains == 2048].assign(arm=lambda x: x.arm.str.replace("ism_one_view", "ism").str.replace("dsm_one_corruption", "dsm") + ":" + x.arch)
    o = _load("outputs/hpc_E33o/oracles.csv")
    floor = o[o.arm == "lmmse_true"].groupby("cell").risk.mean()
    a = pd.concat([k[["cell", "T", "arm", "risk"]], n[["cell", "T", "arm", "risk"]]], ignore_index=True)
    a = a[a.cell.str.contains("iso") | (a.cell == "T0.3_h1_L32")]
    g = a.groupby(["cell", "T", "arm"]).risk.mean().reset_index()
    g["ratio"] = g.risk / g.cell.map(floor)
    return g


def misspec_table():
    d = pd.read_csv(ROOT / "outputs/E35/augmented.csv")
    return d.groupby(["b", "n_chains", "arm"]).risk.mean().reset_index()


# --------------------------------------------------------------------------- panels
def panel_gain(ax, g, n_chains, legend=True):
    s = g[g.n_chains == n_chains]
    for a in NETS:
        r = s[s.arm == f"net:{a}"].sort_values("t")
        ax.plot(r.t, r.risk_level, color=C_NET, lw=0.6, alpha=0.7)
    best = s[s.arm.str.startswith("net:")].groupby("t").risk_level.min()
    ax.plot(best.index, best.values, color=fs.INK2, lw=1.2, label="best network")
    for arm, c, ls, lab in (("lmmse", C_GAUSS, "--", "Gaussian score"), ("em_bp", C_EM, "-", "EM--BP"), ("isotropic", C_ISO, ":", "isotropic")):
        r = s[s.arm == arm].sort_values("t")
        ax.plot(r.t, r.risk_level, color=c, lw=1.3 if arm == "em_bp" else 1.0, ls=ls, label=lab.replace("--", "–"))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"diffusion time $t$"); fs.ygrid(ax)
    if legend:
        ax.legend(loc="lower left", fontsize=6.3, handlelength=1.8, labelspacing=0.25, borderpad=0.3, frameon=False)


def panel_eff_laplace(ax, e, legend=True):
    for a in NETS:
        r = e[e.arm == a].sort_values("n_chains")
        ax.plot(r.n_chains, r.risk, color=C_NET, lw=0.7, marker="o", ms=2.2, alpha=0.8)
    best = e[e.arm.isin(NETS)].groupby("n_chains").risk.min()
    ax.plot(best.index, best.values, color=fs.INK2, lw=1.2, label="best network")
    r = e[e.arm == "em_bp"].sort_values("n_chains")
    ax.plot(r.n_chains, r.risk, color=C_EM, lw=1.4, marker="o", ms=3, label="EM–BP")
    for arm, c, ls, lab in (("lmmse_true", C_GAUSS, "--", "LMMSE floor"), ("isotropic", C_ISO, ":", "isotropic")):
        ax.axhline(float(e[e.arm == arm].risk.mean()), color=c, lw=0.9, ls=ls, label=lab)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel(r"training sequences $M$"); fs.ygrid(ax)
    _plain(ax, x=[32, 128, 512, 2048], y=[0.02, 0.05, 0.1, 0.2, 0.5])
    if legend:
        ax.legend(loc="lower left", fontsize=6.3, handlelength=1.8, labelspacing=0.25, borderpad=0.3, frameon=False, ncol=2, columnspacing=0.8)


FIELD_ARMS = [("em_pair_quartic", "site energy", fs.CAT[0], "o"), ("em_tilted_spline", "spline tilt", fs.CAT[2], "s"),
              ("em_mdn", "MDN", fs.CAT[6], "^"), ("em_mixture", "mixture", fs.CAT[1], "D")]
FIELD_NETS = [("dsm:window", "window", fs.CAT[3]), ("dsm:transformer", "transformer", fs.CAT[7])]


def panel_eff_field(ax, f, floor, legend=True):
    for arm, lab, c, mk in FIELD_ARMS:
        r = f[f.arm == arm].sort_values("n_chains")
        ax.plot(r.n_chains, r.risk, color=c, lw=1.1, marker=mk, ms=3, label=lab)
    for arm, lab, c in FIELD_NETS:
        r = f[f.arm == arm].sort_values("n_chains")
        ax.plot(r.n_chains, r.risk, color=c, lw=0.9, ls="-.", marker="x", ms=3.5, label=lab)
    ax.axhline(floor, color=C_GAUSS, lw=0.9, ls="--")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel(r"training sequences $M$"); fs.ygrid(ax)
    _plain(ax, x=[32, 128, 512, 2048], y=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    if legend:
        ax.legend(loc="lower left", fontsize=6.0, handlelength=1.6, labelspacing=0.2, borderpad=0.3, frameon=False, ncol=2, columnspacing=0.8)


def panel_scoreboard(ax, g, legend=True):
    for arm, lab, c, mk in FIELD_ARMS:
        r = g[g.arm == arm].sort_values("T")
        ax.plot(r["T"], r.ratio, color=c, lw=1.1, marker=mk, ms=3, label=lab)
    for arm, lab, c in FIELD_NETS:
        r = g[g.arm == arm].sort_values("T")
        ax.plot(r["T"], r.ratio, color=c, lw=0.9, ls="-.", marker="x", ms=3.5, label=lab)
    ax.axhline(1.0, color=C_GAUSS, lw=0.9, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"temperature $\Theta$")
    ax.set_xticks([0.3, 0.5, 1, 2]); ax.set_xticklabels(["0.3", "0.5", "1", "2"]); fs.ygrid(ax)
    if legend:
        ax.legend(loc="lower right", fontsize=6.0, handlelength=1.6, labelspacing=0.2, borderpad=0.3, frameon=False, ncol=2, columnspacing=0.8)


def panel_misspec(ax, m, n_chains, legend=True):
    s = m[m.n_chains == n_chains]
    # linear axis: at b = 0 the chain is Gaussian and the linear floor IS the exact score (risk ~1e-14)
    for arm, lab, c, ls, mk in (("em_first_order", "first order", fs.CAT[1], "-", "s"), ("em_augmented", "augmented", fs.CAT[0], "-", "o"),
                                ("lmmse_true", "linear floor", fs.INK2, "--", None)):
        r = s[s.arm == arm].sort_values("b")
        ax.plot(r.b, r.risk, color=c, ls=ls, lw=1.2, marker=mk, ms=3.2, label=lab)
    ax.set_ylim(0, None); ax.set_xlabel(r"latent strength $b$"); ax.set_xticks([0, 1, 2]); fs.ygrid(ax)
    if legend:
        ax.legend(loc="upper left", fontsize=6.3, handlelength=1.8, labelspacing=0.25, borderpad=0.3, frameon=False)


# --------------------------------------------------------------------------- figures
def fig_gain(g):
    fig, axes = plt.subplots(1, 2, figsize=(fs.FULL_W, 2.4), sharey=True, gridspec_kw=dict(wspace=0.08))
    panel_gain(axes[0], g, 128, legend=True); panel_gain(axes[1], g, 2048, legend=False)
    axes[0].set_ylabel(r"$\|\hat S_t-S_t\|/\|S_t\|$")
    for ax, n, lab in ((axes[0], 128, "(a)"), (axes[1], 2048, "(b)")):
        fs.panel_label(ax, lab); ax.set_title(rf"$M={n}$", fontsize=8)
    fs.save(fig, OUT / "ff-nongaussian-gain.pdf")


def fig_efficiency(e, f, floor):
    fig, axes = plt.subplots(1, 2, figsize=(fs.FULL_W, 2.45), gridspec_kw=dict(wspace=0.25))
    panel_eff_laplace(axes[0], e); panel_eff_field(axes[1], f, floor)
    axes[0].set_ylabel(r"pooled relative score error")
    fs.panel_label(axes[0], "(a)"); fs.panel_label(axes[1], "(b)")
    fs.save(fig, OUT / "ff-sample-efficiency.pdf")


def fig_misspec(m):
    fig, axes = plt.subplots(1, 2, figsize=(fs.FULL_W, 2.3), sharey=True, gridspec_kw=dict(wspace=0.08))
    panel_misspec(axes[0], m, 512, legend=True); panel_misspec(axes[1], m, 2048, legend=False)
    axes[0].set_ylabel(r"pooled relative score error")
    for ax, n, lab in ((axes[0], 512, "(a)"), (axes[1], 2048, "(b)")):
        fs.panel_label(ax, lab); ax.set_title(rf"$M={n}$", fontsize=8)
    fs.save(fig, OUT / "ff-misspec-repair.pdf")


def fig_summary(g, e, sb):
    """Four panels, one colour per model class across all of them: blue the paper's mixture kernel (EM--BP on the
    Laplace chain, `em_mixture` on the field), green the site energy, yellow the networks, dark dashed the Gaussian
    score / linear floor."""
    import make_field_runtime_fig as rt
    T = rt.table()
    MIX, SITE, NET, FLOOR = fs.CAT[0], fs.CAT[2], fs.CAT[3], fs.INK2
    fig, axes = plt.subplots(1, 4, figsize=(fs.FULL_W, 2.25), gridspec_kw=dict(wspace=0.45))
    fig.subplots_adjust(bottom=0.30)       # the key sits inside the canvas: outside it the PDF clipped it (figure 6)
    ax = axes[0]
    s2 = g[g.n_chains == 2048]
    for arm, c, ls in (("em_bp", MIX, "-"), ("lmmse", FLOOR, "--")):
        r = s2[s2.arm == arm].sort_values("t"); ax.plot(r.t, r.risk_level, color=c, ls=ls, lw=1.3)
    best = s2[s2.arm.str.startswith("net:")].groupby("t").risk_level.min(); ax.plot(best.index, best.values, color=NET, lw=1.3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"diffusion time $t$"); ax.set_ylabel(r"$\|\hat S-S\|/\|S\|$")
    _plain(ax, x=[0.1, 1], y=[1e-4, 1e-3, 1e-2, 1e-1], fmt="{:g}")
    ax = axes[1]
    r = e[e.arm == "em_bp"].sort_values("n_chains"); ax.plot(r.n_chains, r.risk, color=MIX, lw=1.3, marker="o", ms=2.6)
    b = e[e.arm.isin(NETS)].groupby("n_chains").risk.min(); ax.plot(b.index, b.values, color=NET, lw=1.3, marker="o", ms=2.6)
    ax.axhline(float(e[e.arm == "lmmse_true"].risk.mean()), color=FLOOR, ls="--", lw=1.0)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel(r"sequences $M$"); ax.set_ylabel(r"pooled risk")
    _plain(ax, x=[32, 512], y=[0.02, 0.05, 0.1, 0.2])
    ax = axes[2]
    for arm, c in (("em_pair_quartic", SITE), ("em_mixture", MIX), ("dsm:window", NET)):
        r = sb[sb.arm == arm].sort_values("T"); ax.plot(r["T"], r.ratio, color=c, lw=1.3, marker="o", ms=2.6)
    ax.axhline(1.0, color=FLOOR, ls="--", lw=1.0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"temperature $\Theta$"); ax.set_ylabel(r"risk / linear floor")
    _plain(ax, x=[0.3, 1, 2], y=[0.03, 0.1, 0.3, 1, 3])
    ax = axes[3]
    for arm, c in (("em_pair_quartic", SITE), ("em_mixture", MIX), ("dsm:window", NET)):
        r = T[T.arm == arm].sort_values("L"); ax.plot(r.infer_ms_per_seq, r.risk_over_floor, color=c, lw=1.3, marker="o", ms=2.6)
    ax.axhline(1.0, color=FLOOR, ls="--", lw=1.0)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"ms per sequence"); ax.set_ylabel(r"risk / linear floor")
    _plain(ax, x=[0.01, 0.1, 1], y=[0.03, 0.1, 0.3, 1])
    for a, lab in zip(axes, "abcd"):
        fs.ygrid(a); fs.panel_label(a, f"({lab})", x=-0.34)
        a.tick_params(labelsize=6.5); a.xaxis.label.set_size(7); a.yaxis.label.set_size(7)
    handles = [plt.Line2D([], [], color=MIX, lw=1.4), plt.Line2D([], [], color=SITE, lw=1.4),
               plt.Line2D([], [], color=NET, lw=1.4), plt.Line2D([], [], color=FLOOR, lw=1.0, ls="--")]
    fig.legend(handles, ["mixture kernel (EM–BP)", "site energy", "best network / window", "Gaussian score / linear floor"],
               loc="lower center", ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.0), handlelength=2.0, columnspacing=1.4)
    fs.save(fig, OUT / "ff-summary.pdf")


def main() -> int:
    fs.apply()
    g, e = gain_table(), efficiency_laplace()
    f, floor = efficiency_field()
    sb, m = scoreboard(), misspec_table()
    fig_gain(g); fig_efficiency(e, f, floor); fig_misspec(m); fig_summary(g, e, sb)
    with pd.option_context("display.width", 200):
        print(g[(g.n_chains == 2048) & g.arm.isin(["em_bp", "lmmse"])].pivot_table(index="t", columns="arm", values="risk_level").round(4))
        print(e.pivot_table(index="n_chains", columns="arm", values="risk").round(3))
        print(m.pivot_table(index=["n_chains", "b"], columns="arm", values="risk").round(4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
