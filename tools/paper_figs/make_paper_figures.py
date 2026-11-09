#!/usr/bin/env python3
"""Figures for the hand-written paper, in the style Jérôme asked for on 29 Oct 2026.

    fig_locality         |Q_t| of the Gaussian chain at five noise levels, t* (densest precision) marked
    fig_locality_curve   coupling outside the band of Q_t against t, with t*
    fig_laplace          the Laplace chain: law of the increment, and the score at two noise levels
    fig_factor_graph     the posterior's factor graph: a chain, hence a tree (TikZ, compiled by tectonic)
    fig_em_loop          EM through BP: learn the kernel, then use it (TikZ, compiled by tectonic)
    fig_grid_gaussian    grid BP against the exact score on the Gaussian chain
    fig_closure          Gaussian closure error against the grid's own error, Laplace chain
    fig_em_innovation    what EM recovers: the learned increment law and its moments against N (thesis Fig. 6.3)
    fig_value_of_structure  EM-BP, window CNN and global MLP on one plot (thesis Fig. 8.1a-b merged)
    ... and the restructured paper's figures (8 Nov 2026), in figs_v2.py: fig_factor_graph, fig_em_loop, fig_routes,
    fig_closure (2,000 sequences), fig_results, fig_steps, fig_wall, fig_networks, fig_em_trace, fig_nonmarkov_delta,
    fig_nonmarkov_ar2, fig_field (four panels)

Style (transcript 45:26-46:57): a box around every panel, text in the paper's font, labels the size of the body text
at placement, minor ticks, no background grid, markers for measurements and lines for closed forms.

Data are read from the vendored thesis generators and the frozen outputs (read only, nothing is written there):

    .venv/bin/python tools/paper_figs/make_paper_figures.py [--layout onecol|twocol] [--font times|cm] [name ...]

The thesis chapter modules are vendored in tools/paper_figs/thesis/; their data is outputs/frozen (same bytes as the archived thesis).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
THESIS_TOOLS = HERE / "thesis"                   # vendored thesis chapter modules (read their figures' data from outputs/frozen)
sys.path.insert(0, str(THESIS_TOOLS))            # the thesis modules import their own `figstyle` from here

import matplotlib                                # noqa: E402
matplotlib.use("pdf")
import matplotlib.pyplot as plt                  # noqa: E402
from matplotlib.colors import LogNorm            # noqa: E402
from matplotlib.patches import Rectangle         # noqa: E402

_spec = importlib.util.spec_from_file_location("paper_figstyle", REPO / "tools" / "figstyle.py")
fs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fs)                     # the repo's save gate (no prose, symbols as math), under its own name

BLUE, ORANGE, GREEN = fs.CAT[0], fs.CAT[1], fs.CAT[2]
INK = "black"
GREY = "#6b6b6b"

LAYOUTS = {"onecol": {"full": 5.5, "one": 3.25},       # NeurIPS-style 5.5 in text block
           "twocol": {"full": 6.75, "one": 3.25}}      # ICML / Physical Review column grid
W = dict(LAYOUTS["onecol"])
OUT = REPO / "reports" / "paper-figures" / "onecol"
FONT = "times"


def apply_style(font: str) -> None:
    """The repo's typography (Times text, CM math), then Jérôme's taste on top of it."""
    fs.apply()
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "axes.grid": False, "lines.linewidth": 1.5, "lines.markersize": 4.5,
        "legend.handlelength": 2.0, "legend.labelspacing": 0.3,
    })
    if font == "cm":
        plt.rcParams.update({"font.serif": ["cmr10"], "axes.formatter.use_mathtext": True})


def figure(ncols: int = 1, width: float | None = None, height: float = 2.4, **kw):
    return plt.subplots(1, ncols, figsize=(width or W["one"], height), constrained_layout=True, **kw)


def save(fig, name: str) -> None:
    fs.save(fig, OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  {name}.pdf")


def panel(ax, letter: str) -> None:
    """Panel letter above the top-left corner, outside the data."""
    ax.set_title(f"({letter})", loc="left")


def plain_log_y(ax, ticks) -> None:
    """Log y axis with plain tick labels (0.05, 0.1, ...) instead of 5 x 10^-2."""
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:g}" for v in ticks])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def log_n_axis(ax, ns) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{int(n)}" for n in ns])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


# ------------------------------------------------------------------------------------- Gaussian chain --
def _precision_snapshots(L: int, alpha: float):
    import fig_ch05 as c5
    S0 = c5.ar1_cov(L, alpha)
    ref = c5.dist_mass(c5.precision_t(S0, 0.0), 1, 1)
    grid = np.logspace(-3.0, 0.62, 4000)
    off = np.array([c5.dist_mass(c5.precision_t(S0, t), 2, L) / ref for t in grid])
    return c5, S0, grid, off, float(grid[int(np.argmax(off))])


def fig_locality(L: int = 24, alpha: float = 0.85) -> None:
    c5, S0, _, _, t_star = _precision_snapshots(L, alpha)
    ts = [0.0, 0.01, t_star, 0.3, 1.8]
    fig, axes = figure(len(ts), width=W["full"], height=1.45)
    im = None
    for ax, t in zip(axes, ts):
        im = ax.imshow(np.abs(c5.precision_t(S0, t)), norm=LogNorm(vmin=1e-3, vmax=5.0), cmap="Greys",
                       interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xticks([], minor=True); ax.set_yticks([], minor=True)
        ax.set_title(rf"$t^{{*}}={t_star:.3f}$" if t == t_star else rf"$t={t:g}$")
    axes[2].add_patch(Rectangle((-0.5, -0.5), L, L, fill=False, ec=BLUE, lw=1.6, clip_on=False, zorder=5))
    axes[0].set_ylabel(r"$|Q_t|$")
    cb = fig.colorbar(im, ax=list(axes), location="right", shrink=0.92, pad=0.015)
    cb.ax.minorticks_off()
    save(fig, "fig_locality")
    print(f"    t* = {t_star:.4f}")


def fig_locality_curve(L: int = 24, alpha: float = 0.85) -> None:
    _, _, grid, off, t_star = _precision_snapshots(L, alpha)
    fig, ax = figure(height=2.3)
    ax.semilogx(grid, off, color=INK, lw=1.6)
    ax.axvline(t_star, color=BLUE, lw=1.0, ls="--")
    ax.text(t_star * 1.12, off.max() * 0.08, r"$t^\star$", color=BLUE, va="bottom")
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(0.0, off.max() * 1.08)
    ax.set_xlabel(r"diffusion time $t$")
    ax.set_ylabel(r"coupling beyond $|i-j|=1$")
    save(fig, "fig_locality_curve")


# ------------------------------------------------------------------------------------- Laplace chain --
def fig_laplace(alpha: float = 0.85, L: int = 9) -> None:
    import fig_ch07 as c7
    b = c7.laplace_scale(alpha)
    sig = np.sqrt(1.0 - alpha ** 2)
    k = L // 2
    fig, (a0, a1, a2) = figure(3, width=W["full"], height=2.2)

    e = np.linspace(-3.2, 3.2, 800)
    a0.semilogy(e, np.exp(-np.abs(e) / b) / (2 * b), color=ORANGE, label="Laplace")
    a0.semilogy(e, np.exp(-e ** 2 / (2 * sig ** 2)) / np.sqrt(2 * np.pi * sig ** 2), color=BLUE, ls="--",
                label="Gaussian")
    a0.set_xlim(e[0], e[-1]); a0.set_ylim(1e-4, 3e2)
    a0.set_xlabel(r"$\varepsilon$"); a0.set_ylabel(r"$\rho(\varepsilon)$")
    a0.legend(loc="upper right", frameon=False, handlelength=1.6)
    panel(a0, "a")

    xk = np.linspace(-4.0, 4.0, 121)
    for ax, t in ((a1, 0.05), (a2, 1.0)):
        sl, sg = [], []
        for v in xk:
            x = np.zeros(L); x[k] = v
            sl.append(c7.grid_bp_score(x, alpha, t, k))
            sg.append(c7.gaussian_score(x, alpha, t, k))
        ax.plot(xk, sl, color=ORANGE, label="Laplace chain")
        ax.plot(xk, sg, color=BLUE, ls="--", label="Gaussian chain")
        ax.set_xlim(xk[0], xk[-1]); ax.set_ylim(-16, 16)
        ax.set_xlabel(r"$x_k$")
        ax.set_title(rf"$t={t:g}$")
    panel(a1, "b"); panel(a2, "c")
    a1.set_ylabel(r"score $S_k(x)$")
    a2.set_yticklabels([])
    a2.legend(loc="lower left", frameon=False)
    save(fig, "fig_laplace")


# ------------------------------------------------------------------------------------- grid BP --
def fig_grid_gaussian() -> None:
    import fig_ch08 as c8
    by: dict[int, dict[float, float]] = defaultdict(dict)
    for r in c8.read("exp_01/grid_sweep_M.csv"):
        if abs(float(r["rho"]) - 0.8) < 1e-9:
            by[int(r["grid_size"])][float(r["t"])] = float(r["score_rel_error_mean"])
    sizes = [101, 201, 401]
    fig, ax = figure(height=2.45)
    for M, c, m in zip(sizes, (fs.BLUE_RAMP[0], fs.BLUE_RAMP[2], fs.BLUE_RAMP[4]), "o^s"):
        ts = np.array(sorted(by[M]))
        er = np.array([max(by[M][t], 1e-16) for t in ts])
        ax.loglog(ts, er, marker=m, color=c, lw=1.0, ms=4.0, label=rf"$N_g={M}$")
    ax.axhline(2.2e-16, color=GREY, lw=0.8, ls=":")
    ax.text(0.03, 2.2e-16 / 2.5, "round-off", transform=ax.get_yaxis_transform(), ha="left", va="top",
            color=GREY, fontsize=9)
    ax.set_xlabel(r"diffusion time $t$")
    ax.set_ylabel("relative score error")
    ax.set_ylim(1e-17, 1e-1)
    ax.legend(loc="upper right", frameon=False)
    save(fig, "fig_grid_gaussian")


# ------------------------------------------------------------------------------------- EM --
def fig_em_innovation() -> None:
    import fig_ch08 as c8
    dens = c8.read("exp_18/innovation_density.csv")
    summ = c8.read("exp_18/innovation_summary.csv")
    fig, (aL, aR) = figure(2, width=W["full"], height=2.45)

    def curve(arm: str, budget: int, C: int):
        pts = sorted((float(r["e"]), float(r["density"])) for r in dens
                     if r["arm"] == arm and int(r["budget"]) == budget and int(r["n_components"]) == C)
        return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])

    budgets = [128, 512, 2048]
    for nb, c in zip(budgets, (fs.BLUE_RAMP[1], fs.BLUE_RAMP[3], fs.BLUE_RAMP[4])):
        aL.semilogy(*curve("fitted", nb, 8), color=c, label=rf"$N={nb}$")
    aL.semilogy(*curve("true", 0, 0), color=INK, ls="--", label="Laplace")
    aL.set_xlim(-2.6, 2.6); aL.set_ylim(1e-4, 6.0)
    aL.set_xlabel(r"$\varepsilon$"); aL.set_ylabel(r"$\rho(\varepsilon)$")
    aL.legend(loc="lower center", frameon=False, ncol=2)
    panel(aL, "a")

    rows = sorted((r for r in summ if int(r["n_components"]) == 8), key=lambda r: int(r["n_chains"]))
    n = np.array([int(r["n_chains"]) for r in rows], float)

    def rel(est: str, true: str) -> np.ndarray:
        return np.array([abs(float(r[est]) - float(r[true])) / float(r[true]) for r in rows])

    aR.loglog(n, rel("innovation_excess_kurtosis", "innovation_excess_kurtosis_true"), "o-", color=ORANGE,
              lw=1.0, label="kurtosis")
    aR.loglog(n, rel("innovation_var", "innovation_var_true"), "s-", color=GREEN, lw=1.0, label="variance")
    aR.loglog(n, rel("rho_hat", "rho_true"), "^-", color=BLUE, lw=1.0, label=r"$\alpha$")
    log_n_axis(aR, n)
    aR.set_xlim(n[0] / 1.3, n[-1] * 1.3); aR.set_ylim(1e-4, 1.0)
    aR.set_xlabel(r"training sequences $N$"); aR.set_ylabel("relative error")
    aR.legend(loc="lower left", frameon=False)
    panel(aR, "b")
    save(fig, "fig_em_innovation")


def _paired_e03() -> dict[str, dict[int, dict[int, float]]]:
    """Pooled test risk per arm, size and seed: E02a's EM-BP (C = 8) and E03's networks, on the same test bundles.

    Both experiments draw the exp_31 bundles from the same tags, so a row of one joins a row of the other on
    (seed, n_chains, test_bundle); a network row without its EM-BP partner is an error, not a gap.
    """
    import csv
    risk: dict[str, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    bundle: dict[tuple[int, int], str] = {}
    with open(REPO / "outputs" / "E02a" / "capacity.csv") as fh:
        for r in csv.DictReader(fh):
            if r["region"] == "all" and r["arm"] == "em_bp" and int(r["components"]) == 8:
                key = (int(r["seed"]), int(r["n_chains"]))
                risk["em_bp"][key[1]][key[0]] = float(r["risk"])
                bundle[key] = r["test_bundle"]
    with open(REPO / "outputs" / "E03" / "confirm.csv") as fh:
        for r in csv.DictReader(fh):
            if r["region"] != "all":
                continue
            key = (int(r["seed"]), int(r["n_chains"]))
            if bundle.get(key) != r["test_bundle"]:
                raise SystemExit(f"E03 row {key} {r['arch']} has no EM-BP partner on its test bundle")
            risk[r["arch"]][key[1]][key[0]] = float(r["risk"])
    return risk


def fig_value_of_structure() -> None:
    risk = _paired_e03()
    ns = sorted(risk["em_bp"])
    mean = {a: np.array([np.mean(list(risk[a][n].values())) for n in ns]) for a in ("em_bp", "window", "mlp", "bigru")}
    fig, ax = figure(height=2.6)
    ax.loglog(ns, mean["mlp"], "o-", color=ORANGE, lw=1.0)
    ax.loglog(ns, mean["window"], "^-", color=GREEN, lw=1.0)
    ax.loglog(ns, mean["bigru"], "D-", color=GREY, lw=1.0, ms=4.0)
    ax.loglog(ns, mean["em_bp"], "s-", color=BLUE, lw=1.0)
    k = _e56("kernels.csv")
    if k is not None:                  # E56's neural energy kernel, paired with E03 on the same test bundles (read_e56)
        log_n_axis(ax, ns)
        ax.set_xlim(ns[0] / 1.4, ns[-1] * 1.4)
        ax.set_ylim(0.015, 1.2)
        e = k[(k.failed == 0) & (k.arm == "energy")].drop_duplicates(["seed", "n_chains"])
        en = e.groupby("n_chains").risk_pooled.mean().reindex(ns).to_numpy()
        ax.loglog(ns, en, "p--", color=fs.CAT[4], lw=1.0, ms=4.5)
        # on-curve label in the gap between the neural kernel and BiGRU, along their common slope (N = 128..512)
        xm = np.sqrt(ns[1] * ns[2])
        ym = (np.sqrt(en[1] * en[2]) * np.sqrt(mean["bigru"][1] * mean["bigru"][2])) ** 0.5
        p0, p1 = ax.transData.transform([(ns[1], en[1]), (ns[2], en[2])])
        ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        ax.text(xm, ym, "neural kernel", color=fs.CAT[4], va="center", ha="center", rotation=ang,
                rotation_mode="anchor")
    log_n_axis(ax, ns)
    ax.set_xlim(ns[0] / 1.4, ns[-1] * 1.4)
    ticks = [0.02, 0.05, 0.1, 0.2, 0.5]
    ax.set_ylim(0.015, 1.2)
    ax.set_yticks(ticks); ax.set_yticklabels([f"{v:g}" for v in ticks])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.text(ns[0], mean["mlp"][0] * 1.25, "global MLP", color=ORANGE, va="bottom")
    ax.text(ns[-1] * 1.3, mean["window"][-1] * 1.15, "window CNN", color=GREEN, va="bottom", ha="right")
    ax.text(ns[-1] * 1.3, mean["bigru"][-1] / 1.2, "BiGRU", color=GREY, va="top", ha="right")
    ax.text(ns[0], mean["em_bp"][0] / 1.35, "EM–BP", color=BLUE, va="top")
    ax.set_xlabel(r"training sequences $N$")
    ax.set_ylabel("relative score error")
    save(fig, "fig_value_of_structure")
    for a in ("mlp", "window", "bigru", "unet", "transformer", "tcn"):
        rat = [np.mean([risk[a][n][s] / risk["em_bp"][n][s] for s in risk["em_bp"][n]]) for n in ns]
        print(f"    {a:12s} / EM-BP, mean per-seed ratio over N = {ns}: " + " ".join(f"{v:.2f}" for v in rat))


# ------------------------------------------------------------------------------------- kernel families (E56) --
FAMILIES = (("spline", "B-spline"), ("rbf", "RBF"), ("fourier", "Fourier"), ("legendre", "Legendre"),
            ("loglinear2d", "log-linear 2-D"), ("energy", "neural kernel"))
SHIFT = ("spline", "rbf", "fourier", "legendre")


def _e56(name: str):
    """E56's per-shard CSVs concatenated, or None while the cluster fits have not landed."""
    import glob

    import pandas as pd
    paths = sorted(glob.glob(str(REPO / "outputs" / "hpc_E56_s*" / name)))
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else None


def fig_families() -> None:
    """Thesis Fig. 6.3 for every family: learned increment law against the Laplace truth at N = 128, 512, 2048.

    The mixture (C = 8) is the thesis fit (exp_18). Shift families plot log phi(eps); the two agnostic ones plot
    log K(eps | a = 0), the same function for a shift kernel. Until E56's densities exist only the truth is drawn there.
    """
    import fig_ch07 as c7
    import fig_ch08 as c8
    from matplotlib.lines import Line2D
    dens = _e56("densities.csv")
    mix = c8.read("exp_18/innovation_density.csv")
    b = c7.laplace_scale(0.85)
    eps = np.linspace(-2.6, 2.6, 400)
    fig, axes = plt.subplots(2, 4, figsize=(W["full"], 3.1), sharex=True, sharey=True, constrained_layout=True)
    sizes = (128, 512, 2048)
    cols = (fs.BLUE_RAMP[1], fs.BLUE_RAMP[3], fs.BLUE_RAMP[4])
    panels = (("mixture", "mixture, $C=8$"),) + FAMILIES
    for ax, (arm, label) in zip(axes.flat, panels):
        for n, c in zip(sizes, cols):
            if arm == "mixture":
                pts = sorted((float(r["e"]), float(r["density"])) for r in mix
                             if r["arm"] == "fitted" and int(r["budget"]) == n and int(r["n_components"]) == 8)
                if pts:
                    ax.semilogy([p[0] for p in pts], [p[1] for p in pts], color=c)
            elif dens is not None:
                d = dens[(dens.arm == arm) & (dens.n_chains == n) & (dens.seed == 0)]
                d = d[d.what == "log_phi"] if arm in SHIFT else d[(d.what == "log_K") & (d.a.abs() < 1e-9)]
                if len(d):
                    d = d.sort_values("x")
                    ax.semilogy(d.x, np.exp(d.value), color=c)
        ax.semilogy(eps, np.exp(-np.abs(eps) / b) / (2 * b), color=INK, ls="--")
        ax.set_title(label)
        ax.set_xlim(eps[0], eps[-1]); ax.set_ylim(1e-4, 6.0)
    for ax in axes[1]:
        ax.set_xlabel(r"$\varepsilon$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\rho(\varepsilon)$")
    last = axes.flat[-1]
    last.set_axis_off()
    handles = [Line2D([], [], color=c, label=rf"$N={n}$") for n, c in zip(sizes, cols)]
    handles.append(Line2D([], [], color=INK, ls="--", label="Laplace"))
    last.legend(handles=handles, loc="center", frameon=False)
    save(fig, "fig_families")


def fig_em_dynamics() -> None:
    """Step efficiency. (a) EM's log-likelihood against iteration from four starts (thesis exp_18, C = 8). (b) Validation
    score error against parameter updates at N = 2048, seed means: EM iterations for the spline and the neural energy
    kernel (E56's ladder), SGD steps for E03's networks."""
    import csv
    fig, (aL, aR) = figure(2, width=W["full"], height=2.45)
    with open(REPO / "outputs" / "frozen" / "exp_18" / "em_trace.csv") as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["n_components"]) == 8]
    by: dict[float, list] = defaultdict(list)
    for r in rows:
        by[float(r["rho_init"])].append((int(r["iteration"]) + 1, float(r["log_evidence"]) / 1e5))
    for (a0, pts), c in zip(sorted(by.items()), fs.BLUE_RAMP[1:]):
        it, ll = zip(*sorted(pts))
        aL.semilogx(it, ll, color=c, label=rf"$\alpha_0={a0:g}$")
    aL.set_ylabel(r"log-likelihood $/10^{5}$")
    aL.legend(loc="lower right", frameon=False)
    aL.set_xlabel("EM iteration")

    curves: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(REPO / "outputs" / "E03" / "confirm.csv") as fh:
        for r in csv.DictReader(fh):
            if r["region"] == "all" and int(r["n_chains"]) == 2048 and r["arch"] in ("mlp", "window", "bigru"):
                for item in r["val_curve"].split(";"):
                    step, v = item.split(":")
                    curves[r["arch"]][int(step)].append(float(v))
    for arch, c, m, lab in (("mlp", ORANGE, "o", "global MLP"), ("window", GREEN, "^", "window CNN"),
                            ("bigru", GREY, "D", "BiGRU")):
        st = sorted(curves[arch])
        aR.loglog(st, [np.mean(curves[arch][s]) for s in st], marker=m, color=c, lw=1.0, ms=4.0, label=lab)
    k = _e56("kernels.csv")
    if k is not None:
        for arm, c, m, lab in (("spline", BLUE, "s", "spline kernel"), ("energy", fs.CAT[4], "p", "neural kernel")):
            ev = k[(k.failed == 0) & (k.n_chains == 2048) & (k.arm == arm)].drop_duplicates(["seed"])
            pts: dict[int, list[float]] = defaultdict(list)
            for vc in ev.val_curve:
                for item in str(vc).split(";"):
                    it, v = item.split(":")
                    pts[int(it)].append(float(v))
            st = [s for s in sorted(pts) if len(pts[s]) == len(ev)]      # iterations every seed reached
            aR.loglog(st, [np.mean(pts[s]) for s in st], marker=m, color=c, lw=1.0, ms=4.0, label=lab)
    aR.set_xlim(5, 5e4)
    aR.set_ylim(0.015, 1.5)
    plain_log_y(aR, [0.02, 0.05, 0.1, 0.2, 0.5, 1])
    aR.set_xlabel("parameter updates")
    aR.set_ylabel("relative score error")
    aR.legend(loc="upper right", frameon=False, fontsize=8, ncol=2, columnspacing=1.0)
    panel(aL, "a"); panel(aR, "b")
    save(fig, "fig_em_dynamics")


# ------------------------------------------------------------------------------------- breaking Markov --
def _core(name: str):
    """A registered reader from tools/core (the paper reads the frozen outputs through the same code as the verdicts)."""
    sys.path.insert(0, str(REPO / "tools" / "core"))
    return importlib.import_module(name)


def fig_nonmarkov() -> None:
    """(a) Network / EM-BP error ratio against the distance delta to the best first-order chain: thesis exp_21_clean
    (global latent, long-range couplings; delta from the covariance) and E62's Laplace AR(2) chains (delta exact, the
    conditional mutual information), N = 2048. (b) E62: error against phi2 at N = 2048 for first-order EM-BP, the
    pair-state kernel, the best network and the true-covariance linear denoiser; the line is the exact error of the best
    first-order kernel (E62's registered calibration)."""
    import fig_ch10 as c10
    from matplotlib.lines import Line2D
    adv = c10._nonmarkov()
    betas, gammas = [0.1, 0.25, 0.5, 1.0], [0.05, 0.1, 0.2, 0.4]
    kl_b = [c10.kl_per_site(c10.global_cov(x), c10.CHOW_LIU[("global", x)]) for x in betas]
    kl_g = [c10.kl_per_site(c10.longrange_cov(x), c10.CHOW_LIU[("longrange", x)]) for x in gammas]
    r62 = _core("read_e62")
    arms, _ = r62.load()
    cal = r62.calibration()
    fig, (ax, bx) = figure(2, width=W["full"], height=2.4)
    series = ((("gauss", "global"), betas, kl_b, "o"), (("laplace", "global"), betas, kl_b, "s"),
              (("gauss", "longrange"), gammas, kl_g, "^"))
    phis = [p for p in r62.PHI2 if p > 0]
    d62 = [cal[p]["delta_exact"] for p in phis]
    for net, c in (("cnn", GREEN), ("mlp", ORANGE)):
        for key, xs, kl, m in series:
            ax.loglog(kl, [adv[(key[0], key[1], s)][net] for s in xs], m, color=c, ms=4.5, ls="none")
        net62 = "window" if net == "cnn" else "mlp"
        ax.loglog(d62, [r62.seed_mean(arms, net62, 2048, p) / r62.seed_mean(arms, "first", 2048, p) for p in phis],
                  "D", color=c, ms=4.0, ls="none")
    ax.axhline(1.0, color=INK, lw=0.8, ls="--")
    ax.set_ylim(0.3, 60)
    plain_log_y(ax, [0.5, 1, 2, 5, 10, 20, 50])
    ax.set_xlabel(r"distance to a Markov chain $\delta$")
    ax.set_ylabel("network / EM–BP error")
    handles = [Line2D([], [], color=GREEN, marker="o", ls="none", label="window CNN"),
               Line2D([], [], color=ORANGE, marker="o", ls="none", label="global MLP"),
               Line2D([], [], color=INK, marker="o", ls="none", mfc="none", label="latent, Gauss"),
               Line2D([], [], color=INK, marker="s", ls="none", mfc="none", label="latent, Laplace"),
               Line2D([], [], color=INK, marker="^", ls="none", mfc="none", label="long-range"),
               Line2D([], [], color=INK, marker="D", ls="none", mfc="none", label="AR(2), Laplace")]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)

    P = list(r62.PHI2)
    sm = lambda a: [r62.seed_mean(arms, a, 2048, p) for p in P]
    best = [min(r62.seed_mean(arms, a, 2048, p) for a in r62.NETS) for p in P]
    bx.plot(P, [cal[p]["first_order_floor"] for p in P], "-", color=BLUE, lw=1.0)
    bx.plot(P, sm("first"), "o", color=BLUE, ms=4.5, ls="none", label="first order")
    bx.plot(P, sm("pair"), "s", color=fs.CAT[4], ms=4.5, ls="none", label="pair state")
    bx.plot(P, best, "D", color=GREY, ms=4.0, ls="none", label="best network")
    bx.plot(P, sm("lmmse"), "--", color=INK, lw=0.8, label="linear")
    bx.set_xlim(-0.36, 0.66)
    bx.set_ylim(0, 0.37)
    bx.set_xlabel(r"second-order coefficient $\phi_2$")
    bx.set_ylabel("relative score error")
    bx.legend(loc="upper left", frameon=False, fontsize=8)
    panel(ax, "a"); panel(bx, "b")
    save(fig, "fig_nonmarkov")
    for key, xs, kl, _ in series:
        print(f"    {key}: " + "  ".join(f"d={d:.2e} cnn={adv[(key[0], key[1], s)]['cnn']:.2f} "
                                         f"mlp={adv[(key[0], key[1], s)]['mlp']:.2f}" for s, d in zip(xs, kl)))
    print("    E62 N=2048: " + "  ".join(f"phi2={p:g} d={cal[p]['delta_exact']:.3f} first={f:.4f} pair={q:.4f} net={b:.4f}"
                                         for p, f, q, b in zip(P, sm("first"), sm("pair"), best)))


# ------------------------------------------------------------------------------------- tables --
def _write_table(name: str, body: str) -> None:
    (OUT / f"{name}.tex").write_text(body)
    print(f"  {name}.tex")


def _field_rows():
    """Allen-Cahn rows, relative score error divided by the linear floor per seed, then averaged over seeds.
    E33/E43/E60 on seeds 0-15 (frozen); E55's neural energy kernel and U-Net on seeds 0-7 (finished cells, fetched 7 Nov
    2026). E55's fit parts share one draws hash per (cell, seed); the oracle hash differs only where the recorded cell scale
    differs in its last bits between the CPU and GPU nodes (checked here), so E33's per-seed floor applies."""
    import glob

    import pandas as pd
    r60 = _core("read_e60")
    risk, _ = r60.load()
    orc = pd.read_csv(REPO / "outputs/hpc_E33o/oracles.csv")
    floor = orc[orc.arm == "lmmse_true"].drop_duplicates(["cell", "seed"]).set_index(["cell", "seed"]).risk
    cat = lambda pat: pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(REPO / pat)))], ignore_index=True)
    ll = cat("outputs/hpc_E43a_s*/kernels.csv")
    ll = ll[(ll.selection == "risk") & (ll.arm == "em_loglinear")].assign(arm="loglinear2d")
    ts = cat("outputs/hpc_E33k_s*/kernels.csv")
    ll = pd.concat([ll, ts[(ts.selection == "risk") & (ts.arm == "em_tilted_spline")]], ignore_index=True)
    nets55, k55 = cat("outputs/hpc_E55_nets_*/nets.csv"), cat("outputs/hpc_E55_kern_*/kernels.csv")
    fits = pd.concat([d[["cell", "seed", "test_draws"]] for d in (nets55, k55)])
    assert (fits.groupby(["cell", "seed"]).test_draws.nunique() == 1).all(), "E55 fit parts disagree on the test draws"
    o55 = cat("outputs/hpc_E55_oracles/oracles.csv").drop_duplicates("cell").set_index("cell").scale
    s55 = nets55.drop_duplicates("cell").set_index("cell").scale
    assert float((s55 / o55.reindex(s55.index) - 1).abs().max()) < 1e-14, "E55 cell scale differs beyond round-off"
    e55 = pd.concat([k55[k55.selection == "risk"].assign(arm="em_energy"), nets55[nets55.arm == "unet"]])
    e55 = e55[e55.seed < 8]
    extra = pd.concat([ll, e55], ignore_index=True)
    extra = extra.assign(risk=extra.risk / floor.reindex(pd.MultiIndex.from_frame(extra[["cell", "seed"]])).to_numpy())
    cols = ["cell", "arm", "n_chains", "seed", "risk", "n_params"]
    return r60, pd.concat([risk.reindex(columns=cols), extra.reindex(columns=cols)], ignore_index=True)


FIELD_ROWS = (("site energy", "em_pair_quartic", "5"), ("neural kernel", "em_energy", None),
              ("free mixture", "em_free_C4", "24"), ("tilted spline", "em_tilted_spline", "25"),
              ("log-linear 2-D", "loglinear2d", "100"), ("MDN", "em_mdn", "1{,}516"),
              ("shift mixture", "em_mixture", "25"), (None, None, None),
              ("U-Net", "unet", r"$1.2\times10^{7}$"), ("best of four", "best", None))


def tab_field() -> None:
    """Table 1 and its M = 128 twin: the Allen-Cahn cells, every kernel family and network, over the linear floor."""
    r60, df = _field_rows()
    cells = ("T0.3_h1_L32", "T0.5_iso_L32", "T1_iso_L32", "T2_iso_L32")
    for m, name in ((2048, "tab_field"), (128, "tab_field_128")):
        lines = [r"\begin{tabular}{@{}lrcccc@{}}", r"\toprule",
                 r" & param. & $\Theta=0.3$ & $0.5$ & $1$ & $2$\\", r"\midrule"]
        for label, arm, npar in FIELD_ROWS:
            if label is None:
                lines.append(r"\midrule")
                continue
            vals = []
            for c in cells:
                if arm == "best":
                    v = min(r60.seed_mean(df, c, a, m) for a in r60.NETS)
                else:
                    d = df[(df.cell == c) & (df.arm == arm) & (df.n_chains == m)]
                    v = float(d.risk.mean()) if len(d) and d.risk.notna().all() else float("nan")
                vals.append("--" if not np.isfinite(v) else f"{v:.3f}")
            if npar is None and arm == "em_energy":
                ps = sorted({int(p) for p in df[(df.arm == arm) & df.cell.isin(cells)].n_params.dropna()})
                npar = "--".join(f"{p:,}".replace(",", "{,}") for p in (ps[0], ps[-1])) if len(ps) > 1 else f"{ps[0]}"
            lines.append(f"{label} & {npar or ''} & " + " & ".join(vals) + r"\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        _write_table(name, "\n".join(lines) + "\n")


def tab_chains() -> None:
    """Table 2: relative score error at N = 2048 on E61's and E57's chains (seed means over seeds 0-7)."""
    r61, r57 = _core("read_e61"), _core("read_e57")
    d61 = r61.load()
    d57, _ = r57.load()
    f = lambda v: "--" if not np.isfinite(v) else ("0" if abs(v) < 5e-5 else f"{v:.3f}") 
    rows = {"inside": [("Gaussian", "gauss", 61), ("Student-$t$", "student", 61), ("uniform", "uniform", 61),
                       ("bimodal$^\\dagger$", "bimodal", 57)],
            "outside": [("skewed", "skew", 61), ("tanh drift$^\\dagger$", "tanh", 57),
                        ("heteroscedastic$^\\dagger$", "hetero", 57), ("double well", "langevin", 61),
                        ("Pearson", "pearson", 61)]}
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r" & & \multicolumn{3}{c}{kernel inside BP} & \\", r"\cmidrule(lr){3-5}",
             r"chain & linear & shift & neural & neural$^{\ast}$ & network\\", r"\midrule"]
    for group, title in (("inside", "inside the shift class"), ("outside", "outside it")):
        lines.append(r"\multicolumn{6}{@{}l}{\emph{" + title + r"}}\\")
        for label, chain, src in rows[group]:
            r, d = (r61, d61) if src == 61 else (r57, d57)
            vals = [r.seed_mean(d, chain, "lmmse", 2048), r.seed_mean(d, chain, "mixture", 2048),
                    r.seed_mean(d, chain, "energy", 2048),
                    r.seed_mean(d, chain, "energy_clean", 2048) if src == 61 else float("nan"),
                    min(r.seed_mean(d, chain, a, 2048) for a in r.NETS)]
            lines.append(f"{label} & " + " & ".join(f(v) for v in vals) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write_table("tab_chains", "\n".join(lines) + "\n")


# ------------------------------------------------------------------------------------- TikZ --
_PREAMBLE = r"""\documentclass[border=2pt]{standalone}
%s
\usepackage{amsmath,amssymb}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,calc,positioning}
\begin{document}
%s
\end{document}
"""


def _tikz(name: str, body: str) -> None:
    font = r"\usepackage{times}" if FONT == "times" else r"\usepackage{lmodern}"
    tex = OUT / f"{name}.tex"
    tex.write_text(_PREAMBLE % (font, body))
    (OUT / f"{name}.tikz").write_text(body + "\n")      # the bare picture, for \input in the paper
    tectonic = shutil.which("tectonic")
    if tectonic is None:
        print(f"  {name}.tex written; tectonic not found, PDF not built")
        return
    subprocess.run([tectonic, "-X", "compile", "--outdir", str(OUT), str(tex)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  {name}.pdf (from {name}.tex)")


import figs_v2  # the restructured paper's figures, 8 Nov 2026 (tools/paper_figs/figs_v2.py)


def _v2(name: str):
    """A figs_v2 figure, run against this module's layout, style and save gate."""
    def run() -> None:
        getattr(figs_v2, name)(sys.modules[__name__])
    run.__name__ = name
    return run


ALL = {f.__name__: f for f in (fig_locality, fig_locality_curve, fig_laplace, fig_grid_gaussian, fig_em_innovation,
                               fig_value_of_structure, fig_families, fig_em_dynamics,
                               fig_nonmarkov, tab_field, tab_chains)}
ALL.update({n: _v2(n) for n in figs_v2.NAMES})


def main() -> int:
    global OUT, FONT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("names", nargs="*", help=f"any of {', '.join(ALL)} (default: all)")
    ap.add_argument("--layout", choices=list(LAYOUTS), default="onecol")
    ap.add_argument("--font", choices=["times", "cm"], default="times")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    unknown = [n for n in args.names if n not in ALL]
    if unknown:
        ap.error(f"unknown figure(s): {', '.join(unknown)}")
    W.update(LAYOUTS[args.layout])
    FONT = args.font
    OUT = args.out or REPO / "reports" / "paper-figures" / args.layout
    OUT.mkdir(parents=True, exist_ok=True)
    apply_style(args.font)
    print(f"layout {args.layout} (full width {W['full']} in), font {args.font} -> {OUT}")
    for name in args.names or ALL:
        ALL[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
