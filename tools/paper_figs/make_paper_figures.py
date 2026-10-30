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

Style (transcript 45:26-46:57): a box around every panel, text in the paper's font, labels the size of the body text
at placement, minor ticks, no background grid, markers for measurements and lines for closed forms.

Data are read from the thesis generators and its frozen outputs (read only, nothing is written there):

    .venv/bin/python tools/paper_figs/make_paper_figures.py [--layout onecol|twocol] [--font times|cm] [name ...]

THESIS_TOOLS overrides the location of Diffusion/Markov_Score_Diffusion/thesis/tools.
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
THESIS_TOOLS = Path(os.environ.get(
    "THESIS_TOOLS", REPO.parent / "Diffusion" / "Markov_Score_Diffusion" / "thesis" / "tools"))
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
        ax.set_title(rf"$t^\star={t_star:.3f}$" if t == t_star else rf"$t={t:g}$")
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


def fig_closure() -> None:
    import csv
    with open(THESIS_TOOLS / "data" / "ch07_metrics.csv") as fh:
        rows = list(csv.DictReader(fh))
    d = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    t = d["t"]
    fig, ax = figure(height=2.45)
    ax.fill_between(t, d["score_q10"], d["score_q90"], color=ORANGE, alpha=0.18, lw=0)
    ax.loglog(t, d["score_med"], color=ORANGE)
    ax.loglog(t, d["grid_selfconv_med"], color=BLUE, ls="--")
    i = int(np.argmin(np.abs(t - 0.0725)))
    ax.annotate("", xy=(t[i], d["score_med"][i] * 0.8), xytext=(t[i], d["grid_selfconv_med"][i] * 1.4),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9, shrinkA=0, shrinkB=0))
    ax.text(t[0] * 1.15, d["score_q90"][0] * 2.5, "Gaussian closure", color=ORANGE, va="bottom")
    ax.text(t[0] * 1.15, d["grid_selfconv_med"][0] / 4, "grid BP", color=BLUE, va="top")
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(1e-8, 30.0)
    ax.set_xlabel(r"diffusion time $t$")
    ax.set_ylabel("relative score error")
    save(fig, "fig_closure")


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

FACTOR_GRAPH = r"""\begin{tikzpicture}[
    var/.style={circle,draw,thick,minimum size=8mm,inner sep=0pt},
    obs/.style={circle,draw,thick,fill=black!12,minimum size=8mm,inner sep=0pt},
    fac/.style={rectangle,fill=black,minimum size=2.4mm,inner sep=0pt},
    lab/.style={font=\small}]
  \foreach \i/\n in {0/1,1/2,2/3,3/L} {
    \node[var] (a\i) at (1.6*\i,0) {$a_{\n}$};
    \node[fac] (o\i) at (1.6*\i,-0.95) {};
    \node[obs] (x\i) at (1.6*\i,-1.9) {$x_{\n}$};
    \draw (a\i) -- (o\i) -- (x\i);
  }
  \foreach \i/\j in {0/1,1/2} {
    \node[fac] (k\i) at ($(a\i)!0.5!(a\j)$) {};
    \draw (a\i) -- (k\i) -- (a\j);
  }
  \draw (a2) -- ($(a2)!0.32!(a3)$);
  \node at ($(a2)!0.5!(a3)$) {$\cdots$};
  \draw ($(a2)!0.68!(a3)$) -- (a3);
  \node[lab,above=1.2mm of k1] {$K(a_3\,|\,a_2)$};
  \node[lab,right=1.2mm of o3] {$\ell_t(x_L\mid a_L)$};
  \draw[-{Stealth[length=2mm]},thick] ($(a0)+(-0.1,0.95)$) -- node[above,lab]{$\mu_{\to}$} ($(a1)+(0.3,0.95)$);
  \draw[-{Stealth[length=2mm]},thick] ($(a3)+(0.1,0.95)$) -- node[above,lab]{$\mu_{\leftarrow}$} ($(a2)+(-0.3,0.95)$);
\end{tikzpicture}"""

EM_LOOP = r"""\begin{tikzpicture}[
    box/.style={draw,thick,align=center,font=\small,minimum height=9mm,inner sep=3pt,text width=2.0cm},
    data/.style={box,fill=black!8},
    arr/.style={-{Stealth[length=2mm]},thick},
    lab/.style={font=\small,align=center}]
  \node[font=\small\bfseries,anchor=west] at (-1.2,1.0) {learn};
  \node[data] (d) at (0,0) {noisy\\ sequences};
  \node[box] (p) at (3.4,0) {pair\\ posteriors};
  \node[box] (k) at (6.8,0) {kernel $K_\theta$};
  \draw[arr] (d) -- node[above,lab]{BP} node[below,lab]{E-step} (p);
  \draw[arr] (p) -- node[above,lab]{refit} node[below,lab]{M-step} (k);
  \draw[arr] (k.north) to[out=160,in=20] node[above,lab]{repeat} (d.north);
  \node[font=\small\bfseries,anchor=west] at (-1.2,-1.55) {use};
  \node[data] (x) at (0,-2.5) {noisy\\ sequence $x$};
  \node[box] (m) at (3.4,-2.5) {$\mathbb{E}[a\mid x]$};
  \node[box] (s) at (6.8,-2.5) {score $S_t(x)$};
  \draw[arr] (x) -- node[below,lab]{BP} (m);
  \draw[arr] (m) -- (s);
  \draw[thick,dashed,-{Stealth[length=2mm]}] (k.south) -- (6.8,-1.25) -| (1.7,-2.47);
\end{tikzpicture}"""


def _tikz(name: str, body: str) -> None:
    font = r"\usepackage{times}" if FONT == "times" else r"\usepackage{lmodern}"
    tex = OUT / f"{name}.tex"
    tex.write_text(_PREAMBLE % (font, body))
    tectonic = shutil.which("tectonic")
    if tectonic is None:
        print(f"  {name}.tex written; tectonic not found, PDF not built")
        return
    subprocess.run([tectonic, "-X", "compile", "--outdir", str(OUT), str(tex)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  {name}.pdf (from {name}.tex)")


def fig_factor_graph() -> None:
    _tikz("fig_factor_graph", FACTOR_GRAPH)


def fig_em_loop() -> None:
    _tikz("fig_em_loop", EM_LOOP)


ALL = {f.__name__: f for f in (fig_locality, fig_locality_curve, fig_laplace, fig_factor_graph, fig_em_loop,
                               fig_grid_gaussian, fig_closure, fig_em_innovation, fig_value_of_structure)}


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
