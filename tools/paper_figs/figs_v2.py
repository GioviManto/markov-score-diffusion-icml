"""Figures for the restructured paper (8 Nov 2026), on the user's plan after the 7 Nov draft.

Main text
    fig_factor_graph   the posterior's factor graph with the forward and backward messages f_k, b_k (TikZ)
    fig_em_loop        learn the kernel by EM through BP, then use it through BP (TikZ)
    fig_routes         the two clean-data routes: kernel by maximum likelihood then BP, network by DSM (TikZ)
    fig_closure        grid BP against the Gaussian approximation (Kalman smoother), Laplace chain, 2,000 sequences
    fig_results        shift mixture, neural kernel, global MLP, window CNN against N: Laplace chain, Allen-Cahn field
    fig_steps          test error against parameter updates at N = 2048 (E65 refits)
Appendix
    fig_wall           the same curves against wall-clock seconds on one CPU thread (E63)
    fig_networks       the six networks against the two kernels (E02a, E56, E03)
    fig_nonmarkov_delta, fig_nonmarkov_ar2   the two panels of fig_nonmarkov, as separate and simpler figures
    fig_families_error every kernel family and the best network, one row each, at N = 128 and N = 2048
    fig_field          the Allen-Cahn field: potential, samples, single-site law, kernel at three states

Each function takes the base module (make_paper_figures, as `ctx`) for its layout, style and save gate, so the
house rules (box axes, no grid, minor ticks, markers for measurements and lines for closed forms, no prose in a
chart) are the same as for every other figure.
"""
from __future__ import annotations

import glob
import sys

import numpy as np

T12 = (0.05, 0.0725, 0.1053, 0.1527, 0.2216, 0.3215, 0.4665, 0.6769, 0.9821, 1.425, 2.0676, 3.0)
LADDER = (10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200, 1600, 2400)
NK_INNER = 10            # Adam steps per M-step of the neural energy kernel (E56/E58/E65 energy_inner = 10)


def _c(ctx) -> dict[str, str]:
    """One colour per model, the same in every figure (validated together: tools/PALETTE.md checks pass)."""
    fs = ctx.fs
    return {"mix": fs.CAT[0], "nk": fs.CAT[6], "mlp": fs.CAT[1], "cnn": fs.CAT[2], "ink": "black", "grey": "#6b6b6b"}


def _pd():
    import pandas as pd
    return pd


def _cat(ctx, pattern: str):
    pd = _pd()
    paths = sorted(glob.glob(str(ctx.REPO / pattern)))
    if not paths:
        raise SystemExit(f"no files match {pattern}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def _log2_n(ctx, ax, ns) -> None:
    ctx.log_n_axis(ax, ns)
    ax.set_xlim(ns[0] / 1.5, ns[-1] * 1.5)


# ===================================================================================== TikZ diagrams ==
FACTOR_GRAPH = r"""\begin{tikzpicture}[
    var/.style={circle,draw,thick,minimum size=8.2mm,inner sep=0pt,font=\small},
    obs/.style={circle,draw,thick,fill=black!12,minimum size=8.2mm,inner sep=0pt,font=\small},
    fac/.style={rectangle,fill=black,minimum size=2.2mm,inner sep=0pt},
    lab/.style={font=\small,inner sep=1pt},
    msg/.style={-{Stealth[length=2mm]},thick}]
  \node[var] (a1) at (0,0) {$a_1$};
  \node (d1) at (1.15,0) {$\cdots$};
  \node[var] (ak) at (2.3,0) {$a_k$};
  \node[var] (ak1) at (4.0,0) {$a_{k+1}$};
  \node (d2) at (5.15,0) {$\cdots$};
  \node[var] (aL) at (6.3,0) {$a_L$};
  \node[fac] (m0) at (-0.8,0) {};
  \draw (m0) -- (a1);
  \node[lab,above=0.8mm of m0] {$\mu$};
  \foreach \u/\v/\n in {a1/d1/ka,d1/ak/kb,ak/ak1/kc,ak1/d2/kd,d2/aL/ke} {
    \node[fac] (\n) at ($(\u)!0.5!(\v)$) {};
    \draw (\u) -- (\n) -- (\v);
  }
  \node[lab,above=0.8mm of kc] {$K$};
  \foreach \i/\n in {a1/1,ak/k,ak1/{k+1},aL/L} {
    \node[fac] (o\i) at ($(\i)+(0,-0.9)$) {};
    \node[obs] (x\i) at ($(\i)+(0,-1.8)$) {$x_{\n}$};
    \draw (\i) -- (o\i) -- (x\i);
  }
  \node[lab,left=0.8mm of oak] {$\ell_t$};
  \draw[msg] ($(a1.north)+(-0.25,0.5)$) -- node[lab,above]{$f_k$} ($(ak.north)+(-0.1,0.5)$);
  \draw[msg] ($(aL.north)+(0.25,0.5)$) -- node[lab,above]{$b_k$} ($(ak.north)+(0.1,0.5)$);
\end{tikzpicture}"""

EM_LOOP = r"""\begin{tikzpicture}[
    box/.style={draw,thick,align=center,font=\small,minimum height=8.5mm,inner sep=2pt,text width=1.55cm},
    data/.style={box,fill=black!8},
    arr/.style={-{Stealth[length=2mm]},thick},
    lab/.style={font=\footnotesize,align=center,inner sep=1.5pt},
    stage/.style={font=\small\itshape,anchor=west,inner sep=0pt}]
  \node[stage] at (-0.85,1.2) {learn};
  \node[data] (x) at (0,0) {noisy\\ sequences};
  \node[box] (xi) at (3.05,0) {counts\\ $\Xi(a,a')$};
  \node[box] (k) at (6.1,0) {kernel\\ $K_\theta(a'|a)$};
  \draw[arr] (x) -- node[lab,above]{E-step} node[lab,below]{BP} (xi);
  \draw[arr] (xi) -- node[lab,above]{M-step} node[lab,below]{$\max_\theta Q$} (k);
  \draw[arr] (k.north) |- node[lab,above,pos=0.75]{repeat} (1.52,1.0) -- (1.52,0.52);
  \node[stage] at (-0.85,-1.5) {use};
  \node[data] (y) at (0,-2.55) {noisy\\ sequence $\mathbf{x}$};
  \node[box] (m) at (3.05,-2.55) {$\mathbb{E}[\mathbf{a}\mid\mathbf{x}]$};
  \node[box] (s) at (6.1,-2.55) {score\\ $\mathbf{S}_t(\mathbf{x})$};
  \draw[arr] (y) -- node[lab,above]{BP} node[lab,below]{any $t$} (m);
  \draw[arr] (m) -- node[lab,above]{Tweedie} (s);
  \draw[thick,dashed,-{Stealth[length=2mm]}] (k.south) |- (1.52,-1.45) -- (1.52,-2.05);
\end{tikzpicture}"""

ROUTES = r"""\begin{tikzpicture}[
    box/.style={draw,thick,align=center,font=\small,minimum height=8.5mm,inner sep=2pt,text width=1.75cm},
    data/.style={box,fill=black!8,text width=2.3cm},
    arr/.style={-{Stealth[length=2mm]},thick},
    lab/.style={font=\footnotesize,align=center,inner sep=2pt}]
  \node[data] (a) at (0,0) {clean sequences};
  \node[box] (k) at (-1.9,-1.75) {kernel\\ $\widehat K_\theta(a'|a)$};
  \node[box] (n) at (1.9,-1.75) {network\\ $\boldsymbol{\epsilon}_\phi(\mathbf{x},t)$};
  \node[box] (s1) at (-1.9,-3.5) {score\\ $\mathbf{S}_t(\mathbf{x})$};
  \node[box] (s2) at (1.9,-3.5) {score\\ $\widehat{\mathbf{S}}_t(\mathbf{x})$};
  \draw[arr] (a.west) -| node[lab,left,pos=0.75]{max.\ likelihood} (k.north);
  \draw[arr] (a.east) -| node[lab,right,pos=0.75]{noise, DSM} (n.north);
  \draw[arr] (k) -- node[lab,left]{BP, any $t$} (s1);
  \draw[arr] (n) -- node[lab,right]{$-\boldsymbol{\epsilon}_\phi/\sqrt{\Delta_t}$} (s2);
\end{tikzpicture}"""


def fig_factor_graph(ctx) -> None:
    ctx._tikz("fig_factor_graph", FACTOR_GRAPH)


def fig_em_loop(ctx) -> None:
    ctx._tikz("fig_em_loop", EM_LOOP)


def fig_routes(ctx) -> None:
    ctx._tikz("fig_routes", ROUTES)


# ===================================================================== grid BP against the Gaussian approx ==
def _grid_bp_means(X: np.ndarray, t: float, alpha: float, b: float, N: int, A: float = 8.0) -> np.ndarray:
    """Posterior means of a batch of Laplace chains (B, L) by grid BP: the recursion of fig_ch07.grid_bp_marginals,
    vectorised over sequences (same grid, same N(0, 1) law for the first site, per-site rescaling)."""
    mu, D = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
    g = np.linspace(-A, A, N)
    K = np.exp(-np.abs(g[:, None] - alpha * g[None, :]) / b) / (2.0 * b)          # K[out, in]
    B, L = X.shape
    ll = -((X[:, :, None] - mu * g[None, None, :]) ** 2) / (2.0 * D)
    ell = np.exp(ll - ll.max(axis=2, keepdims=True))
    fwd = np.empty((L, B, N))
    m = np.exp(-g ** 2 / 2.0)[None, :] * ell[:, 0, :]
    fwd[0] = m / m.sum(axis=1, keepdims=True)
    for i in range(1, L):
        m = (fwd[i - 1] @ K.T) * ell[:, i, :]
        fwd[i] = m / m.sum(axis=1, keepdims=True)
    bwd = np.ones((L, B, N))
    for i in range(L - 2, -1, -1):
        m = (ell[:, i + 1, :] * bwd[i + 1]) @ K
        bwd[i] = m / m.max(axis=1, keepdims=True)
    post = fwd * bwd
    post /= post.sum(axis=2, keepdims=True)
    return np.einsum("lbn,n->bl", post, g)


def _closure_sweep(ctx, n_seq: int = 2000, L: int = 32, alpha: float = 0.85, batch: int = 200):
    """Relative score error per sequence of (i) the Gaussian approximation (the Kalman smoother of the
    covariance-matched Gaussian chain, i.e. the best linear denoiser) and (ii) grid BP on 401 points, both against
    grid BP on 801 points, at the paper's twelve levels. Cached in reports/paper-figures/data/ (deterministic)."""
    pd = _pd()
    path = ctx.REPO / "reports" / "paper-figures" / "data" / f"closure_sweep_{n_seq}.csv"
    if path.exists():
        return pd.read_csv(path)
    b = float(np.sqrt((1.0 - alpha ** 2) / 2.0))
    d = np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
    S0 = alpha ** d
    rows = []
    for t in T12:
        rng = np.random.default_rng([20261108, round(t * 1e4)])
        A = np.empty((n_seq, L))
        A[:, 0] = rng.standard_normal(n_seq)
        E = rng.laplace(0.0, b, size=(n_seq, L - 1))
        for i in range(1, L):
            A[:, i] = alpha * A[:, i - 1] + E[:, i - 1]
        mu, D = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
        X = mu * A + np.sqrt(D) * rng.standard_normal((n_seq, L))
        lin_err, grid_err = [], []
        for s in range(0, n_seq, batch):
            x = X[s:s + batch]
            m_ref = _grid_bp_means(x, t, alpha, b, 801)
            m_401 = _grid_bp_means(x, t, alpha, b, 401)
            m_lin = mu * (S0 @ np.linalg.solve(mu * mu * S0 + D * np.eye(L), x.T)).T
            s_ref, s_401, s_lin = ((mu * m - x) / D for m in (m_ref, m_401, m_lin))
            nrm = np.linalg.norm(s_ref, axis=1)
            lin_err.append(np.linalg.norm(s_lin - s_ref, axis=1) / nrm)
            grid_err.append(np.linalg.norm(s_401 - s_ref, axis=1) / nrm)
        le, ge = np.concatenate(lin_err), np.concatenate(grid_err)
        rows.append({"t": t, "n_seq": n_seq, "lin_med": np.median(le), "lin_q10": np.quantile(le, 0.1),
                     "lin_q90": np.quantile(le, 0.9), "grid_med": np.median(ge), "grid_q90": np.quantile(ge, 0.9)})
        print(f"    t={t:<7g} Gaussian approx. median {rows[-1]['lin_med']:.4f}  grid {rows[-1]['grid_med']:.2e}")
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def fig_closure(ctx) -> None:
    C = _c(ctx)
    d = _closure_sweep(ctx)
    t = d.t.to_numpy()
    fig, ax = ctx.figure(height=2.45)
    ax.fill_between(t, d.lin_q10, d.lin_q90, color=C["grey"], alpha=0.18, lw=0)
    ax.loglog(t, d.lin_med, "o-", color=C["grey"], lw=1.0, ms=4.0)
    ax.loglog(t, d.grid_med, "s-", color=C["mix"], lw=1.0, ms=4.0)
    i = 1
    ax.annotate("", xy=(t[i], d.lin_q10[i] * 0.8), xytext=(t[i], d.grid_med[i] * 1.5),
                arrowprops={"arrowstyle": "<->", "color": C["ink"], "lw": 0.9, "shrinkA": 0, "shrinkB": 0})
    ax.text(t[0] * 1.08, 1.2, "Gaussian approximation", color=C["grey"], va="bottom", ha="left")
    ax.text(t[0] * 1.08, 8e-8, "grid BP", color=C["mix"], va="top", ha="left")
    ax.set_xlim(t[0] / 1.2, t[-1] * 1.2)
    ax.set_ylim(1e-9, 10.0)
    ax.set_xlabel(r"diffusion time $t$")
    ax.set_ylabel("relative score error")
    ctx.save(fig, "fig_closure")
    print(f"    t=0.05: Gaussian approx. median {d.lin_med[0]:.3f} (10-90%: {d.lin_q10[0]:.2f}-{d.lin_q90[0]:.2f}), "
          f"grid {d.grid_med[0]:.1e}; t=3: {d.lin_med.iloc[-1]:.1e}, grid {d.grid_med.iloc[-1]:.1e}")


# ================================================================================== results against N ==
def _results_data(ctx):
    """Seed medians of the pooled relative score error, by panel and arm, plus the linear floor.

    Laplace chain (16 seeds, E03's bundles): the C = 8 mixture by EM (E02a), the neural energy kernel by EM (E56),
    the neural energy kernel by maximum likelihood on the clean sequences (E58, width 16), the global MLP and the
    window CNN (E58's networks, E03's recipe at every N; E58's gate G2 reproduces E03 at N <= 2048).
    Allen-Cahn field, Theta = 0.3 (seeds 0-7, E33's bundles): the shift mixture by EM (E33), the neural energy kernel
    by EM (E55), and E58's clean kernel and networks. The linear floor is the best linear denoiser (E58 `lmmse`).
    Medians, because a few neural-kernel fits stalled at the linear floor (App. C); the means are in the text."""
    pd = _pd()
    r58 = ctx._core("read_e58")
    d58 = r58.load()
    med = lambda df, col: df.groupby("n_chains")[col].median()
    e58 = lambda cell, arm: med(d58[(d58.cell == cell) & (d58.arm == arm)], "risk")
    e02 = pd.read_csv(ctx.REPO / "outputs" / "E02a" / "capacity.csv")
    e02 = e02[(e02.region == "all") & (e02.arm == "em_bp") & (e02.components == 8)].drop_duplicates(["seed", "n_chains"])
    e56 = ctx._e56("kernels.csv")
    e56 = e56[(e56.failed == 0) & (e56.arm == "energy")].drop_duplicates(["seed", "n_chains"])
    k55 = _cat(ctx, "outputs/hpc_E55_kern_T0.3_h1_L32/kernels.csv")
    k55 = k55[(k55.selection == "risk") & (k55.seed < 8)].drop_duplicates(["seed", "n_chains"])
    r60 = ctx._core("read_e60")
    risk60, _ = r60.load()
    m33 = risk60[(risk60.cell == "T0.3_h1_L32") & (risk60.arm == "em_mixture") & (risk60.seed < 8)]
    m33 = m33.drop_duplicates(["seed", "n_chains"])
    lap = {"mix": med(e02, "risk"), "nk": med(e56, "risk_pooled"), "nk_clean": e58("laplace", "energy_h16"),
           "mlp": e58("laplace", "mlp"), "cnn": e58("laplace", "window"), "lin": e58("laplace", "lmmse")}
    cell = "T0.3_h1_L32"
    fld = {"mix": med(m33, "raw_risk"), "nk": med(k55, "risk"), "nk_clean": e58(cell, "energy_h16"),
           "mlp": e58(cell, "mlp"), "cnn": e58(cell, "window"), "lin": e58(cell, "lmmse")}
    return lap, fld


SERIES = (("mix", "mixture (EM)", "s", "-", True), ("nk", "neural kernel (EM)", "p", "-", True),
          ("nk_clean", "neural kernel (clean)", "p", "--", False), ("mlp", "global MLP", "o", "-", True),
          ("cnn", "window CNN", "^", "-", True))


def fig_results(ctx) -> None:
    from matplotlib.lines import Line2D
    C = _c(ctx)
    lap, fld = _results_data(ctx)
    fig, axes = ctx.figure(2, width=ctx.W["full"], height=2.55, sharey=True)
    ns = [32, 128, 512, 2048, 8192, 32768]
    for ax, data, title in zip(axes, (lap, fld), ("(a) Laplace chain", r"(b) Allen–Cahn, $\Theta=0.3$")):
        ax.axhline(float(data["lin"].median()), color=C["ink"], lw=0.9, ls=":")
        for key, _, m, ls, filled in SERIES:
            s = data[key]
            col = C["nk"] if key.startswith("nk") else C[key]
            ax.loglog(s.index, s.values, marker=m, ls=ls, color=col, lw=1.0, ms=4.5,
                      mfc=col if filled else "white", mec=col, mew=1.0)
        _log2_n(ctx, ax, ns)
        ax.set_xlabel(r"training sequences $N$")
        ax.set_title(title, loc="left")
    axes[0].set_ylim(1.8e-3, 0.9)
    ctx.plain_log_y(axes[0], [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    axes[0].set_ylabel("relative score error")
    handles = [Line2D([], [], color=(C["nk"] if k.startswith("nk") else C[k]), marker=m, ls=ls, lw=1.0, ms=4.5,
                      mfc=((C["nk"] if k.startswith("nk") else C[k]) if f else "white"), label=lab)
               for k, lab, m, ls, f in SERIES]
    handles.append(Line2D([], [], color=C["ink"], ls=":", lw=0.9, label="best linear"))
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, fontsize=8.5,
               handlelength=2.4, columnspacing=1.6)
    ctx.save(fig, "fig_results")
    for name, data in (("Laplace", lap), ("Allen-Cahn T0.3", fld)):
        print(f"    {name}: " + "  ".join(f"{k} " + " ".join(f"{v:.4f}" for v in data[k].reindex(ns).to_numpy())
                                         for k, *_ in SERIES) + f"  lin {float(data['lin'].median()):.4f}")


# ============================================================================ step efficiency (E65, E63) ==
def _curve_table(ctx, n: int = 2048):
    """E65's refit curves at N = n: test error and cumulative wall-clock at every checkpoint, one row per
    (arm, seed, checkpoint); the wall column is E63's per-update timing applied to E65's update counts."""
    pd = _pd()
    w = pd.read_csv(ctx.REPO / "reports" / "core" / "e63_curves_wall.csv")
    return w[(w.n_chains == n) & (w.failed == 0)]


def _ladder_median(d, arm: str, col: str):
    """Median over seeds of `col` at each checkpoint of the arm's ladder. A kernel fit that converged before the last
    ladder point keeps its final parameters, so its last value is carried forward (it would not change)."""
    a = d[d.arm == arm]
    pts = sorted(c for c in a.checkpoint.unique() if (c in LADDER) or c >= 500 and c % 500 == 0)
    rows = []
    for _, g in a.groupby("seed"):
        g = g.sort_values("checkpoint")
        rows.append([g[g.checkpoint <= c][col].iloc[-1] if (g.checkpoint <= c).any() else np.nan for c in pts])
    return np.array(pts, float), np.nanmedian(np.array(rows, float), axis=0)


STEP_ARMS = (("em_bp", "mix", "mixture (EM)", "s", "-", True, 1),
             ("energy", "nk", "neural kernel (EM)", "p", "-", True, NK_INNER),
             ("clean_energy", "nk", "neural kernel (clean)", "p", "--", False, NK_INNER),
             ("mlp", "mlp", "global MLP", "o", "-", True, 1),
             ("window", "cnn", "window CNN", "^", "-", True, 1))


def _step_figure(ctx, name: str, xcol: str | None) -> None:
    from matplotlib.lines import Line2D
    C = _c(ctx)
    d = _curve_table(ctx)
    fig, ax = ctx.figure(height=2.75)
    for arm, ck, _, m, ls, filled, scale in STEP_ARMS:
        pts, err = _ladder_median(d, arm, "test_all")
        if xcol is None:
            x = pts * scale
        else:
            _, x = _ladder_median(d, arm, xcol)
        col = C[ck]
        ax.loglog(x, err, marker=m, ls=ls, color=col, lw=1.0, ms=3.8, mfc=col if filled else "white", mec=col)
        print(f"    {name} {arm:13s} x {x[0]:.3g}..{x[-1]:.3g}  error {err[0]:.4f} -> min {np.nanmin(err):.4f} "
              f"at x={x[int(np.nanargmin(err))]:.3g}, last {err[-1]:.4f}")
    ax.set_ylim(0.005, 0.6)
    ctx.plain_log_y(ax, [0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    ax.set_xlabel("parameter updates" if xcol is None else "wall-clock seconds")
    ax.set_ylabel("relative score error")
    handles = [Line2D([], [], color=C[ck], marker=m, ls=ls, lw=1.0, ms=3.8, mfc=(C[ck] if f else "white"),
                      label=lab) for _, ck, lab, m, ls, f, _ in STEP_ARMS]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False, fontsize=8,
               handlelength=2.2, columnspacing=1.0)
    ctx.save(fig, name)


def fig_steps(ctx) -> None:
    _step_figure(ctx, "fig_steps", None)


def fig_wall(ctx) -> None:
    _step_figure(ctx, "fig_wall", "wall")


# ============================================================================================ appendix ==
NETS6 = (("mlp", "global MLP", "o"), ("window", "window CNN", "^"), ("tcn", "TCN", "v"), ("bigru", "BiGRU", "D"),
         ("transformer", "transformer", "<"), ("unet", "U-Net", ">"))


def fig_networks(ctx) -> None:
    """Every network of E03 against the two kernels learned by EM, Laplace chain, seed means (16 seeds, paired)."""
    from matplotlib.lines import Line2D
    C = _c(ctx)
    risk = ctx._paired_e03()
    ns = sorted(risk["em_bp"])
    mean = lambda a: np.array([np.mean(list(risk[a][n].values())) for n in ns])
    k = ctx._e56("kernels.csv")
    k = k[(k.failed == 0) & (k.arm == "energy")].drop_duplicates(["seed", "n_chains"])
    nk = k.groupby("n_chains").risk_pooled.mean().reindex(ns).to_numpy()
    greys = ("#3b3b3b", "#6b6b6b", "#8f8f8f", "#b0b0b0")
    fig, ax = ctx.figure(height=2.9)
    handles = []
    gi = 0
    for arch, lab, m in NETS6:
        col = C["mlp"] if arch == "mlp" else C["cnn"] if arch == "window" else greys[gi]
        gi += arch not in ("mlp", "window")
        ax.loglog(ns, mean(arch), marker=m, color=col, lw=1.0, ms=4.0)
        handles.append(Line2D([], [], color=col, marker=m, lw=1.0, ms=4.0, label=lab))
    ax.loglog(ns, mean("em_bp"), "s-", color=C["mix"], lw=1.0, ms=4.0)
    ax.loglog(ns, nk, "p-", color=C["nk"], lw=1.0, ms=4.5)
    handles += [Line2D([], [], color=C["mix"], marker="s", lw=1.0, label="mixture (EM)"),
                Line2D([], [], color=C["nk"], marker="p", lw=1.0, label="neural kernel (EM)")]
    _log2_n(ctx, ax, ns)
    ax.set_ylim(0.015, 0.8)
    ctx.plain_log_y(ax, [0.02, 0.05, 0.1, 0.2, 0.5])
    ax.set_xlabel(r"training sequences $N$")
    ax.set_ylabel("relative score error")
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, fontsize=7.5, columnspacing=0.8,
               handlelength=1.8)
    ctx.save(fig, "fig_networks")


def _nonmarkov_data(ctx):
    import fig_ch10 as c10
    adv = c10._nonmarkov()
    betas, gammas = [0.1, 0.25, 0.5, 1.0], [0.05, 0.1, 0.2, 0.4]
    kl_b = [c10.kl_per_site(c10.global_cov(x), c10.CHOW_LIU[("global", x)]) for x in betas]
    kl_g = [c10.kl_per_site(c10.longrange_cov(x), c10.CHOW_LIU[("longrange", x)]) for x in gammas]
    r62 = ctx._core("read_e62")
    arms, _ = r62.load()
    cal = r62.calibration()
    return adv, betas, gammas, kl_b, kl_g, r62, arms, cal


def fig_nonmarkov_delta(ctx) -> None:
    """Network error over first-order EM-BP error against delta, the distance per site to the closest first-order
    chain, N = 2048. One point per non-Markov data set: a global latent variable (Gaussian and Laplace chains),
    long-range couplings (thesis exp_21_clean) and second-order Laplace chains (E62). One colour per network."""
    from matplotlib.lines import Line2D
    C = _c(ctx)
    adv, betas, gammas, kl_b, kl_g, r62, arms, cal = _nonmarkov_data(ctx)
    fig, ax = ctx.figure(height=2.55)
    series = ((("gauss", "global"), betas, kl_b), (("laplace", "global"), betas, kl_b),
              (("gauss", "longrange"), gammas, kl_g))
    phis = [p for p in r62.PHI2 if p > 0]
    d62 = [cal[p]["delta_exact"] for p in phis]
    for net, col in (("cnn", C["cnn"]), ("mlp", C["mlp"])):
        xs_all, ys_all = [], []
        for key, xs, kl in series:
            xs_all += list(kl)
            ys_all += [adv[(key[0], key[1], s_)][net] for s_ in xs]
        net62 = "window" if net == "cnn" else "mlp"
        xs_all += d62
        ys_all += [r62.seed_mean(arms, net62, 2048, p) / r62.seed_mean(arms, "first", 2048, p) for p in phis]
        ax.loglog(xs_all, ys_all, "o", color=col, ms=4.5, ls="none")
    ax.axhline(1.0, color=C["ink"], lw=0.8, ls="--")
    ax.text(2e-6, 1.25, "kernel better", color=C["grey"], va="bottom", ha="left")
    ax.text(2e-6, 0.8, "network better", color=C["grey"], va="top", ha="left")
    ax.set_xlim(1.2e-6, 0.6)
    ax.set_ylim(0.3, 60)
    ctx.plain_log_y(ax, [0.5, 1, 2, 5, 10, 20, 50])
    ax.set_xlabel(r"distance from a first-order chain $\delta$")
    ax.set_ylabel("network error / kernel error")
    handles = [Line2D([], [], color=C["cnn"], marker="o", ls="none", label="window CNN"),
               Line2D([], [], color=C["mlp"], marker="o", ls="none", label="global MLP")]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False, fontsize=8, columnspacing=1.2)
    ctx.save(fig, "fig_nonmarkov_delta")


def fig_nonmarkov_ar2(ctx) -> None:
    """E62: error against phi2 at N = 2048, first-order EM-BP, pair-state EM-BP, the best network and the linear
    denoiser of the true covariance; the line is the exact error of the best first-order kernel (calibration)."""
    from matplotlib.lines import Line2D
    C = _c(ctx)
    _, _, _, _, _, r62, arms, cal = _nonmarkov_data(ctx)
    P = list(r62.PHI2)
    sm = lambda a: [r62.seed_mean(arms, a, 2048, p) for p in P]
    best = [min(r62.seed_mean(arms, a, 2048, p) for a in r62.NETS) for p in P]
    fig, ax = ctx.figure(height=2.6)
    ax.plot(P, [cal[p]["first_order_floor"] for p in P], "-", color=C["mix"], lw=1.0)
    ax.plot(P, sm("first"), "o", color=C["mix"], ms=4.5, ls="none")
    ax.plot(P, sm("pair"), "s", color=C["nk"], ms=4.5, ls="none")
    ax.plot(P, best, "D", color=C["grey"], ms=4.0, ls="none")
    ax.plot(P, sm("lmmse"), "--", color=C["ink"], lw=0.8)
    ax.set_xlim(-0.36, 0.66)
    ax.set_ylim(0, 0.37)
    ax.set_xlabel(r"second-order coefficient $\phi_2$")
    ax.set_ylabel("relative score error")
    handles = [Line2D([], [], color=C["mix"], marker="o", ls="none", label="first-order kernel"),
               Line2D([], [], color=C["mix"], ls="-", label="best first-order"),
               Line2D([], [], color=C["nk"], marker="s", ls="none", label="BP on pairs"),
               Line2D([], [], color=C["grey"], marker="D", ls="none", label="best network"),
               Line2D([], [], color=C["ink"], ls="--", label="best linear")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, fontsize=8, columnspacing=1.0)
    ctx.save(fig, "fig_nonmarkov_ar2")


def fig_field(ctx) -> None:
    """The Allen-Cahn field. (a) The double-well potential. (b) One exact sample at Theta = 0.3 and one at Theta = 2.
    (c) Single-site law of the standardised field at the four temperatures. (d) The exact kernel K(a'|a) at
    Theta = 0.3 for three current states: its shape, not only its centre, changes with a. Closed form or exact
    sampling (E32 cells)."""
    import json
    fs = ctx.fs
    C = _c(ctx)
    sys.path.insert(0, str(ctx.REPO / "src"))
    from markov_score.field import Phi4Chain
    from markov_score.utils import rng_for
    with open(ctx.REPO / "outputs" / "E32" / "cells.json") as fh:
        cells = {c["cell"]: c for c in json.load(fh)}
    tcells = (("T0.3_h1_L32", "0.3"), ("T0.5_iso_L32", "0.5"), ("T1_iso_L32", "1"), ("T2_iso_L32", "2"))
    ramp = (fs.BLUE_RAMP[4], fs.BLUE_RAMP[3], fs.BLUE_RAMP[2], fs.BLUE_RAMP[0])
    fig, (a0, a1, a2, a3) = ctx.figure(4, width=ctx.W["full"], height=1.95)
    phi = np.linspace(-1.7, 1.7, 400)
    a0.plot(phi, 0.25 * (phi ** 2 - 1) ** 2, color=C["ink"], lw=1.2)
    a0.set_xlim(phi[0], phi[-1]); a0.set_ylim(0, 0.75)
    a0.set_xlabel(r"$\phi$"); a0.set_ylabel(r"$V(\phi)$")
    chains = {cid: Phi4Chain(cells[cid]["T"], cells[cid]["h"]) for cid, _ in tcells}
    for (cid, lab), col in ((tcells[0], ramp[0]), (tcells[3], ramp[3])):
        A = chains[cid].sample_batch(rng_for(f"fig-field-chain-{cid}"), 2, 128)
        a1.plot(np.arange(128), A[0], color=col, lw=0.9, label=rf"$\Theta={lab}$")
    a1.set_xlim(0, 127); a1.set_ylim(-3.0, 4.6)
    a1.set_xlabel(r"site $i$"); a1.set_ylabel(r"$a_i$")
    a1.legend(loc="upper center", frameon=False, fontsize=7.5, handlelength=1.2, ncol=2, columnspacing=0.8)
    g = np.linspace(-3.0, 3.0, 801)
    for (cid, lab), col in zip(tcells, ramp):
        a2.plot(g, np.exp(chains[cid].log_stationary(g)), color=col, label=rf"$\Theta={lab}$")
    a2.set_xlim(g[0], g[-1]); a2.set_ylim(0, 0.75)
    a2.set_xlabel(r"$a$"); a2.set_ylabel(r"$\pi(a)$")
    a2.legend(loc="upper center", frameon=False, fontsize=7, handlelength=1.0, ncol=2, columnspacing=0.6)
    base = chains["T0.3_h1_L32"]
    gg = np.linspace(-3.0, 3.0, 601)
    logk = base.log_transition_matrix(gg)                  # [out, in]
    for a_now, col, ls in ((0.0, C["nk"], "-"), (0.6, C["mix"], "--"), (1.2, C["grey"], "-.")):
        j = int(np.argmin(np.abs(gg - a_now)))
        a3.plot(gg, np.exp(logk[:, j]), color=col, ls=ls, label=rf"$a={a_now:g}$")
    a3.set_xlim(-2.6, 2.6); a3.set_ylim(bottom=0)
    a3.set_xlabel(r"$a'$"); a3.set_ylabel(r"$K(a'\,|\,a)$")
    a3.legend(loc="upper left", frameon=False, fontsize=7.5, handlelength=1.6)
    for ax, letter in zip((a0, a1, a2, a3), "abcd"):
        ctx.panel(ax, letter)
    ctx.save(fig, "fig_field")


FAMILY_ROWS = (("mixture", "mixture", "mix"), ("fourier", "Fourier", "mix"), ("spline", "B-spline", "mix"),
               ("rbf", "RBF", "mix"), ("legendre", "Legendre", "mix"), ("energy", "neural kernel", "nk"),
               ("loglinear2d", "log-linear 2-D", "nk"), ("best", "best network", "grey"))


def fig_families_error(ctx) -> None:
    """Every kernel family learned by EM on the Laplace chain, and the best of the six networks: relative score error at
    N = 128 (open) and N = 2048 (filled), seed means over 16 seeds, on the same test sequences (E02a, E56, E03).
    Blue: families that assume the shift form; indigo: families that do not; grey: networks. Dotted: best linear."""
    from matplotlib.lines import Line2D
    C = _c(ctx)
    risk = ctx._paired_e03()
    k = ctx._e56("kernels.csv")
    k = k[k.failed == 0].drop_duplicates(["seed", "n_chains", "arm"])
    nets = ("mlp", "window", "bigru", "unet", "transformer", "tcn")
    mean_net = lambda a, n: float(np.mean(list(risk[a][n].values())))

    def value(arm: str, n: int) -> float:
        if arm == "mixture":
            return mean_net("em_bp", n)
        if arm == "best":
            return min(mean_net(a, n) for a in nets)
        return float(k[(k.arm == arm) & (k.n_chains == n)].risk_pooled.mean())

    r58 = ctx._core("read_e58")
    d58 = r58.load()
    lin = float(d58[(d58.cell == "laplace") & (d58.arm == "lmmse")].risk.mean())
    fig, ax = ctx.figure(height=2.6)
    rows = list(FAMILY_ROWS)
    for i, (arm, _, ck) in enumerate(rows):
        y = len(rows) - 1 - i
        lo, hi = value(arm, 128), value(arm, 2048)
        col = C[ck]
        ax.plot([lo, hi], [y, y], "-", color=col, lw=1.0)
        ax.plot([lo], [y], "o", color=col, mfc="white", ms=5)
        ax.plot([hi], [y], "o", color=col, ms=5)
        print(f"    {arm:12s} N=128 {lo:.4f}  N=2048 {hi:.4f}")
    ax.axvline(lin, color=C["ink"], lw=0.9, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(0.015, 0.4)
    ax.set_xticks([0.02, 0.05, 0.1, 0.2])
    ax.set_xticklabels(["0.02", "0.05", "0.1", "0.2"])
    ax.xaxis.set_minor_formatter(ctx.matplotlib.ticker.NullFormatter())
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for _, lab, _ in rows][::-1])
    ax.set_yticks([], minor=True)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("relative score error")
    handles = [Line2D([], [], color=C["ink"], marker="o", mfc="white", ls="none", label=r"$N=128$"),
               Line2D([], [], color=C["ink"], marker="o", ls="none", label=r"$N=2048$"),
               Line2D([], [], color=C["ink"], ls=":", lw=0.9, label="best linear")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, fontsize=8, columnspacing=1.2)
    ctx.save(fig, "fig_families_error")


NAMES = ("fig_factor_graph", "fig_em_loop", "fig_routes", "fig_closure", "fig_results", "fig_steps", "fig_wall",
         "fig_networks", "fig_nonmarkov_delta", "fig_nonmarkov_ar2", "fig_field", "fig_families_error")
