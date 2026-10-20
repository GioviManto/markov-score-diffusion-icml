"""The clean figure set after the 19 Oct 2026 call with Jerome. One function per figure, frozen data only.

    python tools/core/figures.py [names...]      # all figures by default

Writes reports/core/figures/<name>.pdf (+ .png) and prints every number it draws.

  summary     (a) |Q_t| of the noisy Gaussian chain at t = 0, t*, 1.5; (b) Laplace chain: EM-BP vs a CNN against M;
              (c) phi^4 field: site energy, neural kernel in BP and CNN against M            [E30, E03, E28, E33, E54]
  grid        relative score error of grid BP against the number of grid points                [E39, E47a]
  laplace     (a) the three estimators against M; (b) error per noise level at M = 128           [E30, E03, E28]
  field       (a) potential read off site-energy and MDN kernels; (b) temperature read off both  [E51]
  speciation  (a) polarisation map of one chain; (b) commitment time against distance to a kink  [E53]

Scores are compared through the pooled relative score error (docs/METRICS.md section 1) divided by the linear floor
of the same cell (the best linear estimator, = Gaussian closure): 1 means no gain over a linear denoiser.
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
from matplotlib.colors import LogNorm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import figstyle as fs  # noqa: E402

OUT = ROOT / "reports" / "core" / "figures"
BASE = "T0.3_h1_L32"
SIZES = (32, 128, 512, 2048)
C_BP, C_KER, C_NET, C_FLOOR = fs.CAT[0], fs.CAT[2], fs.CAT[1], fs.INK2


def _csvs(pattern):
    return pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(ROOT / pattern)))], ignore_index=True)


def _mean_se(df, key="ratio"):
    g = df.groupby("n_chains")[key]
    return g.mean(), g.std(ddof=1) / np.sqrt(g.count())


def _m_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(m) for m in SIZES])
    ax.set_xlabel(r"training sequences $M$")
    ax.minorticks_off()


# ------------------------------------------------------------------------------------------------ data


def laplace_rows():
    """Per-seed risk / floor on E03's bundles (rho = 0.85, L = 32, Laplace driving noise)."""
    floor = float(pd.read_csv(ROOT / "outputs/hpc_E28/per_level.csv").query("arm == 'lmmse'").groupby("seed").risk_pooled.first().mean())
    e30 = _csvs("outputs/hpc_E30_s*/joint.csv").groupby(["arm", "n_chains", "seed"]).risk_pooled.first().reset_index()
    e30 = e30.rename(columns={"risk_pooled": "risk"})
    nets = pd.read_csv(ROOT / "outputs/E03/confirm.csv").query("region == 'all'")
    cnn = nets[nets.arch == "window"][["n_chains", "seed", "risk"]].assign(arm="cnn")
    rows = pd.concat([e30, cnn], ignore_index=True)
    rows["ratio"] = rows.risk / floor
    best_net = nets.groupby(["arch", "n_chains"]).risk.mean().unstack().min(axis=0)
    return rows, floor, best_net


def field_rows():
    """Per-seed risk / floor on E33's bundles, every cell: site energy, spline, MDN, CNN (standard DSM, E54)."""
    floor = pd.read_csv(ROOT / "outputs/hpc_E33o/oracles.csv").query("arm == 'lmmse_true'").groupby("cell").risk.mean()
    k = pd.concat([_csvs("outputs/hpc_E33k_s*/kernels.csv"), _csvs("outputs/hpc_E33q_s*/kernels.csv")], ignore_index=True)
    k = k[(k.selection == "risk") & k.arm.isin(["em_pair_quartic", "em_tilted_spline", "em_mdn"])][["cell", "arm", "n_chains", "seed", "risk"]]
    parts = [k]
    if (ROOT / "outputs/E54/nets.csv").exists():
        parts.append(pd.read_csv(ROOT / "outputs/E54/nets.csv")[["cell", "n_chains", "seed", "risk"]].assign(arm="cnn"))
    rows = pd.concat(parts, ignore_index=True)
    rows["ratio"] = rows.risk / rows.cell.map(floor)
    return rows, floor


# ------------------------------------------------------------------------------------------------ panels


def panel_precision(fig, spec, rho=0.85, L=24):
    idx = np.arange(L)
    S0 = rho ** np.abs(idx[:, None] - idx[None, :])
    Q = lambda t: np.linalg.inv(np.exp(-2 * t) * S0 + (1 - np.exp(-2 * t)) * np.eye(L))  # noqa: E731
    dist = np.abs(idx[:, None] - idx[None, :])
    ts = np.geomspace(1e-3, 3, 4000)
    frob = lambda M, sel: float(np.sqrt(np.sum(M[sel] ** 2)))  # noqa: E731  (the thesis's dist_mass, fig_ch05.py)
    ref = frob(Q(0.0), dist == 1)
    far = np.array([frob(Q(t), dist >= 2) / ref for t in ts])
    t_star = float(ts[np.argmax(far)])
    print(f"  precision: t* = {t_star:.3f} (largest Frobenius mass beyond the first off-diagonal, {far.max():.3f} of Q_0's neighbour coupling)")
    sub = spec.subgridspec(1, 3, wspace=0.08)
    axes = []
    for j, (t, title) in enumerate([(0.0, r"$t=0$"), (t_star, r"$t=t^\star$"), (1.5, r"$t=1.5$")]):
        ax = fig.add_subplot(sub[0, j])
        ax.imshow(np.abs(Q(t)), cmap="Greys", norm=LogNorm(1e-3, 5.0), interpolation="nearest")   # one scale for all three
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.4); s.set_color(fs.INK2)
        ax.set_title(title, pad=2)
        axes.append(ax)
    return axes


def panel_efficiency(ax, rows, arms, title=None):
    for arm, label, color, ls, marker in arms:
        sub = rows[rows.arm == arm]
        if sub.empty:
            continue
        m, se = _mean_se(sub)
        ax.errorbar(m.index, m.values, yerr=se.values, color=color, ls=ls, marker=marker, capsize=0, lw=1.3, label=label)
        print(f"    {label:28s} " + "  ".join(f"M={n}: {v:.3f}" for n, v in m.items()))
    ax.axhline(1.0, color=C_FLOOR, ls="--", lw=0.9, label="best linear")
    ax.set_yscale("log")
    ticks = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_ylabel(r"score error / linear floor")
    _m_axis(ax)
    fs.ygrid(ax)
    if title:
        ax.set_title(title, pad=3)


# ------------------------------------------------------------------------------------------------ figures


def summary():
    lap, lap_floor, _ = laplace_rows()
    fld, fld_floor = field_rows()
    fig = plt.figure(figsize=(fs.FULL_W, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 1], wspace=0.42, bottom=0.33, top=0.9)
    pa = panel_precision(fig, gs[0, 0])
    fs.panel_label(pa[0], "(a)", x=-0.25, y=1.12)
    print(f"  (b) Laplace chain, floor {lap_floor:.4f}")
    ax = fig.add_subplot(gs[0, 1])
    panel_efficiency(ax, lap, [("em_bp", "EM-BP, 25 par.", C_BP, "-", "o"), ("cnn", "CNN, DSM", C_NET, "-", "s")])
    fs.panel_label(ax, "(b)")
    print(f"  (c) phi^4 base cell, floor {fld_floor[BASE]:.4f}")
    ax2 = fig.add_subplot(gs[0, 2], sharey=ax)
    panel_efficiency(ax2, fld[fld.cell == BASE], [("em_pair_quartic", "site energy, 5 par.", C_BP, "-", "o"),
                                                 ("em_mdn", "neural kernel", C_KER, "--", "^"), ("cnn", "CNN, DSM", C_NET, "-", "s")])
    ax2.set_ylabel("")
    fs.panel_label(ax2, "(c)")
    h, lab = [], []
    for a in (ax, ax2):
        for hh, ll in zip(*a.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh); lab.append(ll)
    fig.legend(h, lab, loc="lower center", ncol=5, bbox_to_anchor=(0.6, 0.0), handlelength=1.8, columnspacing=1.2)
    fs.save(fig, OUT / "fig1-summary.pdf")


def grid():
    e39 = pd.read_csv(ROOT / "outputs/E39/sanity.csv").query("part == 'grid' and arm == 'grid_bp' and half_width == 8.0")
    g = e39.groupby("n_grid").err_closed_form.max()
    e47 = pd.read_csv(ROOT / "outputs/E47a/refinement.csv").query("cell == @BASE")
    f = e47.groupby("grid_size").rel_error.max()
    fig, ax = plt.subplots(figsize=(fs.COL_W, 2.0))
    ax.plot(g.index, g.values, color=C_BP, marker="o", label="Gaussian chain")
    ax.plot(f.index, f.values, color=C_NET, marker="s", label=r"$\phi^4$ chain")
    ax.axvline(401, color=C_FLOOR, ls=":", lw=0.9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"grid points $N_g$")
    ax.set_ylabel("relative score error")
    ax.legend(loc="lower left")
    fs.ygrid(ax)
    print("  Gaussian vs closed form: " + ", ".join(f"{n}: {v:.1e}" for n, v in g.items()))
    print("  phi^4 vs 1601 points:    " + ", ".join(f"{n}: {v:.1e}" for n, v in f.items()))
    fs.save(fig, OUT / "fig2-grid.pdf")


def laplace():
    lap, floor, best = laplace_rows()
    print(f"  floor {floor:.4f}; best of six networks per M: " + ", ".join(f"{n}: {v:.4f}" for n, v in best.items()))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(fs.FULL_W, 2.1), gridspec_kw=dict(wspace=0.32))
    panel_efficiency(ax, lap, [("em_bp", "EM-BP, 25 par.", C_BP, "-", "o"), ("mdn", "neural kernel", C_KER, "--", "^"),
                               ("cnn", "CNN, DSM", C_NET, "-", "s")])
    ax.legend(loc="lower left")
    fs.panel_label(ax, "(a)")
    pl = pd.read_csv(ROOT / "outputs/hpc_E28/per_level.csv").query("region == 'all' and n_chains == 128")
    for arm, label, color, ls in [("em_bp", "EM-BP", C_BP, "-"), ("net:window", "CNN", C_NET, "-"), ("lmmse", "best linear", C_FLOOR, "--")]:
        s = pl[pl.arm == arm].groupby("t").risk_level.mean()
        bx.plot(s.index, s.values, color=color, ls=ls, marker="o" if arm != "lmmse" else None, label=label)
        print(f"    per level, {label}: " + ", ".join(f"{t:.2f}: {v:.3f}" for t, v in s.items()))
    bx.set_xscale("log"); bx.set_yscale("log")
    bx.set_xlabel(r"diffusion time $t$")
    bx.set_ylabel("relative score error")
    bx.legend(loc="lower left")
    fs.ygrid(bx)
    fs.panel_label(bx, "(b)")
    fs.save(fig, OUT / "fig3-laplace.pdf")


def field():
    from scipy.interpolate import CubicSpline
    sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
    import e33_field_learning as e33
    from e34_field_physics import kernel_from_row
    from markov_score.bp_grid import make_grid
    from markov_score.field import potential
    from markov_score.kernel_readings import bond_reading, site_energy_from_diagonal
    rd = pd.read_csv(ROOT / "outputs/E51/readings.csv")
    k = pd.concat([_csvs("outputs/hpc_E33k_s*/kernels.csv"), _csvs("outputs/hpc_E33q_s*/kernels.csv")], ignore_index=True)
    k = k[(k.selection == "risk") & (k.cell == BASE) & (k.n_chains == 2048) & k.arm.isin(["em_pair_quartic", "em_mdn"])]
    grid, w = make_grid(8.0, 401)
    phi = np.linspace(-2.0, 2.0, 401)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(fs.FULL_W, 2.1), gridspec_kw=dict(wspace=0.32))
    for arm, color, z in (("em_mdn", C_KER, 2), ("em_pair_quartic", C_BP, 3)):
        for _, r in k[k.arm == arm].iterrows():
            kern = kernel_from_row(r, dict(e33.SETTINGS), grid)
            lk = kern.log_transition_matrix(grid)
            with np.errstate(invalid="ignore", divide="ignore"):
                theta = bond_reading(lk, grid, w)["gamma_hat"] * float(r["scale"]) ** 2 / float(r["h"])
            f = site_energy_from_diagonal(lk)
            fin = np.isfinite(f)
            v = (2 * theta / float(r["h"])) * CubicSpline(grid[fin], f[fin])(phi / float(r["scale"]))
            inner = np.abs(phi) <= 1.6
            v = v - np.mean((v - potential(phi))[inner])
            ax.plot(phi, v, color=color, lw=0.6, alpha=0.55, zorder=z)
    ax.plot(phi, potential(phi), color=fs.INK, ls="--", lw=1.2, zorder=4)
    ax.plot([], [], color=C_BP, label="site energy"); ax.plot([], [], color=C_KER, label="neural kernel")
    ax.plot([], [], color=fs.INK, ls="--", label=r"true $V$")
    ax.set_ylim(-0.1, 0.9)
    ax.set_xlabel(r"$\phi$"); ax.set_ylabel(r"$\hat V(\phi)$")
    ax.legend(loc="upper center", ncol=1, handlelength=1.6)
    fs.ygrid(ax)
    fs.panel_label(ax, "(a)")
    for j, (arm, color, marker) in enumerate((("em_pair_quartic", C_BP, "o"), ("em_mdn", C_KER, "^"))):
        sub = rd[(rd.arm == arm) & (rd.cell == BASE)]
        med = sub.groupby("n_chains").theta_ratio.median()
        jit = 1.0 + (j - 0.5) * 0.12
        bx.scatter(sub.n_chains * jit, sub.theta_ratio, s=5, color=color, alpha=0.35, lw=0, marker=marker)
        bx.plot(med.index * jit, med.values, color=color, marker=marker, label="site energy" if j == 0 else "neural kernel")
        print(f"    Theta_hat/Theta median, {arm}: " + ", ".join(f"{n}: {v:.3f}" for n, v in med.items()))
    bx.axhline(1.0, color=fs.INK, ls="--", lw=0.9)
    bx.set_ylabel(r"$\hat\Theta/\Theta$")
    _m_axis(bx)
    bx.legend(loc="upper right")
    fs.ygrid(bx)
    fs.panel_label(bx, "(b)")
    fs.save(fig, OUT / "fig4-field-readings.pdf")


def speciation():
    z = np.load(ROOT / "outputs/E53/maps.npz")
    lv = z["levels"]
    df = pd.read_csv(ROOT / "outputs/E53/sites.csv", keep_default_na=False)
    df["a"] = df.a.astype(float); df["tau"] = df.tau.astype(float)
    cell = "T0.2_h1_L256"
    a = df[(df.cell == cell) & (df.chain == 0)].sort_values("site").a.values
    tau = df[(df.cell == cell) & (df.chain == 0)].sort_values("site").tau.values
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(fs.FULL_W, 2.2), gridspec_kw=dict(width_ratios=[2.2, 1], wspace=0.28))
    m = z[cell]
    ax.pcolormesh(np.arange(len(a)), lv, m, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto", rasterized=True)
    ax.plot(np.arange(len(a)), np.where(tau > 0, tau, np.nan), "k.", ms=1.2)
    for kk in np.flatnonzero(np.sign(a[1:]) != np.sign(a[:-1])) + 0.5:
        ax.axvline(kk, color=fs.INK, lw=0.4, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel(r"site $i$"); ax.set_ylabel(r"diffusion time $t$")
    fs.panel_label(ax, "(a)", x=-0.09)
    for cell_, color, label in (("T0.2_h1_L256", C_BP, r"$\Theta=0.2$"), ("T0.3_h1_L256", C_NET, r"$\Theta=0.3$")):
        d = df[(df.cell == cell_) & (df.domain != "") & (df.a.abs() >= 0.5)].copy()
        parts = []
        for _, g in d.groupby("domain"):
            g = g.sort_values("site")
            n = len(g)
            parts.append(pd.DataFrame({"dist": np.minimum(np.arange(n), np.arange(n)[::-1]), "tau": g.tau.values}))
        o = pd.concat(parts)
        o = o[o.dist <= 12]
        med = o.groupby("dist").tau.median()
        bx.plot(med.index, med.values, color=color, marker="o", label=label)
        print(f"    {cell_} median tau by distance to kink: " + ", ".join(f"{i}: {v:.2f}" for i, v in med.items()))
    bx.set_xlabel("distance to nearest kink"); bx.set_ylabel(r"commitment time $\tau$")
    bx.legend(loc="lower right")
    fs.ygrid(bx)
    fs.panel_label(bx, "(b)")
    fs.save(fig, OUT / "fig5-speciation.pdf")


FIGURES = {"summary": summary, "grid": grid, "laplace": laplace, "field": field, "speciation": speciation}


def main(argv) -> int:
    fs.apply()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (argv or list(FIGURES)):
        print(f"[{name}]")
        FIGURES[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
