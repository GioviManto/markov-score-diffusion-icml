#!/usr/bin/env python
"""Figure: the coarse grid (E47) -- what a 4x smaller quadrature grid costs in time and in risk.

    python tools/make_fig_coarse_grid.py                       # paper/figures/fig_grid_coarse.pdf (+png)
    python tools/make_fig_coarse_grid.py --out outputs/scratch/coarse   # dry run, nothing in paper/

Two panels, full width, frozen outputs only:

(a) Belief-propagation time per sequence (ms) against the number of grid points N_g = 41..401 (log-log), from
    outputs/hpc_E47t/scaling.csv (arm bp_grid, factor grid_size, batch 512, 32 sites; wall_median / 512; one BLAS
    thread on an exclusive node). Dashed horizontal references are three networks' forward passes at the same batch
    and thread count (the same file, factor == batch, value == 512). N_g = 101 (the coarse grid) and N_g = 401 (the
    frozen operating grid of E33) are marked.
(b) Risk at N_g relative to the 401-point fit, drawn as |seed-mean ratio - 1| on a log axis:
      site energy (em_pair_quartic) and tilted spline (em_tilted_spline), base cell T0.3_h1_L32, M = 2048: E47b
        (outputs/hpc_E47b_s*/kernels.csv, seeds 0-7, N_g = 51, 101, 201) against E33's frozen N_g = 401 fits on the
        same seeds and the same test bundle (outputs/hpc_E33q_s*, hpc_E33k_s*; selection == 'risk') -- the pairing
        tools/make_extension_numbers.read_e47_learned reads;
      the linear chain's headline kernels with C = 8 and C = 1 components, M = 2048: E47c (outputs/E47c/refinement.csv,
        columns risk_em_C8, risk_em_C1, 16 seeds, N_g = 51, 101, 151, 201), ratio of seed means to N_g = 401.
    A ratio BELOW 1 (the coarse fit is the better one) is drawn with a hollow marker. A deviation under FLOOR = 1e-4 is
    drawn on the floor line, not omitted, and the four series are dodged horizontally by DODGE_PT points (display
    space; the data x are exact) so that the markers that share the floor at N_g = 201 stay visible.

House rules: tools/figstyle (no prose in a figure, symbols in math mode, one width per placement).
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import figstyle as fs  # noqa: E402

FLOOR = 1e-4                       # a deviation under this is drawn ON the floor line
DODGE_PT = 3.9                     # horizontal offset between adjacent series in panel (b), in points
BATCH, N_SITES = 512, 32
OPERATING, COARSE = 401, 101
BASE_CELL, M_LEARNED = "T0.3_h1_L32", 2048
NETWORKS = (("transformer", "--"), ("bigru", "-."), ("window", ":"))
LEARNED_ARMS = (("em_pair_quartic", "site energy", "q"), ("em_tilted_spline", "spline tilt", "k"))
HEADLINE_ARMS = (("risk_em_C8", r"mixture $C=8$"), ("risk_em_C1", r"mixture $C=1$"))


# --------------------------------------------------------------------------- data
def _load(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(str(ROOT / pattern)))
    if not paths:
        raise FileNotFoundError(pattern)
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def bp_ms_per_sequence(scaling: pd.DataFrame) -> pd.Series:
    """Grid-BP time per sequence in ms, indexed by the number of grid points."""
    b = scaling[(scaling.arm == "bp_grid") & (scaling.factor == "grid_size")
                & (scaling.batch == BATCH) & (scaling.n_sites == N_SITES)]
    return (b.groupby("grid_size").wall_median.median() / BATCH * 1000.0).sort_index()


def network_ms_per_sequence(scaling: pd.DataFrame) -> dict[str, float]:
    """Forward-pass time per sequence in ms of each reference network at batch 512."""
    out = {}
    for arm, _ in NETWORKS:
        r = scaling[(scaling.arm == arm) & (scaling.factor == "batch") & (scaling.value == BATCH)]
        if r.empty:
            raise ValueError(f"no batch-{BATCH} row for {arm} in the scaling file")
        out[arm] = float(r.wall_median.median() / BATCH * 1000.0)
    return out


def deviation(ratio: float) -> tuple[float, bool]:
    """(|ratio - 1| clipped up to FLOOR, ratio < 1): what is drawn, and whether the marker is hollow."""
    return max(abs(ratio - 1.0), FLOOR), bool(ratio < 1.0)


def learned_ratios(d47: pd.DataFrame, d33: pd.DataFrame, arm: str, cell: str = BASE_CELL, n_chains: int = M_LEARNED) -> pd.Series:
    """Seed-mean risk at each N_g over E33's seed-mean risk at N_g = 401, on E47's seeds (the pairing of
    make_extension_numbers.read_e47_learned). A different test bundle, a failed fit or a missing seed raises."""
    a_all = d47[(d47.arm == arm) & (d47.selection == "risk") & (d47.cell == cell) & (d47.n_chains == n_chains)]
    b = d33[(d33.arm == arm) & (d33.selection == "risk") & (d33.cell == cell) & (d33.n_chains == n_chains)].set_index("seed")
    ratios = {}
    for ng, a in a_all.groupby("grid_size"):
        a = a.set_index("seed").sort_index()
        ref = b.reindex(a.index)
        if ref.risk.isna().any() or a.risk.isna().any():
            raise ValueError(f"{arm} N_g={ng}: a missing or failed fit; the paired reading is not valid")
        if (a.test_bundle.values != ref.test_bundle.values).any():
            raise ValueError("E47 and E33 were scored on different test bundles; the paired reading is not valid")
        ratios[int(ng)] = float(a.risk.mean() / ref.risk.mean())
    if not ratios:
        raise ValueError(f"no E47b rows for {arm} / {cell} / M={n_chains}")
    return pd.Series(ratios).sort_index()


def headline_ratios(d47c: pd.DataFrame, col: str, n_chains: int = M_LEARNED) -> pd.Series:
    """Seed-mean headline-kernel risk at each N_g over its seed-mean at N_g = 401 (E47c, half-width 8)."""
    d = d47c[(d47c.n_chains == n_chains) & (d47c.half_width == 8.0)]
    m = d.groupby("n_grid")[col].mean()
    if OPERATING not in m.index or m.isna().any():
        raise ValueError(f"E47c {col}: no N_g = {OPERATING} reference or a missing risk")
    return (m / m[OPERATING]).drop(OPERATING).sort_index()


# --------------------------------------------------------------------------- drawing
def _plain(axis, values, fmt="{:g}"):
    """Explicit major ticks with plain labels and no minor labels (a log axis otherwise prints 6x10^-1 clutter)."""
    axis.set_major_locator(FixedLocator(values))
    axis.set_major_formatter(FixedFormatter([fmt.format(v) for v in values]))
    axis.set_minor_formatter(NullFormatter())


def panel_time(ax, bp: pd.Series, nets: dict[str, float]) -> None:
    ax.plot(bp.index, bp.values, color=fs.INK, lw=1.3, marker="o", ms=2.4, zorder=3)
    ax.plot([OPERATING], [bp[OPERATING]], marker="s", ms=5.5, mfc="white", mec=fs.INK, mew=1.3, ls="none", zorder=5)
    ax.plot([COARSE], [bp[COARSE]], marker="o", ms=5.5, mfc=fs.CAT[1], mec=fs.INK, mew=0.8, ls="none", zorder=5)
    for (arm, ls) in NETWORKS:
        ax.axhline(nets[arm], color=fs.MUTED, lw=0.9, ls=ls, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(36, 480); ax.set_ylim(0.014, 1.15)
    _plain(ax.xaxis, [41, 101, 201, 401]); _plain(ax.yaxis, [0.02, 0.05, 0.1, 0.2, 0.5, 1])
    ax.set_xlabel(r"grid points $N_{\mathrm{g}}$"); ax.set_ylabel("BP time per sequence (ms)")
    fs.ygrid(ax)
    # the network names sit under their own line at the right margin: the BP curve is above 0.3 ms there
    x_right = ax.get_xlim()[1] / 1.04
    for arm, _ in NETWORKS:
        t = ax.annotate(arm, xy=(x_right, nets[arm]), xytext=(0, -2.0), textcoords="offset points",
                        ha="right", va="top", fontsize=7, color=fs.INK2, zorder=6)
        fs.halo(t, lw=1.8)
    # the two marked grids
    t = ax.annotate(rf"$N_{{\mathrm{{g}}}}={OPERATING}$", xy=(OPERATING, bp[OPERATING]), xytext=(-6, 3), textcoords="offset points",
                    ha="right", va="bottom", fontsize=7, color=fs.INK, zorder=6)
    fs.halo(t, lw=1.8)
    t = ax.annotate(rf"$N_{{\mathrm{{g}}}}={COARSE}$", xy=(COARSE, bp[COARSE]), xytext=(-5, 4), textcoords="offset points",
                    ha="right", va="bottom", fontsize=7, color=fs.INK, zorder=6)
    fs.halo(t, lw=1.8)


def dodge_offsets(n: int, step: float = DODGE_PT) -> np.ndarray:
    """Symmetric horizontal offsets, in points, for n series."""
    return (np.arange(n) - (n - 1) / 2.0) * step


def panel_risk(ax, series) -> None:
    """series: list of (label, ratios by N_g, colour, marker, linestyle)."""
    from matplotlib.transforms import ScaledTranslation
    handles = []
    for (lab, ratios, col, mk, ls), dx in zip(series, dodge_offsets(len(series))):
        tr = ax.transData + ScaledTranslation(dx / 72.0, 0.0, ax.figure.dpi_scale_trans)
        xs = np.array(ratios.index, dtype=float)
        dev = [deviation(r) for r in ratios.values]
        ys = np.array([d for d, _ in dev]); hollow = np.array([h for _, h in dev])
        ax.plot(xs, ys, color=col, ls=ls, lw=1.1, zorder=3, transform=tr)
        ax.plot(xs[~hollow], ys[~hollow], ls="none", marker=mk, ms=3.8, mfc=col, mec=col, zorder=4, transform=tr)
        ax.plot(xs[hollow], ys[hollow], ls="none", marker=mk, ms=3.8, mfc="white", mec=col, mew=1.0, zorder=4, transform=tr)
        handles.append(plt.Line2D([], [], color=col, ls=ls, lw=1.1, marker=mk, ms=3.8, label=lab))
    ax.axhline(FLOOR, color=fs.INK2, lw=0.8, ls=":", zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(44, 230); ax.set_ylim(FLOOR / 2.2, 8.0)
    _plain(ax.xaxis, [51, 101, 151, 201])
    ax.yaxis.set_major_locator(FixedLocator([1e-4, 1e-3, 1e-2, 1e-1, 1]))
    ax.yaxis.set_major_formatter(FixedFormatter([r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$1$"]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"grid points $N_{\mathrm{g}}$")
    ax.set_ylabel(r"$|\,\mathrm{risk}(N_{\mathrm{g}})\,/\,\mathrm{risk}(401)-1\,|$")
    fs.ygrid(ax)
    ax.legend(handles=handles, loc="upper right", fontsize=7, handlelength=2.0, labelspacing=0.25,
              borderaxespad=0.3, borderpad=0.2, frameon=False)


def make_figure(scaling, d47b, d33, d47c, out: Path) -> dict:
    bp, nets = bp_ms_per_sequence(scaling), network_ms_per_sequence(scaling)
    series = []
    for (arm, lab, which), col, mk, ls in zip(LEARNED_ARMS, (fs.CAT[0], fs.CAT[2]), fs.MARKERS[:2], ("-", "--")):
        src = d33[which]
        series.append((lab, learned_ratios(d47b, src, arm), col, mk, ls))
    for (col_name, lab), col, mk, ls in zip(HEADLINE_ARMS, (fs.CAT[1], fs.CAT[3]), fs.MARKERS[2:], ("-.", ":")):
        series.append((lab, headline_ratios(d47c, col_name), col, mk, ls))
    fs.apply()
    fig, axes = plt.subplots(1, 2, figsize=(fs.FULL_W, 2.3), gridspec_kw=dict(wspace=0.32))
    panel_time(axes[0], bp, nets); panel_risk(axes[1], series)
    fs.panel_label(axes[0], "(a)", x=-0.2); fs.panel_label(axes[1], "(b)", x=-0.2)
    out.parent.mkdir(parents=True, exist_ok=True)
    fs.save(fig, out)
    return dict(fig=fig, axes=axes, bp=bp, nets=nets, series=series)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None, help="dry-run directory (no paper file written)")
    a = ap.parse_args()
    scaling = pd.read_csv(ROOT / "outputs/hpc_E47t/scaling.csv")
    d47b = _load("outputs/hpc_E47b_s*/kernels.csv")
    d33 = {"q": _load("outputs/hpc_E33q_s*/kernels.csv"), "k": _load("outputs/hpc_E33k_s*/kernels.csv")}
    d47c = pd.read_csv(ROOT / "outputs/E47c/refinement.csv")
    out = (ROOT / a.out / "fig_grid_coarse.pdf") if a.out else (ROOT / "paper/figures/fig_grid_coarse.pdf")
    r = make_figure(scaling, d47b, d33, d47c, out)
    print("BP ms per sequence:", ", ".join(f"{int(g)}: {v:.4f}" for g, v in r["bp"].items()))
    print("networks ms per sequence:", ", ".join(f"{k}: {v:.4f}" for k, v in r["nets"].items()))
    for lab, ratios, *_ in r["series"]:
        print(f"{lab:14s} ratio - 1: " + ", ".join(f"{int(n)}: {v - 1:+.2e}" for n, v in ratios.items()))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
