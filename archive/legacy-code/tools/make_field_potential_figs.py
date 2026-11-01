"""The potential and the temperature read from kernels learned through noise (E33, E34).

Every kernel of the gradient class (em_tilted_poly, em_tilted_spline, em_pair_quartic) is the Gibbs
chain of W(a, a') = (a' - a)^2 / 2 gamma + f(a) + f(a') (docs/FIELD_MODEL.md, reversibility), so a
fitted kernel reads as physics: Theta_hat = gamma_hat s^2 / h and V_hat(phi) = (2 Theta_hat / h)
f_hat(phi / s) + const. This tool rebuilds each risk-selected kernel from its `theta` column and
writes

    reports/figures/fig-field-potential.pdf
        (a) V_hat(phi) against the true double well at the base cell, one curve per arm at the
            largest M in the inputs, every seed drawn; (b) the same at the Theta = 2 cell;
        (c) Theta_hat / Theta against M per arm, seeds as points, both cells.
    reports/numbers/field_potential.csv
        per (source, cell, arm, M): Theta_hat / Theta, barrier, well position, potential RMS,
        mean and spread over seeds, with the seed count.

Inputs are any E33 kernels.csv shards and E34 clean.csv shards (`--kernels`, `--clean`). The
defaults are the laptop PREVIEW runs of 1 October (two cells, two seeds, M <= 2048, labelled
preview in every output); the confirmatory shards replace them when they land.
House rules: tools/figstyle (no prose in a figure, every symbol in math mode, fs.save).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import figstyle as fs  # noqa: E402
from markov_score.bp_grid import make_grid  # noqa: E402
from markov_score.field import potential  # noqa: E402
from markov_score.pair_energy_kernel import potential_readings  # noqa: E402
from frozen_config import FROZEN  # noqa: E402

ARMS = {"em_pair_quartic": ("site energy, 5", fs.CAT[0]), "em_tilted_poly": ("Legendre, 7", fs.CAT[1]), "em_tilted_spline": ("spline, 25", fs.CAT[2])}
# drawing order: the five-parameter site energy and the true potential on top of the flexible tilts
ZORDER = {"em_pair_quartic": 4, "em_tilted_poly": 3, "em_tilted_spline": 2}
DEFAULT_KERNELS = ["outputs/E33_preview_laptop/kernels.csv", "outputs/E33_preview_laptop_pair/kernels.csv",
                   "outputs/E33_preview_laptop_M2048/kernels.csv", "outputs/E33_preview_laptop_pair_M2048/kernels.csv"]
DEFAULT_CLEAN = ["outputs/E34_preview_laptop/clean.csv", "outputs/E34_preview_laptop_pairclean/clean.csv"]
CELL_LABEL = {"T0.3_h1_L32": r"$\Theta{=}0.3$", "T2_iso_L32": r"$\Theta{=}2$", "T0.5_iso_L32": r"$\Theta{=}0.5$",
              "T1_iso_L32": r"$\Theta{=}1$", "T0.3_h0.5_L64": r"$h{=}0.5$", "T0.3_h0.25_L128": r"$h{=}0.25$"}


def _load(paths, source):
    frames = []
    for p in paths:
        q = ROOT / p
        if q.exists():
            df = pd.read_csv(q)
            df["source"] = source
            df["path"] = p
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df[(df.selection == "risk") & df.arm.isin(ARMS)].copy()


def _readings(df, grid):
    from e34_field_physics import kernel_from_row
    import e33_field_learning as e33
    cfg = dict(e33.SETTINGS)
    rows, curves = [], {}
    phi = np.linspace(-2.0, 2.0, 801)
    for _, r in df.iterrows():
        k = kernel_from_row(r, cfg, grid)
        f = k.site_energy(grid)
        rd = potential_readings(f, grid, k.gamma, float(r["scale"]), float(r["h"]))
        # coefficients at the optimiser's bound (+-200): the tilt was pushed down where no data arrive
        # (docs/NEGATIVE_RESULTS.md N-33); harmless for a local basis, a warning sign for a global one
        coef = np.asarray(k.coef, dtype=float)
        rows.append({"source": r["source"], "cell": r["cell"], "arm": r["arm"], "n_chains": int(r["n_chains"]), "seed": int(r["seed"]),
                     "T": float(r["T"]), "h": float(r["h"]), "risk": float(r["risk"]), "theta_ratio": rd["theta_hat"] / float(r["T"]),
                     "n_at_bound": int((np.abs(coef) >= 199.9).sum()), **rd})
        from scipy.interpolate import CubicSpline
        fin = np.isfinite(f)
        v = (2.0 * rd["theta_hat"] / float(r["h"])) * CubicSpline(grid[fin], f[fin], extrapolate=False)(phi / float(r["scale"]))
        ok = np.isfinite(v) & (np.abs(phi) <= 1.6)
        v = v - np.mean((v - potential(phi))[ok])
        curves[(r["source"], r["cell"], r["arm"], int(r["n_chains"]), int(r["seed"]))] = (phi, v)
    return pd.DataFrame(rows), curves


def figure(tab, curves, out):
    cells = [c for c in ("T0.3_h1_L32", "T2_iso_L32") if c in set(tab.cell)]
    fig, axes = plt.subplots(1, 3, figsize=(fs.FULL_W, 2.15), gridspec_kw=dict(wspace=0.36))
    phi = np.linspace(-2.0, 2.0, 801)
    for ax, cell in zip(axes[:2], cells):
        ax.plot(phi, potential(phi), color=fs.INK, lw=1.3, ls="--", label=r"$V$", zorder=6)
        sub = tab[(tab.cell == cell) & (tab.source == "noisy")]
        if sub.empty:
            continue
        M = int(sub.n_chains.max())
        for arm, (lab, col) in ARMS.items():
            for seed in sorted(sub[(sub.arm == arm) & (sub.n_chains == M)].seed.unique()):
                key = ("noisy", cell, arm, M, int(seed))
                if key in curves:
                    p, v = curves[key]
                    ax.plot(p, v, color=col, lw=0.9 if arm == "em_pair_quartic" else 0.7, alpha=0.9 if arm == "em_pair_quartic" else 0.55,
                            zorder=ZORDER[arm], label=lab if seed == sorted(sub.seed.unique())[0] else None)
            for seed in sorted(tab[(tab.cell == cell) & (tab.source == "clean") & (tab.arm == arm) & (tab.n_chains == M)].seed.unique())[:1]:
                key = ("clean", cell, arm, M, int(seed))
                if key in curves:
                    p, v = curves[key]
                    ax.plot(p, v, color=col, lw=0.7, ls=":", alpha=0.9)
        ax.set_xlabel(r"$\phi$"); ax.set_ylabel(r"$\hat V(\phi)$"); ax.set_ylim(-0.12, 0.75); ax.set_xlim(-2, 2)
        fs.ygrid(ax); fs.panel_label(ax, "(a)" if ax is axes[0] else "(b)")
        ax.set_title(CELL_LABEL.get(cell, cell) + f", $M{{=}}{M}$", fontsize=6.6, pad=2.5)
    # the region above the barrier, between the wells, is the one every curve leaves empty
    axes[0].legend(fontsize=5.8, loc="upper center", ncol=1, handlelength=1.5, labelspacing=0.2, borderpad=0.25)
    cx = axes[2]
    for j, cell in enumerate(cells):
        for arm, (lab, col) in ARMS.items():
            sub = tab[(tab.cell == cell) & (tab.source == "noisy") & (tab.arm == arm)]
            if sub.empty:
                continue
            g = sub.groupby("n_chains").theta_ratio
            m = g.mean(); x = m.index.values * (1.0 + 0.06 * (list(ARMS).index(arm) - 1))
            cx.plot(x, m.values, color=col, marker="o" if j == 0 else "s", ms=3.0, lw=1.0, ls="-" if j == 0 else "--",
                    label=f"{lab}, " + CELL_LABEL.get(cell, cell))
            cx.scatter(np.repeat(x, g.count().values), np.concatenate([v.values for _, v in g]), color=col, s=4, alpha=0.5, zorder=3)
    cx.axhline(1.0, color=fs.INK2, lw=0.8, ls="--")
    cx.set_xscale("log"); cx.set_xlabel("$M$"); cx.set_ylabel(r"$\hat\Theta/\Theta$")
    cx.set_xticks([32, 128, 512, 2048]); cx.set_xticklabels(["32", "128", "512", "2048"]); cx.minorticks_off()
    fs.ygrid(cx); fs.panel_label(cx, "(c)")
    cx.legend(fontsize=5.4, loc="best", handlelength=1.6, labelspacing=0.15, borderpad=0.25)
    fs.save(fig, out)


def table(tab, out):
    keys = ["source", "cell", "arm", "n_chains"]
    agg = tab.groupby(keys).agg(n_seeds=("seed", "count"), theta_ratio=("theta_ratio", "mean"), theta_ratio_sd=("theta_ratio", "std"),
                                barrier=("barrier_hat", "mean"), barrier_sd=("barrier_hat", "std"), well=("well_hat", "mean"),
                                potential_rms=("potential_rms", "mean"), risk=("risk", "mean"), n_at_bound=("n_at_bound", "max")).reset_index()
    out.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(out, index=False, float_format="%.4f")
    return agg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kernels", nargs="*", default=DEFAULT_KERNELS)
    ap.add_argument("--clean", nargs="*", default=DEFAULT_CLEAN)
    ap.add_argument("--fig", default="reports/figures/fig-field-potential.pdf")
    ap.add_argument("--table", default="reports/numbers/field_potential.csv")
    ap.add_argument("--arms", nargs="*", default=list(ARMS),
                    help="subset of the gradient-class arms to draw (the frozen Legendre shards contain failed fits)")
    args = ap.parse_args()
    for arm in [a for a in ARMS if a not in args.arms]:
        del ARMS[arm]
    fs.apply()
    grid, _ = make_grid(FROZEN.half_width, FROZEN.n_grid)
    noisy = _load(args.kernels, "noisy"); clean = _load(args.clean, "clean")
    df = pd.concat([d for d in (noisy, clean) if not d.empty], ignore_index=True)
    if df.empty:
        raise SystemExit("no risk-selected rows of the gradient-class arms in the inputs")
    tab, curves = _readings(df, grid)
    agg = table(tab, ROOT / args.table)
    with pd.option_context("display.width", 200):
        print(agg.to_string(index=False))
    figure(tab, curves, ROOT / args.fig)
    print("wrote", args.fig, "and", args.table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
