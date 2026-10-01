#!/usr/bin/env python
"""E34 -- the physics of the field, the clean-data interface, and what the learned kernels generate.

Jerome's consolidated note (30 September 2026, "Structured denoising with EM-BP and
one-dimensional gradient SPDEs", section 24.5) asks for physical statistics beside the score
error: the correlation function C(r), the correlation length, the kink density, the
domain-length distribution, marginal bimodality, and generated-sample quality. Section 16.1
asks for clean-data maximum likelihood of the pair energy as the baseline that needs no
noising (Marc's remark, `em.fit_clean`). E33 measures score error only. This driver adds the
rest, in three parts that share E33's streams so every row pairs with E33's.

physics_truth   Deterministic. Per cell: the statistics above on exact stationary samples
                (the chain's own sampler) and, where a closed form exists, from the spectrum
                (C(r) by the spectral sum, the correlation length 1/log(lambda_0/lambda_1),
                the marginal kurtosis), so the sample statistics are checked against the
                exact ones in the same table. Runs on the laptop.
clean           Per (cell, seed, M): the four kernel arms of E33 fitted by CLEAN-data maximum
                likelihood on the very training chains E33's kernel arms see through one
                noised view (same tag, seed, cell, M), scored on E33's test bundles with
                E33's estimand. The difference between this row and E33's is the price of
                the channel, per kernel family. No E-step: the M-step runs on the histogram
                statistic, so a cell is minutes.
physics_learned After E33 lands. For each fitted kernel at M = 2048 (E33's `theta` column):
                ancestral samples on the grid from the fitted kernel started from its own
                invariant law (e27's sampler), and the same physical statistics as
                physics_truth, against the truth. This is "generated-sample quality" in the
                physical statistics the note names, not in a distributional distance.

PREDECLARED (registry E34.json):
  P1  the exact samples reproduce the spectral C(r) to within Monte Carlo error at every lag
      on every cell; and the kink density per unit length stays within 20% across the
      refinement cells, since the physical system is the same.
      SECOND HALF FALSIFIED on the first run (1 Oct): 0.120, 0.175, 0.247 per unit length at
      h = 1, 0.5, 0.25 -- the raw sign-change density grows like 1/sqrt(h). Registered as
      N-31 and replaced, BEFORE the coarse statistic was computed, by
  P1b the COARSE kink density per unit length (runs shorter than the kink width merged) is
      within 20% across the refinement cells, and the coarse mean domain length in physical
      units within 20%.
  P2  clean-data MLE beats one-view EM at every M for every kernel arm (16/16 paired), and
      the gap shrinks with M; for the exactly specified tilted kernels the clean-data risk at
      M = 2048 is within 10% of the exact score's zero... stated usably: below 0.02 pooled.
  P3  the mixture kernel's clean-data fit does NOT reach the tilted kernels' at M = 2048 on
      any cell: misspecification, not the channel, is what limits it (the E33 P1 mechanism
      isolated from the noise).
  P4  samples from the fitted tilted-spline kernel reproduce the truth's kink density within
      15% and its mean domain length within 15% at M = 2048 on every cell; samples from the
      fitted mixture kernel overshoot the kink density (a linear drift cannot hold a domain).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.em import clean_statistics  # noqa: E402
from markov_score.kernels import MDNKernel, MixtureInnovationKernel  # noqa: E402
from markov_score.nnet import MLP  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.pair_energy_kernel import PairEnergyKernel  # noqa: E402
from markov_score.tilted_kernel import TiltedWalkKernel  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402
from protocol import PROTOCOL  # noqa: E402

import e14_regime_baselines as e14  # noqa: E402
import e33_field_learning as e33  # noqa: E402
from e32_field_oracles import CELLS, resolve_cell  # noqa: E402

T_SCHEDULE = tuple(FROZEN.t_grid)

SETTINGS = dict(
    cells=tuple(c["id"] for c in CELLS),
    n_physics=4096, reps=(0, 1),
    sizes=PROTOCOL.sizes, seeds=PROTOCOL.seeds, n_val=PROTOCOL.n_val, n_test=PROTOCOL.n_test,
    grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    arms=e33.KERNEL_ARMS, components=8, poly_degree=6, spline_basis=24, mdn_components=4, mdn_hidden=32,
    em_init_rho=PROTOCOL.em_init_rho, em_init_var=PROTOCOL.em_init_var,
    clean_checkpoints=(1, 2, 4, 8, 16, 32, 64, 120, 220, 400),
    learned_sizes=(2048,), learned_seeds=(0, 1, 2, 3), n_learned=2048,
    kernels_csv=(),                     # physics_learned: E33's kernels.csv shards
)
QUICK = dict(cells=("T0.3_h1_L32",), n_physics=256, reps=(0,), sizes=(32,), seeds=(10_000,), n_val=32, n_test=32,
             grid_size=201, clean_checkpoints=(1, 2, 4), arms=("em_mixture", "em_tilted_poly"), learned_seeds=(10_000,),
             n_learned=128)


# ------------------------------------------------------------------ physical statistics


def _coarse_runs(sgn_row: np.ndarray, min_len: int) -> np.ndarray:
    """Run lengths of constant sign after merging every run shorter than `min_len` into its
    neighbours (a run that is too short to be a domain is a crossing of the barrier inside a
    domain, not a kink). Iterated until no short run remains; returns the interior runs."""
    bounds = np.concatenate([[0], np.flatnonzero(sgn_row[:-1] * sgn_row[1:] < 0) + 1, [len(sgn_row)]])
    lengths = np.diff(bounds).tolist()
    changed = True
    while changed and len(lengths) >= 3:
        changed = False
        for i in range(1, len(lengths) - 1):             # interior runs only
            if lengths[i] < min_len:
                lengths[i - 1] += lengths[i] + lengths[i + 1]
                del lengths[i:i + 2]
                changed = True
                break
    return np.asarray(lengths[1:-1], dtype=float) if len(lengths) >= 3 else np.empty(0)


def physical_statistics(A: np.ndarray, scale: float, L: int, h: float = 1.0) -> dict:
    """Kink density, domain lengths, C(r), bimodality of a batch of standardised chains.

    A kink is a sign change between neighbours (the wells sit at +-1/s in standardised
    units, the barrier at 0). Kink density is per bond; the physical kink density per unit
    length is (kinks per bond) / h, reported by the caller. Domain lengths are run lengths
    of constant sign within a chain, censored at the chain ends (reported separately).

    THE RAW SIGN-CHANGE DENSITY IS NOT A PHYSICAL OBSERVABLE UNDER REFINEMENT (found 1 October
    2026, E34 P1's second half falsified): along the refinement cells it read 0.120, 0.175,
    0.247 per unit length at h = 1, 0.5, 0.25, growing like 1/sqrt(h), because a rough field
    crosses the barrier many times inside one domain and finer lattices resolve more of those
    crossings. The COARSE statistics merge every run shorter than the kink width sqrt(2)
    (in sites: sqrt(2)/h) into its neighbours, so that a kink is a run boundary between two
    domains at least one kink width long; that is the observable the refinement cells are
    predicted to share (registry E34, P1b).
    """
    n, L = A.shape
    sgn = np.sign(A)
    kinks = (sgn[:, :-1] * sgn[:, 1:] < 0)
    kink_density = float(kinks.mean())
    min_len = max(1, int(np.ceil(np.sqrt(2.0) / h)))
    runs, censored, coarse = [], 0, []
    for row in sgn:
        change = np.flatnonzero(row[:-1] * row[1:] < 0) + 1
        bounds = np.concatenate([[0], change, [L]])
        lengths = np.diff(bounds)
        if len(lengths) >= 3:
            runs.extend(lengths[1:-1].tolist())        # interior domains only
        censored += 2 if len(lengths) >= 2 else 1
        coarse.append(_coarse_runs(row, min_len))
    runs = np.asarray(runs, dtype=float)
    coarse_runs = np.concatenate(coarse) if coarse else np.empty(0)
    # coarse kinks per bond: interior coarse domain boundaries over bonds, per chain
    coarse_kinks = float(sum(len(c) + 1 if len(c) else 0 for c in coarse)) / (n * (L - 1))
    corr = np.array([float(np.mean(A[:, :L - r] * A[:, r:])) for r in range(L)])
    # the Monte Carlo error of each lag from the scatter of per-chain means (chains are
    # independent; sites within a chain are not, which a naive 1/sqrt(nL) would ignore)
    corr_se = np.array([float(np.std((A[:, :L - r] * A[:, r:]).mean(axis=1), ddof=1) / np.sqrt(n)) for r in range(L)])
    var = float(A.var())
    kurt = float(np.mean(A**4) / var**2 - 3.0)
    skew = float(np.mean(A**3) / var**1.5)
    barrier = float(np.mean(np.abs(A) < 0.5 / scale))
    return dict(kink_density=kink_density, mean_domain=float(runs.mean()) if runs.size else np.nan,
                median_domain=float(np.median(runs)) if runs.size else np.nan, n_domains=int(runs.size),
                censored_domains=int(censored),
                coarse_min_len=int(min_len), coarse_kink_density=coarse_kinks,
                coarse_mean_domain=float(coarse_runs.mean()) if coarse_runs.size else np.nan,
                coarse_n_domains=int(coarse_runs.size),
                var=var, excess_kurtosis=kurt, skew=skew,
                bimodality_coefficient=float((skew**2 + 1.0) / (kurt + 3.0)), barrier_mass=barrier,
                corr=";".join(f"{c:.6f}" for c in corr), corr_se_lags=";".join(f"{c:.6f}" for c in corr_se))


def _physics_cell(args):
    cfg, cell_id, rep = args
    cell = next(c for c in CELLS if c["id"] == cell_id)
    chain, info = resolve_cell(cell)
    L = cell["L"]
    A = chain.sample_batch(rng_for("e34-physics", cell_id, rep), cfg["n_physics"], L)
    st = physical_statistics(A, chain.scale, L, info["h"])
    cov = chain.covariance(L)[0]
    corr_sample = np.array([float(x) for x in st["corr"].split(";")])
    corr_se = np.array([float(x) for x in st["corr_se_lags"].split(";")])
    z = np.abs(corr_sample - cov) / np.maximum(corr_se, 1e-12)
    rows = [{**info, "rep": rep, "source": "exact_samples", "n_chains": cfg["n_physics"], **st,
             "kink_density_physical": st["kink_density"] / info["h"],
             "mean_domain_physical": st["mean_domain"] * info["h"],
             "coarse_kink_density_physical": st["coarse_kink_density"] / info["h"],
             "coarse_mean_domain_physical": st["coarse_mean_domain"] * info["h"],
             "corr_max_abs_diff": float(np.max(np.abs(corr_sample - cov))), "corr_max_z": float(z.max()),
             "corr_se": float(corr_se[0])},
            {**info, "rep": rep, "source": "spectral", "n_chains": 0,
             "kink_density": np.nan, "mean_domain": np.nan, "median_domain": np.nan, "n_domains": 0,
             "censored_domains": 0, "coarse_min_len": 0, "coarse_kink_density": np.nan, "coarse_mean_domain": np.nan,
             "coarse_n_domains": 0, "coarse_kink_density_physical": np.nan, "coarse_mean_domain_physical": np.nan,
             "var": 1.0, "excess_kurtosis": info["excess_kurtosis"], "skew": 0.0,
             "bimodality_coefficient": float(1.0 / (info["excess_kurtosis"] + 3.0)),
             "barrier_mass": float(np.exp(chain.log_stationary(np.linspace(-0.5 / chain.scale, 0.5 / chain.scale, 201))).mean() / chain.scale),
             "corr": ";".join(f"{c:.6f}" for c in cov), "corr_se_lags": "", "kink_density_physical": np.nan,
             "mean_domain_physical": np.nan, "corr_max_abs_diff": 0.0, "corr_max_z": 0.0, "corr_se": 0.0}]
    print(f"  [E34 physics] {cell_id} rep={rep}: kinks/bond={st['kink_density']:.4f} "
          f"(coarse {st['coarse_kink_density']:.4f}, /unit length {rows[0]['coarse_kink_density_physical']:.3f}) "
          f"mean domain={st['mean_domain']:.2f} kurt={st['excess_kurtosis']:+.3f} (exact {info['excess_kurtosis']:+.3f}) "
          f"max|dC|={rows[0]['corr_max_abs_diff']:.4f} (max z={rows[0]['corr_max_z']:.2f})", flush=True)
    return rows


def part_physics_truth(cfg, out, workers):
    cells = [(cfg, c, r) for c in cfg["cells"] for r in cfg["reps"]]
    write_csv(out / "physics.csv", run_cells(cells, _physics_cell, workers, out, "physics.csv"))


# ------------------------------------------------------------------ the clean-data interface


def _fit_clean_ladder(kernel, grid, stats, checkpoints):
    """Repeated M-steps on the clean histogram statistic, snapshotting at the ladder."""
    saved, current = {}, kernel
    for it in range(1, max(checkpoints) + 1):
        current = current.m_step(stats, grid)
        if it in checkpoints:
            saved[it] = current
    return saved


def _clean_cell(args):
    cfg, cell_id, seed, n = args
    chain, grid, w, log_k, log_mu, sigma0, A, Av, val, test, base = e33._common(cfg, cell_id, seed, n)
    L = base["n_sites"]
    stats = clean_statistics(grid, A)
    mu_fn = e33._stationary_mu(grid, w)
    rows = []
    for arm in cfg["arms"]:
        t0 = time.time()
        k0 = e33._init_kernel(arm, cfg, grid, seed, cell_id, n)
        ck = tuple(int(c) for c in cfg["clean_checkpoints"])
        if arm.startswith("em_tilted"):
            ck = tuple(c for c in ck if c <= 4) or (1,)   # one L-BFGS solve is the maximiser; a few repeats for safety
        saved = _fit_clean_ladder(k0, grid, stats, set(ck))
        best, best_v, curve, mus = None, np.inf, {}, {}
        for it in sorted(saved):
            mu = mu_fn(saved[it]); mus[it] = mu
            lk = saved[it].log_transition_matrix(grid)
            v = e14._pooled(lambda X, t, lk=lk, mu=mu: grid_bp_batch(grid, w, lk, X, *alpha_delta(t), log_mu=mu)[0], val)
            curve[it] = v
            if v < best_v:
                best, best_v = it, v
        kb = saved[best]
        lk = kb.log_transition_matrix(grid)
        risk = e14._pooled(lambda X, t, lk=lk, mu=mus[best]: grid_bp_batch(grid, w, lk, X, *alpha_delta(t), log_mu=mu)[0], test)
        rows.append({**base, **e33.ROW_DEFAULTS, "arm": arm, "selection": "risk", "risk": risk, "checkpoint": best,
                     "at_cap": int(best == max(saved)), "val_selected": best_v, "n_params": int(len(np.asarray(kb.theta))),
                     "wall": time.time() - t0, "n_iters": max(saved), **e33._kernel_diagnostics(kb, grid),
                     "theta": ";".join(f"{v:.8g}" for v in np.asarray(kb.theta, dtype=float)),
                     "val_curve": ";".join(f"{i}:{a:.5f}" for i, a in sorted(curve.items()))})
    print(f"  [E34 clean] {cell_id} seed={seed} M={n}: " + " ".join(f"{r['arm']}={r['risk']:.4f}" for r in rows), flush=True)
    return rows


def part_clean(cfg, out, workers):
    cells = [(cfg, c, s, n) for n in sorted(cfg["sizes"], reverse=True) for c in cfg["cells"] for s in cfg["seeds"]]
    print(f"  clean: {len(cells)} cells x {len(cfg['arms'])} arms", flush=True)
    write_csv(out / "clean.csv", run_cells(cells, _clean_cell, workers, out, "clean.csv"))


# ------------------------------------------------------------------ what the learned kernels generate


def kernel_from_row(row, cfg, grid):
    """Rebuild a fitted kernel from an E33/E34 CSV row (its `theta` column and the settings)."""
    theta = np.array([float(v) for v in str(row["theta"]).split(";")])
    arm = row["arm"]
    if arm == "em_mixture":
        C = cfg["components"]
        return MixtureInnovationKernel(rho=float(theta[0]), pi=theta[1:1 + C], mu=theta[1 + C:1 + 2 * C], s2=theta[1 + 2 * C:1 + 3 * C])
    if arm in ("em_tilted_poly", "em_tilted_spline"):
        basis = "legendre" if arm.endswith("poly") else "bspline"
        return TiltedWalkKernel(log_gamma=float(theta[0]), coef=theta[1:], basis=basis, half_width=float(np.max(np.abs(grid))))
    if arm == "em_pair_quartic":
        return PairEnergyKernel(log_gamma=float(theta[0]), coef=theta[1:], half_width=float(np.max(np.abs(grid))))
    if arm == "em_mdn":
        k = MDNKernel.init(cfg["mdn_components"], cfg["mdn_hidden"], rng_for("e34-mdn-shape"))
        sizes = [p.size for p in k.net.params]
        if sum(sizes) != theta.size:
            raise ValueError(f"MDN theta has {theta.size} entries, the architecture {sum(sizes)}")
        pos = 0
        for p in k.net.params:
            p[...] = theta[pos:pos + p.size].reshape(p.shape); pos += p.size
        return k
    raise ValueError(arm)


def _learned_cell(args):
    cfg, cell_id, seed, n, rows_for_cell = args
    cell = next(c for c in CELLS if c["id"] == cell_id)
    chain, info = resolve_cell(cell)
    L = cell["L"]
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    import e27_direct_sampler as e27
    e27.N_SITES = L
    truth = physical_statistics(chain.sample_batch(rng_for("e34-learned-truth", cell_id, seed), cfg["n_learned"], L), chain.scale, L, info["h"])
    out = [{**info, "seed": seed, "n_chains": n, "arm": "truth", "selection": "", **truth,
            "kink_density_physical": truth["kink_density"] / info["h"], "mean_domain_physical": truth["mean_domain"] * info["h"]}]
    mu_fn = e33._stationary_mu(grid, w)
    for row in rows_for_cell:
        k = kernel_from_row(row, cfg, grid)
        K = np.exp(k.log_transition_matrix(grid))
        mu = mu_fn(k)
        S = e27.ancestral_grid(K, grid, w, cfg["n_learned"], rng_for("e34-learned", cell_id, seed, row["arm"]),
                               rng_for("e34-learned-jit", cell_id, seed, row["arm"]), jitter=True, log_mu=mu)
        st = physical_statistics(S, chain.scale, L, info["h"])
        out.append({**info, "seed": seed, "n_chains": n, "arm": row["arm"], "selection": row["selection"], **st,
                    "kink_density_physical": st["kink_density"] / info["h"], "mean_domain_physical": st["mean_domain"] * info["h"]})
    print(f"  [E34 learned] {cell_id} seed={seed} M={n}: " + " ".join(
        f"{r['arm']}:kinks={r['kink_density']:.4f}/dom={r['mean_domain']:.1f}" for r in out), flush=True)
    return out


def part_physics_learned(cfg, out, workers):
    import pandas as pd
    paths = list(cfg["kernels_csv"])
    if not paths:
        raise SystemExit("physics_learned needs --set kernels_csv=('outputs/hpc_E33k_s0/kernels.csv', ...)")
    df = pd.concat([pd.read_csv(Path(__file__).resolve().parents[1] / p) for p in paths], ignore_index=True)
    df = df[(df.selection == "risk") & df.n_chains.isin(cfg["learned_sizes"]) & df.seed.isin(cfg["learned_seeds"]) & df.cell.isin(cfg["cells"])]
    cells = []
    for (c, s, n), sub in df.groupby(["cell", "seed", "n_chains"]):
        cells.append((cfg, c, int(s), int(n), sub.to_dict("records")))
    print(f"  physics_learned: {len(cells)} cells from {len(df)} fitted kernels", flush=True)
    write_csv(out / "learned.csv", run_cells(cells, _learned_cell, workers, out, "learned.csv",
                                             key=lambda cell: f"{cell[1]}_s{cell[2]}_M{cell[3]}"))


PARTS = {"physics_truth": part_physics_truth, "clean": part_clean, "physics_learned": part_physics_learned}


def main():
    ap = experiment_parser("e34_field_physics", "physical statistics of the field, the clean-data interface, generated samples")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.list_parts:
        print("\n".join(PARTS)); return
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    out = ensure_dir(Path(args.output_dir))
    cells_info = [resolve_cell(c)[1] for c in CELLS if c["id"] in cfg["cells"]]
    write_json(out / "cells.json", cells_info)
    for name, fn in select_parts(PARTS, args.only).items():
        fn(cfg, out, args.workers)
        write_json(out / f"params_{name}.json", {**provenance(), "settings": cfg, "cells": cells_info,
                                                 "quick": bool(args.quick), "workers": args.workers})


if __name__ == "__main__":
    main()
