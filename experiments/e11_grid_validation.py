#!/usr/bin/env python
"""E11 — is the tree-exact reference valid everywhere we use it, or only at the frozen point?

Every number in this project is measured against a reference computed by grid belief propagation
at N_g = 401 points on [-8, 8]. That choice was validated once, at the frozen configuration
(Laplace innovation, rho = 0.85, the twelve-level schedule). Every other configuration the project
touches -- the four other innovation families, the five correlations of the phase sweep, the
generalised-Gaussian family, the longer chains of the length study -- inherits that validation
without having earned it. This experiment earns it, by sweeping the quadrature against the
configurations rather than assuming one transfers.

For each (family, rho, chain length, noise level) and each candidate grid (N_g, A), three numbers:

  boundary   the posterior mass in the outermost 2% of the domain, worst chain in the bundle --
             how much of the belief the truncation is cutting off;
  colmass    max_j |sum_i K(g_i | g_j) w_i - 1| -- how far the discretised transition kernel is
             from a probability distribution, which is the quadrature error at its source;
  drift      the relative RMS difference between this grid's posterior-mean field and the finest
             grid's on the SAME chains -- what a reported number would actually move by.

`drift` is the one that matters for a claim; the other two say why it is small. The reference for
`drift` is the finest grid in the ladder, not the truth, so the quantity is a convergence
diagnostic and is reported as such.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import GaussianAR1, GaussianMixtureAR1, LaplaceAR1, StudentTAR1, UniformAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

FAMILIES = {"gaussian": GaussianAR1, "laplace": LaplaceAR1, "student": StudentTAR1,
            "uniform": UniformAR1, "bimodal": GaussianMixtureAR1}

SETTINGS = dict(
    families=("gaussian", "laplace", "student", "uniform", "bimodal"),
    rhos=(0.0, 0.5, 0.85, 0.95),
    lengths=(32,),
    levels=tuple(FROZEN.t_grid),
    grids=((201, 8.0), (401, 6.0), (401, 8.0), (401, 10.0), (801, 8.0), (801, 12.0), (1601, 8.0)),
    reference_grid=(3201, 12.0),
    n_chains=256, seeds=(0, 1, 2, 3),
    boundary_frac=0.02,
)
QUICK = dict(families=("laplace",), rhos=(0.85,), levels=(0.05, 1.0), grids=((201, 8.0), (401, 8.0)),
             reference_grid=(801, 8.0), n_chains=32, seeds=(0,))


def _prior(family, rho):
    cls = FAMILIES[family]
    return cls(rho)


def _cell(args):
    """One (family, rho, L, seed): every grid in the ladder on the same chains, at every level."""
    cfg, family, rho, L, seed = args
    prior = _prior(family, rho)
    rng = rng_for("e11-chains", family, rho, L, seed)
    A = np.stack([prior.sample(rng, L) for _ in range(cfg["n_chains"])])
    X = {}
    for t in cfg["levels"]:
        al, de = alpha_delta(t)
        X[t] = al * A + np.sqrt(de) * rng.standard_normal(A.shape)

    def field(ng, hw, want_diag):
        grid, w = make_grid(hw, ng)
        log_k = prior.log_transition_matrix(grid)
        col = float(np.abs((np.exp(log_k) * w[:, None]).sum(axis=0) - 1.0).max())
        means, bfwd, bbel = {}, 0.0, 0.0
        for t in cfg["levels"]:
            al, de = alpha_delta(t)
            if want_diag:
                m, _, d = grid_bp_batch(grid, w, log_k, X[t], al, de, boundary_frac=cfg["boundary_frac"], chunk=None)
                bfwd = max(bfwd, d["forward_edge_max"]); bbel = max(bbel, d["belief_edge_max"])
            else:
                m, _ = grid_bp_batch(grid, w, log_k, X[t], al, de)
            means[t] = m
        return means, col, (bfwd, bbel)

    ref_means, _, _ = field(*cfg["reference_grid"], False)
    rows = []
    for ng, hw in cfg["grids"]:
        means, col, (bfwd, bbel) = field(ng, hw, True)
        num = den = 0.0
        per_level = {}
        for t in cfg["levels"]:
            d = float(((means[t] - ref_means[t]) ** 2).sum()); e = float((ref_means[t] ** 2).sum())
            num += d; den += e
            per_level[t] = np.sqrt(d / e) if e > 0 else np.nan
        rows.append(dict(family=family, rho=rho, n_sites=L, seed=seed, n_grid=ng, half_width=hw,
                         n_chains=cfg["n_chains"],
                         boundary_fwd_max=bfwd, boundary_belief_max=bbel, col_mass_max=col,
                         drift_pooled=float(np.sqrt(num / den)),
                         drift_worst_level=float(np.nanmax(list(per_level.values()))),
                         drift_at_lowest_level=float(per_level[min(cfg["levels"])]),
                         ref_n_grid=cfg["reference_grid"][0], ref_half_width=cfg["reference_grid"][1]))
        print(f"  [E11] {family} rho={rho} L={L} seed={seed} grid=({ng},{hw}): "
              f"drift={rows[-1]['drift_pooled']:.2e} fwd_edge={bfwd:.2e} belief_edge={bbel:.2e} colmass={col:.2e}", flush=True)
    return rows


def part_sweep(cfg, out, workers):
    cells = [(cfg, f, r, L, s) for f in cfg["families"] for r in cfg["rhos"]
             for L in cfg["lengths"] for s in cfg["seeds"]]
    print(f"  {len(cells)} cells", flush=True)
    rows = run_cells(cells, _cell, workers, out, "grid.csv")
    write_csv(out / "grid.csv", rows)


PARTS = {"sweep": part_sweep}


def main():
    ap = experiment_parser("e11_grid_validation", "quadrature validity across families, correlations and levels")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.list_parts:
        print("\n".join(PARTS)); return
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    out = ensure_dir(Path(args.output_dir))
    for name, fn in select_parts(PARTS, args.only).items():
        fn(cfg, out, args.workers)
        write_json(out / f"params_{name}.json", {**provenance(), "settings": cfg, "quick": bool(args.quick), "workers": args.workers})


if __name__ == "__main__":
    main()
