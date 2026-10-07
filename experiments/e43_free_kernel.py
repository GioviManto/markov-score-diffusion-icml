"""E43: is the site-energy form wide enough? The price and the benefit of an unrestricted kernel.

The open question the field report puts to Jerome and Marc: the Gaussian-bond-plus-site-energy kernel is the whole class of
discretised gradient SPDEs, and exactly specified for phi^4 with five parameters; would a general kernel be worth its parameters?
E43 fits the most agnostic family there is, markov_score.loglinear_kernel (a log-linear conditional kernel with a 2D spline score,
100 parameters, exact EM, no penalty), in two laboratories.

  Lab A  E33's own phi^4 cells (T0.3_h1_L32 and T1_iso_L32; every one of E33's data, validation and test bundles, seeds and
         initialisations): where the site energy is EXACTLY specified, what does generality cost? The comparators are E33's frozen
         em_pair_quartic, em_tilted_spline and em_mixture fits on the same seeds (bundle hashes compared at read time).
  Lab B  a chain OUTSIDE the tilted-walk class (markov_score.grid_pair_chain: a Laplace bond on a double-well site energy, calibrated
         to the phi^4 base cell's correlation and kurtosis before any fit): exact samples, exact oracle. Here the Gaussian-bond families
         are misspecified. Arms fitted fresh: em_loglinear, em_pair_quartic, em_tilted_spline, em_mixture.

This is the frozen e33 driver with two additions, applied IN PROCESS (e33_field_learning.py itself is not modified): the arm
`em_loglinear` in the arm constructor and the cell `LAP_B_L32` in the cell list. Rows have E33's schema, so the same readings apply.

Predictions (experiments/registry/E43.json):
  Lab A  P1  M = 128: loglinear / site energy >= 2 in >= 14/16 seeds on both cells (the price of generality when data are few)
         P2  the seed-mean ratio loglinear / site energy is smaller at M = 2048 than at M = 128 on both cells, and below 2 at 2048
         P3  M = 2048: loglinear below the mixture in >= 14/16 seeds on both cells (it represents the nonlinear drift)
         P4  M = 128: loglinear above the 25-parameter tilted spline in >= 12/16 seeds on both cells
  Lab B  P5  M = 2048: loglinear below the site energy in >= 7/8 seeds, seed-mean ratio site energy / loglinear >= 1.25
         P6  M = 2048: loglinear below the tilted spline and below the mixture, each in >= 7/8 seeds
         P7  the crossover: the seed-mean ratio loglinear / site energy is above 1 at M = 128 and below 1 at M = 2048
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e33_field_learning as e33  # noqa: E402
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from markov_score import theory  # noqa: E402
from markov_score.grid_pair_chain import laplace_double_well  # noqa: E402
from markov_score.loglinear_kernel import LogLinearKernel  # noqa: E402
from markov_score.utils import ensure_dir, write_csv, write_json  # noqa: E402

e14 = e33.e14

LAB_B = dict(id="LAP_B_L32", T=float("nan"), h=float("nan"), L=32, role="lab_b")
BOND, DEPTH = 0.40, 2.0          # calibrated on the chain's own variance, lag-1 correlation and kurtosis, before any fit
_CHAIN_B = laplace_double_well(bond=BOND, depth=DEPTH)

_resolve_e33, _init_e33 = e33.resolve_cell, e33._init_kernel


def _resolve(cell):
    if cell["id"] == LAB_B["id"]:
        info = dict(cell=cell["id"], T=float("nan"), h=float("nan"), n_sites=cell["L"], role=cell["role"],
                    rho1=_CHAIN_B.lag1_correlation, corr_length_sites=_CHAIN_B.corr_length_sites(),
                    scale=float(np.sqrt(_CHAIN_B.variance)), excess_kurtosis=_CHAIN_B.excess_kurtosis)
        return _CHAIN_B, info
    return _resolve_e33(cell)


def _init(arm, cfg, grid, seed, cell_id, n):
    if arm == "em_loglinear":
        return LogLinearKernel.init(cfg["loglinear_basis"], rho=cfg["em_init_rho"], var=cfg["em_init_var"],
                                    half_width=float(np.max(np.abs(grid))), grid_size=len(grid))
    return _init_e33(arm, cfg, grid, seed, cell_id, n)


# The two additions, in process only: E33's file is the frozen script and stays byte-identical.
e33.CELLS = list(e33.CELLS) + [LAB_B]
e33.resolve_cell = _resolve
e33._init_kernel = _init

SETTINGS = dict(e33.SETTINGS, loglinear_basis=10, arms=("em_loglinear",), sizes=(128, 512, 2048),
                cells=("T0.3_h1_L32", "T1_iso_L32"))
QUICK = dict(e33.QUICK, arms=("em_loglinear",), loglinear_basis=6)


def _cell(args):
    cfg, cell_id, seed, n = args
    rows = e33._kernels_cell(args)
    if cell_id == LAB_B["id"]:       # the linear floor of the lab-B chain (E33 gets it from its oracles part)
        chain, grid, w, log_k, log_mu, sigma0, A, Av, val, test, base = e33._common(cfg, cell_id, seed, n)
        rows.append({**base, **e33.ROW_DEFAULTS, "arm": "lmmse_true", "selection": "risk", "failed": 0,
                     "risk": e14._pooled(lambda X, t: theory.lmmse_posterior_mean(X, sigma0, t), test)})
    return rows


def part_kernels(cfg, out, workers):
    cells = [(cfg, c, s, n) for n in sorted(cfg["sizes"], reverse=True) for c in cfg["cells"] for s in cfg["seeds"]]
    print(f"  kernels: {len(cells)} cells x {len(cfg['arms'])} arms", flush=True)
    write_csv(out / "kernels.csv", run_cells(cells, _cell, workers, out, "kernels.csv"))


PARTS = {"kernels": part_kernels}


def main():
    ap = experiment_parser("e43_free_kernel", "the price and the benefit of an unrestricted kernel on the phi^4 chain and on an out-of-class chain")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.list_parts:
        print("\n".join(PARTS)); return
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    out = ensure_dir(Path(args.output_dir))
    cells_info = [e33.resolve_cell(c)[1] for c in e33.CELLS if c["id"] in cfg["cells"]]
    write_json(out / "cells.json", cells_info)
    for name, fn in select_parts(PARTS, args.only).items():
        fn(cfg, out, args.workers)
        write_json(out / f"params_{name}.json", {**provenance(), "settings": cfg, "cells": cells_info,
                                                 "quick": bool(args.quick), "workers": args.workers,
                                                 "lab_b": dict(bond=BOND, depth=DEPTH)})


if __name__ == "__main__":
    main()
