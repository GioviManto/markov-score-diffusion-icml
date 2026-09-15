#!/usr/bin/env python
"""E7 — the oracle-ladder phase diagram over correlation, family and noise level.

Deterministic: for rho in `rhos`, each variance-matched family, and every level of the
frozen schedule, the relative score error of the isotropic score and of the LMMSE oracle
(true covariance) against true-kernel grid BP, per level and schedule-pooled; and the
predicted closure constant C_4 = ||kappa_4||_F^2 / 6 and crossover t_* per rho. The
LMMSE-floor surface is what any second-order learner is bounded by, so it is the map on
which the learned estimators of E02/E03 are placed. Cheap (minutes).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score import theory  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import GaussianMixtureAR1, LaplaceAR1, StudentTAR1, UniformAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

N_SITES = FROZEN.n_sites
T_SCHEDULE = tuple(FROZEN.t_grid)

SETTINGS = dict(rhos=(0.0, 0.3, 0.6, 0.85, 0.95), n_chains=512, grid_size=FROZEN.n_grid, half_width=FROZEN.half_width, reps=2)
QUICK = dict(rhos=(0.3, 0.85), n_chains=32, grid_size=201, reps=1)


def _families(rho):
    return {"laplace": LaplaceAR1(rho), "student": StudentTAR1(rho, 5), "uniform": UniformAR1(rho), "bimodal": GaussianMixtureAR1(rho, 0.6)}


def part_ladder(cfg, out):
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    rows = []
    for rho in cfg["rhos"]:
        q = 1 - rho**2
        sigma0 = theory.chain_covariance(rho, q, N_SITES)
        tstar = theory.crossover_time(rho, q, N_SITES) if rho > 0 else 0.0
        for fam, prior in _families(rho).items():
            logK = prior.log_transition_matrix(grid)
            k4 = theory.excess_kurtosis_to_kappa4(prior.innovation_excess_kurtosis, q)
            C4 = theory.score_gap_constant(theory.chain_kappa4(rho, k4, N_SITES)) if rho > 0 else theory.score_gap_constant(theory.chain_kappa4(1e-9, k4, N_SITES))
            for rep in range(cfg["reps"]):
                rng = rng_for("e07", rho, fam, rep)
                A = np.stack([prior.sample(rng, N_SITES) for _ in range(cfg["n_chains"])])
                Z = rng.standard_normal(A.shape)
                num_lm = num_iso = den = 0.0
                for t in T_SCHEDULE:
                    al, de = alpha_delta(t)
                    X = al * A + np.sqrt(de) * Z
                    m, _ = grid_bp_batch(grid, w, logK, X, al, de)
                    s_bp = theory.score_from_mean(X, m, t)
                    e_lm = ((theory.lmmse_score(X, sigma0, t) - s_bp) ** 2).sum()
                    e_iso = ((theory.isotropic_score(X) - s_bp) ** 2).sum()
                    en = (s_bp**2).sum()
                    num_lm += e_lm; num_iso += e_iso; den += en
                    rows.append(dict(rho=rho, family=fam, excess_kurtosis=prior.innovation_excess_kurtosis, rep=rep, t=t, t_star=tstar, C4=C4,
                                     rel_lmmse=np.sqrt(e_lm / en), rel_iso=np.sqrt(e_iso / en), ref_energy=en / A.size, pooled=False))
                rows.append(dict(rho=rho, family=fam, excess_kurtosis=prior.innovation_excess_kurtosis, rep=rep, t=np.nan, t_star=tstar, C4=C4,
                                 rel_lmmse=np.sqrt(num_lm / den), rel_iso=np.sqrt(num_iso / den), ref_energy=den / A.size / len(T_SCHEDULE), pooled=True))
            print(f"  [E07] rho={rho} {fam:8s} pooled lmmse {rows[-1]['rel_lmmse']:.3f} iso {rows[-1]['rel_iso']:.3f}  t*={tstar:.2f}", flush=True)
    write_csv(out / "ladder.csv", rows)


PARTS = {"ladder": part_ladder}


def main():
    ap = experiment_parser("e07_phase_diagram", "oracle ladder over rho, family and level")
    args = ap.parse_args()
    if args.list_parts:
        print("\n".join(PARTS)); return
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    out = ensure_dir(Path(args.output_dir))
    for name, fn in select_parts(PARTS, args.only).items():
        print(f"[e07] part {name}", flush=True)
        fn(cfg, out)
        write_json(out / f"params_{name}.json", {**provenance(), "settings": cfg, "quick": bool(args.quick), "frozen": FROZEN.as_dict()})


if __name__ == "__main__":
    main()
