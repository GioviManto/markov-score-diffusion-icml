#!/usr/bin/env python
"""E1 — order-by-order information decay: exact curves, shape information, closure gap.

Tests Theorems A/B of docs/THEORY_NOTES.md on the chain. Three parts, all local:

  gauss   closed-form Fisher information of the Gaussian chain about (rho, q) on a fine
          t-grid, for several rho: exact curves, local log-slopes, the e^{-4t} asymptote
          with its predicted constant, the crossover t_*, the rho–q correlation.
  shape   outer-product information about (rho, q, beta) for generalised-Gaussian
          chains by batched Fisher-identity scores (markov_score.information), twelve
          levels to t = 3 plus the frozen schedule, N_g = 801, D chains x R replicate
          seeds. The Gaussian member (beta = 2) is compared with `gauss` to locate the
          numerical floor of the outer-product estimate.
  closure relative score error, posterior-mean MSE and 1 - cos of the LMMSE oracle
          against true-kernel grid BP, five families, paired draws across t, N_g = 801,
          to t = 4; plus the predicted constant C_4 = ||kappa_4||_F^2 / 6 for each family.

Every part writes CSVs and a params_<part>.json with full provenance into --out.
Slopes are NOT fitted here; tools/make_fig_theory.py fits them from the CSVs so the
fit window is one declared place.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score import theory  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.information import outer_product_information, per_chain_scores, summarise  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import GaussianAR1, GaussianMixtureAR1, LaplaceAR1, StudentTAR1, UniformAR1  # noqa: E402
from markov_score.shape_kernel import GeneralizedGaussianKernel, generalized_gaussian_excess_kurtosis  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

L = FROZEN.n_sites
T12 = tuple(FROZEN.t_grid)
T_FINE = tuple(float(x) for x in np.round(np.exp(np.linspace(np.log(0.02), np.log(6.0), 40)), 5))
T_EXT = tuple(sorted(set(T12) | {3.5, 4.0}))

SETTINGS = dict(
    rhos=(0.0, 0.3, 0.6, 0.85, 0.95),
    betas=(1.0, 1.25, 1.5, 2.0, 3.0),
    n_grid=801, half_width=8.0,
    n_chains=512, n_reps=8,
    closure_chains=512, closure_reps=4,
)
QUICK = dict(rhos=(0.85,), betas=(1.0, 2.0), n_grid=201, n_chains=32, n_reps=2, closure_chains=32, closure_reps=1)


def part_gauss(cfg, out):
    rows = []
    for rho in cfg["rhos"]:
        q = 1.0 - rho**2
        ts = theory.crossover_time(rho, q, L) if rho > 0 else 0.0
        for t in T_FINE:
            r = theory.gaussian_chain_fisher(rho, q, L, t)
            rows.append(dict(rho=rho, q=q, t=t, t_star=ts,
                             I_rho=r["diag"][0], I_q=r["diag"][1], I_rho_q=r["I"][0, 1],
                             eff_rho=r["efficient"][0], eff_q=r["efficient"][1],
                             corr_rho_q=r["corr_rho_q"],
                             asymptote_rho=r["asymptote_rho"], asymptote_q=r["asymptote_q"],
                             I_rho_scaled=r["diag"][0] * np.exp(4 * t) / r["asymptote_rho"],
                             I_q_scaled=r["diag"][1] * np.exp(4 * t) / r["asymptote_q"]))
    write_csv(out / "gaussian_information.csv", rows)
    print(f"  gauss: {len(rows)} rows; t_* at rho=0.85 = {theory.crossover_time(0.85, None, L):.3f}")


def _sample_gg(rng, kernel, n):
    from exp_22_shape_information import sample_generalized_gaussian
    return np.stack([sample_generalized_gaussian(rng, kernel, L) for _ in range(n)])


def part_shape(cfg, out):
    grid, w = make_grid(cfg["half_width"], cfg["n_grid"])
    rho, q = FROZEN.rho, FROZEN.innovation_variance
    rows = []
    for beta in cfg["betas"]:
        k = GeneralizedGaussianKernel(rho=rho, q=q, beta=beta)
        k4 = generalized_gaussian_excess_kurtosis(beta)
        for rep in range(cfg["n_reps"]):
            rng = rng_for("e01-shape", beta, rep)
            A = _sample_gg(rng, k, cfg["n_chains"])
            Z = rng.standard_normal(A.shape)          # common noise across t within a replicate
            for t in T_EXT:
                al, de = alpha_delta(t)
                X = al * A + np.sqrt(de) * Z
                g = per_chain_scores(grid, w, k, X, al, de)
                J = outer_product_information(g)
                s = summarise(J)
                rows.append(dict(beta=beta, excess_kurtosis=k4, rep=rep, t=t, n_chains=cfg["n_chains"],
                                 diag_rho=s["diag"][0], diag_q=s["diag"][1], diag_beta=s["diag"][2],
                                 eff_rho=s["efficient"][0], eff_q=s["efficient"][1], eff_beta=s["efficient"][2],
                                 corr_rho_q=s["corr"][0, 1], corr_rho_beta=s["corr"][0, 2], corr_q_beta=s["corr"][1, 2],
                                 cond=s["condition_number"]))
            print(f"  shape: beta={beta} rep={rep} done")
    write_csv(out / "shape_information.csv", rows)


def _families(rho):
    return {
        "laplace": LaplaceAR1(rho), "gaussian": GaussianAR1(rho),
        "student": StudentTAR1(rho, 5), "uniform": UniformAR1(rho), "bimodal": GaussianMixtureAR1(rho, 0.6),
    }


def part_closure(cfg, out):
    grid, w = make_grid(cfg["half_width"], cfg["n_grid"])
    rho = FROZEN.rho
    sigma0 = theory.chain_covariance(rho, 1 - rho**2, L)
    rows, consts = [], []
    for name, prior in _families(rho).items():
        logK = prior.log_transition_matrix(grid)
        k4_eps = theory.excess_kurtosis_to_kappa4(prior.innovation_excess_kurtosis, prior.q if hasattr(prior, "q") else 1 - rho**2)
        K4 = theory.chain_kappa4(rho, k4_eps, L)
        C4 = theory.score_gap_constant(K4)
        consts.append(dict(family=name, excess_kurtosis=prior.innovation_excess_kurtosis, kappa4_eps=k4_eps,
                           kappa4_frobenius_sq=float((K4**2).sum()), C4=C4, C4_per_site=C4 / L))
        for rep in range(cfg["closure_reps"]):
            rng = rng_for("e01-closure", name, rep)
            A = np.stack([prior.sample(rng, L) for _ in range(cfg["closure_chains"])])
            Z = rng.standard_normal(A.shape)
            for t in T_EXT:
                al, de = alpha_delta(t)
                X = al * A + np.sqrt(de) * Z
                m_bp, _ = grid_bp_batch(grid, w, logK, X, al, de)
                s_bp = theory.score_from_mean(X, m_bp, t)
                m_lm = theory.lmmse_posterior_mean(X, sigma0, t)
                s_lm = theory.score_from_mean(X, m_lm, t)
                s_iso = theory.isotropic_score(X)
                e_lm = ((s_lm - s_bp) ** 2).sum(); e_iso = ((s_iso - s_bp) ** 2).sum(); en = (s_bp**2).sum()
                pm = ((m_lm - m_bp) ** 2).sum() / A.size
                cos = np.mean(np.sum(s_lm * s_bp, 1) / np.sqrt(np.sum(s_lm**2, 1) * np.sum(s_bp**2, 1)))
                rows.append(dict(family=name, rep=rep, t=t, rel_lmmse=np.sqrt(e_lm / en), rel_iso=np.sqrt(e_iso / en),
                                 sq_gap_per_chain=e_lm / A.shape[0], ref_energy_per_chain=en / A.shape[0],
                                 pm_mse=pm, one_minus_cos=1 - cos,
                                 pred_sq_gap_per_chain=C4 * np.exp(-8 * t)))
        print(f"  closure: {name} done (C4={C4:.4g})")
    write_csv(out / "closure_gap.csv", rows)
    write_csv(out / "closure_constants.csv", consts)


PARTS = {"gauss": part_gauss, "shape": part_shape, "closure": part_closure}


def main():
    ap = experiment_parser("e01_information_decay", "order-by-order information decay (Theorems A/B)")
    args = ap.parse_args()
    if args.list_parts:
        print("\n".join(PARTS)); return
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    out = ensure_dir(Path(args.output_dir))
    for name, fn in select_parts(PARTS, args.only).items():
        print(f"[e01] part {name}")
        fn(cfg, out)
        write_json(out / f"params_{name}.json", {**provenance(), "settings": cfg, "t_ext": T_EXT, "t_fine": T_FINE,
                                                   "frozen": FROZEN.as_dict(), "quick": bool(args.quick)})


if __name__ == "__main__":
    main()
