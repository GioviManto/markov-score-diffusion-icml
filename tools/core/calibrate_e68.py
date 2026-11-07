"""Calibrate E68's non-stationary chains from their TRUE kernels only (no learned arm, no training data).

    PYTHONPATH=src:experiments python tools/core/calibrate_e68.py [d_rho d_logsd ...]

For each of E68's cells (drift at delta 0, 0.5, 1; switch at delta 0.25, 0.5), on a calibration-only bundle (tag e68-calibration,
256 chains, the twelve levels), against exact per-link BP with the true kernels on 401 points:

    linear_floor    the LMMSE with the chain's exact covariance
    shared_floor    BP with the best single kernel for every link: the clean-data maximum-likelihood limit of a shared
                    kernel, i.e. the pooled transition law K(a' | a) = sum_k p_k(a) K_k(a' | a) / sum_k p_k(a), with the
                    true first-site law (what one stationary kernel can at best do on these data)
    e03_kernel      BP with E03's stationary kernel (rho 0.85, s 1) and N(0, 1): the delta = 0 truth used as if nothing
                    had changed
    grid_201        relative change of the reference score from 401 to 201 points (the arms' grid)

Pass (d_rho, d_logsd) pairs to try other strengths; the defaults are E68's registered ones.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments")]

from frozen_config import FROZEN

from markov_score.bp_grid import make_grid
from markov_score.noising import alpha_delta
from markov_score.nonstationary import NonstationaryLaplaceChain
from markov_score.priors import LaplaceAR1
from markov_score.site_bp import site_bp_batch
from markov_score.utils import rng_for

T_SCHEDULE = tuple(FROZEN.t_grid)
DELTAS = {"drift": (0.0, 0.5, 1.0), "switch": (0.25, 0.5)}      # E68's cells


def pooled_kernel(chain, grid, w):
    lks, log_w = chain.log_kernels(grid), np.log(w)
    logp = [chain.log_mu(grid)]
    for k in range(chain.n_sites - 2):
        logp.append(logsumexp(lks[k] + log_w[None, :] + logp[-1][None, :], axis=1))
    log_pair = logsumexp(np.stack([lks[k] + logp[k][None, :] for k in range(chain.n_sites - 1)]), axis=0)
    return log_pair - logsumexp(log_pair + log_w[:, None], axis=0, keepdims=True)


def calibrate(chain, n_chains=256):
    g4, w4 = make_grid(8.0, 401)
    g2, w2 = make_grid(8.0, 201)
    A = chain.sample(rng_for("e68-calibration", chain.schedule, chain.delta), n_chains)
    noise = rng_for("e68-calibration-noise", chain.schedule, chain.delta)
    lk_true, mu_true = chain.log_kernels(g4), chain.log_mu(g4)
    lk_shared = pooled_kernel(chain, g4, w4)
    lk_e03 = LaplaceAR1(rho=0.85).log_transition_matrix(g4)
    mu_e03 = -0.5 * g4**2 - 0.5 * np.log(2 * np.pi)
    cov = chain.covariance()
    num = {k: 0.0 for k in ("linear_floor", "shared_floor", "e03_kernel", "grid_201")}
    den = 0.0
    for t in T_SCHEDULE:
        al, de = alpha_delta(t)
        X = al * A + np.sqrt(de) * noise.standard_normal(A.shape)
        def score(m, al=al, de=de, X=X):
            return (al * m - X) / de
        ref = score(site_bp_batch(g4, w4, lk_true, X, al, de, mu_true).means)
        est = {"linear_floor": score((al * cov @ np.linalg.solve(al**2 * cov + de * np.eye(len(cov)), X.T)).T),
               "shared_floor": score(site_bp_batch(g4, w4, lk_shared, X, al, de, mu_true).means),
               "e03_kernel": score(site_bp_batch(g4, w4, lk_e03, X, al, de, mu_e03).means),
               "grid_201": score(site_bp_batch(g2, w2, chain.log_kernels(g2), X, al, de, chain.log_mu(g2)).means)}
        den += float((ref**2).sum())
        for k, s in est.items():
            num[k] += float(((s - ref) ** 2).sum())
    return {k: float(np.sqrt(v / den)) for k, v in num.items()}


def main(argv):
    pairs = [(float(argv[i]), float(argv[i + 1])) for i in range(0, len(argv), 2)] or [(0.1, 0.35)]
    out = {}
    for d_rho, d_logsd in pairs:
        for schedule in ("drift", "switch"):
            for delta in DELTAS[schedule]:
                ch = NonstationaryLaplaceChain(schedule, delta, d_rho=d_rho, d_logsd=d_logsd)
                r = calibrate(ch)
                key = f"d_rho={d_rho:g} d_logsd={d_logsd:g} {ch.name}"
                out[key] = {"rho_range": [float(ch.rho().min()), float(ch.rho().max())],
                            "sd_range": [float(ch.sd().min()), float(ch.sd().max())], **{k: round(v, 5) for k, v in r.items()}}
                print(key, json.dumps(out[key]), flush=True)
    return out


if __name__ == "__main__":
    main(sys.argv[1:])
