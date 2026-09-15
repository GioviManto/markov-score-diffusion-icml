"""Batched per-chain Fisher-identity scores agree with the legacy one-chain-at-a-time E-step."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from markov_score.bp_grid import make_grid
from markov_score.em import e_step_multi, q_gradient
from markov_score.information import efficient_information, outer_product_information, per_chain_scores, summarise
from markov_score.noising import alpha_delta
from markov_score.shape_kernel import GeneralizedGaussianKernel


def _chains(kernel, n, rng, L=16):
    from exp_22_shape_information import sample_generalized_gaussian
    return np.stack([sample_generalized_gaussian(rng, kernel, L) for _ in range(n)])


def test_batched_scores_match_legacy_loop():
    grid, w = make_grid(8.0, 201)
    k = GeneralizedGaussianKernel(rho=0.7, q=1 - 0.7**2, beta=1.5)
    rng = np.random.default_rng(0)
    A = _chains(k, 12, rng)
    for t in (0.2, 1.0):
        al, de = alpha_delta(t)
        X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
        g_new = per_chain_scores(grid, w, k, X, al, de)
        grad = k.grad_log_transition_matrix(grid); logk = k.log_transition_matrix(grid)
        g_old = np.stack([q_gradient(e_step_multi(grid, w, logk, [(x[None, :], al, de)]), grad) for x in X])
        assert np.abs(g_new - g_old).max() <= 1e-10 * np.abs(g_old).max()


def test_information_summary_is_consistent():
    g = np.random.default_rng(1).standard_normal((500, 3)) @ np.diag([3.0, 1.0, 0.1])
    J = outer_product_information(g)
    s = summarise(J)
    assert np.allclose(s["efficient"], efficient_information(J))
    assert np.all(s["nuisance_cost"] >= 1.0 - 1e-12)
    assert abs(s["corr"][0, 0] - 1.0) < 1e-12
