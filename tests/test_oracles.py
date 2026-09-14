"""The oracle ladder: isotropic score, LMMSE (covariance twin), true-kernel BP.

Pins (i) that `theory.lmmse_score` equals exact Gaussian-chain BP (Proposition 1 in
its Gaussian instance), (ii) that on the Gaussian chain the LMMSE oracle has zero error
against grid BP, and (iii) the audit's headline number: on the Laplace chain at the
frozen configuration the schedule-pooled LMMSE relative error is 0.17 ± 0.01 and the
isotropic one 0.67 ± 0.01 (docs/AUDIT_ORACLE_LADDER.md), i.e. the floor any
Gaussian-innovation kernel is bound by.
"""

from __future__ import annotations

import numpy as np
import pytest

from markov_score import theory
from markov_score.bp_grid import grid_bp_batch, make_grid
from markov_score.exact_scores import exact_gaussian_posterior_mean
from markov_score.noising import alpha_delta
from markov_score.priors import GaussianAR1, LaplaceAR1


def _pooled(prior, rho, n_chains, t_grid, ng=401, seed=0):
    grid, w = make_grid(8.0, ng)
    L = 32
    sigma0 = rho ** np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
    rng = np.random.default_rng(seed)
    A = np.stack([prior.sample(rng, L) for _ in range(n_chains)])
    logK = prior.log_transition_matrix(grid)
    num_lm = num_iso = den = 0.0
    for t in t_grid:
        al, de = alpha_delta(t)
        X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
        m, _ = grid_bp_batch(grid, w, logK, X, al, de)
        s_bp = theory.score_from_mean(X, m, t)
        s_lm = theory.lmmse_score(X, sigma0, t)
        s_iso = theory.isotropic_score(X)
        num_lm += ((s_lm - s_bp) ** 2).sum()
        num_iso += ((s_iso - s_bp) ** 2).sum()
        den += (s_bp**2).sum()
    return np.sqrt(num_lm / den), np.sqrt(num_iso / den)


def test_lmmse_score_equals_exact_gaussian_posterior_mean():
    rho, L = 0.85, 32
    sigma0 = rho ** np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
    rng = np.random.default_rng(3)
    X = rng.standard_normal((5, L))
    for t in (0.1, 1.0):
        al, de = alpha_delta(t)
        m_ref = np.stack([exact_gaussian_posterior_mean(x, sigma0, al, de) for x in X])
        m = theory.lmmse_posterior_mean(X, sigma0, t)
        assert np.allclose(m, m_ref, atol=1e-12)
        assert np.allclose(theory.lmmse_score(X, sigma0, t), theory.score_from_mean(X, m, t), atol=1e-10)


def test_gaussian_chain_lmmse_error_is_zero():
    t_grid = (0.05, 0.3, 1.0, 3.0)
    lm, iso = _pooled(GaussianAR1(0.85), 0.85, 64, t_grid, ng=401, seed=1)
    assert lm < 1e-10, lm
    assert 0.5 < iso < 0.8


@pytest.mark.slow
def test_laplace_chain_lmmse_floor_is_about_0p17():
    t_grid = tuple(np.exp(np.linspace(np.log(0.05), np.log(3.0), 12)))
    lm, iso = _pooled(LaplaceAR1(0.85), 0.85, 256, t_grid, ng=401, seed=12345)
    assert abs(lm - 0.168) < 0.015, lm
    assert abs(iso - 0.67) < 0.02, iso
