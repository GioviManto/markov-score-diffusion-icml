"""The regime kernel on the doubled grid: valid densities, exact reduction to the Gaussian chain, EM recovers it."""
from __future__ import annotations

import numpy as np

from markov_score.bp_grid import grid_bp_batch, make_grid
from markov_score.em import clean_statistics, fit_em
from markov_score.noising import alpha_delta
from markov_score.priors import GaussianAR1
from markov_score.regime_kernel import RegimeKernel, doubled, sample_regime_chain


def test_columns_are_densities():
    g, w = make_grid(8.0, 201)
    G, W = doubled(g, w)
    K = np.exp(RegimeKernel.init(0.8, 0.3, 0.9, 0.95, 0.9).log_transition_matrix(G))
    assert np.allclose((K * W[:, None]).sum(axis=0), 1.0, atol=1e-12)


def test_no_switching_and_equal_scales_is_the_gaussian_chain():
    """p00 = p11 = 1 with s0 = s1: BP on the doubled grid gives the Gaussian chain's posterior mean."""
    g, w = make_grid(8.0, 401)
    G, W = doubled(g, w)
    s = np.sqrt(1 - 0.8 ** 2)
    k = RegimeKernel(0.8, s, s, 1.0 - 1e-12, 1.0 - 1e-12)
    X = np.random.default_rng(0).standard_normal((4, 16))
    al, de = alpha_delta(0.3)
    log_mu = np.concatenate([-0.5 * g ** 2, -0.5 * g ** 2]) - np.log(2.0)
    m2 = grid_bp_batch(G, W, k.log_transition_matrix(G), X, al, de, log_mu=log_mu)[0]
    m1 = grid_bp_batch(g, w, GaussianAR1(0.8).log_transition_matrix(g), X, al, de)[0]
    assert np.max(np.abs(m2 - m1)) < 1e-8


def test_em_recovers_the_regimes():
    truth = RegimeKernel(0.85, 0.27, 0.80, 0.95, 0.95)
    rng = np.random.default_rng(1)
    A = sample_regime_chain(truth, rng, 400, 32)
    g, w = make_grid(8.0, 161)
    G, W = doubled(g, w)
    al, de = alpha_delta(0.1)
    X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
    log_mu = np.concatenate([-0.5 * g ** 2, -0.5 * g ** 2]) - np.log(2.0)
    k, _ = fit_em(RegimeKernel.init(0.5, 0.4, 1.0, 0.8, 0.8), G, W, [(X, al, de)], n_iters=60, log_mu=log_mu)
    assert abs(k.alpha - 0.85) < 0.05
    assert abs(k.p00 - 0.95) < 0.05 and abs(k.p11 - 0.95) < 0.05
    assert abs(k.s1 / k.s0 - truth.s1 / truth.s0) / (truth.s1 / truth.s0) < 0.2


def test_m_step_on_clean_statistics_is_finite():
    g, w = make_grid(8.0, 101)
    G, _ = doubled(g, w)
    A = sample_regime_chain(RegimeKernel(0.85, 0.27, 0.8, 0.95, 0.95), np.random.default_rng(2), 50, 32)
    stats = clean_statistics(G, A)
    k = RegimeKernel.init().m_step(stats, G)
    assert np.all(np.isfinite(k.theta))
