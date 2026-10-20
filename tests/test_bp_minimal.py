"""The minimal grid BP equals the production one, and both equal the closed form on a Gaussian chain."""
import numpy as np
import pytest

from markov_score import bp_minimal
from markov_score.bp_grid import grid_bp_batch
from markov_score.field import Phi4Chain
from markov_score.priors import GaussianAR1, LaplaceAR1
from markov_score.theory import lmmse_score

L, B = 32, 6
LEVELS = (0.05, 0.5, 2.0)


def _chains(name):
    rng = np.random.default_rng(7)
    if name == "field":
        chain = Phi4Chain(0.3, 1.0)
        return chain, chain.sample_batch(rng, B, L)
    chain = {"gaussian": GaussianAR1(0.85), "laplace": LaplaceAR1(0.85)}[name]
    return chain, np.stack([chain.sample(rng, L) for _ in range(B)])


@pytest.mark.parametrize("name", ["gaussian", "laplace", "field"])
def test_minimal_equals_production(name):
    chain, A = _chains(name)
    u, w = bp_minimal.make_grid(8.0, 401)
    log_K = chain.log_transition_matrix(u)
    log_mu = chain.log_stationary(u) if name == "field" else -0.5 * u**2 - 0.5 * np.log(2 * np.pi)
    rng = np.random.default_rng(11)
    for t in LEVELS:
        al, de = np.exp(-t), 1 - np.exp(-2 * t)
        X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
        m_prod, _ = grid_bp_batch(u, w, log_K, X, al, de, log_mu=log_mu)
        m_min = np.stack([bp_minimal.posterior_mean(x, t, np.exp(log_K), np.exp(log_mu), u, w) for x in X])
        assert np.max(np.abs(m_min - m_prod)) < 1e-10


def test_minimal_is_exact_on_the_gaussian_chain():
    chain, A = _chains("gaussian")
    u, w = bp_minimal.make_grid(8.0, 401)
    K, mu = np.exp(chain.log_transition_matrix(u)), np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
    rng = np.random.default_rng(12)
    for t in (0.5, 2.0):
        al, de = np.exp(-t), 1 - np.exp(-2 * t)
        X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
        s_bp = np.stack([bp_minimal.score(x, t, K, mu, u, w) for x in X])
        s_exact = lmmse_score(X, chain.covariance(L), t)
        assert np.max(np.abs(s_bp - s_exact)) / np.max(np.abs(s_exact)) < 1e-8
