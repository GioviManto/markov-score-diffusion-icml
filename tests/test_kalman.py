"""The Kalman/RTS smoother against the closed form and against information-form BP.

The proposition on the Gaussian chain (paper, appendix A) says three computations of one conditional law agree:
the closed form alpha Sigma0 Sigma_t^{-1} x, the RTS smoother, and sum-product. These tests pin the first two
to each other and to `bp_gaussian.gaussian_chain_bp`; E39 then measures grid BP against both.
"""
import numpy as np
import pytest

from markov_score.bp_gaussian import gaussian_chain_bp
from markov_score.kalman import closed_form_gaussian, kalman_rts
from markov_score.noising import alpha_delta
from markov_score.priors import GaussianAR1


def _data(t, n_chains=40, n_sites=17, rho=0.85, seed=3):
    rng = np.random.default_rng(seed)
    prior = GaussianAR1(rho)
    A = np.stack([prior.sample(rng, n_sites) for _ in range(n_chains)])
    alpha, delta = alpha_delta(t)
    X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
    return prior, X, alpha, delta


@pytest.mark.parametrize("t", [0.05, 0.4, 1.5, 3.0])
def test_rts_equals_closed_form(t):
    prior, X, alpha, delta = _data(t)
    k = kalman_rts(X, prior.rho, prior.q, alpha, delta)
    means, scores = closed_form_gaussian(X, prior.covariance(X.shape[1]), alpha, delta)
    assert np.max(np.abs(k.smoothed_means - means)) < 1e-12
    s_rts = -(X - alpha * k.smoothed_means) / delta          # Tweedie on the smoother's mean
    assert np.linalg.norm(s_rts - scores) / np.linalg.norm(scores) < 1e-12


@pytest.mark.parametrize("t", [0.05, 1.0])
def test_rts_variances_are_the_posterior_diagonal(t):
    prior, X, alpha, delta = _data(t)
    k = kalman_rts(X, prior.rho, prior.q, alpha, delta)
    s0 = prior.covariance(X.shape[1])
    post = np.linalg.inv(np.linalg.inv(s0) + alpha**2 / delta * np.eye(len(s0)))
    assert np.allclose(k.smoothed_vars, np.diag(post), rtol=1e-11, atol=0)


def test_rts_equals_information_form_bp_and_filter_ends_agree():
    prior, X, alpha, delta = _data(0.7)
    k = kalman_rts(X, prior.rho, prior.q, alpha, delta)
    for b in range(X.shape[0]):
        r = gaussian_chain_bp(X[b], prior.rho, prior.q, alpha, delta)
        assert np.max(np.abs(r.means - k.smoothed_means[b])) < 1e-12
        assert np.max(np.abs(r.variances - k.smoothed_vars)) < 1e-12
    # at the last site, filtering and smoothing condition on the same data
    assert np.array_equal(k.filtered_means[:, -1], k.smoothed_means[:, -1])
