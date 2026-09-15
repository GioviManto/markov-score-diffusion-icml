"""Generation metrics: zero on identical samples, positive on distinct laws, and the BP evidence
log-likelihood recovers the Gaussian closed form."""

from __future__ import annotations

import numpy as np

from markov_score import gen_metrics, theory
from markov_score.bp_grid import make_grid
from markov_score.exact_scores import gaussian_log_evidence
from markov_score.noising import alpha_delta
from markov_score.priors import GaussianAR1, LaplaceAR1


def test_metrics_vanish_on_identical_samples_and_separate_laws():
    rng = np.random.default_rng(0)
    L, rho = 16, 0.7
    lap, gau = LaplaceAR1(rho), GaussianAR1(rho)
    A = np.stack([lap.sample(rng, L) for _ in range(600)])
    B = np.stack([gau.sample(rng, L) for _ in range(600)])
    assert gen_metrics.sliced_wasserstein2(A, A) < 1e-12
    assert gen_metrics.energy_distance(A, A) < 1e-9
    assert gen_metrics.sliced_wasserstein2(A, B) > 0.02
    ra, rb = gen_metrics.ar_residuals(A, rho), gen_metrics.ar_residuals(B, rho)
    assert gen_metrics.excess_kurtosis(ra) > 1.5 and abs(gen_metrics.excess_kurtosis(rb)) < 0.5
    assert gen_metrics.histogram_kl(ra.ravel(), rb.ravel()) > 0.01


def test_bp_log_likelihood_matches_gaussian_closed_form():
    L, rho, t = 12, 0.6, 0.4
    grid, w = make_grid(8.0, 401)
    prior = GaussianAR1(rho)
    rng = np.random.default_rng(2)
    A = np.stack([prior.sample(rng, L) for _ in range(20)])
    al, de = alpha_delta(t)
    X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
    ll = gen_metrics.log_likelihood_true(X, grid, w, prior.log_transition_matrix(grid), t)
    sigma0 = theory.chain_covariance(rho, 1 - rho**2, L)
    ref = np.array([gaussian_log_evidence(x, sigma0, al, de) for x in X])
    assert np.abs(ll - ref).max() < 1e-6
