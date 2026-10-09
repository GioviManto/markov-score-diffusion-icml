"""E50's commitment measurement: the sign probabilities of grid BP, the Gaussian twin's closed form, the crossing time."""
from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.stats import norm

from markov_score.bp_grid import grid_bp_batch, make_grid
from markov_score.commitment import bulk_sites, commitment, crossing_time, gaussian_twin, sign_correct


def _small_problem(seed=0, n_grid=15, n_sites=3, n_chains=4, symmetric=False):
    rng = np.random.default_rng(seed)
    grid, w = make_grid(3.0, n_grid)
    if symmetric:
        log_k = -0.5 * (grid[:, None] - 0.6 * grid[None, :]) ** 2 / 0.5 - 0.1 * grid[:, None] ** 4
        log_mu = -0.3 * grid**2 - 0.05 * grid**4
    else:
        log_k = rng.normal(size=(n_grid, n_grid))
        log_mu = rng.normal(size=n_grid)
    log_k = log_k - np.log(np.exp(log_k).T @ w)[None, :]          # every column a density under w
    log_mu = log_mu - np.log(np.exp(log_mu) @ w)
    X = rng.normal(size=(n_chains, n_sites))
    return grid, w, log_k, log_mu, X


def test_sign_probability_matches_enumeration():
    grid, w, log_k, log_mu, X = _small_problem()
    alpha, delta = 0.8, 0.36
    _, _, p = grid_bp_batch(grid, w, log_k, X, alpha, delta, log_mu=log_mu, return_sign_prob=True)
    K, mu = np.exp(log_k), np.exp(log_mu)
    pos = np.where(grid > 0, 1.0, np.where(grid == 0, 0.5, 0.0))
    n, L = len(grid), X.shape[1]
    for b in range(X.shape[0]):
        ell = np.exp(-0.5 * (X[b][:, None] - alpha * grid[None, :]) ** 2 / delta)   # (L, n)
        num, tot = np.zeros(L), 0.0
        for idx in itertools.product(range(n), repeat=L):
            m = mu[idx[0]] * w[idx[0]] * ell[0, idx[0]]
            for i in range(1, L):
                m *= K[idx[i], idx[i - 1]] * w[idx[i]] * ell[i, idx[i]]
            tot += m
            num += m * pos[list(idx)]
        np.testing.assert_allclose(p[b], num / tot, rtol=1e-10, atol=1e-13)


def test_sign_probability_leaves_means_and_variances_unchanged_and_chunks():
    grid, w, log_k, log_mu, X = _small_problem(seed=1, n_chains=7, n_sites=5)
    m0, v0 = grid_bp_batch(grid, w, log_k, X, 0.7, 0.51, log_mu=log_mu)
    m1, v1, p1 = grid_bp_batch(grid, w, log_k, X, 0.7, 0.51, log_mu=log_mu, return_sign_prob=True)
    m2, v2, p2 = grid_bp_batch(grid, w, log_k, X, 0.7, 0.51, log_mu=log_mu, return_sign_prob=True, chunk=3)
    np.testing.assert_array_equal(m0, m1)
    np.testing.assert_array_equal(v0, v1)
    np.testing.assert_allclose(m2, m1, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(p2, p1, rtol=1e-12, atol=1e-14)
    assert np.all((p1 >= 0) & (p1 <= 1))


def test_sign_probability_reflection_symmetry():
    grid, w, log_k, log_mu, X = _small_problem(seed=2, symmetric=True)
    _, _, p = grid_bp_batch(grid, w, log_k, X, 0.6, 0.64, log_mu=log_mu, return_sign_prob=True)
    _, _, q = grid_bp_batch(grid, w, log_k, -X, 0.6, 0.64, log_mu=log_mu, return_sign_prob=True)
    np.testing.assert_allclose(p + q, 1.0, atol=1e-12)


def test_sign_probability_refuses_other_outputs():
    grid, w, log_k, log_mu, X = _small_problem()
    with pytest.raises(ValueError):
        grid_bp_batch(grid, w, log_k, X, 0.8, 0.36, log_mu=log_mu, return_sign_prob=True, return_evidence=True)


def test_gaussian_twin_closed_form_matches_monte_carlo():
    rng = np.random.default_rng(3)
    L, rho, n = 6, 0.8, 200_000
    lag = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
    cov = rho**lag
    alpha, delta = 0.5, 0.75
    c_closed, v_closed = gaussian_twin(cov, alpha, delta)
    a = rng.multivariate_normal(np.zeros(L), cov, size=n)
    x = alpha * a + np.sqrt(delta) * rng.standard_normal((n, L))
    S_t = alpha**2 * cov + delta * np.eye(L)
    mu = alpha * x @ np.linalg.solve(S_t, cov)                  # rows: alpha Sigma S_t^{-1} x (Sigma symmetric)
    v = np.diag(cov - alpha**2 * cov @ np.linalg.solve(S_t, cov))
    np.testing.assert_allclose(v_closed, v, rtol=1e-12)
    c_mc = commitment(norm.cdf(mu / np.sqrt(v))).mean(axis=0)
    np.testing.assert_allclose(c_closed, c_mc, atol=4e-3)
    # calibration in the twin world: the posterior's sign guess is right (1 + c) / 2 of the time
    acc = sign_correct(norm.cdf(mu / np.sqrt(v)), a).mean(axis=0)
    np.testing.assert_allclose(acc, 0.5 * (1 + c_closed), atol=4e-3)


def test_gaussian_twin_crosses_one_half_where_posterior_variance_halves():
    cov = np.array([[1.0]])
    # scalar: v = delta / (alpha^2 + delta); v = 1/2 at alpha^2 = delta, i.e. e^{-2t} = 1 - e^{-2t}
    t = 0.5 * np.log(2.0)
    alpha, delta = np.exp(-t), 1 - np.exp(-2 * t)
    c, v = gaussian_twin(cov, alpha, delta)
    assert v[0] == pytest.approx(0.5)
    assert c[0] == pytest.approx(0.5)


def test_crossing_time():
    ts = np.geomspace(0.01, 10.0, 400)
    assert crossing_time(ts, np.exp(-ts)) == pytest.approx(np.log(2.0), rel=1e-3)
    assert np.isnan(crossing_time(ts, 0.4 * np.ones_like(ts)))          # starts below the level
    assert np.isnan(crossing_time(ts, 0.9 * np.ones_like(ts)))          # never falls through it
    assert crossing_time(ts[::-1], np.exp(-ts)[::-1]) == pytest.approx(np.log(2.0), rel=1e-3)   # order-free


def test_bulk_sites():
    assert bulk_sites(4).tolist() == [1, 2]
    assert bulk_sites(8).tolist() == [2, 3, 4, 5]
    assert bulk_sites(32).tolist() == list(range(8, 24))
    assert bulk_sites(256).tolist() == list(range(120, 136))


def test_exact_posterior_is_calibrated_on_the_field():
    from markov_score.field import Phi4Chain
    from markov_score.noising import alpha_delta
    chain = Phi4Chain(0.5, 1.0)
    grid, w = make_grid(8.0, 201)
    rng = np.random.default_rng(4)
    A = chain.sample_batch(rng, 1024, 8)
    al, de = alpha_delta(0.6)
    X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
    _, _, p = grid_bp_batch(grid, w, chain.log_transition_matrix(grid), X, al, de,
                            log_mu=chain.log_stationary(grid), return_sign_prob=True)
    assert abs(sign_correct(p, A).mean() - 0.5 * (1 + commitment(p).mean())) < 0.02
