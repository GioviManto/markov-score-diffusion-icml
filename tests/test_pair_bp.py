"""Pair-state BP (markov_score.pair_bp): it IS first-order grid BP when the kernel ignores a_{t-2}, and it returns the
exact Gaussian posterior mean of a stationary Gaussian AR(2) (up to the grid)."""

from __future__ import annotations

import numpy as np

from markov_score.bp_grid import grid_bp_batch, make_grid
from markov_score.pair_bp import pair_bp_batch


def _norm(lk, w, axis=0):
    return lk - np.log(np.sum(np.exp(lk) * np.expand_dims(w, tuple(i for i in range(lk.ndim) if i != axis)), axis=axis,
                              keepdims=True))


def test_a_kernel_that_ignores_the_second_lag_is_first_order_bp():
    grid, w = make_grid(6.0, 61)
    rng = np.random.default_rng(0)
    lk1 = _norm(-0.5 * (grid[:, None] - 0.7 * grid[None, :]) ** 2 / 0.51 - 0.3 * np.abs(grid[:, None]), w)
    log_mu = np.log(np.exp(-0.5 * grid ** 2) / np.sum(np.exp(-0.5 * grid ** 2) * w))
    lk2 = np.repeat(lk1[:, :, None], len(grid), axis=2)                          # [k, j, i], no dependence on i
    log_mu2 = lk1 + log_mu[None, :]                                              # p(a_2 = j, a_1 = i)
    X = rng.standard_normal((7, 12)) * 1.3
    for al, de in ((0.9, 0.19), (0.4, 0.84)):
        m1, v1 = grid_bp_batch(grid, w, lk1, X, al, de, log_mu=log_mu)
        m2, v2 = pair_bp_batch(grid, w, lk2, X, al, de, log_mu2)
        np.testing.assert_allclose(m2, m1, atol=1e-10)
        np.testing.assert_allclose(v2, v1, atol=1e-10)
    m3, _ = pair_bp_batch(grid, w, lk2, X, 0.9, 0.19, log_mu2, chunk=3)         # chunking changes nothing
    np.testing.assert_allclose(m3, grid_bp_batch(grid, w, lk1, X, 0.9, 0.19, log_mu=log_mu)[0], atol=1e-10)


def test_the_gaussian_ar2_posterior_mean_is_exact():
    grid, w = make_grid(7.0, 141)
    r1, p2 = 0.85, 0.4
    p1 = r1 * (1 - p2)
    s2 = (1 - r1 ** 2) * (1 - p2 ** 2)                                           # unit stationary variance
    L = 10
    rho = [1.0, r1]
    for _ in range(2, L):
        rho.append(p1 * rho[-1] + p2 * rho[-2])
    i = np.arange(L)
    cov = np.array(rho)[np.abs(i[:, None] - i[None, :])]
    a, ap, app = grid[None, None, :], grid[None, :, None], grid[:, None, None]  # [k, j, i]: app = a_t, ap = a_{t-1}
    lk2 = _norm(-0.5 * (app - p1 * ap - p2 * a) ** 2 / s2 + 0 * a, w)
    c2 = np.array([[1.0, r1], [r1, 1.0]])
    pts = np.stack(np.meshgrid(grid, grid, indexing="ij"), -1)                  # [j = a_2, i = a_1]
    q = np.einsum("jix,xy,jiy->ji", pts[..., ::-1], np.linalg.inv(c2), pts[..., ::-1])
    log_mu2 = -0.5 * q
    log_mu2 -= np.log(np.sum(np.exp(log_mu2) * w[:, None] * w[None, :]))
    rng = np.random.default_rng(1)
    A = rng.multivariate_normal(np.zeros(L), cov, size=20)
    for al, de in ((0.95, 0.0975), (0.6, 0.64)):
        X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
        exact = X @ (al * cov @ np.linalg.inv(al ** 2 * cov + de * np.eye(L))).T
        m, _ = pair_bp_batch(grid, w, lk2, X, al, de, log_mu2)
        np.testing.assert_allclose(m, exact, atol=2e-3)
