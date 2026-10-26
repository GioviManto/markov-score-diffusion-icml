"""The neural-energy kernel of E55: columns are densities of the discretised model, the zero start is the uniform kernel,
an M-step never lowers Q and never mutates its input, EM through noise is monotone and learns a Gaussian chain, and the
kernel has an invariant law the learning drivers can use."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.em import ExpectedStatistics, fit_em  # noqa: E402
from markov_score.energy_kernel import NeuralEnergyKernel  # noqa: E402
from markov_score.stationary import invariant_log_density_eig  # noqa: E402

A_HALF = 6.0


def _grid(n=81):
    return make_grid(A_HALF, n)


def test_columns_are_densities_and_the_start_is_uniform():
    grid, w = _grid()
    k = NeuralEnergyKernel.init(hidden=16, seed=0, half_width=A_HALF)
    lk = k.log_transition_matrix(grid)
    np.testing.assert_allclose(np.exp(lk).T @ w, 1.0, atol=1e-12)
    np.testing.assert_allclose(np.exp(lk), 1.0 / (2 * A_HALF), rtol=1e-12)
    assert k.n_params == 2 * 16 + 16 + 16 * 16 + 16 + 16 + 1


def _random_stats(grid, rng):
    xi = rng.gamma(1.0, size=(len(grid), len(grid))) * np.exp(-0.5 * np.subtract.outer(grid, 0.8 * grid) ** 2)
    xi *= 1000.0 / xi.sum()
    return ExpectedStatistics(xi=xi, site1=np.ones(len(grid)), log_evidence=0.0, n_edges=1000, n_chains=10)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_m_step_never_lowers_q_and_returns_a_new_kernel(seed):
    grid, w = _grid()
    rng = np.random.default_rng(seed)
    stats = _random_stats(grid, rng)
    k = NeuralEnergyKernel.init(hidden=16, seed=seed, half_width=A_HALF)
    q = lambda kk: float(np.sum(stats.xi * kk.log_transition_matrix(grid)))  # noqa: E731
    theta0 = k.theta.copy()
    ks = [k]
    for _ in range(5):
        ks.append(ks[-1].m_step(stats, grid))
    np.testing.assert_array_equal(k.theta, theta0)            # the input is never mutated
    qs = [q(kk) for kk in ks]
    assert all(b >= a - 1e-9 for a, b in zip(qs, qs[1:]))
    assert qs[-1] > qs[0] + 1.0                                # and it actually learns
    assert len({kk.theta.tobytes() for kk in ks}) == len(ks)   # every iterate is its own snapshot


def _gaussian_chains(rng, n, L, rho):
    A = np.zeros((n, L))
    A[:, 0] = rng.standard_normal(n)
    for i in range(1, L):
        A[:, i] = rho * A[:, i - 1] + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    return A


def test_em_through_noise_is_monotone_and_learns_a_gaussian_chain():
    grid, w = _grid()
    rng = np.random.default_rng(0)
    L, rho, levels = 16, 0.8, (0.1, 0.4, 1.0)
    A = _gaussian_chains(rng, 384, L, rho)
    groups = []
    for t, part in zip(levels, np.array_split(A, len(levels))):
        al, de = np.exp(-t), 1 - np.exp(-2 * t)
        groups.append((al * part + np.sqrt(de) * rng.standard_normal(part.shape), al, de))
    k0 = NeuralEnergyKernel.init(hidden=16, seed=0, half_width=A_HALF, n_inner=10, lr=2e-2)
    # with a FIXED initial law the M-step is a generalised M-step for the whole likelihood: exactly monotone
    _, fixed = fit_em(k0, grid, w, groups, n_iters=30, log_mu=-0.5 * grid**2 - 0.5 * np.log(2 * np.pi), tol=0.0)
    assert fixed.monotone_violation == 0.0
    # with the kernel's own stationary law (what E55 uses) only the site-1 term can dip; fit_em records it
    log_mu = lambda kern: invariant_log_density_eig(kern, grid, w).log_density  # noqa: E731
    k, trace = fit_em(k0, grid, w, groups, n_iters=60, log_mu=log_mu, tol=0.0)
    assert trace.monotone_violation < 1e-3 * len(A)          # nats, against an evidence of ~8000
    assert trace.log_evidence[-1] > trace.log_evidence[0]
    # score error of the learned kernel against the true Gaussian kernel, on fresh chains
    true_lk = np.log(np.exp(-0.5 * (grid[:, None] - rho * grid[None, :]) ** 2 / (1 - rho**2))
                     / np.sqrt(2 * np.pi * (1 - rho**2)))
    true_lk -= np.log(np.exp(true_lk).T @ w)[None, :]
    true_mu = -0.5 * grid**2 - 0.5 * np.log(2 * np.pi)
    lk, mu = k.log_transition_matrix(grid), log_mu(k)
    Xc = _gaussian_chains(rng, 128, L, rho)
    num = den = 0.0
    for t in (0.1, 1.0):
        al, de = np.exp(-t), 1 - np.exp(-2 * t)
        X = al * Xc + np.sqrt(de) * rng.standard_normal(Xc.shape)
        m_ref = grid_bp_batch(grid, w, true_lk, X, al, de, log_mu=true_mu)[0]
        m_hat = grid_bp_batch(grid, w, lk, X, al, de, log_mu=mu)[0]
        num += (((m_hat - m_ref) * al / de) ** 2).sum(); den += (((X - al * m_ref) / de) ** 2).sum()
    assert np.sqrt(num / den) < 0.1


def test_the_fitted_kernel_has_an_invariant_law():
    grid, w = _grid()
    stats = _random_stats(grid, np.random.default_rng(5))
    k = NeuralEnergyKernel.init(hidden=16, seed=0, half_width=A_HALF)
    for _ in range(3):
        k = k.m_step(stats, grid)
    d = invariant_log_density_eig(k, grid, w)
    np.testing.assert_allclose(np.exp(d.log_density) @ w, 1.0, atol=1e-8)
