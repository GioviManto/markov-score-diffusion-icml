"""markov_score.grid_pair_chain: a reversible grid chain from a symmetric pair energy (E43's out-of-class laboratory)."""
from __future__ import annotations

import numpy as np
import pytest

from markov_score.bp_grid import grid_bp_batch
from markov_score.grid_pair_chain import GridPairChain, laplace_double_well
from markov_score.noising import alpha_delta
from markov_score.utils import rng_for

CHAIN = laplace_double_well(bond=0.40, depth=2.0)
GRID, W = CHAIN.grid, CHAIN.w


def test_columns_are_normalised_under_the_quadrature_weights():
    assert np.allclose((np.exp(CHAIN.log_transition_matrix(GRID)) * W[:, None]).sum(0), 1.0)


def test_the_stationary_law_is_invariant_and_the_chain_is_reversible():
    P = np.exp(CHAIN.log_transition_matrix(GRID)) * W[:, None]            # P[k, j] = prob(j -> k)
    assert np.allclose(P @ CHAIN.p, CHAIN.p, atol=1e-12)
    flow = P * CHAIN.p[None, :]                                           # flow[k, j] = p_j P[k, j]
    assert np.allclose(flow, flow.T, atol=1e-14)                          # detailed balance


def test_the_stationary_log_density_is_finite_everywhere_and_normalised():
    ld = CHAIN.log_stationary(GRID)
    assert np.isfinite(ld).all() and abs((np.exp(ld) * W).sum() - 1.0) < 1e-12
    assert np.allclose(np.exp(ld) * W, CHAIN.p, atol=1e-15)


def test_the_pair_energy_must_be_symmetric():
    with pytest.raises(ValueError):
        GridPairChain(np.triu(np.ones((401, 401))), GRID, W)


def test_the_calibration_is_comparable_to_the_phi4_cells():
    assert 0.55 < CHAIN.variance < 0.8 and 0.8 < CHAIN.lag1_correlation < 0.9 and -1.5 < CHAIN.excess_kurtosis < -1.0


def test_samples_follow_the_stationary_law_and_the_exact_covariance():
    A = CHAIN.sample_batch(rng_for("test-grid-chain"), 6000, 12)
    idx = np.searchsorted(GRID, A[:, 0] - 1e-9)
    emp = np.bincount(idx, minlength=len(GRID)) / len(idx)
    assert np.abs(np.cumsum(emp) - np.cumsum(CHAIN.p)).max() < 0.03       # Kolmogorov distance of the first site
    C = CHAIN.covariance(12)
    emp_c = (A.T @ A) / A.shape[0]
    assert np.abs(emp_c - C).max() < 0.05 and np.allclose(C, C.T)
    assert abs(A.mean()) < 0.05


def test_grid_bp_with_the_chains_own_kernel_is_the_exact_posterior_mean():
    """Two sites, noised: the exact posterior mean by enumeration over the grid pair equals grid BP with this kernel."""
    A = CHAIN.sample_batch(rng_for("test-grid-chain-bp"), 3, 2)
    al, de = alpha_delta(0.8)
    X = al * A + np.sqrt(de) * np.random.default_rng(0).standard_normal(A.shape)
    mean = grid_bp_batch(GRID, W, CHAIN.log_transition_matrix(GRID), X, al, de, log_mu=CHAIN.log_stationary(GRID))[0]
    P = np.exp(CHAIN.log_transition_matrix(GRID)) * W[:, None]            # P[k, j]
    for b in range(3):
        lik0 = np.exp(-0.5 * (X[b, 0] - al * GRID) ** 2 / de)
        lik1 = np.exp(-0.5 * (X[b, 1] - al * GRID) ** 2 / de)
        joint = (CHAIN.p * lik0)[None, :] * P * lik1[:, None]            # [k, j]: first site j, second site k
        joint /= joint.sum()
        assert np.allclose(mean[b], [(joint.sum(0) * GRID).sum(), (joint.sum(1) * GRID).sum()], atol=2e-3)
