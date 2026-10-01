"""Moment-projected BP: Proposition 1 holds on linear transitions and is measured, not
assumed, on the phi^4 chain."""

from __future__ import annotations

import numpy as np
import pytest

from markov_score import theory
from markov_score.bp_grid import grid_bp_batch, grid_bp_projected_batch, make_grid
from markov_score.field import Phi4Chain
from markov_score.noising import alpha_delta
from markov_score.priors import GaussianAR1, LaplaceAR1
from markov_score.utils import rng_for


def _data(prior, n, L, t, tag):
    rng = rng_for(tag, 0)
    A = np.stack([prior.sample(rng, L) for _ in range(n)])
    al, de = alpha_delta(t)
    return A, al * A + np.sqrt(de) * rng.standard_normal(A.shape), al, de


# A grid wide enough for the BACKWARD messages, which are centred at x/(alpha rho): the
# operating (401, 8) grid truncates them at t = 1 and the projection then describes a stump
# (bp_grid.grid_bp_projected_batch docstring). Half-width 32 holds them with margin.
WIDE = {0.2216: (1601, 16.0), 0.9821: (3201, 32.0)}


@pytest.mark.parametrize("t", [0.2216, 0.9821])
def test_gaussian_chain_projection_is_the_identity_and_the_lmmse(t):
    n_g, hw = WIDE[t]
    g, w = make_grid(hw, n_g)
    prior = GaussianAR1(0.85)
    A, X, al, de = _data(prior, 32, 16, t, "proj-gauss")
    log_k = prior.log_transition_matrix(g)
    m_bp, _ = grid_bp_batch(g, w, log_k, X, al, de)
    m_pr, _, edge = grid_bp_projected_batch(g, w, log_k, X, al, de, return_diagnostics=True)
    m_lin = theory.lmmse_posterior_mean(X, theory.chain_covariance(0.85, 1 - 0.85**2, 16), t)
    assert edge < 1e-6
    assert np.max(np.abs(m_pr - m_bp)) < 1e-6
    assert np.max(np.abs(m_pr - m_lin)) < 1e-6


def test_operating_grid_truncates_the_gaussian_chains_backward_messages():
    """The failure mode, pinned: at t = 1 on (401, 8) the diagnostic fires and the projection
    departs from exact BP. This is the reason the field chain's arm carries the diagnostic."""
    g, w = make_grid(8.0, 401)
    prior = GaussianAR1(0.85)
    A, X, al, de = _data(prior, 32, 16, 0.9821, "proj-gauss")
    log_k = prior.log_transition_matrix(g)
    m_bp, _ = grid_bp_batch(g, w, log_k, X, al, de)
    m_pr, _, edge = grid_bp_projected_batch(g, w, log_k, X, al, de, return_diagnostics=True)
    assert edge > 1e-2
    assert np.max(np.abs(m_pr - m_bp)) > 1e-2


@pytest.mark.parametrize("t", [0.2216, 0.9821])
def test_laplace_chain_projection_is_the_lmmse(t):
    """Proposition 1: on a linear transition the projected recursion returns the linear floor."""
    n_g, hw = WIDE[t]
    g, w = make_grid(hw, n_g)
    prior = LaplaceAR1(0.85)
    A, X, al, de = _data(prior, 32, 16, t, "proj-laplace")
    m_pr, _, edge = grid_bp_projected_batch(g, w, prior.log_transition_matrix(g), X, al, de, return_diagnostics=True)
    m_lin = theory.lmmse_posterior_mean(X, theory.chain_covariance(0.85, 1 - 0.85**2, 16), t)
    m_bp, _ = grid_bp_batch(g, w, prior.log_transition_matrix(g), X, al, de)
    assert edge < 1e-6
    assert np.max(np.abs(m_pr - m_lin)) < 1e-5
    assert np.max(np.abs(m_bp - m_lin)) > 1e-3   # and the exact score is not the floor


def test_phi4_chain_projection_is_not_well_posed_on_the_operating_grid():
    """The finding of 1 October 2026, pinned. From far outside the bulk the phi^4 kernel jumps
    to a point that grows only like a^(1/3), so the backward message E[ell(a_{i+1}) | a_i = a]
    is nearly FLAT across the operating grid: its moment projection is a near-uniform Gaussian
    that discards the backward information. The diagnostic must fire, and the arm reported for
    this chain as "Gaussian closure" is therefore the covariance twin (LMMSE), as Proposition 1
    identifies the two on linear transitions."""
    g, w = make_grid(8.0, 401)
    chain = Phi4Chain(0.5, 1.0)
    rng = rng_for("proj-phi4", 0)
    A = chain.sample_batch(rng, 32, 16)
    al, de = alpha_delta(0.4665)
    X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
    m_pr, v_pr, edge = grid_bp_projected_batch(g, w, chain.log_transition_matrix(g), X, al, de,
                                               log_mu=chain.log_stationary(g), return_diagnostics=True)
    m_lin = theory.lmmse_posterior_mean(X, chain.covariance(16), 0.4665)
    assert np.all(np.isfinite(m_pr)) and np.all(v_pr > 0)
    assert m_pr.shape == m_lin.shape
    assert edge > 0.1
