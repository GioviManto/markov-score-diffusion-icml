"""The 2-D state module: with no rotation the chain is two independent Laplace chains, so 2-D BP reproduces 1-D grid BP
on each component (means and evidence); the expected transitions sum to the number of edges; the VAR(1) M-step recovers
the transition matrix from clean-like data; the neural M-step never lowers Q."""

from __future__ import annotations

import math

import numpy as np
import pytest

from markov_score.bp_grid import grid_bp_batch
from markov_score.noising import alpha_delta
from markov_score.priors import LaplaceAR1
from markov_score.vector_chain import (GaussianVAR1, RotatedLaplaceVAR, Stats2D, bp2d, e_step2d, normalise, state_grid)


def _one_d_normalised(g, w):
    lk = LaplaceAR1(0.85).log_transition_matrix(g)
    m = lk.max(axis=0, keepdims=True)
    return lk - (m + np.log((np.exp(lk - m) * w[:, None]).sum(axis=0, keepdims=True)))


def test_separable_chain_equals_two_one_d_chains():
    g, w, states, w2 = state_grid(6.0, 21)
    lk1 = _one_d_normalised(g, w)
    n = len(g)
    lk2 = (lk1[:, None, :, None] + lk1[None, :, None, :]).reshape(n * n, n * n)     # [(i',j'), (i,j)]
    rng = np.random.default_rng(0)
    chain = RotatedLaplaceVAR(theta=0.0)
    B = chain.sample(rng, 5, 7)
    al, de = alpha_delta(0.3)
    X = al * B + math.sqrt(de) * rng.standard_normal(B.shape)
    means, st = bp2d(states, w2, lk2, X, al, de, want_xi=True)
    ev = 0.0
    for c in range(2):
        m1, _, e1 = grid_bp_batch(g, w, lk1, X[:, :, c], al, de, return_evidence=True)
        np.testing.assert_allclose(means[:, :, c], m1, atol=1e-10)
        ev += float(np.sum(e1))
    assert abs(st.log_evidence - ev) < 1e-6 * abs(ev)
    np.testing.assert_allclose(st.xi.sum(), st.n_edges, rtol=1e-10)


def test_rotated_kernel_columns_are_densities_and_covariance_is_stationary():
    _, _, states, w2 = state_grid(5.0, 21)
    chain = RotatedLaplaceVAR()
    lk = chain.log_transition_matrix(states, w2)
    np.testing.assert_allclose((np.exp(lk) * w2[:, None]).sum(axis=0), 1.0, atol=1e-10)
    C = chain.covariance(3)
    np.testing.assert_allclose(C[:2, :2], np.eye(2)); np.testing.assert_allclose(C[2:4, :2], chain.A)


def test_var1_m_step_recovers_the_transition_matrix():
    _, _, states, w2 = state_grid(5.0, 31)
    chain = RotatedLaplaceVAR()
    B = chain.sample(np.random.default_rng(1), 4000, 6)
    S = len(states)
    near = lambda Y: np.argmin(((Y[:, None, :] - states[None]) ** 2).sum(-1), axis=1)  # noqa: E731
    xi = np.zeros((S, S))
    for k in range(B.shape[1] - 1):
        np.add.at(xi, (near(B[:, k + 1]), near(B[:, k])), 1.0)
    k = GaussianVAR1.init().m_step(Stats2D(xi=xi, log_evidence=0.0, n_edges=int(xi.sum())), states, w2)
    np.testing.assert_allclose(k.A, chain.A, atol=0.03)
    np.testing.assert_allclose(np.diag(k.Q), 1 - 0.85 ** 2, atol=0.03)


def test_neural_m_step_never_lowers_q():
    pytest.importorskip("torch")
    from markov_score.vector_chain import NeuralEnergyKernel2D
    _, _, states, w2 = state_grid(5.0, 11)
    chain = RotatedLaplaceVAR()
    rng = np.random.default_rng(2)
    B = chain.sample(rng, 64, 6)
    al, de = alpha_delta(0.5)
    X = al * B + math.sqrt(de) * rng.standard_normal(B.shape)
    k = NeuralEnergyKernel2D.init(hidden=8, seed=0, half_width=5.0)
    for _ in range(3):
        st = e_step2d(states, w2, k.log_transition_matrix(states, w2), [(X, al, de)])
        q0 = float((st.xi * k.log_transition_matrix(states, w2)).sum())
        k = k.m_step(st, states, w2)
        assert float((st.xi * k.log_transition_matrix(states, w2)).sum()) >= q0 - 1e-9
