"""The strided (subsampled) phi^4 chain: exactly Markov with the stride-power kernel."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from markov_score.bp_grid import make_grid
from markov_score.field import Phi4Chain, StridedChain
from markov_score.utils import rng_for


@pytest.fixture(scope="module")
def fine():
    return Phi4Chain(0.3, 0.25)


def test_stride_one_is_the_fine_chain(fine):
    g, w = make_grid(8.0, 401)
    s = StridedChain(fine, 1)
    assert np.array_equal(s.log_transition_matrix(g), fine.log_transition_matrix(g))
    assert s.h == fine.h and s.rho == fine.rho


@pytest.mark.parametrize("stride", [2, 4])
def test_strided_kernel_is_a_proper_kernel_with_the_same_invariant_law(fine, stride):
    g, w = make_grid(8.0, 401)
    s = StridedChain(fine, stride)
    lk = s.log_transition_matrix(g)
    col = logsumexp(lk + np.log(w)[:, None], axis=0)
    # the fine kernel is normalised analytically (ground-state transform on the fine grid), so on
    # the operating grid its columns integrate to one only to quadrature precision (a few 1e-7)
    assert np.max(np.abs(col)) < 1e-5
    pi = np.exp(fine.log_stationary(g))
    K = np.exp(lk)
    pi_next = K @ (w * pi)
    m = pi > 1e-10
    assert np.max(np.abs(pi_next - pi)[m] / pi[m]) < 1e-6   # pi is invariant for K^s
    assert abs(s.rho - fine.rho**stride) < 1e-12 and s.h == fine.h * stride


def test_chapman_kolmogorov(fine):
    g, w = make_grid(8.0, 401)
    k2 = np.exp(StridedChain(fine, 2).log_transition_matrix(g))
    k4 = np.exp(StridedChain(fine, 4).log_transition_matrix(g))
    k22 = k2 @ (w[:, None] * k2)
    assert np.max(np.abs(k4 - k22)) < 1e-10 * k4.max()


def test_strided_samples_match_the_strided_covariance(fine):
    rng = rng_for("strided-cov", 0)
    s = StridedChain(fine, 4)
    L = 16
    A = s.sample_batch(rng, 4096, L)
    assert A.shape == (4096, L)
    emp = np.cov(A.T)
    cov = s.covariance(L)
    assert np.max(np.abs(emp - cov)) < 0.08                  # 4096 chains: s.e. ~ 0.02-0.05
    assert abs(cov[0, 1] / cov[0, 0] - fine.rho**4) < 0.05   # lag-one correlation of the strided chain


def test_strided_kernel_is_not_a_tilted_walk(fine):
    """For stride >= 2 the mixed derivative of log K^s is not constant: the bond is no longer
    Gaussian, so the gradient-class kernels are misspecified on the subsampled chain."""
    g, w = make_grid(8.0, 401)
    bulk = np.abs(g) <= 1.5
    for stride, flat in ((1, True), (4, False)):
        lk = StridedChain(fine, stride).log_transition_matrix(g)
        sub = lk[np.ix_(bulk, bulk)]
        mixed = np.diff(np.diff(sub, axis=0), axis=1) / (g[1] - g[0]) ** 2
        spread = float(np.std(mixed) / abs(np.mean(mixed)))
        assert (spread < 1e-6) == flat, (stride, spread)
