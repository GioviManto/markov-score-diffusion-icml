"""The tilted-walk kernel: the Gaussian embedding, Fisher's identity, the concave M-step,
and recovery of the phi^4 kernel from clean data."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from markov_score.bp_grid import grid_bp_batch, make_grid
from markov_score.em import clean_statistics, e_step_multi, fit_clean, fit_em, q_gradient, q_value
from markov_score.field import Phi4Chain
from markov_score.kernels import GaussianAR1Kernel
from markov_score.noising import alpha_delta
from markov_score.priors import GaussianAR1
from markov_score.tilted_kernel import TiltedWalkKernel
from markov_score.utils import rng_for


def _grid(n=201):
    return make_grid(8.0, n)


def _normalise_columns(log_k, w):
    return log_k - logsumexp(log_k + np.log(w)[:, None], axis=0)[None, :]


@pytest.mark.parametrize("basis,n", [("legendre", 4), ("bspline", 12)])
def test_init_is_the_gaussian_ar1_kernel(basis, n):
    g, w = _grid()
    rho, q = 0.6, 0.64
    k = TiltedWalkKernel.init(basis, n, g, rho=rho, var=q)
    ref = _normalise_columns(GaussianAR1Kernel(rho, q).log_transition_matrix(g), w)
    got = k.log_transition_matrix(g)
    # in the bulk (three standard deviations of a unit-variance chain), where the fixed soft
    # wall of the tilt is below 2e-3; a spline only approximates a parabola
    bulk = np.abs(g) <= 3.0
    mask = (np.exp(ref) > 1e-8) & bulk[:, None] & bulk[None, :]
    tol = 5e-3 if basis == "legendre" else 2e-2
    assert np.max(np.abs(got - ref)[mask]) < tol
    # and the tilt itself is the quadratic embedding plus the wall, exactly for a polynomial basis
    if basis == "legendre":
        from markov_score.tilted_kernel import soft_wall
        target = -(1 - rho) * g**2 / (2 * q) + soft_wall(k.half_width, g)
        assert np.max(np.abs((k.log_tilt(g) - target) - (k.log_tilt(g) - target).mean())) < 1e-8


def test_gradient_matches_finite_differences_and_fishers_identity():
    g, w = _grid()
    prior = GaussianAR1(0.7)
    rng = rng_for("tilted-fd", 0)
    A = np.stack([prior.sample(rng, 12) for _ in range(24)])
    groups = []
    for t in (0.2216, 0.9821):
        al, de = alpha_delta(t)
        groups.append((al * A + np.sqrt(de) * rng.standard_normal(A.shape), al, de))
    k = TiltedWalkKernel.init("legendre", 3, g, rho=0.5, var=0.7)
    stats = e_step_multi(g, w, k.log_transition_matrix(g), groups)
    grad = q_gradient(stats, k.grad_log_transition_matrix(g))
    theta = k.theta
    eps = 1e-5
    for p in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[p] += eps; tm[p] -= eps
        kp = TiltedWalkKernel(tp[0], tp[1:], k.basis, k.half_width)
        km = TiltedWalkKernel(tm[0], tm[1:], k.basis, k.half_width)
        # (i) the Q-gradient against a finite difference of Q at fixed Xi
        fd_q = (q_value(stats, kp.log_transition_matrix(g)) - q_value(stats, km.log_transition_matrix(g))) / (2 * eps)
        assert abs(grad[p] - fd_q) < 1e-5 * max(1.0, abs(fd_q)), p
        # (ii) Fisher's identity: the same number is the gradient of the exact evidence
        ep = e_step_multi(g, w, kp.log_transition_matrix(g), groups).log_evidence
        em = e_step_multi(g, w, km.log_transition_matrix(g), groups).log_evidence
        fd_l = (ep - em) / (2 * eps)
        assert abs(grad[p] - fd_l) < 1e-4 * max(1.0, abs(fd_l)), p


def test_m_step_ascends_and_returns_a_new_kernel():
    g, w = _grid()
    prior = GaussianAR1(0.7)
    rng = rng_for("tilted-mstep", 0)
    A = np.stack([prior.sample(rng, 16) for _ in range(32)])
    al, de = alpha_delta(0.4665)
    groups = [(al * A + np.sqrt(de) * rng.standard_normal(A.shape), al, de)]
    k0 = TiltedWalkKernel.init("legendre", 4, g, rho=0.3, var=0.8)
    stats = e_step_multi(g, w, k0.log_transition_matrix(g), groups)
    k1 = k0.m_step(stats, g)
    assert k1 is not k0
    assert np.array_equal(k0.theta, TiltedWalkKernel.init("legendre", 4, g, rho=0.3, var=0.8).theta)
    assert q_value(stats, k1.log_transition_matrix(g)) >= q_value(stats, k0.log_transition_matrix(g)) - 1e-10
    assert k1.inner_q_gain > 0.0


def test_clean_mle_recovers_the_gaussian_chain():
    g, w = _grid(401)
    prior = GaussianAR1(0.7)
    rng = rng_for("tilted-clean", 0)
    A = np.stack([prior.sample(rng, 32) for _ in range(512)])
    k = TiltedWalkKernel.init("legendre", 2, g, rho=0.3, var=0.8)
    fitted, _ = fit_clean(k, g, A, n_iters=3)
    ref = _normalise_columns(GaussianAR1Kernel(0.7, 1 - 0.7**2).log_transition_matrix(g), w)
    got = fitted.log_transition_matrix(g)
    # 512 chains put the sampling error of rho near 0.01, which in the kernel's tails is a
    # log-ratio of order e * a * drho / q ~ 0.15; the bulk is where the comparison is sharp
    bulk = np.abs(g) <= 3.0   # the stationary bulk, where the soft wall is below 2e-3
    mask = (np.exp(ref) > 1e-2) & bulk[:, None] & bulk[None, :]
    assert np.max(np.abs(got - ref)[mask]) < 0.1
    # and the implied (rho, q): gamma = q/rho and tilt curvature -(1-rho)/2q
    assert abs(fitted.gamma - (1 - 0.7**2) / 0.7) < 0.1


def test_clean_mle_recovers_the_phi4_kernel():
    """On the second laboratory the tilted family is exactly specified: the clean MLE with a
    flexible tilt reproduces the true kernel where it has mass."""
    g, w = _grid(401)
    chain = Phi4Chain(0.5, 1.0)
    A = chain.sample_batch(rng_for("tilted-phi4", 0), 1024, 32)
    k = TiltedWalkKernel.init("bspline", 24, g, rho=0.3, var=0.8)
    fitted, stats = fit_clean(k, g, A, n_iters=4)
    truth = chain.log_transition_matrix(g)
    pi = np.exp(chain.log_stationary(g))
    joint = np.exp(truth) * pi[None, :]
    mask = joint > 1e-3 * joint.max()
    diff = np.abs(fitted.log_transition_matrix(g) - truth)[mask]
    assert np.median(diff) < 0.1
    assert abs(fitted.gamma - 0.5 * 1.0 / chain.scale**2) < 0.15 * 0.5 / chain.scale**2


def test_em_through_noise_improves_the_denoiser():
    g, w = _grid(201)
    chain = Phi4Chain(0.5, 1.0)
    rng = rng_for("tilted-em", 0)
    A = chain.sample_batch(rng, 96, 16)
    groups = []
    for t in (0.2216, 0.4665):
        al, de = alpha_delta(t)
        groups.append((al * A + np.sqrt(de) * rng.standard_normal(A.shape), al, de))
    k0 = TiltedWalkKernel.init("legendre", 4, g, rho=0.3, var=0.8)
    k1, trace = fit_em(k0, g, w, groups, n_iters=6, tol=1e-12)
    assert trace.monotone_violation < 1e-6
    assert trace.log_evidence[-1] > trace.log_evidence[0]
    X, al, de = groups[0]
    m0, _ = grid_bp_batch(g, w, k0.log_transition_matrix(g), X, al, de)
    m1, _ = grid_bp_batch(g, w, k1.log_transition_matrix(g), X, al, de)
    assert np.mean((m1 - A) ** 2) < np.mean((m0 - A) ** 2)
