"""The pair-energy kernel (the physicist's parametrisation of the learned pair energy) and the
reversibility identities of the tilted walk.

Pinned here:
  * every tilted walk is reversible, with invariant density g Z (closed form = eigensolve) and
    site energy -(1/2) log(g / Z), which for the phi^4 chain is the potential (h / 2T) V(s a);
  * the pair-energy kernel with the field's own (gamma, c) IS the field's kernel on the grid;
  * its Gaussian-AR(1) initialisation is the Gaussian chain, like the tilted kernel's;
  * its analytic gradient (perturbation theory of the Perron eigenvector) matches finite
    differences, and Fisher's identity holds;
  * clean-data MLE from the field's chains recovers the temperature and the barrier.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from markov_score.bp_grid import make_grid
from markov_score.em import clean_statistics, e_step_multi, fit_clean, q_gradient, q_value
from markov_score.field import Phi4Chain, potential
from markov_score.kernels import GaussianAR1Kernel
from markov_score.noising import alpha_delta
from markov_score.pair_energy_kernel import PairEnergyKernel, potential_readings
from markov_score.priors import GaussianAR1
from markov_score.stationary import invariant_log_density_eig
from markov_score.tilted_kernel import TiltedWalkKernel
from markov_score.utils import rng_for


def _normalise_columns(log_k, w):
    return log_k - logsumexp(log_k + np.log(w)[:, None], axis=0)[None, :]


# ---------------------------------------------------------------------- the tilted walk is reversible


def test_tilted_walk_is_reversible_with_invariant_law_g_times_z():
    g, w = make_grid(8.0, 401)
    k = TiltedWalkKernel.init("legendre", 6, g, rho=0.3, var=0.8)
    k = TiltedWalkKernel(k.log_gamma, k.coef + np.array([0.3, -0.5, 0.2, 0.1, -0.05, 0.02]), k.basis, k.half_width)
    closed = k.log_stationary(g)
    eig = invariant_log_density_eig(k, g, w).log_density
    m = np.exp(closed) > 1e-12
    assert np.max(np.abs(closed - eig)[m]) < 1e-8
    joint = closed[None, :] + k.log_transition_matrix(g)        # log pi(a_j) K(a_k | a_j)
    mm = np.exp(joint) > 1e-12
    assert np.max(np.abs(joint - joint.T)[mm & mm.T]) < 1e-10   # detailed balance


@pytest.mark.parametrize("T,h", [(0.3, 1.0), (1.0, 0.419)])
def test_site_energy_of_the_field_kernel_is_the_potential(T, h):
    """Build the tilted walk with the field's (gamma, log g) and read its site energy back."""
    g, w = make_grid(8.0, 401)
    chain = Phi4Chain(T, h)
    s = chain.scale
    gamma = T * h / s**2
    lt = chain.log_tilt(g)
    fin = np.isfinite(lt)
    logN = -0.5 * (g[:, None] - g[None, :]) ** 2 / gamma - 0.5 * np.log(2 * np.pi * gamma)
    logZ = logsumexp((logN + lt[:, None] + np.log(w)[:, None])[fin], axis=0)
    f = -0.5 * (lt - logZ)
    f_true = (h / (2 * T)) * potential(s * g)
    bulk = fin & (np.abs(g) <= 3.0)
    d = (f - f_true)[bulk]
    assert np.max(np.abs(d - d.mean())) < 1e-6
    # and the readings: temperature, barrier 1/4, wells at +-1
    r = potential_readings(np.where(fin, f, np.nan), g, gamma, s, h)
    assert abs(r["theta_hat"] - T) < 1e-9
    assert abs(r["barrier_hat"] - 0.25) < 2e-3
    assert abs(r["well_hat"] - 1.0) < 1e-2
    assert r["potential_rms"] < 2e-3


# ---------------------------------------------------------------------- the pair-energy kernel


def test_field_parameters_give_the_field_kernel():
    """With the field's own (gamma, c_2, c_4) the pair-energy kernel is the chain's kernel, up to
    the coarse (401, 8) eigensolve against the chain's fine (2401, 12) one."""
    g, w = make_grid(8.0, 401)
    chain = Phi4Chain(0.3, 1.0)
    k = PairEnergyKernel.from_field(0.3, 1.0, chain.scale)
    truth = chain.log_transition_matrix(g)
    pi = np.exp(chain.log_stationary(g))
    joint = np.exp(truth) * pi[None, :]
    mask = joint > 1e-6 * joint.max()
    assert np.max(np.abs(k.log_transition_matrix(g) - truth)[mask]) < 2e-3
    lp = k.log_stationary(g)
    m = pi > 1e-8
    assert np.max(np.abs(lp - chain.log_stationary(g))[m]) < 2e-3
    # the invariant law by eigensolve of K agrees with psi^2
    eig = invariant_log_density_eig(k, g, w).log_density
    assert np.max(np.abs(lp - eig)[m]) < 1e-7


def test_init_is_the_gaussian_ar1_kernel():
    g, w = make_grid(8.0, 201)
    rho, var = 0.3, 0.8
    k = PairEnergyKernel.init(4, rho=rho, var=var)
    ref = _normalise_columns(GaussianAR1Kernel(rho, var).log_transition_matrix(g), w)
    got = k.log_transition_matrix(g)
    bulk = np.abs(g) <= 3.0
    mask = (np.exp(ref) > 1e-4) & bulk[:, None] & bulk[None, :]
    assert np.max(np.abs(got - ref)[mask]) < 5e-3     # the soft wall at |a| = 3 is 2e-3


def test_gradient_matches_finite_differences_and_fishers_identity():
    g, w = make_grid(8.0, 201)
    chain = Phi4Chain(0.5, 1.0)
    A = chain.sample_batch(rng_for("pair-grad", 0), 24, 16)
    groups = []
    rng = rng_for("pair-grad-noise", 0)
    for t in (0.2216, 0.9821):
        al, de = alpha_delta(t)
        groups.append((al * A + np.sqrt(de) * rng.standard_normal(A.shape), al, de))
    k = PairEnergyKernel.init(4, rho=0.4, var=0.7)
    k = PairEnergyKernel(k.log_gamma, k.coef + np.array([0.1, 0.2, -0.05, 0.3]))
    stats = e_step_multi(g, w, k.log_transition_matrix(g), groups)
    grad = q_gradient(stats, k.grad_log_transition_matrix(g))
    eps = 1e-5
    def shifted(i, d):
        z = k.theta.copy(); z[i] += d
        return PairEnergyKernel(float(z[0]), z[1:].copy())   # copy: the dataclass must not hold a view of z

    for i in range(1 + k.degree):
        fd = (q_value(stats, shifted(i, eps).log_transition_matrix(g))
              - q_value(stats, shifted(i, -eps).log_transition_matrix(g))) / (2 * eps)
        assert abs(fd - grad[i]) < 1e-4 * max(1.0, abs(fd)), (i, fd, grad[i])
    # Fisher: d log evidence / d theta = d Q / d theta at the current theta
    for i in (0, 2):
        ep = e_step_multi(g, w, shifted(i, eps).log_transition_matrix(g), groups).log_evidence
        em = e_step_multi(g, w, shifted(i, -eps).log_transition_matrix(g), groups).log_evidence
        assert abs((ep - em) / (2 * eps) - grad[i]) < 1e-3 * max(1.0, abs(grad[i]))


def test_m_step_ascends_and_returns_a_new_kernel():
    g, w = make_grid(8.0, 201)
    chain = Phi4Chain(0.5, 1.0)
    A = chain.sample_batch(rng_for("pair-mstep", 0), 64, 16)
    al, de = alpha_delta(0.4665)
    groups = [(al * A + np.sqrt(de) * rng_for("pair-mstep-noise", 0).standard_normal(A.shape), al, de)]
    k = PairEnergyKernel.init(4, rho=0.3, var=0.8)
    stats = e_step_multi(g, w, k.log_transition_matrix(g), groups)
    q0 = q_value(stats, k.log_transition_matrix(g))
    k1 = k.m_step(stats, g)
    assert k1 is not k and not np.array_equal(k1.theta, k.theta)
    assert q_value(stats, k1.log_transition_matrix(g)) > q0
    assert k1.inner_q_gain > 0


def test_clean_mle_recovers_temperature_and_barrier():
    """The physicist's test: from 1024 clean chains of the field at (T, h) = (0.3, 1), the
    five-parameter fit reads the temperature within 5% and the barrier within 10%."""
    g, w = make_grid(8.0, 401)
    chain = Phi4Chain(0.3, 1.0)
    A = chain.sample_batch(rng_for("pair-clean", 0), 1024, 32)
    k = PairEnergyKernel.init(4, rho=0.3, var=0.8)
    fitted, _ = fit_clean(k, g, A, n_iters=3)
    r = potential_readings(fitted.site_energy(g), g, fitted.gamma, chain.scale, 1.0)
    assert abs(r["theta_hat"] - 0.3) < 0.015, r
    assert abs(r["barrier_hat"] - 0.25) < 0.025, r
    assert abs(r["well_hat"] - 1.0) < 0.05, r
    # and the kernel itself, where the chain has mass
    truth = chain.log_transition_matrix(g)
    pi = np.exp(chain.log_stationary(g))
    joint = np.exp(truth) * pi[None, :]
    mask = joint > 1e-3 * joint.max()
    assert np.median(np.abs(fitted.log_transition_matrix(g) - truth)[mask]) < 0.05


def test_tilted_spline_site_energy_reads_the_potential_from_clean_data():
    """The same readings through the flexible tilt: its site energy is the potential too."""
    g, w = make_grid(8.0, 401)
    chain = Phi4Chain(0.3, 1.0)
    A = chain.sample_batch(rng_for("pair-clean", 0), 1024, 32)
    k = TiltedWalkKernel.init("bspline", 24, g, rho=0.3, var=0.8)
    fitted, _ = fit_clean(k, g, A, n_iters=4)
    r = potential_readings(fitted.site_energy(g), g, fitted.gamma, chain.scale, 1.0)
    assert abs(r["theta_hat"] - 0.3) < 0.03, r
    assert abs(r["barrier_hat"] - 0.25) < 0.05, r
    assert abs(r["well_hat"] - 1.0) < 0.08, r
