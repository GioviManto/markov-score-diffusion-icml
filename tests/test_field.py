"""The phi^4 chain: the eigensolve, the tilted-walk identity, the two samplers, the cumulants.

Each test here is a claim in `field.py`'s docstring checked numerically, with a control where
the arithmetic could hide a wrong answer (verification-discipline memory: refine the grid to
test quadrature, widen the domain to test tails, and compare two independent routes).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp

from markov_score.bp_grid import grid_bp_batch, make_grid
from markov_score.field import Phi4Chain, h_for_correlation, solve_spectrum
from markov_score.utils import rng_for

CELLS = [(0.5, 1.0), (1.0, 0.5), (0.3, 1.0)]


@pytest.fixture(scope="module")
def grid():
    return make_grid(8.0, 401)


@pytest.mark.parametrize("T,h", CELLS)
def test_spectrum_is_a_stationary_reversible_chain(T, h, grid):
    chain = Phi4Chain(T, h)
    sp = chain.spectrum
    assert 0.0 < sp.rho1 < 1.0
    assert sp.lam[0] > sp.lam[1] > 0.0
    g, w = grid
    log_k = chain.log_transition_matrix(g)
    log_pi = chain.log_stationary(g)
    assert not np.any(np.isnan(log_k))          # -inf is allowed: an unreachable arrival state
    ok = np.isfinite(log_pi)
    assert np.all(np.isfinite(log_k[np.ix_(ok, ok)]))
    K = np.exp(log_k)
    # every resolved column is a density under the BP grid's quadrature
    col = (K * w[:, None]).sum(axis=0)
    assert np.max(np.abs(col[ok] - 1.0)) < 1e-6
    # stationarity: K pi = pi, and reversibility: pi(a) K(a'|a) symmetric
    pi = np.exp(np.where(ok, log_pi, -np.inf))
    assert abs(float(pi @ w) - 1.0) < 1e-6
    assert np.max(np.abs(K @ (pi * w) - pi)) < 1e-6 * pi.max()
    flux = K * pi[None, :]
    assert np.max(np.abs(flux - flux.T)) < 1e-9 * flux.max()
    # standardised: unit variance, zero mean, to quadrature
    assert abs(float((g**2 * pi) @ w) - 1.0) < 1e-6
    assert abs(float((g * pi) @ w)) < 1e-10


@pytest.mark.parametrize("T,h", CELLS)
def test_tilted_walk_identity(T, h, grid):
    """K = N(a'; a, Th/s^2) g(a') / Z(a) with Z the column normaliser (the h-transform)."""
    chain = Phi4Chain(T, h)
    g, w = grid
    s = chain.scale
    gamma = T * h / s**2
    d = g[:, None] - g[None, :]
    raw = -0.5 * d**2 / gamma - 0.5 * np.log(2 * np.pi * gamma) + chain.log_tilt(g)[:, None]
    lw = np.log(w)
    log_k_tilt = raw - logsumexp(raw + lw[:, None], axis=0)[None, :]
    log_k = chain.log_transition_matrix(g)
    ok = np.isfinite(chain.log_stationary(g))
    mask = np.isfinite(log_k_tilt) & np.isfinite(log_k)
    with np.errstate(invalid="ignore"):
        diff = np.abs(log_k_tilt - log_k)[:, ok]
    m = mask[:, ok] & (np.exp(log_k)[:, ok] > 1e-12)
    assert np.max(diff[m]) < 1e-6


def test_refining_the_fine_grid_moves_nothing():
    """Trapezoid quadrature of the smooth, fast-decaying kernel is spectrally accurate."""
    a = solve_spectrum(0.5, 1.0, 1201, 12.0)
    b = solve_spectrum(0.5, 1.0, 2401, 12.0)
    assert abs(a.lambda0 - b.lambda0) < 1e-10 * b.lambda0
    assert abs(a.rho1 - b.rho1) < 1e-10
    assert abs(a.scale - b.scale) < 1e-9
    assert abs(a.excess_kurtosis - b.excess_kurtosis) < 1e-8


def test_widening_the_domain_moves_nothing():
    a = solve_spectrum(0.5, 1.0, 2401, 12.0)
    b = solve_spectrum(0.5, 1.0, 3201, 16.0)
    assert abs(a.rho1 - b.rho1) < 1e-10
    assert abs(a.scale - b.scale) < 1e-9


def test_overlaps_are_complete():
    sp = solve_spectrum(0.5, 1.0)
    assert abs(float((sp.overlap**2).sum()) - sp.scale**2) < 1e-10 * sp.scale**2


def test_covariance_matches_sample_autocorrelation_and_two_samplers_agree():
    chain = Phi4Chain(0.5, 1.0)
    L, n = 16, 3000
    A = chain.sample_batch(rng_for("field-cont", 0), n, L)
    B = chain.sample_lattice(rng_for("field-latt", 0), n, L)
    cov = chain.covariance(L)
    assert abs(cov[0, 0] - 1.0) < 1e-12
    for Z in (A, B):
        var = Z.var()
        lag1 = np.mean(Z[:, :-1] * Z[:, 1:])
        kurt = np.mean(Z**4) / var**2 - 3.0
        assert abs(var - 1.0) < 0.04
        assert abs(lag1 - cov[0, 1]) < 0.04
        assert abs(kurt - chain.innovation_excess_kurtosis) < 0.15
    # the two routes: same marginal, same correlation, within Monte Carlo error
    assert abs(A.var() - B.var()) < 0.05
    assert abs(np.mean(A[:, :-1] * A[:, 1:]) - np.mean(B[:, :-1] * B[:, 1:])) < 0.05
    # stationary: the first and the last site have the same law
    assert abs(A[:, 0].var() - A[:, -1].var()) < 0.08


def test_single_chain_sampler_is_the_batch_sampler():
    chain = Phi4Chain(0.5, 1.0)
    a = chain.sample(rng_for("field-one", 1), 8)
    b = chain.sample_batch(rng_for("field-one", 1), 1, 8)[0]
    assert np.array_equal(a, b)


def test_fourth_cumulant_diagonal_is_the_marginal_excess_kurtosis():
    chain = Phi4Chain(0.5, 1.0)
    kap = chain.fourth_cumulant_tensor(6)
    assert abs(kap[2, 2, 2, 2] - chain.innovation_excess_kurtosis) < 1e-6
    # symmetric under index permutations, by construction
    assert np.allclose(kap, np.transpose(kap, (1, 0, 2, 3)))
    assert np.allclose(kap, np.transpose(kap, (0, 1, 3, 2)))


def test_exact_bp_reduces_posterior_error_below_a_naive_denoiser():
    """A control that BP with this kernel and initial law is a working denoiser."""
    chain = Phi4Chain(0.5, 1.0)
    g, w = make_grid(8.0, 401)
    A = chain.sample_batch(rng_for("field-bp", 0), 64, 16)
    t = 0.4665
    al, de = float(np.exp(-t)), float(1 - np.exp(-2 * t))
    X = al * A + np.sqrt(de) * rng_for("field-bp-noise", 0).standard_normal(A.shape)
    m, _ = grid_bp_batch(g, w, chain.log_transition_matrix(g), X, al, de, log_mu=chain.log_stationary(g))
    err_bp = np.mean((m - A) ** 2)
    err_lin = np.mean((al * X / (al**2 + de) - A) ** 2)   # site-wise Gaussian shrinkage
    assert err_bp < 0.8 * err_lin


def test_iso_correlation_solver_hits_the_target():
    h = h_for_correlation(0.5, 0.85)
    assert abs(Phi4Chain(0.5, h).rho - 0.85) < 2e-3
