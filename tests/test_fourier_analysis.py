"""The three analytical Q5 results, pinned (tools/fourier_analysis.py).

Each test pins a DERIVATION against the project's own exact BP or against a case with a known
answer, not against a previously recorded number. The Cramer tests carry their controls with
them, because the first two explanations for E23's cumulant failure were both wrong and it was
controls that showed it.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "experiments"))

from numpy.polynomial.hermite_e import hermeval  # noqa: E402

from fourier_analysis import (_cramer_standardised, laplace_cf,  # noqa: E402
                             message_cf_closed_form)
from markov_score.bp_grid import grid_bp, make_grid  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402

RHO, L = 0.85, 32


def test_closed_form_matches_grid_bp():
    """phi_{k+1}(w) = phi_mu(rho^k w) prod_j phi_eps(rho^j w), against grid BP with a flat
    likelihood. Tolerance is the grid's quadrature error, not a fitted constant."""
    prior = LaplaceAR1(RHO); b = prior.b
    grid, w = make_grid(12.0, 4001)
    r = grid_bp(grid, w, prior.log_transition_matrix(grid), np.zeros((L, len(grid))),
                np.zeros(L), 1.0, 1.0, return_messages=True)
    omega = np.linspace(0, 40, 801)
    for k in (1, 2, 8, L - 1):
        meas = (r.forward[k][None, :] * np.exp(1j * omega[:, None] * grid[None, :])
                * w[None, :]).sum(axis=1).real
        assert np.abs(meas - message_cf_closed_form(omega, k, RHO, b)).max() < 1e-6


def test_infinite_product_is_the_stationary_law():
    """As k grows the message CF approaches prod_{j>=0} phi_eps(rho^j w) -- the AR(1)'s
    stationary CF. This is why E22's P3 failed: no finite-parameter family contains it."""
    b = float(np.sqrt((1 - RHO**2) / 2))
    omega = np.linspace(0, 20, 401)
    inf_prod = np.ones_like(omega)
    for j in range(2000):
        inf_prod *= laplace_cf(RHO**j * omega, b)
    prev = np.inf
    for k in (4, 8, 16, 32, 64):
        gap = float(np.abs(message_cf_closed_form(omega, k, RHO, b) - inf_prod).max())
        assert gap < prev, f"not approaching the stationary CF at k={k}"
        prev = gap
    assert prev < 1e-6


@pytest.mark.parametrize("rho", [0.5, 0.7, 0.85, 0.95])
def test_mode_decay_law(rho):
    """log|phi_inf(w)| ~ -(log b|w|)^2 / log(1/rho): the slope against (log b w)^2 must be
    -1/log(1/rho) to within 10%, with R^2 above 0.999."""
    b = float(np.sqrt((1 - rho**2) / 2))
    om = np.exp(np.linspace(np.log(3.0), np.log(3e4), 40))
    lp = np.array([-np.sum(np.log1p((b * rho ** np.arange(0, 20000) * o) ** 2)) for o in om])
    X = (np.log(b * om)) ** 2
    A = np.vstack([X, np.ones_like(X)]).T
    c, *_ = np.linalg.lstsq(A, lp, rcond=None)
    r2 = 1 - (lp - A @ c).var() / lp.var()
    assert r2 > 0.999
    assert 0.9 < c[0] / (-1.0 / np.log(1.0 / rho)) < 1.1


def test_tilt_bandwidth_shrinks_with_noise():
    """sigma_w = alpha/sqrt(Delta) is DECREASING in t: the likelihood tilt is widest at low
    noise. Stated as a test because it is the opposite of the natural guess."""
    ts = [0.05, 0.2216, 0.4665, 0.9821, 2.0676, 3.0]
    sw = [np.exp(-t) / np.sqrt(1 - np.exp(-2 * t)) for t in ts]
    assert all(sw[i] < sw[i - 1] for i in range(1, len(sw)))
    assert sw[0] > 3.0 and sw[-1] < 0.06


def test_cramer_control_gaussian_is_exactly_one():
    grid, w = make_grid(8.0, 3201)
    p = np.exp(-0.5 * grid**2 / 0.4) / np.sqrt(2 * np.pi * 0.4)
    val, _ = _cramer_standardised(p, grid, w)
    assert abs(val - 1.0) < 1e-9


def test_cramer_divergence_needs_a_WIDER_domain_not_a_finER_grid():
    """The methodological point that made the first answer wrong: grid refinement tests
    QUADRATURE, domain widening tests TAIL INTEGRABILITY. For a standardised Laplace the
    integral is flat under grid refinement at fixed domain, and explodes when widened."""
    def lap(g):
        return np.exp(-np.abs(g) * np.sqrt(2.0)) * np.sqrt(2.0) / 2

    fixed = []
    for ng in (1601, 3201, 6401, 12801):
        g, w = make_grid(8.0, ng)
        fixed.append(_cramer_standardised(lap(g), g, w)[0])
    assert max(fixed) / min(fixed) < 1.01, f"refinement should NOT move it: {fixed}"

    widened = []
    for half in (8.0, 12.0, 16.0, 20.0):
        g, w = make_grid(half, 20001)
        widened.append(_cramer_standardised(lap(g), g, w)[0])
    assert all(widened[i] > 100 * widened[i - 1] for i in range(1, len(widened))), widened


def test_cramer_fails_for_the_actual_bp_messages():
    """Every standardised BP message on the Laplace chain violates Cramer's condition, at
    every noise level -- so the Gram-Charlier expansion provably cannot converge in L^2."""
    from markov_score.noising import alpha_delta, log_likelihood_matrix
    from markov_score.utils import rng_for
    prior = LaplaceAR1(RHO)
    A = prior.sample(rng_for("fourier-test"), L)
    rng = rng_for("fourier-test-noise")
    for t in (0.05, 0.4665, 3.0):
        al, de = alpha_delta(t)
        x = al * A + np.sqrt(de) * rng.standard_normal(L)
        vals = []
        for half in (8.0, 16.0):
            g, w = make_grid(half, 8001)
            r = grid_bp(g, w, prior.log_transition_matrix(g),
                        log_likelihood_matrix(g, x, al, de), x, al, de, return_messages=True)
            p = r.forward[16]; p = p / (p * w).sum()
            vals.append(_cramer_standardised(p, g, w)[0])
        assert vals[1] / vals[0] > 1e3, f"t={t}: expected divergence, got {vals}"


def test_gram_charlier_machinery_is_exact_on_a_gaussian():
    """The control that ruled out arithmetic as the explanation: for a Gaussian message every
    c_n with n >= 3 vanishes and the order-30 reconstruction is exact."""
    grid, w = make_grid(20.0, 20001)
    p = np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi)
    m = float((grid * p * w).sum()); v = float(((grid - m) ** 2 * p * w).sum())
    s = np.sqrt(v); z = (grid - m) / s
    ph = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    coef = np.zeros(31)
    for n in range(31):
        e = np.zeros(n + 1); e[n] = 1.0
        coef[n] = float((hermeval(z, e) * p * w).sum()) / math.factorial(n)
    assert np.abs(coef[3:]).max() < 1e-12
    rec = ph * hermeval(z, coef) / s
    assert float(np.sqrt(((p - rec) ** 2 * w).sum())) < 1e-10
