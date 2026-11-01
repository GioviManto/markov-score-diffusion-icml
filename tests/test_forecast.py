"""The forecasting maths of E59: on the Gaussian AR(1) chain the h-step predictive law, the Kalman filter and the CRPS
are known in closed form, and the grid versions must reproduce them; the SETAR chain carries its registered calibration
and its sampler draws from its kernel."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from markov_score import forecast as fc
from markov_score.bp_grid import make_grid
from markov_score.kernels import GaussianAR1Kernel
from markov_score.priors import SETARChain

RHO = 0.85
GRID, W = make_grid(8.0, 401)
LOGK = GaussianAR1Kernel(rho=RHO, q=1 - RHO**2).log_transition_matrix(GRID)


def test_predictive_from_a_point_is_the_closed_form():
    a = np.array([-1.234, 0.0, 0.517, 2.02])
    for h in (1, 2, 4, 8):
        p = fc.predictive_from_point(LOGK, GRID, W, a, h)
        np.testing.assert_allclose(p @ W, 1.0, atol=1e-10)
        mean = p @ (W * GRID)
        var = p @ (W * GRID**2) - mean**2
        np.testing.assert_allclose(mean, RHO**h * a, atol=2e-6)
        np.testing.assert_allclose(var, 1 - RHO ** (2 * h), atol=5e-4)


def test_the_filter_is_the_kalman_filter_on_the_gaussian_chain():
    rng = np.random.default_rng(0)
    al, de = np.exp(-0.2), 1 - np.exp(-0.4)
    a = np.zeros((5, 24)); a[:, 0] = rng.standard_normal(5)
    for i in range(1, 24):
        a[:, i] = RHO * a[:, i - 1] + np.sqrt(1 - RHO**2) * rng.standard_normal(5)
    X = al * a + np.sqrt(de) * rng.standard_normal(a.shape)
    p = fc.forward_filter(LOGK, GRID, W, X, al, de)
    m, v = np.zeros(5), np.ones(5)                         # scalar Kalman filter, prior N(0, 1) at site 1
    for i in range(24):
        if i > 0:
            m, v = RHO * m, RHO**2 * v + 1 - RHO**2
        k = v * al / (al**2 * v + de)
        m, v = m + k * (X[:, i] - al * m), (1 - k * al) * v
    np.testing.assert_allclose(p @ (W * GRID), m, atol=1e-8)
    np.testing.assert_allclose(p @ (W * GRID**2) - (p @ (W * GRID)) ** 2, v, atol=1e-8)


def test_quantiles_and_the_crps_of_a_gaussian():
    mu, s = 0.3, 0.7
    p = norm.pdf(GRID, mu, s)[None, :]
    levels = fc.LEVELS
    q = fc.quantiles(p, GRID, W, levels)
    np.testing.assert_allclose(q[0], norm.ppf(levels, mu, s), atol=2e-3)
    y = np.array([1.1])
    z = (y - mu) / s
    exact = s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    np.testing.assert_allclose(fc.crps_exact(p, GRID, W, y), exact, rtol=1e-3)   # the step sits between grid points
    approx = fc.crps_from_quantiles(q, y, levels)
    assert abs(approx[0] - exact[0]) / exact[0] < 0.08          # 19 quantiles: close, the same rule for every method


def test_pinball_and_coverage():
    q = np.array([[-1.0, 0.0, 1.0]]); levels = np.array([0.05, 0.5, 0.95])
    assert fc.coverage(q, np.array([0.5]), levels)[0] == 1.0 and fc.coverage(q, np.array([2.0]), levels)[0] == 0.0
    np.testing.assert_allclose(fc.crps_from_quantiles(q, np.array([0.0]), levels), 2 / 3 * (0.05 + 0 + 0.05))


def test_setar_carries_the_registered_calibration():
    s = SETARChain()
    assert (s.loc, s.scale) == (-0.380078, 1.262109)
    rng = np.random.default_rng(1)
    A = np.stack([s.sample(rng, 200) for _ in range(400)])[:, 50:]
    x, y = A[:, :-1].ravel(), A[:, 1:].ravel()
    assert abs(x.mean()) < 0.05 and abs(x.var() - 1) < 0.06 and abs(np.corrcoef(x, y)[0, 1] - 0.881) < 0.02
    g, w = make_grid(8.0, 401)
    inner = np.abs(g) < 5                                   # at the grid's edge part of the next step falls outside it
    np.testing.assert_allclose((np.exp(s.log_transition_matrix(g)).T @ w)[inner], 1.0, atol=1e-6)


@pytest.mark.parametrize("b", [-1.5, 0.0, 1.2])
def test_setar_kernel_and_sampler_agree_on_the_regimes(b):
    """In standard units: from b the raw value is a = loc + scale b; the next raw value has the regime's mean and sd."""
    s = SETARChain()
    g, w = make_grid(8.0, 801)
    j = int(np.argmin(np.abs(g - b)))
    a = s.loc + s.scale * g[j]
    phi, c, sd = (s.phi_l, s.c_l, s.s_l) if a <= s.r else (s.phi_h, s.c_h, s.s_h)
    col = np.exp(s.log_transition_matrix(g))[:, j]
    m = col @ (w * g)
    np.testing.assert_allclose(m, (phi * a + c - s.loc) / s.scale, atol=1e-6)
    np.testing.assert_allclose(np.sqrt(col @ (w * g**2) - m**2), sd / s.scale, rtol=1e-4)
    rng = np.random.default_rng(2)                          # and the sampler: one step from b, many times
    nxt = np.array([s.step(rng, g[j]) for _ in range(20000)])
    np.testing.assert_allclose(nxt.mean(), m, atol=0.02)
    np.testing.assert_allclose(nxt.std(), sd / s.scale, rtol=0.03)
