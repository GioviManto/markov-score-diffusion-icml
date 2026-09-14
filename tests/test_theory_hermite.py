"""Regression tests for the order-by-order information-decay theory (docs/THEORY_NOTES.md).

Each test pins one statement the paper makes:
  * Theorem A: Hermite moments of degree n attenuate exactly by e^{-nt} (Monte Carlo, scalar);
  * the chain's fourth-cumulant tensor against brute-force cumulants of sampled chains;
  * Theorem B3: the closed-form Gaussian-chain Fisher information tends to
    (1/2)||d Sigma_0||_F^2 e^{-4t} and its local log-slope tends to -4 beyond the crossover;
  * the scalar leading constants kappa_4^2/6 (score gap) and 1/2 (variance information)
    against fine quadrature, which is the check THEORY_NOTES §4 reports.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import gamma as G
from scipy.signal import fftconvolve

from markov_score import theory
from markov_score.priors import LaplaceAR1


def test_hermite_moments_attenuate_exactly():
    rng = np.random.default_rng(0)
    a = rng.laplace(scale=1 / np.sqrt(2), size=200_000)  # unit variance, kappa_4 = 3
    for n, t in [(2, 0.3), (4, 0.3), (4, 1.0), (6, 0.5)]:
        lhs, rhs = theory.hermite_moment_attenuation(a, t, n, rng, n_noise=4)
        # Monte Carlo error of the left side: sd(He_n(x))/sqrt(N); allow 5 sd.
        assert abs(lhs - rhs) < 5 * (abs(rhs) * 0.05 + 0.02), (n, t, lhs, rhs)


def test_chain_kappa4_matches_sampled_cumulants():
    rho, L, n = 0.7, 6, 400_000
    prior = LaplaceAR1(rho)
    rng = np.random.default_rng(1)
    A = np.stack([prior.sample(rng, L) for _ in range(n)])
    K = theory.chain_kappa4(rho, theory.excess_kurtosis_to_kappa4(3.0, prior.q), L)
    # brute-force fourth cumulants for a few index sets (zero-mean, symmetric law)
    def cum4(i, j, k, l):
        a, b, c, d = (A[:, s] for s in (i, j, k, l))
        m = np.mean(a * b * c * d)
        return m - np.mean(a * b) * np.mean(c * d) - np.mean(a * c) * np.mean(b * d) - np.mean(a * d) * np.mean(b * c)
    for idx in [(0, 0, 0, 0), (1, 1, 1, 1), (5, 5, 5, 5), (2, 2, 3, 3), (1, 2, 3, 4), (4, 5, 5, 5)]:
        est = cum4(*idx)
        se = 12.0 / np.sqrt(n)  # crude bound on the sampling sd of a fourth moment of unit-variance data
        assert abs(est - K[idx]) < 4 * se + 0.01, (idx, est, K[idx])
    assert K[0, 0, 0, 0] == 0.0  # the first site is Gaussian


def test_gaussian_chain_information_reaches_the_degree_two_rate():
    rho, q, L = 0.85, 1 - 0.85**2, 32
    out = {t: theory.gaussian_chain_fisher(rho, q, L, t) for t in (2.5, 3.5, 4.5, 5.5)}
    # scaled by e^{4t} the information converges to the constant (1/2)||d Sigma_0||^2
    for key in ("asymptote_rho", "asymptote_q"):
        pass
    for t in (3.5, 4.5, 5.5):
        r = out[t]["diag"][0] * np.exp(4 * t) / out[t]["asymptote_rho"]
        assert abs(r - 1.0) < 0.15 * np.exp(-2 * (t - 2.5)) + 0.02, (t, r)
    slope = (np.log(out[5.5]["diag"][0]) - np.log(out[4.5]["diag"][0])) / 1.0
    assert abs(slope + 4.0) < 0.05, slope
    # below the crossover the slope is shallower than -4, not steeper
    lo = theory.gaussian_chain_fisher(rho, q, L, 0.5)["diag"][0]
    hi = theory.gaussian_chain_fisher(rho, q, L, 1.0)["diag"][0]
    assert (np.log(hi) - np.log(lo)) / 0.5 > -4.0
    assert 1.1 < theory.crossover_time(rho) < 1.4


def _gg_pdf(u, beta):
    s = np.sqrt(G(1 / beta) / G(3 / beta))
    return beta / (2 * s * G(1 / beta)) * np.exp(-((np.abs(u) / s) ** beta))


def _noised(u, p0, t):
    al, de = np.exp(-t), 1 - np.exp(-2 * t)
    du = u[1] - u[0]
    pa = np.interp(u / al, u, p0, left=0, right=0) / al
    g = np.exp(-(u**2) / (2 * de)) / np.sqrt(2 * np.pi * de)
    pt = fftconvolve(pa, g, mode="same") * du
    return pt / np.sum(pt * du)


@pytest.mark.parametrize("beta,k4", [(1.0, 3.0), (4.0, -0.8116)])
def test_scalar_score_gap_constant_is_kappa4_squared_over_six(beta, k4):
    u = np.linspace(-16, 16, 8001)
    du = u[1] - u[0]
    p0 = _gg_pdf(u, beta)
    m2 = np.sum(u**2 * p0 * du)
    m4 = np.sum(u**4 * p0 * du)
    assert abs(m4 / m2**2 - 3 - k4) < 0.01
    vals = []
    for t in (1.25, 1.75):
        pt = _noised(u, p0, t)
        S = np.gradient(np.log(np.maximum(pt, 1e-300)), du)
        mask = np.abs(u) < 9
        vals.append(np.sum(((S + u) ** 2 * pt * du)[mask]) * np.exp(8 * t))
    pred = theory.score_gap_constant(np.array([k4]))
    for v in vals:
        assert abs(v / pred - 1.0) < 0.12, (beta, vals, pred)


def test_scalar_variance_information_constant_is_one_half():
    u = np.linspace(-16, 16, 8001)
    du = u[1] - u[0]
    p = lambda v: _gg_pdf(u / np.sqrt(v), 1.0) / np.sqrt(v)
    for t in (1.5, 2.0):
        h = 1e-3
        pp, pm, p0 = (_noised(u, p(v), t) for v in (1 + h, 1 - h, 1.0))
        dl = (np.log(np.maximum(pp, 1e-300)) - np.log(np.maximum(pm, 1e-300))) / (2 * h)
        I = np.sum((dl**2 * p0 * du)[np.abs(u) < 9])
        assert abs(I * np.exp(4 * t) - 0.5) < 0.02, (t, I * np.exp(4 * t))
