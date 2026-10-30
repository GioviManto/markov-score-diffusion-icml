"""E57's two new chains: their kernels are densities, they carry the calibration E57.json registered (variance 1, lag-one
correlation 0.85, from tools/core/calibrate_e57.py), and their samplers draw from those kernels."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from markov_score.priors import HeteroscedasticAR1, TanhDriftChain

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "core"))
import calibrate_e57 as cal  # noqa: E402

CHAINS = [TanhDriftChain(), HeteroscedasticAR1()]


@pytest.mark.parametrize("prior", CHAINS, ids=lambda p: p.name)
def test_columns_are_densities(prior):
    g = np.linspace(-12, 12, 2401)
    w = np.full(g.size, g[1] - g[0]); w[[0, -1]] *= 0.5
    mass = np.exp(prior.log_transition_matrix(g)).T @ w
    inner = np.abs(g) < 4
    np.testing.assert_allclose(mass[inner], 1.0, atol=1e-10)


def test_the_defaults_are_the_registered_calibration():
    t, h = TanhDriftChain(), HeteroscedasticAR1()
    assert (t.amp, t.sigma) == (1.303746, 0.497509)
    assert (h.rho, h.c) == (0.85, 0.25) and abs(h.s0 - 0.471169) < 1e-6
    _, v, c1, *_ = cal.stats(cal.tanh_log_k(t.amp, t.sigma))
    assert abs(v - 1) < 1e-5 and abs(c1 - 0.85) < 1e-5
    _, v, c1, *_ = cal.stats(cal.hetero_log_k(h.c))
    assert abs(v - 1) < 1e-3 and abs(c1 - 0.85) < 1e-3


@pytest.mark.parametrize("prior", CHAINS, ids=lambda p: p.name)
def test_the_kernel_is_the_one_used_by_the_calibration(prior):
    g = cal.G
    ref = cal.tanh_log_k(prior.amp, prior.sigma) if isinstance(prior, TanhDriftChain) else cal.hetero_log_k(prior.c)
    mine = prior.log_transition_matrix(g)
    d = mine - ref                                       # equal up to a constant per column (the normaliser)
    np.testing.assert_allclose(d - d[0][None, :], 0.0, atol=1e-9)


@pytest.mark.parametrize("prior", CHAINS, ids=lambda p: p.name)
def test_the_sampler_draws_from_the_kernel(prior):
    rng = np.random.default_rng(0)
    A = np.stack([prior.sample(rng, 64) for _ in range(3000)])
    a, b = A[:, 20:-1].ravel(), A[:, 21:].ravel()            # past the start, near-stationary
    assert abs(a.var() - 1) < 0.06 and abs(np.corrcoef(a, b)[0, 1] - 0.85) < 0.02
    for lo, hi in ((-1.5, -0.5), (-0.25, 0.25), (0.5, 1.5)):
        sel = (a > lo) & (a < hi)
        x = a[sel]
        if isinstance(prior, TanhDriftChain):
            np.testing.assert_allclose((b[sel] - prior.amp * np.tanh(x)).mean(), 0.0, atol=0.02)
            np.testing.assert_allclose((b[sel] - prior.amp * np.tanh(x)).std(), prior.sigma, rtol=0.03)
        else:
            z = (b[sel] - prior.rho * x) / (prior.s0 * np.sqrt(1 + prior.c * x**2))
            np.testing.assert_allclose(z.mean(), 0.0, atol=0.03)
            np.testing.assert_allclose(z.std(), 1.0, rtol=0.03)


def test_the_chains_start_from_a_standard_normal():
    rng = np.random.default_rng(1)
    for prior in CHAINS:
        first = np.array([prior.sample(rng, 2)[0] for _ in range(20000)])
        assert abs(first.mean()) < 0.03 and abs(first.std() - 1) < 0.02
