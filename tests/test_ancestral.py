"""The ladder sampler and the grid-chain sampler of E55: the bridge has the right law, the exact denoiser on a Gaussian chain
generates the chain up to the plug-in variance deficit (measured here, so the E55 floor is known), and a grid chain's exact
samples have its moments."""

from __future__ import annotations

import numpy as np

from markov_score import ancestral as an
from markov_score import theory
from markov_score.bp_grid import make_grid

LEVELS = (0.05, 0.0725, 0.1053, 0.1527, 0.2216, 0.3215, 0.4665, 0.6769, 0.9821, 1.425, 2.0676, 3.0)


def test_bridge_step_has_the_ou_law():
    rng = np.random.default_rng(0)
    a, t_hi, t_lo, n = 0.7, 1.0, 0.3, 400_000
    x_hi = np.exp(-t_hi) * a + np.sqrt(1 - np.exp(-2 * t_hi)) * rng.standard_normal(n)
    x_lo = an.bridge_step(x_hi, np.full(n, a), t_hi, t_lo, rng)
    assert abs(x_lo.mean() - np.exp(-t_lo) * a) < 5e-3
    assert abs(x_lo.var() - (1 - np.exp(-2 * t_lo))) < 5e-3


def _ar1(n, L, rho, rng):
    A = np.zeros((n, L)); A[:, 0] = rng.standard_normal(n)
    for i in range(1, L):
        A[:, i] = rho * A[:, i - 1] + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    return A


def test_the_twelve_level_ladder_cannot_generate_even_with_the_exact_denoiser():
    """Measured 28 Oct 2026 (rho = 0.85, L = 32, 20000 sequences, exact linear denoiser): the twelve frozen levels give
    variance 0.675 (plug-in posterior variance), 0.79 (DDPM beta variance), 0.75 (DDIM) against 1. Eleven steps from
    t = 3 to 0.05 are too coarse: generation needs a fine ladder, hence a score at every level of it."""
    rng = np.random.default_rng(1)
    L, rho = 32, 0.85
    sigma = rho ** np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
    G = an.generate(lambda X, t: theory.lmmse_posterior_mean(X, sigma, t), 20_000, L, LEVELS, rng)
    assert 0.6 < G.var() < 0.75


def test_a_fine_ladder_generates_a_gaussian_chain_with_the_exact_denoiser():
    """Measured 28 Oct 2026: 400 geometric levels from 3 to 1e-3 give variance 0.981 and lag-1 correlation 0.851 (target
    1 and 0.85); 200 levels 0.966 / 0.849; 100 levels 0.928 / 0.847."""
    rng = np.random.default_rng(1)
    L, rho = 32, 0.85
    sigma = rho ** np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
    G = an.generate(lambda X, t: theory.lmmse_posterior_mean(X, sigma, t), 20_000, L, np.geomspace(3.0, 1e-3, 401), rng)
    var = G.var()
    assert 0.97 < var < 1.0
    assert abs(np.mean(G[:, :-1] * G[:, 1:]) / var - rho) < 0.01


def test_grid_chain_samples_have_the_chains_moments():
    rng = np.random.default_rng(2)
    grid, w = make_grid(6.0, 201)
    rho = 0.8
    lk = -0.5 * (grid[:, None] - rho * grid[None, :]) ** 2 / (1 - rho**2)
    lk -= np.log(np.exp(lk).T @ w)[None, :]
    lmu = -0.5 * grid**2
    S = an.sample_grid_chain(lk, lmu, grid, w, 20_000, 16, rng)
    assert abs(S.var() - 1.0) < 0.03
    assert abs(np.mean(S[:, :-1] * S[:, 1:]) / S.var() - rho) < 0.02
