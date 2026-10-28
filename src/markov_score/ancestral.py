"""Generation on a fixed ladder of noise levels with any denoiser, and exact sampling of a grid Markov chain (E55).

Why a ladder sampler. E55's networks are trained at the twelve frozen levels only, so a continuous-time reverse SDE would
query them at times they never saw. The DDPM ancestral sampler needs the denoiser at the ladder's levels and nothing else:
going down from level t_hi to t_lo it draws x_lo from the Gaussian bridge q(x_lo | x_hi, a) of the OU channel with the clean
value a replaced by the denoiser's posterior mean m(x_hi) ("x0 prediction", Ho et al. 2020). With

    q(x_lo | a) = N(alpha_lo a, Delta_lo),     q(x_hi | x_lo) = N(r x_lo, 1 - r^2),     r = e^{-(t_hi - t_lo)},

the bridge is N(mu, v) with 1/v = 1/Delta_lo + r^2 / (1 - r^2) and mu = v (alpha_lo m / Delta_lo + r x_hi / (1 - r^2)), site by
site. The output is the posterior-mean readout m(x) at the lowest level. Every arm runs the same sampler with the same noise
(common random numbers), and the exact denoiser through it measures the sampler's own error: plugging in the mean drops the
posterior variance of a, so even exact denoisers generate under-dispersed sequences unless the ladder is fine: on a
Gaussian chain the twelve frozen levels give variance 0.68 (target 1) and 400 geometric levels down to 1e-3 give 0.98
(tests/test_ancestral.py).

Exact sampling of a grid chain: a kernel learned inside BP can also generate WITHOUT diffusion, by ancestral sampling of its
own Markov chain on the grid (first site from its stationary law, each next site from the kernel's column).
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

Denoiser = Callable[[np.ndarray, float], np.ndarray]


def bridge_step(x_hi: np.ndarray, m: np.ndarray, t_hi: float, t_lo: float, rng: np.random.Generator) -> np.ndarray:
    """One draw of x_lo ~ q(x_lo | x_hi, a = m) for the OU channel."""
    a_lo, d_lo = math.exp(-t_lo), 1.0 - math.exp(-2.0 * t_lo)
    r = math.exp(-(t_hi - t_lo))
    d_s = 1.0 - r * r
    v = 1.0 / (1.0 / d_lo + r * r / d_s)
    mu = v * (a_lo * m / d_lo + r * x_hi / d_s)
    return mu + math.sqrt(v) * rng.standard_normal(np.shape(x_hi))


def generate(denoise: Denoiser, n: int, L: int, levels, rng: np.random.Generator) -> np.ndarray:
    """n sequences of L sites: x ~ N(0, 1) at the highest level (the stationary OU marginal of standardised data, up to
    alpha^2 = e^{-2 t_max}), ancestral bridge steps down the ladder, posterior-mean readout at the lowest level."""
    lv = sorted((float(t) for t in levels), reverse=True)
    x = rng.standard_normal((n, L))
    for t_hi, t_lo in zip(lv[:-1], lv[1:]):
        x = bridge_step(x, denoise(x, t_hi), t_hi, t_lo, rng)
    return denoise(x, lv[-1])


def sample_grid_chain(log_k: np.ndarray, log_mu: np.ndarray, grid: np.ndarray, weights: np.ndarray, n: int, L: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Exact ancestral samples of the grid Markov chain: log_k[k, j] = log K(u_k | u_j) (columns densities under
    `weights`), log_mu the first-site density. Values on the grid."""
    p0 = np.exp(log_mu - np.max(log_mu)) * weights
    p0 /= p0.sum()
    P = np.exp(log_k - np.max(log_k, axis=0, keepdims=True)) * weights[:, None]
    C = np.cumsum(P / P.sum(axis=0, keepdims=True), axis=0)          # column-wise CDFs
    idx = np.empty((n, L), dtype=np.int64)
    idx[:, 0] = np.searchsorted(np.cumsum(p0), rng.random(n), side="right").clip(0, len(grid) - 1)
    for i in range(1, L):
        u = rng.random(n)
        cols = C[:, idx[:, i - 1]]                                   # (N_g, n)
        idx[:, i] = (cols < u[None, :]).sum(axis=0).clip(0, len(grid) - 1)
    return grid[idx]
