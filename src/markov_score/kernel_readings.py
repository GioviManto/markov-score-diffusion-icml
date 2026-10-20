"""Physics read off ANY transition kernel on a grid: the bond variance and the site energy.

Why this module exists (Jerome, call of 19 Oct 2026, 00:23-00:28 and 01:38): the temperature and the
potential were read off the spline kernel through its own parameters. Can the same readings be made on a
neural (MDN) kernel, which has no such parameters? They can, through two identities.

A kernel of the gradient class (the phi^4 chain, the site-energy kernel, the spline tilted walk) is

    K(a' | a) = exp( -(a' - a)^2 / (2 gamma) + u(a') + v(a) ),        u + v = -2 f + const,

where f is the site energy. Two facts follow, whatever f is:

  (1) the mixed second difference of log K is exactly 1/gamma, at every pair of grid points,
          [log K(a'+d | a+d) - log K(a'+d | a) - log K(a' | a+d) + log K(a' | a)] / d^2 = 1 / gamma,
      because u(a') and v(a) cancel and so does every term of the bond except the cross term;
  (2) the diagonal gives the site energy, f(a) = -1/2 log K(a | a) + const: the bond vanishes at a' = a.

For a kernel outside the class, (1) is no longer constant over (a, a'). We average it with the weight of the
kernel's own stationary pair law pi(a) K(a'|a), so it is read where the data are, and report how much it
varies (`curvature_cv`, zero inside the class): a reading of how far the kernel is from the gradient class.

Units: on the standardised grid a = phi / s, gamma = Theta h / s^2, so Theta_hat = gamma_hat s^2 / h, and the
potential is V_hat(phi) = (2 Theta_hat / h) f_hat(phi / s) + const (`pair_energy_kernel.potential_readings`).
"""
from __future__ import annotations

import numpy as np

from markov_score.pair_energy_kernel import potential_readings


def mixed_curvature(log_K: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """(N-1, N-1) mixed second differences of log K[out, in], divided by the grid step squared."""
    d = float(grid[1] - grid[0])
    with np.errstate(invalid="ignore"):          # -inf - -inf in unresolved far tails: NaN, masked by the caller
        return (log_K[1:, 1:] - log_K[1:, :-1] - log_K[:-1, 1:] + log_K[:-1, :-1]) / d**2


def stationary_law(log_K: np.ndarray, weights: np.ndarray, iters: int = 20_000, tol: float = 1e-13) -> np.ndarray:
    """The kernel's invariant density on the grid (power iteration, unit quadrature mass)."""
    K = np.exp(log_K)                       # K[out, in], each column integrates to one under `weights`
    p = np.full(len(weights), 1.0 / weights.sum())
    for _ in range(iters):
        q = K @ (p * weights)
        q /= q @ weights
        if np.max(np.abs(q - p)) < tol * np.max(q):
            return q
        p = q
    return p


def bond_reading(log_K: np.ndarray, grid: np.ndarray, weights: np.ndarray) -> dict:
    """gamma_hat = 1 / (pair-law average of the mixed curvature) and the spread of the curvature."""
    pi = stationary_law(log_K, weights)
    pair = np.exp(log_K) * (pi * weights)[None, :] * weights[:, None]      # P(a_in, a_out), sums to one
    w = pair[:-1, :-1]
    c = mixed_curvature(log_K, grid)
    ok = np.isfinite(c) & (w > 0)
    w = np.where(ok, w, 0.0)
    w = w / w.sum()
    mean = float(np.sum(w * np.where(ok, c, 0.0)))
    var = float(np.sum(w * (np.where(ok, c, mean) - mean) ** 2))
    return {"gamma_hat": 1.0 / mean, "curvature_mean": mean, "curvature_cv": float(np.sqrt(var) / abs(mean))}


def site_energy_from_diagonal(log_K: np.ndarray) -> np.ndarray:
    """f_hat(a) = -1/2 log K(a | a), defined up to a constant."""
    return -0.5 * np.diag(log_K).copy()


def readings(log_K: np.ndarray, grid: np.ndarray, weights: np.ndarray, scale: float, h: float) -> dict:
    """Theta_hat, barrier, well and potential RMS of any kernel, plus its distance from the gradient class."""
    b = bond_reading(log_K, grid, weights)
    f = site_energy_from_diagonal(log_K)
    return {**b, **potential_readings(f, grid, b["gamma_hat"], scale, h)}
