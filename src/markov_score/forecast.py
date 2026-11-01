"""Probabilistic forecasting with a Markov kernel on the grid (E59).

The h-step predictive law of a first-order chain observed up to site k is

    p(a_{k+h} | x_{1:k}) = K^h applied to p(a_k | x_{1:k}),

so a forecast is a BP forward filter over the past (or a point mass at the last value when the past is clean) followed
by h applications of the kernel: h matrix-vector products on the grid. Densities live on the grid with the trapezoid
weights w, normalised so that sum_i w_i p_i = 1; K[i, j] = K(g_i | g_j) is column-normalised the same way.

Scores, the same for every method so that a kernel and a quantile network are compared on one scale:
    CRPS from 19 quantiles  (2 / Q) sum_q pinball_q(y - Q_q)
    coverage of the 5-95% interval, the median's error, and for a density the exact CRPS and the log score.
"""
from __future__ import annotations

import numpy as np

LEVELS = np.round(np.arange(0.05, 0.951, 0.05), 2)          # 19 quantile levels, 5% ... 95%


def _column_normalised(log_k: np.ndarray, w: np.ndarray) -> np.ndarray:
    k = np.exp(log_k - log_k.max(axis=0, keepdims=True))
    return k / (k * w[:, None]).sum(axis=0, keepdims=True)


def propagate(p: np.ndarray, log_k: np.ndarray, w: np.ndarray, h: int) -> np.ndarray:
    """(B, Ng) densities after h steps of the chain."""
    k = _column_normalised(log_k, w)
    out = np.atleast_2d(p)
    for _ in range(h):
        out = (out * w[None, :]) @ k.T
    return out


def predictive_from_point(log_k, grid, w, a, h: int) -> np.ndarray:
    """(B, Ng) the law of the value h steps after a known value a (clean past): the kernel's column at a, interpolated
    linearly between the two neighbouring grid columns, then h - 1 further steps."""
    a = np.atleast_1d(np.asarray(a, dtype=float))
    k = _column_normalised(log_k, w)
    pos = np.clip((a - grid[0]) / (grid[1] - grid[0]), 0.0, len(grid) - 1 - 1e-9)
    j = np.floor(pos).astype(int)
    lam = pos - j
    first = (1.0 - lam)[:, None] * k[:, j].T + lam[:, None] * k[:, np.minimum(j + 1, len(grid) - 1)].T
    return propagate(first, log_k, w, h - 1)


def forward_filter(log_k, grid, w, X, al: float, de: float, log_mu: np.ndarray | None = None) -> np.ndarray:
    """(B, Ng) the filtered law of the last site given noisy observations X (B, k) of x = al a + sqrt(de) xi."""
    k = _column_normalised(log_k, w)
    X = np.atleast_2d(X)
    mu = np.exp(log_mu) if log_mu is not None else np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi)

    def lik(x):
        return np.exp(-0.5 * (x[:, None] - al * grid[None, :]) ** 2 / de)

    f = mu[None, :] * lik(X[:, 0])
    f /= (f * w[None, :]).sum(axis=1, keepdims=True)
    for i in range(1, X.shape[1]):
        f = ((f * w[None, :]) @ k.T) * lik(X[:, i])
        f /= (f * w[None, :]).sum(axis=1, keepdims=True)
    return f


def quantiles(p, grid, w, levels=LEVELS) -> np.ndarray:
    """(B, Q) quantiles of grid densities, from the trapezoid CDF by linear interpolation."""
    p = np.atleast_2d(p)
    cell = 0.5 * (p[:, 1:] + p[:, :-1]) * np.diff(grid)[None, :]
    cdf = np.concatenate([np.zeros((len(p), 1)), np.cumsum(cell, axis=1)], axis=1)
    cdf /= cdf[:, -1:]
    return np.stack([np.interp(levels, c, grid) for c in cdf])


def crps_from_quantiles(q, y, levels=LEVELS) -> np.ndarray:
    """(B,) 2/Q times the summed pinball losses: the CRPS of the quantile forecast, the same rule for every method."""
    q = np.sort(np.atleast_2d(q), axis=1)                    # repair quantile crossing, as is standard
    d = np.asarray(y, dtype=float)[:, None] - q
    pin = np.maximum(levels[None, :] * d, (levels[None, :] - 1.0) * d)
    return 2.0 * pin.mean(axis=1)


def crps_exact(p, grid, w, y) -> np.ndarray:
    """(B,) integral of (F(z) - 1{z >= y})^2 over the grid, F the trapezoid CDF of each density.

    The step is evaluated at the grid points, so y between two of them costs a relative error of order 1e-4 at the
    project's spacing (1.4e-4 against the Gaussian closed form in tests/test_forecast.py). E59 uses it as a descriptive
    number only; every registered comparison uses `crps_from_quantiles`, the same rule for every method."""
    p = np.atleast_2d(p)
    cell = 0.5 * (p[:, 1:] + p[:, :-1]) * np.diff(grid)[None, :]
    F = np.concatenate([np.zeros((len(p), 1)), np.cumsum(cell, axis=1)], axis=1)
    F /= F[:, -1:]
    step = (grid[None, :] >= np.asarray(y, dtype=float)[:, None]).astype(float)
    return ((F - step) ** 2 * w[None, :]).sum(axis=1)


def coverage(q, y, levels=LEVELS, lo=0.05, hi=0.95) -> np.ndarray:
    """(B,) 1 where y lies in [Q_lo, Q_hi]."""
    q = np.sort(np.atleast_2d(q), axis=1)
    i, j = int(np.argmin(np.abs(levels - lo))), int(np.argmin(np.abs(levels - hi)))
    y = np.asarray(y, dtype=float)
    return ((y >= q[:, i]) & (y <= q[:, j])).astype(float)


def log_score(p, grid, y) -> np.ndarray:
    """(B,) log predictive density at y (linear interpolation on the grid)."""
    p = np.atleast_2d(p)
    vals = np.array([np.interp(yy, grid, pp) for pp, yy in zip(p, np.asarray(y, dtype=float))])
    return np.log(np.maximum(vals, 1e-300))
