"""Grid BP in its simplest form: the algorithm the paper describes, one sequence at a time.

Data: x_i = alpha a_i + sqrt(delta) z_i on a chain a_1 -> a_2 -> ... -> a_L with transition density K(a' | a)
and first-site law mu(a); alpha = e^{-t}, delta = 1 - e^{-2t}. The posterior of the clean chain given x is a
chain too, so belief propagation is exact on it. The only approximation is to represent every message by its
values on an equally spaced grid u_1..u_n and to integrate with the trapezoid rule (weights w).

    forward   m_1 = mu,         m_{i+1}(u) = sum_v K(u | v) m_i(v) ell_i(v) w(v)
    backward  n_L = 1,          n_{i-1}(v) = sum_u K(u | v) n_i(u) ell_i(u) w(u)
    belief    b_i(u) proportional to m_i(u) ell_i(u) n_i(u)
    mean      E[a_i | x] = sum_u u b_i(u) w(u)
    score     S_i(x) = -(x_i - alpha E[a_i | x]) / delta          (Tweedie)

with ell_i(u) = exp(-(x_i - alpha u)^2 / (2 delta)) the likelihood of site i. Two details keep the numbers
finite and change nothing else: each likelihood row is divided by its maximum, and each message is divided
by its total mass (beliefs are normalised at the end anyway).

The production code (`bp_grid.grid_bp_batch`) runs the same recursion on many sequences at once, on CPU or
GPU, with optional diagnostics. tests/test_bp_minimal.py holds the two to agreement at 1e-10.
"""
from __future__ import annotations

import numpy as np


def make_grid(half_width: float, n: int):
    """n equally spaced points on [-half_width, half_width] and their trapezoid weights."""
    u = np.linspace(-half_width, half_width, n)
    w = np.full(n, u[1] - u[0])
    w[0] /= 2
    w[-1] /= 2
    return u, w


def posterior_mean(x, t, K, mu, u, w):
    """E[a_i | x] for one noisy sequence x of length L. K[j, k] = K(u_j | u_k), mu[k] = mu(u_k)."""
    alpha, delta = np.exp(-t), 1 - np.exp(-2 * t)
    L, n = len(x), len(u)
    log_ell = -(x[:, None] - alpha * u[None, :]) ** 2 / (2 * delta)
    ell = np.exp(log_ell - log_ell.max(axis=1, keepdims=True))

    fwd = np.empty((L, n))
    fwd[0] = mu
    for i in range(L - 1):
        m = K @ (fwd[i] * ell[i] * w)
        fwd[i + 1] = m / (m @ w)

    bwd = np.empty((L, n))
    bwd[L - 1] = 1.0
    for i in range(L - 1, 0, -1):
        m = K.T @ (bwd[i] * ell[i] * w)
        bwd[i - 1] = m / (m @ w)

    belief = fwd * ell * bwd
    belief /= (belief @ w)[:, None]
    return belief @ (u * w)


def score(x, t, K, mu, u, w):
    """The exact score of the noisy chain at level t (up to the grid), by Tweedie's identity."""
    alpha, delta = np.exp(-t), 1 - np.exp(-2 * t)
    return -(x - alpha * posterior_mean(x, t, K, mu, u, w)) / delta
