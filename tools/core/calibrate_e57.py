"""Calibrate E57's two new chains from their TRUE kernels on a wide grid: no data are drawn and nothing is fitted.

    .venv/bin/python tools/core/calibrate_e57.py

Both chains are matched to the Laplace chain of E03/E56 in variance (1) and lag-one correlation (0.85), so that the
covariance a linear method sees is the same in every chain of E56 and E57 and any difference comes from the shape.

    tanh drift        a' = A tanh(a) + sigma eps          (A, sigma) solved for Var = 1, corr = 0.85
    heteroscedastic   a' = 0.85 a + s0 sqrt(1 + c a^2) eps  s0^2 = (1 - 0.85^2) / (1 + c) gives Var = 1 for any c

The stationary law is the leading eigenvector of the grid operator; the grid is wide (|a| <= 12) so that the tails of the
heteroscedastic chain are inside it, and the mass beyond the registered E03 grid (|a| > 8) is printed.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve
from scipy.sparse.linalg import eigs

G = np.linspace(-12.0, 12.0, 1201)
W = np.full(G.size, G[1] - G[0])
W[[0, -1]] *= 0.5
ALPHA, CORR = 0.85, 0.85


def stats(log_k: np.ndarray):
    """Mean, variance, lag-one correlation, excess kurtosis, mass beyond |a| > 8 and the stationary law."""
    k = np.exp(log_k - log_k.max(0, keepdims=True))
    k /= (k * W[:, None]).sum(0, keepdims=True)
    p = k * W[:, None]                                       # column-stochastic: p[i, j] = Prob(g_j -> g_i)
    _, vec = eigs(p, k=1, which="LM")
    pi = np.abs(np.real(vec[:, 0]))
    pi /= pi.sum()
    m = pi @ G
    v = pi @ (G - m) ** 2
    cond_mean = (p * G[:, None]).sum(0)                      # E[a' | a = g_j]
    c1 = (pi * (G - m) * (cond_mean - m)).sum() / v
    kurt = pi @ (G - m) ** 4 / v**2 - 3.0
    return m, v, c1, kurt, float(pi[np.abs(G) > 8].sum()), pi


def tanh_log_k(a_amp: float, sigma: float) -> np.ndarray:
    return -0.5 * (G[:, None] - a_amp * np.tanh(G[None, :])) ** 2 / sigma**2


def hetero_log_k(c: float) -> np.ndarray:
    s0 = np.sqrt((1 - ALPHA**2) / (1 + c))
    sd = s0 * np.sqrt(1 + c * G[None, :] ** 2)
    return -0.5 * ((G[:, None] - ALPHA * G[None, :]) / sd) ** 2 - np.log(sd)


def main() -> None:
    def resid(p):
        _, v, c1, *_ = stats(tanh_log_k(p[0], abs(p[1])))
        return [v - 1.0, c1 - CORR]

    a_amp, sigma = fsolve(resid, [1.5, 0.4], xtol=1e-10)
    sigma = abs(sigma)
    _, v, c1, kurt, tail, pi = stats(tanh_log_k(a_amp, sigma))
    modes = int(((pi[1:-1] > pi[:-2]) & (pi[1:-1] > pi[2:])).sum())
    print(f"tanh drift: A = {a_amp:.6f}, sigma = {sigma:.6f} -> Var {v:.6f}, lag-1 corr {c1:.6f}, "
          f"marginal excess kurtosis {kurt:.3f}, {modes} mode(s), mass beyond |a| > 8: {tail:.1e}")
    for c in (0.25, 0.5, 1.0):
        _, v, c1, kurt, tail, _ = stats(hetero_log_k(c))
        print(f"heteroscedastic c = {c}: s0 = {np.sqrt((1 - ALPHA**2) / (1 + c)):.6f} -> Var {v:.6f}, "
              f"lag-1 corr {c1:.6f}, marginal excess kurtosis {kurt:.3f}, mass beyond |a| > 8: {tail:.1e}")


if __name__ == "__main__":
    main()
