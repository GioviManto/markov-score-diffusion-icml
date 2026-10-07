"""A stationary reversible chain on the BP grid, built from an arbitrary symmetric pair energy W(a, a').

    p(a_1, ..., a_L)  proportional to  prod_i exp W(a_i, a_{i+1}),        W(a, a') = W(a', a).

The transfer matrix T[k, j] = exp W(u_k, u_j) is symmetric; with the quadrature weights w, the symmetrised operator
M = diag(sqrt w) T diag(sqrt w) has a top eigenpair (lambda, v), psi = v / sqrt w, and the Markov kernel is the Doob transform

    K(u_k | u_j) = T[k, j] psi_k / (lambda psi_j),        sum_k w_k K(u_k | u_j) = 1,       stationary density proportional to psi^2.

This is the second laboratory of E43: a chain OUTSIDE the tilted-walk class (docs/FIELD_MODEL.md: Gaussian bond times a function of
the arrival point), obtained by giving the bond a heavy tail (Laplace) on a double-well site energy. Its states are the grid points,
so sampling is EXACT (inverse CDF per step) and the exact posterior mean is grid BP with this very kernel: the oracle of the
estimand, with no discretisation error between the data and the reference.

Orientation as every grid kernel in this package: log_transition_matrix(grid)[k, j] = log K(u_k | u_j), the arrival on the rows.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.special import logsumexp

from markov_score.bp_grid import make_grid


class GridPairChain:
    def __init__(self, W: np.ndarray, grid: np.ndarray, weights: np.ndarray):
        W = np.asarray(W, dtype=float)
        if W.shape != (len(grid), len(grid)) or not np.allclose(W, W.T):
            raise ValueError("W must be a symmetric matrix on the grid")
        self.grid, self.w = np.asarray(grid, dtype=float), np.asarray(weights, dtype=float)
        sw = np.sqrt(self.w)
        Wc = W - W.max()
        M = sw[:, None] * np.exp(Wc) * sw[None, :]
        vals, vecs = eigh(M)
        v = vecs[:, -1] * np.sign(vecs[:, -1].sum())
        if (v < -1e-12).any():
            raise ValueError("the top eigenvector of the transfer operator is not positive")
        self.lam = float(vals[-1])
        log_psi = np.log(np.maximum(v / sw, 1e-300))
        raw = Wc + log_psi[:, None] - log_psi[None, :]                      # [k, j]: arrival k, origin j
        self._logk = raw - np.log(self.lam) - 0.0
        self._logk -= logsumexp(self._logk + np.log(self.w)[:, None], axis=0)[None, :]   # remove round-off drift
        log_pi = 2.0 * log_psi + np.log(self.w)                             # in logs: far in the tails psi^2 underflows
        self._log_density = 2.0 * log_psi - logsumexp(log_pi)               # a DENSITY (integrates to one under w), finite everywhere
        self.p = np.exp(log_pi - logsumexp(log_pi))                         # stationary PROBABILITY of each grid point
        self._P = np.exp(self._logk) * self.w[:, None]                      # P[k, j] = prob(next = k | now = j)
        self._cdf = np.cumsum(self._P, axis=0)

    # ------------------------------------------------------------------ the interface E33's scoring path uses

    def log_transition_matrix(self, grid: np.ndarray, grid_out: np.ndarray | None = None) -> np.ndarray:
        if len(grid) != len(self.grid) or not np.allclose(grid, self.grid):
            raise ValueError("this chain lives on one grid")
        return self._logk

    def log_stationary(self, grid: np.ndarray) -> np.ndarray:
        if len(grid) != len(self.grid) or not np.allclose(grid, self.grid):
            raise ValueError("this chain lives on one grid")
        return self._log_density

    def covariance(self, L: int) -> np.ndarray:
        """C(r) = E[a_0 a_r] under the stationary chain (the mean is zero by the symmetry a -> -a), as a Toeplitz matrix."""
        a = self.grid
        c, m = [float((self.p * a * a).sum())], a.copy()
        for _ in range(1, L):
            m = self._P.T @ m                                               # E[a_next | a_now = u_j]
            c.append(float((self.p * a * m).sum()))
        i = np.arange(L)
        return np.asarray(c)[np.abs(i[:, None] - i[None, :])]

    def sample_batch(self, rng: np.random.Generator, n_chains: int, n_sites: int) -> np.ndarray:
        j = np.minimum(np.searchsorted(np.cumsum(self.p), rng.random(n_chains)), len(self.grid) - 1)
        out = np.empty((n_chains, n_sites))
        out[:, 0] = self.grid[j]
        for i in range(1, n_sites):
            u = rng.random(n_chains)
            j = np.minimum((self._cdf[:, j] < u[None, :]).sum(axis=0), len(self.grid) - 1)
            out[:, i] = self.grid[j]
        return out

    # ------------------------------------------------------------------ diagnostics

    @property
    def variance(self) -> float:
        return float((self.p * self.grid**2).sum())

    @property
    def excess_kurtosis(self) -> float:
        return float((self.p * self.grid**4).sum() / self.variance**2 - 3.0)

    @property
    def lag1_correlation(self) -> float:
        return float(self.covariance(2)[0, 1] / self.variance)

    def corr_length_sites(self, L: int = 64) -> float:
        C = self.covariance(L)[0] / self.variance
        return float(1.0 + 2.0 * np.sum(C[1:]))


def laplace_double_well(bond: float, depth: float, wells: float = 1.0, half_width: float = 8.0, n_grid: int = 401) -> GridPairChain:
    """W(a, a') = -|a' - a| / bond - depth (V(a) + V(a')) / 2,  V(a) = (a^2 - wells^2)^2 / 4: a Laplace bond on a double well."""
    grid, w = make_grid(half_width, n_grid)
    V = (grid**2 - wells**2) ** 2 / 4.0
    W = -np.abs(grid[:, None] - grid[None, :]) / bond - 0.5 * depth * (V[:, None] + V[None, :])
    return GridPairChain(W, grid, w)
