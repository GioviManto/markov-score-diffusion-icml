"""A Laplace chain whose kernel changes along the sequence (E68).

    a_1 ~ N(0, s_1^2),    a_{k+1} = alpha_k a_k + eps_k,    eps_k ~ Laplace(0, b_k),

with the link's lag-one correlation rho_k and the sites' standard deviations s_k as the knobs:
alpha_k = rho_k s_{k+1} / s_k and 2 b_k^2 = s_{k+1}^2 (1 - rho_k^2), so Corr(a_k, a_{k+1}) = rho_k and Var a_k = s_k^2.
A strength `delta` in [0, 1] scales the departure from E03's chain (rho = 0.85, s = 1 everywhere; delta = 0 IS that
chain, `priors.LaplaceAR1(0.85)`):

    drift    rho_k moves linearly from 0.85 - d_rho delta to 0.85 + d_rho delta along the links, and log s_k linearly
             from -d_logsd delta to +d_logsd delta along the sites;
    switch   the first half of the chain has (0.85 + d_rho delta, s = e^{-d_logsd delta}), the second half
             (0.85 - d_rho delta, s = e^{+d_logsd delta}); the link into site L/2 + 1 carries the change.

Every link is a Laplace shift kernel, so a per-link shift family is correctly specified; one kernel shared by all
links is not (unless delta = 0).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RHO0 = 0.85


@dataclass(frozen=True)
class NonstationaryLaplaceChain:
    schedule: str                    # "drift" or "switch" (any schedule at delta = 0 is E03's chain)
    delta: float
    n_sites: int = 32
    d_rho: float = 0.1
    d_logsd: float = 0.35

    @property
    def name(self) -> str:
        return f"{self.schedule}_d{self.delta:g}"

    def rho(self) -> np.ndarray:
        """(L - 1,) lag-one correlation of each link."""
        n = self.n_sites - 1
        if self.schedule == "drift":
            return RHO0 + self.d_rho * self.delta * np.linspace(-1.0, 1.0, n)
        if self.schedule == "switch":
            first = np.arange(n) < self.n_sites // 2 - 1
            return np.where(first, RHO0 + self.d_rho * self.delta, RHO0 - self.d_rho * self.delta)
        raise ValueError(f"unknown schedule {self.schedule!r}")

    def sd(self) -> np.ndarray:
        """(L,) standard deviation of each site."""
        if self.schedule == "drift":
            return np.exp(self.d_logsd * self.delta * np.linspace(-1.0, 1.0, self.n_sites))
        if self.schedule == "switch":
            first = np.arange(self.n_sites) < self.n_sites // 2
            return np.exp(np.where(first, -1.0, 1.0) * self.d_logsd * self.delta)
        raise ValueError(f"unknown schedule {self.schedule!r}")

    def alpha(self) -> np.ndarray:
        s = self.sd()
        return self.rho() * s[1:] / s[:-1]

    def b(self) -> np.ndarray:
        s = self.sd()
        return np.sqrt(s[1:] ** 2 * (1.0 - self.rho() ** 2) / 2.0)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """(n, L) chains; the first site is drawn from N(0, s_1^2), then one Laplace innovation per link."""
        a = np.empty((n, self.n_sites))
        a[:, 0] = self.sd()[0] * rng.standard_normal(n)
        eps = rng.laplace(0.0, 1.0, size=(n, self.n_sites - 1)) * self.b()
        al = self.alpha()
        for k in range(1, self.n_sites):
            a[:, k] = al[k - 1] * a[:, k - 1] + eps[:, k - 1]
        return a

    def log_kernels(self, grid: np.ndarray) -> np.ndarray:
        """(L - 1, N_g, N_g) log K_k(u_i | u_j), the analytic Laplace densities (not renormalised, as in `priors`)."""
        grid = np.asarray(grid, dtype=float)
        e = grid[None, :, None] - self.alpha()[:, None, None] * grid[None, None, :]
        b = self.b()[:, None, None]
        return -np.abs(e) / b - np.log(2.0 * b)

    def log_mu(self, grid: np.ndarray) -> np.ndarray:
        v = self.sd()[0] ** 2
        return -0.5 * np.asarray(grid) ** 2 / v - 0.5 * np.log(2.0 * np.pi * v)

    def covariance(self) -> np.ndarray:
        """Exact covariance: Cov(a_i, a_j) = s_i s_j prod_{k=i}^{j-1} rho_k."""
        s, r = self.sd(), self.rho()
        L = self.n_sites
        logcum = np.concatenate([[0.0], np.cumsum(np.log(r))])
        i = np.arange(L)
        lo, hi = np.minimum.outer(i, i), np.maximum.outer(i, i)
        return np.outer(s, s) * np.exp(logcum[hi] - logcum[lo])
