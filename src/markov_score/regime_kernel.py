"""A first-order chain whose noise switches between two persistent regimes, on a doubled grid (E73).

    r_{k+1} | r_k ~ P[r_{k+1}, r_k],      a_{k+1} = alpha a_k + s_{r_{k+1}} eps_k,   eps ~ N(0, 1).

The value a alone is not first-order Markov (the regime remembers the past), but the pair (r, a) is. On a grid g of
N points the pair lives on the DOUBLED grid [g, g] (index r * N + i): the noisy observation depends on a only, so the
doubled grid's values are the base grid repeated, and every grid routine of the package (grid_bp_batch, em.e_step,
fit_em, the stationary eigensolve) runs on it unchanged and marginalises the regime automatically. The weights of the
doubled grid are the base trapezoid weights repeated.

theta = (alpha, s0, s1, p00, p11); P[0, 0] = p00, P[1, 0] = 1 - p00, P[1, 1] = p11, P[0, 1] = 1 - p11.
The M-step is closed form (an ECM step): stay probabilities from the regime blocks of xi, then alpha by weighted least
squares given the noise scales, then each scale from its block's residuals. Every M-step returns a NEW kernel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def doubled(grid: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.concatenate([grid, grid]), np.concatenate([weights, weights])


def _base(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(grid) // 2
    g = np.asarray(grid[:n], dtype=float)
    if not np.array_equal(g, grid[n:]):
        raise ValueError("RegimeKernel needs the doubled grid [g, g]")
    dx = float(g[1] - g[0])
    w = np.full(n, dx)
    w[0] *= 0.5
    w[-1] *= 0.5
    return g, w


@dataclass(frozen=True)
class RegimeKernel:
    alpha: float
    s0: float
    s1: float
    p00: float
    p11: float

    @property
    def theta(self) -> np.ndarray:
        return np.array([self.alpha, self.s0, self.s1, self.p00, self.p11])

    @property
    def P(self) -> np.ndarray:
        return np.array([[self.p00, 1.0 - self.p11], [1.0 - self.p00, self.p11]])

    @classmethod
    def init(cls, alpha=0.3, s0=0.5, s1=1.0, p00=0.9, p11=0.9) -> "RegimeKernel":
        return cls(float(alpha), float(s0), float(s1), float(p00), float(p11))

    def log_transition_matrix(self, grid: np.ndarray) -> np.ndarray:
        """log K[(r', i), (r, j)] on the doubled grid, each column a density over (r', i) (trapezoid on a)."""
        g, w = _base(grid)
        n = len(g)
        out = np.empty((2 * n, 2 * n))
        logP = np.log(np.clip(self.P, 1e-300, None))
        for ro, s in enumerate((self.s0, self.s1)):
            lg = -0.5 * ((g[:, None] - self.alpha * g[None, :]) / s) ** 2
            lg = lg - np.log((np.exp(lg) * w[:, None]).sum(axis=0))[None, :]        # normalised on the grid
            for ri in range(2):
                out[ro * n:(ro + 1) * n, ri * n:(ri + 1) * n] = logP[ro, ri] + lg
        return out

    def m_step(self, stats, grid: np.ndarray) -> "RegimeKernel":
        g, _ = _base(grid)
        n = len(g)
        xi = np.asarray(stats.xi, dtype=float)
        B = {(ro, ri): xi[ro * n:(ro + 1) * n, ri * n:(ri + 1) * n] for ro in range(2) for ri in range(2)}
        C = {k: float(v.sum()) for k, v in B.items()}
        p00 = C[(0, 0)] / max(C[(0, 0)] + C[(1, 0)], 1e-300)
        p11 = C[(1, 1)] / max(C[(1, 1)] + C[(0, 1)], 1e-300)
        gi, gj = g[:, None], g[None, :]
        s2 = (self.s0 ** 2, self.s1 ** 2)
        num = sum(((B[(ro, 0)] + B[(ro, 1)]) * gi * gj).sum() / s2[ro] for ro in range(2))
        den = sum(((B[(ro, 0)] + B[(ro, 1)]) * gj ** 2).sum() / s2[ro] for ro in range(2))
        alpha = float(num / den)
        scales = []
        for ro in range(2):
            m = B[(ro, 0)] + B[(ro, 1)]
            scales.append(float(np.sqrt(max((m * (gi - alpha * gj) ** 2).sum() / max(m.sum(), 1e-300), 1e-8))))
        return RegimeKernel(alpha, scales[0], scales[1], float(p00), float(p11))


def sample_regime_chain(kernel: RegimeKernel, rng: np.random.Generator, n: int, L: int, burn: int = 300) -> np.ndarray:
    """n chains of length L, started at a = 0, r ~ stationary, after `burn` steps."""
    P = kernel.P
    pi1 = (1 - kernel.p00) / ((1 - kernel.p00) + (1 - kernel.p11))
    r = (rng.random(n) < pi1).astype(int)
    a = np.zeros(n)
    out = np.empty((n, L))
    s = np.array([kernel.s0, kernel.s1])
    for k in range(burn + L):
        stay = rng.random(n) < P[r, r]
        r = np.where(stay, r, 1 - r)
        a = kernel.alpha * a + s[r] * rng.standard_normal(n)
        if k >= burn:
            out[:, k - burn] = a
    return out
