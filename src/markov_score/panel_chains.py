"""The two kinds of chain E61 adds to markov_score.priors: a skewed transformed-Gaussian chain and lattice diffusions.

Both have the priors' interface (`name`, `sample(rng, n)`, `log_transition_matrix(grid)`) and start from a_1 ~ N(0, 1), the
convention of every prior in the package, so exact grid BP with the default initial law is the exact reference.

SinhArcsinhAR1   a = (T(z) - m) / s, z a Gaussian AR(1) of correlation r, T(z) = sinh(asinh(z) + eps): full support,
                 Gaussian tails of different scales on the two sides; kernel and stationary law in closed form.
LatticeDiffusion a reversible 1-D diffusion with closed-form stationary law pi and diffusion coefficient D, sampled every
                 Delta. Generator: the square-root approximation on the 0.02 lattice of [-12, 12] (rate D h^-2
                 sqrt(pi_j / pi_i) between neighbours, detailed balance with pi exactly); transition: its matrix
                 exponential over Delta. A sample jumps between lattice cells and sits uniformly inside its cell, so the
                 kernel at the lattice points is exact for the process sampled. Any grid that is a subset of the lattice
                 (E03's: 401 points on [-8, 8], spacing 0.04) reads the kernel off directly.

`skew_chain()`, `langevin_chain()` and `pearson_chain()` build E61's registered members (experiments/registry/E61.json),
solving for the parameters exactly as tools/core/calibrate_e61.py did; tests/test_panel_chains.py checks them against
the registered values.
"""
from __future__ import annotations

from functools import cache

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

RHO, SKEW = 0.85, 1.0
LATTICE = np.linspace(-12.0, 12.0, 1201)
_H = float(LATTICE[1] - LATTICE[0])


# ------------------------------------------------------------------ the skewed chain


def _t(z, eps):
    return np.sinh(np.arcsinh(z) + eps)


def _t_inv(x, eps):
    return np.sinh(np.arcsinh(x) - eps)


def skew_moments(eps):
    """Mean, sd and skewness of T(z), z ~ N(0, 1) (quadrature on 24001 points over |z| <= 12)."""
    z = np.linspace(-12.0, 12.0, 24001)
    pz = norm.pdf(z) * (z[1] - z[0])
    x = _t(z, eps)
    m = pz @ x
    s = np.sqrt(pz @ (x - m) ** 2)
    return float(m), float(s), float((pz @ (x - m) ** 3) / s ** 3)


def skew_lag1(eps, r):
    m, s, _ = skew_moments(eps)
    zz, ww = np.polynomial.hermite_e.hermegauss(120)
    ww = ww / ww.sum()
    a1 = (_t(zz, eps) - m) / s
    a2 = (_t(r * zz[:, None] + np.sqrt(1 - r ** 2) * zz[None, :], eps) - m) / s
    return float(ww @ (a1[:, None] * a2 @ ww))


class SinhArcsinhAR1:
    def __init__(self, eps: float, r: float):
        self.eps, self.r = float(eps), float(r)
        self.m, self.s, _ = skew_moments(self.eps)

    @property
    def name(self) -> str:
        return "skew"

    def log_transition_matrix(self, grid: np.ndarray) -> np.ndarray:
        x = self.m + self.s * np.asarray(grid)
        z = _t_inv(x, self.eps)
        dz = self.s * np.cosh(np.arcsinh(x) - self.eps) / np.sqrt(1 + x ** 2)
        q = 1 - self.r ** 2
        return -0.5 * (z[:, None] - self.r * z[None, :]) ** 2 / q - 0.5 * np.log(2 * np.pi * q) + np.log(dz)[:, None]

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        a = np.empty(n)
        a[0] = rng.standard_normal()
        z = _t_inv(self.m + self.s * a[0], self.eps)
        for i in range(1, n):
            z = self.r * z + np.sqrt(1 - self.r ** 2) * rng.standard_normal()
            a[i] = (_t(z, self.eps) - self.m) / self.s
        return a


@cache
def skew_chain() -> SinhArcsinhAR1:
    eps = brentq(lambda e: skew_moments(e)[2] - SKEW, 0.01, 2.0, xtol=1e-12)
    r = brentq(lambda r: skew_lag1(eps, r) - RHO, 0.5, 0.99, xtol=1e-12)
    return SinhArcsinhAR1(eps, r)


# ------------------------------------------------------------------ lattice diffusions


class LatticeDiffusion:
    def __init__(self, name: str, log_pi: np.ndarray, d_mid: np.ndarray, delta: float):
        self.name_, self.delta = name, float(delta)
        log_pi = np.maximum(log_pi, log_pi.max() - 60.0)     # mass below e^-60 of the peak: rates stay finite
        up = d_mid / _H ** 2 * np.exp(0.5 * (log_pi[1:] - log_pi[:-1]))       # rate i -> i + 1
        down = d_mid / _H ** 2 * np.exp(0.5 * (log_pi[:-1] - log_pi[1:]))     # rate i + 1 -> i
        q = np.diag(up, -1) + np.diag(down, 1)                                  # q[to, from]
        q -= np.diag(q.sum(0))
        s = np.exp(0.5 * (log_pi - log_pi.max()))
        self._s = s
        self._lam, self._u = np.linalg.eigh((q * s[None, :]) / s[:, None])
        self.p = self.transition(self.delta)                                    # column-stochastic, P[i, j] = Prob(j -> i)
        self._cdf = np.cumsum(self.p, axis=0)

    def transition(self, delta: float) -> np.ndarray:
        p = (self._u * np.exp(self._lam * delta)[None, :]) @ self._u.T
        p = np.clip(p * self._s[:, None] / self._s[None, :], 0.0, None)
        return p / p.sum(0, keepdims=True)

    def lag1(self, delta: float) -> float:
        p = self.transition(delta)
        pi = self._s ** 2 / np.sum(self._s ** 2)
        m = pi @ LATTICE
        v = pi @ (LATTICE - m) ** 2
        return float((pi * (LATTICE - m) * ((p * LATTICE[:, None]).sum(0) - m)).sum() / v)

    @property
    def name(self) -> str:
        return self.name_

    def log_k_lattice(self) -> np.ndarray:
        return np.log(np.maximum(self.p, 1e-300)) - np.log(_H)

    def log_transition_matrix(self, grid: np.ndarray) -> np.ndarray:
        idx = np.rint((np.asarray(grid) - LATTICE[0]) / _H).astype(int)
        if idx.min() < 0 or idx.max() >= LATTICE.size or not np.allclose(LATTICE[idx], grid, atol=1e-9):
            raise ValueError("the grid must be a subset of the diffusion's 0.02 lattice on [-12, 12]")
        return self.log_k_lattice()[np.ix_(idx, idx)]

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        i = int(np.clip(np.rint((rng.standard_normal() - LATTICE[0]) / _H), 0, LATTICE.size - 1))
        out = np.empty(n)
        for k in range(n):
            if k:
                i = min(int(np.searchsorted(self._cdf[:, i], rng.random())), LATTICE.size - 1)
            out[k] = LATTICE[i] + _H * (rng.random() - 0.5)
        return out


def _matched(name, log_pi_fn, d_fn, delta0=0.1):
    """The scale for variance 1, then Delta for a lag-one correlation of 0.85."""
    def var(sc):
        lp = log_pi_fn(sc)
        p = np.exp(lp - lp.max())
        p /= p.sum()
        return p @ LATTICE ** 2 - (p @ LATTICE) ** 2
    sc = brentq(lambda sc: var(sc) - 1.0, 0.2, 5.0, xtol=1e-12)
    g_mid = 0.5 * (LATTICE[1:] + LATTICE[:-1])
    probe = LatticeDiffusion(name, log_pi_fn(sc), d_fn(sc, g_mid), delta0)
    delta = brentq(lambda dl: probe.lag1(dl) - RHO, 1e-3, 10.0, xtol=1e-12)
    return LatticeDiffusion(name, log_pi_fn(sc), d_fn(sc, g_mid), delta)


@cache
def langevin_chain(barrier: float = 2.0) -> LatticeDiffusion:
    """dX = -V'(X) dt + sqrt(2) dW, V = barrier (x^2 / s^2 - 1)^2: a double well at unit temperature."""
    return _matched("langevin", lambda s: -barrier * (LATTICE ** 2 / s ** 2 - 1) ** 2, lambda s, g: np.ones_like(g))


@cache
def pearson_chain(nu: float = 6.0) -> LatticeDiffusion:
    """The Student diffusion dX = -X dt + sqrt(2 (nu + X^2) / (nu - 1)) dW in units a = X / c with Var a = 1:
    pi(a) ~ (1 + c^2 a^2 / nu)^(-(nu + 1) / 2), D(a) = (nu + c^2 a^2) / ((nu - 1) c^2)."""
    return _matched("pearson", lambda c: -0.5 * (nu + 1) * np.log1p(c ** 2 * LATTICE ** 2 / nu),
                    lambda c, g: (nu + c ** 2 * g ** 2) / ((nu - 1) * c ** 2))
