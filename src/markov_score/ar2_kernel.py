"""A second-order transition kernel, the thesis mixture with two lags: E62's pair-state repair.

    a_t = phi1 a_{t-1} + phi2 a_{t-2} + eps,      eps ~ sum_c pi_c N(mu_c, s2_c)              (2 + 3C parameters)

With phi2 = 0 it is markov_score.kernels.MixtureInnovationKernel. log_k2(grid)[k, j, i] = log K2(u_k | a_{t-1} = u_j,
a_{t-2} = u_i), normalised over k with the trapezoid weights, the convention of markov_score.pair_bp.

M-step (ECM, one sweep per EM iteration) on the expected triple counts xi3[k, j, i] (pair_bp.pair_e_step): with phi held,
the mixture by its closed-form weighted update on the residuals e = u_k - phi1 u_j - phi2 u_i; with the mixture held,
(phi1, phi2) by weighted least squares, a 2 x 2 system. The sweep raises the continuous-density Q; it is ACCEPTED only if the
grid-normalised Q (what BP and the E-step use) does not fall, so the M-step is a generalised M-step of the model EM fits.
The variances are floored at a quarter of the grid spacing squared (a component is never narrower than half a cell), and
phi is kept inside the stationary triangle.

fit_pair_em runs EM through pair-state BP with the kernel's own stationary pair law as the initial law (E33's rule),
saving the kernels at a ladder of iterations. Every M-step returns a NEW kernel (amendment N-25).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
from scipy.special import logsumexp

from markov_score.kernels import MixtureInnovationKernel
from markov_score.pair_bp import pair_e_step, stationary_pair


def _residual(phi, grid):
    return grid[:, None, None] - phi[0] * grid[None, :, None] - phi[1] * grid[None, None, :]


def _stationary_phi(phi, margin=1e-3):
    """Project (phi1, phi2) into the stationary triangle phi2 < 1, phi1 + phi2 < 1, phi2 - phi1 < 1 (with a margin)."""
    p1, p2 = float(phi[0]), float(min(phi[1], 1 - margin))
    p2 = max(p2, -1 + margin)
    lim = 1 - margin - p2
    p1 = float(np.clip(p1, -lim, lim))
    return np.array([p1, p2])


@dataclass(frozen=True)
class Ar2MixtureKernel:
    phi: np.ndarray
    pi: np.ndarray
    mu: np.ndarray
    s2: np.ndarray
    accepted: bool = True

    @classmethod
    def init(cls, components: int, rho: float = 0.3, var: float = 0.8, rng=None) -> Ar2MixtureKernel:
        """The AR(1) start of every first-order arm (phi = (rho, 0)) with MixtureInnovationKernel's innovation start."""
        m = MixtureInnovationKernel.init(components, rho=rho, var=var, rng=rng)
        return cls(phi=np.array([float(m.rho), 0.0]), pi=np.asarray(m.pi, float), mu=np.asarray(m.mu, float),
                   s2=np.asarray(m.s2, float))

    @property
    def name(self) -> str:
        return f"ar2_mixture_C{len(self.pi)}"

    @property
    def theta(self) -> np.ndarray:
        return np.concatenate([self.phi, self.pi, self.mu, self.s2])

    def _log_comp(self, e):
        return (np.log(self.pi)[:, None, None, None] - 0.5 * np.log(2 * np.pi * self.s2)[:, None, None, None]
                - 0.5 * (e[None] - self.mu[:, None, None, None]) ** 2 / self.s2[:, None, None, None])

    def log_k2(self, grid: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        lk = logsumexp(self._log_comp(_residual(self.phi, grid)), axis=0)
        if weights is None:
            return lk
        return lk - np.log(np.einsum("kji,k->ji", np.exp(lk), weights))[None]

    def q(self, xi3, grid, weights) -> float:
        return float(np.sum(xi3 * self.log_k2(grid, weights)) / max(np.sum(xi3), 1e-300))

    def m_step(self, xi3: np.ndarray, grid: np.ndarray, weights: np.ndarray) -> Ar2MixtureKernel:
        floor = 0.25 * float(grid[1] - grid[0]) ** 2
        e = _residual(self.phi, grid)
        lc = self._log_comp(e)
        r = np.exp(lc - logsumexp(lc, axis=0, keepdims=True))                  # (C, n, n, n)
        wc = r * xi3[None]
        tot = wc.sum(axis=(1, 2, 3))
        pi = tot / tot.sum()
        mu = (wc * e[None]).sum(axis=(1, 2, 3)) / tot
        s2 = np.maximum((wc * (e[None] - mu[:, None, None, None]) ** 2).sum(axis=(1, 2, 3)) / tot, floor)
        lam = (wc / s2[:, None, None, None]).sum(0)                            # sum_c W_c / s2_c
        lam_mu = (wc * (mu / s2)[:, None, None, None]).sum(0)
        uj, ui, uk = grid[None, :, None], grid[None, None, :], grid[:, None, None]
        a = np.array([[np.sum(lam * uj * uj), np.sum(lam * uj * ui)], [np.sum(lam * uj * ui), np.sum(lam * ui * ui)]])
        y = lam * uk - lam_mu
        b = np.array([np.sum(y * uj), np.sum(y * ui)])
        phi = _stationary_phi(np.linalg.solve(a, b))
        new = replace(self, phi=phi, pi=pi, mu=mu, s2=s2, accepted=True)
        if new.q(xi3, grid, weights) < self.q(xi3, grid, weights) - 1e-12:
            return replace(self, accepted=False)
        return new


def fit_pair_em(kernel, grid, weights, groups, ladder, log=None):
    """EM through pair-state BP; groups = [(X, alpha, delta), ...]; returns {iteration: kernel} at the ladder and the
    seconds per iteration. The initial law of every chain is the current kernel's stationary pair law."""
    saved, mu2, t0 = {}, None, time.perf_counter()
    top = max(ladder)
    for it in range(top + 1):
        if it in ladder:
            saved[it] = kernel
        if it == top:
            break
        lk2 = kernel.log_k2(grid, weights)
        mu2 = stationary_pair(lk2, weights, init=mu2)
        log_mu2 = np.log(np.maximum(mu2, 1e-300))
        xi3 = sum(pair_e_step(grid, weights, lk2, X, al, de, log_mu2) for X, al, de in groups)
        kernel = kernel.m_step(xi3, grid, weights)
        if log is not None and it % 20 == 0:
            log(f"    it {it}: phi {np.round(kernel.phi, 4).tolist()}")
    return saved, (time.perf_counter() - t0) / max(top, 1)


def clean_triples(grid, A):
    """xi3[k, j, i] for CLEAN chains A (B, L): the histogram of observed triples (a_t, a_{t-1}, a_{t-2}), binned
    trilinearly onto the grid (em.clean_statistics' rule in three coordinates: unit mass per triple, first moments kept)."""
    from markov_score.em import _interp_weights

    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = len(grid)
    out = np.zeros((n, n, n))
    ks, js, is_ = (_interp_weights(grid, A[:, s].ravel()) for s in (slice(2, None), slice(1, -1), slice(None, -2)))
    for k_idx, k_w in ((ks[0], 1 - ks[2]), (ks[1], ks[2])):
        for j_idx, j_w in ((js[0], 1 - js[2]), (js[1], js[2])):
            for i_idx, i_w in ((is_[0], 1 - is_[2]), (is_[1], is_[2])):
                np.add.at(out, (k_idx, j_idx, i_idx), k_w * j_w * i_w)
    return out
