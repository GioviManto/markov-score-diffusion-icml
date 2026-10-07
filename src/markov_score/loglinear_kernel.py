"""The most agnostic kernel family of the field laboratory: a log-linear conditional kernel with a 2D spline score.

    K(a' | a) = exp S(a, a') / Z(a),      S(a, a') = sum_{p,q} C_{pq} B_p(a') B_q(a),      Z(a) = int exp S(a, a'') da''

with B_p cubic B-splines on [-A, A]. Nothing else is imposed: no Gaussian bond, no reversibility, no site energy. It contains the
site-energy kernel (a Gaussian bond is a smooth function of a' - a, and the tilt is a function of a' alone), the tilted-walk
splines of E33, and kernels they cannot represent (a heavy-tailed bond, a coupling that is not a function of a' - a). It is the
answer to the open question put to Jerome and Marc (docs/FIELD_MODEL.md, the report's section on questions): is the site-energy form
wide enough, or does the paper need a general pair kernel?

Why this parametrisation. The E-step gives the M-step one matrix, xi[k, j], the expected number of transitions u_j -> u_k
(markov_score.em.ExpectedStatistics). The complete-data log-likelihood is  Q(C) = sum_{kj} xi[k, j] log K[k, j],  which is CONCAVE in
C (a log-linear model: linear score, log-sum-exp normaliser), so the M-step has no local optimum and no eigensolve inside it, unlike
the pair-energy kernel. Its gradient is closed form:

    dQ/dS[k, j] = xi[k, j] - m_j w_k K[k, j],   m_j = sum_k xi[k, j];     dQ/dC = B^T (dQ/dS) B.

The M-step is the exact maximum of Q (default `smooth` = 0, `ridge` = 0), so EM is exact EM and the evidence is monotone like every
other arm's. Regularisation is the validation-selected checkpoint of the ladder, as for all E33 arms. An optional third-difference
P-spline penalty (`smooth` > 0) exists but is NOT used by E43: with clamped boundary knots a quadratic score (the Gaussian chain) is
not exactly free under a penalty on coefficient differences, so a penalised M-step trades plain likelihood for penalty and the
evidence is no longer monotone (found in the first test of the EM loop). There is no ridge on the raw coefficients either: the
coefficients of a Gaussian score are large, so even 1e-3 costs as much as the likelihood of a few hundred transitions. The
flat directions that remain, the functions of the origin a, cancel in Z and cost the optimiser nothing.

Orientation as every grid kernel in this package: log_transition_matrix(grid)[k, j] = log K(u_k | u_j), normalised over k with the
trapezoid weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.special import logsumexp

from markov_score.em import ExpectedStatistics
from markov_score.tilted_kernel import _quad_weights

DEFAULT_SMOOTH = 0.0
DEFAULT_RIDGE = 0.0


def bspline_basis(n_basis: int, half_width: float, x: np.ndarray) -> np.ndarray:
    """(len(x), n_basis) cubic B-splines on [-A, A] (clamped knots); they sum to one."""
    k = 3
    n_inner = n_basis - 2
    if n_inner < 2:
        raise ValueError("a cubic B-spline basis needs n_basis >= 4")
    inner = np.linspace(-half_width, half_width, n_inner)
    t = np.concatenate([np.full(k, -half_width), inner, np.full(k, half_width)])
    xc = np.clip(np.asarray(x, dtype=float), -half_width, half_width * (1 - 1e-12))
    return BSpline.design_matrix(xc, t, k).toarray()


def _third_difference(n: int) -> np.ndarray:
    """Third differences of the coefficients: zero for a quadratic score, so the Gaussian chain is not penalised."""
    return np.diff(np.eye(n), n=3, axis=0)


@dataclass(frozen=True)
class LogLinearKernel:
    """theta = C.ravel(), C of shape (n_basis, n_basis); C[p, q]: arrival basis p, origin basis q."""

    coef: np.ndarray
    half_width: float = 8.0
    smooth: float = DEFAULT_SMOOTH
    ridge: float = DEFAULT_RIDGE
    inner_iters: int = 0

    @property
    def name(self) -> str:
        return f"loglinear{self.n_basis}"

    @property
    def n_basis(self) -> int:
        return int(round(np.sqrt(np.asarray(self.coef).size)))

    @property
    def theta(self) -> np.ndarray:
        return np.asarray(self.coef, dtype=float).ravel()

    @property
    def C(self) -> np.ndarray:
        return np.asarray(self.coef, dtype=float).reshape(self.n_basis, self.n_basis)

    @classmethod
    def init(cls, n_basis: int, rho: float, var: float, half_width: float = 8.0, grid_size: int = 401,
             smooth: float = DEFAULT_SMOOTH, ridge: float = DEFAULT_RIDGE) -> "LogLinearKernel":
        """The Gaussian AR(1) kernel (rho, var) projected on the basis: S(a, a') = -(a' - rho a)^2 / 2 var by least squares."""
        grid = np.linspace(-half_width, half_width, grid_size)
        B = bspline_basis(n_basis, half_width, grid)
        target = -0.5 * (grid[:, None] - rho * grid[None, :]) ** 2 / var
        Bp = np.linalg.pinv(B)
        return cls(coef=(Bp @ target @ Bp.T).ravel(), half_width=float(half_width), smooth=smooth, ridge=ridge)

    # ------------------------------------------------------------------ the kernel

    def _score(self, grid: np.ndarray) -> np.ndarray:
        B = bspline_basis(self.n_basis, self.half_width, grid)
        return B @ self.C @ B.T

    def log_transition_matrix(self, grid: np.ndarray, grid_out: np.ndarray | None = None) -> np.ndarray:
        if grid_out is not None and not (len(grid_out) == len(grid) and np.allclose(grid_out, grid)):
            raise ValueError("LogLinearKernel is evaluated on one grid")
        grid = np.asarray(grid, dtype=float)
        S = self._score(grid)
        lw = np.log(_quad_weights(grid))
        return S - logsumexp(S + lw[:, None], axis=0)[None, :]

    # ------------------------------------------------------------------ the M-step

    def _objective(self, grid, B, lw, xi, col_mass):
        n = self.n_basis
        D = _third_difference(n)
        P2 = D.T @ D

        def f(z):
            C = z.reshape(n, n)
            S = B @ C @ B.T
            logZ = logsumexp(S + lw[:, None], axis=0)
            logK = S - logZ[None, :]
            q = float((xi * logK).sum())
            pen = 0.5 * self.smooth * float((C * (P2 @ C)).sum() + (C * (C @ P2)).sum()) + 0.5 * self.ridge * float((C * C).sum())
            Kmass = np.exp(logK + lw[:, None])
            G = B.T @ (xi - col_mass[None, :] * Kmass) @ B - self.smooth * (P2 @ C + C @ P2) - self.ridge * C
            return -(q - pen), -G.ravel()
        return f

    def m_step(self, stats: ExpectedStatistics, grid: np.ndarray, grid_out: np.ndarray | None = None,
               max_iter: int = 60) -> "LogLinearKernel":
        grid = np.asarray(grid, dtype=float)
        if grid_out is not None and not (len(grid_out) == len(grid) and np.allclose(grid_out, grid)):
            raise ValueError("LogLinearKernel.m_step fits on one grid")
        xi = np.asarray(stats.xi, dtype=float)
        B = bspline_basis(self.n_basis, self.half_width, grid)
        lw = np.log(_quad_weights(grid))
        f = self._objective(grid, B, lw, xi, xi.sum(axis=0))
        z0 = np.asarray(self.coef, dtype=float).ravel()
        res = minimize(f, z0, jac=True, method="L-BFGS-B", options=dict(maxiter=max_iter, maxcor=20, ftol=1e-13, gtol=1e-9))
        # a generalised M-step must not go down: keep the current kernel if the line search found nothing better
        if not np.isfinite(res.fun) or res.fun > f(z0)[0]:
            return self
        return LogLinearKernel(coef=res.x.reshape(self.n_basis, self.n_basis), half_width=self.half_width, smooth=self.smooth,
                               ridge=self.ridge, inner_iters=int(res.nit))
