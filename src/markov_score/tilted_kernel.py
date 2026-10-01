"""The tilted random walk: the transition family that contains the phi^4 chain exactly.

Why this kernel exists
----------------------
`field.py` derives that the Gibbs chain of a one-dimensional gradient SPDE has the kernel

    K(a' | a) = N(a'; a, gamma) g(a') / Z(a),        Z(a) = int N(a'; a, gamma) g(a') da',

a Gaussian step tilted by a positive function of the ARRIVAL point and normalised per column.
That is Jerome's "pair energy" W(a, a') = (a' - a)^2 / 2 gamma - log g(a') with the normaliser
that the h-transform supplies -- and the normaliser is exactly what column normalisation on
the grid computes. Learning this family by EM is therefore the exact maximum-likelihood
counterpart of Boltzmann learning of the pair energy: the gradient of Q is the centred
statistic "posterior expectation minus model expectation" (`m_step`), with no chain-wide
partition function anywhere, because the model is written as a transition kernel.

Parameters: log gamma, and the tilt log g(a') = sum_m theta_m B_m(a') on a linear basis --
Legendre polynomials on [-A, A] (a few global coefficients: the parsimonious rung) or cubic
B-splines (local coefficients: the flexible rung). A constant basis function is deliberately
absent: it is absorbed by Z. The Gaussian AR(1) chain is the member with a quadratic tilt
(`field.py`), which is the initialisation and the first correctness check.

The M-step
----------
Q(theta) = <Xi, log K_theta> is CONCAVE in the tilt coefficients at fixed gamma: it is linear
in theta minus a sum of log-partition functions of exponential families (one per column).
It is maximised jointly in (log gamma, theta) by L-BFGS-B from the current parameters with the
analytic gradient

    dQ/dtheta_m   = sum_k B_m(u_k) sum_j Xi~[k, j],
    dQ/dlog gamma = sum_kj Xi~[k, j] [ (u_k - u_j)^2 / 2 gamma - 1/2 ],
    Xi~[k, j]     = Xi[k, j] - (sum_k Xi[k, j]) p_j(k),

p_j the model's column distribution under the grid quadrature. The optimiser's result is
accepted only if Q did not decrease (a rejected step returns the receiver's parameters in a
NEW object, keeping the checkpoint ladder honest -- tests/test_kernel_mstep_contract.py), so
the outer EM is monotone in the edge term by construction. The record of each inner solve is
carried on the returned kernel, as `MixtureInnovationKernel` carries its ECM sweeps.

Like `MDNKernel`, the kernel is column-normalised under the grid's trapezoid weights, so the
fitted object is a proper density of the discretised model whose likelihood EM maximises.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from markov_score.em import ExpectedStatistics

_LOG_2PI = float(np.log(2.0 * np.pi))
_LOG_GAMMA_BOUNDS = (-9.0, 5.0)
_COEF_BOUND = 200.0
_Q_DECREASE_TOL = 1e-10
WALL_FRACTION = 0.625
WALL_POWER = 12
"""The tilt carries a FIXED soft wall -(a / A_w)^12 with A_w = WALL_FRACTION x half-width (5 on
the (401, 8) grid). Why: the data occupy |a| < 4 on every cell, so the basis functions beyond
that are constrained only through the normalisers of far columns, and a free fit can raise the
tilt at the grid edge and create a second, metastable pocket there -- invisible to the
likelihood, but it is where the kernel's invariant law then puts mass, and the power iteration
for that law stalled at a residual of 1e-8 in the first smoke run (1 October 2026). The wall
is 0.002 at |a| = 3, 0.07 at 4, 1 at 5 and 280 at 8: nothing where the data are, and more
than any coefficient within the optimiser's bounds (200) can undo where the grid ends
(tests/test_stationary.py pins the adversarial case). The true tilt of the phi^4 chain is
already -0.2 a^4 at |a| = 4, so the wall sits far inside it and distorts nothing the fit
can learn; on a Gaussian chain it costs 2e-3 in log K at three standard deviations."""


def soft_wall(half_width: float, x: np.ndarray) -> np.ndarray:
    return -(np.asarray(x, dtype=float) / (WALL_FRACTION * half_width)) ** WALL_POWER


def _quad_weights(grid: np.ndarray) -> np.ndarray:
    dx = float(grid[1] - grid[0])
    w = np.full(len(grid), dx)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def design_matrix(basis: str, n_basis: int, half_width: float, x: np.ndarray) -> np.ndarray:
    """(len(x), n_basis) basis values; no constant function (it is absorbed by Z)."""
    x = np.asarray(x, dtype=float)
    if basis == "legendre":
        from numpy.polynomial.legendre import legvander
        return legvander(np.clip(x / half_width, -1.0, 1.0), n_basis)[:, 1:]
    if basis == "bspline":
        from scipy.interpolate import BSpline
        k = 3
        # len(t) - k - 1 spline functions; we want n_basis + 1 of them (one is dropped below)
        n_inner = n_basis + 5 - 2 * k
        if n_inner < 2:
            raise ValueError("bspline needs n_basis >= 4")
        inner = np.linspace(-half_width, half_width, n_inner)
        t = np.concatenate([np.full(k, -half_width), inner, np.full(k, half_width)])
        xc = np.clip(x, -half_width, half_width * (1 - 1e-12))
        B = BSpline.design_matrix(xc, t, k).toarray()
        return B[:, 1:]  # drop one function: the B-splines sum to one, so one is a constant
    raise ValueError(f"unknown basis {basis!r}")


@dataclass(frozen=True)
class TiltedWalkKernel:
    """K(a'|a) = N(a'; a, gamma) exp(B(a') theta) / Z(a) on the grid. theta = (log gamma, coef)."""

    log_gamma: float
    coef: np.ndarray
    basis: str = "legendre"
    half_width: float = 8.0

    inner_iters: int = 0
    inner_converged: bool = False
    inner_q_gain: float = float("nan")
    """How the last M-step's L-BFGS-B ended: iterations taken, its own convergence flag, and
    the per-edge gain in Q it delivered (0.0 for a rejected step)."""

    @property
    def name(self) -> str:
        return f"tilted_{self.basis}{len(self.coef)}"

    @property
    def n_basis(self) -> int:
        return int(len(self.coef))

    @property
    def gamma(self) -> float:
        return float(np.exp(self.log_gamma))

    @property
    def theta(self) -> np.ndarray:
        return np.concatenate([[self.log_gamma], np.asarray(self.coef, dtype=float)])

    @classmethod
    def init(cls, basis: str, n_basis: int, grid: np.ndarray, rho: float, var: float,
             half_width: float | None = None) -> "TiltedWalkKernel":
        """The Gaussian AR(1) member with (rho, var): gamma = var / rho, tilt -(1 - rho) a^2 / 2 var,
        projected onto the basis by least squares on the grid. The same (rho, var) the mixture
        kernel starts from (`protocol.PROTOCOL.em_init_*`), so the arms share their start."""
        if not 0.0 < rho < 1.0:
            raise ValueError("the Gaussian AR(1) embedding needs 0 < rho < 1")
        A = float(np.max(np.abs(grid))) if half_width is None else float(half_width)
        B = design_matrix(basis, n_basis, A, grid)
        # the quadratic only: the soft wall is added by `log_tilt` on top of the basis, so the
        # embedding is "Gaussian chain times wall", exact in the bulk where the wall is ~0
        target = -(1.0 - rho) * np.asarray(grid, dtype=float) ** 2 / (2.0 * var)
        coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(grid)), B]), target, rcond=None)
        return cls(log_gamma=float(np.log(var / rho)), coef=np.asarray(coef[1:], dtype=float),
                   basis=basis, half_width=A)

    # ------------------------------------------------------------------ densities

    def design(self, grid: np.ndarray) -> np.ndarray:
        return design_matrix(self.basis, self.n_basis, self.half_width, grid)

    def log_tilt(self, grid: np.ndarray) -> np.ndarray:
        return self.design(grid) @ self.coef + soft_wall(self.half_width, grid)

    def _log_step(self, grid: np.ndarray, grid_out: np.ndarray) -> np.ndarray:
        g = self.gamma
        d = grid_out[:, None] - grid[None, :]
        return -0.5 * d**2 / g - 0.5 * (_LOG_2PI + self.log_gamma)

    def _raw(self, grid, grid_out, log_gamma=None, coef=None):
        lg = self.log_gamma if log_gamma is None else log_gamma
        c = self.coef if coef is None else coef
        g = np.exp(lg)
        d = grid_out[:, None] - grid[None, :]
        tilt = self.design(grid_out) @ c + soft_wall(self.half_width, grid_out)
        return -0.5 * d**2 / g - 0.5 * (_LOG_2PI + lg) + tilt[:, None]

    def log_transition_matrix(self, grid: np.ndarray, grid_out: np.ndarray | None = None) -> np.ndarray:
        """Column-normalised under the out-grid's trapezoid weights (see module docstring)."""
        grid = np.asarray(grid, dtype=float)
        out = grid if grid_out is None else np.asarray(grid_out, dtype=float)
        raw = self._raw(grid, out)
        lw = np.log(_quad_weights(out))
        return raw - logsumexp(raw + lw[:, None], axis=0)[None, :]

    def grad_log_transition_matrix(self, grid: np.ndarray) -> np.ndarray:
        """(1 + P, M, M): d log K / d(log gamma, coef). Each is the raw derivative minus its
        expectation under the column distribution, the exponential-family form."""
        grid = np.asarray(grid, dtype=float)
        log_k = self.log_transition_matrix(grid)
        p = np.exp(log_k + np.log(_quad_weights(grid))[:, None])  # column distributions
        d = grid[:, None] - grid[None, :]
        dg = 0.5 * d**2 / self.gamma - 0.5                        # d log step / d log gamma
        B = self.design(grid)                                      # (M, P)
        parts = [dg - (p * dg).sum(axis=0)[None, :]]
        for m in range(self.n_basis):
            bm = B[:, m][:, None] * np.ones_like(d)
            parts.append(bm - (p * bm).sum(axis=0)[None, :])
        return np.stack(parts)

    # ------------------------------------------------------------------ M-step

    def m_step(self, stats: ExpectedStatistics, grid: np.ndarray,
               grid_out: np.ndarray | None = None, max_iter: int = 300) -> "TiltedWalkKernel":
        grid = np.asarray(grid, dtype=float)
        out = grid if grid_out is None else np.asarray(grid_out, dtype=float)
        xi = np.asarray(stats.xi, dtype=float)
        n_edges = max(float(xi.sum()), 1.0)
        col_mass = xi.sum(axis=0)                                  # (M_in,)
        lw = np.log(_quad_weights(out))
        B = self.design(out)                                       # (M_out, P)
        wall = soft_wall(self.half_width, out)
        d2 = (out[:, None] - grid[None, :]) ** 2
        P = self.n_basis

        def unpack(z):
            return float(z[0]), np.asarray(z[1:], dtype=float)

        def q_and_grad(z):
            lg, c = unpack(z)
            raw = -0.5 * d2 * np.exp(-lg) - 0.5 * (_LOG_2PI + lg) + (B @ c + wall)[:, None]
            logz = logsumexp(raw + lw[:, None], axis=0)            # (M_in,)
            log_k = raw - logz[None, :]
            q = float((xi * log_k).sum()) / n_edges
            p = np.exp(log_k + lw[:, None])
            xt = xi - col_mass[None, :] * p                        # centred statistic
            g_coef = B.T @ xt.sum(axis=1)
            g_lg = float((xt * (0.5 * d2 * np.exp(-lg) - 0.5)).sum())
            grad = np.concatenate([[g_lg], g_coef]) / n_edges
            return -q, -grad

        z0 = self.theta
        q0 = -q_and_grad(z0)[0]
        bounds = [_LOG_GAMMA_BOUNDS] + [(-_COEF_BOUND, _COEF_BOUND)] * P
        res = minimize(q_and_grad, z0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options=dict(maxiter=max_iter, ftol=1e-15, gtol=1e-10))
        q1 = -float(res.fun)
        gain = q1 - q0
        if not np.isfinite(q1) or gain < -_Q_DECREASE_TOL:
            # Not an ascent: keep the receiver's parameters in a NEW object (the contract),
            # and say so through the record rather than silently.
            return replace(self, inner_iters=int(res.nit), inner_converged=False, inner_q_gain=0.0)
        lg, c = unpack(res.x)
        return TiltedWalkKernel(log_gamma=lg, coef=c, basis=self.basis, half_width=self.half_width,
                                inner_iters=int(res.nit), inner_converged=bool(res.success),
                                inner_q_gain=float(gain))

    # ------------------------------------------------------------------ diagnostics

    def tilt_diagnostics(self, grid: np.ndarray) -> dict:
        """Whether the fitted step is resolved by the grid, and how far the tilt moved."""
        h = float(grid[1] - grid[0])
        sd = float(np.sqrt(self.gamma))
        return {"n_basis": self.n_basis, "basis": self.basis, "gamma": self.gamma,
                "step_sd_over_h": sd / h, "resolved": bool(sd >= 2.0 * h),
                "tilt_range": float(np.ptp(self.log_tilt(grid)))}
