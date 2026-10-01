"""The pair-energy kernel: Jerome's W_theta learned as a transition kernel, in the physicist's
parametrisation.

What it is
----------
A reversible chain whose pair energy has the gradient form

    W(a, a') = (a' - a)^2 / 2 gamma + f(a) + f(a'),

a Gaussian bond and a site energy f -- exactly the class the discretised gradient SPDE lives in
(f = (h / 2 Theta) V(s a) for the phi^4 field, a quartic). The transition kernel is the
ground-state transform of the symmetric transfer operator T_f(a, a') = e^{-f(a)} N(a'; a, gamma)
e^{-f(a')}:

    K_f(a' | a) = T_f(a, a') psi_f(a') / (lambda_f psi_f(a)),       pi_f = psi_f^2,

psi_f its Perron eigenfunction. Equivalently (field.py's identity) K_f is the tilted walk with
arrival tilt g_f = psi_f e^{-f}; so this kernel and `TiltedWalkKernel` are the SAME model class
(docs/FIELD_MODEL.md, the reversibility proposition: every tilted walk is reversible with
pi = g Z and site energy f = -(1/2) log(g / Z)). They differ in what is parametrised:
the tilt log g = log psi_f - f is a transcendental function of the potential, so it needs a
flexible basis (7 or 25 parameters); the site energy IS the potential, a polynomial of degree
4 for the field, so this kernel is exactly specified with 1 + 4 parameters, and its parameters
read directly as physics: gamma = Theta h / s^2 gives the temperature, f gives V.

The price is one dense eigensolve of the (M x M) transfer operator per evaluation (15 ms on the
401-point grid), and an M-step objective that is no longer concave in the coefficients. The
gradient is analytic through first-order perturbation theory of the Perron eigenvector,

    d psi_0 / d theta = sum_{n >= 1} psi_n <psi_n | dT/dtheta | psi_0> / (lambda_0 - lambda_n),

which is exact for the discretised operator because eigh returns the complete basis. For a
site-energy coefficient, <psi_n | dT/dc_m | psi_0> = -(lambda_0 + lambda_n) <psi_n | B_m psi_0>
(T is symmetric and B_m acts on both ends of the bond); for log gamma it is the full quadratic
form of T (dlog N / dlog gamma) between psi_n and psi_0. The gradient of Q = <Xi, log K> then has
the same centred form as the tilted kernel's, with d log g in place of the basis.

Parameters: log gamma, and f(a) = sum_{m=1}^{d} c_m (a / 2)^m. Monomials in a / 2 keep the
field's coefficients O(1) on the (401, 8) grid (y^4 <= 256 there), so the optimiser's bounds
(+-200) are never active for a physical fit. The constant is absorbed by the eigen-normalisation.
The site energy carries the same fixed soft wall as the tilted kernel, as an ENERGY
+(a / 5)^12, on both ends of the bond.

Contract: `m_step` returns a NEW kernel and never mutates the receiver
(tests/test_kernel_mstep_contract.py); a step that does not increase Q is rejected and recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import logsumexp

from markov_score.em import ExpectedStatistics
from markov_score.tilted_kernel import _LOG_2PI, _LOG_GAMMA_BOUNDS, _quad_weights, soft_wall

BASIS_SCALE = 2.0
_COEF_BOUND = 200.0
_RESOLVED_REL = 1e-13
_Q_DECREASE_TOL = 1e-10


def site_design(degree: int, x: np.ndarray) -> np.ndarray:
    """(len(x), degree) monomials (x / BASIS_SCALE)^m, m = 1..degree; no constant."""
    y = np.asarray(x, dtype=float) / BASIS_SCALE
    return np.stack([y**m for m in range(1, degree + 1)], axis=1)


@dataclass(frozen=True)
class PairEnergyKernel:
    """K_f(a'|a) on the grid for W = (a'-a)^2 / 2 gamma + f(a) + f(a'). theta = (log gamma, coef)."""

    log_gamma: float
    coef: np.ndarray
    half_width: float = 8.0

    inner_iters: int = 0
    inner_converged: bool = False
    inner_q_gain: float = float("nan")

    @property
    def name(self) -> str:
        return f"pair_energy{self.degree}"

    @property
    def degree(self) -> int:
        return int(len(self.coef))

    @property
    def gamma(self) -> float:
        return float(np.exp(self.log_gamma))

    @property
    def theta(self) -> np.ndarray:
        return np.concatenate([[self.log_gamma], np.asarray(self.coef, dtype=float)])

    @classmethod
    def init(cls, degree: int, rho: float, var: float, half_width: float = 8.0) -> "PairEnergyKernel":
        """The Gaussian AR(1) member with (rho, var): gamma = var / rho and f(a) = (1 - rho)^2 a^2 / 4 var
        (the site energy of the quadratic tilt -(1 - rho) a^2 / 2 var, docs/FIELD_MODEL.md)."""
        if not 0.0 < rho < 1.0:
            raise ValueError("the Gaussian AR(1) embedding needs 0 < rho < 1")
        if degree < 2:
            raise ValueError("the site energy needs degree >= 2 to contain the Gaussian chain")
        coef = np.zeros(degree)
        coef[1] = (1.0 - rho) ** 2 * BASIS_SCALE**2 / (4.0 * var)
        return cls(log_gamma=float(np.log(var / rho)), coef=coef, half_width=float(half_width))

    @classmethod
    def from_field(cls, T: float, h: float, scale: float, half_width: float = 8.0) -> "PairEnergyKernel":
        """The phi^4 chain's own parameters in standardised units: gamma = T h / s^2 and
        f(a) = (h / 2T) (s^2 a^2 - 1)^2 / 4, i.e. c_2 = -h s^2 (2/s... written out below."""
        s = float(scale)
        pref = h / (8.0 * T)                                   # (h / 2T) * (1/4)
        coef = np.zeros(4)
        coef[1] = pref * (-2.0 * s**2) * BASIS_SCALE**2        # a^2 term: -2 s^2 a^2
        coef[3] = pref * s**4 * BASIS_SCALE**4                 # a^4 term: s^4 a^4
        return cls(log_gamma=float(np.log(T * h / s**2)), coef=coef, half_width=float(half_width))

    # ------------------------------------------------------------------ energies and spectrum

    def design(self, grid: np.ndarray) -> np.ndarray:
        return site_design(self.degree, grid)

    def site_energy(self, grid: np.ndarray) -> np.ndarray:
        """f(a) on the grid, wall included (a positive energy at the grid edge)."""
        return self.design(grid) @ self.coef - soft_wall(self.half_width, grid)

    def _spectrum(self, grid: np.ndarray, log_gamma: float | None = None, coef: np.ndarray | None = None):
        """Full eigen-decomposition of the quadrature-discretised T_f: (lam desc, psi, w, f, logN).
        psi is quadrature-normalised (sum_k w_k psi_nk^2 = 1), psi_0 positive."""
        lg = self.log_gamma if log_gamma is None else float(log_gamma)
        c = self.coef if coef is None else np.asarray(coef, dtype=float)
        u = np.asarray(grid, dtype=float)
        w = _quad_weights(u)
        f = site_design(len(c), u) @ c - soft_wall(self.half_width, u)
        logN = -0.5 * (u[:, None] - u[None, :]) ** 2 * np.exp(-lg) - 0.5 * (_LOG_2PI + lg)
        logT = logN - f[:, None] - f[None, :]
        shift = float(logT.max())
        sw = np.sqrt(w)
        A = np.exp(logT - shift) * sw[:, None] * sw[None, :]
        lam, v = eigh(A)
        order = np.argsort(lam)[::-1]
        lam, v = lam[order], v[:, order]
        psi = v / sw[:, None]
        if psi[np.argmax(np.abs(psi[:, 0])), 0] < 0:
            psi[:, 0] *= -1.0
        return lam, psi, w, f, logN

    @staticmethod
    def _log_psi0(psi0: np.ndarray) -> np.ndarray:
        """log psi_0, floored at 1e-290 of its maximum where the eigenvector is not resolved
        (eigensolver noise of either sign there). Finite everywhere, so <Xi, log K> never meets
        0 x (-inf); the floored states carry no mass in any posterior (e^-668 relative)."""
        top = float(psi0.max())
        return np.log(np.maximum(psi0, 1e-290 * top))

    def log_tilt(self, grid: np.ndarray) -> np.ndarray:
        """log g_f = log psi_f - f on the grid; -inf where psi_f is unresolved."""
        lam, psi, w, f, _ = self._spectrum(grid)
        return self._log_psi0(psi[:, 0]) - f

    def log_stationary(self, grid: np.ndarray) -> np.ndarray:
        """log pi_f = 2 log psi_f, normalised under the grid's trapezoid weights."""
        lam, psi, w, f, _ = self._spectrum(grid)
        lp = 2.0 * self._log_psi0(psi[:, 0])
        return lp - logsumexp(lp + np.log(w))

    def _raw(self, grid: np.ndarray, grid_out: np.ndarray | None = None) -> np.ndarray:
        lam, psi, w, f, logN = self._spectrum(grid)
        lt = self._log_psi0(psi[:, 0]) - f
        u = np.asarray(grid, dtype=float)
        if grid_out is None or (len(grid_out) == len(u) and np.allclose(grid_out, u)):
            return logN + lt[:, None]
        out = np.asarray(grid_out, dtype=float)
        fin = np.isfinite(lt)
        lt_out = CubicSpline(u[fin], lt[fin], extrapolate=False)(out)
        lt_out = np.where(np.isfinite(lt_out), lt_out, -np.inf)
        logN_out = -0.5 * (out[:, None] - u[None, :]) ** 2 / self.gamma - 0.5 * (_LOG_2PI + self.log_gamma)
        return logN_out + lt_out[:, None]

    def log_transition_matrix(self, grid: np.ndarray, grid_out: np.ndarray | None = None) -> np.ndarray:
        """Column-normalised under the out-grid's trapezoid weights, as every grid kernel is."""
        grid = np.asarray(grid, dtype=float)
        out = grid if grid_out is None else np.asarray(grid_out, dtype=float)
        raw = self._raw(grid, None if grid_out is None else out)
        lw = np.log(_quad_weights(out))
        return raw - logsumexp(raw + lw[:, None], axis=0)[None, :]

    # ------------------------------------------------------------------ derivatives

    def _dlog_tilt(self, grid: np.ndarray, lam, psi, w, f, logN):
        """(d log g / d log gamma, d log g / d coef): (M,) and (M, P), by first-order perturbation
        theory of the Perron eigenvector in the complete discretised eigenbasis."""
        u = np.asarray(grid, dtype=float)
        psi0 = psi[:, 0]
        mask = psi0 > _RESOLVED_REL * psi0.max()
        B = self.design(u)                                                   # (M, P)
        lam0 = lam[0]
        denom = lam0 - lam[1:]                                               # (M-1,) > 0 by Perron-Frobenius
        # site-energy coefficients: <psi_n | dT/dc_m | psi_0> = -(lam_0 + lam_n) <psi_n | B_m psi_0>
        D = psi[:, 1:].T @ (w[:, None] * B * psi0[:, None])                  # (M-1, P)
        coef_c = -(lam0 + lam[1:])[:, None] * D / denom[:, None]
        dpsi_c = psi[:, 1:] @ coef_c                                         # (M, P)
        # log gamma: <psi_n | T o S | psi_0>, S = dlog N / dlog gamma, through the scaled operator
        S = 0.5 * (u[:, None] - u[None, :]) ** 2 * np.exp(-self.log_gamma) - 0.5
        logT = logN - f[:, None] - f[None, :]
        Tsc = np.exp(logT - logT.max())
        v = (Tsc * S) @ (w * psi0)                                           # (M,)
        b = psi[:, 1:].T @ (w * v)                                           # (M-1,)
        # lam is the spectrum of the SAME scaled operator (A = W^1/2 T_scaled W^1/2), so the
        # ratio <psi_n|dT|psi_0> / (lam_0 - lam_n) is consistent without undoing the shift
        dpsi_g = psi[:, 1:] @ (b / denom)                                    # (M,)
        with np.errstate(divide="ignore", invalid="ignore"):
            dlp_c = np.where(mask[:, None], dpsi_c / psi0[:, None], 0.0)
            dlp_g = np.where(mask, dpsi_g / psi0, 0.0)
        return dlp_g, dlp_c - B

    def grad_log_transition_matrix(self, grid: np.ndarray) -> np.ndarray:
        """(1 + P, M, M): d log K / d(log gamma, coef), each the raw derivative minus its
        expectation under the column distribution."""
        grid = np.asarray(grid, dtype=float)
        lam, psi, w, f, logN = self._spectrum(grid)
        dlg_g, dlg_c = self._dlog_tilt(grid, lam, psi, w, f, logN)
        log_k = self.log_transition_matrix(grid)
        p = np.exp(log_k + np.log(w)[:, None])
        S = 0.5 * (grid[:, None] - grid[None, :]) ** 2 / self.gamma - 0.5
        raw_g = S + dlg_g[:, None]
        parts = [raw_g - (p * raw_g).sum(axis=0)[None, :]]
        for m in range(self.degree):
            bm = dlg_c[:, m][:, None] * np.ones_like(S)
            parts.append(bm - (p * bm).sum(axis=0)[None, :])
        return np.stack(parts)

    # ------------------------------------------------------------------ M-step

    def m_step(self, stats: ExpectedStatistics, grid: np.ndarray,
               grid_out: np.ndarray | None = None, max_iter: int = 300) -> "PairEnergyKernel":
        grid = np.asarray(grid, dtype=float)
        if grid_out is not None and not (len(grid_out) == len(grid) and np.allclose(grid_out, grid)):
            raise ValueError("PairEnergyKernel.m_step fits on one grid (the eigensolve lives there)")
        xi = np.asarray(stats.xi, dtype=float)
        n_edges = max(float(xi.sum()), 1.0)
        col_mass = xi.sum(axis=0)
        row_xi = None  # filled per evaluation
        w = _quad_weights(grid)
        lw = np.log(w)
        P = self.degree

        def unpack(z):
            return float(z[0]), np.asarray(z[1:], dtype=float)

        def q_and_grad(z):
            lg, c = unpack(z)
            lam, psi, w_, f, logN = self._spectrum(grid, lg, c)
            lt = self._log_psi0(psi[:, 0]) - f
            raw = logN + lt[:, None]
            logz = logsumexp(raw + lw[:, None], axis=0)
            log_k = raw - logz[None, :]
            with np.errstate(invalid="ignore"):
                q = float(np.where(xi > 0, xi * log_k, 0.0).sum()) / n_edges
            if not np.isfinite(q):
                return np.inf, np.zeros(1 + P)
            p = np.exp(log_k + lw[:, None])
            xt = xi - col_mass[None, :] * p
            tmp = replace(self, log_gamma=lg, coef=c)
            dlg_g, dlg_c = tmp._dlog_tilt(grid, lam, psi, w_, f, logN)
            S = 0.5 * (grid[:, None] - grid[None, :]) ** 2 * np.exp(-lg) - 0.5
            row = xt.sum(axis=1)                                         # (M,) sum over sources
            g_lg = float((xt * S).sum() + row @ dlg_g)
            g_c = dlg_c.T @ row
            grad = np.concatenate([[g_lg], g_c]) / n_edges
            return -q, -grad

        z0 = self.theta
        q0 = -q_and_grad(z0)[0]
        bounds = [_LOG_GAMMA_BOUNDS] + [(-_COEF_BOUND, _COEF_BOUND)] * P
        res = minimize(q_and_grad, z0, jac=True, method="L-BFGS-B", bounds=bounds,
                       options=dict(maxiter=max_iter, ftol=1e-15, gtol=1e-10))
        q1 = -float(res.fun)
        gain = q1 - q0
        if not np.isfinite(q1) or gain < -_Q_DECREASE_TOL:
            return replace(self, inner_iters=int(res.nit), inner_converged=False, inner_q_gain=0.0)
        lg, c = unpack(res.x)
        return PairEnergyKernel(log_gamma=lg, coef=c, half_width=self.half_width,
                                inner_iters=int(res.nit), inner_converged=bool(res.success),
                                inner_q_gain=float(gain))

    # ------------------------------------------------------------------ diagnostics

    def tilt_diagnostics(self, grid: np.ndarray) -> dict:
        h = float(grid[1] - grid[0])
        sd = float(np.sqrt(self.gamma))
        lt = self.log_tilt(grid)
        fin = np.isfinite(lt)
        return {"n_basis": self.degree, "basis": "monomial", "gamma": self.gamma,
                "step_sd_over_h": sd / h, "resolved": bool(sd >= 2.0 * h),
                "tilt_range": float(np.ptp(lt[fin]))}


# ---------------------------------------------------------------------- physical readings


def potential_readings(site_energy: np.ndarray, grid: np.ndarray, gamma: float, scale: float, h: float,
                       bulk: float = 1.6) -> dict:
    """Read (Theta, V) from a fitted site energy f on the standardised grid.

    With a = phi / s and W = (phi' - phi)^2 / 2 Theta h + (h / 2 Theta)[V(phi) + V(phi')], the
    standardised chain has gamma = Theta h / s^2 and f(a) = (h / 2 Theta) V(s a) + const, so

        Theta_hat = gamma s^2 / h,      V_hat(phi) = (2 Theta_hat / h) f(phi / s) + const.

    Returns Theta_hat, the barrier height V_hat(0) - min V_hat, the well position argmin V_hat
    (|phi| <= 2), and the RMS deviation of V_hat from the true double well on |phi| <= `bulk`
    after removing the constant (phys units). The true values are 1/4, 1 and 0.
    """
    from markov_score.field import potential
    grid = np.asarray(grid, dtype=float)
    fin = np.isfinite(site_energy)
    theta_hat = float(gamma * scale**2 / h)
    phi = np.linspace(-2.0, 2.0, 801)
    f_phi = CubicSpline(grid[fin], site_energy[fin], extrapolate=False)(phi / scale)
    v_hat = (2.0 * theta_hat / h) * f_phi
    ok = np.isfinite(v_hat)
    v_true = potential(phi)
    inner = ok & (np.abs(phi) <= bulk)
    shift = float(np.mean((v_hat - v_true)[inner]))
    v_hat = v_hat - shift
    pos = ok & (phi > 0)
    i_min = np.flatnonzero(pos)[np.argmin(v_hat[pos])]
    i0 = int(np.argmin(np.abs(phi)))
    return {"theta_hat": theta_hat,
            "barrier_hat": float(v_hat[i0] - v_hat[i_min]),
            "well_hat": float(phi[i_min]),
            "potential_rms": float(np.sqrt(np.mean((v_hat - v_true)[inner] ** 2)))}
