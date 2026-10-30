"""Shift kernels with a basis-expanded log density: the four kernel families of E56.

    K(a' | a) = phi_theta(a' - alpha a) / Z(a),   log phi_theta(eps) = -lambda eps^2 / 2 + sum_{m=1}^{23} c_m b_m(clip(eps, -W, W)),

the class the Laplace AR(1) chain lives in (every K(a'|a) = phi(a' - alpha a)), with the increment law phi written in one of
four bases at the mixture kernel's parameter count: theta = (alpha, log lambda, c_1 .. c_23), 25 numbers.

    spline    23 cubic B-splines, clamped equally spaced knots on [-W, W] (tilted_kernel.design_matrix("bspline", 23, W))
    rbf       exp(-(eps - mu_m)^2 / 2 ell^2), 23 centres equally spaced on [-W, W], ell their spacing
    fourier   cos(k pi eps / W), k = 1..12, and sin(k pi eps / W), k = 1..11
    legendre  P_k(eps / W), k = 1..23 (tilted_kernel.design_matrix("legendre", 23, W))

Splines and RBFs are LOCAL bases and can bend sharply at one point (the Laplace kink); Fourier and Legendre are GLOBAL and
can only approximate a kink by oscillating (Jerome's "truncated Fourier series" and "polynomial", 29 Oct 2026). Beyond W
every basis is held at its value at +-W by the clip, and the quadratic envelope, whose curvature lambda > 0 is learned,
carries the tails. Z(a) is the grid's trapezoid sum, so every column is a density of the discretised model, the model whose
likelihood EM maximises (as in energy_kernel).

The start alpha = 0.3, lambda = 1 / 0.8, c = 0 is the Gaussian AR(1) kernel N(a'; 0.3 a, 0.8), E30's em_bp start.

Learning: generalised EM, as energy_kernel. The M-step raises Q(theta) = sum_kj xi[k, j] log K_theta[k, j] / n_edges by
`n_inner` Adam steps; the basis is evaluated in torch so autograd reaches alpha through eps = a' - alpha a. If Q did not
increase the step is redone at half the rate, up to `max_halvings` times; if no attempt increases Q the parameters are kept
and `kept_steps` counts it. Q therefore never decreases. Every M-step returns a NEW kernel.
"""

from __future__ import annotations

import copy
import functools
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from markov_score.tilted_kernel import _quad_weights

BASES = ("spline", "rbf", "fourier", "legendre")
N_COEF = 23


# ---------------------------------------------------------------------------------------- bases

def _bspline(x: torch.Tensor, knots: torch.Tensor, degree: int = 3) -> torch.Tensor:
    """(len(x), len(knots) - degree - 1) B-spline values by the Cox-de Boor recursion, differentiable in x."""
    t = knots
    B = ((x[:, None] >= t[None, :-1]) & (x[:, None] < t[None, 1:])).to(x.dtype)
    for p in range(1, degree + 1):
        dl = t[p:-1] - t[:-p - 1]
        dr = t[p + 1:] - t[1:-p]
        left = (x[:, None] - t[None, :-p - 1]) / torch.where(dl > 0, dl, torch.ones_like(dl)) * (dl > 0)
        right = (t[None, p + 1:] - x[:, None]) / torch.where(dr > 0, dr, torch.ones_like(dr)) * (dr > 0)
        B = left * B[:, :-1] + right * B[:, 1:]
    return B


@functools.lru_cache(maxsize=8)
def _spline_table(half_width: float) -> tuple[np.ndarray, np.ndarray, float]:
    """The spline basis as piecewise cubics: (left knots (21,), table (21, 4, 23), knot spacing).

    table[i, d, m] is the coefficient of s^d, s = eps - knot_i, of basis function m on interval i. Built by solving the
    4 x 4 Vandermonde system on each interval against the Cox-de Boor values, so it IS the basis (to ~1e-13), and
    `_spline_score` can contract it with the coefficients first: one (21, 4) table per evaluation instead of a
    (N_g^2, 23) design matrix.
    """
    W = float(half_width)
    knots = np.linspace(-W, W, N_COEF - 1)
    h = float(knots[1] - knots[0])
    s = np.array([0.1, 0.4, 0.7, 0.95]) * h
    V = np.vander(s, 4, increasing=True)
    table = np.empty((len(knots) - 1, 4, N_COEF))
    for i, x0 in enumerate(knots[:-1]):
        B = basis_matrix("spline", torch.as_tensor(x0 + s), W).numpy()
        table[i] = np.linalg.solve(V, B)
    return knots[:-1], table, h


def _spline_score(eps: torch.Tensor, coef: torch.Tensor, half_width: float) -> torch.Tensor:
    """sum_m coef_m b_m(clip(eps, -W, W)) for the spline basis, exactly, without the design matrix."""
    left, table, h = _spline_table(float(half_width))
    W = float(half_width)
    z = eps.reshape(-1).to(torch.float64).clamp(-W, W * (1 - 1e-12))
    idx = torch.clamp(torch.floor((z + W) / h).long(), 0, len(left) - 1)
    T = torch.as_tensor(table, dtype=torch.float64, device=z.device) @ coef        # (21, 4)
    s = z - torch.as_tensor(left, dtype=torch.float64, device=z.device)[idx]
    c = T[idx]
    return c[:, 0] + s * (c[:, 1] + s * (c[:, 2] + s * c[:, 3]))


def basis_matrix(basis: str, eps: torch.Tensor, half_width: float) -> torch.Tensor:
    """(len(eps), 23) values of the named basis at clip(eps, -W, W), as a float64 torch tensor."""
    W = float(half_width)
    eps = eps.reshape(-1).to(torch.float64)
    if basis == "spline":
        k = 3
        inner = torch.linspace(-W, W, N_COEF + 5 - 2 * k, dtype=torch.float64, device=eps.device)
        knots = torch.cat([torch.full((k,), -W, dtype=torch.float64, device=eps.device), inner,
                           torch.full((k,), W, dtype=torch.float64, device=eps.device)])
        z = eps.clamp(-W, W * (1 - 1e-12))
        return _bspline(z, knots, k)[:, 1:]     # the splines sum to one: drop one, a constant is absorbed by Z
    z = eps.clamp(-W, W)
    if basis == "rbf":
        mu = torch.linspace(-W, W, N_COEF, dtype=torch.float64, device=eps.device)
        ell = float(2 * W / (N_COEF - 1))
        return torch.exp(-0.5 * ((z[:, None] - mu[None, :]) / ell) ** 2)
    if basis == "fourier":
        kc = torch.arange(1, 13, dtype=torch.float64, device=eps.device)
        ks = torch.arange(1, 12, dtype=torch.float64, device=eps.device)
        return torch.cat([torch.cos(np.pi * z[:, None] * kc / W), torch.sin(np.pi * z[:, None] * ks / W)], dim=1)
    if basis == "legendre":
        u = z / W
        P = [torch.ones_like(u), u]
        for n in range(1, N_COEF):
            P.append(((2 * n + 1) * u * P[n] - n * P[n - 1]) / (n + 1))
        return torch.stack(P[1:], dim=1)
    raise ValueError(f"unknown basis {basis!r}; one of {BASES}")


# ---------------------------------------------------------------------------------------- the kernel

@dataclass(frozen=True)
class ShiftKernel:
    basis: str
    params: np.ndarray = field(repr=False)     # (alpha, log lambda, c_1 .. c_23)
    half_width_basis: float = 4.0
    lr: float = 1e-2
    lr_max: float = 1e-2
    n_inner: int = 10
    max_halvings: int = 6
    kept_steps: int = 0
    opt_state: dict | None = field(default=None, repr=False)

    @classmethod
    def init(cls, basis: str, rho: float = 0.3, var: float = 0.8, half_width_basis: float = 4.0, lr: float = 1e-2,
             n_inner: int = 10) -> ShiftKernel:
        if basis not in BASES:
            raise ValueError(f"unknown basis {basis!r}; one of {BASES}")
        params = np.concatenate([[rho, np.log(1.0 / var)], np.zeros(N_COEF)])
        return cls(basis=basis, params=params, half_width_basis=float(half_width_basis), lr=lr, lr_max=lr,
                   n_inner=n_inner)

    @classmethod
    def from_theta(cls, basis: str, theta, half_width_basis: float = 4.0) -> ShiftKernel:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.size != 2 + N_COEF:
            raise ValueError(f"theta has {theta.size} entries; a shift kernel has {2 + N_COEF}")
        return cls(basis=basis, params=theta.copy(), half_width_basis=float(half_width_basis))

    @property
    def name(self) -> str:
        return f"shift_{self.basis}"

    @property
    def theta(self) -> np.ndarray:
        return self.params.copy()

    @property
    def n_params(self) -> int:
        return int(self.params.size)

    @property
    def alpha(self) -> float:
        return float(self.params[0])

    def _score(self, p: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """log phi up to its normaliser, at every entry of eps (any shape)."""
        if self.basis == "spline":
            tilt = _spline_score(eps, p[2:], self.half_width_basis)
        else:
            tilt = basis_matrix(self.basis, eps, self.half_width_basis) @ p[2:]
        return (-0.5 * torch.exp(p[1]) * eps.reshape(-1) ** 2 + tilt).reshape(eps.shape)

    def _log_k(self, p: torch.Tensor, grid: np.ndarray) -> torch.Tensor:
        """(N_g, N_g) log K[k, j] = log K(u_k | u_j), normalised over k with the trapezoid weights."""
        u = torch.as_tensor(grid, dtype=torch.float64)
        s = self._score(p, u[:, None] - p[0] * u[None, :])
        log_w = torch.log(torch.as_tensor(_quad_weights(grid), dtype=torch.float64))
        return s - torch.logsumexp(s + log_w[:, None], dim=0, keepdim=True)

    def log_transition_matrix(self, grid: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self._log_k(torch.as_tensor(self.params), grid).numpy()

    def log_phi(self, eps: np.ndarray, half_width: float = 8.0, n_quad: int = 4001) -> np.ndarray:
        """The learned increment law log phi(eps), normalised as a density on [-half_width, half_width]."""
        p = torch.as_tensor(self.params)
        q = np.linspace(-half_width, half_width, n_quad)
        with torch.no_grad():
            sq = self._score(p, torch.as_tensor(q)).numpy()
            se = self._score(p, torch.as_tensor(np.asarray(eps, dtype=np.float64))).numpy()
        m = sq.max()
        return se - (m + np.log(np.exp(sq - m) @ _quad_weights(q)))

    # ------------------------------------------------------------------ generalised M-step

    def m_step(self, stats, grid: np.ndarray) -> ShiftKernel:
        xi = torch.as_tensor(stats.xi, dtype=torch.float64)
        scale = 1.0 / max(int(stats.n_edges), 1)
        p = torch.tensor(self.params, dtype=torch.float64, requires_grad=True)

        def q() -> torch.Tensor:
            return (xi * self._log_k(p, grid)).sum() * scale

        with torch.no_grad():
            q0 = float(q())
        lr = self.lr
        for attempt in range(self.max_halvings + 1):
            with torch.no_grad():
                p.copy_(torch.as_tensor(self.params))
            opt = torch.optim.Adam([p], lr=lr)
            if self.opt_state is not None:
                opt.load_state_dict(copy.deepcopy(self.opt_state))
                for g in opt.param_groups:
                    g["lr"] = lr
            for _ in range(self.n_inner):
                opt.zero_grad(set_to_none=True)
                (-q()).backward()
                opt.step()
            with torch.no_grad():
                q1 = float(q())
            if np.isfinite(q1) and q1 >= q0:
                new_lr = min(lr * 1.25, self.lr_max) if attempt == 0 else lr
                return replace(self, params=p.detach().numpy().copy(), lr=new_lr,
                               opt_state=copy.deepcopy(opt.state_dict()))
            lr *= 0.5
        return replace(self, lr=lr, kept_steps=self.kept_steps + 1)
