"""A transition kernel that is a FREE Gaussian mixture on the pair: the kernel of E60.

C free 2-D Gaussians on (a, a') with weights w_c, means (m_c1, m_c2) and covariances S_c, read as a conditional:

    K(a' | a) = p(a, a') / p(a) = sum_c pi_c(a) N(a'; m_c2 + (l21_c / l11_c)(a - m_c1), l22_c^2),
    pi_c(a) proportional to w_c N(a; m_c1, l11_c^2),

with S_c = L_c L_c^T, L_c = [[l11, 0], [l21, l22]] (log l11, l21, log l22 are the parameters). Each component is a line
through its own centre with its own slope and its own spread, switched on by a Gaussian gate in a. Nothing ties the
components together: no shared autoregression (the shift mixture's a' = rho a + eps, `kernels.MixtureInnovationKernel`, is
the special case of one gate centre and one gate width for every component, slope rho), no symmetry, no stationarity.
6 C parameters. The columns are normalised over a' with the trapezoid weights BP uses, so every column is a density of
the discretised model.

Learning: generalised EM with E55's M-step (`energy_kernel.NeuralEnergyKernel`): n_inner Adam steps on
Q(theta) = sum_kj xi[k, j] log K[k, j] / n_edges, Q then evaluated exactly, the step redone from the same start at half the
rate when Q did not increase (up to max_halvings times), a first-attempt success growing the rate by 25% back to its
initial value. Retries start Adam from a fresh state and an M-step that rejects every attempt drops the carried state: the
rule E58's frozen clean fits lacked (23 of 576 froze on a stale Adam state; `NeuralEnergyKernel.fresh_on_reject`).
log l11 and log l22 are projected onto [LOG_SD_MIN, LOG_SD_MAX] after every Adam step (a standard deviation between
0.0025 and 20 on a grid of spacing 0.04 and half-width 8).

Every M-step returns a NEW kernel and leaves its input untouched (amendment N-25).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np
import torch
from scipy.stats import norm

from markov_score.tilted_kernel import _quad_weights

LOG_SD_MIN, LOG_SD_MAX = -6.0, 3.0
_LOG_2PI = float(np.log(2.0 * np.pi))


def free_log_k(logit: torch.Tensor, mu: torch.Tensor, chol: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """(N_g, N_g) continuous log K[k, j] = log K(u_k | u_j); rows a' (out), columns a (in)."""
    l11, l21, l22 = chol[:, 0].exp(), chol[:, 1], chol[:, 2].exp()
    a, ap = grid[None, None, :], grid[None, :, None]
    log_w = logit - torch.logsumexp(logit, 0)
    s11 = l11 ** 2
    gate = (log_w[:, None, None] - 0.5 * (a - mu[:, 0, None, None]) ** 2 / s11[:, None, None]
            - 0.5 * (_LOG_2PI + s11.log())[:, None, None])
    mean = mu[:, 1, None, None] + (l21 / l11)[:, None, None] * (a - mu[:, 0, None, None])
    v = l22 ** 2
    comp = -0.5 * (ap - mean) ** 2 / v[:, None, None] - 0.5 * (_LOG_2PI + v.log())[:, None, None]
    return torch.logsumexp(gate + comp, 0) - torch.logsumexp(gate, 0)


@dataclass(frozen=True)
class FreeMixtureKernel:
    logit: np.ndarray            # (C,)   log weights, up to a constant
    mu: np.ndarray               # (C, 2) component means of (a, a')
    chol: np.ndarray             # (C, 3) log l11, l21, log l22
    lr: float = 1e-2
    lr_max: float = 1e-2
    n_inner: int = 10
    max_halvings: int = 6
    opt_state: dict | None = None

    @classmethod
    def init(cls, components: int, rho: float = 0.3, var: float = 0.8, lr: float = 1e-2,
             n_inner: int = 10) -> FreeMixtureKernel:
        """Every component is the AR(1) start of the other EM arms, N(a'; rho a, var); the gates sit at the C quantiles of
        that chain's stationary law with its stationary variance, so the first kernel IS that AR(1) and only the gates
        differ (which breaks the symmetry between components)."""
        s2 = var / (1.0 - rho ** 2)
        centres = norm.ppf((np.arange(components) + 0.5) / components) * np.sqrt(s2)
        l11 = np.sqrt(s2)
        chol = np.tile([np.log(l11), rho * l11, 0.5 * np.log(var)], (components, 1))
        return cls(logit=np.zeros(components), mu=np.stack([centres, rho * centres], 1), chol=chol,
                   lr=lr, lr_max=lr, n_inner=n_inner)

    # ------------------------------------------------------------------ the kernel

    @property
    def components(self) -> int:
        return len(self.logit)

    @property
    def name(self) -> str:
        return f"free_mixture_C{self.components}"

    @property
    def theta(self) -> np.ndarray:
        return np.concatenate([self.logit, self.mu.ravel(), self.chol.ravel()])

    @property
    def n_params(self) -> int:
        return 6 * self.components

    @property
    def slopes(self) -> np.ndarray:
        return self.chol[:, 1] / np.exp(self.chol[:, 0])

    def _tensors(self, requires_grad: bool = False) -> list[torch.Tensor]:
        return [torch.tensor(np.asarray(x, dtype=np.float64), requires_grad=requires_grad)
                for x in (self.logit, self.mu, self.chol)]

    @staticmethod
    def _log_k(params, grid: np.ndarray) -> torch.Tensor:
        """The grid-normalised log K (columns sum to one with the trapezoid weights)."""
        g = torch.as_tensor(grid, dtype=torch.float64)
        lk = free_log_k(*params, g)
        log_w = torch.log(torch.as_tensor(_quad_weights(grid), dtype=torch.float64))
        return lk - torch.logsumexp(lk + log_w[:, None], dim=0, keepdim=True)

    def log_transition_matrix(self, grid: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self._log_k(self._tensors(), grid).numpy()

    # ------------------------------------------------------------------ generalised M-step

    def m_step(self, stats, grid: np.ndarray) -> FreeMixtureKernel:
        xi = torch.as_tensor(stats.xi, dtype=torch.float64)
        scale = 1.0 / max(int(stats.n_edges), 1)

        def q(params) -> torch.Tensor:
            return (xi * self._log_k(params, grid)).sum() * scale

        with torch.no_grad():
            q0 = float(q(self._tensors()))
        lr = self.lr
        for attempt in range(self.max_halvings + 1):
            params = self._tensors(requires_grad=True)
            opt = torch.optim.Adam(params, lr=lr)
            if self.opt_state is not None and attempt == 0:
                opt.load_state_dict(copy.deepcopy(self.opt_state))
                for g in opt.param_groups:
                    g["lr"] = lr
            for _ in range(self.n_inner):
                opt.zero_grad(set_to_none=True)
                (-q(params)).backward()
                opt.step()
                with torch.no_grad():
                    params[2][:, 0].clamp_(LOG_SD_MIN, LOG_SD_MAX)
                    params[2][:, 2].clamp_(LOG_SD_MIN, LOG_SD_MAX)
            with torch.no_grad():
                q1 = float(q(params))
            if np.isfinite(q1) and q1 >= q0:
                new_lr = min(lr * 1.25, self.lr_max) if attempt == 0 else lr
                logit, mu, chol = (p.detach().numpy().copy() for p in params)
                return replace(self, logit=logit, mu=mu, chol=chol, lr=new_lr, opt_state=_copy_state(opt.state_dict()))
            lr *= 0.5
        return replace(self, lr=lr, opt_state=None)


def _copy_state(obj):
    """A deep copy of an optimizer state dict (tensors detached and cloned)."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: _copy_state(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_copy_state(v) for v in obj]
    return copy.deepcopy(obj)
