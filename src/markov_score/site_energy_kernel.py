"""A neural energy kernel that also takes the link's position: one network for a chain whose kernel changes (E68).

    K_theta,k(a' | a) = exp(-E_theta(a, a', s_k)) / Z_theta,k(a),      s_k = 2 k / (L - 2) - 1 in [-1, 1],

for the link from site k to site k + 1 (k = 0 .. L - 2), normalised over a' with the trapezoid weights of the grid,
separately for every (k, a). It is `energy_kernel.NeuralEnergyKernel` with a third input. The site enters smoothly, so
neighbouring links share what they learn; the network can still make the kernel change quickly where the data demand it.
The last layer starts at zero, so the first kernel is uniform at every link. Inputs (a, a') are scaled by 2 / half_width
as in NeuralEnergyKernel; float64 throughout.

M-step: the generalised M-step of NeuralEnergyKernel, with `fresh_on_reject` (the E60 rule): `n_inner` Adam steps on
Q(theta) = sum_k sum_ij xi_k[i, j] log K_theta,k[i, j] / n_edges, redone at half the rate if Q did not increase (up to
`max_halvings` times), the rate growing back by 25% after a first-attempt success. Q never decreases. `stats` needs
`xi` of shape (L - 1, N_g, N_g) (per-link statistics, `site_bp.site_bp_batch(pairs=True).xi`) and `n_edges`.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn

from markov_score.tilted_kernel import _quad_weights


def _mlp(hidden: int) -> nn.Module:
    return nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                         nn.Linear(hidden, 1)).double()


def link_positions(n_links: int) -> np.ndarray:
    return np.linspace(-1.0, 1.0, n_links) if n_links > 1 else np.zeros(1)


@dataclass(frozen=True)
class SiteEnergyKernel:
    state: dict
    hidden: int
    half_width: float
    n_links: int
    lr: float = 1e-2
    lr_max: float = 1e-2
    n_inner: int = 10
    max_halvings: int = 6
    opt_state: dict | None = None

    @classmethod
    def init(cls, hidden: int, seed: int, half_width: float, n_links: int, lr: float = 1e-2,
             n_inner: int = 10) -> SiteEnergyKernel:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            net = _mlp(hidden)
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
        return cls(state={k: v.detach().clone() for k, v in net.state_dict().items()}, hidden=hidden,
                   half_width=float(half_width), n_links=int(n_links), lr=lr, lr_max=lr, n_inner=n_inner)

    @property
    def name(self) -> str:
        return f"site_energy_H{self.hidden}"

    @property
    def theta(self) -> np.ndarray:
        return np.concatenate([v.numpy().ravel() for v in self.state.values()])

    def _net(self) -> nn.Module:
        net = _mlp(self.hidden)
        net.load_state_dict(self.state)
        return net

    def _log_k(self, net: nn.Module, grid: np.ndarray) -> torch.Tensor:
        """(L - 1, N_g, N_g) log K_k[i, j] = log K_k(u_i | u_j), each column of each link a density on the grid."""
        u = torch.as_tensor(grid, dtype=torch.float64) * (2.0 / self.half_width)
        s = torch.as_tensor(link_positions(self.n_links), dtype=torch.float64)
        n, m = len(grid), self.n_links
        pairs = torch.stack([u[None, None, :].expand(m, n, n), u[None, :, None].expand(m, n, n),
                             s[:, None, None].expand(m, n, n)], dim=-1)                 # [k, i, j] = (u_j, u_i, s_k)
        logits = -net(pairs.reshape(-1, 3)).reshape(m, n, n)
        log_w = torch.log(torch.as_tensor(_quad_weights(grid), dtype=torch.float64))
        return logits - torch.logsumexp(logits + log_w[None, :, None], dim=1, keepdim=True)

    def log_transition_matrices(self, grid: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self._log_k(self._net(), grid).numpy()

    def m_step(self, stats, grid: np.ndarray) -> SiteEnergyKernel:
        xi = torch.as_tensor(np.asarray(stats.xi), dtype=torch.float64)
        if xi.shape[0] != self.n_links:
            raise ValueError(f"per-link statistics expected: xi has {xi.shape[0]} links, the kernel {self.n_links}")
        scale = 1.0 / max(int(stats.n_edges), 1)
        net = self._net()

        def q() -> torch.Tensor:
            return (xi * self._log_k(net, grid)).sum() * scale

        with torch.no_grad():
            q0 = float(q())
        lr = self.lr
        for attempt in range(self.max_halvings + 1):
            net.load_state_dict(self.state)
            opt = torch.optim.Adam(net.parameters(), lr=lr)
            if self.opt_state is not None and attempt == 0:
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
                return replace(self, state={k: v.detach().clone() for k, v in net.state_dict().items()},
                               lr=new_lr, opt_state=copy.deepcopy(opt.state_dict()))
            lr *= 0.5
        return replace(self, lr=lr, opt_state=None)
