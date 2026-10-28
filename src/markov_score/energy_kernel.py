"""A transition kernel that is a neural-network energy: the agnostic neural kernel of E55.

    K_theta(a' | a) = exp(-E_theta(a, a')) / Z_theta(a),      Z_theta(a) = sum_k w_k exp(-E_theta(a, u_k)),

with E_theta a small MLP from the pair (a, a') to one number, and the normaliser the trapezoid sum BP itself uses, so every
column is a probability density of the discretised model (the model whose likelihood EM maximises). Any positive kernel on
the grid can be written this way, so no shape is assumed. In particular nothing about the field is built in: no Gaussian
bond (the Gaussian components of `kernels.MDNKernel` were one), no site energy, no reversibility, no a -> -a symmetry.
What the kernel is told is what BP needs to be BP: the data are a first-order Markov chain with the same kernel at every
link. The last layer starts at ZERO, so the first kernel is UNIFORM on the grid and everything it learns comes from data.
It is the network version of E43's log-linear kernel (`loglinear_kernel.py`, a 2-D spline energy).

Inputs are scaled by 2 / half_width (the grid becomes [-2, 2]); the network is float64 throughout, so that Q and the EM
evidence keep the digits the monotonicity checks need.

Learning: generalised EM. The E-step (`em.e_step_multi`) gives xi[k, j], the expected number of transitions u_j -> u_k over
the whole dataset; the M-step raises Q(theta) = sum_kj xi[k, j] log K_theta[k, j] (divided by the number of transitions, so
the step size does not depend on the dataset size) by `n_inner` Adam steps. Autograd differentiates through the
log-normaliser, which gives the "observed minus expected" gradient of every exponential family. Q is then evaluated
exactly. If it did not increase, the step is redone from the same start at half the learning rate, up to `max_halvings`
times; if no attempt increases Q the parameters are kept and the halved rate is carried forward (the rule MDNKernel.m_step
adopted after E25, amendment N-24). A first-attempt success lets the rate grow by 25%, back towards its initial value.
So Q never decreases, and EM's evidence never decreases except through the site-1 term when the initial law is the
kernel's own stationary law (recorded by `fit_em` as `monotone_violation`).

Every M-step returns a NEW kernel and leaves its input untouched: `fit_em` keeps checkpoints by reference (amendment N-25,
the MDN's checkpoints were once all aliases of the final kernel).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

import numpy as np
import torch
import torch.nn as nn

from markov_score.tilted_kernel import _quad_weights


def _mlp(hidden: int) -> nn.Module:
    return nn.Sequential(nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                         nn.Linear(hidden, 1)).double()


@dataclass(frozen=True)
class NeuralEnergyKernel:
    state: dict                 # the MLP's state dict, float64, on the CPU
    hidden: int
    half_width: float
    lr: float = 1e-2
    lr_max: float = 1e-2
    n_inner: int = 10
    max_halvings: int = 6
    opt_state: dict | None = None
    device: str = "cpu"

    @classmethod
    def init(cls, hidden: int, seed: int, half_width: float, lr: float = 1e-2, n_inner: int = 10,
             device: str = "cpu") -> "NeuralEnergyKernel":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            net = _mlp(hidden)
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
        return cls(state={k: v.detach().clone() for k, v in net.state_dict().items()}, hidden=hidden,
                   half_width=float(half_width), lr=lr, lr_max=lr, n_inner=n_inner, device=device)

    @classmethod
    def from_theta(cls, theta, hidden: int, half_width: float, device: str = "cpu") -> "NeuralEnergyKernel":
        """Rebuild a fitted kernel from its flat parameter vector (the `theta` column E55 writes)."""
        shapes = {k: v.shape for k, v in _mlp(hidden).state_dict().items()}
        theta = np.asarray(theta, dtype=np.float64)
        if theta.size != sum(int(np.prod(s)) for s in shapes.values()):
            raise ValueError(f"theta has {theta.size} entries; a width-{hidden} energy net has "
                             f"{sum(int(np.prod(s)) for s in shapes.values())}")
        state, pos = {}, 0
        for k, shp in shapes.items():
            n = int(np.prod(shp))
            state[k] = torch.as_tensor(theta[pos:pos + n].reshape(shp), dtype=torch.float64)
            pos += n
        return cls(state=state, hidden=hidden, half_width=float(half_width), device=device)

    # ------------------------------------------------------------------ the kernel

    @property
    def name(self) -> str:
        return f"energy_net_H{self.hidden}"

    @property
    def theta(self) -> np.ndarray:
        return np.concatenate([v.numpy().ravel() for v in self.state.values()])

    @property
    def n_params(self) -> int:
        return int(sum(v.numel() for v in self.state.values()))

    def _net(self) -> nn.Module:
        net = _mlp(self.hidden).to(self.device)
        net.load_state_dict(self.state)
        return net

    def _log_k(self, net: nn.Module, grid: np.ndarray) -> torch.Tensor:
        """(N_g, N_g) torch tensor, log K[k, j] = log K(u_k | u_j), normalised over k with the trapezoid weights."""
        u = torch.as_tensor(grid, dtype=torch.float64, device=self.device) * (2.0 / self.half_width)
        n = len(grid)
        pairs = torch.stack([u[None, :].expand(n, n), u[:, None].expand(n, n)], dim=-1)   # [k, j] = (u_j, u_k)
        logits = -net(pairs.reshape(-1, 2)).reshape(n, n)
        log_w = torch.log(torch.as_tensor(_quad_weights(grid), dtype=torch.float64, device=self.device))
        return logits - torch.logsumexp(logits + log_w[:, None], dim=0, keepdim=True)

    def log_transition_matrix(self, grid: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self._log_k(self._net(), grid).cpu().numpy()

    # ------------------------------------------------------------------ generalised M-step

    def m_step(self, stats, grid: np.ndarray) -> "NeuralEnergyKernel":
        xi = torch.as_tensor(stats.xi, dtype=torch.float64, device=self.device)
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
                return replace(self, state={k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
                               lr=new_lr, opt_state=_cpu(opt.state_dict()))
            lr *= 0.5
        return replace(self, lr=lr)


def _cpu(obj):
    """A deep copy of an optimizer state dict with every tensor on the CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cpu(v) for v in obj]
    return copy.deepcopy(obj)
