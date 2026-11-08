"""A vector state (d = 2): a bivariate Markov chain, BP on a 2-D state grid, kernels learned by EM, two-channel networks.

The chain (E67):  b_{k+1} = A b_k + eps_k,  A = rho R(theta),  eps_k with independent Laplace components of variance
1 - rho^2, so the stationary covariance is the identity. Each component is observed through the OU channel at level t.

The state grid is the product grid g x g (n points per axis, S = n^2 states, flattened row-major: s = i n + j holds
(g_i, g_j)), with product trapezoid weights. A kernel is an S x S matrix log_k[s', s] = log K(b' = state s' | b = s),
columns normalised under the 2-D weights. The recursion is bp_grid.grid_bp_batch's, with a state in place of a number:
forward messages exclude their own site's likelihood, beliefs are fwd * ell * bwd, every message is normalised. Its cost
is O(L S^2) = O(L n^4) per sequence, against O(L n^2) for a scalar state: the point E67 makes.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace

import numpy as np

from markov_score.bp_grid import make_grid


# ---------------------------------------------------------------------------------------- the chain

def rotation(theta: float) -> np.ndarray:
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


@dataclass(frozen=True)
class RotatedLaplaceVAR:
    rho: float = 0.85
    theta: float = math.pi / 6
    burn_in: int = 64

    @property
    def A(self) -> np.ndarray:
        return self.rho * rotation(self.theta)

    @property
    def b(self) -> float:
        """Laplace scale of each innovation component: variance 2 b^2 = 1 - rho^2."""
        return math.sqrt((1.0 - self.rho ** 2) / 2.0)

    def sample(self, rng: np.random.Generator, n: int, L: int) -> np.ndarray:
        """(n, L, 2); the first site after `burn_in` steps from 0 (stationary to ~rho^64)."""
        x = np.zeros((n, 2))
        out = np.empty((n, L, 2))
        for k in range(self.burn_in + L):
            x = x @ self.A.T + rng.laplace(0.0, self.b, size=(n, 2))
            if k >= self.burn_in:
                out[:, k - self.burn_in] = x
        return out

    def covariance(self, L: int) -> np.ndarray:
        """(2L, 2L) stationary covariance, site-major: Cov(b_i, b_j) = A^(i-j) for i >= j (Sigma_0 = I)."""
        C = np.zeros((2 * L, 2 * L))
        P = [np.eye(2)]
        for _ in range(L):
            P.append(P[-1] @ self.A)
        for i in range(L):
            for j in range(L):
                blk = P[i - j] if i >= j else P[j - i].T
                C[2 * i:2 * i + 2, 2 * j:2 * j + 2] = blk
        return C

    def log_transition_matrix(self, states: np.ndarray, w2: np.ndarray) -> np.ndarray:
        eps = states[:, None, :] - (states @ self.A.T)[None, :, :]                 # [s', s] = s' - A s
        log_k = (-np.abs(eps) / self.b - math.log(2.0 * self.b)).sum(axis=2)
        return normalise(log_k, w2)


# ---------------------------------------------------------------------------------------- the grid

def state_grid(half_width: float, n: int):
    """(g, w, states (S, 2), w2 (S,)): the 1-D grid and weights, and the flattened product grid and its weights."""
    g, w = make_grid(half_width, n)
    states = np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1).reshape(-1, 2)
    w2 = np.outer(w, w).ravel()
    return g, w, states, w2


def normalise(log_k: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """Columns normalised so that sum_{s'} w2[s'] K[s', s] = 1."""
    m = log_k.max(axis=0, keepdims=True)
    return log_k - (m + np.log((np.exp(log_k - m) * w2[:, None]).sum(axis=0, keepdims=True)))


def standard_normal_log_mu(states: np.ndarray) -> np.ndarray:
    return -0.5 * (states ** 2).sum(axis=1) - math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------------------- BP and the E-step

@dataclass
class Stats2D:
    xi: np.ndarray            # (S, S): expected transitions, [s', s]
    log_evidence: float
    n_edges: int


def bp2d(states, w2, log_k, X, alpha, delta, log_mu=None, chunk=128, want_xi=False):
    """Posterior means (B, L, 2) of the 2-D chain; with want_xi, also the expected transitions and the evidence.

    X: (B, L, 2) noisy observations, all at one level (alpha, delta). Processed in chunks of sequences.
    """
    S = len(states)
    K = np.exp(log_k)
    if log_mu is None:
        log_mu = standard_normal_log_mu(states)
    mu = np.exp(log_mu - log_mu.max())
    means = np.empty(X.shape)
    xi = np.zeros((S, S)) if want_xi else None
    logz = 0.0
    for c0 in range(0, len(X), chunk):
        Xc = X[c0:c0 + chunk]
        B, L, _ = Xc.shape
        z = Xc[:, :, None, :] - alpha * states[None, None, :, :]                     # (B, L, S, 2)
        log_ell = -0.5 * (z ** 2).sum(axis=3) / delta
        shift = log_ell.max(axis=2, keepdims=True)
        ell = np.exp(log_ell - shift)                                                # (B, L, S)
        fwd = np.empty((L, B, S))
        bwd = np.empty((L, B, S))
        fwd[0] = mu[None, :] / (mu @ w2)                                              # the initial law, grid-normalised
        lz = np.zeros(B)                    # log p(x) = sum of normalisers + row shifts - L log(2 pi delta), added below
        for i in range(L - 1):
            out = (fwd[i] * ell[:, i] * w2) @ K.T
            mass = out @ w2
            if not np.all(np.isfinite(mass)) or np.any(mass <= 0):
                raise FloatingPointError(f"2-D forward message {i + 1} lost mass")
            fwd[i + 1] = out / mass[:, None]
            lz = lz + np.log(mass) + shift[:, i, 0]
        lz = lz + np.log((fwd[L - 1] * ell[:, L - 1]) @ w2) + shift[:, L - 1, 0]
        bwd[L - 1] = 1.0
        for i in range(L - 1, 0, -1):
            out = (bwd[i] * ell[:, i] * w2) @ K
            mass = out @ w2
            if not np.all(np.isfinite(mass)) or np.any(mass <= 0):
                raise FloatingPointError(f"2-D backward message {i - 1} lost mass")
            bwd[i - 1] = out / mass[:, None]
        for i in range(L):
            raw = fwd[i] * ell[:, i] * bwd[i]
            mass = raw @ w2
            bel = raw * w2 / mass[:, None]
            means[c0:c0 + B, i] = bel @ states
        if want_xi:
            for i in range(L - 1):
                u = fwd[i] * ell[:, i] * w2                                          # (B, S): origin side
                v = w2 * ell[:, i + 1] * bwd[i + 1]                                  # (B, S): arrival side
                tot = np.einsum("bs,bs->b", v, u @ K.T)
                xi += K * ((v / tot[:, None]).T @ u)
            # log p(x) per sequence: the forward normalisers, plus the Gaussian constants of the channel
            logz += float(np.sum(lz)) - B * L * math.log(2.0 * math.pi * delta)
    if want_xi:
        return means, Stats2D(xi=xi, log_evidence=logz, n_edges=int(len(X) * (X.shape[1] - 1)))
    return means


def e_step2d(states, w2, log_k, groups, log_mu=None, chunk=128):
    """Sum of the expected transitions over (X, alpha, delta) groups (one noise level each)."""
    tot = None
    for X, alpha, delta in groups:
        _, st = bp2d(states, w2, log_k, X, alpha, delta, log_mu, chunk, want_xi=True)
        if tot is None:
            tot = st
        else:
            tot = Stats2D(xi=tot.xi + st.xi, log_evidence=tot.log_evidence + st.log_evidence, n_edges=tot.n_edges + st.n_edges)
    return tot


# ---------------------------------------------------------------------------------------- kernels

@dataclass(frozen=True)
class GaussianVAR1:
    """K(b' | b) = N(b'; A b, Q), closed-form M-step (the posterior-weighted least squares of a VAR(1))."""
    A: np.ndarray
    Q: np.ndarray

    @classmethod
    def init(cls, rho=0.3, var=0.8):
        return cls(A=rho * np.eye(2), Q=var * np.eye(2))

    @property
    def theta(self):
        return np.concatenate([self.A.ravel(), self.Q.ravel()])

    def log_transition_matrix(self, states, w2):
        e = states[:, None, :] - (states @ self.A.T)[None, :, :]
        Qi = np.linalg.inv(self.Q)
        return normalise(-0.5 * np.einsum("abi,ij,abj->ab", e, Qi, e), w2)

    def m_step(self, stats: Stats2D, states, w2):
        W = stats.xi.sum()
        Sxx = np.einsum("ab,bi,bj->ij", stats.xi, states, states)
        Syx = np.einsum("ab,ai,bj->ij", stats.xi, states, states)
        A = Syx @ np.linalg.inv(Sxx)
        e = states[:, None, :] - (states @ A.T)[None, :, :]
        Q = np.einsum("ab,abi,abj->ij", stats.xi, e, e) / W
        return replace(self, A=A, Q=0.5 * (Q + Q.T))


def _mlp4(hidden: int):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(4, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)).double()


@dataclass(frozen=True)
class NeuralEnergyKernel2D:
    """K(b' | b) proportional to exp(-E_theta(b, b')), E a 4-input MLP with two hidden SiLU layers; uniform start;
    generalised M-step (Adam on Q, step halving, a FRESH optimiser state after a rejected step: the stall fix of
    NeuralEnergyKernel.fresh_on_reject, bf55e2c)."""
    state: dict
    hidden: int
    half_width: float
    lr: float = 1e-2
    lr_max: float = 1e-2
    n_inner: int = 10
    max_halvings: int = 6
    opt_state: dict | None = None

    @classmethod
    def init(cls, hidden, seed, half_width, lr=1e-2, n_inner=10):
        import torch
        import torch.nn as nn
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            net = _mlp4(hidden)
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
        return cls(state={k: v.detach().clone() for k, v in net.state_dict().items()}, hidden=hidden,
                   half_width=float(half_width), lr=lr, lr_max=lr, n_inner=n_inner)

    @property
    def theta(self):
        return np.concatenate([v.numpy().ravel() for v in self.state.values()])

    def _net(self):
        net = _mlp4(self.hidden)
        net.load_state_dict(self.state)
        return net

    def _log_k(self, net, states, w2):
        import torch
        u = torch.as_tensor(states, dtype=torch.float64) * (2.0 / self.half_width)
        S = len(states)
        pairs = torch.cat([u[None, :, :].expand(S, S, 2), u[:, None, :].expand(S, S, 2)], dim=-1)   # [s', s] = (b, b')
        logits = -net(pairs.reshape(-1, 4)).reshape(S, S)
        lw = torch.log(torch.as_tensor(w2, dtype=torch.float64))
        return logits - torch.logsumexp(logits + lw[:, None], dim=0, keepdim=True)

    def log_transition_matrix(self, states, w2):
        import torch
        with torch.no_grad():
            return self._log_k(self._net(), states, w2).numpy()

    def m_step(self, stats: Stats2D, states, w2):
        import torch
        xi = torch.as_tensor(stats.xi, dtype=torch.float64)
        scale = 1.0 / max(int(stats.n_edges), 1)
        net = self._net()

        def q():
            return (xi * self._log_k(net, states, w2)).sum() * scale

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
                return replace(self, state={k: v.detach().clone() for k, v in net.state_dict().items()}, lr=new_lr,
                               opt_state=copy.deepcopy(opt.state_dict()))
            lr *= 0.5
        return replace(self, lr=max(lr, 1e-6), opt_state=None)


def fit_em2d(kernel, states, w2, groups, checkpoints, log_mu=None, chunk=128):
    """EM through 2-D BP; returns ({iteration: kernel} at the checkpoints, [evidence], [seconds])."""
    import time
    saved, ev, secs, cur = {}, [], [], kernel
    for it in range(max(checkpoints) + 1):
        if it in checkpoints:
            saved[it] = cur
        if it == max(checkpoints):
            break
        t0 = time.perf_counter()
        st = e_step2d(states, w2, cur.log_transition_matrix(states, w2), groups, log_mu, chunk)
        ev.append(st.log_evidence)
        cur = cur.m_step(st, states, w2)
        secs.append(time.perf_counter() - t0)
    return saved, ev, secs
