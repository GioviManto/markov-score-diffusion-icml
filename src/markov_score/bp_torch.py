"""Grid BP in torch, differentiable with respect to the kernel: the posterior mean E[a | x] of a first-order chain.

The recursion is `bp_grid.grid_bp_batch`'s, line for line, so the two agree to round-off on the same kernel (tested):
forward messages exclude their own site's likelihood, the belief of site i is fwd_i * ell_i * bwd_i, every message is
normalised by its trapezoid mass. Two differences, both for training (E66): `alpha` and `delta` may differ per sequence
(a DSM batch mixes noise levels), and nothing is detached, so a loss on the posterior mean backpropagates into
`log_k` and through it into whatever parameterises the kernel.

    log_k[i, j] = log K(g_i | g_j), arrival i, origin j, columns normalised so that sum_i w_i K(g_i | g_j) = 1.
"""
from __future__ import annotations

import math

import torch


def posterior_mean(grid: torch.Tensor, weights: torch.Tensor, log_k: torch.Tensor, X: torch.Tensor,
                   alpha, delta, log_mu: torch.Tensor | None = None) -> torch.Tensor:
    """(B, L) posterior means. `alpha` and `delta` are floats or (B,) tensors (one noise level per sequence)."""
    B, L = X.shape
    dt = log_k.dtype
    X = X.to(dt)
    alpha = torch.as_tensor(alpha, dtype=dt, device=X.device).reshape(-1, 1, 1)
    delta = torch.as_tensor(delta, dtype=dt, device=X.device).reshape(-1, 1, 1)
    if log_mu is None:
        log_mu = -0.5 * grid ** 2 - 0.5 * math.log(2.0 * math.pi)
    K = torch.exp(log_k)
    z = X[:, :, None] - alpha * grid[None, None, :]
    log_ell = -0.5 * z ** 2 / delta
    ell = torch.exp(log_ell - log_ell.amax(dim=2, keepdim=True))                    # (B, L, N_g), row-shifted
    w = weights[None, :]

    fwd = [torch.exp(log_mu)[None, :].expand(B, -1)]
    for i in range(L - 1):
        out = (fwd[i] * ell[:, i, :] * w) @ K.T                                     # sum_j K[k, j] w_j f_j ell_j
        fwd.append(out / (out @ weights)[:, None])
    bwd = [None] * L
    bwd[L - 1] = torch.ones(B, len(grid), dtype=dt, device=X.device)
    for i in range(L - 1, 0, -1):
        out = (bwd[i] * ell[:, i, :] * w) @ K                                       # sum_k K[k, j] w_k b_k ell_k
        bwd[i - 1] = out / (out @ weights)[:, None]
    gw = grid * weights
    means = []
    for i in range(L):
        raw = fwd[i] * ell[:, i, :] * bwd[i]
        means.append((raw @ gw) / (raw @ weights))
    return torch.stack(means, dim=1)
