"""Exact grid BP for a SECOND-order chain, on the pair state b_t = (a_{t-1}, a_t): the machinery of E62.

A chain with p(a_t | a_{t-1}, a_{t-2}) is first order in the pair b_t = (a_{t-1}, a_t), with the transition
p(b_{t+1} | b_t) = delta(b_{t+1}^(1) - b_t^(2)) K2(a_{t+1} | a_t, a_{t-1}). The delta means a message never has to
range over pairs of pairs: the forward update

    alpha_t(a_{t-1}, a_t) = phi_t(a_t) sum_{a_{t-2}} w alpha_{t-1}(a_{t-2}, a_{t-1}) K2(a_t | a_{t-1}, a_{t-2})

is, for each value of the shared coordinate a_{t-1}, one (B x n) @ (n x n) product: n^3 per chain and site, not n^4.

Arrays. grid (n,), trapezoid weights w (n,). log_k2[k, j, i] = log K2(u_k | a_{t-1} = u_j, a_{t-2} = u_i), each (j, i)
normalised over k with w. log_mu2[j, i] = log p(a_2 = u_j, a_1 = u_i), normalised with w (x) w. X (B, L) noisy
observations x_t = alpha a_t + sqrt(delta) eps. Returns the posterior means and variances of every a_t, (B, L).

Messages are densities at the grid points; every sum over a coordinate carries its trapezoid weight. The recursion is in
probability space with a per-chain rescaling at every site; the forward messages of a chunk of chains are held in memory
(L n^2 floats per chain), so the chunk is sized to MEMORY_BYTES. `tests/test_pair_bp.py` checks it against `bp_grid.grid_bp_batch` when K2 does not depend on a_{t-2}
and against the Gaussian posterior mean of a Gaussian AR(2).
"""
from __future__ import annotations

import numpy as np

MEMORY_BYTES = 256e6


def _likelihood(grid, X, alpha, delta):
    """phi[b, t, k] = N(x_bt; alpha u_k, delta) up to a per-(b, t) factor (the max is removed for range)."""
    ll = -0.5 * (X[:, :, None] - alpha * grid[None, None, :]) ** 2 / delta
    return np.exp(ll - ll.max(axis=2, keepdims=True))


def pair_bp_batch(grid, weights, log_k2, X, alpha, delta, log_mu2, chunk="auto"):
    """Posterior means and variances of a_t under the second-order chain, for a batch of noisy chains X (B, L)."""
    X = np.asarray(X, dtype=float)
    if chunk == "auto":
        chunk = max(1, int(MEMORY_BYTES // (X.shape[1] * len(grid) ** 2 * 8)))
    if chunk is not None and len(X) > chunk:
        parts = [pair_bp_batch(grid, weights, log_k2, X[s:s + chunk], alpha, delta, log_mu2, chunk=None)
                 for s in range(0, len(X), chunk)]
        return tuple(np.concatenate([p[i] for p in parts]) for i in range(2))
    B, L = X.shape
    n = len(grid)
    w = np.asarray(weights, dtype=float)
    k2 = np.exp(log_k2)                                        # [k, j, i]
    # kw[j, i, k] = w_i K2(u_k | u_j, u_i): the forward step is a batched matmul over the shared coordinate j
    kw = np.ascontiguousarray(np.transpose(k2, (1, 2, 0)) * w[None, :, None])
    # kb[j, k, i] = w_k K2(u_k | u_j, u_i): the backward step, summing over the NEXT value k
    kb = np.ascontiguousarray(np.transpose(k2, (1, 0, 2)) * w[None, :, None])
    phi = _likelihood(grid, X, alpha, delta)                    # (B, L, n)
    mu2 = np.exp(log_mu2)                                       # [j = a_2, i = a_1]
    # forward: fw[t][b, j, k] = p(a_{t-1} = u_j, a_t = u_k, x_1..t), rescaled
    fw = np.empty((L, B, n, n))
    f = mu2.T[None, :, :] * phi[:, 0, :, None] * phi[:, 1, None, :]          # [b, a_1, a_2]
    f /= f.sum(axis=(1, 2), keepdims=True)
    fw[1] = f
    for t in range(2, L):
        # new[b, j, k] = phi_t(k) sum_i f[b, i, j] kw[j, i, k]
        g = np.matmul(np.transpose(f, (2, 0, 1)), kw)                         # (j, b, k)
        f = np.transpose(g, (1, 0, 2)) * phi[:, t, None, :]
        f /= f.sum(axis=(1, 2), keepdims=True)
        fw[t] = f
    # backward: bk[b, i, j] = p(x_{t+1}..L | a_{t-1} = u_i, a_t = u_j), rescaled
    bk = np.ones((B, n, n))
    means, vars_ = np.empty((B, L)), np.empty((B, L))
    for t in range(L - 1, 0, -1):
        post = fw[t] * bk * w[None, :, None] * w[None, None, :]               # mass of (a_{t-1}, a_t)
        post /= post.sum(axis=(1, 2), keepdims=True)
        pt = post.sum(axis=1)
        means[:, t] = pt @ grid
        vars_[:, t] = pt @ grid ** 2 - means[:, t] ** 2
        if t == 1:
            p1 = post.sum(axis=2)
            means[:, 0] = p1 @ grid
            vars_[:, 0] = p1 @ grid ** 2 - means[:, 0] ** 2
            break
        # new[b, i, j] = sum_k K2(k | j, i) w_k phi_t(k) bk[b, j, k]   (i = a_{t-2}, j = a_{t-1})
        h = bk * phi[:, t, None, :]                                            # [b, j, k]
        g = np.matmul(np.transpose(h, (1, 0, 2)), kb)                          # (j, b, i)
        bk = np.transpose(g, (1, 2, 0))                                        # [b, i, j]
        bk /= bk.max(axis=(1, 2), keepdims=True)
    return means, vars_
