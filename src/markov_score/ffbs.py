"""Exact posterior sampling on the grid by forward filtering, backward sampling (FFBS).

The consolidated note, section 9.7: "for posterior sampling after a forward pass, sample X_L ~ p(X_L | y) and then move
backward with p(x_i | x_{i+1}, y) proportional to alpha_i(x_i) K(x_{i+1} | x_i)" -- and "merely starting from a flat
marginal and running BP is not, by itself, a complete sampling algorithm". In the notation of docs/NOTATION.md the
note's alpha_i is m->_i ell_i, the filtering density:

    a_L ~ m->_L ell_L,        a_i | a_{i+1}, x ~ m->_i(a) ell_i(a) K(a_{i+1} | a) w(a)     for i = L-1, ..., 1.

Each draw is an exact sample of the finite-state posterior the grid defines (the same model `grid_bp_batch` computes
marginals of), so its site marginals are BP's beliefs and its pair statistics BP's pair marginals; `jitter` adds a
uniform offset within the sampled grid cell, as `e27_direct_sampler` does for prior samples.
"""

from __future__ import annotations

import numpy as np


def _categorical(rng, p):
    """One index per row of p (rows sum to one), by inverse CDF."""
    c = np.cumsum(p, axis=1)
    u = rng.random((p.shape[0], 1)) * c[:, -1:]
    return np.minimum((c < u).sum(axis=1), p.shape[1] - 1)


def grid_ffbs(grid, weights, log_K, X, alpha, delta, rng, n_samples=1, log_mu=None, jitter=False):
    """Posterior samples, shape (n_samples, B, L), for observations X of shape (B, L)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n_chains, n_sites = X.shape
    n_grid = len(grid)
    w = np.asarray(weights, dtype=float)
    K = np.exp(log_K)                                           # K[out, in]
    if log_mu is None:
        log_mu = -0.5 * grid**2 - 0.5 * np.log(2.0 * np.pi)
    z = X[:, :, None] - alpha * grid[None, None, :]
    log_ell = -0.5 * z**2 / delta
    ell = np.exp(log_ell - log_ell.max(axis=2, keepdims=True))  # (B, L, N_g)
    filt = np.empty((n_sites, n_chains, n_grid))                # m->_i ell_i w, normalised to unit mass
    pred = np.exp(log_mu)[None, :].repeat(n_chains, axis=0)
    for i in range(n_sites):
        f = pred * ell[:, i, :] * w[None, :]
        mass = f.sum(axis=1, keepdims=True)
        if not np.all(np.isfinite(mass)) or np.any(mass <= 0):
            raise FloatingPointError(f"FFBS filter at site {i} lost all mass.")
        filt[i] = f / mass
        if i < n_sites - 1:
            pred = filt[i] @ K.T                                # m->_{i+1} on the grid (up to w, absorbed above)
    out = np.empty((n_samples, n_chains, n_sites))
    du = float(grid[1] - grid[0])
    for s in range(n_samples):
        idx = np.empty((n_chains, n_sites), dtype=int)
        idx[:, -1] = _categorical(rng, filt[-1])
        for i in range(n_sites - 2, -1, -1):
            p = filt[i] * K[idx[:, i + 1], :]                   # K(a_{i+1} | a) as a function of a, per chain
            p = p / p.sum(axis=1, keepdims=True)
            idx[:, i] = _categorical(rng, p)
        a = grid[idx]
        if jitter:
            a = np.clip(a + (rng.random(a.shape) - 0.5) * du, grid[0], grid[-1])
        out[s] = a
    return out
