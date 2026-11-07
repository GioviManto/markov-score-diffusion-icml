"""Grid BP for a chain whose kernel changes from link to link (E68).

The paper's chains use one kernel K at every link. A non-stationary chain uses K_k for the link a_k -> a_{k+1}; the
posterior is still a tree, so BP is still exact, with the same cost: only the matrix of each step changes.

    log_Ks   (L - 1, N_g, N_g): log_Ks[k][i, j] = log K_k(u_i | u_j), the package's [out, in] convention for the link
             from site k to site k + 1 (0-based). One (N_g, N_g) matrix is used at every link (= `grid_bp_batch`).
    alpha, delta   one noise level for the batch, or one per chain.

`site_bp_batch` returns the posterior mean and variance of every site and log p(x) of every chain and, on request, the
EM statistics PER LINK: xi[k][i, j], the expected number of transitions u_j -> u_i at link k summed over chains, plus
the expected first-site mass. A stationary model sums xi over links (`xi.sum(0)` is what `em.e_step` returns); a
site-dependent model keeps them apart. tests/test_site_bp.py holds it to `grid_bp_batch`, to `em.e_step` and to the
exact posterior of a non-stationary Gaussian chain.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SiteBPResult:
    means: np.ndarray                 # (B, L) posterior means
    variances: np.ndarray             # (B, L) posterior variances
    log_evidence: np.ndarray          # (B,) log p(x) of each chain
    xi: np.ndarray | None             # (L - 1, N_g, N_g) expected transition mass per link, summed over chains
    site1: np.ndarray | None          # (N_g,) expected first-site mass, summed over chains (sums to B)


def _link_kernels(log_Ks, n_links: int, n_grid: int) -> np.ndarray:
    log_Ks = np.asarray(log_Ks, dtype=float)
    if log_Ks.shape == (n_grid, n_grid):
        log_Ks = np.broadcast_to(log_Ks, (n_links, n_grid, n_grid))
    if log_Ks.shape != (n_links, n_grid, n_grid):
        raise ValueError(f"log_Ks must be ({n_links}, {n_grid}, {n_grid}) or ({n_grid}, {n_grid}), got {log_Ks.shape}")
    return np.exp(log_Ks)


def site_bp_batch(grid, weights, log_Ks, X, alpha, delta, log_mu, *, pairs: bool = False,
                  chunk: int = 256) -> SiteBPResult:
    X = np.atleast_2d(np.asarray(X, dtype=float))
    B, L = X.shape
    grid, weights = np.asarray(grid, dtype=float), np.asarray(weights, dtype=float)
    K = _link_kernels(log_Ks, L - 1, len(grid))
    alpha = np.broadcast_to(np.asarray(alpha, dtype=float), (B,))
    delta = np.broadcast_to(np.asarray(delta, dtype=float), (B,))
    mu = np.exp(np.asarray(log_mu, dtype=float))
    parts = [_block(grid, weights, K, X[s:s + chunk], alpha[s:s + chunk], delta[s:s + chunk], mu, pairs)
             for s in range(0, B, chunk)]
    def cat(name):
        return np.concatenate([p[name] for p in parts])
    return SiteBPResult(means=cat("means"), variances=cat("variances"), log_evidence=cat("log_evidence"),
                        xi=sum(p["xi"] for p in parts) if pairs else None,
                        site1=sum(p["site1"] for p in parts) if pairs else None)


def _block(grid, w, K, X, alpha, delta, mu, pairs):
    L = X.shape[1]
    ll = (-0.5 * (X[:, :, None] - alpha[:, None, None] * grid) ** 2 / delta[:, None, None]
          - 0.5 * np.log(2 * np.pi * delta)[:, None, None])                 # log l(x_k | u), (B, L, N_g)
    llmax = ll.max(axis=2, keepdims=True)
    lik = np.exp(ll - llmax)
    log_ev = llmax.sum(axis=(1, 2))

    F = np.empty_like(lik)                                                  # law of a_k given x_1..x_k
    f = mu[None, :] * lik[:, 0]
    for k in range(L):
        if k:
            f = ((F[:, k - 1] * w) @ K[k - 1].T) * lik[:, k]
        z = f @ w
        if not np.all(z > 0) or not np.all(np.isfinite(z)):
            raise FloatingPointError(f"forward message lost all mass at site {k}")
        F[:, k] = f / z[:, None]
        log_ev += np.log(z)
    G = np.ones_like(lik)                                                   # what x_{k+1}..x_L say about a_k
    for k in range(L - 2, -1, -1):
        g = (lik[:, k + 1] * G[:, k + 1] * w) @ K[k]
        G[:, k] = g / g.max(axis=1, keepdims=True)

    P = F * G
    P /= (P @ w)[:, :, None]
    means = P @ (w * grid)
    out = {"means": means, "variances": P @ (w * grid**2) - means**2, "log_evidence": log_ev}
    if pairs:
        xi = np.empty_like(K)
        for k in range(L - 1):
            left = F[:, k] * w                                              # a_k = u_j
            right = lik[:, k + 1] * G[:, k + 1] * w                         # a_{k+1} = u_i
            z = ((left @ K[k].T) * right).sum(axis=1)
            xi[k] = K[k] * ((right / z[:, None]).T @ left)
        out["xi"] = xi
        out["site1"] = (P[:, 0] * w).sum(axis=0)
    return out
