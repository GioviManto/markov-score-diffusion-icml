"""Per-chain scores of the marginal log-likelihood and outer-product information.

Fisher's identity: for one noised chain x at level t and a kernel with parameters
theta, the gradient of log P_t,theta(x) is the posterior expectation of the
complete-data gradient, which lives on the edges:

    d_theta log P(x) = sum_i  < b_{i,i+1}, d_theta log K >,

with b_{i,i+1} the pairwise posterior on edge (i, i+1). The legacy `e_step` sums the
pairwise masses over the dataset into one (M, M) matrix, which is what EM needs but
loses the per-chain scores that an information matrix needs. `experiments/exp_22`
recovered them by calling the E-step one chain at a time (slow). This module computes
them batched: with forward/backward messages F, R of shape (n, B, M) the per-chain,
per-parameter score is a batched bilinear form

    g[b, p] = sum_i  g_{i+1}^T (G_p) f_i / Z_i,   G_p = d_p log K * K  (elementwise),

which never materialises a per-chain (M, M) matrix.

`outer_product_information(g)` = g^T g / B is the empirical outer-product estimate of
the Fisher information (equal to it in expectation at the true parameter of a
correctly specified model; it is not the observed information). `efficient` =
1 / (J^{-1})_pp is the information about coordinate p after the others are estimated.
"""

from __future__ import annotations

import numpy as np

from markov_score.noising import log_likelihood_matrix


def forward_backward_batch(grid, weights, log_K, X, alpha, delta, log_mu=None):
    """Normalised forward (excluding local factor) and backward messages, batched.

    Returns F, R of shape (n, B, M) and ELL of shape (n, B, M) (row-shifted likelihoods),
    following `bp_grid.grid_bp` conventions: F[i] carries factors up to site i-1's
    likelihood; R[i] carries factors from site i+1's likelihood on.
    """
    B, n = X.shape
    M = grid.size
    if log_mu is None:
        log_mu = -0.5 * grid**2 - 0.5 * np.log(2.0 * np.pi)
    K = np.exp(log_K)                      # (M_out, M_in)
    ELL = np.exp(np.stack([log_likelihood_matrix(grid, x, alpha, delta) for x in X], axis=1))  # (n, B, M)
    F = np.empty((n, B, M)); R = np.empty((n, B, M))
    F[0] = np.exp(log_mu)[None]
    for i in range(n - 1):
        inc = F[i] * ELL[i] * weights[None]          # (B, M)
        out = inc @ K.T                              # (B, M_out)
        c = (out * weights[None]).sum(axis=1, keepdims=True)
        F[i + 1] = out / c
    R[-1] = 1.0
    for i in range(n - 1, 0, -1):
        inc = R[i] * ELL[i] * weights[None]
        out = inc @ K                                # (B, M_in)
        c = (out * weights[None]).sum(axis=1, keepdims=True)
        R[i - 1] = out / c
    return F, R, ELL


def per_chain_scores(grid, weights, kernel, X, alpha, delta, log_mu=None) -> np.ndarray:
    """(B, P) matrix of per-chain gradients of log P_t,theta(x) by Fisher's identity."""
    log_K = kernel.log_transition_matrix(grid)
    grad = kernel.grad_log_transition_matrix(grid)     # (P, M_out, M_in)
    K = np.exp(log_K)
    G = grad * K[None]                                  # (P, M_out, M_in)
    F, R, ELL = forward_backward_batch(grid, weights, log_K, X, alpha, delta, log_mu)
    n, B, M = F.shape
    P = G.shape[0]
    g = np.zeros((B, P))
    for i in range(n - 1):
        f = F[i] * ELL[i] * weights[None]               # (B, M_in)
        h = R[i + 1] * ELL[i + 1] * weights[None]       # (B, M_out)
        hK = h @ K                                      # (B, M_in)
        Z = np.einsum("bi,bi->b", hK, f)                # (B,)
        hG = np.einsum("bo,poi->bpi", h, G, optimize=True)   # (B, P, M_in)
        num = np.einsum("bpi,bi->bp", hG, f)            # (B, P)
        g += num / Z[:, None]
    return g


def outer_product_information(g: np.ndarray) -> np.ndarray:
    """J = g^T g / B: outer-product estimate of the per-chain Fisher information."""
    return g.T @ g / g.shape[0]


def efficient_information(J: np.ndarray) -> np.ndarray:
    """1 / (J^{-1})_pp for every coordinate."""
    return 1.0 / np.diag(np.linalg.inv(J))


def summarise(J: np.ndarray) -> dict:
    d = np.diag(J)
    eff = efficient_information(J)
    corr = J / np.sqrt(np.outer(d, d))
    return {
        "diag": d,
        "efficient": eff,
        "nuisance_cost": d / eff,
        "corr": corr,
        "condition_number": float(np.linalg.cond(J)),
    }
