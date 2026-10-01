"""State augmentation for a global two-valued latent: the repair of the consolidated note's section 8.

The misspecification. Let X_i = G b + U_i with G in {-1, +1} a fair coin drawn once per sequence
and U a stationary first-order chain. Conditioned on G the sequence is first-order Markov;
marginalised over G it is not (the whole history informs G), and a first-order kernel fitted
to such data converges to the KL-closest first-order law and keeps an asymptotic bias
(E08 measured a Gaussian version of this with a rank-one latent). The note's repair is to
enlarge the state to S_i = (X_i, G): a hidden Markov model whose latent chain is the trivial
one (G never changes), i.e. a MIXTURE OF TWO CHAINS with the mixture weight inferred per
sequence from the evidence.

The augmentation here uses the Z_2 symmetry of the truth (U symmetric): the minus component
is the reflection of the plus one, K_-(x'|x) = K_+(-x'|-x) and mu_-(x) = mu_+(-x), so the
augmented model has exactly the parameters of one kernel, and no extra parameter for b: an
AR(1)-type kernel with an innovation mean already represents a chain with mean +b. On the
symmetric grid the reflection is index reversal.

What is computed exactly
    augmented_posterior      per component: exact grid BP with its evidence; responsibilities
                             r_g(y) = p(G = g | y) from the two evidences; posterior mean
                             sum_g r_g E[X | y, G = g].
    augmented_e_step         the E-step of the shared kernel: each component's pair statistic,
                             weighted by its responsibility (em.e_step's chain_weights), the
                             minus one reflected onto the plus frame, summed. Fisher's identity
                             holds for the mixture evidence, so EM is exact maximum likelihood
                             for the augmented model.
    fit_em_augmented         the EM loop with the checkpoint ladder, the monotone check and the
                             stopping rule of em.fit_em (same conventions: a checkpoint is a
                             kernel whose evidence was computed).
    two_phase_gaussian_posterior_mean
                             the closed-form reference when U is Gaussian AR(1): a mixture of
                             two Gaussian chains, E[X|y] = sum_g r_g(y) [g b 1 + alpha
                             Sigma_0 Sigma_t^{-1} (y - alpha g b 1)], so the exact score of the
                             truth needs no grid at all (the test that validates the grid).
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from markov_score.bp_grid import grid_bp_batch
from markov_score.em import ExpectedStatistics, e_step
from markov_score.noising import alpha_delta


def reflect_kernel(log_k: np.ndarray) -> np.ndarray:
    """K_-(u_k | u_j) = K_+(u_{M-1-k} | u_{M-1-j}) on a symmetric grid."""
    return log_k[::-1, ::-1]


def reflect_law(log_mu: np.ndarray) -> np.ndarray:
    return log_mu[::-1]


def augmented_posterior(grid, w, log_k_plus, X, alpha, delta, log_mu_plus, prior_plus=0.5):
    """Posterior mean under the Z_2-augmented chain, with the per-chain mixture log evidence and
    the responsibilities (2, B)."""
    comps = ((log_k_plus, log_mu_plus, np.log(prior_plus)),
             (reflect_kernel(log_k_plus), reflect_law(log_mu_plus), np.log1p(-prior_plus)))
    means, levs = [], []
    for lk, lm, lp in comps:
        m, _, ev = grid_bp_batch(grid, w, lk, X, alpha, delta, log_mu=lm, return_evidence=True)
        means.append(m); levs.append(np.asarray(ev, dtype=float) + lp)
    lev = np.stack(levs)                       # (2, B)
    tot = logsumexp(lev, axis=0)               # (B,)
    r = np.exp(lev - tot[None, :])
    m = r[0][:, None] * means[0] + r[1][:, None] * means[1]
    return m, tot, r


def augmented_e_step(grid, w, log_k_plus, groups, log_mu_plus, prior_plus=0.5):
    """Expected statistics of the SHARED kernel over `groups` = [(X, alpha, delta), ...], and the
    total mixture log evidence. xi sums to the edge count exactly (r_+ + r_- = 1 per chain)."""
    lk_minus, lm_minus = reflect_kernel(log_k_plus), reflect_law(log_mu_plus)
    xi = site1 = None
    n_edges = n_chains = 0
    total_ev = 0.0
    for X, alpha, delta in groups:
        _, tot, r = augmented_posterior(grid, w, log_k_plus, X, alpha, delta, log_mu_plus, prior_plus)
        sp = e_step(grid, w, log_k_plus, X, alpha, delta, log_mu=log_mu_plus, chain_weights=r[0])
        sm = e_step(grid, w, lk_minus, X, alpha, delta, log_mu=lm_minus, chain_weights=r[1])
        xi_g = sp.xi + sm.xi[::-1, ::-1]
        s1 = sp.site1 + sm.site1[::-1]
        xi = xi_g if xi is None else xi + xi_g
        site1 = s1 if site1 is None else site1 + s1
        n_edges += sp.n_edges; n_chains += sp.n_chains
        total_ev += float(tot.sum())
    stats = ExpectedStatistics(xi=xi, site1=site1, log_evidence=total_ev, n_edges=n_edges, n_chains=n_chains)
    return stats, total_ev


def fit_em_augmented(kernel, grid, w, groups, n_iters, log_mu, tol=1e-9, patience=3, checkpoints=None,
                     prior_plus=0.5):
    """EM for the augmented model with a shared kernel. `log_mu` is an array or a callable of
    the kernel (the component's own invariant law). Returns (kernel, trace dict, saved), where
    saved[it] is the kernel whose evidence is trace['log_evidence'][it - 1], as in em.fit_em.
    The returned kernel is the last SCORED one (n_iters is always treated as a checkpoint)."""
    checkpoints = set(checkpoints or ()) | {int(n_iters)}
    current = kernel
    evid, saved = [], {}
    small, converged, reason = 0, False, "censored"
    violation = 0.0
    for it in range(1, n_iters + 1):
        mu = log_mu(current) if callable(log_mu) else log_mu
        stats, lev = augmented_e_step(grid, w, current.log_transition_matrix(grid), groups, mu, prior_plus)
        evid.append(lev)
        if it in checkpoints:
            saved[it] = current
        if len(evid) > 1:
            gain = (evid[-1] - evid[-2]) / max(stats.n_edges, 1)
            violation = max(violation, -gain)
            small = small + 1 if 0.0 <= gain <= tol else 0
            if small >= max(1, patience):
                converged, reason = True, "converged"
                break
        current = current.m_step(stats, grid)
    last_scored = saved[max(saved)] if saved else current
    trace = dict(log_evidence=evid, converged=converged, stop_reason=reason, n_iters=len(evid),
                 monotone_violation=float(violation))
    return last_scored, trace, saved


def two_phase_gaussian_posterior_mean(X, sigma0, b, t, prior_plus=0.5):
    """Exact E[X | y] for X = G b 1 + U, U ~ N(0, sigma0), G = +-1, y = alpha X + sqrt(delta) Z."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    alpha, delta = alpha_delta(t)
    n = sigma0.shape[0]
    st = alpha**2 * sigma0 + delta * np.eye(n)
    chol = np.linalg.cholesky(st)
    ones = np.ones(n)
    levs, means = [], []
    for g, lp in ((1.0, np.log(prior_plus)), (-1.0, np.log1p(-prior_plus))):
        resid = X - alpha * g * b * ones[None, :]
        z = np.linalg.solve(chol, resid.T)                      # (n, B)
        levs.append(-0.5 * (z**2).sum(axis=0) + lp)             # common constants cancel in r
        means.append(g * b * ones[None, :] + (alpha * sigma0 @ np.linalg.solve(st, resid.T)).T)
    lev = np.stack(levs)
    r = np.exp(lev - logsumexp(lev, axis=0)[None, :])
    return r[0][:, None] * means[0] + r[1][:, None] * means[1]
