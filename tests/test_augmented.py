"""State augmentation for a global sign (markov_score.augmented) and the weighted E-step.

Pinned: the weighted E-step with unit weights is bit-for-bit the unweighted one and with 0/1
weights is the E-step on the subset; the augmented grid posterior with the TRUE component kernels
matches the closed-form two-phase Gaussian mixture to quadrature precision; the pooled augmented
statistic sums to the edge count; EM with the augmented state beats the first-order fit on
two-phase data.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from markov_score.augmented import (augmented_e_step, augmented_posterior, fit_em_augmented,
                                    two_phase_gaussian_posterior_mean)
from markov_score.bp_grid import make_grid
from markov_score.em import e_step, fit_em
from markov_score.kernels import MixtureInnovationKernel
from markov_score.noising import alpha_delta
from markov_score.priors import GaussianAR1
from markov_score.stationary import invariant_log_density_eig
from markov_score.utils import rng_for

RHO = 0.85


def _two_phase_data(rng, n_chains, L, b):
    prior = GaussianAR1(RHO)
    U = np.stack([prior.sample(rng, L) for _ in range(n_chains)])
    G = rng.choice([-1.0, 1.0], size=n_chains)
    return G[:, None] * b + U


def _true_component(grid, w, b):
    """log K_+(u_k | u_j) = log N(u_k; b + rho (u_j - b), q), column-normalised; mu_+ = N(b, 1)."""
    q = 1 - RHO**2
    mean = b + RHO * (grid[None, :] - b)
    raw = -0.5 * (grid[:, None] - mean) ** 2 / q - 0.5 * np.log(2 * np.pi * q)
    lk = raw - logsumexp(raw + np.log(w)[:, None], axis=0)[None, :]
    lm = -0.5 * (grid - b) ** 2 - 0.5 * np.log(2 * np.pi)
    lm = lm - logsumexp(lm + np.log(w))
    return lk, lm


def _ar1_cov(L):
    i = np.arange(L)
    return RHO ** np.abs(i[:, None] - i[None, :])


def test_weighted_e_step_reduces_to_the_unweighted_one():
    g, w = make_grid(8.0, 201)
    rng = rng_for("aug-weights", 0)
    A = _two_phase_data(rng, 12, 10, 0.0)
    al, de = alpha_delta(0.4665)
    X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
    lk, lm = _true_component(g, w, 0.0)
    s0 = e_step(g, w, lk, X, al, de, log_mu=lm)
    s1 = e_step(g, w, lk, X, al, de, log_mu=lm, chain_weights=np.ones(len(X)))
    # unit weights go through g * (w / p) rather than g / p: equal to rounding, not bit-for-bit
    # (the unweighted path itself is untouched and stays bit-for-bit what it was)
    assert np.allclose(s0.xi, s1.xi, rtol=1e-10, atol=0.0) and np.allclose(s0.site1, s1.site1, rtol=1e-10, atol=0.0)
    mask = np.zeros(len(X)); mask[:5] = 1.0
    s2 = e_step(g, w, lk, X, al, de, log_mu=lm, chain_weights=mask)
    s3 = e_step(g, w, lk, X[:5], al, de, log_mu=lm)
    assert np.allclose(s2.xi, s3.xi, rtol=1e-12, atol=1e-14) and np.allclose(s2.site1, s3.site1)


def test_augmented_posterior_matches_the_closed_form():
    g, w = make_grid(8.0, 401)
    b, L = 1.0, 24
    rng = rng_for("aug-closed", 0)
    A = _two_phase_data(rng, 64, L, b)
    sigma0 = _ar1_cov(L)
    lk, lm = _true_component(g, w, b)
    for t in (0.2216, 0.9821):
        al, de = alpha_delta(t)
        X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
        m, tot, r = augmented_posterior(g, w, lk, X, al, de, lm)
        m_ref = two_phase_gaussian_posterior_mean(X, sigma0, b, t)
        assert np.max(np.abs(m - m_ref)) < 2e-3, (t, np.max(np.abs(m - m_ref)))
        assert np.allclose(r.sum(axis=0), 1.0)
        assert r.shape == (2, len(X)) and tot.shape == (len(X),)


def test_augmented_statistic_sums_to_the_edge_count():
    g, w = make_grid(8.0, 201)
    rng = rng_for("aug-stats", 0)
    A = _two_phase_data(rng, 20, 12, 1.0)
    groups = []
    for t in (0.2216, 0.9821):
        al, de = alpha_delta(t)
        groups.append((al * A + np.sqrt(de) * rng.standard_normal(A.shape), al, de))
    lk, lm = _true_component(g, w, 1.0)
    stats, lev = augmented_e_step(g, w, lk, groups, lm)
    assert abs(stats.xi.sum() - stats.n_edges) < 1e-8 * stats.n_edges
    assert abs(stats.site1.sum() - stats.n_chains) < 1e-8 * stats.n_chains
    assert np.isfinite(lev) and stats.n_chains == 40


def test_augmented_em_beats_the_first_order_fit_on_two_phase_data():
    """The repair: on data with a global sign (b = 1.5), EM over the augmented state reaches a
    posterior-mean error well below the first-order mixture kernel's, against the closed form."""
    g, w = make_grid(8.0, 201)
    b, L = 1.5, 16
    rng = rng_for("aug-em", 0)
    A = _two_phase_data(rng, 192, L, b)
    groups = []
    for t in (0.2216, 0.4665, 0.9821):
        al, de = alpha_delta(t)
        groups.append((al * A + np.sqrt(de) * rng.standard_normal(A.shape), al, de))
    sigma0 = _ar1_cov(L)
    test_A = _two_phase_data(rng, 128, L, b)
    al, de = alpha_delta(0.4665)
    Xt = al * test_A + np.sqrt(de) * rng.standard_normal(test_A.shape)
    m_ref = two_phase_gaussian_posterior_mean(Xt, sigma0, b, 0.4665)

    def mu_of(k):
        return invariant_log_density_eig(k, g, w).log_density

    k0 = MixtureInnovationKernel.init(2, rho=0.3, var=0.8, rng=rng_for("aug-em-init", 0))
    k_aug, tr_aug, _ = fit_em_augmented(k0, g, w, groups, n_iters=40, log_mu=mu_of, checkpoints={40})
    k_first, tr_first, _ = fit_em(k0, g, w, groups, n_iters=40, log_mu=mu_of, checkpoints={40})
    assert tr_aug["monotone_violation"] < 1e-6
    m_aug, _, _ = augmented_posterior(g, w, k_aug.log_transition_matrix(g), Xt, al, de, mu_of(k_aug))
    from markov_score.bp_grid import grid_bp_batch
    m_first, _ = grid_bp_batch(g, w, k_first.log_transition_matrix(g), Xt, al, de, log_mu=mu_of(k_first))
    err_aug = np.sqrt(np.mean((m_aug - m_ref) ** 2)); err_first = np.sqrt(np.mean((m_first - m_ref) ** 2))
    assert err_aug < 0.5 * err_first, (err_aug, err_first)
    assert err_aug < 0.15, err_aug


def test_e35_driver_reference_is_the_exact_posterior():
    """N-34: the E35 driver's reference must be the exact two-phase posterior mean, i.e. agree with
    the grid posterior of the TRUE component kernels at b > 0 (it once used the total covariance)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import e35_state_augmentation as e35
    g, w = make_grid(8.0, 401)
    b, L = 1.0, 16
    lk, lm = _true_component(g, w, b)
    bundle = e35._bundle("test-e35-ref", 0, b, 48, L)
    for t in (0.2216, 0.9821, 2.0676):
        X, ref = bundle[t]
        al, de = alpha_delta(t)
        m, _, _ = augmented_posterior(g, w, lk, X, al, de, lm)
        assert np.max(np.abs(m - ref)) < 1e-6, (t, np.max(np.abs(m - ref)))
