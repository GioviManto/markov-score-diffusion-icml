"""markov_score.loglinear_kernel: the log-linear conditional kernel with a 2D spline score (E43's agnostic arm)."""
from __future__ import annotations

import numpy as np

from markov_score.bp_grid import make_grid
from markov_score.em import ExpectedStatistics, fit_em
from markov_score.loglinear_kernel import LogLinearKernel, bspline_basis
from markov_score.noising import alpha_delta
from markov_score.priors import GaussianAR1
from markov_score.stationary import invariant_log_density_eig
from markov_score.utils import rng_for

GRID, W = make_grid(8.0, 401)
RHO, VAR = 0.8, 0.36


def _truth():
    lk = -0.5 * (GRID[:, None] - RHO * GRID[None, :]) ** 2 / VAR
    lk -= np.log((np.exp(lk) * W[:, None]).sum(0))[None, :]
    pi = np.exp(-0.5 * GRID**2 / (VAR / (1 - RHO**2)))
    pi /= (pi * W).sum()
    return lk, pi


def _stats(n_edges):
    lk, pi = _truth()
    xi = n_edges * np.exp(lk) * W[:, None] * (pi * W)[None, :]
    return ExpectedStatistics(xi=xi, site1=np.zeros(len(GRID)), log_evidence=0.0, n_edges=int(n_edges), n_chains=1), lk


def test_the_basis_is_a_partition_of_unity():
    B = bspline_basis(10, 8.0, GRID)
    assert B.shape == (401, 10) and np.allclose(B.sum(1), 1.0)


def test_columns_are_normalised_under_the_trapezoid_weights():
    k = LogLinearKernel.init(8, 0.5, 1.0)
    assert np.allclose((np.exp(k.log_transition_matrix(GRID)) * W[:, None]).sum(0), 1.0)


def test_init_is_close_to_the_gaussian_chain_it_projects():
    st, lk = _stats(1e5)
    k = LogLinearKernel.init(10, RHO, VAR)
    gap = (float((st.xi * lk).sum()) - float((st.xi * k.log_transition_matrix(GRID)).sum())) / st.n_edges
    assert abs(gap) < 1e-9                                      # a quadratic score is reproduced exactly by cubic splines


def test_the_m_step_never_decreases_the_likelihood_and_recovers_the_kernel():
    st, lk = _stats(1e5)
    k = LogLinearKernel.init(10, 0.3, 1.0)
    q = []
    for _ in range(10):
        q.append(float((st.xi * k.log_transition_matrix(GRID)).sum()) / st.n_edges)
        k = k.m_step(st, GRID)
    assert all(b >= a - 1e-9 for a, b in zip(q, q[1:])), q      # exact EM: the likelihood never decreases
    gap = (float((st.xi * lk).sum()) - float((st.xi * k.log_transition_matrix(GRID)).sum())) / st.n_edges
    assert gap < 1e-3                                           # nats per transition against the truth


def _sampled_stats(n_edges, seed=0):
    """Counts actually drawn (multinomial) from the truth: expected counts carry no sampling noise, so the exact MLE would return
    the truth at any n and the price of 100 parameters could not show."""
    _, lk = _stats(1.0)
    st1, _ = _stats(1.0)
    p = (st1.xi / st1.xi.sum()).ravel()
    xi = np.random.default_rng(seed).multinomial(int(n_edges), p).reshape(st1.xi.shape).astype(float)
    return ExpectedStatistics(xi=xi, site1=np.zeros(len(GRID)), log_evidence=0.0, n_edges=int(n_edges), n_chains=1), lk


def test_with_little_data_it_pays_for_its_parameters():
    """The held-out gap to the truth, per transition: the variance of a 100-parameter kernel at 1e3 against 1e5 transitions."""
    _, lk = _stats(1.0)
    pi_xi, _ = _stats(1e6)                                         # the true expected counts as the test measure
    gaps = []
    for n in (1e5, 1e3):
        st, _ = _sampled_stats(n)
        k = LogLinearKernel.init(10, 0.3, 1.0)
        for _ in range(10):
            k = k.m_step(st, GRID)
        gaps.append((float((pi_xi.xi * lk).sum()) - float((pi_xi.xi * k.log_transition_matrix(GRID)).sum())) / pi_xi.n_edges)
    assert gaps[1] > 10 * gaps[0] and gaps[0] < 5e-3


def test_it_runs_inside_the_real_em_loop_with_a_monotone_evidence():
    grid, w = make_grid(6.0, 161)
    prior = GaussianAR1(0.7)
    rng = rng_for("test-loglinear-em")
    A = np.stack([prior.sample(rng, 10) for _ in range(40)])
    al, de = alpha_delta(0.5)
    X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)

    def mu(kernel):
        return invariant_log_density_eig(kernel, grid, w).log_density

    k0 = LogLinearKernel.init(8, 0.5, 1.0, half_width=6.0, grid_size=161)
    k, trace, saved = fit_em(k0, grid, w, [(X, al, de)], n_iters=6, tol=1e-9, checkpoints={0, 3, 5}, log_mu=mu)
    ev = np.asarray(trace.log_evidence)
    assert np.all(np.isfinite(ev)) and ev[-1] > ev[0]
    assert trace.monotone_violation < 1e-6 * abs(ev[0])        # exact M-step: no decrease beyond round-off
    assert {0, 3, 5} <= set(saved) and k.theta.size == 64
