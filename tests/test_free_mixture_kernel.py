"""The free 2-D Gaussian-mixture kernel of E60: columns are densities of the discretised model, the start is the AR(1) the
other EM arms start from, one gate for every component is exactly the shift mixture, an M-step never lowers Q and never
mutates its input, the M-step recovers a kernel of its own family, and a rejected M-step drops its Adam state."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from markov_score.bp_grid import make_grid  # noqa: E402
from markov_score.em import ExpectedStatistics  # noqa: E402
from markov_score.free_mixture_kernel import FreeMixtureKernel  # noqa: E402
from markov_score.kernels import MixtureInnovationKernel  # noqa: E402
from markov_score.stationary import invariant_log_density_eig  # noqa: E402

A_HALF = 6.0


def _grid(n=81):
    return make_grid(A_HALF, n)


def _normalised(lk, w):
    return lk - np.log(np.exp(lk).T @ w)[None, :]


def _q(stats, k, grid):
    return float(np.sum(stats.xi * k.log_transition_matrix(grid)) / stats.n_edges)


def _stats_from(lk, grid, w, n_edges=10**5):
    p_in = np.exp(-0.5 * grid ** 2) * w                       # columns weighted by N(0, 1) on the grid
    xi = np.exp(lk) * w[:, None] * (p_in / p_in.sum())[None, :] * n_edges
    return ExpectedStatistics(xi=xi, site1=np.ones(len(grid)), log_evidence=0.0, n_edges=n_edges, n_chains=1)


def test_columns_are_densities_and_the_start_is_the_ar1():
    grid, w = _grid()
    k = FreeMixtureKernel.init(4, rho=0.3, var=0.8)
    lk = k.log_transition_matrix(grid)
    np.testing.assert_allclose(np.exp(lk).T @ w, 1.0, atol=1e-12)
    ar1 = _normalised(-0.5 * (grid[:, None] - 0.3 * grid[None, :]) ** 2 / 0.8, w)
    np.testing.assert_allclose(lk, ar1, atol=1e-10)
    assert k.n_params == k.theta.size == 24
    np.testing.assert_allclose(k.slopes, 0.3)


def test_one_gate_for_every_component_is_the_shift_mixture():
    grid, w = _grid()
    rho, pi, m, s2 = 0.7, np.array([0.2, 0.5, 0.3]), np.array([-1.0, 0.0, 0.8]), np.array([0.3, 0.1, 0.5])
    shift = MixtureInnovationKernel(rho=rho, pi=pi, mu=m, s2=s2)
    l11 = 1.7                                                 # any common gate width: the gate then cancels
    free = FreeMixtureKernel(logit=np.log(pi), mu=np.stack([np.zeros(3), m], 1),
                             chol=np.stack([np.full(3, np.log(l11)), np.full(3, rho * l11), 0.5 * np.log(s2)], 1))
    np.testing.assert_allclose(free.log_transition_matrix(grid), _normalised(shift.log_transition_matrix(grid), w),
                               atol=1e-10)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_m_step_never_lowers_q_and_returns_a_new_kernel(seed):
    grid, _ = _grid()
    rng = np.random.default_rng(seed)
    xi = rng.gamma(1.0, size=(len(grid), len(grid))) * np.exp(-0.5 * np.subtract.outer(grid, 0.8 * grid) ** 2)
    xi *= 1000.0 / xi.sum()
    stats = ExpectedStatistics(xi=xi, site1=np.ones(len(grid)), log_evidence=0.0, n_edges=1000, n_chains=10)
    k = FreeMixtureKernel.init(4)
    theta0 = k.theta.copy()
    ks = [k]
    for _ in range(8):
        ks.append(ks[-1].m_step(stats, grid))
    np.testing.assert_array_equal(k.theta, theta0)            # the input is never mutated
    qs = [_q(stats, kk, grid) for kk in ks]
    assert all(b >= a - 1e-12 for a, b in zip(qs, qs[1:]))
    assert qs[-1] > qs[0] + 0.05                               # and it actually learns
    assert len({kk.theta.tobytes() for kk in ks}) == len(ks)   # every iterate is its own snapshot


def test_the_m_step_recovers_a_two_line_kernel_and_has_an_invariant_law():
    # two gated lines with different slopes: out of every shift class
    grid, w = _grid()
    true = FreeMixtureKernel(logit=np.zeros(2), mu=np.array([[-1.0, -0.9], [1.0, 0.9]]),
                             chol=np.array([[np.log(0.8), 0.8 * 0.6, np.log(0.45)], [np.log(0.8), -0.8 * 0.4, np.log(0.45)]]))
    stats = _stats_from(true.log_transition_matrix(grid), grid, w)
    k = FreeMixtureKernel.init(2, lr=5e-2, n_inner=20)
    q_true, q0 = _q(stats, true, grid), _q(stats, k, grid)
    for _ in range(150):
        k = k.m_step(stats, grid)
    gap = q_true - _q(stats, k, grid)                          # the KL per edge from the true kernel
    assert gap < 0.02 * (q_true - q0)
    assert gap < 5e-3
    d = invariant_log_density_eig(k, grid, w)
    np.testing.assert_allclose(np.exp(d.log_density) @ w, 1.0, atol=1e-8)


def test_a_rejected_m_step_drops_its_adam_state():
    grid, w = _grid()
    stats = _stats_from(-0.5 * (grid[:, None] - 0.8 * grid[None, :]) ** 2 / 0.36, grid, w)
    k = FreeMixtureKernel.init(3).m_step(stats, grid).m_step(stats, grid)
    params = k._tensors(requires_grad=True)
    (-(torch.as_tensor(stats.xi) * k._log_k(params, grid)).sum()).backward()
    state = copy.deepcopy(k.opt_state)
    for i, p in enumerate(params):                             # Adam moments pointing DOWNHILL in Q
        state["state"][i]["exp_avg"] = -1e3 * p.grad.sign()
        state["state"][i]["exp_avg_sq"] = torch.full_like(p, 1e6)
    poisoned = replace(k, opt_state=state)
    k1 = replace(poisoned, max_halvings=0).m_step(stats, grid)
    np.testing.assert_array_equal(k1.theta, k.theta)
    assert k1.opt_state is None
    assert _q(stats, k1.m_step(stats, grid), grid) > _q(stats, k, grid)
    assert _q(stats, poisoned.m_step(stats, grid), grid) > _q(stats, k, grid)   # a fresh retry climbs at once
