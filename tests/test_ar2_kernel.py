"""The two-lag mixture kernel of E62: phi2 = 0 is the shift mixture, columns are densities, the M-step never lowers the
grid-normalised Q and never mutates its input, it recovers (phi1, phi2) from exact triple counts, and EM through
pair-state BP moves phi towards the truth from noisy data."""

from __future__ import annotations

import numpy as np

from markov_score.ar2_kernel import Ar2MixtureKernel, fit_pair_em
from markov_score.bp_grid import make_grid
from markov_score.kernels import MixtureInnovationKernel
from markov_score.pair_bp import stationary_pair


def _truth():
    return Ar2MixtureKernel(phi=np.array([0.51, 0.4]), pi=np.array([0.5, 0.5]), mu=np.array([-0.25, 0.25]),
                            s2=np.array([0.12, 0.12]))


def _triples(k, grid, w, n=10**5):
    lk2 = k.log_k2(grid, w)
    mu2 = stationary_pair(lk2, w)                              # [j = a_{t-1}, i = a_{t-2}]
    xi = np.exp(lk2) * mu2[None] * w[:, None, None] * w[None, :, None] * w[None, None, :]
    return xi / xi.sum() * n


def test_phi2_zero_is_the_shift_mixture_and_columns_are_densities():
    grid, w = make_grid(5.0, 41)
    m = MixtureInnovationKernel(rho=0.7, pi=np.array([0.3, 0.7]), mu=np.array([-0.4, 0.2]), s2=np.array([0.2, 0.1]))
    k = Ar2MixtureKernel(phi=np.array([0.7, 0.0]), pi=m.pi, mu=m.mu, s2=m.s2)
    lk2 = k.log_k2(grid)
    np.testing.assert_allclose(lk2, np.repeat(m.log_transition_matrix(grid)[:, :, None], len(grid), axis=2), atol=1e-12)
    np.testing.assert_allclose(np.einsum("kji,k->ji", np.exp(k.log_k2(grid, w)), w), 1.0, atol=1e-12)


def test_the_m_step_never_lowers_q_and_recovers_phi():
    grid, w = make_grid(5.0, 41)
    xi = _triples(_truth(), grid, w)
    k = Ar2MixtureKernel.init(2, rho=0.3, var=0.8, rng=np.random.default_rng(0))
    theta0 = k.theta.copy()
    ks = [k]
    for _ in range(150):
        ks.append(ks[-1].m_step(xi, grid, w))
    np.testing.assert_array_equal(k.theta, theta0)
    qs = [kk.q(xi, grid, w) for kk in ks]
    assert all(b >= a - 1e-12 for a, b in zip(qs, qs[1:]))
    np.testing.assert_allclose(ks[-1].phi, [0.51, 0.4], atol=0.02)


def test_em_through_pair_state_bp_moves_phi_towards_the_truth():
    grid, w = make_grid(5.0, 31)
    rng = np.random.default_rng(1)
    phi1, phi2 = 0.51, 0.4
    A = np.zeros((400, 600 + 12))
    e = rng.laplace(0, np.sqrt((1 - 0.85 ** 2) * (1 - phi2 ** 2) / 2), A.shape)
    for t in range(2, A.shape[1]):
        A[:, t] = phi1 * A[:, t - 1] + phi2 * A[:, t - 2] + e[:, t]
    A = A[:, 600:]
    groups = []
    for part, (al, de) in zip(np.array_split(A, 2), ((0.95, 0.0975), (0.8, 0.36))):
        groups.append((al * part + np.sqrt(de) * rng.standard_normal(part.shape), al, de))
    k0 = Ar2MixtureKernel.init(3, rho=0.3, var=0.8, rng=np.random.default_rng(2))
    saved, sec = fit_pair_em(k0, grid, w, groups, ladder={0, 30})
    assert abs(saved[30].phi[1] - phi2) < abs(saved[0].phi[1] - phi2) - 0.2
    assert sec > 0
