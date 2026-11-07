"""site_bp_batch: per-link kernels, against grid_bp_batch, em.e_step and the exact non-stationary Gaussian posterior."""
import numpy as np
import pytest

from markov_score.bp_grid import grid_bp_batch, make_grid
from markov_score.em import e_step
from markov_score.priors import LaplaceAR1
from markov_score.site_bp import site_bp_batch


def _normal(grid, var=1.0):
    return -0.5 * grid**2 / var - 0.5 * np.log(2 * np.pi * var)


def test_one_kernel_equals_grid_bp_and_e_step():
    grid, w = make_grid(8.0, 201)
    prior = LaplaceAR1(rho=0.85)
    rng = np.random.default_rng(0)
    X = 0.8 * np.stack([prior.sample(rng, 12) for _ in range(9)]) + 0.6 * rng.standard_normal((9, 12))
    lk = prior.log_transition_matrix(grid)
    r = site_bp_batch(grid, w, lk, X, 0.8, 0.36, _normal(grid), pairs=True, chunk=4)
    m, v = grid_bp_batch(grid, w, lk, X, 0.8, 0.36, log_mu=_normal(grid))
    assert np.abs(r.means - m).max() < 1e-12 and np.abs(r.variances - v).max() < 1e-12
    st = e_step(grid, w, lk, X, 0.8, 0.36, log_mu=_normal(grid))
    assert np.abs(r.xi.sum(0) - st.xi).max() < 1e-10
    assert abs(r.log_evidence.sum() - st.log_evidence) < 1e-9
    assert np.abs(r.site1 - st.site1).max() < 1e-10


def test_nonstationary_gaussian_chain_is_exact():
    grid, w = make_grid(8.0, 401)
    L = 10
    sd = np.linspace(0.7, 1.4, L)
    rho = np.linspace(0.6, 0.95, L - 1)
    alpha_k = rho * sd[1:] / sd[:-1]
    q = sd[1:] ** 2 * (1 - rho**2)
    lks = np.stack([-0.5 * (grid[:, None] - a * grid[None, :]) ** 2 / s - 0.5 * np.log(2 * np.pi * s)
                    for a, s in zip(alpha_k, q)])
    cov = np.empty((L, L))
    for i in range(L):
        for j in range(L):
            lo, hi = min(i, j), max(i, j)
            cov[i, j] = sd[i] * sd[j] * np.prod(rho[lo:hi])
    rng = np.random.default_rng(1)
    A = rng.multivariate_normal(np.zeros(L), cov, size=6)
    al, de = np.exp(-0.2), 1 - np.exp(-0.4)
    X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
    r = site_bp_batch(grid, w, lks, X, al, de, _normal(grid, sd[0] ** 2))
    exact = (al * cov @ np.linalg.solve(al**2 * cov + de * np.eye(L), X.T)).T
    assert np.abs(r.means - exact).max() < 1e-10


def test_per_link_statistics_are_consistent():
    grid, w = make_grid(8.0, 121)
    rng = np.random.default_rng(2)
    L, B = 7, 5
    lks = np.stack([LaplaceAR1(rho=r).log_transition_matrix(grid) for r in np.linspace(0.5, 0.9, L - 1)])
    X = rng.standard_normal((B, L))
    al = np.exp(-rng.uniform(0.1, 1.0, B))
    r = site_bp_batch(grid, w, lks, X, al, 1 - al**2, _normal(grid), pairs=True)
    assert np.allclose(r.xi.sum(axis=(1, 2)), B)                    # each link: one transition per chain
    P_site = np.stack([site_bp_batch(grid, w, lks, X[b:b + 1], al[b], 1 - al[b] ** 2, _normal(grid)).means
                       for b in range(B)])[:, 0]
    assert np.abs(P_site - r.means).max() < 1e-12                   # per-chain levels = one chain at a time
    # the next-site marginal of link k's pair mass is the posterior mass of site k + 1, summed over chains
    lik_free = r.xi[2].sum(axis=1)                                  # sum over u_j -> mass of a_3 on u_i
    mean_from_xi = (lik_free * grid).sum()
    assert mean_from_xi == pytest.approx(r.means[:, 3].sum(), abs=1e-9)
