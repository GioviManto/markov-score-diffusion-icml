"""The shift families of E56: K(a'|a) = phi(a' - alpha a) with log phi in four bases at 25 parameters. The bases agree
with the package's numpy ones where those exist and are held constant beyond W; columns are densities of the discretised
model; the start is E30's Gaussian AR(1); an M-step never lowers Q and never mutates its input; EM through noise is
monotone; on the exact pair law of the Laplace chain the fit finds the true alpha; a kernel is rebuilt from its theta."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from markov_score.bp_grid import make_grid  # noqa: E402
from markov_score.em import ExpectedStatistics, fit_em  # noqa: E402
from markov_score.kernels import LaplaceAR1Kernel  # noqa: E402
from markov_score.shift_kernel import BASES, ShiftKernel, basis_matrix  # noqa: E402
from markov_score.tilted_kernel import design_matrix  # noqa: E402

W = 4.0


def _grid(n=101, half_width=8.0):
    return make_grid(half_width, n)


def _np(basis, x):
    return basis_matrix(basis, torch.as_tensor(x, dtype=torch.float64), W).numpy()


@pytest.mark.parametrize("basis", ["spline", "legendre"])
def test_the_bases_match_the_package_numpy_ones(basis):
    x = np.linspace(-6.0, 6.0, 997)
    ref = design_matrix("bspline" if basis == "spline" else "legendre", 23, W, x)
    np.testing.assert_allclose(_np(basis, x), ref, atol=1e-12)


def test_the_fast_spline_score_is_the_basis_contracted_with_its_coefficients():
    from markov_score.shift_kernel import _spline_score
    rng = np.random.default_rng(1)
    eps = torch.as_tensor(np.concatenate([rng.uniform(-6, 6, 4000), [-W, W, -9.0, 9.0, 0.0]]))
    for _ in range(3):
        c = torch.as_tensor(rng.standard_normal(23))
        ref = basis_matrix("spline", eps, W) @ c
        np.testing.assert_allclose(_spline_score(eps, c, W).numpy(), ref.numpy(), atol=1e-11)
    # and its derivative in eps (what reaches alpha) is the basis's own
    e = eps.clone().requires_grad_(True)
    _spline_score(e, c, W).sum().backward()
    e2 = eps.clone().requires_grad_(True)
    (basis_matrix("spline", e2, W) @ c).sum().backward()
    inside = (eps.abs() < W - 1e-6).numpy()
    np.testing.assert_allclose(e.grad.numpy()[inside], e2.grad.numpy()[inside], atol=1e-9)


def test_fourier_and_rbf_are_what_the_registry_says():
    x = np.linspace(-W, W, 41)
    F = _np("fourier", x)
    k = np.arange(1, 13)
    np.testing.assert_allclose(F[:, :12], np.cos(np.pi * np.outer(x, k) / W), atol=1e-14)
    np.testing.assert_allclose(F[:, 12:], np.sin(np.pi * np.outer(x, np.arange(1, 12)) / W), atol=1e-14)
    R = _np("rbf", x)
    mu = np.linspace(-W, W, 23)
    ell = mu[1] - mu[0]
    np.testing.assert_allclose(R, np.exp(-0.5 * (x[:, None] - mu[None, :]) ** 2 / ell**2), atol=1e-14)


@pytest.mark.parametrize("basis", BASES)
def test_every_basis_has_23_functions_held_constant_beyond_w(basis):
    B = _np(basis, np.array([-9.0, -W, W * (1 - 1e-12), 9.0]))
    assert B.shape == (4, 23)
    np.testing.assert_allclose(B[0], B[1], atol=1e-10)
    np.testing.assert_allclose(B[3], B[2], atol=1e-10)


@pytest.mark.parametrize("basis", BASES)
def test_columns_are_densities_and_the_start_is_e30s_gaussian(basis):
    grid, w = _grid()
    k = ShiftKernel.init(basis, rho=0.3, var=0.8, half_width_basis=W)
    assert k.n_params == 25 and k.theta.size == 25
    lk = k.log_transition_matrix(grid)
    np.testing.assert_allclose(np.exp(lk).T @ w, 1.0, atol=1e-12)
    g = -0.5 * (grid[:, None] - 0.3 * grid[None, :]) ** 2 / 0.8
    g -= np.log(np.exp(g).T @ w)[None, :]
    np.testing.assert_allclose(lk, g, atol=1e-10)


def _laplace_pair_stats(grid, w, n_edges=4000, rho=0.85):
    """The EXACT pair law of the stationary Laplace chain on the grid, as expected counts (clean-data statistics)."""
    b = np.sqrt((1 - rho**2) / 2)
    lk = LaplaceAR1Kernel(rho=rho, b=b).log_transition_matrix(grid)
    lk -= np.log(np.exp(lk).T @ w)[None, :]
    P = np.exp(lk) * w[:, None]                          # P[k, j] = Prob(u_j -> u_k)
    vals, vecs = np.linalg.eig(P)
    pi = np.abs(np.real(vecs[:, np.argmax(np.real(vals))]))
    pi /= pi.sum()
    xi = P * pi[None, :] * n_edges
    return ExpectedStatistics(xi=xi, site1=pi * 10, log_evidence=0.0, n_edges=n_edges, n_chains=10)


@pytest.mark.parametrize("basis", BASES)
def test_m_step_never_lowers_q_and_returns_a_new_kernel(basis):
    grid, w = _grid()
    stats = _laplace_pair_stats(grid, w)
    k = ShiftKernel.init(basis, rho=0.3, var=0.8, half_width_basis=W)
    theta0 = k.theta.copy()
    ks = [k]
    for _ in range(6):
        ks.append(ks[-1].m_step(stats, grid))
    np.testing.assert_array_equal(k.theta, theta0)
    qs = [float(np.sum(stats.xi * kk.log_transition_matrix(grid))) for kk in ks]
    assert all(b >= a - 1e-9 for a, b in zip(qs, qs[1:]))
    assert qs[-1] > qs[0] + 1.0
    assert len({kk.theta.tobytes() for kk in ks}) == len(ks)


@pytest.mark.parametrize("basis", BASES)
def test_on_the_exact_laplace_pair_law_the_fit_finds_alpha(basis):
    grid, w = _grid(161)
    stats = _laplace_pair_stats(grid, w)
    k = ShiftKernel.init(basis, rho=0.3, var=0.8, half_width_basis=W, lr=5e-2)
    for _ in range(150):
        k = k.m_step(stats, grid)
    assert abs(k.alpha - 0.85) < 0.02


def test_em_through_noise_is_monotone_with_a_fixed_initial_law():
    grid, w = _grid(81, 6.0)
    rng = np.random.default_rng(0)
    L, rho, n = 12, 0.85, 240
    b = np.sqrt((1 - rho**2) / 2)
    A = np.zeros((n, L))
    A[:, 0] = rng.laplace(0.0, np.sqrt(0.5), n)
    for i in range(1, L):
        A[:, i] = rho * A[:, i - 1] + rng.laplace(0.0, b, n)
    groups = []
    for t, part in zip((0.1, 0.5, 1.0), np.array_split(A, 3)):
        al, de = np.exp(-t), 1 - np.exp(-2 * t)
        groups.append((al * part + np.sqrt(de) * rng.standard_normal(part.shape), al, de))
    k0 = ShiftKernel.init("spline", rho=0.3, var=0.8, half_width_basis=W)
    k, trace = fit_em(k0, grid, w, groups, n_iters=25, log_mu=-0.5 * grid**2 - 0.5 * np.log(2 * np.pi), tol=0.0)
    assert trace.monotone_violation == 0.0
    assert trace.log_evidence[-1] > trace.log_evidence[0]
    assert 0.6 < k.alpha < 1.0


@pytest.mark.parametrize("basis", BASES)
def test_a_kernel_is_rebuilt_exactly_from_its_saved_theta(basis):
    grid, w = _grid()
    stats = _laplace_pair_stats(grid, w)
    k = ShiftKernel.init(basis, rho=0.3, var=0.8, half_width_basis=W)
    for _ in range(3):
        k = k.m_step(stats, grid)
    k2 = ShiftKernel.from_theta(basis, k.theta, half_width_basis=W)
    np.testing.assert_allclose(k2.log_transition_matrix(grid), k.log_transition_matrix(grid), atol=1e-12)
    eps = np.linspace(-3, 3, 7)
    lp = k.log_phi(eps)
    assert lp.shape == (7,) and np.all(np.isfinite(lp))
