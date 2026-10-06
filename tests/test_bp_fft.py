"""FFT-transition BP against dense grid BP and against the exact Gaussian answer.

The FFT path differs from the dense one only by the interpolation that the rescaling a -> rho a needs (bp_fft
docstring), so: (1) its pieces are exact where no interpolation is involved (rho = 1 reduces both steps to a pure
convolution), (2) it converges to dense grid BP as the grid is refined, (3) on the Gaussian chain it converges to the
closed form.
"""
import numpy as np
import pytest

from markov_score.bp_fft import FFTTransition, _cubic_gather, fft_bp_batch, residual_log_density
from markov_score.bp_grid import grid_bp_batch, make_grid
from markov_score.kalman import closed_form_gaussian
from markov_score.noising import alpha_delta
from markov_score.priors import GaussianAR1, LaplaceAR1


def _lags(prior, grid):
    du = grid[1] - grid[0]
    m = np.arange(-(len(grid) - 1), len(grid))
    return residual_log_density(prior, m * du)


def test_residual_log_density_matches_the_kernel():
    prior = LaplaceAR1(0.85)
    d = np.linspace(-3, 3, 13)
    assert np.allclose(residual_log_density(prior, d), -np.abs(d) / prior.b - np.log(2 * prior.b))


def test_cubic_gather_is_exact_on_quadratics_and_at_nodes():
    # Catmull--Rom (Keys a = -1/2) reproduces polynomials up to degree two exactly
    grid, _ = make_grid(4.0, 81)
    f = 0.5 * grid - grid**2 + 2.0
    x = np.linspace(-3.5, 3.5, 57)
    idx, w = _cubic_gather(len(grid), grid[1] - grid[0], 4.0, x)
    assert np.allclose((f[idx] * w).sum(1), 0.5 * x - x**2 + 2.0, atol=1e-10)
    idx, w = _cubic_gather(len(grid), grid[1] - grid[0], 4.0, grid[5:-5])
    assert np.allclose((f[idx] * w).sum(1), f[5:-5], atol=1e-12)


def test_transition_matches_dense_matmul_when_no_interpolation_is_needed():
    # rho = 1: K(u_k|u_j) = p(u_k - u_j), so both steps are pure (Toeplitz) convolutions
    grid, w = make_grid(6.0, 121)
    p = lambda e: np.exp(-0.5 * e**2 / 0.3)
    m = np.arange(-120, 121) * (grid[1] - grid[0])
    tr = FFTTransition(grid, w, np.log(p(m)), rho=1.0)
    rng = np.random.default_rng(0)
    f = rng.random((3, len(grid)))
    K = p(grid[:, None] - grid[None, :])
    assert np.allclose(tr.forward(f), (K @ (f * w).T).T, atol=1e-12)
    assert np.allclose(tr.backward(f), (K.T @ (f * w).T).T, atol=1e-12)


@pytest.mark.parametrize("prior", [GaussianAR1(0.85), LaplaceAR1(0.85)], ids=["gaussian", "laplace"])
def test_converges_to_dense_grid_bp(prior):
    rng = np.random.default_rng(1)
    A = np.stack([prior.sample(rng, 12) for _ in range(24)])
    errs = []
    for n in (101, 201, 401):
        grid, w = make_grid(8.0, n)
        tr = FFTTransition(grid, w, _lags(prior, grid), prior.rho)
        e = []
        for t in (0.1, 1.0):
            al, de = alpha_delta(t)
            X = al * A + np.sqrt(de) * np.random.default_rng(2).standard_normal(A.shape)
            m_fft, _ = fft_bp_batch(grid, w, tr, X, al, de)
            m_dense, _ = grid_bp_batch(grid, w, prior.log_transition_matrix(grid), X, al, de)
            e.append(np.linalg.norm(m_fft - m_dense) / np.linalg.norm(m_dense))
        errs.append(max(e))
    assert errs[0] > errs[1] > errs[2]
    assert errs[2] < 1e-3


def test_gaussian_chain_against_the_closed_form():
    prior = GaussianAR1(0.85)
    rng = np.random.default_rng(3)
    A = np.stack([prior.sample(rng, 16) for _ in range(20)])
    grid, w = make_grid(8.0, 801)
    tr = FFTTransition(grid, w, _lags(prior, grid), prior.rho)
    al, de = alpha_delta(0.5)
    X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
    m_fft, _ = fft_bp_batch(grid, w, tr, X, al, de)
    m_cf, _ = closed_form_gaussian(X, prior.covariance(16), al, de)
    assert np.linalg.norm(m_fft - m_cf) / np.linalg.norm(m_cf) < 1e-4
