"""E27's one-sweep samplers must agree with the project's own exact BP.

Jerome's Q8 strong form -- "start from a flat marginal and run BP once" -- is only a
claim about generation if the sampler is exact. These tests pin it against `grid_bp`,
which is already tested independently, rather than against a second implementation of
the same idea.

Four properties, in increasing depth:
  1. the FFBS forward filter IS normalise(grid_bp forward message * local likelihood);
  2. so the last site's filter is the last site's BP belief (nothing lies to its right);
  3. the FFBS *draws* reproduce BP's posterior marginals, site by site;
  4. with a flat likelihood the backward message of a grid-normalised kernel is constant,
     which is why the ancestral arm is the flat-marginal limit of one BP sweep and not a
     different algorithm.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from markov_score.bp_grid import grid_bp, make_grid  # noqa: E402
from markov_score.noising import alpha_delta, log_likelihood_matrix  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.utils import rng_for  # noqa: E402

import e27_direct_sampler as e27  # noqa: E402

RHO = 0.85
L = e27.N_SITES


@pytest.fixture(scope="module")
def setup():
    grid, w = make_grid(8.0, 401)
    prior = LaplaceAR1(RHO)
    log_k = prior.log_transition_matrix(grid)
    return grid, w, prior, log_k, np.exp(log_k)


def _obs(prior, n, alpha, delta, tag):
    rng = rng_for(tag)
    A = np.stack([prior.sample(rng, L) for _ in range(n)])
    return alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)


def _filters(K, grid, w, X, alpha, delta):
    """The forward filter e27.ffbs_grid computes, isolated so it can be checked directly."""
    z = X[:, :, None] - alpha * grid[None, None, :]
    le = -0.5 * z**2 / delta
    ell = np.exp(le - le.max(axis=2, keepdims=True))
    log_mu = -0.5 * grid**2 - 0.5 * np.log(2 * np.pi)
    F = np.empty((len(X), L, len(grid)))
    f = np.exp(log_mu - log_mu.max())[None, :] * ell[:, 0]
    F[:, 0] = f / (f * w).sum(axis=1, keepdims=True)
    for i in range(1, L):
        f = (F[:, i - 1] * w) @ K.T * ell[:, i]
        F[:, i] = f / (f * w).sum(axis=1, keepdims=True)
    return F


@pytest.mark.parametrize("t", [0.1, 0.5, 2.0])
def test_forward_filter_matches_grid_bp(setup, t):
    grid, w, prior, log_k, K = setup
    al, de = alpha_delta(t)
    X = _obs(prior, 4, al, de, f"ffbs-filter-{t}")
    F = _filters(K, grid, w, X, al, de)
    worst = 0.0
    for b in range(len(X)):
        log_ell = log_likelihood_matrix(grid, X[b], al, de)
        r = grid_bp(grid, w, log_k, log_ell, X[b], al, de, return_messages=True)
        ell = np.exp(log_ell)
        for i in range(L):
            filt = r.forward[i] * ell[i]
            filt /= (filt * w).sum()
            worst = max(worst, float(np.abs(filt - F[b, i]).max()))
    assert worst < 1e-12, f"filter disagrees with grid_bp by {worst:.3e}"


@pytest.mark.parametrize("t", [0.1, 0.5, 2.0])
def test_last_filter_is_last_belief(setup, t):
    """No message arrives from the right of the final site, so its filter is its belief."""
    grid, w, prior, log_k, K = setup
    al, de = alpha_delta(t)
    X = _obs(prior, 3, al, de, f"ffbs-last-{t}")
    F = _filters(K, grid, w, X, al, de)
    for b in range(len(X)):
        r = grid_bp(grid, w, log_k, log_likelihood_matrix(grid, X[b], al, de), X[b], al, de)
        assert np.abs(F[b, -1] - r.beliefs[-1]).max() < 1e-12


def test_ffbs_draws_reproduce_bp_marginals(setup):
    """The DRAWS, not just the messages: 20000 sweeps of one fixed observation, against
    the posterior mean and variance BP reports at every site.

    Tolerance is the Monte Carlo standard error of the mean of 20000 draws, times four.
    It is computed from BP's own posterior variance rather than hardcoded, so the test
    does not silently loosen if the configuration changes."""
    grid, w, prior, log_k, K = setup
    t = 0.4
    al, de = alpha_delta(t)
    X1 = _obs(prior, 1, al, de, "ffbs-draws")
    r = grid_bp(grid, w, log_k, log_likelihood_matrix(grid, X1[0], al, de), X1[0], al, de)

    n = 20_000
    Xr = np.repeat(X1, n, axis=0)
    A = e27.ffbs_grid(K, grid, w, Xr, al, de,
                      rng_for("ffbs-draw-idx"), rng_for("ffbs-draw-jit"), jitter=True)
    se = np.sqrt(r.variances / n)
    assert np.abs(A.mean(axis=0) - r.means).max() < 4 * se.max(), (
        f"means off by {np.abs(A.mean(axis=0) - r.means).max():.4f}, 4 s.e. = {4*se.max():.4f}")
    # Variance: the jittered draw adds the within-cell uniform variance h^2/12.
    h = float(grid[1] - grid[0])
    assert np.abs(A.var(axis=0) - (r.variances + h**2 / 12.0)).max() < 0.02


def test_flat_likelihood_backward_message_is_constant():
    """Why the ancestral arm IS one BP sweep, not a different algorithm: the backward
    recursion R_{i-1} = K^T (R_i * w) has R == const as its fixed point exactly when the
    grid kernel integrates to one in its output argument.

    It does NOT hold to machine precision, and the first version of this test asserted
    1e-9 and failed. The trapezoid rule has a kink to integrate at a' = rho a (the Laplace
    innovation's), so the column mass is 1 + O(h^2): measured 3.8e-3 at N_g = 201 and
    9.6e-4 at 401, a ratio of 4.0. What the test therefore pins is the CONVERGENCE, which
    is the property the ancestral arm needs, rather than a tolerance chosen to pass.

    It also separates the grid's two errors, which behave differently and are worth having
    on the record:
      * quadrature -- refining h drives the interior non-constancy down at h^2
        (4.0e-3, 9.8e-4, 2.4e-4, 6.4e-5 at N_g = 201, 401, 801, 1601);
      * truncation -- over the FULL grid it saturates at 2.1e-2 and does not improve,
        because at |a| ~ A = 8 the kernel loses mass off the edge and the backward message
        really is not constant there. That region carries no prior mass (|a| > 5 is beyond
        five standard deviations of a unit-variance chain), which is why the ancestral arm
        is unaffected, but it is a truncation effect and refining N_g will never remove it.
    """
    rows = []
    for Ng in (201, 401, 801):
        grid, w = make_grid(8.0, Ng)
        K = np.exp(LaplaceAR1(RHO).log_transition_matrix(grid))
        interior = np.abs(grid) < 5.0
        col_mass_err = float(np.abs((K * w[:, None]).sum(axis=0)[interior] - 1.0).max())
        R = np.ones(Ng)
        for _ in range(L - 1):
            out = K.T @ (R * w)
            R = out / (out * w).sum()
        rows.append((Ng, col_mass_err,
                     float(np.abs(R[interior] / R[interior].mean() - 1.0).max()),
                     float(np.abs(R / R.mean() - 1.0).max())))

    for i in range(1, len(rows)):
        for col in (1, 2):                      # column mass, and interior non-constancy
            ratio = rows[i - 1][col] / rows[i][col]
            assert 3.0 < ratio < 5.0, (
                f"expected second-order convergence (ratio ~4) for column {col} between "
                f"N_g={rows[i-1][0]} and {rows[i][0]}, got {ratio:.2f}: {rows}")
    # Truncation does not converge; assert it is bounded and stays put, so a future change
    # that made it grow would be caught.
    assert all(r[3] < 0.05 for r in rows), rows
    assert rows[-1][3] / rows[0][3] > 1.0, "full-grid non-constancy is truncation, not quadrature"
