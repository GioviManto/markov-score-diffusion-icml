"""Reference grid BP on a chain posterior.

Probability objects
-------------------
The posterior on a chain is

    p_t(a | x)  propto  mu(a_1) prod_i K(a_i | a_{i-1}) prod_i ell_i(a_i; x_i).

BP on this tree is *exact at the level of continuous message functions*; the only
approximation made here is the representation of those functions by their values
on an equally spaced grid u_1..u_{N_g} on [-A, A] with composite trapezoidal
quadrature. Two error sources follow:
  (i)  tail truncation: posterior mass outside [-A, A] is discarded;
  (ii) quadrature: O(dx^2) trapezoid error, which becomes serious only when an
       integrand (in practice the likelihood at small t) is narrower than a few
       grid steps.
Layer-2 experiments quantify both against the exactly solvable Gaussian case.

Messages are kept in the linear domain but renormalized to unit quadrature mass
after every step; the discarded constants are accumulated so that the model
evidence log p_t(x) can also be returned (testable against the closed-form
Gaussian marginal likelihood).

What that returned number is, precisely: a **quadrature estimate of the marginal
evidence under the truncated grid representation**. It used to be documented as
the "exact model evidence", which cannot be right two paragraphs after listing
truncation and O(dx^2) quadrature as error sources. What *is* exact is the
bookkeeping -- no rescaling constant is lost, so the returned value is the
evidence of the finite-state model the grid defines, to floating point. Reserve
"exact" for three things: the analytic Gaussian likelihood, the continuous
functional recursion before discretisation, and finite-state BP once the grid
model has been declared to be the model of interest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from markov_score.backend import to_device


def make_grid(half_width: float, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Equally spaced grid on [-A, A] with composite trapezoidal weights."""
    grid = np.linspace(-half_width, half_width, size)
    dx = float(grid[1] - grid[0])
    weights = np.full(size, dx)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return grid, weights


@dataclass(frozen=True)
class GridBPResult:
    """Posterior summaries from grid BP.

    means, variances : posterior marginal mean / variance per site (exact up to
                       grid error).
    beliefs          : normalized posterior marginals on the grid, shape (L, N_g).
    log_evidence     : log p_t(x) under the truncated grid model -- a quadrature
                       estimate of the continuous evidence, including all
                       constants (likelihood normalizers and message rescalings).
                       Exact for the finite-state model the grid defines; see the
                       module docstring on what "exact" does and does not cover.
    forward, backward: the normalized cavity messages m->_i and m<-_i on the grid, shape
                       (L, N_g) each, present only when `grid_bp(..., return_messages=True)`
                       was asked for and None otherwise.

                       These are the objects the BP literature calls mu_k, and they are
                       returned because a question was asked about their SHAPE, not only
                       about the marginals they produce (docs/JEROME_CALL_2026-09-25.md
                       Q3: "if you look at the messages mu_K, what do they look like?").
                       Exposing them from this recursion is deliberate: the alternative
                       is a second implementation of the same sweep inside an experiment,
                       which is exactly what the module docstring and `backend.py` argue
                       against. m->_i and m<-_i EXCLUDE the local likelihood ell_i, matching
                       the convention in the recursion below.
    """

    means: np.ndarray
    variances: np.ndarray
    beliefs: np.ndarray
    log_evidence: float
    forward: np.ndarray | None = None
    backward: np.ndarray | None = None


def grid_bp(
    grid: np.ndarray,
    weights: np.ndarray,
    log_K: np.ndarray,
    log_ell: np.ndarray,
    x: np.ndarray,
    alpha: float,
    delta: float,
    log_mu: np.ndarray | None = None,
    *,
    return_messages: bool = False,
) -> GridBPResult:
    """Sum-product BP with grid-represented messages on the chain posterior.

    Parameters
    ----------
    log_K   : log K[out, in] = log p(a_out | a_in) on the grid, shape (N_g, N_g).
    log_ell : row-shifted log-likelihood table from `noising.log_likelihood_matrix`,
              shape (L, N_g). Row shifts are undone in the evidence computation.
    x       : observations (needed only to restore the exact evidence constants).
    log_mu  : log initial density on the grid; defaults to standard normal.

    The forward cavity message m->_i (array `fwd`) excludes the local likelihood ell_i; the
    backward one m<-_i (`bwd`) likewise; the site marginal is m->_i ell_i m<-_i (the notation
    of docs/NOTATION.md; the collaborators' note writes alpha_i = m->_i ell_i, beta_i = m<-_i).
    Recursions (trapezoidal quadrature of the exact continuous ones):

        m->_{i+1} = K @ (m->_i * ell_i * w),      m<-_{i-1} = K^T @ (m<-_i * ell_i * w).
    """
    n_sites, n_grid = log_ell.shape
    if log_mu is None:
        log_mu = -0.5 * grid**2 - 0.5 * np.log(2.0 * np.pi)

    K = np.exp(log_K)
    ell = np.exp(log_ell)

    # Evidence bookkeeping: true log ell_i(a) = log_ell[i] + shift_i + gauss_norm,
    # where shift_i is the row max removed in noising.log_likelihood_matrix.
    z = x[:, None] - alpha * grid[None, :]
    row_shift = (-0.5 * z**2 / delta).max(axis=1)
    gauss_norm = -0.5 * np.log(2.0 * np.pi * delta)
    log_const = float(np.sum(row_shift) + n_sites * gauss_norm)

    fwd = np.empty((n_sites, n_grid))
    bwd = np.empty((n_sites, n_grid))
    fwd[0] = np.exp(log_mu)
    log_z_fwd = 0.0  # accumulated log of message rescaling constants

    for i in range(n_sites - 1):
        incoming = fwd[i] * ell[i] * weights
        out = K @ incoming
        c = float(np.sum(out * weights))
        if not np.isfinite(c) or c <= 0.0:
            raise FloatingPointError(f"Forward message {i + 1} lost all mass.")
        fwd[i + 1] = out / c
        log_z_fwd += np.log(c)

    bwd[-1] = np.ones(n_grid)
    for i in range(n_sites - 1, 0, -1):
        incoming = bwd[i] * ell[i] * weights
        out = K.T @ incoming
        c = float(np.sum(out * weights))
        if not np.isfinite(c) or c <= 0.0:
            raise FloatingPointError(f"Backward message {i - 1} lost all mass.")
        bwd[i - 1] = out / c

    beliefs = np.empty((n_sites, n_grid))
    means = np.empty(n_sites)
    variances = np.empty(n_sites)
    for i in range(n_sites):
        raw = fwd[i] * ell[i] * bwd[i]
        mass = float(np.sum(raw * weights))
        if not np.isfinite(mass) or mass <= 0.0:
            raise FloatingPointError(f"Belief {i} lost all mass.")
        b = raw / mass
        beliefs[i] = b
        means[i] = float(np.sum(grid * b * weights))
        variances[i] = float(np.sum((grid - means[i]) ** 2 * b * weights))

    # Evidence: p(x) = (prod of forward rescale constants) * int m->_L ell_L da,
    # where m->_L is the *normalized* forward message at the last site; restore the
    # removed likelihood constants at the end.
    tail = float(np.sum(fwd[-1] * ell[-1] * weights))
    log_evidence = log_z_fwd + np.log(tail) + log_const

    return GridBPResult(
        means=means, variances=variances, beliefs=beliefs, log_evidence=log_evidence,
        # Normalized to unit quadrature mass, as the recursion leaves them, so they are
        # densities on the grid and can be compared with candidate message families
        # directly. bwd[-1] is the constant 1 by construction and is returned as such.
        forward=(fwd.copy() if return_messages else None),
        backward=(bwd.copy() if return_messages else None),
    )


def score_from_posterior_mean(
    x: np.ndarray, means: np.ndarray, alpha: float, delta: float
) -> np.ndarray:
    """Exact OU identity  s_i(x, t) = -(x_i - alpha_t E[a_i | x]) / Delta_t.

    Feeding *approximate* posterior means through this identity produces an
    approximate score whose error is exactly (alpha/delta) times the mean error.
    """
    return -(x - alpha * means) / delta


def grid_bp_batch(
    grid: np.ndarray,
    weights: np.ndarray,
    log_K: np.ndarray,
    X: np.ndarray,
    alpha: float,
    delta: float,
    log_mu: np.ndarray | None = None,
    xp=None,
    *,
    return_evidence: bool = False,
    boundary_frac: float | None = None,
    chunk: int | None = 256,
    return_sign_prob: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized grid BP over a batch of observations X of shape (B, L).

    ``return_sign_prob=True`` returns (means, variances, p_pos) instead, p_pos[b, i] the
    posterior mass of a_i > 0 (a grid point at exactly 0 counts one half), read off the same
    beliefs: E50's commitment of each site to a sign. It cannot be combined with
    ``return_evidence`` or ``boundary_frac``.

    ``chunk`` bounds the rows processed at once (memory knob, default 256): the chains are
    independent, so the recursion is run on consecutive blocks of ``chunk`` rows and the
    outputs concatenated; the result is the same to floating-point round-off
    (``tests/test_bp_grid_chunk.py``). At N_g = 401 grid points and L = 32 sites the
    unchunked pass over a 2048-chain bundle holds about 1 GB of (B, L, N_g) temporaries,
    which is what pushed six concurrent experiment workers into swap on an 8 GB laptop.

    Same algorithm as `grid_bp` (independent chains, one per batch row), with
    the site loop shared and each message update a (N_g, N_g) @ (N_g, B) matmul.
    Used by the reverse-dynamics experiments, where BP must run at every
    integration step for every trajectory. Returns (means, variances), each of
    shape (B, L), or (means, variances, log_evidence) with `return_evidence=True`,
    the third being the per-chain log p_t(x) under the grid model, shape (B,).

    The evidence is the same quantity `grid_bp` returns for a single chain, computed
    the same way -- forward rescaling constants, tail integral, and the restored
    per-(chain, site) likelihood shifts -- and `tests/test_bp_evidence_batch.py`
    asserts the two agree. It is off by default because almost every caller wants
    only the means, and because a per-chain evidence is what the non-Markov
    reference needs to weight a latent variable it is marginalising over.

    ``xp`` selects the array module (numpy, or cupy for GPU execution). This is the *same*
    recursion either way -- there is deliberately no second implementation, because a
    separately written GPU version would put the exactness guarantees at risk in the least
    visible way possible. ``tests/test_backend_parity.py`` asserts the two devices agree to
    machine precision on identical inputs. Inputs are moved to the device here, and the
    result is returned in ``xp``'s own array type; callers that need numpy should use
    ``backend.to_host``.
    """
    if xp is None:
        xp = np
    if return_sign_prob and (return_evidence or boundary_frac):
        raise ValueError("return_sign_prob cannot be combined with return_evidence or boundary_frac.")
    if chunk is not None and X.shape[0] > chunk:
        parts = [grid_bp_batch(grid, weights, log_K, X[s : s + chunk], alpha, delta, log_mu, xp,
                               return_evidence=return_evidence, boundary_frac=boundary_frac, chunk=None,
                               return_sign_prob=return_sign_prob)
                 for s in range(0, X.shape[0], chunk)]
        return tuple(xp.concatenate([pt[i] for pt in parts], axis=0) for i in range(len(parts[0])))
    grid = to_device(grid, xp)
    weights = to_device(weights, xp)
    log_K = to_device(log_K, xp)
    X = to_device(X, xp)

    b_size, n_sites = X.shape
    n_grid = len(grid)
    if log_mu is None:
        log_mu = -0.5 * grid**2 - 0.5 * xp.log(2.0 * xp.pi)
    else:
        log_mu = to_device(log_mu, xp)
    K = xp.exp(log_K)

    # ell[i] has shape (B, N_g): row-shifted per (batch, site). The shifts are kept rather
    # than discarded so the evidence can be restored exactly, as `grid_bp` does.
    z = X[:, :, None] - alpha * grid[None, None, :]
    log_ell = -0.5 * z**2 / delta
    row_shift = log_ell.max(axis=2)  # (B, L)
    log_ell = log_ell - row_shift[:, :, None]
    ell = xp.exp(log_ell)  # (B, L, N_g)

    fwd = xp.empty((n_sites, b_size, n_grid))
    bwd = xp.empty((n_sites, b_size, n_grid))
    fwd[0] = xp.exp(log_mu)[None, :]
    log_z_fwd = xp.zeros(b_size)

    for i in range(n_sites - 1):
        incoming = fwd[i] * ell[:, i, :] * weights[None, :]  # (B, N_g)
        out = (K @ incoming.T).T
        mass = out @ weights
        if not bool(xp.all(xp.isfinite(mass))) or bool(xp.any(mass <= 0.0)):
            raise FloatingPointError(f"Batched forward message {i + 1} lost mass.")
        fwd[i + 1] = out / mass[:, None]
        log_z_fwd = log_z_fwd + xp.log(mass)

    bwd[-1] = 1.0
    for i in range(n_sites - 1, 0, -1):
        incoming = bwd[i] * ell[:, i, :] * weights[None, :]
        out = (K.T @ incoming.T).T
        mass = out @ weights
        if not bool(xp.all(xp.isfinite(mass))) or bool(xp.any(mass <= 0.0)):
            raise FloatingPointError(f"Batched backward message {i - 1} lost mass.")
        bwd[i - 1] = out / mass[:, None]

    means = xp.empty((b_size, n_sites))
    variances = xp.empty((b_size, n_sites))
    bel_edge = xp.zeros((b_size, n_sites)) if boundary_frac else None
    if return_sign_prob:
        p_pos = xp.empty((b_size, n_sites))
        pos_weights = weights * xp.where(grid > 0.0, 1.0, xp.where(grid == 0.0, 0.5, 0.0))
    for i in range(n_sites):
        raw = fwd[i] * ell[:, i, :] * bwd[i]
        mass = raw @ weights
        if not bool(xp.all(xp.isfinite(mass))) or bool(xp.any(mass <= 0.0)):
            raise FloatingPointError(f"Batched belief {i} lost mass.")
        bel = raw / mass[:, None]
        means[:, i] = bel @ (grid * weights)
        variances[:, i] = bel @ (grid**2 * weights) - means[:, i] ** 2
        if return_sign_prob:
            p_pos[:, i] = bel @ pos_weights
        if boundary_frac:
            e = max(1, int(boundary_frac * n_grid))
            bel_edge[:, i] = (bel[:, :e] @ weights[:e]) + (bel[:, -e:] @ weights[-e:])

    if boundary_frac:
        # How much of the mass sits in the outermost `boundary_frac` of the
        # domain -- for the objects the recursion actually propagates.
        #
        # WHY THIS LIVES HERE (round-two review, item 9.1). The convergence tool
        # used to compute its own "boundary mass" as
        #     ell * exp(log_K.max(axis=0))
        # normalised over the grid axis. That is the unary likelihood times the
        # column-wise maximum of the transition matrix. It involves no forward
        # message, no backward message, no recursion, no quadrature weights and
        # no initial law, so whatever it measured, it was not the boundary mass
        # of anything BP propagates -- and the appendix quoted it as such.
        #
        # Computing it inside the recursion is the only way to be sure the
        # diagnostic and the algorithm refer to the same object.
        e = max(1, int(boundary_frac * n_grid))
        fwd_edge = (fwd[:, :, :e] @ weights[:e]) + (fwd[:, :, -e:] @ weights[-e:])
        diag = {
            # L is normalised to unit mass under the quadrature, so these are
            # already fractions.
            "forward_edge_max": float(xp.max(fwd_edge)),
            "forward_edge_mean": float(xp.mean(fwd_edge)),
            "belief_edge_max": float(xp.max(bel_edge)),
            "belief_edge_mean": float(xp.mean(bel_edge)),
            "edge_cells": int(e),
        }
        if return_evidence:
            tail = (fwd[-1] * ell[:, -1, :]) @ weights
            log_const = row_shift.sum(axis=1) + n_sites * (-0.5 * xp.log(2.0 * xp.pi * delta))
            return means, variances, log_z_fwd + xp.log(tail) + log_const, diag
        return means, variances, diag

    if return_evidence:
        tail = (fwd[-1] * ell[:, -1, :]) @ weights
        log_const = row_shift.sum(axis=1) + n_sites * (-0.5 * xp.log(2.0 * xp.pi * delta))
        return means, variances, log_z_fwd + xp.log(tail) + log_const
    if return_sign_prob:
        return means, variances, p_pos
    return means, variances


def grid_bp_projected_batch(
    grid: np.ndarray,
    weights: np.ndarray,
    log_K: np.ndarray,
    X: np.ndarray,
    alpha: float,
    delta: float,
    log_mu: np.ndarray | None = None,
    chunk: int | None = 256,
    *,
    return_diagnostics: bool = False,
    edge_frac: float = 0.05,
) -> tuple:
    """Moment-projected sum-product on the grid: Gaussian closure for ANY kernel.

    The same recursion as `grid_bp_batch`, except that every message is replaced, after each
    transition integral (and the initial law before the first), by the Gaussian on the grid
    with the message's mean and variance -- the projection P_G of Proposition 1.

    Why this exists (1 October 2026). Proposition 1 identifies this algorithm with the LMMSE
    estimator of the covariance twin, but its proof uses the LINEAR transition a' = rho a + eps:
    the first two moments of a convolution depend on the driving noise only through its first two
    moments. For a kernel whose conditional mean is nonlinear (the phi^4 chain, `field.py`),
    the projected message after a transition depends on the incoming message beyond its mean
    and variance, so the identity need not hold and "Gaussian closure" and "the linear floor"
    become two different estimators. This routine measures the first; `theory.lmmse_posterior_mean`
    is the second. On the linear-transition families the two agree to quadrature accuracy,
    which is the test (tests/test_bp_projected.py).

    Returns (means, variances) of the site beliefs m->_i ell_i m<-_i formed from the PROJECTED
    messages, each of shape (B, n); with `return_diagnostics=True` a third value, the largest
    fraction of any projected message's mass in the outermost `edge_frac` of the grid.

    THE GRID CAN DEFEAT THE PROJECTION, AND THE DIAGNOSTIC SAYS WHEN. Moment projection needs
    the message's bulk ON the grid. Forward messages are prior-like and narrow, but a backward
    message is a likelihood in the site variable: for a linear kernel it is centred at
    x/(alpha rho), which at t = 1 and |x| = 2 is 6.5 for rho = 0.85 and leaves [-8, 8] at larger
    t, so its grid moments describe a truncated stump and the projected Gaussian is wrong.
    Found on 1 October 2026 as a 0.3 disagreement with exact BP on the GAUSSIAN chain, where
    the two must coincide; on a grid wide enough for the messages (half-width 32 at t = 1)
    they do, to 1e-8 (tests/test_bp_projected.py).

    ON THE PHI^4 CHAIN THE PROJECTION IS NOT WELL-POSED AT ALL, for the opposite reason. Its
    kernel is N(a'; a, gamma) g(a') / Z(a) with a super-Gaussian tilt g, so from a state far
    outside the bulk the next state lands near a' ~ (a / 4 c gamma)^{1/3} -- it is pulled back
    almost regardless of a. The backward message R(a) = E[ell(a_{i+1}) | a_i = a] is then nearly
    constant across the whole grid (it decays only as exp(-c' a^{2/3}) far beyond it), and the
    Gaussian with its moments is close to flat: the projection discards the backward
    information. Measured: 28% of a projected message's mass in the outer 5% of the grid at
    t = 0.47 on the (401, 8) grid. The "Gaussian closure" arm for this chain is therefore the
    covariance twin (`theory.lmmse_posterior_mean` with the chain's true covariance), which is
    what Proposition 1 proves the projected recursion computes on linear transitions; this
    routine is kept as the DIAGNOSTIC that says so (E32, E33 record its edge mass per cell).
    """
    if chunk is not None and X.shape[0] > chunk:
        parts = [grid_bp_projected_batch(grid, weights, log_K, X[s : s + chunk], alpha, delta,
                                         log_mu, chunk=None, return_diagnostics=return_diagnostics,
                                         edge_frac=edge_frac) for s in range(0, X.shape[0], chunk)]
        out = [np.concatenate([pt[i] for pt in parts], axis=0) for i in range(2)]
        if return_diagnostics:
            out.append(max(pt[2] for pt in parts))
        return tuple(out)
    grid = np.asarray(grid, dtype=float)
    weights = np.asarray(weights, dtype=float)
    K = np.exp(np.asarray(log_K, dtype=float))
    X = np.asarray(X, dtype=float)
    b_size, n_sites = X.shape
    n_grid = len(grid)
    gw = grid * weights
    g2w = grid**2 * weights
    e = max(1, int(edge_frac * n_grid))
    edge_mass = 0.0

    def project(msg):
        """Replace each row by the grid Gaussian with the row's mean and variance."""
        nonlocal edge_mass
        mass = msg @ weights
        if not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
            raise FloatingPointError("Projected message lost all mass.")
        edge_mass = max(edge_mass, float(np.max((msg[:, :e] @ weights[:e] + msg[:, -e:] @ weights[-e:]) / mass)))
        mu = (msg @ gw) / mass
        var = (msg @ g2w) / mass - mu**2
        var = np.maximum(var, 1e-12)
        out = np.exp(-0.5 * (grid[None, :] - mu[:, None]) ** 2 / var[:, None])
        return out / (out @ weights)[:, None]

    if log_mu is None:
        log_mu = -0.5 * grid**2 - 0.5 * np.log(2.0 * np.pi)
    z = X[:, :, None] - alpha * grid[None, None, :]
    log_ell = -0.5 * z**2 / delta
    ell = np.exp(log_ell - log_ell.max(axis=2, keepdims=True))

    fwd = np.empty((n_sites, b_size, n_grid))
    bwd = np.empty((n_sites, b_size, n_grid))
    with np.errstate(divide="ignore"):
        mu0 = np.exp(np.asarray(log_mu, dtype=float))
    fwd[0] = project(np.broadcast_to(mu0, (b_size, n_grid)).copy())
    for i in range(n_sites - 1):
        incoming = fwd[i] * ell[:, i, :] * weights[None, :]
        fwd[i + 1] = project((K @ incoming.T).T)
    bwd[-1] = 1.0
    for i in range(n_sites - 1, 0, -1):
        incoming = bwd[i] * ell[:, i, :] * weights[None, :]
        bwd[i - 1] = project((K.T @ incoming.T).T)

    means = np.empty((b_size, n_sites))
    variances = np.empty((b_size, n_sites))
    for i in range(n_sites):
        raw = fwd[i] * ell[:, i, :] * bwd[i]
        mass = raw @ weights
        if not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
            raise FloatingPointError(f"Projected belief {i} lost mass.")
        bel = raw / mass[:, None]
        means[:, i] = bel @ gw
        variances[:, i] = bel @ g2w - means[:, i] ** 2
    if return_diagnostics:
        return means, variances, edge_mass
    return means, variances
