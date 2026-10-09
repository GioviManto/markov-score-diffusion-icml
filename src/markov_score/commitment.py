"""Commitment of the exact denoiser to the sign of each site: "speciation" on the phi^4 chain (E50).

Jerome's email of 30 Sep 2026 and the consolidated note (section 19) ask to watch the denoiser commit
to whole domains as the corruption level falls. For a noisy sequence x_t the exact posterior gives each
clean site a probability p_i = P(a_i > 0 | x_t) (`bp_grid.grid_bp_batch(..., return_sign_prob=True)`).
The site's COMMITMENT is |2 p_i - 1|: 0 when the posterior is undecided between the two wells, 1 when it
has chosen one. Averaged over sites and data it falls from 1 (no noise) to 0 (pure noise, by the
reflection symmetry), and the level t_c where it crosses 1/2 is the commitment time.

For an exact (calibrated) posterior the probability of guessing the sign right is (1 + commitment) / 2 on
average, which gives a check of the whole machinery (E50 P1).

The GAUSSIAN TWIN is the zero-mean Gaussian chain with the same covariance. Its posterior of a_i is
N(mu_i, v_i) with mu_i ~ N(0, Sigma_ii - v_i) over the data, so with R_i^2 = 1 - v_i / Sigma_ii the
commitment E|2 Phi(mu_i / sqrt(v_i)) - 1| = (2 / pi) arctan(R_i / sqrt(1 - R_i^2)) = (2 / pi) arcsin(R_i),
in closed form (E|2 Phi(sZ) - 1| = (2 / pi) arctan(s) for Z ~ N(0, 1)). The twin commits at t_c exactly
where the posterior variance is half the prior one. Comparing the two answers whether commitment on
the field is a second-order phenomenon or uses the bimodality.
"""
from __future__ import annotations

import numpy as np


def commitment(p_pos: np.ndarray) -> np.ndarray:
    """|2 p - 1| elementwise: 0 undecided, 1 committed."""
    return np.abs(2.0 * np.asarray(p_pos, dtype=float) - 1.0)


def sign_correct(p_pos: np.ndarray, a: np.ndarray) -> np.ndarray:
    """1 where the posterior's more likely sign is the true one, 0 where not, 1/2 on an exact tie."""
    p_pos = np.asarray(p_pos, dtype=float)
    a = np.asarray(a, dtype=float)
    guess_pos = p_pos > 0.5
    guess_neg = p_pos < 0.5
    return np.where(guess_pos, (a > 0).astype(float), np.where(guess_neg, (a < 0).astype(float), 0.5))


def gaussian_twin(cov: np.ndarray, alpha: float, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-site commitment and posterior variance of the zero-mean Gaussian chain with covariance `cov`,
    observed as x = alpha a + sqrt(delta) z. Returns (commitment, posterior variance), each of shape (L,)."""
    cov = np.asarray(cov, dtype=float)
    L = cov.shape[0]
    gain = np.linalg.solve(alpha**2 * cov + delta * np.eye(L), cov)      # (alpha^2 S + delta I)^{-1} S
    post_var = np.diag(cov) - alpha**2 * np.einsum("ij,ji->i", cov, gain)
    r2 = np.clip(1.0 - post_var / np.diag(cov), 0.0, 1.0)
    return (2.0 / np.pi) * np.arcsin(np.sqrt(r2)), post_var


def crossing_time(ts, values, level: float = 0.5) -> float:
    """The level t at which a decreasing curve `values(t)` first falls through `level`, scanning from the
    smallest t; linear interpolation in log t between the two bracketing levels. NaN if it never does
    (the curve starts below the level or ends above it)."""
    ts = np.asarray(ts, dtype=float)
    v = np.asarray(values, dtype=float)
    order = np.argsort(ts)
    ts, v = ts[order], v[order]
    if not (v[0] >= level):
        return float("nan")
    for k in range(len(ts) - 1):
        if v[k] >= level > v[k + 1]:
            lo, hi = np.log(ts[k]), np.log(ts[k + 1])
            frac = (v[k] - level) / (v[k] - v[k + 1])
            return float(np.exp(lo + frac * (hi - lo)))
    return float("nan")


def bulk_sites(n_sites: int, half_max: int = 8) -> np.ndarray:
    """The central sites of a chain: 2 * min(n_sites // 4, half_max) of them (at least 2), away from the ends."""
    half = max(1, min(n_sites // 4, half_max))
    mid = n_sites // 2
    return np.arange(mid - half, mid + half)
