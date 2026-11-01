"""Diagnostics of fitted field kernels that the likelihood cannot see (N-33).

`invariant_mass_out(row)`: the mass that a fitted kernel's own invariant law puts beyond the data
range |a| > EDGE. The estimator uses that law as the law of the first site, and a law parked outside
the data drags every posterior with it. On the 1 October previews it separated the five failed
Legendre-tilt fits (mass ~1) from the 63 others (mass < 1.1e-4) without exception; the coefficient
bound did not. A row is reported as PARKED when the mass exceeds PARKED (0.5).

The law measured is the one the ESTIMATOR used for the first site, computed by the same function
(e33 `_stationary_mu`): the direct eigensolve, or the standard normal when the eigensolve raises.
On 3 Oct 2026 it raised on 20 of 768 confirmatory clean Legendre fits, and a closed-form substitute
was tried and rejected: a fitted tilted walk can be REDUCIBLE on the grid (a bulk and a pocket beyond
the data with no transition between them, two eigenvalues equal to one), so its invariant law is not
unique. The eigensolve picks one closed class, pi = g Z a particular mixture of the two, and on one
preview fit they disagree (0 against 0.0225 beyond |a| = 3.5) while both are exactly invariant.
`annotate` therefore also reports `eig_failed` (the estimator fell back to N(0, 1)) and
`gz_mass_out` (the mass of pi = g Z, information only).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

EDGE = 3.5
PARKED = 0.5
KERNEL_ARMS = ("em_tilted_poly", "em_tilted_spline", "em_pair_quartic", "em_mixture", "em_mdn")


def _kernel(row, grid, cfg=None):
    import e33_field_learning as e33
    from e34_field_physics import kernel_from_row
    return kernel_from_row(row, cfg or dict(e33.SETTINGS), grid)


def eig_fails(row, grid, w, cfg=None) -> bool:
    """True when the direct eigensolve of the fitted kernel raises (the estimator then used N(0, 1))."""
    from markov_score.stationary import invariant_log_density_eig
    try:
        invariant_log_density_eig(_kernel(row, grid, cfg), grid, w)
    except (RuntimeError, FloatingPointError, ValueError):
        return True
    return False


def _mass(pi, grid, w, edge):
    return float((pi * w)[np.abs(grid) > edge].sum())


def invariant_mass_out(row, grid, w, cfg=None, edge: float = EDGE) -> float:
    """Mass beyond |a| > edge of the first-site law the estimator used: the eigensolve's invariant
    law, or the standard normal where the eigensolve raises (e33 `_stationary_mu`, verbatim)."""
    from markov_score.stationary import invariant_log_density_eig
    try:
        log_pi = invariant_log_density_eig(_kernel(row, grid, cfg), grid, w).log_density
    except (FloatingPointError, RuntimeError, ValueError):
        log_pi = -0.5 * grid**2 - 0.5 * np.log(2.0 * np.pi)
    return _mass(np.exp(log_pi), grid, w, edge)


def gz_mass_out(row, grid, w, cfg=None, edge: float = EDGE) -> float:
    """Mass beyond |a| > edge of pi = g Z / int g Z (tilted walks only; NaN otherwise). Information
    only: for a kernel that is reducible on the grid this is one invariant law among several."""
    k = _kernel(row, grid, cfg)
    if not hasattr(k, "log_stationary"):
        return float("nan")
    lp = k.log_stationary(grid)
    pi = np.exp(lp - lp.max())
    return _mass(pi / float((pi * w).sum()), grid, w, edge)


def annotate(df, grid, w):
    """Add `invariant_mass_out` and `parked` to every kernel row with a `theta` (others get NaN)."""
    import pandas as pd
    out = []
    for _, r in df.iterrows():
        ok = r.get("arm") in KERNEL_ARMS and isinstance(r.get("theta"), str) and r["theta"]
        out.append(invariant_mass_out(r, grid, w) if ok else np.nan)
    df = df.copy()
    df["invariant_mass_out"] = out
    has = [bool(r.get("arm") in KERNEL_ARMS and isinstance(r.get("theta"), str) and r["theta"]) for _, r in df.iterrows()]
    df["eig_failed"] = [eig_fails(r, grid, w) if h else False for (_, r), h in zip(df.iterrows(), has)]
    df["gz_mass_out"] = [gz_mass_out(r, grid, w) if h else np.nan for (_, r), h in zip(df.iterrows(), has)]
    df["parked"] = pd.Series(out, index=df.index) > PARKED
    return df
