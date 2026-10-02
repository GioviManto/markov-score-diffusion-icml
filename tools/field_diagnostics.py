"""Diagnostics of fitted field kernels that the likelihood cannot see (N-33).

`invariant_mass_out(row)`: the mass that a fitted kernel's own invariant law puts beyond the data
range |a| > EDGE. The estimator uses that law as the law of the first site, and a law parked outside
the data drags every posterior with it. On the 1 October previews it separated the five failed
Legendre-tilt fits (mass ~1) from the 63 others (mass < 1.1e-4) without exception; the coefficient
bound did not. A row is reported as PARKED when the mass exceeds PARKED (0.5).
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


def invariant_mass_out(row, grid, w, cfg=None, edge: float = EDGE) -> float:
    from e34_field_physics import kernel_from_row
    import e33_field_learning as e33
    from markov_score.stationary import invariant_log_density_eig
    k = kernel_from_row(row, cfg or dict(e33.SETTINGS), grid)
    pi = np.exp(invariant_log_density_eig(k, grid, w).log_density)
    return float((pi * w)[np.abs(grid) > edge].sum())


def annotate(df, grid, w):
    """Add `invariant_mass_out` and `parked` to every kernel row with a `theta` (others get NaN)."""
    import pandas as pd
    out = []
    for _, r in df.iterrows():
        ok = r.get("arm") in KERNEL_ARMS and isinstance(r.get("theta"), str) and r["theta"]
        out.append(invariant_mass_out(r, grid, w) if ok else np.nan)
    df = df.copy()
    df["invariant_mass_out"] = out
    df["parked"] = pd.Series(out, index=df.index) > PARKED
    return df
