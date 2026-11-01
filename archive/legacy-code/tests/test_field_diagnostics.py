"""tools/field_diagnostics.py: the parked-fit diagnostic of E33 amendment 2 (N-33).

The diagnostic is the mass, beyond |a| = 3.5, of the first-site law the ESTIMATOR used: the fitted
kernel's invariant law from the direct eigensolve, or the standard normal where the eigensolve
raises (e33 `_stationary_mu`). On 3 Oct 2026 the eigensolve raised on 20 of the 768 confirmatory
clean tilted fits (all Legendre) and crashed the diagnostic. A closed-form substitute (pi = g Z) was
tried and rejected: a fitted tilted walk can be reducible on the grid, with two invariant laws, and
on one preview fit the eigensolve and g Z disagree (0 against 0.0225) while both are exactly
invariant. Pinned: the diagnostic measures exactly the law `_stationary_mu` hands the estimator, on
a working fit, on the failing fit and on the reducible one (parameters copied from those rows), and
`annotate` reports the eigensolve failure and the g Z mass beside it.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "experiments"))
import e33_field_learning as e33
import field_diagnostics as fd

from markov_score.bp_grid import make_grid
from markov_score.stationary import invariant_log_density_eig

# E34 clean, em_tilted_poly, T0.5_iso_L32, seed 9, M = 2048 (outputs/hpc_E34c/clean.csv, job 678166)
EIG_FAILS = "-1.0248158;2.5841096;89.05953;2.5412084;124.84477;0.68277226;52.784163"
# E33 preview, em_tilted_poly, T0.3_h1_L32, seed 1, M = 512 (outputs/E33_preview_laptop/kernels.csv)
REDUCIBLE = "-0.78700451;138.74129;-194.14702;150.24627;121.64374;46.3683;125.37871"
REGULAR = "-1.2;0.3;-0.8;0.1;-0.2;0.05;-0.01"


@pytest.fixture(scope="module")
def grid():
    return make_grid(8.0, 401)


def _row(theta, arm="em_tilted_poly"):
    return pd.Series({"arm": arm, "theta": theta})


def _estimator_mass(theta, g, w, arm="em_tilted_poly"):
    """The mass beyond 3.5 of the law the E33 driver itself would use for the first site."""
    with pytest.warns(RuntimeWarning) if theta == EIG_FAILS else nullcontext():
        log_pi = e33._stationary_mu(g, w)(fd._kernel(_row(theta, arm), g))
    pi = np.exp(log_pi)
    return float((pi * w)[np.abs(g) > fd.EDGE].sum())


@pytest.mark.parametrize("theta", [REGULAR, REDUCIBLE, EIG_FAILS])
def test_the_diagnostic_measures_the_estimators_own_first_site_law(grid, theta):
    g, w = grid
    assert fd.invariant_mass_out(_row(theta), g, w) == pytest.approx(_estimator_mass(theta, g, w), abs=1e-15)


def test_the_failing_fit_falls_back_to_the_standard_normal(grid):
    g, w = grid
    k = fd._kernel(_row(EIG_FAILS), g)
    with pytest.raises(RuntimeError):
        invariant_log_density_eig(k, g, w)
    assert fd.invariant_mass_out(_row(EIG_FAILS), g, w) == pytest.approx(4.65e-4, rel=0.01)  # 2 Phi(-3.5)
    assert fd.eig_fails(_row(EIG_FAILS), g, w) and not fd.eig_fails(_row(REGULAR), g, w)


def test_a_reducible_fit_has_two_invariant_laws(grid):
    """Why g Z is reported beside, not instead: both laws are invariant, they differ beyond 3.5."""
    g, w = grid
    k = fd._kernel(_row(REDUCIBLE), g)
    K = np.exp(k.log_transition_matrix(g))
    eig = np.exp(invariant_log_density_eig(k, g, w).log_density)
    lp = k.log_stationary(g); gz = np.exp(lp - lp.max()); gz /= (gz * w).sum()
    for p in (eig, gz):
        assert float((np.abs(K @ (p * w) - p) * w).sum()) < 1e-10
    assert fd.invariant_mass_out(_row(REDUCIBLE), g, w) < 1e-6
    assert fd.gz_mass_out(_row(REDUCIBLE), g, w) == pytest.approx(0.0225, abs=5e-4)


def test_annotate_reports_the_failure_and_the_gz_mass(grid):
    g, w = grid
    df = pd.DataFrame([dict(arm="em_tilted_poly", theta=EIG_FAILS), dict(arm="em_tilted_poly", theta=REGULAR),
                       dict(arm="em_pair_quartic", theta="-1.3;0.5;-2.0;0.1;1.0")])
    out = fd.annotate(df, g, w)
    assert list(out.eig_failed) == [True, False, False]
    assert out.invariant_mass_out.notna().all() and out.gz_mass_out.notna().all()
    assert not out.parked.any()
