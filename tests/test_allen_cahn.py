"""markov_score.allen_cahn: the lattice stochastic Allen-Cahn dynamics against the exact phi^4 chain."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from e32_field_oracles import CELLS, resolve_cell  # noqa: E402
from markov_score.allen_cahn import quench, windows  # noqa: E402
from markov_score.utils import rng_for  # noqa: E402

CELL = next(c for c in CELLS if c["id"] == "T0.3_h1_L32")
CHAIN, INFO = resolve_cell(CELL)


def _stats(a):
    var = a.var()
    return var, (a[:, :-1] * a[:, 1:]).mean() / var, (a**4).mean() / var**2 - 3.0


def test_windows_are_non_overlapping_and_shaped():
    phi = np.arange(2 * 256, dtype=float).reshape(2, 256)
    w = windows(phi, 32, 64)
    assert w.shape == (8, 32) and np.array_equal(w[0], phi[0, :32]) and np.array_equal(w[1], phi[1, :32])


def test_a_long_run_reproduces_the_exact_equilibrium_chain():
    phi = quench(rng_for("test-ac-eq"), 0.3, 1.0, tau=80.0, n_runs=160, dt=0.05)
    a = windows(phi, 32, 64) / INFO["scale"]
    var, rho1, kurt = _stats(a)
    assert abs(var - 1.0) < 0.05 and abs(rho1 - INFO["rho1"]) < 0.03 and abs(kurt - INFO["excess_kurtosis"]) < 0.15


def test_an_early_quench_is_not_the_gibbs_measure():
    phi = quench(rng_for("test-ac-early"), 0.3, 1.0, tau=3.0, n_runs=160, dt=0.05)
    var, rho1, _ = _stats(windows(phi, 32, 64) / INFO["scale"])
    assert var < 0.9 and rho1 < INFO["rho1"] - 0.05          # the field has not yet reached the equilibrium amplitude and correlation


def test_the_dynamics_are_symmetric_under_phi_to_minus_phi():
    phi = quench(rng_for("test-ac-sym"), 0.3, 1.0, tau=10.0, n_runs=200, dt=0.05)
    assert abs(phi.mean()) < 0.03
