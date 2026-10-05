"""tools/make_replication_numbers.py, pinned on the frozen two-seed designs before E35x and E36x land.

On E35's and E36's own files the tool must give their registered readings: E36's MDN clause UNRESOLVED (the MDN at
its last checkpoint in both stride-4 fits), the mixture within 3% of E36's floors (1.026, 1.007, 1.012), E35's
first/augmented seed means 1.27 < 8.49 < 13.28 at M = 2048, and the floors read from E36's pooled rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_replication_numbers as mr


def test_bootstrap_is_deterministic_and_brackets_the_mean():
    v = np.array([0.1, -0.2, 0.05, 0.3, -0.1])
    assert mr.boot_ci(v, b=5000) == mr.boot_ci(v, b=5000)
    lo, hi = mr.boot_ci(v, b=5000)
    assert lo < v.mean() < hi


def test_e36_floors_are_the_pooled_rows():
    fl = mr.e36_floors(ROOT / "outputs/E36/oracles.csv")
    assert fl == pytest.approx({1: 0.0975, 2: 0.1578, 4: 0.2347}, abs=5e-5)


def test_the_frozen_two_seed_designs_read_as_registered():
    R = pd.DataFrame(mr.read_e35x(pd.read_csv(ROOT / "outputs/E35/augmented.csv"))
                     + mr.read_e36x(pd.read_csv(ROOT / "outputs/E36/clean.csv"), mr.e36_floors(ROOT / "outputs/E36/oracles.csv")))
    p2 = R[(R.experiment == "E36x") & (R.prediction == "P2")].iloc[0]
    assert p2.verdict == "UNRESOLVED" and "2/2" in p2.note
    p3 = R[(R.experiment == "E36x") & (R.prediction == "P3")]
    assert list(p3.value.round(3)) == [1.026, 1.007, 1.012]
    rising = R[R.clause.str.contains("rising")].iloc[0]
    assert rising.note == "1.27 < 8.49 < 13.28"
