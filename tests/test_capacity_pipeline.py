"""End-to-end pin of the capacity experiment at quick scale: a fitted Gaussian-innovation
kernel (C = 1) lands on the LMMSE floor of the same test bundle (Proposition 1 at the
estimator level), and the mixture kernel (C = 8) goes below it on the Laplace chain."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_quick_capacity_run_reproduces_the_floor(tmp_path):
    out = tmp_path / "e02_quick"
    r = subprocess.run([sys.executable, "e02_capacity.py", "--quick", "--only", "capacity", "--output-dir", str(out)],
                       cwd=ROOT / "experiments", capture_output=True, text=True, env={**__import__("os").environ, "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"})
    assert r.returncode == 0, r.stderr[-2000:]
    df = pd.read_csv(out / "capacity.csv"); df = df[df.region == "all"]
    floor = float(df[df.arm == "lmmse_true"].risk.iloc[0])
    c1 = float(df[(df.arm == "em_bp") & (df.components == 1)].risk.iloc[0])
    c8 = float(df[(df.arm == "em_bp") & (df.components == 8)].risk.iloc[0])
    assert abs(c1 - floor) < 0.03 * floor + 0.005, (c1, floor)
    assert c8 < 0.8 * floor, (c8, floor)
    assert (df[df.arm == "em_bp"].theta.str.count(";") > 0).all()
