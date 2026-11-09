"""E69's driver runs end to end at --quick and writes the rows the reader expects."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_quick_run(tmp_path):
    env = {"PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'experiments'}", "PATH": "/usr/bin:/bin"}
    subprocess.run([sys.executable, str(ROOT / "experiments" / "e69_small_t.py"), "--quick", "--output-dir",
                    str(tmp_path)], check=True, env=env, cwd=ROOT, capture_output=True)
    d = pd.read_csv(tmp_path / "small_t.csv")
    assert {"true_401", "lmmse", "em_bp", "window_small"} <= set(d.arm)
    assert set(d.t) == {0.005, 0.05}
    assert (d.failed == 0).all()
    assert (tmp_path / "gate.json").exists()
