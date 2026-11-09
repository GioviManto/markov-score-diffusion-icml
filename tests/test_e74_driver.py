"""E74's driver runs end to end at --quick and keeps the samples the image figures need."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_quick(tmp_path):
    env = {"PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'experiments'}", "PATH": "/usr/bin:/bin"}
    subprocess.run([sys.executable, str(ROOT / "experiments" / "e74_generation_hard.py"), "--quick", "--output-dir",
                    str(tmp_path)], check=True, env=env, cwd=ROOT, capture_output=True)
    g = pd.read_csv(tmp_path / "generation.csv")
    assert {"exact", "oracle", "energy_clean", "window_cont"} <= set(g.arm) and (g.failed == 0).all()
    z = np.load(tmp_path / "samples" / "langevin_n128.npz")
    assert {"reference", "oracle", "energy_clean", "window_cont"} <= set(z.files)
    assert np.load(tmp_path / "long" / "T0.3_h1_L32.npz")["oracle"].shape[1] == 40


def test_switch_rate():
    sys.path.insert(0, str(ROOT / "experiments")); sys.path.insert(0, str(ROOT / "src"))
    import e74_generation_hard as e
    assert e.switch_rate(np.array([[1.0, -1.0, -2.0, 3.0]])) == 2 / 3
