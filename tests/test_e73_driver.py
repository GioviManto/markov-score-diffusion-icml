"""E73's driver runs end to end at --quick; the real series load with the registered splits."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))


def test_quick(tmp_path):
    env = {"PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'experiments'}", "PATH": "/usr/bin:/bin"}
    subprocess.run([sys.executable, str(ROOT / "experiments" / "e73_real_and_hidden.py"), "--quick", "--output-dir",
                    str(tmp_path)], check=True, env=env, cwd=ROOT, capture_output=True)
    r = pd.read_csv(tmp_path / "regime.csv")
    assert {"regime", "first", "window", "lmmse"} <= set(r.arm) and (r.failed == 0).all()
    q = pd.read_csv(tmp_path / "real.csv")
    assert set(q.series) == {"dtb3", "milan"}


def test_series_splits():
    import e73_real_and_hidden as e
    for name, n_all in (("dtb3", 398), ("milan", 913)):
        z, split = e.series(name)
        assert set(np.unique(split)) == {0, 1, 2}
        assert abs(np.nanmean(z[split == 0])) < 1e-9 and abs(np.nanstd(z[split == 0]) - 1) < 1e-9
        tr, _, _ = e.real_data(e.SETTINGS, name, 0, 10**6)
        assert len(tr) == n_all and np.isfinite(tr).all()
