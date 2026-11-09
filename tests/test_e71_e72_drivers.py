"""E71's and E72's drivers run end to end at --quick; E71's K = 1 groups are E03's one-view interface exactly."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))
ENV = {"PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'experiments'}", "PATH": "/usr/bin:/bin"}


def _run(script, out):
    subprocess.run([sys.executable, str(ROOT / "experiments" / script), "--quick", "--output-dir", str(out)],
                   check=True, env=ENV, cwd=ROOT, capture_output=True)


def test_e71_quick(tmp_path):
    _run("e71_views.py", tmp_path)
    d = pd.read_csv(tmp_path / "views.csv")
    assert set(d.views) == {1, 4} and (d.failed == 0).all()
    one = d[d.views == 1].n_noisy.iloc[0]
    four = d[d.views == 4].n_noisy.iloc[0]
    assert four == 4 * one


def test_e71_one_view_is_e03():
    import e03_baselines as e03
    import e71_views as e71
    A = np.random.default_rng(0).standard_normal((24, 32))
    g = e71.view_groups(A, 3, 24, 1)
    ref = e03._one_view_groups(A, 3)
    assert len(g) == len(ref)
    for (X, *_), (Xr, _) in zip(g, ref):
        assert np.array_equal(X, Xr)


def test_e71_twelve_views_cover_every_level():
    import e71_views as e71
    A = np.random.default_rng(0).standard_normal((10, 32))
    g = e71.view_groups(A, 0, 10, 12)
    assert len(g) == 12 and all(len(X) == 10 for X, *_ in g)


def test_e72_quick(tmp_path):
    _run("e72_small_t_field.py", tmp_path)
    d = pd.read_csv(tmp_path / "small_t_field.csv")
    assert {"lmmse", "energy_h16", "window_small"} <= set(d.arm)
    assert (d.failed == 0).all() and (tmp_path / "gate.json").exists()
