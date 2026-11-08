"""E67's reader with made-up tables in both directions; the grid gate refuses a coarse reference."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "core"))
import read_e67 as r  # noqa: E402


def _d(energy, var1, net):
    rows = []
    for seed in r.SEEDS:
        for n in (128, 512, 2048):
            rows += [dict(seed=seed, n_chains=n, arm="energy2d", risk=energy), dict(seed=seed, n_chains=n, arm="var1", risk=var1)]
            rows += [dict(seed=seed, n_chains=n, arm=a, risk=net + 0.01 * i) for i, a in enumerate(r.NETS)]
    return pd.DataFrame(rows)


def test_confirmed_and_falsified():
    ok = r.verdicts(_d(0.05, 0.10, 0.20), pd.DataFrame([dict(arm="energy2d", sec_per_iter=120.0)]))
    assert all(v for v, _ in ok.values()), ok
    bad = r.verdicts(_d(0.30, 0.10, 0.20), pd.DataFrame([dict(arm="energy2d", sec_per_iter=900.0)]))
    assert not any(v for v, _ in bad.values()), bad


def test_grid_gate():
    r.gates(pd.DataFrame([dict(t=-1.0, rel=3e-3)]))
    with pytest.raises(AssertionError, match="G1"):
        r.gates(pd.DataFrame([dict(t=-1.0, rel=2e-2)]))
