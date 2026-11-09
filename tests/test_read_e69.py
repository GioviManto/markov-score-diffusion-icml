"""E69's reader: verdicts move in both directions on made-up tables."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "core"))
import read_e69 as r  # noqa: E402

LEVELS = (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05)


def _table(kernel=0.05, net=0.2, grid_small=2e-3, grid_large=1e-5, lin=0.5, slope=2.0):
    rows = []
    for n in r.SIZES:
        for s in r.SEEDS:
            for t in LEVELS:
                f = slope if t == 0.005 else 1.0
                for arm, v in (("em_bp", kernel * f), ("spline", kernel * f), ("energy", kernel * f),
                               ("clean_energy", kernel * f / 2), ("mlp_small", net * f * f * 1.5),
                               ("window_small", net * f * f * 1.5), ("lmmse", lin if t == 0.005 else 0.25),
                               ("true_401", grid_small if t == 0.005 else grid_large)):
                    rows.append(dict(arm=arm, n_chains=n, seed=s, t=t, risk=v))
    return pd.DataFrame(rows)


def test_all_confirmed_on_a_table_built_to_confirm():
    v = r.verdicts(_table())
    assert v["P1"] and v["P2"] and v["P3"] and v["P4"] and v["P5"]


def test_p1_falsified_when_networks_win():
    assert not r.verdicts(_table(kernel=0.3, net=0.05))["P1"]


def test_p3_falsified_when_the_coarse_grid_is_fine():
    assert not r.verdicts(_table(grid_small=5e-4))["P3"]


def test_p4_falsified_when_linear_error_is_small():
    assert not r.verdicts(_table(lin=0.2))["P4"]


def test_missing_seed_counts_against():
    d = _table()
    d = d[~((d.arm == "clean_energy") & (d.seed == 0) & (d.t == 0.005))]
    assert np.isnan(r.seed_mean(d, "clean_energy", 128, 0.005))
    assert r.wins(d, "clean_energy", "mlp_small", 128, 0.005) == 7
