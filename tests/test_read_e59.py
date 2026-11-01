"""E59's reader, tested before any E59 data exist: each registered prediction holds on a made-up table built to satisfy
it and flips on one built to break it, and a missing row counts AGAINST the prediction it enters."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "core"))
import read_e59 as r  # noqa: E402

METHODS = ("kernel", "gauss_ar1", "oracle", "qmlp", "qgru", "persistence", "climatology")


def world(b, scen, method, n, h):
    base = {"oracle": 0.20, "kernel": 0.22, "gauss_ar1": 0.30, "qmlp": 0.32, "qgru": 0.31, "persistence": 0.40,
            "climatology": 0.60}[method]
    return base * (1 + 0.1 * h) * (1 + 3.0 / np.sqrt(n) * (method != "oracle"))


def table(level=world, cov=lambda b, s, m, n, h, seed: 0.9, drop=()):
    rows = []
    for b in ("setar", "wind"):
        for scen in ("clean", "noisy"):
            for m in METHODS:
                if b == "wind" and m == "oracle":
                    continue
                for n in r.SIZES[b]:
                    for h in r.HORIZONS:
                        for s in r.SEEDS:
                            v = level(b, scen, m, n, h) * (1 + 0.01 * ((s % 3) - 1))
                            if (b, scen, m, n, h, s) in drop:
                                continue
                            rows.append(dict(benchmark=b, scenario=scen, method=m, n_chains=n, h=h, seed=s, crps=v,
                                             cov90=cov(b, scen, m, n, h, s), target_hash=f"{b}{s}"))
    return pd.DataFrame(rows)


def test_all_true_in_the_registered_world():
    v = r.verdicts(table())
    assert all(ok for ok, _ in v.values()), {k: d for k, (ok, d) in v.items() if not ok}


def _flip(b, scen, m, factor, where=lambda n, h: True):
    return lambda bb, ss, mm, n, h: world(bb, ss, mm, n, h) * (factor if (bb, ss, mm) == (b, scen, m) and where(n, h) else 1)


@pytest.mark.parametrize("key, level, cov", [
    ("P1", _flip("setar", "clean", "gauss_ar1", 0.5, lambda n, h: (n, h) == (512, 4)), None),
    ("P2", _flip("setar", "clean", "qgru", 0.5, lambda n, h: n == 32), None),
    ("P3", _flip("setar", "noisy", "qmlp", 0.5, lambda n, h: (n, h) == (2048, 1)), None),
    ("P4", _flip("wind", "clean", "gauss_ar1", 0.5, lambda n, h: (n, h) == (1917, 2)), None),
    ("P5", world, lambda b, s, m, n, h, seed: 0.80 if (b, s, m, n, h) == ("wind", "clean", "kernel", 1917, 1) and seed < 3 else 0.9),
])
def test_each_prediction_can_fail(key, level, cov):
    t = table(level) if cov is None else table(level, cov)
    v = r.verdicts(t)
    assert not v[key][0], v[key][1]


def test_a_missing_row_counts_against():
    v = r.verdicts(table(drop={("setar", "clean", "kernel", 512, 2, s) for s in range(2)}))
    assert not v["P1"][0] and not v["P2"][0]
