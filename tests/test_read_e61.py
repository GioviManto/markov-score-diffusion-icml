"""tools/core/read_e61.py: every verdict in both directions on made-up tables, a missing or failed row counting against,
and the bundle gate."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "core"))
import read_e61 as r  # noqa: E402


def _level(chain, arm, m):
    """A world in which every prediction holds."""
    shrink = {32: 1.0, 128: 0.6, 512: 0.4, 2048: 0.3}[m]
    if arm in r.NETS:
        return 0.30 if arm == "window" else 0.45
    if arm == "lmmse":
        return 0.17
    base = {"mixture": 0.10, "free4": 0.11, "energy": 0.12, "energy_clean": 0.05}[arm]
    if chain in r.OUTSIDE and arm == "mixture":
        base = 0.25
    return base * shrink


def _table():
    rng = np.random.default_rng(0)
    rows = [dict(chain=c, arm=a, n_chains=m, seed=s, risk=_level(c, a, m) * (1 + 0.01 * rng.standard_normal()))
            for c in r.CHAINS for a in r.KERNELS + r.NETS + ("lmmse",) for m in r.SIZES for s in r.SEEDS]
    return pd.DataFrame(rows)


def _set(df, arm, m, value, chains, seeds=r.SEEDS):
    df = df.copy()
    sel = (df.arm == arm) & (df.n_chains == m) & df.chain.isin(chains) & df.seed.isin(seeds)
    df.loc[sel, "risk"] = value
    return df


def test_every_prediction_holds_in_the_made_up_world():
    v = r.verdicts(_table())
    assert all(ok for ok, _ in v.values()), v


def test_each_prediction_can_fail():
    df = _table()
    assert not r.p1(_set(df, "free4", 128, 0.5, ["gauss"], seeds=range(2)))[0]               # 6/8 against a network
    assert not r.p2(_set(df, "mixture", 32, 0.5, ["uniform"], seeds=range(2)))[0]            # loses to the CNN
    assert not r.p2(_set(df, "mixture", 2048, 0.2, ["laplace"]))[0]                          # 0.2 > 1.5 x 0.033
    assert not r.p3(_set(df, "energy", 2048, 0.9, ["langevin"], seeds=range(2)))[0]
    assert not r.p4(_set(df, "energy_clean", 512, 0.9, ["gauss", "skew"]))[0]                # 2 chains fail: 5 of 7
    assert r.p4(_set(df, "energy_clean", 512, 0.9, ["gauss"]))[0]                             # 6 of 7 is enough
    assert not r.p5(_set(df, "window", 128, 0.001, ["pearson"]))[0]


def test_a_missing_or_failed_row_counts_against():
    df = _table()
    one = df[~((df.chain == "skew") & (df.arm == "free4") & (df.n_chains == 2048) & (df.seed == 3))]
    assert r.p3(one)[0]                                                                      # 7/8 wins still meet >= 7/8
    two = one[~((one.chain == "skew") & (one.arm == "free4") & (one.n_chains == 2048) & (one.seed == 4))]
    assert not r.p3(two)[0]                                                                  # 6/8: the missing rows lose
    failed = _set(df, "mixture", 2048, np.nan, ["student"], seeds=(0,))
    assert not r.p2(failed)[0]                                                               # its seed mean is undefined


def test_the_bundle_gate_stops_a_mismatched_read():
    ok = pd.DataFrame(dict(chain=["c"] * 2, seed=[0, 0], n_chains=[32, 32], arm=["mlp", "mixture"], test_bundle=["x", "x"]))
    r.check_bundles(ok)
    with pytest.raises(SystemExit):
        r.check_bundles(ok.assign(test_bundle=["x", "y"]))
