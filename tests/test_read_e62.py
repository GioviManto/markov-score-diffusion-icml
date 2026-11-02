"""tools/core/read_e62.py: every verdict in both directions on made-up tables, missing rows counting against, the bundle
gate, and the break-even interpolation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "core"))
import read_e62 as r  # noqa: E402

FLOOR = r.calibration()


def _level(p, arm, m):
    """A world in which every prediction holds: first sits on the calibrated floor, pair near the truth."""
    f = FLOOR[p]["first_order_floor"]
    if arm == "first":
        return max(f, 0.01) * (1.0 if m >= 512 else 1.2) + 0.01
    if arm == "pair":
        return 0.02 + 0.02 * (2048 / m) ** 0.3
    if arm == "window":
        return 0.09 + 0.15 * abs(p) if p >= 0 else 0.15
    if arm == "mlp":
        return 0.4
    return 0.16


def _arms():
    rng = np.random.default_rng(0)
    return pd.DataFrame([dict(phi2=p, arm=a, n_chains=m, seed=s, risk=_level(p, a, m) * (1 + 0.005 * rng.standard_normal()))
                         for p in r.PHI2 for a in ("first", "pair", "mlp", "window", "lmmse") for m in r.SIZES
                         for s in r.SEEDS])


def _wind():
    return pd.DataFrame([dict(arm=a, n_chains=m, seed=s, risk={"pair": 0.40, "first": 0.45, "mlp": 0.6, "window": 0.5}[a])
                         for a in ("first", "pair", "mlp", "window") for m in r.WIND_SIZES for s in r.SEEDS])


def _set(df, value, **sel):
    df = df.copy()
    mask = np.ones(len(df), bool)
    for k, v in sel.items():
        mask &= df[k].isin(v if isinstance(v, (list, tuple, range)) else [v]).to_numpy()
    df.loc[mask, "risk"] = value
    return df


def test_every_prediction_holds_in_the_made_up_world():
    v = r.verdicts(_arms(), _wind(), FLOOR)
    assert all(ok for ok, _ in v.values()), v


def test_each_prediction_can_fail():
    a, w = _arms(), _wind()
    assert not r.p1(_set(a, 0.5, phi2=0.3, arm="first", n_chains=2048), FLOOR)[0]                 # 2.9x the floor
    assert not r.p1(_set(a, 0.2, phi2=0.2, arm="first", n_chains=512), FLOOR)[0]                  # 2048 >10% below 512
    assert not r.p2(_set(a, 0.9, phi2=0.4, arm="pair", n_chains=2048, seed=range(2)))[0]
    assert not r.p3(_set(a, 0.5, phi2=0.1, arm="first", n_chains=2048))[0]                        # first loses at 0.1
    assert not r.p3(_set(a, 0.9, phi2=0.4, arm="window"))[0]                                      # first still wins at 0.4
    assert not r.p4(_set(a, 0.001, phi2=-0.3, arm="mlp", n_chains=128))[0]
    assert not r.p5(_set(w, 0.9, arm="pair", n_chains=32, seed=range(2)))[0]


def test_a_missing_row_counts_against():
    a = _arms()
    gone = a[~((a.arm == "first") & (a.n_chains == 2048) & np.isclose(a.phi2, 0.3) & (a.seed == 2))]
    assert not r.p1(gone, FLOOR)[0]                                    # its seed mean is undefined
    two = a[~((a.arm == "pair") & (a.n_chains == 2048) & np.isclose(a.phi2, 0.6) & a.seed.isin((0, 1)))]
    assert not r.p2(two)[0]


def test_the_bundle_gate_and_the_break_even():
    ok = pd.DataFrame(dict(phi2=[0.1, 0.1], seed=[0, 0], n_chains=[128, 128], test_bundle=["x", "x"]))
    r.check_bundles(ok, ("phi2", "seed", "n_chains"))
    with pytest.raises(SystemExit):
        r.check_bundles(ok.assign(test_bundle=["x", "y"]), ("phi2", "seed", "n_chains"))
    be = r.break_even(_arms(), FLOOR)
    assert FLOOR[0.0]["delta_exact"] <= be <= FLOOR[0.6]["delta_exact"]
