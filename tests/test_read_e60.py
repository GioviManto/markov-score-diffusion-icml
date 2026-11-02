"""tools/core/read_e60.py: every verdict in both directions on made-up tables, a missing row counting against, the bundle
gate, and the loader on E33's real frozen rows (a fake E60 file made of E33's own em_mixture rows)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "core"))
import read_e60 as r  # noqa: E402

# a world in which every prediction holds (risk in units of the floor)
LEVEL = {"em_free_C4": {128: 0.30, 512: 0.15, 2048: 0.08}, "em_free_C8": {128: 0.45, 512: 0.18, 2048: 0.08},
         "em_mixture": {128: 1.0, 512: 1.0, 2048: 1.0}, "em_mdn": {128: 0.60, 512: 0.30, 2048: 0.10},
         "em_pair_quartic": {128: 0.10, 512: 0.04, 2048: 0.02},
         **{a: {128: 0.50, 512: np.nan, 2048: 0.40} for a in r.NETS}}


def _table(level=LEVEL):
    rng = np.random.default_rng(0)
    rows = [dict(cell=c, arm=a, n_chains=m, seed=s, risk=v * (1 + 0.01 * rng.standard_normal()))
            for c in r.CELLS for a, by_m in level.items() for m, v in by_m.items() if np.isfinite(v) for s in r.SEEDS]
    return pd.DataFrame(rows)


def _set(df, arm, m, value, cells=r.CELLS, seeds=r.SEEDS):
    df = df.copy()
    sel = (df.arm == arm) & (df.n_chains == m) & df.cell.isin(cells) & df.seed.isin(seeds)
    df.loc[sel, "risk"] = value
    return df


def test_every_prediction_holds_in_the_made_up_world():
    v = r.verdicts(_table())
    assert all(ok for ok, _ in v.values()), v


def test_each_prediction_can_fail():
    df = _table()
    assert not r.p1(_set(df, "em_free_C4", 2048, 0.6, cells=r.CELLS[:1]))[0]                 # above half the floor
    assert not r.p1(_set(df, "em_free_C4", 512, 2.0, cells=r.CELLS[:1], seeds=range(3)))[0]  # 13/16 wins
    assert not r.p2(_set(df, "em_free_C4", 128, 0.9, cells=r.CELLS[:2]))[0]                  # only 2 of 4 cells
    assert r.p2(_set(df, "em_free_C4", 128, 0.9, cells=r.CELLS[:1]))[0]                      # 3 of 4 is enough
    assert not r.p3(_set(df, "em_free_C8", 2048, 0.5, cells=r.CELLS[3:]))[0]
    assert not r.p4(_set(df, "em_pair_quartic", 2048, 0.5, cells=r.CELLS[:2]))[0]
    assert not r.p5(_set(df, "window_ism_one_view", 128, 0.05, cells=r.CELLS[2:3]))[0]       # one network beats it


def test_a_missing_or_failed_row_counts_against():
    df = _table()
    gone = df[~((df.arm == "em_free_C4") & (df.n_chains == 2048) & (df.cell == r.CELLS[0]) & (df.seed == 5))]
    assert not r.p1(gone)[0]                                   # the seed mean at 2048 is undefined
    failed = _set(df, "em_free_C4", 2048, np.nan, cells=r.CELLS[:1], seeds=range(3))
    assert not r.p5(failed)[0]                                 # 13/16 paired wins
    assert not r.p3(_set(df, "em_free_C8", 128, np.nan, cells=r.CELLS[:1], seeds=(0,)))[0]


def test_the_bundle_gate_stops_a_mismatched_read():
    ref = pd.DataFrame(dict(cell=["c"] * 2, seed=[0, 1], n_chains=[128] * 2, val_bundle=["a", "b"], test_bundle=["x", "y"]))
    r.check_bundles(ref.copy(), ref)
    bad = ref.assign(test_bundle=["x", "z"])
    with pytest.raises(SystemExit):
        r.check_bundles(bad, ref)
    with pytest.raises(SystemExit):
        r.check_bundles(ref.assign(seed=[0, 7]), ref)


def test_the_loader_on_e33s_own_rows(tmp_path, monkeypatch):
    k33 = pd.concat([pd.read_csv(p) for p in sorted(ROOT.glob("outputs/hpc_E33k_s*/kernels.csv"))])
    fake = k33[(k33.arm == "em_mixture") & k33.cell.isin(r.CELLS) & k33.n_chains.isin(r.SIZES)].assign(arm="em_free_C4")
    d = tmp_path / "outputs" / "hpc_E60_s0"
    d.mkdir(parents=True)
    fake.to_csv(d / "kernels.csv", index=False)
    real = r._csvs
    monkeypatch.setattr(r, "_csvs", lambda *p: real(*p) if "E60" not in p[0] else pd.read_csv(d / "kernels.csv"))
    df, _ = r.load()
    a = df[df.arm == "em_free_C4"].set_index(r.KEY).risk.sort_index()
    b = df[df.arm == "em_mixture"].set_index(r.KEY).risk.sort_index()
    pd.testing.assert_series_equal(a, b.loc[a.index])
    assert set(df.arm) >= set(r.NETS) | set(r.E33_KERNELS)
    assert np.nanmedian(df[df.arm == "em_mixture"].risk) == pytest.approx(1.0, abs=0.05)   # the mixture sits at the floor
