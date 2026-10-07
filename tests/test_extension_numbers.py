"""tools/make_extension_numbers.py: the pinned readings of E41 and E42 give the verdict a constructed table implies.

Each clause is tested in both directions on synthetic data, and a failed fit must count against a clause, never for it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_extension_numbers as men  # noqa: E402

CELLS = men.CELLS41
SEEDS = range(8)


def _kernels(arm, risk, cells=CELLS, n=2048, at_cap=0, bundle="b"):
    rows = [dict(cell=c, seed=s, n_chains=n, arm=arm, selection="risk", risk=risk, at_cap=at_cap, test_bundle=f"{bundle}{c}{s}")
            for c in cells for s in SEEDS]
    return pd.DataFrame(rows)


def _frames(mdn41=0.48, mdn33=0.6, spline=0.3, mixture=1.0, pair=0.2, at_cap41=0):
    d41 = _kernels("em_mdn", mdn41, at_cap=at_cap41)
    d33 = pd.concat([_kernels("em_mdn", mdn33, at_cap=1), _kernels("em_tilted_spline", spline), _kernels("em_mixture", mixture),
                     _kernels("em_pair_quartic", pair)], ignore_index=True)
    return d41, d33


def _verdicts(rows):
    return {r["prediction"]: r["verdict"] for r in rows}


def test_e41_all_confirmed_on_a_constructed_table():
    assert set(_verdicts(men.read_e41(*_frames())).values()) == {"CONFIRMED"}


def test_e41_p1_falls_when_the_budget_changed_nothing():
    v = _verdicts(men.read_e41(*_frames(mdn41=0.6)))
    assert v["P1"] == "FALSIFIED"


def test_e41_p2_falls_when_fits_are_still_at_the_cap():
    assert _verdicts(men.read_e41(*_frames(at_cap41=1)))["P2"] == "FALSIFIED"


def test_e41_p3_falls_when_the_gap_to_the_spline_closes():
    assert _verdicts(men.read_e41(*_frames(mdn41=0.4, spline=0.3)))["P3"] == "FALSIFIED"      # ratio 1.33 < 1.5


def test_e41_p5_falls_when_the_mdn_reaches_the_site_energy():
    assert _verdicts(men.read_e41(*_frames(mdn41=0.15, pair=0.2)))["P5"] == "FALSIFIED"


def test_e41_a_failed_fit_counts_against_not_for():
    d41, d33 = _frames()
    bad = d41.index[(d41.cell == CELLS[0]) & (d41.seed.isin([0, 1]))]
    d41.loc[bad, "risk"] = np.nan                  # two failed MDN fits on one cell: 6/8 < 7/8
    v = _verdicts(men.read_e41(d41, d33))
    assert v["P4"] == "FALSIFIED" and v["P5"] == "FALSIFIED"


def test_e41_refuses_different_test_bundles():
    d41, d33 = _frames()
    d41["test_bundle"] = "other"
    with pytest.raises(ValueError):
        men.read_e41(d41, d33)


def _tab(err_by_n, signed=False, theta=1.0):
    rows = []
    for c in men.TCELL:
        for n, e in err_by_n[c].items():
            for s in range(16):
                sign = (1 if s % 2 == 0 else -1) if signed else 1
                rows.append(dict(cell=c, n_chains=n, seed=s, barrier_hat=0.25 + sign * e, theta_ratio=theta))
    return pd.DataFrame(rows)


GOOD = {"T1_iso_L32": {512: 0.08, 2048: 0.04, 8192: 0.02, 32768: 0.01},
        "T2_iso_L32": {512: 0.16, 2048: 0.08, 8192: 0.04, 32768: 0.02}}


def test_e42_all_confirmed_on_a_constructed_table():
    assert set(_verdicts(men.read_e42(_tab(GOOD, signed=True), {})).values()) == {"CONFIRMED"}


def test_e42_p1_falls_on_a_floor():
    floor = {c: {**v, 8192: v[2048], 32768: v[2048]} for c, v in GOOD.items()}
    assert _verdicts(men.read_e42(_tab(floor, signed=True), {}))["P1"] == "FALSIFIED"


def test_e42_p3_falls_when_t2_stays_unresolved():
    bad = {**GOOD, "T2_iso_L32": {512: 0.2, 2048: 0.15, 8192: 0.1, 32768: 0.08}}
    assert _verdicts(men.read_e42(_tab(bad, signed=True), {}))["P3"] == "FALSIFIED"


def test_e42_p5_detects_a_bias():
    assert _verdicts(men.read_e42(_tab(GOOD, signed=False), {}))["P5"] == "FALSIFIED"        # all errors positive: bias 0.02


def test_e42_a_failed_fit_counts_against():
    tab = _tab(GOOD, signed=True)
    keep = ~((tab.cell == "T2_iso_L32") & (tab.n_chains == 32768) & (tab.seed < 2))        # two fits missing from the table
    v = _verdicts(men.read_e42(tab[keep], {("T2_iso_L32", 32768): 2}))
    assert v["P3"] == "FALSIFIED"                                                            # 14/16 < 15/16


def test_e42_p6_falls_when_the_temperature_drifts():
    assert _verdicts(men.read_e42(_tab(GOOD, signed=True, theta=1.2), {}))["P6"] == "FALSIFIED"
