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


# --------------------------------------------------------------------------- E46
def _e46(sym_vs_aug=0.0, sym_vs_first=-0.2, sym_vs_2m=0.0, n=16, bundle_ok=True, nan_seed=None):
    """Synthetic b = 0 tables: per-seed first-order risk 0.02 (jittered); the other arms follow from the log ratios."""
    rng = np.random.default_rng(1)
    d35, d46 = [], []
    for M in (512, 2048):
        for s in range(n):
            first = 0.02 * np.exp(rng.normal(0, 0.1))
            sym = first * np.exp(sym_vs_first + rng.normal(0, 0.05))
            aug = sym * np.exp(-sym_vs_aug + rng.normal(0, 0.05))
            two = sym * np.exp(-sym_vs_2m + rng.normal(0, 0.05))
            bnd = f"t{M}{s}"
            d35 += [dict(b=0.0, n_chains=M, seed=s, arm="em_first_order", risk=first, test_bundle=bnd),
                    dict(b=0.0, n_chains=M, seed=s, arm="em_augmented", risk=aug, test_bundle=bnd)]
            sr = np.nan if nan_seed == s else sym
            d46 += [dict(b=0.0, n_chains=M, seed=s, arm="em_first_order_sym", risk=sr, test_bundle=bnd if bundle_ok else "x"),
                    dict(b=0.0, n_chains=M, seed=s, arm="em_first_order_2m", risk=two, test_bundle=bnd if bundle_ok else "x")]
    return pd.DataFrame(d46), pd.DataFrame(d35)


def test_e46_all_confirmed_when_the_symmetry_explains_the_gain():
    assert set(_verdicts(men.read_e46(*_e46())).values()) == {"CONFIRMED"}


def test_e46_p1_falls_when_augmented_is_still_better_than_the_symmetrised_fit():
    assert _verdicts(men.read_e46(*_e46(sym_vs_aug=0.3)))["P1"] == "FALSIFIED"      # augmented 26% below sym


def test_e46_p2_falls_when_mirror_data_do_not_help():
    assert _verdicts(men.read_e46(*_e46(sym_vs_first=0.0)))["P2"] == "FALSIFIED"


def test_e46_p3_falls_when_doubling_the_data_is_much_better_than_reflection():
    assert _verdicts(men.read_e46(*_e46(sym_vs_2m=0.5)))["P3"] == "FALSIFIED"        # sym 65% above the 2m fit


def test_e46_a_missing_fit_counts_against():
    v = _verdicts(men.read_e46(*_e46(nan_seed=3)))
    assert v["P1"] == "FALSIFIED" and v["P2"] == "FALSIFIED"


def test_e46_refuses_different_test_bundles():
    with pytest.raises(ValueError):
        men.read_e46(*_e46(bundle_ok=False))


# --------------------------------------------------------------------------- E43
def _rows(cell, arm, n, risks, bundle="h"):
    return [dict(cell=cell, seed=i, n_chains=n, arm=arm, selection="risk", risk=r, test_bundle=f"{bundle}{cell}{n}{i}") for i, r in enumerate(risks)]


def _e43(price128=3.0, price2048=1.3, vs_mixture=0.5, vs_spline128=1.5, lab_b=(1.4, 0.7, 1.3), n=16, nb=8):
    """Lab A: pair risk 0.01 (jittered), loglinear = ratio x pair; mixture and spline from the given ratios. Lab B: ratios site-energy-relative."""
    rng = np.random.default_rng(2)
    dA, d33 = [], []
    for cell in ("T0.3_h1_L32", "T1_iso_L32"):
        for M, ratio in ((128, price128), (2048, price2048)):
            pair = 0.01 * np.exp(rng.normal(0, 0.05, n))
            dA += _rows(cell, "em_loglinear", M, pair * ratio)
            d33 += _rows(cell, "em_pair_quartic", M, pair)
            d33 += _rows(cell, "em_tilted_spline", M, pair * ratio / vs_spline128 if M == 128 else pair * 2)
            d33 += _rows(cell, "em_mixture", M, pair * ratio / vs_mixture if M == 2048 else pair * 9)
    dB = []
    r128, r2048, vs = lab_b
    for M, ratio in ((128, r128), (2048, r2048)):
        pair = 0.02 * np.exp(rng.normal(0, 0.05, nb))
        ll = pair * ratio
        dB += _rows("LAP_B_L32", "em_pair_quartic", M, pair, "b") + _rows("LAP_B_L32", "em_loglinear", M, ll, "b")
        dB += _rows("LAP_B_L32", "em_tilted_spline", M, ll * vs, "b") + _rows("LAP_B_L32", "em_mixture", M, ll * vs, "b")
    return pd.DataFrame(dA), pd.DataFrame(d33), pd.DataFrame(dB)


def test_e43_all_confirmed_on_a_constructed_table():
    assert set(_verdicts(men.read_e43(*_e43())).values()) == {"CONFIRMED"}


def test_e43_p1_falls_when_generality_is_cheap_at_small_data():
    assert _verdicts(men.read_e43(*_e43(price128=1.2)))["P1"] == "FALSIFIED"


def test_e43_p2_falls_when_the_price_does_not_shrink():
    assert _verdicts(men.read_e43(*_e43(price128=3.0, price2048=3.5)))["P2"] == "FALSIFIED"


def test_e43_p3_falls_when_the_mixture_is_better_at_2048():
    assert _verdicts(men.read_e43(*_e43(vs_mixture=2.0)))["P3"] == "FALSIFIED"


def test_e43_p4_falls_when_loglinear_beats_the_spline_at_128():
    assert _verdicts(men.read_e43(*_e43(vs_spline128=0.5)))["P4"] == "FALSIFIED"


def test_e43_lab_b_p5_falls_when_the_site_energy_still_wins():
    v = _verdicts(men.read_e43(*_e43(lab_b=(1.4, 1.1, 1.3))))
    assert v["P5"] == "FALSIFIED" and v["P7"] == "FALSIFIED"


def test_e43_lab_b_p6_falls_when_the_mixture_is_better():
    assert _verdicts(men.read_e43(*_e43(lab_b=(1.4, 0.7, 0.8))))["P6"] == "FALSIFIED"


def test_e43_a_failed_fit_counts_against():
    dA, d33, dB = _e43()
    bad = dA.index[(dA.cell == "T0.3_h1_L32") & (dA.n_chains == 2048) & (dA.seed < 3)]
    dA.loc[bad, "risk"] = np.nan                                  # 13/16 < 14/16 below the mixture
    assert _verdicts(men.read_e43(dA, d33, dB))["P3"] == "FALSIFIED"


def test_e43_refuses_different_test_bundles():
    dA, d33, dB = _e43()
    dA["test_bundle"] = "x"
    with pytest.raises(ValueError):
        men.read_e43(dA, d33, dB)


# --------------------------------------------------------------------------- E45
def _e45(oracle_cov=0.90, site_cov=0.90, lin_cov=0.80, site_err=1.02, lin_err=1.0, lin_width=1.25, n=16, fail_site_seed=None):
    rows = []
    for cell in ("T0.3_h1_L32", "T1_iso_L32"):
        for pattern in ("random50", "gap8", "gap16"):
            for s in range(n):
                base = 0.5 * (1 + 0.01 * s)
                for arm, cov, err, wid in (("oracle", oracle_cov, 1.0, 2.0), ("em_pair_quartic", site_cov, site_err, 2.05),
                                           ("em_tilted_spline", site_cov, 1.05, 2.1), ("em_mixture", 0.88, 1.2, 2.3),
                                           ("linear", lin_cov, lin_err, 2.0 * lin_width)):
                    failed = int(arm == "em_pair_quartic" and fail_site_seed == s)
                    rows.append(dict(cell=cell, seed=s, pattern=pattern, arm=arm, failed=failed, rel_err=np.nan if failed else base * err,
                                     coverage90=np.nan if failed else cov, width90=wid))
    return pd.DataFrame(rows)


def test_e45_all_confirmed_on_a_constructed_table():
    assert set(_verdicts(men.read_e45(_e45())).values()) == {"CONFIRMED"}


def test_e45_p1_falls_when_the_oracle_is_miscalibrated():
    assert _verdicts(men.read_e45(_e45(oracle_cov=0.80)))["P1"] == "FALSIFIED"


def test_e45_p2_falls_on_a_coverage_miss_or_an_error_miss():
    assert _verdicts(men.read_e45(_e45(site_cov=0.80)))["P2"] == "FALSIFIED"
    assert _verdicts(men.read_e45(_e45(site_err=1.3)))["P2"] == "FALSIFIED"


def test_e45_p3_falls_when_the_linear_intervals_are_calibrated():
    assert _verdicts(men.read_e45(_e45(lin_cov=0.90)))["P3"] == "FALSIFIED"


def test_e45_p4_falls_when_the_linear_mean_is_much_worse():
    assert _verdicts(men.read_e45(_e45(lin_err=1.5)))["P4"] == "FALSIFIED"


def test_e45_p5_falls_when_the_linear_interval_is_not_wider():
    assert _verdicts(men.read_e45(_e45(lin_width=1.02)))["P5"] == "FALSIFIED"


def test_e45_a_failed_rebuild_counts_against():
    assert _verdicts(men.read_e45(_e45(fail_site_seed=3)))["P2"] == "FALSIFIED"


# --------------------------------------------------------------------------- E44
def _e44(site=0.85, spline_off=0.80, spline_eq=1.2, n=8, fail=None):
    """Risks as ratios of the linear arm (risk 1.0): site energy = `site`; the spline is `spline_off` x site off equilibrium, `spline_eq` x at equilibrium."""
    rows = []
    for tau in (1.5, 3.0, 6.0, 12.0, float("inf")):
        for s in range(n):
            j = 1 + 0.01 * s
            sp = spline_eq if np.isinf(tau) else spline_off
            for arm, r in (("linear_empirical", 1.0), ("em_pair_quartic", site * j), ("em_tilted_spline", site * j * sp)):
                rows.append(dict(tau=tau, seed=s, arm=arm, selection="risk", risk=np.nan if (fail == (tau, s) and arm == "em_pair_quartic") else r))
    return pd.DataFrame(rows)


def test_e44_all_confirmed_on_a_constructed_table():
    assert set(_verdicts(men.read_e44(_e44())).values()) == {"CONFIRMED"}


def test_e44_p1_and_p2_fall_when_the_site_energy_loses_to_linear():
    v = _verdicts(men.read_e44(_e44(site=1.1)))
    assert v["P1"] == "FALSIFIED" and v["P2"] == "FALSIFIED"


def test_e44_p3_falls_when_the_site_energy_stays_ahead_of_the_spline():
    assert _verdicts(men.read_e44(_e44(spline_off=1.3)))["P3"] == "FALSIFIED"


def test_e44_p4_falls_when_the_spline_wins_at_equilibrium():
    assert _verdicts(men.read_e44(_e44(spline_eq=0.8)))["P4"] == "FALSIFIED"


def test_e44_a_failed_fit_counts_against():
    v = _verdicts(men.read_e44(_e44(fail=(float("inf"), 2))))
    assert v["P1"] == "FALSIFIED"                                      # 7/8 is not 8/8
