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


# --------------------------------------------------------------------------- E47
def _e47(r101=1.0, r201=1.0, r51_base=1.01, r51_fine=1.05, r101_fine=1.0, wall101=0.2, wall51=0.1, n=8, fail=None):
    """E33's N_g = 401 rows (risk 0.01, wall 100) and E47's coarse rows as constructed ratios of them."""
    cells = ("T0.3_h1_L32", "T0.3_h0.25_L128")
    d33, d47 = [], []
    for cell in cells:
        for arm in ("em_pair_quartic", "em_tilted_spline"):
            for M in (128, 2048):
                for s in range(n):
                    bnd = f"b{cell}{M}{s}"
                    d33.append(dict(cell=cell, arm=arm, selection="risk", n_chains=M, seed=s, risk=0.01, wall=100.0, test_bundle=bnd))
                    for ng in (51, 101, 201):
                        if cell == cells[0]:
                            r = {51: r51_base, 101: r101, 201: r201}[ng]
                        else:
                            r = {51: r51_fine, 101: r101_fine, 201: 1.0}[ng]
                        w = {51: wall51, 101: wall101, 201: 0.5}[ng] * 100.0
                        risk = np.nan if fail == (cell, arm, M, ng, s) else 0.01 * r
                        d47.append(dict(cell=cell, arm=arm, selection="risk", n_chains=M, seed=s, grid_size=ng, risk=risk, wall=w, test_bundle=bnd))
    return pd.DataFrame(d47), pd.DataFrame(d33)


def test_e47_learned_all_confirmed_on_a_constructed_table():
    assert set(_verdicts(men.read_e47_learned(*_e47())).values()) == {"CONFIRMED"}


def test_e47_p1_falls_when_n101_moves_the_risk():
    assert _verdicts(men.read_e47_learned(*_e47(r101=1.03)))["P1"] == "FALSIFIED"


def test_e47_p2_and_p3_fall_on_a_degraded_coarse_grid():
    v = _verdicts(men.read_e47_learned(*_e47(r51_base=1.08, r51_fine=1.25)))
    assert v["P2"] == "FALSIFIED" and v["P3"] == "FALSIFIED"


def test_e47_p4_falls_when_the_small_data_fit_depends_on_the_grid():
    assert _verdicts(men.read_e47_learned(*_e47(r101_fine=1.05)))["P4"] == "FALSIFIED"


def test_e47_p5_falls_when_the_fit_is_not_faster():
    assert _verdicts(men.read_e47_learned(*_e47(wall101=0.4)))["P5"] == "FALSIFIED"


def test_e47_a_failed_fit_counts_against():
    d47, d33 = _e47(fail=("T0.3_h1_L32", "em_pair_quartic", 2048, 101, 3))
    assert _verdicts(men.read_e47_learned(d47, d33))["P1"] == "FALSIFIED"


def test_e47_refuses_different_test_bundles():
    d47, d33 = _e47()
    d47["test_bundle"] = "x"
    with pytest.raises(ValueError):
        men.read_e47_learned(d47, d33)


def _e47t(ms):
    return pd.DataFrame([dict(arm="bp_grid", factor="grid_size", batch=512, n_sites=32, grid_size=g, wall_median=v * 512 / 1000) for g, v in ms.items()])


GOODT = {41: 0.040, 51: 0.050, 61: 0.058, 81: 0.075, 101: 0.114, 151: 0.17, 201: 0.237, 401: 0.524}


def test_e47_time_confirmed_and_falsified():
    assert set(_verdicts(men.read_e47_time(_e47t(GOODT))).values()) == {"CONFIRMED"}
    slow = {**GOODT, 51: 0.09}
    assert _verdicts(men.read_e47_time(_e47t(slow)))["P6"] == "FALSIFIED"
    steep = {g: 0.001 * (g / 41) ** 2 for g in GOODT}
    assert _verdicts(men.read_e47_time(_e47t(steep)))["P7"] == "FALSIFIED"


# --------------------------------------------------------------------------- E48
def _e48(mdn4=0.5, mdn1=3.0, mdn2=0.9, at_cap4=0, n=16, fail=None):
    rng = np.random.default_rng(3)
    d48, d36 = [], []
    for st, ratio in ((1, mdn1), (2, mdn2), (4, mdn4)):
        for s in range(n):
            site = 0.01 * np.exp(rng.normal(0, 0.05))
            d36.append(dict(stride=st, seed=s, arm="em_pair_quartic", risk=site, test_bundle=f"t{st}{s}"))
            d48.append(dict(stride=st, seed=s, arm="em_mdn", risk=np.nan if fail == (st, s) else site * ratio * np.exp(rng.normal(0, 0.03)),
                            at_cap=int(st == 4 and s < at_cap4), checkpoint=16000 if (st == 4 and s < at_cap4) else 5000, test_bundle=f"t{st}{s}"))
    return pd.DataFrame(d48), pd.DataFrame(d36)


def test_e48_all_confirmed_on_a_constructed_table():
    assert set(_verdicts(men.read_e48(*_e48())).values()) == {"CONFIRMED"}


def test_e48_p1_falls_when_the_mdn_is_still_capped():
    assert _verdicts(men.read_e48(*_e48(at_cap4=8)))["P1"] == "FALSIFIED"


def test_e48_p2_p3_p4_p5_fall_in_their_directions():
    assert _verdicts(men.read_e48(*_e48(mdn4=1.2)))["P2"] == "FALSIFIED"
    assert _verdicts(men.read_e48(*_e48(mdn1=0.8)))["P3"] == "FALSIFIED"
    assert _verdicts(men.read_e48(*_e48(mdn1=6.0)))["P4"] == "FALSIFIED"
    assert _verdicts(men.read_e48(*_e48(mdn2=1.2)))["P5"] == "FALSIFIED"


def test_e48_a_failed_fit_counts_against():
    v = _verdicts(men.read_e48(*_e48(fail=(1, 2))))
    assert v["P3"] == "FALSIFIED"                                  # 15/16 above is not 16/16


def test_e48_refuses_different_test_bundles():
    d48, d36 = _e48()
    d48["test_bundle"] = "x"
    with pytest.raises(ValueError):
        men.read_e48(d48, d36)


# --------------------------------------------------------------------------- E49
def _e49(mdn_vs_site=0.8, mdn_vs_mix=0.5, ll_vs_mdn=0.8, at_cap=0, n=8):
    rng = np.random.default_rng(4)
    d49, dB = [], []
    for s in range(n):
        site = 0.10 * np.exp(rng.normal(0, 0.03)); b = f"b{s}"
        mdn = site * mdn_vs_site
        for arm, risk in (("em_pair_quartic", site), ("em_mixture", mdn / mdn_vs_mix), ("em_loglinear", mdn * ll_vs_mdn)):
            dB.append(dict(cell="LAP_B_L32", seed=s, n_chains=2048, arm=arm, selection="risk", risk=risk, test_bundle=b))
        d49.append(dict(cell="LAP_B_L32", seed=s, n_chains=2048, arm="em_mdn", selection="risk", risk=mdn, at_cap=int(s < at_cap), test_bundle=b))
    return pd.DataFrame(d49), pd.DataFrame(dB)


def test_e49_all_confirmed_on_a_constructed_table():
    assert set(_verdicts(men.read_e49(*_e49())).values()) == {"CONFIRMED"}


def test_e49_clauses_fall_in_their_directions():
    assert _verdicts(men.read_e49(*_e49(mdn_vs_site=1.2)))["P1"] == "FALSIFIED"
    assert _verdicts(men.read_e49(*_e49(mdn_vs_mix=1.5)))["P2"] == "FALSIFIED"
    assert _verdicts(men.read_e49(*_e49(ll_vs_mdn=1.3)))["P3"] == "FALSIFIED"
    assert _verdicts(men.read_e49(*_e49(at_cap=4)))["P4"] == "FALSIFIED"


def test_e49_refuses_different_test_bundles():
    d49, dB = _e49()
    d49["test_bundle"] = "x"
    with pytest.raises(ValueError):
        men.read_e49(d49, dB)


def _e47c(move201=0.0, move101=0.0, resolved101=1.0, resolved51=0.5, n=16):
    rows = []
    for M in (32, 2048):
        for seed in range(n):
            for ng, mv, res in ((401, 0.0, 1.0), (201, move201, 1.0), (101, move101, resolved101), (51, 0.1, resolved51)):
                rows.append(dict(seed=seed, n_chains=M, n_grid=ng, half_width=8.0, risk_em_C1=0.1 * (1 + mv), risk_em_C8=0.08 * (1 + mv),
                                 resolved_C1=True, resolved_C8=bool(seed < res * n)))
    return pd.DataFrame(rows)


def test_e47_headline_confirmed_and_each_clause_falsified():
    assert set(_verdicts(men.read_e47_headline(_e47c())).values()) == {"CONFIRMED"}
    assert _verdicts(men.read_e47_headline(_e47c(move201=0.02)))["P8"] == "FALSIFIED"
    assert _verdicts(men.read_e47_headline(_e47c(move101=0.08)))["P9"] == "FALSIFIED"
    assert _verdicts(men.read_e47_headline(_e47c(resolved101=0.5)))["P10"] == "FALSIFIED"
    assert _verdicts(men.read_e47_headline(_e47c(resolved51=1.0)))["P10"] == "FALSIFIED"      # not lower at 51 than at 101


# --------------------------------------------------------------------------- E50
E50_LEVELS = np.geomspace(0.02, 4.0, 48)
E50_AXIS_T0 = dict(zip(men.E50_AXIS, (1.50, 1.30, 1.15, 1.05, 0.95, 0.85, 0.75, 0.70)))
E50_AXIS_EXCESS = dict(zip(men.E50_AXIS, (0.20, 0.15, 0.10, 0.06, 0.04, 0.03, 0.02, 0.01)))
E50_SERIES_T0 = {4: 0.60, 8: 0.80, 16: 1.00, 32: 1.20, 64: 1.30, 128: 1.35, 256: 1.36}


def _curve(t0):
    return 1.0 / (1.0 + (E50_LEVELS / t0) ** 4)


def _e50(axis_t0=None, excess=None, series_t0=None, dev=0.0, dev_cell=None, slope_scale=1.0, never=None):
    axis_t0, excess, series_t0 = axis_t0 or E50_AXIS_T0, excess or E50_AXIS_EXCESS, series_t0 or E50_SERIES_T0
    cells = {c: (t0, t0 - excess[c], t0, 10.0) for c, t0 in axis_t0.items()}
    for L, t0 in series_t0.items():
        lam = np.expm1(2.0 * (0.4 + (t0 - 0.6) / slope_scale))          # (1/2) ln(1 + lam) moves 1/slope_scale per unit t_c
        tb = cells[f"T0.2_h1_L{L}"][0] if L == 32 else t0
        cells[f"T0.2_h1_L{L}"] = (tb, tb - 0.1, t0, lam)
    rows = []
    for cell, (t_all, t_twin, t_bulk, lam) in cells.items():
        for rep in (0, 1):
            c_all = np.zeros_like(E50_LEVELS) + 0.4 if cell == never else _curve(t_all)
            for lev, ca, cb, ct in zip(E50_LEVELS, c_all, _curve(t_bulk), _curve(t_twin)):
                rows.append(dict(cell=cell, rep=rep, level=lev, commit_all=ca, commit_bulk=cb, twin_commit_all=ct, lambda_max=lam,
                                 correct_all=0.5 * (1 + ca) + (dev if cell == dev_cell else 0.0)))
    return pd.DataFrame(rows)


def test_e50_commitment_all_confirmed_on_a_constructed_table():
    v = {k: x for k, x in _verdicts(men.read_e50_commitment(_e50())).items() if k != "commitment"}
    assert v == {"P1": "CONFIRMED", "P2": "CONFIRMED", "P3": "CONFIRMED", "P4": "CONFIRMED", "P5": "CONFIRMED"}


def test_e50_p1_falls_on_a_miscalibrated_cell():
    assert _verdicts(men.read_e50_commitment(_e50(dev=0.02, dev_cell="T1_h1_L32")))["P1"] == "FALSIFIED"


def test_e50_p2_falls_when_the_order_breaks():
    t0 = {**E50_AXIS_T0, "T1_h1_L32": 0.97}
    assert _verdicts(men.read_e50_commitment(_e50(axis_t0=t0)))["P2"] == "FALSIFIED"


def test_e50_a_curve_that_never_crosses_counts_against():
    assert _verdicts(men.read_e50_commitment(_e50(never="T0.7_h1_L32")))["P2"] == "FALSIFIED"


def test_e50_p3_falls_when_the_twin_commits_first():
    ex = {**E50_AXIS_EXCESS, "T0.3_h1_L32": -0.05}
    assert _verdicts(men.read_e50_commitment(_e50(excess=ex)))["P3"] == "FALSIFIED"
    ex = {**E50_AXIS_EXCESS, "T2_h1_L32": 0.5}
    assert _verdicts(men.read_e50_commitment(_e50(excess=ex)))["P3"] == "FALSIFIED"


def test_e50_p4_falls_without_saturation_or_without_growth():
    assert _verdicts(men.read_e50_commitment(_e50(series_t0={**E50_SERIES_T0, 256: 1.42})))["P4"] == "FALSIFIED"
    assert _verdicts(men.read_e50_commitment(_e50(series_t0={**E50_SERIES_T0, 8: 0.61})))["P4"] == "FALSIFIED"


def test_e50_p5_falls_on_a_shallow_slope():
    assert _verdicts(men.read_e50_commitment(_e50(slope_scale=0.5)))["P5"] == "FALSIFIED"


def _kink(a=men.KINK_ENERGY, b=0.5):
    T = np.array([0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3])
    rows = [dict(T=t, h=h, n_fine=2401, xi_phys=np.exp(a / t + b * np.log(t) + 0.3 + 0.2 * t)) for h in (1 / 16, 1 / 32) for t in T]
    return pd.DataFrame(rows)


def test_e50_kink_confirmed_and_falsified():
    assert set(_verdicts(men.read_e50_kink(_kink())).values()) == {"CONFIRMED"}
    assert _verdicts(men.read_e50_kink(_kink(a=0.85)))["P6"] == "FALSIFIED"
    assert _verdicts(men.read_e50_kink(_kink(b=1.0)))["P7"] == "FALSIFIED"
