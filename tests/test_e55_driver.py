"""E55's driver against E33's: the same data, array for array, and the same kernel fit, number for number, on the CPU;
the registered screen rule; the site-energy fits E55 adds are exactly those E33 lacks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import e14_regime_baselines as e14  # noqa: E402
import e33_field_learning as e33  # noqa: E402
import e55_unet_energy as e55  # noqa: E402

SMALL = dict(n_val=24, n_test=12, grid_size=81)          # >= 12 chains: one-view groups span the 12 levels


def _cfg(**kw):
    c = dict(e33.SETTINGS); c.update(SMALL); c.update(kw)
    return c


def test_common_reproduces_e33_common_on_an_e33_cell():
    cfg = _cfg()
    a = e33._common(cfg, "T0.5_iso_L32", 3, 16)
    b = e55.common(cfg, "T0.5_iso_L32", 3, 16, np)
    for i in (2, 3, 4, 5, 6, 7):                       # w, log_k, log_mu, sigma0, A, Av
        np.testing.assert_array_equal(a[i], b[i])
    for bundle_a, bundle_b in ((a[8], b[8]), (a[9], b[9])):
        for t in e33.T_SCHEDULE:
            np.testing.assert_array_equal(bundle_a[t][0], bundle_b[t][0])
            np.testing.assert_array_equal(bundle_a[t][1], bundle_b[t][1])
    assert {k: v for k, v in b[10].items() if k in a[10]} == a[10]
    assert set(b[10]) - set(a[10]) == {"val_draws", "test_draws"}
    assert e55.draws_hash(b[9]) == b[10]["test_draws"] != b[10]["test_bundle"]


def test_the_longer_chains_are_the_base_field():
    chain_a, info_a = e55.resolve_cell(e55.find_cell("T0.3_h1_L32"))
    chain_b, info_b = e55.resolve_cell(e55.find_cell("T0.3_h1_L512"))
    assert (info_a["T"], info_a["h"], info_a["rho1"]) == (info_b["T"], info_b["h"], info_b["rho1"])
    assert info_b["n_sites"] == 512


def test_fit_kernel_reproduces_e33_fit_arm_for_the_site_energy():
    cfg = _cfg(em_checkpoints=(2, 4))
    cell, seed, n = "T0.3_h1_L32", 2, 24
    chain, grid, w, log_k, log_mu, sigma0, A, Av, val, test, base = e33._common(cfg, cell, seed, n)
    L = base["n_sites"]
    groups = e14._one_view_groups(A, f"e33-em-{cell}-M{n}", seed, L)
    val_groups = e14._one_view_groups(Av, f"e33-emval-{cell}", seed, L)
    ref = e33._fit_arm(cfg, "em_pair_quartic", grid, w, groups, val, val_groups, test, seed, cell, L, n)
    k0 = e33._init_kernel("em_pair_quartic", cfg, grid, seed, cell, n)
    got = e55.fit_kernel(cfg, "em_pair_quartic", k0, grid, w, groups, val, val_groups, test, L, cfg["em_checkpoints"], np)
    assert [r["selection"] for r in ref] == [r["selection"] for r in got]
    for r, g in zip(ref, got):
        assert r["checkpoint"] == g["checkpoint"]
        assert abs(r["risk"] - g["risk"]) <= 1e-12 * abs(r["risk"])
        assert abs(r["val_selected"] - g["val_selected"]) <= 1e-12 * abs(r["val_selected"])


def _row(cell, arm, conf, n, v):
    import json
    return dict(cell=cell, screen_arm=arm, conf_key=json.dumps(conf, sort_keys=True), n_chains=n, val_selected=v)


def test_screen_rule_mean_log_risk_complete_and_finite_ties_to_the_smaller():
    small, mid, big = (dict(base=32, lr=2e-4), dict(base=64, lr=2e-4), dict(base=128, lr=2e-4))
    rows = [_row("c", "unet", small, 32, 0.4), _row("c", "unet", small, 2048, 0.1),     # mean log = log 0.2
            _row("c", "unet", mid, 32, 0.8), _row("c", "unet", mid, 2048, 0.05),        # mean log = log 0.2 (tie)
            _row("c", "unet", big, 2048, 0.01)]                                         # missing a size: cannot win
    w = e55.choose_winners(rows)
    assert w["c|unet"]["conf"] == small
    rows += [_row("c", "unet", big, 32, float("nan"))]                                   # complete but non-finite
    assert e55.choose_winners(rows)["c|unet"]["conf"] == small
    rows = [_row("c", "unet", small, 32, 0.4), _row("c", "unet", small, 2048, 0.1),
            _row("c", "unet", mid, 32, 0.3), _row("c", "unet", mid, 2048, 0.1)]
    assert e55.choose_winners(rows)["c|unet"]["conf"] == mid


def test_pair_targets_are_exactly_what_e33_lacks():
    t = set(e55.pair_targets(dict(e55.SETTINGS)))
    for c in e55.E33_CELLS:
        for n in (32, 128, 512, 2048):
            assert (c, n) not in t
        assert (c, 8192) in t
    assert ("T0.3_h1_L32", 32768) in t
    assert {n for c, n in t if c == "T0.3_h1_L512"} == {32, 128, 512, 2048}
    assert {n for c, n in t if c == "T0.3_h1_L128"} == {32, 128, 512, 2048, 8192}


def test_the_largest_fits_have_about_a_million_transitions():
    for c, sizes in e55.SIZES.items():
        L = e55.find_cell(c)["L"]
        assert max(sizes) * (L - 1) <= 1.1e6
