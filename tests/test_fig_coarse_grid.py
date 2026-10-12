"""tools/make_fig_coarse_grid.py: the data behind the coarse-grid figure (E47) is read the way the registered reader reads it.

The pure functions are tested on constructed frames (both directions: a mismatched test bundle, a failed fit and a
missing reference must raise, never be dropped), the floor and the sign convention by their definition, and the two
learned arms against tools/make_extension_numbers.read_e47_learned on the frozen files.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_extension_numbers as men  # noqa: E402
import make_fig_coarse_grid as cg  # noqa: E402


def _e47(arm="em_pair_quartic", risks=None, n=2048, bundle=None):
    risks = risks or {51: 1.003, 101: 0.999, 201: 1.000}
    rows = [dict(cell=cg.BASE_CELL, seed=s, n_chains=n, arm=arm, selection="risk", grid_size=ng, risk=r * (1 + 0.01 * s),
                 test_bundle=(bundle or f"b{s}")) for ng, r in risks.items() for s in range(4)]
    return pd.DataFrame(rows)


def _e33(arm="em_pair_quartic", n=2048, bundle=None, nan=False):
    rows = [dict(cell=cg.BASE_CELL, seed=s, n_chains=n, arm=arm, selection="risk", risk=(np.nan if nan else 1.0 * (1 + 0.01 * s)),
                 test_bundle=(bundle or f"b{s}")) for s in range(4)]
    return pd.DataFrame(rows)


def test_deviation_clips_to_the_floor_and_marks_the_sign():
    assert cg.deviation(1.0) == (cg.FLOOR, False)
    assert cg.deviation(1.0 + 0.5 * cg.FLOOR) == (cg.FLOOR, False)       # within the floor: drawn ON it, not omitted
    assert cg.deviation(1.0 - 0.5 * cg.FLOOR) == (cg.FLOOR, True)
    d, below = cg.deviation(0.9)
    assert np.isclose(d, 0.1) and below                                  # the coarse fit is better: hollow marker
    d, below = cg.deviation(1.5)
    assert np.isclose(d, 0.5) and not below


def test_bp_ms_per_sequence_divides_by_the_batch():
    rows = [dict(arm="bp_grid", factor="grid_size", batch=512, n_sites=32, grid_size=g, wall_median=w, value=g)
            for g, w in ((41, 0.0215), (101, 0.0550), (401, 0.3415))]
    rows += [dict(arm="bp_grid", factor="grid_size", batch=64, n_sites=32, grid_size=101, wall_median=9.0, value=101)]   # not the registered cell
    ms = cg.bp_ms_per_sequence(pd.DataFrame(rows))
    assert list(ms.index) == [41, 101, 401]
    assert np.allclose(ms.values, np.array([0.0215, 0.0550, 0.3415]) / 512 * 1000)


def test_network_ms_uses_the_batch_rows_and_names_a_missing_network():
    rows = [dict(arm=a, factor="batch", value=512, wall_median=w) for a, w in (("transformer", 0.1), ("bigru", 0.03), ("window", 0.012))]
    rows += [dict(arm="transformer", factor="n_sites", value=512, wall_median=99.0)]
    out = cg.network_ms_per_sequence(pd.DataFrame(rows))
    assert np.isclose(out["transformer"], 0.1 / 512 * 1000) and set(out) == {"transformer", "bigru", "window"}
    with pytest.raises(ValueError, match="bigru"):
        cg.network_ms_per_sequence(pd.DataFrame(rows[:1] + rows[2:3]))


def test_learned_ratio_is_the_ratio_of_seed_means_on_the_same_seeds():
    r = cg.learned_ratios(_e47(), _e33(), "em_pair_quartic")
    # risk(ng) = c_ng (1 + 0.01 s) and the reference is (1 + 0.01 s): the ratio of seed means is c_ng exactly
    assert np.allclose(r.values, [1.003, 0.999, 1.000]) and list(r.index) == [51, 101, 201]


def test_learned_ratio_refuses_a_different_test_bundle():
    with pytest.raises(ValueError, match="test bundles"):
        cg.learned_ratios(_e47(), _e33(bundle="other"), "em_pair_quartic")


def test_learned_ratio_refuses_a_failed_fit():
    with pytest.raises(ValueError, match="failed"):
        cg.learned_ratios(_e47(), _e33(nan=True), "em_pair_quartic")


def test_headline_ratio_is_against_the_401_point_value_and_needs_it():
    rows = [dict(seed=s, n_chains=2048, n_grid=ng, half_width=8.0, risk_em_C8=r * (1 + 0.1 * s))
            for ng, r in ((401, 0.2), (201, 0.2002), (101, 0.21), (51, 0.4)) for s in range(3)]
    d = pd.DataFrame(rows)
    r = cg.headline_ratios(d, "risk_em_C8")
    assert list(r.index) == [51, 101, 201] and np.allclose(r.values, [2.0, 1.05, 1.001])
    with pytest.raises(ValueError, match="401"):
        cg.headline_ratios(d[d.n_grid != 401], "risk_em_C8")


def test_dodge_offsets_are_symmetric_and_distinct():
    o = cg.dodge_offsets(4)
    assert np.allclose(o, -o[::-1]) and len(set(np.round(o, 6))) == 4


# ---------------------------------------------------------------- against the registered reader, on the frozen files
@pytest.fixture(scope="module")
def frozen():
    d47b = cg._load("outputs/hpc_E47b_s*/kernels.csv")
    d33 = {"q": cg._load("outputs/hpc_E33q_s*/kernels.csv"), "k": cg._load("outputs/hpc_E33k_s*/kernels.csv")}
    return d47b, d33


def test_learned_ratios_reproduce_the_registered_reader(frozen):
    d47b, d33 = frozen
    rows = men.read_e47_learned(d47b, pd.concat([d33["k"], d33["q"]], ignore_index=True))
    seen = {}
    for r in rows:
        for m in re.finditer(r"(T0\.3_h1_)/M(\d+)/N(\d+)/(pair|tilt): ([0-9.]+)", r["note"]):      # the base cell only
            seen[(int(m.group(2)), int(m.group(3)), m.group(4))] = float(m.group(5))
    short = {"em_pair_quartic": "pair", "em_tilted_spline": "tilt"}
    checked = 0
    for arm, which in (("em_pair_quartic", "q"), ("em_tilted_spline", "k")):
        for ng, ratio in cg.learned_ratios(d47b, d33[which], arm).items():
            key = (cg.M_LEARNED, ng, short[arm])
            if key in seen:
                assert abs(seen[key] - ratio) < 6e-4, (key, seen[key], ratio)       # the reader prints three decimals
                checked += 1
    assert checked >= 4


def test_frozen_figure_data_have_the_expected_shape(frozen, tmp_path):
    scaling = pd.read_csv(ROOT / "outputs/hpc_E47t/scaling.csv")
    bp = cg.bp_ms_per_sequence(scaling)
    assert list(bp.index) == [41, 51, 61, 81, 101, 151, 201, 401] and bp.is_monotonic_increasing
    nets = cg.network_ms_per_sequence(scaling)
    assert nets["window"] < nets["bigru"] < nets["transformer"]
    d47c = pd.read_csv(ROOT / "outputs/E47c/refinement.csv")
    for col, _ in cg.HEADLINE_ARMS:
        r = cg.headline_ratios(d47c, col)
        assert list(r.index) == [51, 101, 151, 201] and (r.values >= 1.0 - 1e-3).all()
        assert r.loc[51] > r.loc[101] > r.loc[151]                      # a coarser grid never helps the mixtures
    d47b, d33 = frozen
    r = cg.make_figure(scaling, d47b, d33, d47c, tmp_path / "fig_grid_coarse.pdf")          # fs.save: no prose, symbols in math
    assert (tmp_path / "fig_grid_coarse.pdf").exists() and (tmp_path / "fig_grid_coarse.png").exists()
    assert abs(r["fig"].get_size_inches()[0] - cg.fs.FULL_W) < 1e-9
