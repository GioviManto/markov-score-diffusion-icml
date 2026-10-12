"""tools/make_fig_speciation.py: the figure draws E50's curves and crossing times as the registered reader computes them.

The registered predictions P2 (t_c falls with Theta), P3 (the exact posterior commits at higher noise than its twin),
P4 and P5 (t_c against L and against (1/2) ln(1 + lambda_max)) are re-read from the figure's own tables, so a change that
silently moves a plotted point breaks them. The crossing logic is tested on a curve with a known crossing.
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
import make_fig_speciation as sp  # noqa: E402


def _synthetic(cell="T0.2_h1_L32", twin_shift=0.0):
    """Two reps of a logistic commitment curve in log t that crosses 1/2 at t = 1 (exact) and 1 - twin_shift (twin)."""
    levels = np.geomspace(0.02, 4.0, 48)
    rows = []
    for rep, wiggle in ((0, 0.01), (1, -0.01)):
        c = 1.0 / (1.0 + np.exp(3.0 * np.log(levels)))
        c2 = 1.0 / (1.0 + np.exp(3.0 * (np.log(levels) + twin_shift)))
        rows += [dict(cell=cell, rep=rep, level=l, commit_all=a + wiggle, commit_bulk=a, twin_commit_all=b, twin_commit_bulk=b,
                      lambda_max=10.0, T=0.2, h=1.0, n_sites=32) for l, a, b in zip(levels, c, c2)]
    return pd.DataFrame(rows)


def test_pooling_averages_the_two_replicates_before_anything_else():
    g = sp.pooled(_synthetic())
    assert len(g) == 48
    base = 1.0 / (1.0 + np.exp(3.0 * np.log(g.level.to_numpy())))
    assert np.allclose(g.commit_all.to_numpy(), base)                    # the +-0.01 wiggles cancel in the mean


def test_crossing_is_where_the_pooled_curve_falls_through_one_half():
    g = sp.pooled(_synthetic(twin_shift=0.4))
    assert abs(sp.crossing(g, "T0.2_h1_L32", "commit_all") - 1.0) < 2e-2          # linear interpolation in log t
    assert sp.crossing(g, "T0.2_h1_L32", "twin_commit_all") < sp.crossing(g, "T0.2_h1_L32", "commit_all")


def test_a_missing_cell_raises():
    with pytest.raises(KeyError):
        sp.curve(sp.pooled(_synthetic()), "T9_h1_L32", "commit_all")


@pytest.fixture(scope="module")
def frozen():
    d = pd.read_csv(ROOT / "outputs/E50/commitment.csv")
    return d, sp.pooled(d)


def test_the_plotted_crossings_are_the_registered_readers(frozen):
    d, g = frozen
    _, tc = men.e50_crossings(d)
    tab = sp.temperature_axis(g)
    for r in tab.itertuples():
        c = sp.cell_fixed(r.theta)
        assert r.exact == pytest.approx(tc[c]["all"]) and r.twin == pytest.approx(tc[c]["twin"])
    lab = sp.length_axis(g)
    for r in lab.itertuples():
        c = sp.cell_length(r.L)
        assert r.tc == pytest.approx(tc[c]["bulk"]) and r.lam == pytest.approx(tc[c]["lam"])
        assert r.predicted == pytest.approx(0.5 * np.log1p(tc[c]["lam"]))


def test_the_registered_predictions_hold_on_the_plotted_tables(frozen):
    _, g = frozen
    tab, lab = sp.temperature_axis(g), sp.length_axis(g)
    assert (np.diff(tab.exact.to_numpy()) < 0).all()                                   # P2: t_c falls with Theta
    assert (tab.exact > tab.twin).all()                                                # P3: the exact posterior commits earlier in the reverse process
    assert tab.exact.iloc[0] - tab.twin.iloc[0] > tab.exact.iloc[-1] - tab.twin.iloc[-1]
    tc = lab.set_index("L").tc
    assert all(tc[b] - tc[a] > 0.02 for a, b in ((4, 8), (8, 16), (16, 32), (32, 64)))   # P4: rises with L, then saturates
    assert abs(tc[256] - tc[128]) < 0.03
    low = lab[lab.L <= 16]
    slope = float(np.polyfit(low.predicted, low.tc, 1)[0])                              # P5: slope against (1/2) ln(1 + lambda_max)
    assert 0.8 <= slope <= 1.2


def test_the_figure_is_written_at_full_width_through_the_save_gate(frozen, tmp_path):
    d, _ = frozen
    r = sp.make_figure(d, tmp_path / "fig_speciation.pdf")
    assert (tmp_path / "fig_speciation.pdf").exists() and (tmp_path / "fig_speciation.png").exists()
    assert abs(r["fig"].get_size_inches()[0] - sp.fs.FULL_W) < 1e-9
