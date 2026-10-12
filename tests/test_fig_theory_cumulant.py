"""tools/make_fig_theory.py, panel (b): information about the cumulants kappa_2, kappa_4 and the numbers quoted from it.

The panel is a change of coordinates of E01's FROZEN shape_information.csv, not a relabel, and the macros it emits
(paper/sections/cumulant-numbers.tex) must (i) be exactly what a constructed curve with known slopes implies, (ii) agree with the
ledger's 163.7 and the relabel range 1.6-1526 of docs/JEROME_CALL_2026-09-25.md Q10, and (iii) leave theory-numbers.tex alone.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import make_fig_theory as th  # noqa: E402

T_LEVELS = [0.05, 0.1, 0.5, 1.0, 2.2, 3.0, 3.5, 4.0]


def _frame(k2_rate=4.0, k4_rate=8.0, relabel=(1.6, 1526.0)):
    rows = []
    for beta in th.BETAS:
        for rep in range(3):
            for t in T_LEVELS:
                rows.append(dict(beta=beta, rep=rep, t=t, eff_kappa2=100.0 * np.exp(-k2_rate * t) * (1 + 0.01 * rep),
                                 eff_kappa4=10.0 * np.exp(-k4_rate * t) * (1 + 0.01 * rep),
                                 relabel_ratio=relabel[0] if (beta == 1.0 and t == 0.05) else (relabel[1] if (beta == 3.0 and t == 4.0) else 10.0)))
    return pd.DataFrame(rows)


def test_slopes_recover_the_constructed_rates_above_t_star_plus_one():
    m = th.cumulant_macros(_frame(), t_star=1.17)
    assert float(m["cumSlopeKappatwoBetaone"]) == pytest.approx(-4.0, abs=0.006)
    assert float(m["cumSlopeKappafourBetaone"]) == pytest.approx(-8.0, abs=0.006)
    assert {"cumSlopeKappatwoBetaoneptwofive", "cumSlopeKappafourBetaonepfive", "cumSlopeKappafourBetathree"} <= set(m)


def test_decay_ratio_is_the_ratio_of_ratios_between_the_two_ends():
    # eff4 / eff2 = 0.1 exp(-4 t): between t = 0.05 and 4 it falls by exp(4 * 3.95)
    m = th.cumulant_macros(_frame(), t_star=1.17)
    assert float(m["cumDecayRatio"]) == pytest.approx(np.exp(4 * 3.95), rel=1e-3)
    assert float(m["cumDecayRatioThree"]) == pytest.approx(np.exp(4 * 2.95), rel=1e-3)
    assert float(m["cumRatioLow"]) == pytest.approx(0.1 * np.exp(-4 * 0.05), abs=5e-4)


def test_relabel_range_is_the_median_over_replicates_then_min_and_max():
    m = th.cumulant_macros(_frame(relabel=(1.6, 1526.0)), t_star=1.17)
    assert m["cumRelabelMin"] == "1.6" and m["cumRelabelMax"] == "1526"


def test_macro_names_have_no_digits_and_do_not_collide_with_the_existing_fragment():
    m = th.cumulant_macros(_frame(), t_star=1.17)
    names = set(m)
    assert all(n.isalpha() for n in names)                                  # \newcommand names are letters only
    existing = (ROOT / "paper/sections/theory-numbers.tex").read_text()
    assert not any(f"\\{n}}}" in existing for n in names)


def test_write_fragment_is_a_generated_fragment(tmp_path):
    p = tmp_path / "cumulant-numbers.tex"
    th.write_fragment({"cumDecayRatio": "163.7"}, 1.17, p)
    head = "\n".join(p.read_text().splitlines()[:8]).lower()
    assert "generated" in head                                                # the typed-literal gate skips a file that says so
    assert "\\newcommand{\\cumDecayRatio}{163.7}" in p.read_text()


# ---------------------------------------------------------------- on the frozen files
@pytest.fixture(scope="module")
def shape():
    return pd.read_csv(ROOT / "outputs/E01/shape_information.csv")


def test_cumulant_frame_matches_its_derived_copy_and_keeps_the_unit_jacobian_rows(shape):
    cum = th.cumulant_frame(shape)                  # raises SystemExit if outputs/E01/cumulant_information.csv drifted from the transform
    assert len(cum) == len(shape)
    assert np.allclose(cum.eff_rho.to_numpy(), shape.eff_rho.to_numpy(), rtol=1e-9)         # rho: an exact relabel
    assert np.allclose(cum.eff_kappa2.to_numpy(), shape.eff_q.to_numpy(), rtol=1e-9)        # kappa_2 = q: an exact relabel
    assert not np.allclose(cum.eff_kappa4.to_numpy(), shape.eff_beta.to_numpy(), rtol=1e-3)  # kappa_4 is NOT beta relabelled


def test_the_ledger_numbers_are_reproduced_on_the_frozen_file(shape):
    g = pd.read_csv(ROOT / "outputs/E01/gaussian_information.csv")
    t_star = float(g[g.rho == 0.85].t_star.iloc[0])
    m = th.cumulant_macros(th.cumulant_frame(shape), t_star)
    assert m["cumDecayRatio"] == "163.7"                                      # Q10: 'the fourth cumulant falls 163.7 times more'
    assert m["cumRelabelMin"] == "1.6" and m["cumRelabelMax"] == "1526"       # Q10: 'wrong by a factor between 1.6 and 1526'
    # fourth-order information above t_* + 1 falls at about e^{-8t} where the estimator is above its floor (beta >= 2)
    assert float(m["cumSlopeKappafourBetatwo"]) == pytest.approx(-7.94, abs=0.01)
    assert float(m["cumSlopeKappatwoBetaone"]) == pytest.approx(-4.0, abs=0.05)


def test_theory_numbers_macros_are_unchanged_by_this_work():
    """shapeSlopeRhoone, shapeSlopeBetatwo, shapeSlopeBetathree and the rest keep their values: the new numbers live in a new fragment."""
    t = (ROOT / "paper/sections/theory-numbers.tex").read_text()
    for name, val in (("shapeSlopeRhoone", "-4.00"), ("shapeSlopeBetatwo", "-7.94"), ("shapeSlopeBetathree", "-8.01"),
                      ("shapeSlopeBetaone", "-2.59"), ("shapeReps", "8"), ("shapeChains", "512")):
        assert f"\\newcommand{{\\{name}}}{{{val}}}" in t
