"""E33 amendment 3: a fit that fails is a FAILED row, and the cell's other arms survive it.

On 2-3 Oct 2026 one fit on the L128, M = 2048 cell raised FloatingPointError ('Pairwise belief
lost all mass') inside EM's E-step, and the exception stopped all four kernel jobs (678087-678090),
losing every cell in flight. Pinned here, on the quick configuration: the failing arm gives two
rows (risk and evidence selection) with risk NaN, failed = 1 and the exception in stop_reason;
the other arm in the same cell is fitted and kept with failed = 0; and a ladder checkpoint whose
validation scoring raises is not selectable (the next one is selected instead).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import e33_field_learning as e33

CELL = "T0.3_h1_L32"


def _cfg(arms):
    cfg = dict(e33.SETTINGS)
    cfg.update(e33.QUICK)
    cfg["arms"] = arms
    return cfg


def test_a_failed_fit_is_a_failed_row_and_the_other_arm_survives(monkeypatch):
    real = e33.fit_em

    def fit_em(kernel, *a, **k):
        if kernel.name.startswith("tilted_legendre"):
            raise FloatingPointError("Pairwise belief lost all mass.")
        return real(kernel, *a, **k)

    monkeypatch.setattr(e33, "fit_em", fit_em)
    rows = e33._kernels_cell((_cfg(("em_mixture", "em_tilted_poly")), CELL, 10_000, 32))
    poly = [r for r in rows if r["arm"] == "em_tilted_poly"]
    mix = [r for r in rows if r["arm"] == "em_mixture"]
    assert sorted(r["selection"] for r in poly) == ["evidence", "risk"]
    assert all(r["failed"] == 1 and np.isnan(r["risk"]) for r in poly)
    assert all("FloatingPointError" in r["stop_reason"] and "lost all mass" in r["stop_reason"] for r in poly)
    assert all(r["n_params"] > 0 for r in poly)
    assert len(mix) == 2 and all(r["failed"] == 0 and np.isfinite(r["risk"]) for r in mix)


def test_a_checkpoint_that_cannot_be_scored_is_not_selectable(monkeypatch):
    real, real_common = e33.grid_bp_batch, e33._common
    state = {"armed": False, "n": 0}

    def _common(*a, **k):  # the oracle reference also runs grid_bp_batch: arm the trap after it
        out = real_common(*a, **k)
        state["armed"] = True
        return out

    def grid_bp_batch(*a, **k):
        if state["armed"]:
            state["n"] += 1
            if state["n"] == 1:  # the first validation level of the first checkpoint
                raise FloatingPointError("Forward message 3 lost all mass.")
        return real(*a, **k)

    monkeypatch.setattr(e33, "_common", _common)
    monkeypatch.setattr(e33, "grid_bp_batch", grid_bp_batch)
    rows = e33._kernels_cell((_cfg(("em_mixture",)), CELL, 10_000, 32))
    risk_row = next(r for r in rows if r["selection"] == "risk")
    first, last = sorted(int(c) for c in e33.QUICK["em_checkpoints"])
    assert risk_row["checkpoint"] == last and risk_row["failed"] == 0 and np.isfinite(risk_row["risk"])
    assert f"{first}:inf" in risk_row["val_curve"]


def test_other_exceptions_still_stop_the_cell(monkeypatch):
    """Only FloatingPointError is a failed fit (amendment 3); anything else is a bug and stays loud."""
    def fit_em(*a, **k):
        raise ValueError("a bug, not a failed fit")

    monkeypatch.setattr(e33, "fit_em", fit_em)
    with pytest.raises(ValueError):
        e33._kernels_cell((_cfg(("em_mixture",)), CELL, 10_000, 32))
