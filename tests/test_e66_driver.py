"""E66's driver: a quick screen picks a learning rate from the pooled validation error, the fit writes one row per level
on E03's test bundle, and the trained checkpoints are energy kernels evaluated on the 401-point grid."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
pytest.importorskip("torch")
import e66_bp_dsm as e66  # noqa: E402

QUICK = dict(e66.SETTINGS, **e66.QUICK)


def test_winner_is_the_lowest_pooled_error():
    rows = [dict(lr=1e-3, sq_err=4.0, sq_ref=100.0), dict(lr=1e-3, sq_err=4.0, sq_ref=100.0),
            dict(lr=3e-3, sq_err=1.0, sq_ref=100.0), dict(lr=3e-3, sq_err=9.0, sq_ref=100.0)]
    assert e66._winner(rows)["lr"] == 1e-3                       # pooled 8/200 < 10/200


def test_quick_screen_then_fit(tmp_path):
    e66.part_screen(QUICK, tmp_path, 1)
    assert json.loads((tmp_path / "winners.json").read_text())["lr"] in QUICK["lrs"]
    e66.part_fit(QUICK, tmp_path, 1)
    rows = list(csv.DictReader(open(tmp_path / "kernels.csv")))
    assert len(rows) == len(e66.T_SCHEDULE) * len(QUICK["seeds"]) * len(QUICK["sizes"])
    _, _, _, base = e66.e03._cell_common(QUICK, 0, 32)
    assert {r["test_bundle"] for r in rows} == {base["test_bundle"]}
    assert all(r["failed"] == "0" and 0 < float(r["risk_pooled"]) < 2 for r in rows)
