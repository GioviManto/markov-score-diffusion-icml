"""E67's driver: a quick run of every part writes every arm, the bundles are cached and reused, the LMMSE arm uses the
chain's covariance, and the timing part records one row per EM arm."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
pytest.importorskip("torch")
import e67_vector_state as e67  # noqa: E402

QUICK = dict(e67.SETTINGS, **e67.QUICK)


def test_quick_parts(tmp_path):
    e67.part_kernels(QUICK, tmp_path, 1)
    e67.part_nets(QUICK, tmp_path, 1)
    e67.part_timing(QUICK, tmp_path, 1)
    arms = {r["arm"] for r in csv.DictReader(open(tmp_path / "arms.csv"))} | {r["arm"] for r in csv.DictReader(open(tmp_path / "nets.csv"))}
    assert arms == set(QUICK["kernel_arms"]) | set(QUICK["net_arms"])
    assert {r["arm"] for r in csv.DictReader(open(tmp_path / "timing.csv"))} == {"var1", "energy2d"}
    assert (tmp_path / "bundles" / "s0.npz").exists()
    rows = list(csv.DictReader(open(tmp_path / "arms.csv")))
    assert all(0 < float(r["risk_pooled"]) < 1.5 for r in rows)
