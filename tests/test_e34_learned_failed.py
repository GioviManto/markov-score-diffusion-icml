"""E34 physics_learned on a FAILED E33 fit (E33 amendment 3).

A failed fit reaches E34 as a kernels.csv row with failed = 1 and no theta. physics_learned rebuilt
every kernel from theta and would have crashed on it, and with it the whole part. Pinned: the
failed fit gives a row with failed = 1 and NaN statistics (so E34's P4 counts it against the
prediction), the other fits of the cell are sampled as before, and a row written before the
amendment (no `failed` value, read back as NaN) is a normal fit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import e34_field_physics as e34

CELL = "T0.3_h1_L32"
# a real fit: E33 pair shard, T0.3_h1_L32, seed 0, M = 2048 (outputs/hpc_E33q_s0/kernels.csv, job 678260)
GOOD = {"arm": "em_pair_quartic", "selection": "risk", "theta": "-0.8021042;0.0069176617;-2.2433916;-0.016062394;3.0059337", "failed": float("nan")}
FAILED = {"arm": "em_tilted_poly", "selection": "risk", "theta": float("nan"), "failed": 1}


def _cfg():
    cfg = dict(e34.SETTINGS)
    cfg.update(e34.QUICK)
    return cfg


def test_a_failed_fit_is_a_failed_row_and_the_others_are_sampled():
    rows = e34._learned_cell((_cfg(), CELL, 10_000, 2048, [GOOD, FAILED]))
    by_arm = {r["arm"]: r for r in rows}
    assert set(by_arm) == {"truth", "em_pair_quartic", "em_tilted_poly"}
    bad = by_arm["em_tilted_poly"]
    assert bad["failed"] == 1 and np.isnan(bad["kink_density"]) and np.isnan(bad["mean_domain"])
    good = by_arm["em_pair_quartic"]
    assert good["failed"] == 0 and np.isfinite(good["kink_density"]) and good["kink_density"] > 0
    assert set(bad) == set(good)  # one column set, so learned.csv stays rectangular
