"""E63's driver: a quick timing run writes one positive timing per (arm, N) for every update kind and the use times,
and the EM arms are the frozen recipes (E30's mixture, E56's spline and energy)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
pytest.importorskip("torch")
import e63_compute as e63  # noqa: E402

from markov_score.energy_kernel import NeuralEnergyKernel  # noqa: E402
from markov_score.kernels import MixtureInnovationKernel  # noqa: E402

QUICK = dict(e63.SETTINGS, **e63.QUICK)


def test_the_em_arms_are_the_frozen_recipes():
    assert isinstance(e63._em_kernel(QUICK, "em_bp", 32), MixtureInnovationKernel)
    k = e63._em_kernel(QUICK, "energy", 32)
    assert isinstance(k, NeuralEnergyKernel) and k.hidden == 16
    assert e63._em_kernel(QUICK, "spline", 32).name.startswith("shift")


def test_timing_writes_every_arm_with_positive_seconds(tmp_path):
    e63.part_timing(QUICK, tmp_path, 1)
    rows = list(csv.DictReader(open(tmp_path / "timing.csv")))
    updates = {(r["arm"], int(r["n_chains"])) for r in rows if r["what"] == "update"}
    want = {(a, n) for a in QUICK["em_arms"] + QUICK["clean_arms"] + QUICK["nets"] for n in QUICK["sizes"]}
    assert updates == want
    uses = {r["arm"] for r in rows if r["what"] == "use"}
    assert uses == set(QUICK["nets"]) | {"bp"}
    assert all(float(r["seconds"]) > 0 for r in rows)
    units = {r["arm"]: r["unit"] for r in rows if r["what"] == "update"}
    assert units["em_bp"] == "em_iteration" and units["clean_energy"] == "clean_m_step" and units["bigru"] == "sgd_step"
