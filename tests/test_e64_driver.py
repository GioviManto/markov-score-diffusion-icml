"""E64's driver: a quick cell writes one row per arm, every arm against the same reference, the frozen kernels are
rebuilt from the shipped table, and the sampler noise is the same for every arm."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
pytest.importorskip("torch")
import e64_generation as e64  # noqa: E402

QUICK = dict(e64.SETTINGS, **e64.QUICK)


def test_the_shipped_table_has_every_kernel():
    import pandas as pd
    d = pd.read_csv(ROOT / e64.SETTINGS["kernels_csv"])
    have = set(zip(d.seed, d.n_chains, d.arm))
    assert {(s, n, a) for s in e64.SETTINGS["seeds"] for n in e64.SETTINGS["sizes"] for a in e64.KERNEL_ARMS} <= have


def test_quick_cell(tmp_path):
    e64.part_gen(QUICK, tmp_path, 1)
    rows = list(csv.DictReader(open(tmp_path / "generation.csv")))
    arms = {r["arm"] for r in rows}
    want = {"exact", "oracle_sampler"} | set(e64.KERNEL_ARMS) | {f"{a}_direct" for a in e64.DIRECT_ARMS} \
        | {f"{a}_cont" for a in QUICK["net_arms"]}
    assert arms == want
    assert len({r["ref_hash"] for r in rows}) == 1
    assert all(r["failed"] == "0" for r in rows)
