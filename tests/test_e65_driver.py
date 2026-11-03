"""E65's driver: every (seed, N, arm) curve flags exactly one checkpoint per rule; the evidence is MAXIMISED and every
other criterion minimised; the networks are E03's winners with E03's seed stream; the clean validation chains are the
ones behind E03's validation bundle."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
pytest.importorskip("torch")
import e65_selection as e65  # noqa: E402

QUICK = dict(e65.SETTINGS, **e65.QUICK)


def test_flag_maximises_evidence_and_minimises_the_rest():
    rows = [dict(checkpoint=c, val_all=v, criterion=k) for c, v, k in ((10, 0.3, -5.0), (20, 0.2, -4.0), (40, 0.25, -6.0))]
    out = e65._flag([dict(r) for r in rows], "val_all", "heldout_evidence")
    assert [r["checkpoint"] for r in out if r["sel_oracle"]] == [20]
    assert [r["checkpoint"] for r in out if r["sel_free"]] == [20]          # the largest evidence
    out = e65._flag([dict(r) for r in rows], "val_all", "val_dsm")
    assert [r["checkpoint"] for r in out if r["sel_free"]] == [40]          # the smallest loss


def test_the_clean_validation_chains_are_e03s():
    from markov_score.bp_grid import make_grid
    from markov_score.noising import alpha_delta
    a_val = e65._a_val(QUICK, 0)
    _, val, _, _ = e65.e03._cell_common(QUICK, 0, 32)
    t = e65.T_SCHEDULE[0]
    X, _ = val[t]
    al, de = alpha_delta(t)
    resid = (X - al * a_val) / np.sqrt(de)                 # the bundle's own noise: standard normal, not a mismatch
    assert abs(resid.std() - 1.0) < 0.2 and abs(resid.mean()) < 0.2


def test_quick_run_flags_one_checkpoint_per_rule(tmp_path):
    e65.part_kernels(QUICK, tmp_path, 1)
    e65.part_nets(QUICK, tmp_path, 1)
    rows = list(csv.DictReader(open(tmp_path / "curves_kernels.csv"))) + list(csv.DictReader(open(tmp_path / "curves_nets.csv")))
    arms = {r["arm"] for r in rows}
    assert arms == set(QUICK["kernel_arms"]) | set(QUICK["nets"])
    for rule in ("sel_oracle", "sel_free"):
        c = Counter((r["seed"], r["n_chains"], r["arm"]) for r in rows if r[rule] == "1")
        assert set(c.values()) == {1} and len(c) == len(arms)
    assert all(np.isfinite(float(r["test_all"])) for r in rows)
    crit = {r["arm"]: r["criterion_name"] for r in rows}
    assert crit["em_bp"] == "heldout_evidence" and crit["clean_energy"] == "heldout_clean_nll" and crit["bigru"] == "val_dsm"
