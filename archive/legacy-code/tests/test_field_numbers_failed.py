"""E33 amendment 3 in the reading tool: a failed fit can falsify a clause and never confirm one.

A failed row (risk NaN, failed = 1) enters every paired table as a NaN ratio: it is counted in n,
never in k, means skip it, and any clause whose group holds one is not confirmed even when the
mean of the remaining fits would pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import make_field_numbers as mfn


def _rows(arm, risks, failed=None):
    d = pd.DataFrame({"cell": "c", "seed": range(len(risks)), "n_chains": 2048, "test_bundle": "b",
                      "arm": arm, "selection": "risk", "risk": risks})
    if failed is not None:
        d["failed"] = failed
    return d


def test_mark_failed_reads_the_column_and_nan_risk():
    old = mfn.mark_failed(_rows("a", [0.1, 0.2]))  # a shard written before the amendment
    assert list(old.failed) == [0, 0]
    new = mfn.mark_failed(_rows("a", [0.1, np.nan], failed=[0, 1]))
    assert list(new.failed) == [0, 1]
    nan_only = mfn.mark_failed(_rows("a", [np.nan, 0.3]))
    assert list(nan_only.failed) == [1, 0]


def test_a_failed_fit_is_counted_but_never_a_success():
    a = mfn.mark_failed(_rows("a", [0.1, 0.1, np.nan], failed=[0, 0, 1]))
    b = mfn.mark_failed(_rows("b", [0.2, 0.2, 0.2]))
    j = mfn.pair(a, b)
    assert len(j) == 3 and mfn.nf(j) == 1
    assert int((j.ratio < 1).sum()) == 2  # k = 2 of n = 3: the clause "a below b in every seed" fails
    assert np.isclose(j.ratio.mean(), 0.5)  # the mean is over the fits that did not fail
    (s,) = mfn.summarise(j, "a/b")
    assert s["n_failed"] == 1 and s["n_seeds"] == 3 and np.isfinite(s["ci_lo"])


def test_a_clause_with_a_failure_is_not_confirmed():
    R = pd.DataFrame([dict(ok=True, n_failed=1), dict(ok=True, n_failed=0), dict(ok=False, n_failed=0), dict(ok=None, n_failed=2)])
    assert list(mfn.finalize_ok(R).ok) == [False, True, False, None]
