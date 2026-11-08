"""E63's reader, driven with made-up tables in both directions: every prediction can come out CONFIRMED and FALSIFIED,
a curve that never reaches the target is infinitely slow, and the gates refuse a missing timing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "core"))
import read_e63 as r  # noqa: E402

CK_EM = (10, 20, 40, 80, 160)
CK_NET = (500, 1000, 2000, 4000, 8000)


def _timing(em_sec, net_sec, bp_use, net_use):
    rows = []
    for n in r.SIZES:
        for a in ("energy", "clean_energy"):
            rows.append(dict(what="update", arm=a, n_chains=n, seconds=em_sec))
        rows.append(dict(what="update", arm="bigru", n_chains=n, seconds=net_sec))
        rows.append(dict(what="use", arm="bigru", n_chains=n, seconds=net_use))
    rows.append(dict(what="use", arm="bp", n_chains=2048, seconds=bp_use))
    return pd.DataFrame(rows)


def _curves(em_errors, net_errors, em_sel, net_sel):
    rows = []
    for seed in r.SEEDS:
        for n in r.SIZES:
            for a in ("energy", "clean_energy"):
                rows += [dict(seed=seed, n_chains=n, arm=a, checkpoint=c, test_all=e, sel_oracle=int(c == em_sel))
                         for c, e in zip(CK_EM, em_errors)]
            rows += [dict(seed=seed, n_chains=n, arm="bigru", checkpoint=c, test_all=e, sel_oracle=int(c == net_sel))
                     for c, e in zip(CK_NET, net_errors)]
    return pd.DataFrame(rows)


def test_all_confirmed():
    # EM reaches BiGRU's 0.06 at checkpoint 20 (20 s) < BiGRU's 8000 x 0.01 = 80 s; EM finishes at 160 s > 80 s
    t = _timing(em_sec=1.0, net_sec=0.01, bp_use=1e-2, net_use=1e-4)
    c = _curves(em_errors=(0.2, 0.05, 0.04, 0.03, 0.025), net_errors=(0.3, 0.2, 0.1, 0.08, 0.06), em_sel=160, net_sel=8000)
    r.gates(t, c)
    v = r.verdicts(t, c)
    assert all(ok for ok, _ in v.values()), v


def test_all_falsified():
    # EM never gets below 0.07 (never reaches 0.06), finishes early; BP use time only 2x
    t = _timing(em_sec=1.0, net_sec=0.01, bp_use=2e-4, net_use=1e-4)
    c = _curves(em_errors=(0.2, 0.1, 0.08, 0.07, 0.07), net_errors=(0.3, 0.2, 0.1, 0.08, 0.06), em_sel=10, net_sel=8000)
    v = r.verdicts(t, c)
    assert not any(ok for ok, _ in v.values()), v


def test_never_reaching_the_target_is_infinitely_slow():
    curve = pd.DataFrame(dict(checkpoint=[1, 2], test_all=[0.5, 0.4], wall=[1.0, 2.0]))
    assert r.time_to(curve, 0.3) == np.inf
    assert r.time_to(curve, 0.45) == 2.0


def test_the_gate_refuses_a_missing_timing():
    t = _timing(1.0, 0.01, 1e-2, 1e-4)
    t = t[~((t.arm == "energy") & (t.n_chains == 2048))]
    c = _curves((0.2, 0.05, 0.04, 0.03, 0.025), (0.3, 0.2, 0.1, 0.08, 0.06), 160, 8000)
    with pytest.raises(AssertionError, match="G1"):
        r.gates(t, c)
