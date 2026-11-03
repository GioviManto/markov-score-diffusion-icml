"""E64's reader with made-up generation tables, in both directions; the rebuild gate refuses a kernel that does not
score its frozen row."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "core"))
import read_e64 as r  # noqa: E402


def _table(kernel_sw2, net_sw2, oracle_sw2, kernel_kurt, net_kurt, kernel_gap, net_gap, rebuilt=0.05):
    rows = []
    for seed in r.SEEDS:
        for n in r.SIZES:
            def row(arm, sw2, kurt, gap, **x):
                rows.append(dict(seed=seed, n_chains=n, arm=arm, sw2=sw2, energy=0.0, w1_marginal=0.0, cov_fro=0.0,
                                 kurt_gen=3.0 + kurt, kurt_ref=3.0, acf_err_max=0.0, loglik_gap=gap, failed=0,
                                 ref_hash=f"h{seed}{n}", frozen_risk=x.get("fr"), rebuilt_risk=x.get("rb")))
            row("exact", 0.10, 0.0, 0.0)
            row("oracle_sampler", oracle_sw2, 0.0, 0.0)
            row("clean_energy", kernel_sw2, kernel_kurt, kernel_gap, fr=0.05, rb=rebuilt)
            for i, net in enumerate(r.NETS):
                row(net, net_sw2 + 0.01 * i, net_kurt, net_gap)
    return pd.DataFrame(rows).assign(abs_kurt=lambda x: (x.kurt_gen - x.kurt_ref).abs(), abs_gap=lambda x: x.loglik_gap.abs())


def test_all_confirmed():
    d = _table(kernel_sw2=0.12, net_sw2=0.30, oracle_sw2=0.11, kernel_kurt=-0.1, net_kurt=-1.0, kernel_gap=-0.05, net_gap=2.0)
    r.gates(d)
    v = r.verdicts(d)
    assert all(ok for ok, _ in v.values()), v


def test_all_falsified():
    d = _table(kernel_sw2=0.40, net_sw2=0.30, oracle_sw2=0.30, kernel_kurt=-1.5, net_kurt=-0.1, kernel_gap=-3.0, net_gap=0.5)
    v = r.verdicts(d)
    assert not any(ok for ok, _ in v.values()), v


def test_the_gap_is_read_in_absolute_value():
    d = _table(kernel_sw2=0.12, net_sw2=0.30, oracle_sw2=0.11, kernel_kurt=-0.1, net_kurt=-1.0, kernel_gap=-5.0, net_gap=1.0)
    assert not r.verdicts(d)["P4"][0]                    # -5 is a bigger error than +1, not a smaller one


def test_the_rebuild_gate():
    with pytest.raises(AssertionError, match="G1"):
        r.gates(_table(0.12, 0.30, 0.11, -0.1, -1.0, -0.05, 2.0, rebuilt=0.06))
