"""E65's reader, driven with made-up tables in both directions: the reproduction gate passes and fails, an arm that
fails it voids its other verdicts, and P2-P4 come out CONFIRMED and FALSIFIED."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "core"))
import read_e65 as r  # noqa: E402


def _tables(kernel_oracle=0.03, kernel_free=0.031, net_oracle=0.10, net_free=0.105, frozen_shift=1.0):
    rows, frozen = [], []
    for seed in r.SEEDS:
        for n in r.SIZES:
            for arm in r.KERNEL_ARMS + r.NETS:
                is_net = arm in r.NETS
                lo, hi = (net_oracle, net_free) if is_net else (kernel_oracle, kernel_free)
                # checkpoint 1 is the oracle's pick, checkpoint 2 the oracle-free criterion's
                rows.append(dict(seed=seed, n_chains=n, arm=arm, checkpoint=1, test_all=lo, val_all=lo, criterion=0.0,
                                 sel_oracle=1, sel_free=0, failed=0, test_bundle=f"b{seed}_{n}"))
                rows.append(dict(seed=seed, n_chains=n, arm=arm, checkpoint=2, test_all=hi, val_all=hi * 1.1, criterion=1.0,
                                 sel_oracle=0, sel_free=1, failed=0, test_bundle=f"b{seed}_{n}"))
                frozen.append(dict(arm=arm, seed=seed, n_chains=n, risk=lo * frozen_shift, test_bundle=f"b{seed}_{n}"))
    return pd.DataFrame(rows), pd.DataFrame(frozen)


def test_all_confirmed():
    c, f = _tables()
    r.gates(c, f)
    v = r.verdicts(c, f)
    assert all(verdict == "CONFIRMED" for verdict, _ in v.values()), v


def test_free_selection_costs_too_much_and_networks_win():
    c, f = _tables(kernel_oracle=0.03, kernel_free=0.2, net_oracle=0.10, net_free=0.15)
    v = r.verdicts(c, f)
    assert v["P1"][0] == "CONFIRMED"
    assert v["P2"][0] == "FALSIFIED" and v["P3"][0] == "FALSIFIED" and v["P4"][0] == "FALSIFIED", v


def test_a_failed_reproduction_voids_the_rest():
    c, f = _tables(frozen_shift=1.10)
    v = r.verdicts(c, f)
    assert v["P1"][0] == "FALSIFIED"
    assert v["P2"][0] == v["P3"][0] == v["P4"][0] == "VOID"


def test_a_missing_seed_makes_the_seed_mean_undefined():
    c, f = _tables()
    c = c[~((c.arm == "energy") & (c.seed == 3))]
    o = r.selected(c, "oracle")
    assert pd.isna(r.seed_mean(o, "energy", 2048))
