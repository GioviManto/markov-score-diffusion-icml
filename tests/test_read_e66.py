"""E66's reader, driven with made-up tables in both directions, and a missing seed counting against."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "core"))
import read_e66 as r  # noqa: E402


def _table(bp, clean, noisy, net):
    rows = []
    for seed in r.SEEDS:
        for n in r.SIZES:
            rows += [dict(arm="bp_dsm", n_chains=n, seed=seed, risk=bp),
                     dict(arm="e58_clean_energy", n_chains=n, seed=seed, risk=clean),
                     dict(arm="e56_noisy_energy", n_chains=n, seed=seed, risk=noisy)]
            rows += [dict(arm=a, n_chains=n, seed=seed, risk=net + 0.01 * i) for i, a in enumerate(r.NETS)]
    return pd.DataFrame(rows)


def test_all_confirmed():
    v = r.verdicts(_table(bp=0.03, clean=0.025, noisy=0.06, net=0.08))
    assert all(ok for ok, _ in v.values()), v


def test_all_falsified():
    v = r.verdicts(_table(bp=0.09, clean=0.02, noisy=0.05, net=0.08))
    assert not any(ok for ok, _ in v.values()), v


def test_a_missing_seed_counts_against():
    df = _table(bp=0.03, clean=0.025, noisy=0.06, net=0.08)
    df.loc[(df.arm == "bp_dsm") & (df.seed < 5), "risk"] = np.nan
    v = r.verdicts(df)
    assert not v["P1"][0] and not v["P2"][0] and not v["P3"][0]
