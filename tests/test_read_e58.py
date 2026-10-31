"""E58's reader, tested before any E58 data exist: each registered prediction holds on a made-up table built to satisfy
it and flips on one built to break it, and a failed fit counts AGAINST every prediction it enters."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "core"))
import read_e58 as r  # noqa: E402

E03_ARCHS = ("window", "mlp", "bigru", "tcn", "transformer", "unet")


def world(cell, arm, m):
    """Every prediction true: the clean energy kernel beats the networks everywhere, beats the noisy one, and is beaten
    on the field only by the exact five-parameter family; the networks plateau."""
    s = m ** -0.4
    if arm == "energy_h16":
        return 0.5 * s
    if arm == "pair_clean":
        return 0.2 * s
    if arm == "e56_energy":
        return 1.2 * s
    if arm in ("mlp", "e03_mlp"):
        return 6.0 * max(s, 0.1)
    if arm in ("window", "e03_window"):
        return 3.0 * max(s, 0.1)
    if arm.startswith("e03_"):
        return 2.3 * max(s, 0.1)
    return 1.0


def table(level=world, drop=()):
    rows = []
    for cell in r.CELLS:
        seeds = r.seeds_of(cell)
        arms = ("energy_h16", "mlp", "window") + (("e56_energy",) + tuple(f"e03_{a}" for a in E03_ARCHS)
                                                  if cell == "laplace" else ("pair_clean",))
        for arm in arms:
            for m in r.SIZES:
                if (arm == "e56_energy" or arm.startswith("e03_")) and m > 2048:
                    continue
                for s in seeds:
                    risk = level(cell, arm, m) * (1 + 0.01 * ((s % 3) - 1))
                    if (cell, arm, m, s) in drop:
                        risk = np.nan
                    rows.append(dict(cell=cell, arm=arm, n_chains=m, seed=s, risk=risk))
    return pd.DataFrame(rows)


def test_all_true_in_the_registered_world():
    v = r.verdicts(table())
    assert all(ok for ok, _ in v.values()), {k: d for k, (ok, d) in v.items() if not ok}


def _flip(cell, arm, factor, where=lambda m: True):
    return lambda c, a, m: world(c, a, m) * (factor if (c, a) == (cell, arm) and where(m) else 1.0)


@pytest.mark.parametrize("key, level", [
    ("P1", _flip("laplace", "mlp", 0.01, lambda m: m == 32768)),              # the MLP catches up with big data
    ("P1", _flip("laplace", "window", 0.05, lambda m: m == 128)),
    ("P2", _flip("laplace", "e56_energy", 0.2, lambda m: m == 512)),          # noisy EM beats the clean fit
    ("P3", _flip("laplace", "e03_bigru", 0.05, lambda m: m == 2048)),         # the best network beats the kernel
    ("P4", _flip("T1_iso_L32", "window", 0.01, lambda m: m == 8192)),
    ("P5", _flip("T0.3_h1_L32", "pair_clean", 20.0, lambda m: m == 128)),      # the general kernel beats the right family
    ("P6", _flip("T0.3_h1_L32", "energy_h16", 50.0, lambda m: m == 128)),      # not data-efficient on the cold cell
])
def test_each_prediction_can_fail(key, level):
    v = r.verdicts(table(level))
    assert not v[key][0], v[key][1]


def test_a_failed_fit_counts_against_every_prediction_it_enters():
    drop = {("laplace", "energy_h16", 128, s) for s in range(3)}
    v = r.verdicts(table(drop=drop))
    assert not v["P1"][0] and not v["P2"][0] and not v["P6"][0]
    v = r.verdicts(table(drop={("T1_iso_L32", "pair_clean", 32, 0)}))
    assert not v["P5"][0]


def test_paired_wins_need_both_rows():
    t = table(drop={("laplace", "energy_h16", 32, 0)})
    assert r.wins(t, "laplace", "energy_h16", "mlp", 32) == 15
    assert r.wins(t, "laplace", "mlp", "energy_h16", 32) == 0
