"""E56's reader, tested before any E56 data exist: each registered prediction holds on a made-up table built to satisfy
it and flips on one built to break it, and a failed fit counts AGAINST every prediction it enters (E33 amendment 3)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "core"))
import read_e56 as r  # noqa: E402

ARMS = r.SHIFT + r.AGNOSTIC + ("em_bp", "mdn") + r.NETS


def world(arm, m):
    """Every prediction true: shift families near the mixture, agnostic ones between it and the CNN/MLP, the local bases
    closest to the true kernel."""
    s = m ** -0.4
    return {"spline": 1.00 * s, "rbf": 1.05 * s, "fourier": 1.10 * s, "legendre": 1.15 * s, "em_bp": 1.0 * s,
            "loglinear2d": 2.0 * s, "energy": 2.2 * s, "mdn": 2.5 * s, "bigru": 2.3 * s, "tcn": 2.6 * s,
            "transformer": 2.7 * s, "unet": 2.8 * s, "window": 3.5 * s, "mlp": 8.0 * s}[arm]


def hell(arm, m):
    return {"spline": 0.02, "rbf": 0.025, "fourier": 0.05, "legendre": 0.06}.get(arm, 0.1)


def table(level=world, hellinger=hell, drop=()):
    rows = []
    for arm in ARMS:
        for m in r.SIZES:
            for s in r.SEEDS:
                risk = level(arm, m) * (1 + 0.01 * ((s % 3) - 1))
                if (arm, m, s) in drop:
                    risk = np.nan
                rows.append(dict(arm=arm, n_chains=m, seed=s, risk=risk, at_cap=0, checkpoint=100,
                                 kernel_hellinger=hellinger(arm, m) if arm in r.SHIFT + r.AGNOSTIC else np.nan))
    return pd.DataFrame(rows)


def test_all_true_in_the_registered_world():
    v = r.verdicts(table())
    assert all(ok for ok, _ in v.values()), {k: d for k, (ok, d) in v.items() if not ok}


def _flip(arm, factor, where=lambda m: True, base=world):
    return lambda a, m: base(a, m) * (factor if a == arm and where(m) else 1.0)


@pytest.mark.parametrize("key, level, hellinger", [
    ("P1", _flip("legendre", 3.0, lambda m: m == 32), hell),                      # one shift family loses to the best net
    ("P2", _flip("fourier", 1.6, lambda m: m == 512), hell),                      # outside the factor 1.5
    ("P2", _flip("rbf", 1 / 1.6, lambda m: m == 2048), hell),                     # ... in either direction
    ("P3", _flip("energy", 0.4, lambda m: m == 128), hell),                       # an agnostic kernel beats the mixture
    ("P4", _flip("loglinear2d", 2.0, lambda m: m == 2048), hell),                 # it loses to the window CNN
    ("P5", world, lambda a, m: {"fourier": 0.01}.get(a, hell(a, m))),             # a global basis fits the kink best
])
def test_each_prediction_can_fail(key, level, hellinger):
    v = r.verdicts(table(level, hellinger))
    assert not v[key][0], v[key][1]
    others = [k for k, (ok, _) in v.items() if not ok and k != key]
    assert key not in others


def test_a_failed_fit_counts_against_every_prediction_it_enters():
    # five failed seeds of the spline arm at N = 2048: P1, P2 (seed mean undefined) and P5 (11/16 < 12) all fall
    drop = {("spline", 2048, s) for s in range(5)}
    t = table(drop=drop)
    t.loc[t.risk.isna(), "kernel_hellinger"] = np.nan
    v = r.verdicts(t)
    assert not v["P1"][0] and not v["P2"][0] and not v["P5"][0]
    # a failed AGNOSTIC fit counts against P3 too, although P3 predicts that arm to be the worse one
    v = r.verdicts(table(drop={("energy", 128, s) for s in range(4)}))
    assert not v["P3"][0]


def test_paired_wins_need_both_rows():
    t = table(drop={("spline", 32, 0)})
    assert r.wins(t, "spline", "window", 32) == 15
    assert r.wins(t, "window", "spline", 32) == 0


def test_at_cap_is_read_from_the_selected_checkpoint_not_the_drivers_flag():
    # the driver flags best == max(saved); after an early convergence the final iterate (here 900) is saved too
    rows = pd.DataFrame({"checkpoint": [2400, 900, 1600, np.nan], "at_cap": [1, 1, 0, 0]})
    assert r.e56_at_cap(rows).tolist() == [1, 0, 0, 0]


def test_the_cap_flag():
    t = table()
    t.loc[(t.arm == "energy") & (t.n_chains == 2048) & (t.seed < 9), "at_cap"] = 1
    assert r.cap_flags(t) == [("energy", 2048, 9)]
