"""E57's reader, tested before any E57 data exist: each registered prediction holds on a made-up table built to satisfy
it and flips on one built to break it, and a failed fit counts AGAINST every prediction it enters."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "core"))
import read_e57 as r  # noqa: E402

ARMS = r.SHIFT + ("mixture",) + r.AGNOSTIC + r.NETS + ("lmmse",)


def world(chain, arm, m):
    """Every prediction true: inside the class every kernel is good; outside it the shift kernels sit near the linear
    floor while the agnostic ones keep improving; the networks are in between."""
    s = m ** -0.4
    inside = chain == "bimodal"
    if arm == "lmmse":
        return 1.0                                          # the linear floor does not learn
    nets = {"bigru": 2.3, "window": 3.0, "unet": 2.6, "mlp": 7.0}
    if arm in nets:
        return nets[arm] * s
    if arm in r.AGNOSTIC:
        return (1.4 if arm == "loglinear2d" else 1.6) * s
    return (1.0 if inside else 3.5) * s * (1.0 if arm == "spline" else 1.05)


def hell(chain, arm, m):
    if arm in r.AGNOSTIC:
        return 0.02 if arm == "loglinear2d" else 0.03
    return 0.02 if chain == "bimodal" else 0.08


def table(level=world, hellinger=hell, drop=()):
    rows = []
    for c in r.CHAINS:
        for arm in ARMS:
            for m in r.SIZES:
                for s in r.SEEDS:
                    risk = level(c, arm, m) * (1 + 0.01 * ((s % 3) - 1))
                    if (c, arm, m, s) in drop:
                        risk = np.nan
                    kern = arm in r.SHIFT + ("mixture",) + r.AGNOSTIC
                    rows.append(dict(chain=c, arm=arm, n_chains=m, seed=s, risk=risk, checkpoint=100, at_cap=0,
                                     kernel_hellinger=(hellinger(c, arm, m) if kern and np.isfinite(risk) else np.nan)))
    return pd.DataFrame(rows)


def test_all_true_in_the_registered_world():
    v = r.verdicts(table())
    assert all(ok for ok, _ in v.values()), {k: d for k, (ok, d) in v.items() if not ok}


def _flip(chain, arm, factor, where=lambda m: True):
    return lambda c, a, m: world(c, a, m) * (factor if (c, a) == (chain, arm) and where(m) else 1.0)


@pytest.mark.parametrize("key, level, hellinger", [
    ("P1", _flip("bimodal", "fourier", 4.0, lambda m: m == 128), hell),           # a shift family loses to a net inside
    ("P1", _flip("bimodal", "mixture", 4.0, lambda m: m == 2048), hell),
    ("P2", _flip("tanh", "legendre", 0.3, lambda m: m == 2048), hell),            # a shift family beats the agnostic
    ("P2", _flip("hetero", "mixture", 0.3, lambda m: m == 2048), hell),
    ("P3", lambda c, a, m: world(c, a, m) * (3.0 if c == "hetero" and a in r.AGNOSTIC and m == 512 else 1.0), hell),
    ("P4", world, lambda c, a, m: 0.001 if (c, a) == ("tanh", "rbf") else hell(c, a, m)),
])
def test_each_prediction_can_fail(key, level, hellinger):
    v = r.verdicts(table(level, hellinger))
    assert not v[key][0], v[key][1]


def test_a_failed_fit_counts_against_the_predictions_it_enters():
    # two failed seeds of the spline arm on the bimodal chain at N = 32: P1 needs 7/8 paired wins and a seed mean
    assert not r.verdicts(table(drop={("bimodal", "spline", 32, s) for s in range(2)}))["P1"][0]
    # two failed seeds of BOTH agnostic arms on tanh at N = 2048: P2, P3 and P4 all fall
    drop = {("tanh", a, 2048, s) for a in r.AGNOSTIC for s in range(2)}
    v = r.verdicts(table(drop=drop))
    assert not v["P2"][0] and not v["P3"][0] and not v["P4"][0]


def test_paired_wins_need_both_rows():
    t = table(drop={("tanh", "spline", 32, 0)})
    assert r.wins(t, "tanh", "spline", "mlp", 32) == 7
    assert r.wins(t, "tanh", "mlp", "spline", 32) == 0


def test_at_cap_is_read_from_the_selected_checkpoint():
    rows = pd.DataFrame({"checkpoint": [2400, 900, np.nan], "at_cap": [1, 1, 0]})
    assert r.at_cap(rows).tolist() == [1, 0, 0]
