"""read_e68's verdicts on made-up tables, in both directions, and its gates."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "core"))
import read_e68 as r

FLOOR = {"stationary": 0.0, "drift05": 0.03, "drift10": 0.063, "switch025": 0.094, "switch05": 0.174}


def _value(cell, arm, n):
    """A world where every prediction holds."""
    floor = FLOOR[cell]
    est = 0.4 / np.sqrt(n)                                   # estimation error, shrinking with N
    if arm in ("shared_mixture", "shared_energy"):
        return np.hypot(est, floor) * (1.0 if arm == "shared_mixture" else 1.02)
    if arm == "persite_mixture":
        return 6.0 * est                                     # pays (L - 1) in data, no floor
    if arm == "site_energy":
        return 0.9 * min(np.hypot(est, floor), 6.0 * est, 0.11) * (1.0 if cell != "stationary" else 1.02 / 0.9)
    if arm == "window":
        return 0.12 + floor
    if arm == "window_pos":
        return 0.12 + 0.2 * floor
    if arm == "mlp":
        return 0.25 + floor
    return 0.16


def world(edit=None):
    rows = []
    for c in r.CELLS:
        for a in ("shared_mixture", "persite_mixture", "shared_energy", "site_energy", "mlp", "window", "window_pos", "lmmse"):
            for n in r.SIZES:
                for s in r.SEEDS:
                    rows.append({"cell": c, "arm": a, "n_chains": n, "seed": s,
                                 "risk": _value(c, a, n) * (1 + 0.001 * s)})
    df = pd.DataFrame(rows)
    if edit:
        edit(df)
    return df


def _ok(df):
    return {k: ok for k, (ok, _) in r.verdicts(df, FLOOR).items()}


def test_all_confirmed_in_the_designed_world():
    assert all(_ok(world()).values()), r.verdicts(world(), FLOOR)


def test_p1_falsified_when_a_shared_arm_beats_its_floor():
    def edit(df):
        df.loc[(df.cell == "drift10") & (df.arm == "shared_energy") & (df.n_chains == 2048), "risk"] = 0.01
    v = _ok(world(edit))
    assert not v["P1"]


def test_p2_falsified_when_persite_never_overtakes():
    def edit(df):
        sel = (df.cell == "switch05") & (df.arm == "persite_mixture") & (df.n_chains == 2048)
        df.loc[sel, "risk"] = 1.0
    assert not _ok(world(edit))["P2"]


def test_p3_falsified_when_a_network_wins_and_by_a_failed_fit():
    def net_wins(df):
        df.loc[(df.cell == "drift05") & (df.arm == "window_pos") & (df.n_chains == 32), "risk"] = 1e-3
    assert not _ok(world(net_wins))["P3"]

    def failed(df):
        df.loc[(df.cell == "drift10") & (df.arm == "site_energy") & (df.n_chains == 512) & (df.seed == 3), "risk"] = np.nan
    assert not _ok(world(failed))["P3"]                       # a failed fit counts against


def test_p4_and_p5_falsified():
    def window_flat(df):
        df.loc[(df.arm == "window_pos"), "risk"] = df.loc[(df.arm == "window_pos"), "risk"] + 1.0
    assert not _ok(world(window_flat))["P4"]

    def site_costs(df):
        df.loc[(df.cell == "stationary") & (df.arm == "site_energy") & (df.n_chains == 2048), "risk"] *= 2
    assert not _ok(world(site_costs))["P5"]


def test_load_gates(tmp_path):
    base = dict(cell="drift05", seed=0, n_chains=32, t=0.05, failed=0, risk_pooled=0.1, test_bundle="h")
    d = tmp_path / "outputs" / "hpc_E68_drift05"
    d.mkdir(parents=True)
    pd.DataFrame([{**base, "arm": "mlp"}, {**base, "arm": "window"}]).to_csv(d / "arms.csv", index=False)
    pd.DataFrame([{"kind": "test", "cell": "drift05", "seed": 0, "ref_check_rel": 5e-5}]).to_csv(d / "bundles.csv", index=False)
    assert len(r.load(tmp_path)) == 2
    pd.DataFrame([{**base, "arm": "mlp"}, {**base, "arm": "window", "test_bundle": "other"}]).to_csv(d / "arms.csv", index=False)
    with pytest.raises(AssertionError):
        r.load(tmp_path)
    pd.DataFrame([{**base, "arm": "mlp"}]).to_csv(d / "arms.csv", index=False)
    pd.DataFrame([{"kind": "test", "cell": "drift05", "seed": 0, "ref_check_rel": 3e-4}]).to_csv(d / "bundles.csv", index=False)
    with pytest.raises(AssertionError):
        r.load(tmp_path)
