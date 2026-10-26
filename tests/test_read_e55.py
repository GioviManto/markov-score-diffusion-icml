"""E55's reader, tested before any E55 data exist: each registered prediction flips when the made-up table is built to
satisfy it and when it is built to break it, and a missing fit counts against the prediction it touches."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "core"))
import read_e55 as r  # noqa: E402

CELLS = r.E33_CELLS + (r.L128, r.L512)
SIZES = {**{c: (32, 128, 512, 2048, 8192) for c in r.E33_CELLS}, r.BASE: (32, 128, 512, 2048, 8192, 32768),
         r.L128: (32, 128, 512, 2048, 8192), r.L512: (32, 128, 512, 2048)}


def table(level):
    """level(arm, cell, m) -> risk; every arm at every (cell, M, seed), seeds jittered by +-1%, nets not at the cap."""
    rows = []
    for arm in ("unet", "window", "em_energy", "site_energy", "mdn", "cnn_e54"):
        for c in CELLS:
            for m in SIZES[c]:
                for s in r.SEEDS:
                    rows.append(dict(arm=arm, cell=c, n_chains=m, seed=s, at_cap=0,
                                     risk=level(arm, c, m) * (1 + 0.01 * ((s % 3) - 1))))
    return pd.DataFrame(rows), pd.Series(1.0, index=list(CELLS))


def kernel_world(arm, c, m):
    """Every prediction true: kernels learn as M^-1/2 per transition, the networks slowly; the U-Net beats the window."""
    L = int(c.rsplit("_L", 1)[1])
    n = m * (L - 1)
    return {"site_energy": 2.0 * n ** -0.5, "em_energy": 10.0 * n ** -0.5 if m > 32 else 5.0, "mdn": 1.0,
            "unet": 3.0 * m ** -0.1, "window": 4.0 * m ** -0.1, "cnn_e54": 5.0}[arm]


def test_all_true_in_the_kernel_world():
    df, floor = table(kernel_world)
    v = r.verdicts(df, floor)
    assert all(ok for ok, _ in v.values()), {k: d for k, (ok, d) in v.items() if not ok}


def test_each_prediction_can_fail():
    def flip(arm_to_worsen, factor, where=lambda c, m: True):
        return lambda a, c, m: kernel_world(a, c, m) * (factor if a == arm_to_worsen and where(c, m) else 1.0)
    cases = {
        "P1": flip("em_energy", 100.0, lambda c, m: m == 2048),
        "P2": flip("unet", 1e-3, lambda c, m: m == 8192),
        "P3": flip("site_energy", 1e3, lambda c, m: c == "T1_iso_L32" and m == 512),
        "P4": flip("site_energy", 1e3, lambda c, m: m == 32),
        "P5": flip("unet", 100.0, lambda c, m: m == 32),
        "P7": flip("window", 10.0, lambda c, m: m == 2048),
        "P8": flip("unet", 10.0, lambda c, m: m == 2048),
        "P9a": flip("site_energy", 3.0, lambda c, m: c == r.BASE and m == 32768),
        "P9b": flip("em_energy", 1e3, lambda c, m: c == r.BASE and m == 32768),
        "P10a": flip("site_energy", 3.0, lambda c, m: c == r.L512),
        "P10b": flip("em_energy", 3.0, lambda c, m: c == r.L512),
    }
    for key, world in cases.items():
        df, floor = table(world)
        ok, detail = r.verdicts(df, floor)[key]
        assert not ok, f"{key} survived a table built to break it: {detail}"


def test_p6_reads_the_cap_and_a_missing_fit_counts_as_capped():
    df, floor = table(kernel_world)
    sel = (df.arm == "unet") & df.n_chains.isin([512, 2048, 8192]) & df.cell.isin(r.E33_CELLS)
    df.loc[sel & (df.seed < 2), "at_cap"] = 1                      # 2/8 = 0.25: not below 0.25
    assert not r.verdicts(df, floor)["P6"][0]
    df, floor = table(kernel_world)
    df = df[~((df.arm == "window") & (df.cell == "T2_iso_L32") & (df.seed < 8) & df.n_chains.isin([512, 2048, 8192]))]
    assert r.at_cap_share(df, "window", r.E33_CELLS, (512, 2048, 8192)) == 3 / 18


def test_a_missing_or_nan_fit_counts_against():
    df, floor = table(kernel_world)
    drop = (df.arm == "em_energy") & (df.n_chains == 2048) & (df.seed == 0) & df.cell.isin(r.E33_CELLS[:3])
    df2 = df[~drop]
    assert not r.verdicts(df2, floor)["P1"][0]                     # 3 of the 6 cells lose their seed mean
    df3 = df.copy()
    df3.loc[(df3.arm == "site_energy") & (df3.cell == r.BASE) & (df3.n_chains == 32768) & (df3.seed == 3), "risk"] = np.nan
    assert not r.verdicts(df3, floor)["P9a"][0]
    assert r.wins(df, "em_energy", "unet", r.BASE, 2048) == 8
    df4 = df[~((df.arm == "em_energy") & (df.cell == r.BASE) & (df.n_chains == 2048) & (df.seed == 5))]
    assert r.wins(df4, "em_energy", "unet", r.BASE, 2048) == 7
    df5 = df[~((df.arm == "unet") & (df.cell == r.BASE) & (df.n_chains == 2048) & (df.seed == 5))]
    assert r.wins(df5, "em_energy", "unet", r.BASE, 2048) == 8     # the loser's missing row is the winner's win


def test_triplets_have_matched_transition_counts():
    for t in r.TRIPLETS:
        n = [m * (int(c.rsplit("_L", 1)[1]) - 1) for c, m in t]
        assert max(n) / min(n) < 1.04
