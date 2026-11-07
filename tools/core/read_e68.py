"""Read E68 against its registered predictions (experiments/registry/E68.json).

    python tools/core/read_e68.py            # gates, verdicts P1-P5, descriptive D1, D3

Cells: stationary (E03's chain), drift05, drift10, switch025, switch05. Arms: shared_mixture, persite_mixture,
shared_energy, site_energy (kernels by EM through per-link BP), mlp, window, window_pos (E03's recipe), lmmse. Every
number is the pooled relative score error over the test bundle's chains, sites and twelve levels.

A FAILED or missing fit counts AGAINST every prediction it enters: a seed mean needs every seed, a paired comparison
needs BOTH rows. The verdict functions take one long table (cell, arm, n_chains, seed, risk), so tests/test_read_e68.py
drives them with made-up tables in both directions.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CELLS = ("stationary", "drift05", "drift10", "switch025", "switch05")
NONSTAT = CELLS[1:]
SIZES = (32, 128, 512, 2048)
SEEDS = tuple(range(8))
NETS = ("mlp", "window", "window_pos")
REF_CHECK_MAX = 1e-4


def floors():
    reg = json.loads((ROOT / "experiments/registry/E68.json").read_text())
    return {c: v["shared_floor"] for c, v in reg["calibration_2026-11-07"]["per_cell"].items()}


def load(root=ROOT):
    paths = sorted(glob.glob(str(root / "outputs/hpc_E68_*/arms.csv")))
    if not paths:
        raise SystemExit("no outputs/hpc_E68_*/arms.csv")
    raw = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    dup = raw.duplicated(["cell", "seed", "n_chains", "arm", "t"])
    assert not dup.any(), f"duplicate (cell, seed, n, arm, t) rows: {raw[dup].head()}"
    ok = raw[raw.failed.fillna(0) == 0]
    n_bundles = ok.groupby(["cell", "seed", "n_chains"]).test_bundle.nunique()
    assert (n_bundles <= 1).all(), f"arms scored on different test bundles: {n_bundles[n_bundles > 1]}"
    checks = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(root / "outputs/hpc_E68_*/bundles.csv")))])
    checks = checks.dropna(subset=["ref_check_rel"])
    assert set(checks.cell) >= set(raw.cell), "a cell has no reference check"
    worst = float(checks.ref_check_rel.max())
    assert worst <= REF_CHECK_MAX, f"reference grid check {worst:.2e} > {REF_CHECK_MAX:.0e}"
    rows = raw.drop_duplicates(["cell", "seed", "n_chains", "arm"]).copy()
    rows["risk"] = np.where(rows.failed.fillna(0) == 0, rows.risk_pooled, np.nan)
    return rows[["cell", "arm", "n_chains", "seed", "risk"]]


# ------------------------------------------------------------------ the verdict arithmetic

def _series(df, cell, arm, m):
    x = df[(df.cell == cell) & (df.arm == arm) & (df.n_chains == m)].set_index("seed")["risk"].reindex(SEEDS)
    return x.to_numpy(dtype=float)


def seed_mean(df, cell, arm, m):
    x = _series(df, cell, arm, m)
    return float(x.mean()) if np.isfinite(x).all() else np.nan


def below(df, cell, a, b, m):
    """Seeds in which arm a is below arm b (both rows present and finite)."""
    xa, xb = _series(df, cell, a, m), _series(df, cell, b, m)
    return int((np.isfinite(xa) & np.isfinite(xb) & (xa < xb)).sum())


def _lt(a, b):
    return bool(np.isfinite(a) and np.isfinite(b) and a < b)


def verdicts(df, floor=None):
    floor = floor or floors()
    v = {}
    bad = []
    for c in ("drift10", "switch025", "switch05"):
        for arm in ("shared_mixture", "shared_energy"):
            m2048, m512 = seed_mean(df, c, arm, 2048), seed_mean(df, c, arm, 512)
            if not (np.isfinite(m2048) and np.isfinite(m512) and m2048 >= 0.8 * floor[c] and m2048 >= 0.9 * m512):
                bad.append((c, arm))
    v["P1"] = (not bad, "shared arms at N = 2048 >= 0.8 x the shared floor and not > 10% below N = 512 (drift10, "
               "switch025, switch05)" + ("" if not bad else f"; FAILS at {bad}"))
    bad = [c for c in ("stationary", "drift05") if below(df, c, "shared_mixture", "persite_mixture", 32) < 6]
    bad += [c for c in ("drift10", "switch025", "switch05") if below(df, c, "persite_mixture", "shared_mixture", 2048) < 7]
    v["P2"] = (not bad, "persite above shared_mixture at N = 32 (stationary, drift05, >= 6/8), below it at N = 2048 "
               "(drift10, switch025, switch05, >= 7/8)" + ("" if not bad else f"; FAILS at {bad}"))
    bad = [("a", c, m) for c in NONSTAT for m in (128, 512, 2048) if below(df, c, "site_energy", "shared_energy", m) < 6]
    bad += [("b", c, m) for c in ("drift05", "drift10") for m in SIZES if below(df, c, "site_energy", "persite_mixture", m) < 6]
    for c in CELLS:
        for m in SIZES:
            best = min((seed_mean(df, c, a, m) for a in NETS), key=lambda x: np.inf if np.isnan(x) else x)
            if any(np.isnan(seed_mean(df, c, a, m)) for a in NETS) or not _lt(seed_mean(df, c, "site_energy", m), best):
                bad.append(("c", c, m))
    v["P3"] = (not bad, "site_energy below shared_energy (N >= 128, non-stationary, >= 6/8), below persite (drift "
               "cells, every N, >= 6/8), below the best network (every cell and N)" + ("" if not bad else f"; FAILS at {bad}"))
    bad = [c for c in NONSTAT if below(df, c, "window_pos", "window", 2048) < 6]
    for c in ("drift10", "switch05"):
        if not _lt(seed_mean(df, "stationary", "window", 2048), seed_mean(df, c, "window", 2048)):
            bad.append(("window rises", c))
    v["P4"] = (not bad, "window_pos below window at N = 2048 (non-stationary, >= 6/8); window higher in drift10 and "
               "switch05 than stationary" + ("" if not bad else f"; FAILS at {bad}"))
    site, shared = seed_mean(df, "stationary", "site_energy", 2048), seed_mean(df, "stationary", "shared_energy", 2048)
    bad = [] if (np.isfinite(site) and np.isfinite(shared) and abs(site / shared - 1) <= 0.15) else ["site vs shared"]
    bad += [m for m in SIZES if below(df, "stationary", "shared_mixture", "persite_mixture", m) < 7]
    v["P5"] = (not bad, "stationary: site_energy within 15% of shared_energy at N = 2048; persite above "
               "shared_mixture at every N (>= 7/8)" + ("" if not bad else f"; FAILS at {bad}"))
    return v


def descriptive(df):
    d = {}
    arms = ("shared_mixture", "persite_mixture", "shared_energy", "site_energy", "mlp", "window", "window_pos", "lmmse")
    for c in CELLS:
        tab = pd.DataFrame({m: {a: seed_mean(df, c, a, m) for a in arms} for m in SIZES})
        med = pd.DataFrame({m: {a: float(np.median(_series(df, c, a, m))) for a in arms} for m in SIZES})
        d[f"D1 {c} (seed means)"] = "\n" + tab.to_string(float_format=lambda x: f"{x:.4f}")
        d[f"D1 {c} (medians)"] = "\n" + med.to_string(float_format=lambda x: f"{x:.4f}")
        diff = [np.log(seed_mean(df, c, "persite_mixture", m) / seed_mean(df, c, "shared_mixture", m)) for m in SIZES]
        cross = "none in range"
        for (m0, d0), (m1, d1) in zip(zip(SIZES, diff), zip(SIZES[1:], diff[1:])):
            if np.isfinite(d0) and np.isfinite(d1) and d0 > 0 >= d1:
                cross = f"N ~ {np.exp(np.log(m0) + (np.log(m1) - np.log(m0)) * d0 / (d0 - d1)):.0f}"
        d[f"D3 {c}"] = f"persite overtakes shared_mixture at {cross}"
    return d


def main(argv) -> int:
    df = load()
    print(f"E68: {len(df)} (cell, arm, N, seed) rows; failed or missing risks: {int(df.risk.isna().sum())}")
    for key, (ok, detail) in verdicts(df).items():
        print(f"{key}: {'CONFIRMED' if ok else 'FALSIFIED'} -- {detail}")
    for key, text in descriptive(df).items():
        print(f"{key}: {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
