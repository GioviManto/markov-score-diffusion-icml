"""Read E67 against its registered predictions (experiments/registry/E67.json).

    python tools/core/read_e67.py          # gates, table, verdicts P1-P3, descriptive D1-D2

Inputs: outputs/hpc_E67_*/arms.csv (lmmse, var1, energy2d), nets.csv (window2, bigru2), gridcheck.csv (the reference
on 41 against 61 points per axis), timing.csv (seconds per EM iteration at N = 2048 with the job's threads).
A FAILED or missing fit counts AGAINST every prediction it enters. The verdict functions take plain tables, so
tests/test_read_e67.py drives them with made-up data in both directions.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(8))
NETS = ("window2", "bigru2")


def _csvs(pattern):
    paths = sorted(glob.glob(str(ROOT / pattern)))
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else pd.DataFrame()


def load():
    d = pd.concat([_csvs("outputs/hpc_E67_*/arms.csv"), _csvs("outputs/hpc_E67_*/nets.csv")], ignore_index=True)
    d = d.assign(risk=np.where(d.failed.fillna(0) == 0, d.risk_pooled, np.nan))
    return d, _csvs("outputs/hpc_E67_*/gridcheck.csv"), _csvs("outputs/hpc_E67_*/timing.csv")


def _v(d, arm, n):
    return d[(d.arm == arm) & (d.n_chains == n)].set_index("seed").risk.reindex(SEEDS).to_numpy(dtype=float)


def below(d, a, b, n):
    xa, xb = _v(d, a, n), _v(d, b, n)
    return int((np.isfinite(xa) & np.isfinite(xb) & (xa < xb)).sum())


def best_net(d, n):
    m = {a: (np.mean(_v(d, a, n)) if np.isfinite(_v(d, a, n)).all() else np.inf) for a in NETS}
    return min(m, key=m.get)


def gates(grid):
    pooled = grid[grid.t < 0].rel
    assert len(pooled) and float(pooled.iloc[0]) < 1e-2, "G1: the 41-point reference is not within 1e-2 of the 61-point one"


def verdicts(d, timing):
    p1 = {n: below(d, "energy2d", best_net(d, n), n) for n in (128, 512)}
    p2 = below(d, "energy2d", "var1", 2048)
    t = timing[(timing.arm == "energy2d")].sec_per_iter
    p3 = float(t.iloc[0]) if len(t) else np.inf
    return {
        "P1": (all(v >= 6 for v in p1.values()), "energy2d below the best network: " + ", ".join(f"N={n} {v}/8 vs {best_net(d, n)}" for n, v in p1.items())),
        "P2": (p2 >= 7, f"energy2d below var1 at N = 2048 in {p2}/8 seeds (need 7)"),
        "P3": (p3 <= 600, f"one energy2d EM iteration at N = 2048: {p3:.0f} s (need <= 600 s on 16 threads)"),
    }


def main(argv) -> int:
    d, grid, timing = load()
    gates(grid)
    print(d.groupby(["arm", "n_chains"]).risk.mean().unstack().to_string(float_format=lambda v: f"{v:.4f}"))
    print("D2  reference grid check: " + ", ".join(f"t={r.t:.2f} {r.rel:.1e}" for r in grid.itertuples()))
    print("D1  seconds per EM iteration: " + ", ".join(f"{r.arm} {r.sec_per_iter:.1f} s" for r in timing.itertuples()))
    for k, (ok, msg) in verdicts(d, timing).items():
        print(f"{k}  {'CONFIRMED' if ok else 'FALSIFIED'}: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
