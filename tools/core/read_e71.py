"""Read E71 against its registered predictions (experiments/registry/E71.json).

    python tools/core/read_e71.py        # gate P4, then P1-P3; exit 1 if the gate fails

A failed or missing row counts AGAINST every prediction it enters.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS, SIZES, VIEWS, ARMS = tuple(range(8)), (32, 128, 512), (1, 4, 12), ("em_bp", "energy")


def load(pattern="outputs/hpc_E71_*/views.csv") -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(ROOT / pattern)))], ignore_index=True)
    df = df[df.failed == 0].drop_duplicates(["arm", "seed", "n_chains", "views"])
    return df.assign(risk=df.risk_pooled)


def seed_mean(df, arm, n, k):
    x = df[(df.arm == arm) & (df.n_chains == n) & (df.views == k)].set_index("seed").risk.reindex(SEEDS)
    return float(x.mean()) if x.notna().all() else float("nan")


def gate(df) -> tuple[bool, float]:
    e30 = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(ROOT / "outputs/hpc_E30_s*/joint.csv")))])
    e30 = e30[e30.arm == "em_bp"].drop_duplicates(["seed", "n_chains"]).set_index(["seed", "n_chains"]).risk_pooled
    mine = df[(df.arm == "em_bp") & (df.views == 1)].set_index(["seed", "n_chains"]).risk
    common = mine.index.intersection(e30.index)
    worst = float(((mine[common] - e30[common]).abs() / e30[common]).max()) if len(common) else float("inf")
    return worst <= 1e-6 and len(common) == len(mine), worst


def verdicts(df, clean=None) -> dict:
    m = {(a, n, k): seed_mean(df, a, n, k) for a in ARMS for n in SIZES for k in VIEWS}
    p1 = all(m[(a, n, 1)] > m[(a, n, 4)] > m[(a, n, 12)] for a in ARMS for n in SIZES)
    p2 = all((m[(a, n, 1)] - m[(a, n, 4)]) > (m[(a, n, 4)] - m[(a, n, 12)]) for a in ARMS for n in SIZES)
    p3 = bool(clean is not None and m[("energy", 512, 12)] <= 1.5 * clean)
    return {"P1": bool(p1), "P2": bool(p2), "P3": p3, "means": m}


def main() -> int:
    df = load()
    ok, worst = gate(df)
    print(f"gate P4 (K = 1 reproduces E30's em_bp): worst rel diff {worst:.2e} -> {'pass' if ok else 'FAIL'}")
    sys.path.insert(0, str(ROOT / "tools" / "core"))
    import read_e58
    d58 = read_e58.load()
    clean = float(d58[(d58.cell == "laplace") & (d58.arm == "energy_h16") & (d58.n_chains == 512)
                      & d58.seed.isin(SEEDS)].risk.mean())
    v = verdicts(df, clean)
    print({k: ("CONFIRMED" if x else "FALSIFIED") for k, x in v.items() if k != "means"})
    for k, x in sorted(v["means"].items()):
        print(f"  {k}: {x:.4f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
