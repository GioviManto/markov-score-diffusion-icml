"""Write experiments/data/e64_kernels.csv: the frozen fitted kernels E64 samples from, one row per (seed, N, arm).

    python tools/core/make_e64_kernels.py

Sources (frozen, read only): E02a's C = 8 mixture (outputs/E02a/capacity.csv, arm em_bp, all sites), E56's noisy-EM
spline and energy kernels (outputs/hpc_E56_s*/kernels.csv), E58's clean-ML energy_h16 on the Laplace chain
(outputs/hpc_E58_kern_lap_s*/kernels.csv). Seeds 0-7, N in (128, 2048). Each row carries the source file's
risk_pooled so E64 can check that it rebuilt the same kernel. Refuses to write if any (seed, N, arm) is missing.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS, SIZES = tuple(range(8)), (128, 2048)
OUT = ROOT / "experiments" / "data" / "e64_kernels.csv"


def _csvs(pattern):
    return pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(ROOT / pattern)))], ignore_index=True)


def main() -> int:
    rows = []
    e02 = pd.read_csv(ROOT / "outputs/E02a/capacity.csv")
    e02 = e02[(e02.arm == "em_bp") & (e02.components == 8) & (e02.region == "all")]
    rows.append(e02.assign(arm="em_bp", source="E02a", hp='{"components": 8}', risk_pooled=e02.risk))
    e56 = _csvs("outputs/hpc_E56_s*/kernels.csv")
    e56 = e56[e56.arm.isin(["spline", "energy"]) & (e56.failed == 0)].drop_duplicates(["seed", "n_chains", "arm"])
    rows.append(e56.assign(source="E56"))
    e58 = _csvs("outputs/hpc_E58_kern_lap_s*/kernels.csv")
    e58 = e58[(e58.cell == "laplace") & (e58.arm == "energy_h16")].drop_duplicates(["seed", "n_chains"])
    rows.append(e58.assign(arm="clean_energy", source="E58", hp='{"hidden": 16}'))
    d = pd.concat([r[["seed", "n_chains", "arm", "source", "hp", "theta", "risk_pooled"]] for r in rows], ignore_index=True)
    d = d[d.seed.isin(SEEDS) & d.n_chains.isin(SIZES)].sort_values(["arm", "n_chains", "seed"])
    want = {(s, n, a) for s in SEEDS for n in SIZES for a in ("em_bp", "spline", "energy", "clean_energy")}
    have = set(zip(d.seed, d.n_chains, d.arm))
    if want - have:
        raise SystemExit(f"missing frozen kernels: {sorted(want - have)[:6]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(d)} kernels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
