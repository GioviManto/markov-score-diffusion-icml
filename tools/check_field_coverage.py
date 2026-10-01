"""Coverage of every field-laboratory shard against its registered design, before anything is read.

N-27: a merged or truncated CSV looks complete. Each shard is checked against the exact set of
(cell, seed, size, arm, selection) rows its job was asked to produce, read from the registered
settings in the drivers (not retyped here), and every missing key is printed. A shard is
COMPLETE, PARTIAL (with the missing count and the first missing keys) or ABSENT. Exit status is
non-zero unless every present shard is complete.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))
import e33_field_learning as e33  # noqa: E402
import e34_field_physics as e34  # noqa: E402

CELLS = list(e33.SETTINGS["cells"])
SIZES = list(e33.SETTINGS["sizes"])
NET_SIZES = list(e33.SETTINGS["net_sizes"])
ARCHS = list(e33.SETTINGS["archs"])


def kernel_keys(seeds, arms):
    return {(c, s, n, a, sel) for c, s, n, a, sel in itertools.product(CELLS, seeds, SIZES, arms, ("risk", "evidence"))}


SHARDS = {
    "hpc_E33o": ("oracles.csv", ["cell", "seed", "arm"],
                 {(c, s, a) for c in CELLS for s in range(16) for a in ("lmmse_true", "projected_diagnostic", "isotropic")}),
    **{f"hpc_E33k_s{k}": ("kernels.csv", ["cell", "seed", "n_chains", "arm", "selection"],
                          kernel_keys(range(k, k + 4), e33.KERNEL_ARMS)) for k in (0, 4, 8, 12)},
    **{f"hpc_E33q_s{k}": ("kernels.csv", ["cell", "seed", "n_chains", "arm", "selection"],
                          kernel_keys(range(k, k + 4), e33.PAIR_ARMS)) for k in (0, 4, 8, 12)},
    **{f"hpc_E33n_s{k}": ("nets.csv", ["cell", "seed", "n_chains", "arm", "arch"],
                          {(c, s, n, a, arch) for c in CELLS for s in range(k, k + 8) for n in NET_SIZES
                           for a in ("ism_one_view", "dsm_one_corruption") for arch in ARCHS}) for k in (0, 8)},
    "hpc_E34c": ("clean.csv", ["cell", "seed", "n_chains", "arm"],
                 {(c, s, n, a) for c in CELLS for s in range(16) for n in SIZES for a in e34.SETTINGS["arms"]}),
}


def main() -> int:
    bad = 0
    for shard, (csv, cols, expected) in SHARDS.items():
        path = ROOT / "outputs" / shard / csv
        if not path.exists():
            print(f"  {shard:14s} ABSENT")
            continue
        df = pd.read_csv(path)
        got = {tuple(r) for r in df[cols].itertuples(index=False, name=None)}
        missing = sorted(expected - got, key=str)
        extra = len(got - expected)
        if missing:
            bad = 1
            print(f"  {shard:14s} PARTIAL  {len(expected) - len(missing)}/{len(expected)} keys, {len(missing)} missing, e.g. {missing[:3]}")
        else:
            print(f"  {shard:14s} COMPLETE {len(expected)}/{len(expected)} keys" + (f" (+{extra} unexpected)" if extra else ""))
    screen = ROOT / "outputs/hpc_E33n/screening.csv"
    print(f"  {'hpc_E33n':14s} " + (f"screening.csv with {len(pd.read_csv(screen))} rows (no fixed design count; winners read per cell)" if screen.exists() else "ABSENT"))
    return bad


if __name__ == "__main__":
    sys.exit(main())
