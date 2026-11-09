"""Read E74 against its registered predictions (experiments/registry/E74.json).

    python tools/core/read_e74.py      # P1-P4 and the metric table

A failed or missing row counts AGAINST every prediction it enters.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS, SIZES = tuple(range(8)), (128, 2048)
CELLS = ("langevin", "skew", "pearson", "T0.3_h1_L32")
NETS = ("mlp_cont", "window_cont", "unet_cont")


def load(d=ROOT / "outputs" / "hpc_E74") -> pd.DataFrame:
    g = pd.read_csv(d / "generation.csv")
    return g[g.failed == 0].drop_duplicates(["cell", "arm", "seed", "n_chains"])


def s(g, cell, arm, n, col):
    return g[(g.cell == cell) & (g.arm == arm) & (g.n_chains == n)].set_index("seed")[col].reindex(SEEDS)


def count(x, y):
    return int(((x < y) & x.notna() & y.notna()).sum())


def verdicts(g) -> dict:
    p1 = all(count(s(g, c, "oracle", n, "sw2"), 1.5 * s(g, c, "exact", n, "sw2")) >= 7 for c in CELLS for n in SIZES)
    p2 = all(count(s(g, c, "energy_clean", 128, "sw2"), s(g, c, net, 128, "sw2")) >= 6 for c in CELLS for net in NETS)

    def best_net(c, n):
        return min(NETS, key=lambda a: s(g, c, a, n, "switch_err").mean())
    p3 = all(count(s(g, c, "energy_clean", n, "switch_err"), s(g, c, best_net(c, n), n, "switch_err")) >= 6
             for c in ("langevin", "T0.3_h1_L32") for n in SIZES)
    p4 = all(count(s(g, "T0.3_h1_L32", "energy_clean", n, "sw2"), s(g, "T0.3_h1_L32", "mixture", n, "sw2")) >= 7
             for n in SIZES)
    return {"P1": p1, "P2": p2, "P3": p3, "P4": p4}


def main() -> int:
    g = load()
    print({k: ("CONFIRMED" if v else "FALSIFIED") for k, v in verdicts(g).items()})
    pd.set_option("display.width", 200)
    for col in ("sw2", "switch_err"):
        print(col)
        print(g.groupby(["cell", "n_chains", "arm"])[col].mean().unstack().round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
