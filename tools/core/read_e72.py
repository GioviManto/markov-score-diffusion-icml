"""Read E72 against its registered predictions (experiments/registry/E72.json).

    python tools/core/read_e72.py [--dir outputs/hpc_E72]     # gates, P1-P4, D1; exit 1 if a gate fails
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS, SIZES = tuple(range(8)), (128, 2048)
SMALL = (0.005, 0.0075, 0.01, 0.015, 0.02)
LEVELS = (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05)
NETS = ("mlp_small", "window_small")
LEARNED = ("em_energy", "energy_h16", "mixture_clean") + NETS


def load(d: Path) -> pd.DataFrame:
    df = pd.read_csv(d / "small_t_field.csv")
    df = df[df.failed == 0].copy()
    df["risk"] = np.sqrt(df.sq_err / df.sq_ref)
    return df


def series(df, arm, n, t):
    return df[(df.arm == arm) & (df.n_chains == n) & np.isclose(df.t, t)].set_index("seed").risk.reindex(SEEDS)


def seed_mean(df, arm, n, t):
    s = series(df, arm, n, t)
    return float(s.mean()) if s.notna().all() else float("nan")


def wins(df, a, b, n, t):
    x, y = series(df, a, n, t), series(df, b, n, t)
    return int(((x < y) & x.notna() & y.notna()).sum())


def gates(d, df) -> dict:
    g1 = json.loads((d / "gate.json").read_text())
    reb = df[df.arm.isin(("em_energy", "energy_h16", "mixture_clean"))].drop_duplicates(["arm", "seed", "n_chains"])
    worst = float(((reb.rebuilt_risk - reb.frozen_risk).abs() / reb.frozen_risk).max()) if len(reb) else float("inf")
    return {"G1": bool(g1["pass"]), "G1_rel_diff": g1["rel_diff"], "G2": worst <= 1e-6, "G2_worst": worst}


def verdicts(df) -> dict:
    p1 = all(wins(df, "energy_h16", net, n, t) >= 7 for net in NETS for n in SIZES for t in SMALL)
    p2 = all(wins(df, "em_energy", net, 2048, t) >= 6 for net in NETS for t in SMALL)
    p3 = all(abs(seed_mean(df, "mixture_clean", n, t) / seed_mean(df, "lmmse", n, t) - 1) <= 0.05
             for n in SIZES for t in LEVELS)
    p4 = all(seed_mean(df, a, n, 0.005) > seed_mean(df, a, n, 0.05) for a in LEARNED for n in SIZES)
    return {"P1": bool(p1), "P2": bool(p2), "P3": bool(p3), "P4": bool(p4)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "outputs" / "hpc_E72")
    args = ap.parse_args(argv)
    df = load(args.dir)
    g = gates(args.dir, df)
    print("gates:", g)
    if not (g["G1"] and g["G2"]):
        print("a gate failed: no prediction is read")
        return 1
    print({k: ("CONFIRMED" if v else "FALSIFIED") for k, v in verdicts(df).items()})
    pd.set_option("display.width", 200)
    print(df.groupby(["n_chains", "arm", "t"]).risk.mean().unstack("t").round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
