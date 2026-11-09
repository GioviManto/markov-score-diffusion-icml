"""Read E69 against its registered predictions (experiments/registry/E69.json).

    python tools/core/read_e69.py [--dir outputs/hpc_E69]     # gates, P1-P5, D1 table; exit 1 if a gate fails

Every number is the relative score error pooled over test chains and sites at one level. A failed or missing row counts
AGAINST every prediction it enters. The verdict functions take one long table (arm, n_chains, seed, t, risk), so
tests/test_read_e69.py drives them with made-up tables in both directions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(8))
SIZES = (128, 2048)
SMALL = (0.005, 0.0075, 0.01, 0.015, 0.02)
LEARNED = ("em_bp", "spline", "energy", "clean_energy", "mlp_small", "window_small")
NETS = ("mlp_small", "window_small")


def load(d: Path) -> pd.DataFrame:
    df = pd.read_csv(d / "small_t.csv")
    df = df[df.failed == 0].copy()
    df["risk"] = np.sqrt(df.sq_err / df.sq_ref)
    return df


def series(df, arm, n, t):
    x = df[(df.arm == arm) & (df.n_chains == n) & np.isclose(df.t, t)].set_index("seed").risk
    return x.reindex(SEEDS)


def seed_mean(df, arm, n, t):
    s = series(df, arm, n, t)
    return float(s.mean()) if s.notna().all() else float("nan")


def wins(df, a, b, n, t):
    x, y = series(df, a, n, t), series(df, b, n, t)
    return int(((x < y) & x.notna() & y.notna()).sum())


def gates(d: Path, df: pd.DataFrame) -> dict:
    g1 = json.loads((d / "gate.json").read_text())
    reb = df[df.arm.isin(("em_bp", "spline", "energy", "clean_energy"))].drop_duplicates(["arm", "seed", "n_chains"])
    worst = float(((reb.rebuilt_risk - reb.frozen_risk).abs() / reb.frozen_risk).max()) if len(reb) else float("inf")
    return {"G1": bool(g1["pass"]) and g1["rel_diff"] <= 1e-4, "G1_rel_diff": g1["rel_diff"],
            "G2": worst <= 1e-6, "G2_worst": worst}


def verdicts(df: pd.DataFrame) -> dict:
    p1 = all(wins(df, k, net, n, t) >= 7 for k in ("em_bp", "clean_energy") for net in NETS for n in SIZES
             for t in SMALL)
    p2 = all(seed_mean(df, a, n, 0.005) > seed_mean(df, a, n, 0.05) for a in LEARNED for n in SIZES)
    p3 = (np.nanmean([seed_mean(df, "true_401", n, 0.005) for n in SIZES]) > 1e-3
          and np.nanmax([seed_mean(df, "true_401", n, 0.05) for n in SIZES]) < 1e-4)
    p4 = all(seed_mean(df, "lmmse", n, 0.005) > 0.3 and seed_mean(df, "lmmse", n, 0.005) > seed_mean(df, "lmmse", n, 0.05)
             for n in SIZES)

    def ratio(t):
        r = series(df, "window_small", 2048, t) / series(df, "em_bp", 2048, t)
        return float(r.mean()) if r.notna().all() else float("nan")
    p5 = ratio(0.005) > ratio(0.05)
    return {"P1": bool(p1), "P2": bool(p2), "P3": bool(p3), "P4": bool(p4), "P5": bool(p5),
            "P5_ratio_0.005": ratio(0.005), "P5_ratio_0.05": ratio(0.05)}


def table(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["n_chains", "arm", "t"]).risk.mean().unstack("t")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "outputs" / "hpc_E69")
    args = ap.parse_args(argv)
    df = load(args.dir)
    g = gates(args.dir, df)
    print("gates:", g)
    if not (g["G1"] and g["G2"]):
        print("a gate failed: no prediction is read")
        return 1
    print("verdicts:", {k: ("CONFIRMED" if v is True else "FALSIFIED" if v is False else v)
                        for k, v in verdicts(df).items()})
    pd.set_option("display.width", 200)
    print(table(df).round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
