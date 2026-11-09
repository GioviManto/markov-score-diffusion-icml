"""Read E73 against its registered predictions (experiments/registry/E73.json).

    python tools/core/read_e73.py       # P1-P7 and the learned regime parameters

A failed or missing row counts AGAINST every prediction it enters. Real series: paired differences only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(8))


def load(name):
    d = pd.read_csv(ROOT / f"outputs/hpc_E73_{name}" / f"{name}.csv")
    d = d[d.failed == 0]
    keys = ["arm", "seed", "n_chains"] + (["series"] if "series" in d else [])
    return d.drop_duplicates(keys).assign(risk=lambda x: x.risk_pooled)


def s(d, arm, n, series=None):
    q = d[(d.arm == arm) & (d.n_chains == n)]
    if series is not None:
        q = q[q.series == series]
    return q.set_index("seed").risk.reindex(SEEDS)


def wins(d, a, b, n, series=None):
    x, y = s(d, a, n, series), s(d, b, n, series)
    return int(((x < y) & x.notna() & y.notna()).sum())


def verdicts(reg, real) -> dict:
    p1 = all(wins(reg, "regime", k, n) >= 7 for k in ("first", "energy") for n in (512, 2048))
    p2 = all(wins(reg, "regime", net, n) >= 7 for net in ("mlp", "window") for n in (128, 512, 2048))
    p3 = wins(reg, "energy", "window", 2048) >= 6
    r = reg[(reg.arm == "regime") & (reg.n_chains == 2048)].set_index("seed").reindex(SEEDS)
    p4 = bool(r.fit_p00.notna().all() and abs(r.fit_p00.mean() - 0.95) <= 0.03 and abs(r.fit_p11.mean() - 0.95) <= 0.03
              and abs((r.fit_s1 / r.fit_s0).mean() - 3) / 3 <= 0.10)
    p5 = all(wins(real, "energy", net, n, "dtb3") >= 7 for net in ("mlp", "window") for n in (32, 128, 398))
    p6 = all(wins(real, "energy", "window", n, "milan") >= 7 for n in (32, 128))
    p7 = all(abs(s(real, "pair", n, "milan").mean() - s(real, "first", n, "milan").mean())
             <= 0.02 * s(real, "first", n, "milan").mean() for n in (32, 128, 512, 913))
    return {f"P{i}": bool(v) for i, v in enumerate((p1, p2, p3, p4, p5, p6, p7), start=1)}


def main() -> int:
    reg, real = load("regime"), load("real")
    print({k: ("CONFIRMED" if v else "FALSIFIED") for k, v in verdicts(reg, real).items()})
    pd.set_option("display.width", 200)
    print(reg.groupby(["n_chains", "arm"]).risk.mean().unstack().round(4).to_string())
    print(real.groupby(["series", "n_chains", "arm"]).risk.mean().unstack().round(4).to_string())
    r = reg[reg.arm == "regime"].groupby("n_chains")[["fit_alpha", "fit_s0", "fit_s1", "fit_p00", "fit_p11"]].mean()
    print(r.round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
