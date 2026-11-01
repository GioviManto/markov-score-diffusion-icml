"""Read E59 against its registered predictions (experiments/registry/E59.json).

    python tools/core/read_e59.py            # gate, tables, verdicts P1-P5, descriptive D1-D2

Benchmarks `setar` and `wind`; scenarios `clean` and `noisy`; methods kernel, gauss_ar1, oracle (SETAR), qmlp, qgru,
persistence, climatology; horizons 1, 2, 4, 8. Every number is a mean over the test windows of one (benchmark, scenario,
seed, N, method, h): the quantile CRPS (the same rule for every method), the 5-95% coverage, the median's RMSE.

A missing row counts AGAINST every prediction it enters: a paired win needs both rows. The verdict functions take one
long table, so tests/test_read_e59.py drives them with made-up tables in both directions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(8))
HORIZONS = (1, 2, 4, 8)
SIZES = {"setar": (32, 128, 512, 2048), "wind": (32, 128, 512, 1917)}


def load(path="outputs/hpc_E59/forecasts.csv"):
    df = pd.read_csv(ROOT / path)
    # gate: every method of a (benchmark, seed) was scored on the same test targets
    h = df.groupby(["benchmark", "seed"]).target_hash.nunique()
    assert (h == 1).all(), f"methods disagree on the test targets: {h[h > 1]}"
    dup = df.duplicated(["benchmark", "scenario", "seed", "n_chains", "method", "h"])
    assert not dup.any(), "duplicate rows"
    return df


def _series(df, b, scen, method, n, h, col="crps"):
    x = df[(df.benchmark == b) & (df.scenario == scen) & (df.method == method) & (df.n_chains == n) & (df.h == h)]
    return x.set_index("seed")[col].reindex(SEEDS).to_numpy(dtype=float)


def _mean_h(df, b, scen, method, n):
    return np.mean([_series(df, b, scen, method, n, h) for h in HORIZONS], axis=0)   # NaN if any horizon is missing


def wins(xa, xb):
    return int((np.isfinite(xa) & np.isfinite(xb) & (xa < xb)).sum())


def verdicts(df):
    v = {}
    bad = [(n, h) for n in (128, 512, 2048) for h in HORIZONS
           if wins(_series(df, "setar", "clean", "kernel", n, h), _series(df, "setar", "clean", "gauss_ar1", n, h)) < 7]
    v["P1"] = (not bad, "SETAR clean: kernel CRPS below gauss_ar1 at N >= 128, every h, >= 7/8 paired"
               + ("" if not bad else f"; FAILS at {bad}"))
    bad = [(n, net) for n in (32, 128, 512) for net in ("qmlp", "qgru")
           if wins(_mean_h(df, "setar", "clean", "kernel", n), _mean_h(df, "setar", "clean", net, n)) < 7]
    v["P2"] = (not bad, "SETAR clean: kernel CRPS (mean over h) below both quantile networks at N <= 512, >= 7/8 paired"
               + ("" if not bad else f"; FAILS at {bad}"))
    bad = [m for m in ("gauss_ar1", "qmlp", "qgru")
           if wins(_series(df, "setar", "noisy", "kernel", 2048, 1), _series(df, "setar", "noisy", m, 2048, 1)) < 7]
    v["P3"] = (not bad, "SETAR noisy, N = 2048, h = 1: kernel below gauss_ar1, qmlp and qgru, >= 7/8 paired"
               + ("" if not bad else f"; FAILS against {bad}"))
    bad = [(n, h) for n in (128, 512, 1917) for h in (1, 2)
           if wins(_series(df, "wind", "clean", "kernel", n, h), _series(df, "wind", "clean", "gauss_ar1", n, h)) < 7]
    v["P4"] = (not bad, "wind clean: kernel CRPS below gauss_ar1 at h = 1, 2 and N >= 128, >= 7/8 paired"
               + ("" if not bad else f"; FAILS at {bad}"))
    cov = _series(df, "wind", "clean", "kernel", 1917, 1, "cov90")
    ok = int(((cov >= 0.85) & (cov <= 0.95)).sum())
    v["P5"] = (ok >= 6, f"wind clean, N = 1917, h = 1: kernel 5-95% coverage in [0.85, 0.95] in {ok}/8 seeds (>= 6); "
               f"values {np.round(cov, 3).tolist()}")
    return v


def descriptive(df):
    d = {}
    for b in ("setar", "wind"):
        for scen in ("clean", "noisy"):
            sub = df[(df.benchmark == b) & (df.scenario == scen)]
            tab = sub.groupby(["method", "n_chains", "h"]).crps.mean().unstack(["n_chains"]).round(4)
            d[f"D2 {b} {scen} (seed-mean CRPS)"] = "\n" + tab.to_string()
    w = df[(df.benchmark == "wind") & (df.scenario == "clean")]
    d["D1 wind vs networks"] = "\n" + w[w.method.isin(["kernel", "qmlp", "qgru", "gauss_ar1"])].groupby(
        ["method", "n_chains"]).crps.mean().unstack().round(4).to_string()
    return d


def main(argv) -> int:
    df = load(*argv[:1])
    print(f"E59: {len(df)} rows")
    for k, (ok, detail) in verdicts(df).items():
        print(f"{k}: {'CONFIRMED' if ok else 'FALSIFIED'} -- {detail}")
    for k, text in descriptive(df).items():
        print(f"{k}: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
