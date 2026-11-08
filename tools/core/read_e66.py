"""Read E66 against its registered predictions (experiments/registry/E66.json).

    python tools/core/read_e66.py          # gates, table, verdicts P1-P3, descriptive D1-D3

Arms: E66's bp_dsm (the neural energy kernel, width 16, trained by DSM through BP on clean chains); comparators on the
same test bundles: E03's six networks (confirm.csv, all sites), E58's clean-ML energy_h16 and E56's noisy-EM energy
kernel (Laplace). Every number is the pooled relative score error over the test bundle's chains, sites and twelve levels.

A FAILED or missing fit counts AGAINST every prediction it enters. The verdict functions take one long table
(arm, n_chains, seed, risk), so tests/test_read_e66.py drives them with made-up tables in both directions.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(16))
SIZES = (32, 128, 512, 2048)
NETS = ("mlp", "window", "tcn", "bigru", "transformer", "unet")


def _csvs(pattern):
    paths = sorted(glob.glob(str(ROOT / pattern)))
    if not paths:
        raise SystemExit(f"no files match {pattern}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def load(pattern="outputs/hpc_E66_h*/kernels.csv"):
    k = _csvs(pattern)
    k = k.drop_duplicates(["seed", "n_chains", "arm"]).assign(risk=lambda d: np.where(d.failed == 0, d.risk_pooled, np.nan))
    e03 = pd.read_csv(ROOT / "outputs/E03/confirm.csv")
    e03 = e03[e03.region == "all"].rename(columns={"arch": "arm"})
    assert set(zip(k.seed, k.n_chains, k.test_bundle)) <= set(zip(e03.seed, e03.n_chains, e03.test_bundle)), \
        "G1: E66's test bundles are not E03's"
    e58 = _csvs("outputs/hpc_E58_kern_lap_s*/kernels.csv")
    e58 = e58[(e58.cell == "laplace") & (e58.arm == "energy_h16")].drop_duplicates(["seed", "n_chains"]) \
        .assign(arm="e58_clean_energy", risk=lambda d: d.risk_pooled)
    e56 = _csvs("outputs/hpc_E56_s*/kernels.csv")
    e56 = e56[(e56.arm == "energy") & (e56.failed == 0)].drop_duplicates(["seed", "n_chains"]) \
        .assign(arm="e56_noisy_energy", risk=lambda d: d.risk_pooled)
    cols = ["arm", "n_chains", "seed", "risk"]
    extra = k[["arm", "n_chains", "seed", "checkpoint", "kernel_hellinger", "sec_per_step"]]
    return pd.concat([k[cols], e03[cols], e58[cols], e56[cols]], ignore_index=True), extra, e58


def _series(df, arm, n):
    return df[(df.arm == arm) & (df.n_chains == n)].set_index("seed").risk.reindex(SEEDS)


def seed_mean(df, arm, n):
    x = _series(df, arm, n)
    return float(x.mean()) if np.isfinite(x.to_numpy(dtype=float)).all() else np.nan


def wins(df, a, b, n):
    xa, xb = _series(df, a, n).to_numpy(dtype=float), _series(df, b, n).to_numpy(dtype=float)
    return int((np.isfinite(xa) & np.isfinite(xb) & (xa < xb)).sum())


def best_net(df, n):
    m = {a: seed_mean(df, a, n) for a in NETS}
    return min(m, key=lambda a: m[a] if np.isfinite(m[a]) else np.inf)


def verdicts(df):
    p1 = {n: wins(df, "bp_dsm", best_net(df, n), n) for n in SIZES}
    p2 = {n: seed_mean(df, "bp_dsm", n) / seed_mean(df, "e58_clean_energy", n) for n in SIZES}
    p3 = {n: wins(df, "bp_dsm", "e56_noisy_energy", n) for n in SIZES}
    return {
        "P1": (all(v >= 12 for v in p1.values()),
               "bp_dsm below the best network: " + ", ".join(f"N={n} {v}/16 vs {best_net(df, n)}" for n, v in p1.items())),
        "P2": (all(np.isfinite(v) and 1 / 1.5 <= v <= 1.5 for v in p2.values()),
               "bp_dsm / E58 clean-ML energy_h16, seed means: " + ", ".join(f"N={n} {v:.2f}" for n, v in p2.items())),
        "P3": (all(v >= 12 for v in p3.values()),
               "bp_dsm below E56's noisy-EM energy kernel: " + ", ".join(f"N={n} {v}/16" for n, v in p3.items())),
    }


def main(argv) -> int:
    df, extra, e58 = load(*argv[:1])
    arms = ("bp_dsm", "e58_clean_energy", "e56_noisy_energy") + NETS
    tab = pd.DataFrame({n: {a: seed_mean(df, a, n) for a in arms} for n in SIZES})
    print("seed-mean test error\n" + tab.to_string(float_format=lambda v: f"{v:.4f}"))
    print("D1  median selected step: " + ", ".join(f"N={n} {extra[extra.n_chains == n].checkpoint.median():.0f}" for n in SIZES))
    h58 = e58.groupby("n_chains").kernel_hellinger.median()
    print("D2  median kernel Hellinger, bp_dsm vs E58: " + ", ".join(
        f"N={n} {extra[extra.n_chains == n].kernel_hellinger.median():.4f} / {h58.get(n, np.nan):.4f}" for n in SIZES))
    print(f"D3  median seconds per step: {extra.sec_per_step.median() * 1e3:.0f} ms")
    for k, (ok, msg) in verdicts(df).items():
        print(f"{k}  {'CONFIRMED' if ok else 'FALSIFIED'}: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
