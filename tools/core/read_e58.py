"""Read E58 against its registered predictions (experiments/registry/E58.json).

    python tools/core/read_e58.py            # gates, tables, verdicts P1-P6, descriptive D1-D3

Cells: `laplace` (E03's bundles, seeds 0-15) and the Allen-Cahn cells T0.3_h1_L32 and T1_iso_L32 (E33's bundles, seeds
0-7). Arms: E58's clean-data kernels (energy_h16/h32/h64, laplace_clean or pair_clean = the arm `ideal`, mixture_clean),
the linear floor (lmmse) and E58's networks (mlp, window); on the Laplace chain also E56's noisy-EM energy kernel
(e56_energy) and E03's six frozen networks (e03_<arch>). Every number is the pooled relative score error over the test
bundle's chains, sites and twelve levels.

A FAILED or missing fit counts AGAINST every prediction it enters: a seed mean needs every seed, a paired win needs BOTH
rows. The verdict functions take one long table (cell, arm, n_chains, seed, risk), so tests/test_read_e58.py drives them
with made-up tables in both directions.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CELLS = ("laplace", "T0.3_h1_L32", "T1_iso_L32")
FIELD = ("T0.3_h1_L32", "T1_iso_L32")
SIZES = (32, 128, 512, 2048, 8192, 32768)
E03_ARCHS = ("window", "mlp", "bigru", "tcn", "transformer", "unet")
SQ_REF_RTOL, FLOOR_RTOL = 1e-12, 1e-10


def seeds_of(cell):
    return tuple(range(16)) if cell == "laplace" else tuple(range(8))


def _csvs(*patterns):
    paths = sorted(p for pattern in patterns for p in glob.glob(str(ROOT / pattern)))
    if not paths:
        raise SystemExit(f"no files match {patterns}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def load():
    k = _csvs("outputs/hpc_E58_kern_*/kernels.csv")
    n = _csvs("outputs/hpc_E58_nets/nets.csv")
    # G1, Laplace: E58's sq_ref equals E30's on every shared (seed, N, t), so E58 is on E03's bundles
    e30 = _csvs("outputs/hpc_E30_s*/joint.csv")
    a = k[(k.cell == "laplace") & (k.failed == 0)].groupby(["seed", "n_chains", "t"]).sq_ref.first()
    b = e30.groupby(["seed", "n_chains", "t"]).sq_ref.first()
    common = a.index.intersection(b.index)
    assert len(common) > 0, "no Laplace cell in common with E30"
    worst = float(((a[common] - b[common]).abs() / b[common].abs()).max())
    assert worst <= SQ_REF_RTOL, f"E58 is not on E03's bundles: sq_ref differs by {worst:.2e}"
    # G1, field: E58's linear floor equals E33's lmmse_true on every (cell, seed), so E58 is on E33's bundles
    o = pd.read_csv(ROOT / "outputs/hpc_E33o/oracles.csv")
    o = o[o.arm == "lmmse_true"].set_index(["cell", "seed"]).risk
    f = k[(k.arm == "lmmse") & k.cell.isin(FIELD)].drop_duplicates(["cell", "seed"]).set_index(["cell", "seed"]).risk_pooled
    dev = max(abs(f[i] / o[i] - 1) for i in f.index)
    assert dev <= FLOOR_RTOL, f"E58's field bundles are not E33's: the linear floor differs by {dev:.2e}"
    rows = [k.drop_duplicates(["cell", "seed", "n_chains", "arm"]).assign(risk=lambda d: d.risk_pooled),
            n.drop_duplicates(["cell", "seed", "n_chains", "arm"]).assign(risk=lambda d: d.risk_pooled)]
    rows[0].loc[rows[0].arm == "ideal", "arm"] = np.where(rows[0].loc[rows[0].arm == "ideal", "cell"] == "laplace",
                                                          "laplace_clean", "pair_clean")
    e56 = _csvs("outputs/hpc_E56_s*/kernels.csv")
    e56 = e56[e56.arm == "energy"].drop_duplicates(["seed", "n_chains"]).assign(cell="laplace", arm="e56_energy",
                                                                               risk=lambda d: d.risk_pooled)
    e03 = pd.read_csv(ROOT / "outputs/E03/confirm.csv")
    e03 = e03[e03.region == "all"].assign(cell="laplace", arm=lambda d: "e03_" + d.arch)
    cols = ["cell", "arm", "n_chains", "seed", "risk", "checkpoint", "at_cap", "failed"]
    df = pd.concat([d.reindex(columns=cols) for d in rows + [e56, e03]], ignore_index=True)
    dup = df.duplicated(["cell", "arm", "n_chains", "seed"])
    assert not dup.any(), f"duplicate rows: {df[dup].head()}"
    return df


# ------------------------------------------------------------------ the verdict arithmetic


def _series(df, cell, arm, m, col="risk"):
    x = df[(df.cell == cell) & (df.arm == arm) & (df.n_chains == m)].set_index("seed")[col].reindex(seeds_of(cell))
    return x.to_numpy(dtype=float)


def seed_mean(df, cell, arm, m):
    x = _series(df, cell, arm, m)
    return float(x.mean()) if np.isfinite(x).all() else np.nan


def wins(df, cell, a, b, m):
    xa, xb = _series(df, cell, a, m), _series(df, cell, b, m)
    return int((np.isfinite(xa) & np.isfinite(xb) & (xa < xb)).sum())


def _lt(a, b):
    return bool(np.isfinite(a) and np.isfinite(b) and a < b)


def _beats(df, cell, a, b, m, need):
    return _lt(seed_mean(df, cell, a, m), seed_mean(df, cell, b, m)) and wins(df, cell, a, b, m) >= need


def best_e03(df, m):
    means = {f"e03_{a}": seed_mean(df, "laplace", f"e03_{a}", m) for a in E03_ARCHS}
    if not all(np.isfinite(v) for v in means.values()):
        return None
    return min(means, key=means.get)


def verdicts(df):
    v = {}
    bad = [(net, m) for net in ("mlp", "window") for m in SIZES if not _beats(df, "laplace", "energy_h16", net, m, 13)]
    v["P1"] = (not bad, "Laplace: energy_h16 below the MLP and the window CNN at every N, >= 13/16 paired"
               + ("" if not bad else f"; FAILS at {bad}"))
    bad = [m for m in (32, 128, 512, 2048) if not _beats(df, "laplace", "energy_h16", "e56_energy", m, 13)]
    v["P2"] = (not bad, "Laplace: clean energy_h16 below E56's noisy energy kernel at N <= 2048, >= 13/16 paired"
               + ("" if not bad else f"; FAILS at N = {bad}"))
    bad = []
    for m in (512, 2048):
        best = best_e03(df, m)
        if best is None or not _beats(df, "laplace", "energy_h16", best, m, 13):
            bad.append((m, best))
    v["P3"] = (not bad, "Laplace: energy_h16 below the best of E03's six networks at N = 512, 2048, >= 13/16 paired"
               + ("" if not bad else f"; FAILS at {bad}"))
    bad = [(c, net, m) for c in FIELD for net in ("mlp", "window") for m in SIZES
           if not _beats(df, c, "energy_h16", net, m, 7)]
    v["P4"] = (not bad, "Allen-Cahn: energy_h16 below the MLP and the window CNN at every N on both cells, >= 7/8 paired"
               + ("" if not bad else f"; FAILS at {bad}"))
    bad = [(c, m) for c in FIELD for m in (32, 128, 512, 2048) if not _beats(df, c, "pair_clean", "energy_h16", m, 7)]
    v["P5"] = (not bad, "Allen-Cahn: the clean site energy below energy_h16 at N <= 2048 on both cells, >= 7/8 paired"
               + ("" if not bad else f"; FAILS at {bad}"))
    bad = [c for c in ("laplace", "T0.3_h1_L32")
           if not _lt(seed_mean(df, c, "energy_h16", 128), seed_mean(df, c, "window", 32768))]
    v["P6"] = (not bad, "data efficiency: energy_h16 at N = 128 below the window CNN at N = 32768 (Laplace, cold cell)"
               + ("" if not bad else f"; FAILS on {bad}"))
    return v


def descriptive(df):
    d = {}
    for c in CELLS:
        arms = [a for a in ("energy_h16", "energy_h32", "energy_h64", "laplace_clean", "pair_clean", "mixture_clean",
                            "mlp", "window", "lmmse") if (df.cell.eq(c) & df.arm.eq(a)).any()]
        tab = pd.DataFrame({m: {a: seed_mean(df, c, a, m) for a in arms} for m in SIZES})
        d[f"D1 {c}"] = "\n" + tab.to_string(float_format=lambda x: f"{x:.4f}")
    d["D3 laplace"] = ", ".join(f"N={m}: noisy/clean {seed_mean(df, 'laplace', 'e56_energy', m) / seed_mean(df, 'laplace', 'energy_h16', m):.2f}"
                                for m in (32, 128, 512, 2048))
    d["G2 reproduction"] = ", ".join(f"{a} N={m}: E58 {seed_mean(df, 'laplace', a, m):.4f} vs E03 "
                                     f"{seed_mean(df, 'laplace', 'e03_' + a, m):.4f}"
                                     for a in ("mlp", "window") for m in (32, 128, 512, 2048))
    return d


def main(argv) -> int:
    df = load()
    print(f"E58: {len(df)} (cell, arm, N, seed) rows; failed fits: {int(df.failed.fillna(0).sum())}")
    for key, (ok, detail) in verdicts(df).items():
        print(f"{key}: {'CONFIRMED' if ok else 'FALSIFIED'} -- {detail}")
    for key, text in descriptive(df).items():
        print(f"{key}: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
