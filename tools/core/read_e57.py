"""Read E57 against its registered predictions (experiments/registry/E57.json).

    python tools/core/read_e57.py            # gates, tables, verdicts P1-P4, descriptive D2-D4

Chains: bimodal (inside the shift class), tanh and hetero (outside it). Arms: the four shift families and the C = 8
mixture, the two agnostic kernels (loglinear2d, energy), the four networks (mlp, window, bigru, unet) and the linear floor
(lmmse). Every number is the pooled relative score error over the test bundle's chains, sites and twelve levels (E03's
estimand); kernel_hellinger is the stationary-weighted Hellinger distance to the chain's true kernel.

A FAILED or missing fit counts AGAINST every prediction it enters (E57.json G3): a seed mean needs all eight seeds
finite, a paired win needs BOTH rows, and "the better agnostic arm" is undefined (so the prediction fails) unless both
agnostic arms have a seed mean.

The verdict functions take one long table (chain, arm, n_chains, seed, risk, checkpoint, at_cap, kernel_hellinger), so
tests/test_read_e57.py drives them with made-up tables in both directions.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(8))
SIZES = (32, 128, 512, 2048)
CHAINS = ("bimodal", "tanh", "hetero")
OUTSIDE = ("tanh", "hetero")
SHIFT = ("spline", "rbf", "fourier", "legendre")
AGNOSTIC = ("loglinear2d", "energy")
NETS = ("mlp", "window", "bigru", "unet")
KERNELS = SHIFT + ("mixture",) + AGNOSTIC
COLS = ["chain", "arm", "n_chains", "seed", "test_bundle", "risk", "checkpoint", "at_cap", "kernel_hellinger"]
CAP = 2400          # the last checkpoint of the registered ladder (E56/E57)


def at_cap(rows: pd.DataFrame) -> pd.Series:
    """G4 from the selected checkpoint: the driver's best == max(saved) also flags fits that converged early."""
    return (pd.to_numeric(rows.checkpoint, errors="coerce") >= CAP).astype(int)


def _csvs(*patterns):
    paths = sorted(p for pattern in patterns for p in glob.glob(str(ROOT / pattern)))
    if not paths:
        raise SystemExit(f"no files match {patterns}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def load(kern="outputs/hpc_E57_kern_*/kernels.csv", nets="outputs/hpc_E57_nets_*/nets.csv"):
    k = _csvs(kern)
    n = _csvs(nets)
    k = k.drop_duplicates(["chain", "seed", "n_chains", "arm"]).assign(risk=lambda d: d.risk_pooled)
    k.loc[k.arm.isin(KERNELS), "at_cap"] = at_cap(k[k.arm.isin(KERNELS)])
    n = n.drop_duplicates(["chain", "seed", "n_chains", "arm"]).assign(risk=lambda d: d.risk_pooled, kernel_hellinger=np.nan)
    df = pd.concat([d.reindex(columns=COLS) for d in (k, n)], ignore_index=True)
    df = df[df.seed.isin(SEEDS) & df.n_chains.isin(SIZES)]
    # G1: every arm of a (chain, seed, n) was scored on the same test bundle (one deployment for kernels and networks)
    bundles = df.dropna(subset=["test_bundle"]).groupby(["chain", "seed", "n_chains"]).test_bundle.nunique()
    assert (bundles == 1).all(), f"arms disagree on the test bundle: {bundles[bundles > 1]}"
    dup = df.duplicated(["chain", "arm", "n_chains", "seed"])
    assert not dup.any(), f"duplicate rows: {df[dup].head()}"
    return df, k


# ------------------------------------------------------------------ the verdict arithmetic


def _series(df, chain, arm, m, col="risk"):
    x = df[(df.chain == chain) & (df.arm == arm) & (df.n_chains == m)].set_index("seed")[col].reindex(SEEDS)
    return x.to_numpy(dtype=float)


def seed_mean(df, chain, arm, m, col="risk"):
    x = _series(df, chain, arm, m, col)
    return float(x.mean()) if np.isfinite(x).all() else np.nan


def wins(df, chain, a, b, m, col="risk"):
    """Seeds in which arm a is below arm b, BOTH rows finite."""
    xa, xb = _series(df, chain, a, m, col), _series(df, chain, b, m, col)
    return int((np.isfinite(xa) & np.isfinite(xb) & (xa < xb)).sum())


def _lt(a, b):
    return bool(np.isfinite(a) and np.isfinite(b) and a < b)


def best_of(df, chain, arms, m, col="risk"):
    """The arm with the lowest seed mean; None unless every arm has one (a failure makes the choice undefined)."""
    means = {a: seed_mean(df, chain, a, m, col) for a in arms}
    if not all(np.isfinite(v) for v in means.values()):
        return None
    return min(means, key=means.get)


def verdicts(df):
    v = {}
    bad = []
    for m in SIZES:
        net = best_of(df, "bimodal", NETS, m)
        for a in SHIFT + ("mixture",):
            if net is None or not (_lt(seed_mean(df, "bimodal", a, m), seed_mean(df, "bimodal", net, m))
                                   and wins(df, "bimodal", a, net, m) >= 7):
                bad.append((a, m, net))
    v["P1"] = (not bad, "bimodal: every shift family and the mixture below the best network at every N, >= 7/8 paired"
               + ("" if not bad else f"; FAILS at {bad}"))
    bad = []
    for c in OUTSIDE:
        agn = best_of(df, c, AGNOSTIC, 2048)
        for a in SHIFT + ("mixture",):
            if agn is None or wins(df, c, agn, a, 2048) < 7:
                bad.append((c, agn, a))
    v["P2"] = (not bad, "tanh, hetero at N = 2048: the better agnostic kernel below every shift family and the mixture, "
               ">= 7/8 paired" + ("" if not bad else f"; FAILS at {bad}"))
    bad = []
    for c in OUTSIDE:
        for m in (512, 2048):
            agn, net = best_of(df, c, AGNOSTIC, m), best_of(df, c, NETS, m)
            if agn is None or net is None or not (_lt(seed_mean(df, c, agn, m), seed_mean(df, c, net, m))
                                                  and wins(df, c, agn, net, m) >= 6):
                bad.append((c, m, agn, net))
    v["P3"] = (not bad, "tanh, hetero at N = 512, 2048: the better agnostic kernel below the best network, >= 6/8 paired"
               + ("" if not bad else f"; FAILS at {bad}"))
    bad = []
    for c in OUTSIDE:
        agn = best_of(df, c, AGNOSTIC, 2048, "kernel_hellinger")
        for a in SHIFT + ("mixture",):
            if agn is None or wins(df, c, agn, a, 2048, "kernel_hellinger") < 7:
                bad.append((c, agn, a))
    v["P4"] = (not bad, "tanh, hetero at N = 2048: the better agnostic kernel's kernel error below every shift family's "
               "and the mixture's, >= 7/8 paired" + ("" if not bad else f"; FAILS at {bad}"))
    return v


def descriptive(df):
    d = {}
    d["D2"] = "; ".join(f"{c} N={m}: " + ", ".join(
        f"{a} {seed_mean(df, c, a, m) / seed_mean(df, c, 'lmmse', m):.2f}" for a in SHIFT + ("mixture",))
        for c in CHAINS for m in (128, 2048)) + " (shift family / linear floor)"
    d["D3"] = "; ".join(f"{c}: " + " ".join(f"N={m} {best_of(df, c, NETS, m)}" for m in SIZES) for c in CHAINS)
    return d


def cap_flags(df):
    out = []
    for c in CHAINS:
        for a in KERNELS:
            for m in SIZES:
                k = int(np.nansum(_series(df, c, a, m, "at_cap")))
                if k > 4:
                    out.append((c, a, m, k))
    return out


def main(argv) -> int:
    df, k = load(*argv[:2])
    n_failed = int(k.get("failed", pd.Series(dtype=float)).fillna(0).sum())
    print(f"E57: {len(df)} (chain, arm, N, seed) rows; failed kernel fits: {n_failed}")
    for c in CHAINS:
        tab = pd.DataFrame({m: {a: seed_mean(df, c, a, m) for a in KERNELS + NETS + ("lmmse",)} for m in SIZES})
        print(f"\n{c}: seed-mean relative score error\n" + tab.to_string(float_format=lambda x: f"{x:.4f}"))
    flags = cap_flags(df)
    print("\nG4 (> 4/8 at the cap, read as upper bounds): " + (", ".join(map(str, flags)) if flags else "none"))
    for key, (ok, detail) in verdicts(df).items():
        print(f"{key}: {'CONFIRMED' if ok else 'FALSIFIED'} -- {detail}")
    for key, text in descriptive(df).items():
        print(f"{key}: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
