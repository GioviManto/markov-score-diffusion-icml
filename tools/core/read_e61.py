"""Read E61 against its registered predictions (experiments/registry/E61.json).

    python tools/core/read_e61.py            # gates, the D1 table, verdicts P1-P5, descriptive D2-D5

Seven chains matched to the Laplace chain in variance and lag-one correlation: gauss (the control), laplace, student,
uniform (inside the shift class), skew, langevin, pearson (outside it). Kernel arms: mixture, free4, energy (EM through
BP from one noisy view), energy_clean (clean-data maximum likelihood); networks: mlp, window; the linear floor lmmse.
Every number is the pooled relative score error over the test bundle's chains, sites and twelve levels (E03's estimand).

A FAILED or missing fit counts AGAINST every prediction it enters: a seed mean needs all eight seeds finite, a paired win
needs BOTH rows, and "the best" arm is the lowest seed mean among those that have one.

The verdict functions take one long table (chain, arm, n_chains, seed, risk), so tests/test_read_e61.py drives them with
made-up tables in both directions.
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
CHAINS = ("gauss", "laplace", "student", "uniform", "skew", "langevin", "pearson")
INSIDE = ("laplace", "student", "uniform")
OUTSIDE = ("skew", "langevin", "pearson")
NOISY = ("mixture", "free4", "energy")
KERNELS = NOISY + ("energy_clean",)
NETS = ("mlp", "window")
CAP = 2400
COLS = ["chain", "arm", "n_chains", "seed", "test_bundle", "risk", "checkpoint", "kernel_hellinger", "final_lr", "failed"]


def _csvs(*patterns):
    paths = sorted(p for pattern in patterns for p in glob.glob(str(ROOT / pattern)))
    if not paths:
        raise SystemExit(f"no files match {patterns}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def check_bundles(df: pd.DataFrame) -> None:
    """G2: every arm of a (chain, seed, n), kernels and networks, was scored on the same test bundle."""
    b = df.dropna(subset=["test_bundle"]).groupby(["chain", "seed", "n_chains"]).test_bundle.nunique()
    if (b > 1).any():
        raise SystemExit(f"arms disagree on the test bundle: {b[b > 1].head()}")


def load(kern="outputs/hpc_E61_k*/kernels.csv", nets="outputs/hpc_E61_nets/nets.csv"):
    k, n = _csvs(kern), _csvs(nets)
    for name, d in (("kernels", k), ("nets", n)):
        dup = d.duplicated(["chain", "seed", "n_chains", "arm", "t"])
        assert not dup.any(), f"duplicate {name} rows: {d[dup].head()}"
    df = pd.concat([d.drop_duplicates(["chain", "seed", "n_chains", "arm"]) for d in (k, n)], ignore_index=True)
    df = df.assign(risk=df.risk_pooled).reindex(columns=COLS)
    df.loc[df.failed.fillna(0).astype(int) == 1, "risk"] = np.nan
    df = df[df.chain.isin(CHAINS) & df.seed.isin(SEEDS) & df.n_chains.isin(SIZES)]
    check_bundles(df)
    return df


# ------------------------------------------------------------------ the verdict arithmetic


def _series(df, chain, arm, m, col="risk"):
    x = df[(df.chain == chain) & (df.arm == arm) & (df.n_chains == m)].set_index("seed")[col].reindex(SEEDS)
    return x.to_numpy(dtype=float)


def seed_mean(df, chain, arm, m, col="risk"):
    x = _series(df, chain, arm, m, col)
    return float(x.mean()) if np.isfinite(x).all() else np.nan


def wins(df, chain, a, b, m):
    xa, xb = _series(df, chain, a, m), _series(df, chain, b, m)
    return int((np.isfinite(xa) & np.isfinite(xb) & (xa < xb)).sum())


def best_of(df, chain, arms, m):
    means = {a: seed_mean(df, chain, a, m) for a in arms}
    means = {a: v for a, v in means.items() if np.isfinite(v)}
    return min(means, key=means.get) if means else None


def p1(df):
    bad = [(a, net, m, wins(df, "gauss", a, net, m)) for a in KERNELS for net in NETS for m in SIZES
           if wins(df, "gauss", a, net, m) < 7]
    return not bad, "gauss: every kernel arm below both networks at every N, >= 7/8" + (f"; FAILS {bad}" if bad else "")


def p2(df):
    bad_w = [(c, m, wins(df, c, "mixture", "window", m)) for c in INSIDE for m in SIZES if wins(df, c, "mixture", "window", m) < 7]
    ratio = {}
    for c in INSIDE:
        best = min((seed_mean(df, c, a, 2048) for a in NOISY), default=np.nan)
        ratio[c] = seed_mean(df, c, "mixture", 2048) / best if np.isfinite(best) else np.nan
    bad_r = {c: r for c, r in ratio.items() if not (np.isfinite(r) and r <= 1.5)}
    ok = not bad_w and not bad_r
    return ok, ("inside: the mixture below the window CNN at every N (>= 7/8) and within 1.5x of the best noisy-EM kernel "
                f"at 2048; ratios {({c: round(r, 3) for c, r in ratio.items()})}" + ("" if ok else f"; FAILS {bad_w} {bad_r}"))


def p3(df):
    w = {(c, a): wins(df, c, a, "mixture", 2048) for c in OUTSIDE for a in ("free4", "energy")}
    ok = all(v >= 7 for v in w.values())
    return ok, f"outside, N = 2048: free4 and energy each below the mixture in >= 7/8 on every chain; wins {w}"


def p4(df):
    per = {c: min(wins(df, c, "energy_clean", "energy", m) for m in SIZES) for c in CHAINS}
    ok = sum(v >= 6 for v in per.values()) >= 6
    return ok, f"energy_clean below energy at every N in >= 6/8 on >= 6 of 7 chains; worst wins per chain {per}"


def p5(df):
    res = {}
    for c in CHAINS:
        for m in SIZES:
            bk, bn = best_of(df, c, KERNELS, m), best_of(df, c, NETS, m)
            res[(c, m)] = (bk, bn, wins(df, c, bk, bn, m) if bk and bn else 0)
    bad = {k: v for k, v in res.items() if v[2] < 7}
    return not bad, "every chain and N: the best kernel below the best network in >= 7/8" + (f"; FAILS {bad}" if bad else "")


PREDICTIONS = {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5}


def verdicts(df):
    return {k: f(df) for k, f in PREDICTIONS.items()}


# ------------------------------------------------------------------ descriptive


def d1_table(df):
    arms = KERNELS + NETS + ("lmmse",)
    return {c: pd.DataFrame({m: {a: seed_mean(df, c, a, m) for a in arms} for m in SIZES}) for c in CHAINS}


def main() -> int:
    df = load()
    print("D1 seed-mean pooled relative score error (NaN = a seed failed or is missing)")
    for c, t in d1_table(df).items():
        print(f"  {c}\n    " + t.round(4).to_string().replace("\n", "\n    "))
    for k, (ok, text) in verdicts(df).items():
        print(f"{k}: {'CONFIRMED' if ok else 'FALSIFIED'} -- {text}")
    print("D2 kernel Hellinger to the true kernel at N = 2048 (seed mean)")
    for c in CHAINS:
        print(f"  {c}: " + ", ".join(f"{a} {seed_mean(df, c, a, 2048, 'kernel_hellinger'):.4f}" for a in KERNELS))
    try:
        e57 = _csvs("outputs/hpc_E57_kern_*/kernels.csv", "outputs/hpc_E57_nets_*/nets.csv")
        e57 = e57.drop_duplicates(["chain", "seed", "n_chains", "arm"])
        t = e57[e57.arm.isin(("mixture", "energy", "mlp", "window"))].groupby(["chain", "arm", "n_chains"]).risk_pooled.mean()
        print("D3 E57's chains, shared arms (E57's energy WITHOUT the fresh-state rule)\n  " + t.unstack().round(4).to_string().replace("\n", "\n  "))
    except SystemExit:
        print("D3 E57's rows are not fetched yet")
    k = df[df.arm.isin(KERNELS)]
    print("D4 median selected checkpoint, share at the cap, failed fits")
    for a in KERNELS:
        s = k[k.arm == a]
        print(f"  {a}: median {s.checkpoint.median()}, at cap {(s.checkpoint >= CAP).mean():.3f}, failed {int(s.failed.fillna(0).sum())}")
    stalled = k[k.arm.isin(("energy", "energy_clean")) & (k.final_lr < 1e-10)]
    print(f"D5 stalled energy fits (final lr < 1e-10): {len(stalled)} of {int(k.arm.isin(('energy', 'energy_clean')).sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
