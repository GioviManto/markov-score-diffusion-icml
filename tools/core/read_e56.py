"""Read E56 against its registered predictions (experiments/registry/E56.json).

    python tools/core/read_e56.py            # gates, tables, verdicts P1-P5, descriptive D1-D4

Arms: the four shift families (spline, rbf, fourier, legendre) and the two agnostic kernels (loglinear2d, energy) from
E56; em_bp (the C = 8 mixture) and mdn from E30, refitted on the same bundles; E03's six networks (window, mlp, bigru, tcn,
transformer, unet). Every number is the pooled relative score error over the test bundle's chains, sites and twelve
levels (E03's estimand); kernel_hellinger is E56's stationary-weighted Hellinger distance to the true kernel.

A FAILED or missing fit counts AGAINST every prediction it enters (E56.json G3, E33 amendment 3): a seed mean needs all
sixteen seeds finite, and a paired win needs BOTH rows, whichever arm the prediction favours.

The verdict functions take one long table (arm, n_chains, seed, risk, at_cap, checkpoint, kernel_hellinger), so
tests/test_read_e56.py drives them with made-up tables in both directions.
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
SHIFT = ("spline", "rbf", "fourier", "legendre")
AGNOSTIC = ("loglinear2d", "energy")
NETS = ("window", "mlp", "bigru", "tcn", "transformer", "unet")
COLS = ["arm", "n_chains", "seed", "test_bundle", "risk", "at_cap", "checkpoint", "kernel_hellinger"]
SQ_REF_RTOL = 1e-12
CAP = 2400          # the last checkpoint of E56's registered ladder (E56.json; e56_kernel_families.LADDER)


def e56_at_cap(e56: pd.DataFrame) -> pd.Series:
    """G4's "at the ladder's cap", from the selected checkpoint.

    The driver's own at_cap column is best == max(saved checkpoints), and fit_em also saves the FINAL iterate of a run
    that converged early under that iterate's index, so a converged fit whose last iterate won the validation selection
    reads at_cap = 1 there although it never reached the cap. Recomputed here from the checkpoint the fit selected.
    """
    return (pd.to_numeric(e56.checkpoint, errors="coerce") >= CAP).astype(int)


def _csvs(*patterns):
    paths = sorted(p for pattern in patterns for p in glob.glob(str(ROOT / pattern)))
    if not paths:
        raise SystemExit(f"no files match {patterns}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def load(e56_pattern="outputs/hpc_E56_s*/kernels.csv"):
    e56 = _csvs(e56_pattern)
    e30 = _csvs("outputs/hpc_E30_s*/joint.csv")
    e03 = pd.read_csv(ROOT / "outputs/E03/confirm.csv")
    # gate G1: the sq_ref of every (seed, n, t) equals E30's, so E56 IS on E03's bundles (part verify, recomputed here)
    ok = e56[e56.failed == 0]
    a = ok.groupby(["seed", "n_chains", "t"]).sq_ref.first()
    b = e30.groupby(["seed", "n_chains", "t"]).sq_ref.first()
    common = a.index.intersection(b.index)
    assert len(common) == len(a), f"{len(a) - len(common)} E56 cells have no E30 partner"
    worst = float(((a[common] - b[common]).abs() / b[common].abs()).max())
    assert worst <= SQ_REF_RTOL, f"E56 is not on E03's bundles: sq_ref differs by {worst:.2e} relative"
    # every row is then keyed by (seed, n_chains, test_bundle); a network row must pair with it
    k56 = e56.drop_duplicates(["seed", "n_chains", "arm"]).assign(risk=lambda d: d.risk_pooled,
                                                                   at_cap=lambda d: e56_at_cap(d))
    k30 = e30.drop_duplicates(["seed", "n_chains", "arm"]).assign(risk=lambda d: d.risk_pooled, kernel_hellinger=np.nan)
    n03 = e03[e03.region == "all"].assign(arm=lambda d: d.arch, kernel_hellinger=np.nan)
    df = pd.concat([d.reindex(columns=COLS) for d in (k56, k30, n03)], ignore_index=True)
    df = df[df.seed.isin(SEEDS) & df.n_chains.isin(SIZES)]
    # The bundle HASH covers the reference means' bytes, which differ in the last bits between machines (E54: laptop and
    # cluster hashes differ on identical draws), so it pairs the frozen comparators with each other only; E56 is paired
    # to them by G1 above, the sq_ref of every cell.
    frozen = df[~df.arm.isin(SHIFT + AGNOSTIC)]
    bundles = frozen.groupby(["seed", "n_chains"]).test_bundle.nunique()
    assert (bundles == 1).all(), f"E30 and E03 disagree on the test bundle: {bundles[bundles > 1]}"
    dup = df.duplicated(["arm", "n_chains", "seed"])
    assert not dup.any(), f"duplicate rows: {df[dup].head()}"
    return df, e56


# ------------------------------------------------------------------ the verdict arithmetic


def _series(df, arm, m, col="risk"):
    return df[(df.arm == arm) & (df.n_chains == m)].set_index("seed")[col].reindex(SEEDS)


def seed_mean(df, arm, m, col="risk"):
    """Seed mean; NaN unless all sixteen seeds have a finite row."""
    x = _series(df, arm, m, col)
    return float(x.mean()) if np.isfinite(x.to_numpy(dtype=float)).all() else np.nan


def wins(df, a, b, m, col="risk"):
    """Seeds in which arm a is below arm b, BOTH rows finite: a missing row of either side is not a win."""
    xa, xb = _series(df, a, m, col).to_numpy(dtype=float), _series(df, b, m, col).to_numpy(dtype=float)
    return int((np.isfinite(xa) & np.isfinite(xb) & (xa < xb)).sum())


def _lt(a, b):
    return bool(np.isfinite(a) and np.isfinite(b) and a < b)


def best_net(df, m):
    means = {a: seed_mean(df, a, m) for a in NETS}
    return min(means, key=lambda a: means[a] if np.isfinite(means[a]) else np.inf)


def cap_flags(df):
    """(arm, n, fits at the cap) wherever more than 8 of 16 E56 fits stopped at the ladder's last checkpoint (G4)."""
    out = []
    for arm in SHIFT + AGNOSTIC:
        for m in SIZES:
            c = int(_series(df, arm, m, "at_cap").fillna(0).astype(float).sum())
            if c > 8:
                out.append((arm, m, c))
    return out


def verdicts(df):
    v = {}
    mean = lambda arm, m: seed_mean(df, arm, m)  # noqa: E731
    bad = [(a, m, best_net(df, m)) for a in SHIFT for m in SIZES
           if not (_lt(mean(a, m), mean(best_net(df, m), m)) and wins(df, a, best_net(df, m), m) >= 13)]
    v["P1"] = (not bad, "every shift family below the best network at every N, >= 13/16 paired"
               + ("" if not bad else f"; FAILS at {bad}"))
    ratios = {(a, m): mean(a, m) / mean("em_bp", m) for a in SHIFT for m in (128, 512, 2048)}
    bad = [k for k, x in ratios.items() if not (np.isfinite(x) and 1 / 1.5 <= x <= 1.5)]
    v["P2"] = (not bad, "shift / em_bp seed-mean ratios " + ", ".join(f"{a}@{m}={x:.2f}" for (a, m), x in ratios.items())
               + ("" if not bad else f"; outside [0.67, 1.5] at {bad}"))
    bad = [(a, m) for a in AGNOSTIC for m in SIZES
           if not (_lt(mean("em_bp", m), mean(a, m)) and wins(df, "em_bp", a, m) >= 13)]
    v["P3"] = (not bad, "both agnostic kernels above em_bp at every N, >= 13/16 paired"
               + ("" if not bad else f"; FAILS at {bad}"))
    bad = [(a, net, m) for a in AGNOSTIC for net in ("mlp", "window") for m in (128, 512, 2048)
           if not (_lt(mean(a, m), mean(net, m)) and wins(df, a, net, m) >= 12)]
    v["P4"] = (not bad, "both agnostic kernels below the MLP and the window CNN at N >= 128, >= 12/16 paired"
               + ("" if not bad else f"; FAILS at {bad}"))
    pairs = {(lo, gl): wins(df, lo, gl, 2048, "kernel_hellinger") for lo in ("spline", "rbf") for gl in ("fourier", "legendre")}
    v["P5"] = (all(w >= 12 for w in pairs.values()), "N = 2048 kernel error, local below global: "
               + ", ".join(f"{lo}<{gl} {w}/16" for (lo, gl), w in pairs.items()) + " (each >= 12)")
    return v


def descriptive(df):
    d = {}
    med = lambda arm, m: float(_series(df, arm, m, "checkpoint").median())  # noqa: E731
    d["D1"] = "median selected checkpoint (EM iterations | network steps) at N = 32/128/512/2048: " + "; ".join(
        f"{a} " + "/".join(f"{med(a, m):.0f}" for m in SIZES) for a in SHIFT + AGNOSTIC + ("em_bp", best_net(df, 2048)))
    d["D4"] = "; ".join(f"{a} vs bigru {wins(df, a, 'bigru', m)}/16 @N={m}" for a in AGNOSTIC for m in SIZES) + \
        "; " + "; ".join(f"energy vs mdn {wins(df, 'energy', 'mdn', m)}/16 @N={m}" for m in SIZES)
    return d


def main(argv) -> int:
    df, e56 = load(*argv[:1])       # optional: another glob for E56's kernels.csv (a smoke run's output)
    n_failed = int(e56.drop_duplicates(["seed", "n_chains", "arm"]).failed.sum())
    print(f"E56: {len(df)} (arm, N, seed) rows; failed E56 fits: {n_failed}")
    tab = pd.DataFrame({m: {a: seed_mean(df, a, m) for a in SHIFT + AGNOSTIC + ("em_bp", "mdn") + NETS} for m in SIZES})
    print("\nseed-mean relative score error\n" + tab.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nbest network per N: " + ", ".join(f"{m}: {best_net(df, m)}" for m in SIZES))
    flags = cap_flags(df)
    print("G4 (> 8/16 at the cap, read as upper bounds): " + (", ".join(map(str, flags)) if flags else "none"))
    print()
    for k, (ok, detail) in verdicts(df).items():
        print(f"{k}: {'CONFIRMED' if ok else 'FALSIFIED'} -- {detail}")
    for k, text in descriptive(df).items():
        print(f"{k}: {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
