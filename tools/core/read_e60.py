"""Read E60 against its registered predictions (experiments/registry/E60.json).

    python tools/core/read_e60.py            # gates, the D1 table, verdicts P1-P5, descriptive D2-D5

Arms: em_free_C4 and em_free_C8 (E60's rows), against E33's frozen em_mixture, em_mdn (hpc_E33k_*), em_pair_quartic
(hpc_E33q_*) and the four networks (window and transformer x dsm_one_corruption and ism_one_view, hpc_E33n*), on the four
slide-15 cells. Every risk is E33's pooled relative score error with RISK selection, divided by the same cell and seed's
lmmse_true (hpc_E33o): 1 is the linear floor.

A FAILED or missing row counts AGAINST every prediction it enters: a seed mean needs all sixteen seeds finite, a paired win
needs BOTH rows, and "the best network" is the lowest seed mean among the networks that have one.

The verdict functions take one long table (cell, arm, n_chains, seed, risk), so tests/test_read_e60.py drives them with
made-up tables in both directions.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(16))
SIZES = (128, 512, 2048)
CELLS = ("T0.3_h1_L32", "T0.5_iso_L32", "T1_iso_L32", "T2_iso_L32")
FREE = ("em_free_C4", "em_free_C8")
E33_KERNELS = ("em_mixture", "em_mdn", "em_pair_quartic")
NETS = tuple(f"{a}_{o}" for a in ("window", "transformer") for o in ("dsm_one_corruption", "ism_one_view"))
CAP = 1200          # the last checkpoint of E33's ladder
KEY = ["cell", "seed", "n_chains"]


def _csvs(*patterns):
    paths = sorted(p for pattern in patterns for p in glob.glob(str(ROOT / pattern)))
    if not paths:
        raise SystemExit(f"no files match {patterns}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def check_bundles(new: pd.DataFrame, ref: pd.DataFrame) -> None:
    """G2: every E60 (cell, seed, M) was scored on E33's validation and test bundles."""
    r = ref.drop_duplicates(KEY).set_index(KEY)[["val_bundle", "test_bundle"]]
    n = new.drop_duplicates(KEY).set_index(KEY)[["val_bundle", "test_bundle"]]
    missing = n.index.difference(r.index)
    if len(missing):
        raise SystemExit(f"E60 cells with no E33 counterpart: {list(missing)[:3]}")
    bad = (n != r.loc[n.index]).any(axis=1)
    if bad.any():
        raise SystemExit(f"E60 and E33 bundles differ at {list(n.index[bad])[:3]}")


def load(e60="outputs/hpc_E60_*/kernels.csv"):
    new = _csvs(e60)
    k33 = _csvs("outputs/hpc_E33k_s*/kernels.csv", "outputs/hpc_E33q_s*/kernels.csv")
    n33 = _csvs("outputs/hpc_E33n*/nets.csv")
    orc = pd.read_csv(ROOT / "outputs/hpc_E33o/oracles.csv")
    # G1: one row per (cell, seed, M, arm, selection)
    dup = new.duplicated(KEY + ["arm", "selection"])
    assert not dup.any(), f"duplicate E60 rows: {new[dup].head()}"
    new = new[new.cell.isin(CELLS) & new.seed.isin(SEEDS) & new.n_chains.isin(SIZES)]
    check_bundles(new, k33)
    # G3: E33's test bundle does not depend on M, so lmmse_true (scored at M = 2048) is every M's floor
    tb = k33[k33.cell.isin(CELLS)].groupby(["cell", "seed"]).test_bundle.nunique()
    assert (tb == 1).all(), f"E33's test bundle depends on M: {tb[tb > 1].head()}"
    floor = orc[orc.arm == "lmmse_true"].drop_duplicates(["cell", "seed"]).set_index(["cell", "seed"]).risk
    n33 = n33.assign(arm=n33.arch + "_" + n33.arm, selection="risk")
    rows = pd.concat([new, k33[k33.arm.isin(E33_KERNELS)], n33[n33.arm.isin(NETS)]], ignore_index=True)
    rows = rows[rows.cell.isin(CELLS) & rows.seed.isin(SEEDS)]
    f = floor.reindex(pd.MultiIndex.from_frame(rows[["cell", "seed"]])).to_numpy()
    rows = rows.assign(raw_risk=rows.risk, risk=rows.risk / f)
    rows.loc[rows.failed.fillna(0).astype(int) == 1, "risk"] = np.nan
    risk = rows[rows.selection == "risk"]
    assert not risk.duplicated(KEY + ["arm"]).any(), "duplicate risk-selected rows"
    return risk, rows


# ------------------------------------------------------------------ the verdict arithmetic


def _series(df, cell, arm, m, col="risk"):
    x = df[(df.cell == cell) & (df.arm == arm) & (df.n_chains == m)].set_index("seed")[col].reindex(SEEDS)
    return x.to_numpy(dtype=float)


def seed_mean(df, cell, arm, m, col="risk"):
    x = _series(df, cell, arm, m, col)
    return float(x.mean()) if np.isfinite(x).all() else np.nan


def wins(df, cell, a, b, m):
    """Seeds in which arm a is below arm b, BOTH rows finite."""
    xa, xb = _series(df, cell, a, m), _series(df, cell, b, m)
    return int((np.isfinite(xa) & np.isfinite(xb) & (xa < xb)).sum())


def best_net(df, cell, m):
    means = {a: seed_mean(df, cell, a, m) for a in NETS}
    means = {a: v for a, v in means.items() if np.isfinite(v)}
    return min(means, key=means.get) if means else None


def p1(df):
    bad = [(c, m, wins(df, c, "em_free_C4", "em_mixture", m)) for c in CELLS for m in (512, 2048)
           if wins(df, c, "em_free_C4", "em_mixture", m) < 14]
    high = [(c, seed_mean(df, c, "em_free_C4", 2048)) for c in CELLS
            if not seed_mean(df, c, "em_free_C4", 2048) < 0.5]
    ok = not bad and not high
    return ok, ("em_free_C4 below em_mixture at M = 512, 2048 in >= 14/16 on every cell and below 0.5 of the floor at 2048"
                + ("" if ok else f"; FAILS wins {bad}, seed-mean {high}"))


def p2(df):
    w = {c: wins(df, c, "em_free_C4", "em_mdn", 128) for c in CELLS}
    ok = sum(v >= 12 for v in w.values()) >= 3
    return ok, f"M = 128: em_free_C4 below em_mdn in >= 12/16 on >= 3 of 4 cells; wins {w}"


def p3(df):
    r = {c: tuple(seed_mean(df, c, "em_free_C8", m) / seed_mean(df, c, "em_free_C4", m) for m in (128, 2048))
         for c in CELLS}
    ok = all(np.isfinite(a) and np.isfinite(b) and b < a for a, b in r.values())
    return ok, "seed-mean C8/C4 smaller at 2048 than at 128 on every cell; (128, 2048): " + str(
        {c: (round(a, 3), round(b, 3)) for c, (a, b) in r.items()})


def p4(df):
    w = {c: wins(df, c, "em_pair_quartic", "em_free_C4", 2048) for c in CELLS}
    ok = sum(v >= 12 for v in w.values()) >= 3
    return ok, f"M = 2048: em_pair_quartic below em_free_C4 in >= 12/16 on >= 3 of 4 cells; wins {w}"


def p5(df):
    res = {}
    for c in CELLS:
        for m in (128, 2048):
            b = best_net(df, c, m)
            res[(c, m)] = (b, wins(df, c, "em_free_C4", b, m) if b else 0)
    ok = all(v >= 14 for _, v in res.values())
    return ok, f"em_free_C4 below the best network at M = 128, 2048 in >= 14/16 on every cell; {res}"


PREDICTIONS = {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "P5": p5}


def verdicts(df):
    return {k: f(df) for k, f in PREDICTIONS.items()}


# ------------------------------------------------------------------ descriptive


def d1_table(df):
    arms = FREE + E33_KERNELS + NETS
    out = {}
    for c in CELLS:
        out[c] = pd.DataFrame({m: {a: seed_mean(df, c, a, m) for a in arms} for m in SIZES})
    return out


def free_shape(theta: str, components: int):
    """Effective number of components and the sorted slopes of a fitted free mixture, from its theta column."""
    t = np.array([float(v) for v in theta.split(";")])
    logit, chol = t[:components], t[3 * components:].reshape(components, 3)
    pi = np.exp(logit - np.logaddexp.reduce(logit))
    eff = float(np.exp(-(pi * np.log(pi)).sum()))
    return eff, np.sort(chol[:, 1] / np.exp(chol[:, 0]))


def main() -> int:
    df, rows = load()
    print("D1 seed-mean risk / lmmse_true (risk selection; NaN = a seed failed or is missing)")
    for c, t in d1_table(df).items():
        print(f"  {c}\n" + t.round(3).to_string().replace("\n", "\n    "))
    for k, (ok, text) in verdicts(df).items():
        print(f"{k}: {'CONFIRMED' if ok else 'FALSIFIED'} -- {text}")
    print("D2 em_free_C4 at M = 2048: median effective components and sorted slopes over seeds")
    for c in CELLS:
        sel = df[(df.cell == c) & (df.arm == "em_free_C4") & (df.n_chains == 2048)].dropna(subset=["theta"])
        shapes = [free_shape(t, 4) for t in sel.theta]
        if shapes:
            print(f"  {c}: eff {np.median([s[0] for s in shapes]):.2f}, slopes {np.round(np.median([s[1] for s in shapes], 0), 2).tolist()}")
    print("D3 evidence- against risk-selected seed-mean risk / floor")
    for a in FREE:
        ev = rows[(rows.arm == a) & (rows.selection == "evidence")]
        for c in CELLS:
            print(f"  {a} {c}: " + ", ".join(f"M={m}: {seed_mean(ev, c, a, m):.3f} vs {seed_mean(df, c, a, m):.3f}" for m in SIZES))
    print("D4 selected checkpoint (median) and share at the cap")
    for a in FREE:
        s = df[df.arm == a]
        g = s.groupby("n_chains").checkpoint
        print(f"  {a}: median " + str(g.median().to_dict()) + ", at cap " + str((s.checkpoint >= CAP).groupby(s.n_chains).mean().round(3).to_dict()))
    print("D5 median wall seconds per fit")
    w = rows[rows.arm.isin(FREE + E33_KERNELS) & (rows.selection == "risk")].groupby(["arm", "n_chains"]).wall.median()
    print("  " + w.unstack().round(0).to_string().replace("\n", "\n  "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
