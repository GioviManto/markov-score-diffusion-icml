"""Read E62 against its registered predictions (experiments/registry/E62.json).

    python tools/core/read_e62.py            # gates, the D1 table against delta, verdicts P1-P5, descriptive D2-D4

Laplace AR(2) chains at lag-one correlation 0.85, phi2 in (-0.3, 0, 0.1, 0.2, 0.3, 0.4, 0.6). Arms: first (first-order
EM-BP, the thesis mixture), pair (the two-lag mixture by EM through pair-state BP), mlp and window, lmmse (the true AR(2)
covariance). Risks are E03's pooled relative score error against the pair-state reference; on wind, against the clean
value (a paired comparison ranks methods as the score error does).

A FAILED or missing fit counts AGAINST every prediction it enters: a seed mean needs all eight seeds finite, a paired win
needs BOTH rows, and "the best network" is the lower seed mean of the two that have one. P1's floor is the registered
calibration (E62.json, calibration_2026-11-02.per_phi2.first_order_floor).

The verdict functions take long tables (phi2, arm, n_chains, seed, risk) and (arm, n_chains, seed, risk), so
tests/test_read_e62.py drives them with made-up tables in both directions.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(8))
SIZES = (128, 512, 2048)
WIND_SIZES = (32, 128, 512, 1917)
PHI2 = (-0.3, 0.0, 0.1, 0.2, 0.3, 0.4, 0.6)
NETS = ("mlp", "window")


def calibration():
    c = json.loads((ROOT / "experiments/registry/E62.json").read_text())["calibration_2026-11-02"]["per_phi2"]
    return {round(float(k), 2): v for k, v in c.items()}


def _csvs(*patterns):
    paths = sorted(p for pattern in patterns for p in glob.glob(str(ROOT / pattern)))
    if not paths:
        raise SystemExit(f"no files match {patterns}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def check_bundles(df: pd.DataFrame, keys) -> None:
    b = df.dropna(subset=["test_bundle"]).groupby(list(keys)).test_bundle.nunique()
    if (b > 1).any():
        raise SystemExit(f"arms disagree on the test bundle: {b[b > 1].head()}")


def _one_per_fit(d, keys):
    dup = d.duplicated(list(keys) + ["arm", "t"])
    assert not dup.any(), f"duplicate rows: {d[dup].head()}"
    d = d.drop_duplicates(list(keys) + ["arm"]).copy()
    d["risk"] = d.risk_pooled
    d.loc[d.failed.fillna(0).astype(int) == 1, "risk"] = np.nan
    return d


def load(arms="outputs/hpc_E62_p*/arms.csv", wind="outputs/hpc_E62_wind/wind.csv"):
    a = _one_per_fit(_csvs(arms), ("phi2", "seed", "n_chains"))
    a["phi2"] = a.phi2.round(2)
    a = a[a.phi2.isin(PHI2) & a.seed.isin(SEEDS) & a.n_chains.isin(SIZES)]
    check_bundles(a, ("phi2", "seed", "n_chains"))
    w = _one_per_fit(_csvs(wind), ("seed", "n_chains"))
    w = w[w.seed.isin(SEEDS) & w.n_chains.isin(WIND_SIZES)]
    check_bundles(w, ("seed",))
    return a, w


# ------------------------------------------------------------------ the verdict arithmetic


def _series(df, arm, m, phi2=None):
    sel = (df.arm == arm) & (df.n_chains == m)
    if phi2 is not None:
        sel &= np.isclose(df.phi2, phi2)
    return df[sel].set_index("seed").risk.reindex(SEEDS).to_numpy(dtype=float)


def seed_mean(df, arm, m, phi2=None):
    x = _series(df, arm, m, phi2)
    return float(x.mean()) if np.isfinite(x).all() else np.nan


def wins(df, a, b, m, phi2=None):
    xa, xb = _series(df, a, m, phi2), _series(df, b, m, phi2)
    return int((np.isfinite(xa) & np.isfinite(xb) & (xa < xb)).sum())


def best_net(df, m, phi2=None):
    means = {a: seed_mean(df, a, m, phi2) for a in NETS}
    means = {a: v for a, v in means.items() if np.isfinite(v)}
    return min(means, key=means.get) if means else None


def p1(df, floor):
    res = {}
    for p in (0.2, 0.3, 0.4, 0.6):
        s2048, s512 = seed_mean(df, "first", 2048, p), seed_mean(df, "first", 512, p)
        f = floor[p]["first_order_floor"]
        res[p] = (round(s2048 / f, 3) if np.isfinite(s2048) else np.nan, bool(np.isfinite(s2048) and np.isfinite(s512)
                                                                                and s2048 >= 0.9 * s512))
    ok = all(np.isfinite(r) and 0.75 <= r <= 1.25 and flat for r, flat in res.values())
    return ok, f"first at 2048 within +-25% of the calibrated floor and not >10% below 512, phi2 >= 0.2; (ratio, flat) {res}"


def p2(df):
    w = {p: wins(df, "pair", "first", 2048, p) for p in (0.1, 0.2, 0.3, 0.4, 0.6)}
    return all(v >= 7 for v in w.values()), f"pair below first at 2048 in >= 7/8, phi2 >= 0.1; wins {w}"


def p3(df):
    below = {p: (seed_mean(df, "first", 2048, p), seed_mean(df, best_net(df, 2048, p), 2048, p) if best_net(df, 2048, p)
                 else np.nan) for p in (0.0, 0.1, 0.4, 0.6)}
    ok = all(np.isfinite(a) and np.isfinite(b) and (a < b if p <= 0.1 else a > b) for p, (a, b) in below.items())
    return ok, "first below the best network at phi2 <= 0.1 and above it at phi2 >= 0.4 (2048); (first, best net) " + str(
        {p: (round(a, 4), round(b, 4)) for p, (a, b) in below.items()})


def p4(df):
    res = {(p, m): (best_net(df, m, p), wins(df, "pair", best_net(df, m, p), m, p) if best_net(df, m, p) else 0)
           for p in PHI2 for m in SIZES}
    bad = {k: v for k, v in res.items() if v[1] < 7}
    return not bad, "pair below the best network at every phi2 and M in >= 7/8" + (f"; FAILS {bad}" if bad else "")


def p5(wind):
    w = {m: wins(wind, "pair", "first", m) for m in WIND_SIZES}
    return all(v >= 7 for v in w.values()), f"wind: pair below first in >= 7/8 at every N; wins {w}"


def verdicts(arms, wind, floor):
    return {"P1": p1(arms, floor), "P2": p2(arms), "P3": p3(arms), "P4": p4(arms), "P5": p5(wind)}


def break_even(df, floor, m=2048):
    """D2: the delta_exact at which first's seed mean crosses the best network's (linear interpolation in delta)."""
    pts = sorted((floor[p]["delta_exact"], seed_mean(df, "first", m, p) - seed_mean(df, best_net(df, m, p), m, p))
                 for p in PHI2 if p >= 0 and best_net(df, m, p))
    for (d0, g0), (d1, g1) in zip(pts, pts[1:]):
        if np.isfinite(g0) and np.isfinite(g1) and g0 < 0 <= g1:
            return d0 + (d1 - d0) * (-g0) / (g1 - g0)
    return np.nan


def main() -> int:
    arms, wind = load()
    floor = calibration()
    print("D1 seed-mean risk per phi2 (rows) against delta, by arm and M")
    for m in SIZES:
        t = pd.DataFrame({a: {p: seed_mean(arms, a, m, p) for p in PHI2} for a in ("first", "pair", "mlp", "window", "lmmse")})
        t.insert(0, "delta_exact", [floor[p]["delta_exact"] for p in PHI2])
        t.insert(1, "floor", [floor[p]["first_order_floor"] for p in PHI2])
        print(f"  M = {m}\n    " + t.round(4).to_string().replace("\n", "\n    "))
    for k, (ok, text) in verdicts(arms, wind, floor).items():
        print(f"{k}: {'CONFIRMED' if ok else 'FALSIFIED'} -- {text}")
    print(f"D2 break-even delta_exact (first against the best network, M = 2048): {break_even(arms, floor):.4f}")
    pr = arms[arms.arm == "pair"]
    if "fit_phi2" in pr:
        print("D3 pair's fitted phi2 at M = 2048 (seed mean) against the truth: " + str(
            {p: round(float(pr[(pr.n_chains == 2048) & np.isclose(pr.phi2, p)].fit_phi2.mean()), 3) for p in PHI2}))
    print("D4 wind: seed-mean risk against the clean value by arm and N\n    " + pd.DataFrame(
        {a: {m: seed_mean(wind, a, m) for m in WIND_SIZES} for a in ("first", "pair", "mlp", "window")}).round(4)
        .to_string().replace("\n", "\n    "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
