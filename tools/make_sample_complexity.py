#!/usr/bin/env python
"""Emit the sample-complexity reading from E12 / E12b (predeclared analysis, registry E12.json).

Usage: make_sample_complexity.py [--src outputs/E12] [--dest paper/sections]

The corollary says M*(t), the number of sequences needed for a fixed accuracy at observation
level t, grows as e^{4t} for a covariance coordinate and e^{8t} for a shape coordinate, so
their ratio grows as e^{4t}. This reads that off, and refuses to read it where the run cannot
support it.

What it computes, in the order the registry declares:

  M*(t)     the M at which the seed-pooled RMSE curve first crosses a fixed threshold,
            interpolated log-linearly between the two bracketing rungs. A level whose curve
            never crosses on the ladder yields no M*: it yields a one-sided BOUND, which is
            reported as such and is not silently dropped.
  slope     OLS log-slope of M* against t across the levels that resolved, with a bootstrap
            over seeds. Fewer than three resolved levels means the slope is a two-point
            difference with no residual degrees of freedom; it is still printed, and it is
            labelled.
  ratio     the PAIRED M*_shape / M*_cov per level and the log-slope of that. This is the
            corollary's own statement and the better-conditioned estimand, because the
            common seed noise cancels between the two coordinates.

The budget gate. At each (level, M) the fraction of cells censored by the EM iteration cap is
computed first. A level whose cells are predominantly at the cap did not measure a sample
size, it measured an optimiser budget (docs/NEGATIVE_RESULTS.md N-12), and this tool marks
it CONFOUNDED and excludes it from the slope while still reporting its bound.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KURT_THR = 0.5        # accuracy at which M* is read for the shape coordinate
RHO_THR = 0.015       # and for the covariance coordinate
CENSOR_MAX = 0.75     # above this at-cap fraction a level is a budget statement, not a sample one
B = 4000


def _mstar(sub: pd.DataFrame, col: str, thr: float, sizes: np.ndarray) -> float:
    y = np.array([np.sqrt((sub[sub.n_chains == M][col] ** 2).mean()) for M in sizes])
    for i in range(len(sizes) - 1):
        if y[i] > thr >= y[i + 1]:
            f = (np.log(y[i]) - np.log(thr)) / (np.log(y[i]) - np.log(y[i + 1]))
            return float(np.exp(np.log(sizes[i]) + f * (np.log(sizes[i + 1]) - np.log(sizes[i]))))
    return float("nan")


def _slope(xs, ys):
    """OLS log-slope of M* against t; xs are levels, ys are M*."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(ys)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(xs[ok], np.log(ys[ok]), 1)[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "outputs" / "E12")
    ap.add_argument("--dest", type=Path, default=ROOT / "paper" / "sections")
    ap.add_argument("--kurt-threshold", type=float, default=KURT_THR)
    ap.add_argument("--rho-threshold", type=float, default=RHO_THR)
    a = ap.parse_args()
    dest = a.dest if "dev" not in a.src.name else ROOT / "outputs" / "scratch"
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260918)

    df = pd.read_csv(a.src / "ladder.csv")
    sizes = np.array(sorted(df.n_chains.unique()))
    levels = sorted(df.t_obs.unique())
    seeds = np.array(sorted(df.seed.unique()))

    # --- the budget gate, first, because it decides which levels may be read at all
    cap = df.groupby("t_obs").at_cap.mean() if "at_cap" in df else pd.Series(0.0, index=levels)
    cens = (df.assign(c=(df.stop_reason == "censored")).groupby("t_obs").c.mean()
            if "stop_reason" in df else pd.Series(np.nan, index=levels))
    usable = [t for t in levels if cap.get(t, 0.0) <= CENSOR_MAX]
    blocked = [t for t in levels if t not in usable]

    rows = []
    for t in levels:
        sub = df[df.t_obs == t]
        mk = _mstar(sub, "kurt_err", a.kurt_threshold, sizes)
        mr = _mstar(sub, "rho_err", a.rho_threshold, sizes)
        floor_k = min(np.sqrt((sub[sub.n_chains == M]["kurt_err"] ** 2).mean()) for M in sizes)
        rows.append(dict(t=t, m_shape=mk, m_cov=mr,
                         ratio=(mk / mr) if np.isfinite(mk) and np.isfinite(mr) else np.nan,
                         at_cap=float(cap.get(t, np.nan)), censored=float(cens.get(t, np.nan)),
                         shape_floor=float(floor_k),
                         status=("CONFOUNDED" if t in blocked else
                                 ("resolved" if np.isfinite(mk) else "shape unresolved"))))
    tab = pd.DataFrame(rows)

    # --- slopes, on the usable and resolved levels only
    use = tab[(tab.t.isin(usable))]
    s_shape = _slope(use.t, use.m_shape)
    s_cov = _slope(use.t, use.m_cov)
    s_ratio = _slope(use.t, use.ratio)

    # The bootstrap runs on a dense (seed, level, size) array of SQUARED errors rather than on
    # the frame: resampling seeds is then an index into axis 0 and the pooled RMSE is a mean
    # over it, so all B resamples for all three estimands are a handful of array operations.
    # Building it per resample out of the frame is the same arithmetic and ~10^4 times slower.
    ut = np.asarray(use.t, float)
    sq = {}
    for col in ("kurt_err", "rho_err"):
        A = np.full((len(seeds), len(ut), len(sizes)), np.nan)
        for i, sd in enumerate(seeds):
            for j, t in enumerate(ut):
                for k, M in enumerate(sizes):
                    v = df[(df.seed == sd) & (df.t_obs == t) & (df.n_chains == M)][col].to_numpy()
                    if len(v):
                        A[i, j, k] = float(v[0]) ** 2
        sq[col] = A

    def _cross(y, thr):
        for i in range(len(y) - 1):
            if y[i] > thr >= y[i + 1]:
                f = (np.log(y[i]) - np.log(thr)) / (np.log(y[i]) - np.log(y[i + 1]))
                return float(np.exp(np.log(sizes[i]) + f * (np.log(sizes[i + 1]) - np.log(sizes[i]))))
        return np.nan

    # With few seeds the bootstrap distribution is small enough to ENUMERATE: every multiset
    # of n draws from n seeds, weighted by its multinomial probability. That removes the Monte
    # Carlo error entirely, which matters here because the 97.5th percentile rides on a handful
    # of extreme resamples -- at B = 4000 the upper limit of the covariance slope wandered
    # between 7.7 and 8.2 across streams, i.e. across the prediction it was being tested
    # against. Above the enumeration limit, Monte Carlo with a large B and a reported error.
    n = len(seeds)
    from itertools import combinations_with_replacement
    from math import comb, factorial
    from collections import Counter
    exact = comb(2 * n - 1, n) <= 200_000
    if exact:
        draws = [np.array(ms) for ms in combinations_with_replacement(range(n), n)]
        weights = np.array([factorial(n) / np.prod([factorial(v) for v in Counter(ms).values()]) / n ** n
                            for ms in draws])
    else:
        draws = list(rng.integers(0, n, size=(B, n)))
        weights = np.full(len(draws), 1.0 / len(draws))
    boots = {"shape": ([], []), "cov": ([], []), "ratio": ([], [])}
    for d, w in zip(draws, weights):
        rk = np.sqrt(np.nanmean(sq["kurt_err"][d], axis=0))   # (levels, sizes)
        rr = np.sqrt(np.nanmean(sq["rho_err"][d], axis=0))
        mk = np.array([_cross(rk[j], a.kurt_threshold) for j in range(len(ut))])
        mr = np.array([_cross(rr[j], a.rho_threshold) for j in range(len(ut))])
        for kind, vals in (("shape", mk), ("cov", mr), ("ratio", mk / mr)):
            v = _slope(ut, vals)
            if np.isfinite(v):
                boots[kind][0].append(v); boots[kind][1].append(w)

    def _wq(v, w, q):
        v, w = np.asarray(v, float), np.asarray(w, float)
        o = np.argsort(v); v, w = v[o], w[o]
        cw = np.cumsum(w) / w.sum()
        return float(v[min(np.searchsorted(cw, q), len(v) - 1)])

    def boot(kind):
        """(lower, upper, resolved mass). Weighted percentiles; exact when enumerated."""
        v, w = boots[kind]
        if not v:
            return np.nan, np.nan, 0.0
        return _wq(v, w, 0.025), _wq(v, w, 0.975), float(np.sum(w))

    macros, report = {}, []
    for name, val, pred, key in (("Shape", s_shape, 8, "shape"), ("Cov", s_cov, 4, "cov"),
                                 ("Ratio", s_ratio, 4, "ratio")):
        lo, hi, mass = boot(key)
        excl = "excludes nothing"
        if np.isfinite(lo) and np.isfinite(hi):
            out = [p for p in (4, 8) if not (lo <= p <= hi)]
            if out:
                excl = "excludes " + ", ".join(str(p) for p in out)
        macros[f"scSlope{name}"] = f"{val:.2f}"
        macros[f"scSlope{name}Lo"] = f"{lo:.2f}"; macros[f"scSlope{name}Hi"] = f"{hi:.2f}"
        report.append(f"  {name:6s} slope {val:6.2f}  95% CI [{lo:6.2f}, {hi:6.2f}]  "
                      f"predicted {pred}  -> {excl}  (resolved mass {mass:.3f})")

    # --- one-sided bounds from levels where the shape coordinate never resolved
    bounds = []
    ref = use[np.isfinite(use.m_shape)]
    if len(ref):
        t0, m0 = float(ref.t.iloc[0]), float(ref.m_shape.iloc[0])
        for _, r in tab[~np.isfinite(tab.m_shape)].iterrows():
            if r.t <= t0:
                continue
            lb = np.log(sizes.max() / m0) / (r.t - t0)
            bounds.append((r.t, lb))
            macros[f"scBoundT{str(r.t).replace('.', 'p')}"] = f"{lb:.2f}"

    macros["scKurtThr"] = f"{a.kurt_threshold:g}"; macros["scRhoThr"] = f"{a.rho_threshold:g}"
    macros["scSeeds"] = str(len(seeds)); macros["scLevels"] = str(len(levels))
    macros["scUsable"] = str(len(usable)); macros["scMaxM"] = str(int(sizes.max()))
    (dest / "sample-complexity-numbers.tex").write_text(
        "%% GENERATED by tools/make_sample_complexity.py -- do not hand-edit.\n"
        + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")

    print(tab.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\nusable levels (at-cap <= {CENSOR_MAX}): {usable or 'NONE'}")
    if blocked:
        print(f"CONFOUNDED and excluded from the slopes: {blocked} -- these levels measured the "
              f"iteration budget, not the sample size (see docs/NEGATIVE_RESULTS.md N-12)")
    print(f"\nlog-slopes d log M*/dt on the usable levels "
          f"({'EXACT bootstrap, ' + str(len(draws)) + ' multisets' if exact else 'Monte Carlo, B = ' + str(B)}):")
    print("\n".join(report))
    for t, lb in bounds:
        print(f"  one-sided: shape unresolved at t={t:g} up to M={sizes.max()}, so "
              f"d log M*/dt > {lb:.2f}")
    if len(use) < 3:
        print(f"\nNOTE: {len(use)} usable level(s). A slope from fewer than three points has no "
              f"residual degrees of freedom and is a difference, not a fit. Reported as such.")
    print(f"\nwrote {dest/'sample-complexity-numbers.tex'} ({len(macros)} macros)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
