#!/usr/bin/env python
"""The predeclared readings of E41 and E42 (registered 8 Oct 2026, before submission).

    python tools/make_extension_numbers.py                  # the confirmatory shards, against E33's frozen fits

E41 (experiments/registry/E41.json): the MDN kernel at four times E33's EM budget, M = 2048, seeds 0-7, four cells.
  P1  paired with E33's MDN on the same seeds, the seed-mean risk is lower on all four cells and by > 10% on >= 3
  P2  at most 30% of the fits select the last checkpoint (E33: 78%)
  P3  MDN / spline (seed means) >= 1.5 on every cell, and MDN above the spline in >= 7/8 seeds on every cell
  P4  MDN below the mixture in >= 7/8 seeds on every cell
  P5  MDN above the site energy in >= 7/8 seeds on every cell
E42 (experiments/registry/E42.json): the site energy's barrier read-out at M = 8192, 32768 (T = 1, 2), with E33's M = 512, 2048.
  P1  log-log slope of the seed-mean |barrier - 1/4| against M over (512, 2048, 8192, 32768) in [-0.65, -0.35], both T
  P2  T = 1: within 20% (|barrier - 1/4| <= 0.05) in 16/16 seeds at 8192 and at 32768
  P3  T = 2: within 20% in >= 15/16 seeds at 32768
  P4  the count of seeds within 20% does not fall along 2048, 8192, 32768, both T
  P5  T = 2: |seed-mean signed barrier error| < 0.01 at 32768
  P6  |Theta_hat / Theta - 1| <= 0.10 in 16/16 seeds, both T, both new sizes

E46 (experiments/registry/E46.json): the b = 0 gain of the augmented kernel, tested with the first-order kernel on mirrored data.
  P1  the mean per-seed log(sym / augmented) has a 95% interval containing 0 at both M
  P2  the mean log(sym / first-order) has an interval entirely below 0 at both M
  P3  the upper end of the interval of the mean log(sym / 2m) is below log 1.1 at both M

E43 (experiments/registry/E43.json): the price and the benefit of an unrestricted kernel (em_loglinear).
  Lab A (E33's cells T0.3_h1_L32, T1_iso_L32; comparators are E33's frozen fits on the same seeds)
  P1  M = 128: loglinear / site energy >= 2 in >= 14/16 seeds on both cells
  P2  the seed-mean ratio loglinear / site energy is smaller at 2048 than at 128 on both cells, and below 2 at 2048
  P3  M = 2048: loglinear below the mixture in >= 14/16 seeds on both cells
  P4  M = 128: loglinear above the tilted spline in >= 12/16 seeds on both cells
  Lab B (the Laplace-bond chain, all arms in the run)
  P5  M = 2048: loglinear below the site energy in >= 7/8 seeds, seed-mean ratio site energy / loglinear >= 1.25
  P6  M = 2048: loglinear below the tilted spline and below the mixture, each in >= 7/8 seeds
  P7  the seed-mean ratio loglinear / site energy is above 1 at M = 128 and below 1 at M = 2048

E45 (experiments/registry/E45.json): inpainting with E33's frozen fitted kernels (laptop, FFBS).
  P1  oracle 90% interval coverage in [0.87, 0.93] on every (cell, pattern), pooled over seeds
  P2  site energy: coverage within 0.03 of 0.90 on every (cell, pattern), and hidden-site error within 10% of the oracle's in >= 14/16 seeds
  P3  linear conditional: |coverage - 0.90| > 0.03 on (T0.3_h1_L32, gap16)
  P4  (T0.3_h1_L32, gap16): linear error / oracle error (seed means) in [0.9, 1.1]
  P5  (T0.3_h1_L32, gap16): linear 90% interval width / oracle width (seed means) >= 1.1

E44 (experiments/registry/E44.json): non-equilibrium snapshots; R = arm risk / linear_empirical risk, paired per seed.
  P1  tau = infinity: the site energy beats the empirical linear estimator in 8/8 seeds
  P2  the site energy's R < 1 in >= 7/8 seeds at EVERY tau
  P3  tau = 1.5, 3, 6: the spline's risk is below the site energy's in >= 7/8 seeds, at each
  P4  tau = infinity: the site energy's risk is below the spline's in >= 7/8 seeds

E47 (experiments/registry/E47.json): the coarse grid. Oracle part descriptive (scouted before registration); learned part against E33's N_g = 401 fits:
  P1  base cell, M = 2048: N_g = 101 and 201: seed-mean risk ratio vs 401 in [0.99, 1.01], both arms
  P2  base cell, M = 2048, N_g = 51: both arms in [0.97, 1.03]
  P3  finest lattice, M = 2048: N_g = 101 in [0.99, 1.01] and N_g = 51 in [0.90, 1.10], both arms
  P4  M = 128, both cells: N_g = 101 in [0.99, 1.01], both arms
  P5  site-energy EM fit time, base cell, M = 2048: median wall at 101 <= 1/4 and at 51 <= 1/8 of E33's at 401
  P6  BP time per sequence at N_g = 51 <= 0.07 ms; at 101 within 20% of 0.114 ms
  P7  log-log slope of time vs N_g over 41-401 < 1.5
  P8  headline kernels (E04 on frozen E02a kernels, no refit): at N_g = 201 each arm's seed-mean risk within 1% of N_g = 401; P9  at 101 within 5%
  P10 at least 90% of the C = 8 kernels 'resolved' at N_g = 101, and a lower fraction at 51
E48 (experiments/registry/E48.json): the MDN with a clean ladder to 16000, against E36x's frozen fits.
  P1  stride 4: the MDN selects a checkpoint below the last in >= 12/16 seeds
  P2  stride 4: MDN below the site energy in >= 14/16 seeds
  P3  stride 1: MDN above the site energy in 16/16 seeds
  P4  stride 1: seed-mean ratio MDN / site energy < 5
  P5  stride 2: MDN below the site energy in >= 10/16 seeds
E49 (experiments/registry/E49.json): the MDN on E43's out-of-class chain, against E43's frozen lab-B fits at M = 2048.
  P1  MDN below the site energy in >= 7/8 seeds;  P2  below the mixture in >= 7/8;  P3  the log-linear kernel below the MDN in >= 6/8;  P4  <= 30% of MDN fits at the last checkpoint

A fit that raised in EM is a FAILED row (E33 amendment 3): it counts against every clause it touches and never for one.
The test bundles of E41 and E33 are compared by hash; a mismatch stops the reading.
"""
from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CELLS41 = ("T0.3_h1_L32", "T0.5_iso_L32", "T1_iso_L32", "T2_iso_L32")
TCELL = {"T1_iso_L32": 1.0, "T2_iso_L32": 2.0}
BARRIER = 0.25


def _load(spec):
    paths = sorted(glob.glob(str(ROOT / spec))) if not Path(spec).is_absolute() else sorted(glob.glob(spec))
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else pd.DataFrame()


def row(exp, pred, clause, value, k, n, ok, note=""):
    return dict(experiment=exp, prediction=pred, clause=clause, value=value, k=k, n=n,
                verdict="UNRESOLVED" if ok is None else ("CONFIRMED" if ok else "FALSIFIED"), note=note)


def _risk_rows(d, arm, n=None):
    s = d[(d.arm == arm) & (d.selection == "risk")]
    if n is not None:
        s = s[s.n_chains == n]
    return s.set_index(["cell", "seed"]).sort_index()


# --------------------------------------------------------------------------- E41
def read_e41(d41, d33, size=2048):
    rows = []
    mdn = _risk_rows(d41, "em_mdn", size)
    ns = mdn.index.get_level_values("seed").nunique()
    need = max(1, math.ceil(7 * ns / 8))
    old = _risk_rows(d33, "em_mdn", size).reindex(mdn.index)
    if (old.test_bundle.values != mdn.test_bundle.values).any():
        raise ValueError("E41 and E33 were scored on different test bundles; the paired reading is not valid")
    comp = {a: _risk_rows(d33, a, size).reindex(mdn.index) for a in ("em_tilted_spline", "em_mixture", "em_pair_quartic")}
    cells = [c for c in CELLS41 if c in mdn.index.get_level_values("cell")]
    # P1: the budget was binding
    lower = []
    for c in cells:
        new, prev = mdn.loc[c].risk, old.loc[c].risk
        lower.append((float(new.mean()), float(prev.mean())))
    n_lower = sum(a < b for a, b in lower)
    n_10 = sum(a < 0.9 * b for a, b in lower)
    rows.append(row("E41", "P1", "MDN seed-mean risk lower than E33's on all cells, by > 10% on >= 3", n_10, n_lower, len(cells),
                    n_lower == len(cells) and n_10 >= 3,
                    "; ".join(f"{c}: {a:.4f} vs {b:.4f}" for c, (a, b) in zip(cells, lower))))
    # P2: the budget is no longer binding (a failed fit counts as capped)
    share = float(((mdn.at_cap == 1) | mdn.risk.isna()).mean())
    rows.append(row("E41", "P2", "share of fits at the last checkpoint <= 0.30", share, int(((mdn.at_cap == 1) | mdn.risk.isna()).sum()), len(mdn), share <= 0.30,
                    "E33: 149/192 = 0.78 at M = 2048"))
    # P3-P5: the ranking against E33's frozen fits, paired by seed
    for pred, arm, label, sign in (("P3", "em_tilted_spline", "spline", +1), ("P4", "em_mixture", "mixture", -1), ("P5", "em_pair_quartic", "site energy", +1)):
        ok_all, notes = True, []
        for c in cells:
            m, o = mdn.loc[c].risk, comp[arm].loc[c].risk
            k = int((m > o).sum()) if sign > 0 else int((m < o).sum())
            ratio = float(m.mean() / o.mean())
            good = k >= need and (ratio >= 1.5 if pred == "P3" else True)
            ok_all &= good
            notes.append(f"{c}: {k}/{len(m)}, ratio {ratio:.2f}")
        direction = "above" if sign > 0 else "below"
        extra = " and seed-mean ratio >= 1.5" if pred == "P3" else ""
        rows.append(row("E41", pred, f"MDN {direction} {label} in >= {need}/{ns} seeds on every cell{extra}", np.nan, None, len(cells), bool(ok_all), "; ".join(notes)))
    return rows


# --------------------------------------------------------------------------- E42
def read_e42(tab, failed):
    """tab: one row per good fit with cell, n_chains, seed, barrier_hat, theta_ratio; failed: {(cell, n_chains): count}."""
    rows = []
    t = tab[tab.cell.isin(TCELL)].copy()
    t["err"] = (t.barrier_hat - BARRIER).abs()

    def total(c, n):
        return int(((t.cell == c) & (t.n_chains == n)).sum()) + int(failed.get((c, n), 0))

    def within(c, n):
        return int(((t.cell == c) & (t.n_chains == n) & (t.err <= 0.05)).sum())

    # P1: slopes (failed fits are absent from the means; their count is reported)
    for c, T in TCELL.items():
        s = t[(t.cell == c) & t.n_chains.isin((512, 2048, 8192, 32768))].groupby("n_chains").err.mean()
        slope = float(np.polyfit(np.log(s.index.values.astype(float)), np.log(s.values), 1)[0]) if len(s) >= 3 else float("nan")
        rows.append(row("E42", "P1", f"T = {T:g}: log-log slope of the seed-mean |barrier error| over M = 512..32768 in [-0.65, -0.35]", slope, None, len(s),
                        None if np.isnan(slope) else bool(-0.65 <= slope <= -0.35), "mean error: " + ", ".join(f"{int(n)}: {v:.4f}" for n, v in s.items())))
    # P2: T = 1 resolves
    ks = [(n, within("T1_iso_L32", n), total("T1_iso_L32", n)) for n in (8192, 32768)]
    rows.append(row("E42", "P2", "T = 1: within 20% in 16/16 at M = 8192 and 32768", np.nan, min(k for _, k, _ in ks), 16, all(k == tt == 16 for _, k, tt in ks),
                    ", ".join(f"{n}: {k}/{tt}" for n, k, tt in ks)))
    # P3: T = 2 resolves at 32768
    k, tt = within("T2_iso_L32", 32768), total("T2_iso_L32", 32768)
    rows.append(row("E42", "P3", "T = 2: within 20% in >= 15/16 at M = 32768", np.nan, k, tt, bool(tt == 16 and k >= 15), "E33: 6/16 at M = 2048"))
    # P4: monotone counts
    ok4, notes = True, []
    for c, T in TCELL.items():
        seq = [within(c, n) for n in (2048, 8192, 32768)]
        ok4 &= all(a <= b for a, b in zip(seq, seq[1:]))
        notes.append(f"T = {T:g}: {seq}")
    rows.append(row("E42", "P4", "count within 20% does not fall along M = 2048, 8192, 32768", np.nan, None, 2, bool(ok4), "; ".join(notes)))
    # P5: variance, not bias
    s = t[(t.cell == "T2_iso_L32") & (t.n_chains == 32768)]
    bias = float((s.barrier_hat - BARRIER).mean()) if len(s) else float("nan")
    rows.append(row("E42", "P5", "T = 2: |seed-mean signed barrier error| < 0.01 at M = 32768", bias, None, len(s), None if np.isnan(bias) else bool(abs(bias) < 0.01),
                    "E33: about +0.03 at M = 2048"))
    # P6: the temperature
    ok6, notes = True, []
    for c, T in TCELL.items():
        for n in (8192, 32768):
            g = t[(t.cell == c) & (t.n_chains == n)]
            k, tt = int((np.abs(g.theta_ratio - 1) <= 0.10).sum()), total(c, n)
            ok6 &= (k == tt == 16)
            notes.append(f"T = {T:g}, M = {n}: {k}/{tt}")
    rows.append(row("E42", "P6", "|Theta_hat / Theta - 1| <= 0.10 in 16/16, both T, both sizes", np.nan, None, 4, bool(ok6), "; ".join(notes)))
    return rows


# --------------------------------------------------------------------------- E44
def read_e44(d):
    rows = []
    d = d[d.selection == "risk"]
    taus = sorted(d.tau.unique())
    piv = {tau: g.pivot_table(index="seed", columns="arm", values="risk", dropna=False) for tau, g in d.groupby("tau")}
    ns = d.seed.nunique()
    need = _need(ns, 7 / 8)
    inf = float("inf")
    if inf in piv:
        p = piv[inf]
        k = int((p.em_pair_quartic < p.linear_empirical).sum())
        rows.append(row("E44", "P1", f"tau = inf: site energy beats the empirical linear estimator in {ns}/{ns} seeds", float((p.em_pair_quartic / p.linear_empirical).mean()), k, ns, bool(k == ns)))
        k = int((p.em_pair_quartic < p.em_tilted_spline).sum())
        rows.append(row("E44", "P4", f"tau = inf: site energy below the spline in >= {need}/{ns} seeds", float((p.em_pair_quartic / p.em_tilted_spline).mean()), k, ns, bool(k >= need)))
    ok, notes = True, []
    for tau in taus:
        p = piv[tau]
        k = int((p.em_pair_quartic < p.linear_empirical).sum())
        ok &= k >= need
        notes.append(f"tau {tau:g}: {k}/{ns} (R {float((p.em_pair_quartic / p.linear_empirical).mean()):.2f})")
    rows.append(row("E44", "P2", f"site energy R < 1 in >= {need}/{ns} seeds at every tau", np.nan, None, len(taus), bool(ok), "; ".join(notes)))
    ok, notes = True, []
    for tau in (1.5, 3.0, 6.0):
        if tau not in piv:
            continue
        p = piv[tau]
        k = int((p.em_tilted_spline < p.em_pair_quartic).sum())
        ok &= k >= need
        notes.append(f"tau {tau:g}: {k}/{ns} (spline / site {float((p.em_tilted_spline / p.em_pair_quartic).mean()):.2f})")
    rows.append(row("E44", "P3", f"tau = 1.5, 3, 6: spline below the site energy in >= {need}/{ns} seeds, at each", np.nan, None, len(notes), bool(ok), "; ".join(notes)))
    return rows


# --------------------------------------------------------------------------- E47, E48, E49
def _within(x, lo, hi):
    return bool(np.isfinite(x) and lo <= x <= hi)


def read_e47_learned(d, d33):
    rows = []
    n_seeds = d.seed.nunique()

    def ratio(arm, cell, n, ng):
        a = d[(d.arm == arm) & (d.selection == "risk") & (d.cell == cell) & (d.n_chains == n) & (d.grid_size == ng)].set_index("seed").sort_index()
        b = d33[(d33.arm == arm) & (d33.selection == "risk") & (d33.cell == cell) & (d33.n_chains == n)].set_index("seed").reindex(a.index)
        if len(a) == 0:
            return np.nan, np.nan
        if (a.test_bundle.values != b.test_bundle.values).any():
            raise ValueError("E47 and E33 were scored on different test bundles; the paired reading is not valid")
        if a.risk.isna().any() or b.risk.isna().any():
            return np.nan, np.nan                                      # a failed fit counts against
        return float(a.risk.mean() / b.risk.mean()), float(a.wall.median() / b.wall.median())

    base, fine = "T0.3_h1_L32", "T0.3_h0.25_L128"
    arms = ("em_pair_quartic", "em_tilted_spline")
    def clause(pred, label, cells_n_ng_lo_hi):
        ok, notes = True, []
        for cell, n, ng, lo, hi in cells_n_ng_lo_hi:
            for a in arms:
                r, _ = ratio(a, cell, n, ng)
                ok &= _within(r, lo, hi); notes.append(f"{cell[:8]}/M{n}/N{ng}/{a[3:7]}: {r:.3f}")
        rows.append(row("E47", pred, label, np.nan, None, n_seeds, bool(ok), "; ".join(notes)))
    clause("P1", "base cell, M = 2048: N_g = 101 and 201 within 1% of 401, both arms", [(base, 2048, 101, .99, 1.01), (base, 2048, 201, .99, 1.01)])
    clause("P2", "base cell, M = 2048, N_g = 51 within 3% of 401, both arms", [(base, 2048, 51, .97, 1.03)])
    clause("P3", "finest lattice, M = 2048: N_g = 101 within 1%, N_g = 51 within 10%, both arms", [(fine, 2048, 101, .99, 1.01), (fine, 2048, 51, .90, 1.10)])
    clause("P4", "M = 128, both cells: N_g = 101 within 1% of 401, both arms", [(base, 128, 101, .99, 1.01), (fine, 128, 101, .99, 1.01)])
    w101 = ratio("em_pair_quartic", base, 2048, 101)[1]
    w51 = ratio("em_pair_quartic", base, 2048, 51)[1]
    rows.append(row("E47", "P5", "site-energy EM fit time, base cell, M = 2048: <= 1/4 at N_g = 101 and <= 1/8 at 51 of E33's at 401", w101, None, n_seeds,
                    bool(np.isfinite(w101) and np.isfinite(w51) and w101 <= 0.25 and w51 <= 0.125), f"N_g = 101: {w101:.3f}; N_g = 51: {w51:.3f}"))
    return rows


def read_e47_headline(d):
    d = d[d.half_width == 8.0]
    rows = []
    arms = [("risk_em_C1", "C1"), ("risk_em_C8", "C8")]
    def worst(ng):
        w, notes = 0.0, []
        for n in sorted(d.n_chains.unique()):
            for col, lab in arms:
                a = d[(d.n_chains == n) & (d.n_grid == ng)].set_index("seed")[col].sort_index()
                b = d[(d.n_chains == n) & (d.n_grid == 401)].set_index("seed")[col].reindex(a.index)
                r = float(a.mean() / b.mean()) if len(a) and b.notna().all() and a.notna().all() else np.nan
                w = max(w, abs(r - 1)) if np.isfinite(r) else np.nan
                notes.append(f"M{n}/{lab}: {r:.4f}")
        return w, "; ".join(notes)
    w201, n201 = worst(201)
    rows.append(row("E47", "P8", "headline kernels at N_g = 201: every arm's seed-mean risk within 1% of N_g = 401", w201, None, d.seed.nunique(), None if np.isnan(w201) else bool(w201 < 0.01), n201))
    w101, n101 = worst(101)
    rows.append(row("E47", "P9", "headline kernels at N_g = 101: within 5% of N_g = 401", w101, None, d.seed.nunique(), None if np.isnan(w101) else bool(w101 < 0.05), n101))
    f = {ng: float(d[d.n_grid == ng].resolved_C8.mean()) for ng in (51, 101, 201, 401) if (d.n_grid == ng).any()}
    rows.append(row("E47", "P10", "C = 8 kernels resolved: >= 90% at N_g = 101 and a lower fraction at 51", f.get(101, np.nan), None, len(d[d.n_grid == 101]),
                    bool(f.get(101, 0) >= 0.9 and f.get(51, 1) < f.get(101, 0)), "resolved fraction by N_g: " + ", ".join(f"{k}: {v:.2f}" for k, v in f.items())))
    return rows


def read_e47_time(d):
    b = d[(d.arm == "bp_grid") & (d.factor == "grid_size") & (d.batch == 512) & (d.n_sites == 32)]
    ms = (b.groupby("grid_size").wall_median.median() / 512 * 1000).sort_index()
    rows = []
    t51, t101 = float(ms.get(51, np.nan)), float(ms.get(101, np.nan))
    rows.append(row("E47", "P6", "BP time per sequence <= 0.07 ms at N_g = 51 and within 20% of 0.114 ms at 101", t51, None, len(ms),
                    bool(np.isfinite(t51) and np.isfinite(t101) and t51 <= 0.07 and 0.114 * 0.8 <= t101 <= 0.114 * 1.2), "ms: " + ", ".join(f"{int(g)}: {v:.3f}" for g, v in ms.items())))
    sub_ = ms[(ms.index >= 41) & (ms.index <= 401)]
    slope = float(np.polyfit(np.log(sub_.index.values.astype(float)), np.log(sub_.values), 1)[0]) if len(sub_) >= 3 else float("nan")
    rows.append(row("E47", "P7", "log-log slope of time vs N_g over 41-401 < 1.5", slope, None, len(sub_), None if np.isnan(slope) else bool(slope < 1.5)))
    return rows


def read_e47_oracle(d):
    rows = []
    for cell, g in d.groupby("cell"):
        w = g.groupby("grid_size").rel_error.max()
        rows.append(dict(experiment="E47", prediction="oracle", clause=f"{cell}: worst relative score error vs 1601", value=float(w.get(101, np.nan)), k=None, n=len(w),
                         verdict="DESCRIPTIVE", note=", ".join(f"{int(n)}: {v:.1e}" for n, v in w.items() if n in (21, 31, 41, 51, 61, 81, 101, 201, 401))))
    return rows


def read_e48(d, d36x):
    rows = []
    mdn = d[d.arm == "em_mdn"].set_index(["stride", "seed"]).sort_index()
    ref = d36x[(d36x.arm == "em_pair_quartic")].set_index(["stride", "seed"]).sort_index().reindex(mdn.index)
    if "test_bundle" in mdn and "test_bundle" in ref and (mdn.test_bundle.values != ref.test_bundle.values).any():
        raise ValueError("E48 and E36x were scored on different test bundles; the paired reading is not valid")
    ns = mdn.index.get_level_values("seed").nunique()
    m4, r4 = mdn.loc[4], ref.loc[4]
    k = int((m4.at_cap == 0).sum())
    rows.append(row("E48", "P1", f"stride 4: MDN selects a checkpoint below the last in >= {_need(ns, 12 / 16)}/{ns} seeds", np.nan, k, ns, bool(k >= _need(ns, 12 / 16)), f"largest selected {int(m4.checkpoint.max())}"))
    k = int((m4.risk < r4.risk).sum())
    rows.append(row("E48", "P2", f"stride 4: MDN below the site energy in >= {_need(ns, 14 / 16)}/{ns} seeds", float((m4.risk / r4.risk).mean()), k, ns, bool(k >= _need(ns, 14 / 16))))
    m1, r1 = mdn.loc[1], ref.loc[1]
    k = int((m1.risk > r1.risk).sum())
    rows.append(row("E48", "P3", f"stride 1: MDN above the site energy in {ns}/{ns} seeds", float((m1.risk / r1.risk).mean()), k, ns, bool(k == ns)))
    ratio = float(m1.risk.mean() / r1.risk.mean())
    rows.append(row("E48", "P4", "stride 1: seed-mean ratio MDN / site energy < 5", ratio, None, ns, bool(np.isfinite(ratio) and ratio < 5)))
    m2, r2 = mdn.loc[2], ref.loc[2]
    k = int((m2.risk < r2.risk).sum())
    rows.append(row("E48", "P5", f"stride 2: MDN below the site energy in >= {_need(ns, 10 / 16)}/{ns} seeds", float((m2.risk / r2.risk).mean()), k, ns, bool(k >= _need(ns, 10 / 16))))
    return rows


def read_e49(d, dB):
    rows = []
    n = 2048
    mdn = _risk_rows(d, "em_mdn", n).loc["LAP_B_L32"]
    R = {a: _risk_rows(dB, a, n).loc["LAP_B_L32"].reindex(mdn.index) for a in ("em_pair_quartic", "em_mixture", "em_loglinear")}
    if (R["em_pair_quartic"].test_bundle.values != mdn.test_bundle.values).any():
        raise ValueError("E49 and E43 were scored on different test bundles; the paired reading is not valid")
    ns = len(mdn)
    k = int((mdn.risk < R["em_pair_quartic"].risk).sum())
    rows.append(row("E49", "P1", f"M = 2048: MDN below the site energy in >= {_need(ns, 7 / 8)}/{ns} seeds", float((mdn.risk / R["em_pair_quartic"].risk).mean()), k, ns, bool(k >= _need(ns, 7 / 8))))
    k = int((mdn.risk < R["em_mixture"].risk).sum())
    rows.append(row("E49", "P2", f"M = 2048: MDN below the mixture in >= {_need(ns, 7 / 8)}/{ns} seeds", float((mdn.risk / R["em_mixture"].risk).mean()), k, ns, bool(k >= _need(ns, 7 / 8))))
    k = int((R["em_loglinear"].risk < mdn.risk).sum())
    rows.append(row("E49", "P3", f"M = 2048: log-linear below the MDN in >= {_need(ns, 6 / 8)}/{ns} seeds", float((R["em_loglinear"].risk / mdn.risk).mean()), k, ns, bool(k >= _need(ns, 6 / 8))))
    share = float((mdn.at_cap == 1).mean())
    rows.append(row("E49", "P4", "M = 2048: at most 30% of MDN fits at the last checkpoint", share, int((mdn.at_cap == 1).sum()), ns, bool(share <= 0.30)))
    return rows


# --------------------------------------------------------------------------- E45
def read_e45(d):
    rows = []
    cp = sorted(set(zip(d.cell, d.pattern)))
    cov = d.groupby(["cell", "pattern", "arm"]).coverage90.mean()                 # NaN rows (failed rebuilds) are skipped by mean
    wid = d.groupby(["cell", "pattern", "arm"]).width90.mean()
    n_failed = d[d.failed == 1].groupby(["cell", "pattern", "arm"]).size()
    err = {k: g.pivot_table(index="seed", columns="arm", values="rel_err", dropna=False) for k, g in d.groupby(["cell", "pattern"])}
    ns = d.seed.nunique()
    # P1
    ok, notes = True, []
    for c, p in cp:
        v = float(cov[(c, p, "oracle")]); ok &= 0.87 <= v <= 0.93; notes.append(f"{c}/{p}: {v:.3f}")
    rows.append(row("E45", "P1", "oracle 90% interval coverage in [0.87, 0.93] on every (cell, pattern)", np.nan, None, len(cp), bool(ok), "; ".join(notes)))
    # P2
    ok, notes = True, []
    for c, p in cp:
        v = float(cov[(c, p, "em_pair_quartic")]) if (c, p, "em_pair_quartic") in cov.index else np.nan
        e = err[(c, p)]
        k = int((e["em_pair_quartic"] <= 1.10 * e["oracle"]).sum())
        good = bool(abs(v - 0.90) <= 0.03 and k >= _need(ns, 14 / 16) and n_failed.get((c, p, "em_pair_quartic"), 0) == 0)
        ok &= good; notes.append(f"{c}/{p}: coverage {v:.3f}, error within 10% in {k}/{ns}")
    rows.append(row("E45", "P2", "site energy: coverage within 0.03 of 0.90 and error within 10% of the oracle's in >= 14/16 seeds, every (cell, pattern)", np.nan, None, len(cp), bool(ok), "; ".join(notes)))
    key = ("T0.3_h1_L32", "gap16")
    if key in err:
        v = float(cov[(*key, "linear")])
        rows.append(row("E45", "P3", "linear: |coverage - 0.90| > 0.03 on (T0.3_h1_L32, gap16)", v, None, ns, bool(abs(v - 0.90) > 0.03), f"oracle {float(cov[(*key, 'oracle')]):.3f}"))
        r = float(err[key]["linear"].mean() / err[key]["oracle"].mean())
        rows.append(row("E45", "P4", "linear error / oracle error in [0.9, 1.1] on (T0.3_h1_L32, gap16)", r, None, ns, bool(0.9 <= r <= 1.1)))
        wr = float(wid[(*key, "linear")] / wid[(*key, "oracle")])
        rows.append(row("E45", "P5", "linear 90% interval width / oracle width >= 1.1 on (T0.3_h1_L32, gap16)", wr, None, ns, bool(wr >= 1.1)))
    return rows


# --------------------------------------------------------------------------- E43
def _need(ns, frac):
    return max(1, math.ceil(frac * ns))


def read_e43(dA, d33, dB):
    rows = []
    if not dA.empty:
        loglin = {n: _risk_rows(dA, "em_loglinear", n) for n in (128, 512, 2048)}
        comp = {a: {n: _risk_rows(d33, a, n) for n in (128, 2048)} for a in ("em_tilted_spline", "em_mixture", "em_pair_quartic")}
        cells = sorted(set(dA.cell))
        for n in (128, 2048):
            ref = comp["em_pair_quartic"][n].reindex(loglin[n].index)
            if (ref.test_bundle.values != loglin[n].test_bundle.values).any():
                raise ValueError("E43 lab A and E33 were scored on different test bundles; the paired reading is not valid")
        ns = loglin[128].index.get_level_values("seed").nunique()
        # P1
        ok, notes = True, []
        for c in cells:
            m, o = loglin[128].loc[c].risk, comp["em_pair_quartic"][128].reindex(loglin[128].index).loc[c].risk
            k = int((m / o >= 2).sum()); ok &= k >= _need(ns, 14 / 16); notes.append(f"{c}: {k}/{len(m)}")
        rows.append(row("E43", "P1", f"lab A, M = 128: loglinear / site energy >= 2 in >= {_need(ns, 14 / 16)}/{ns} seeds on both cells", np.nan, None, len(cells), bool(ok), "; ".join(notes)))
        # P2
        ok, notes = True, []
        for c in cells:
            r = {n: float(loglin[n].loc[c].risk.mean() / comp["em_pair_quartic"][n].reindex(loglin[n].index).loc[c].risk.mean()) for n in (128, 2048)}
            ok &= bool(r[2048] < r[128] and r[2048] < 2.0); notes.append(f"{c}: {r[128]:.2f} -> {r[2048]:.2f}")
        rows.append(row("E43", "P2", "lab A: seed-mean loglinear / site energy smaller at 2048 than at 128, and < 2 at 2048", np.nan, None, len(cells), bool(ok), "; ".join(notes)))
        # P3, P4
        for pred, arm, n, sign, frac, label in (("P3", "em_mixture", 2048, -1, 14 / 16, "below the mixture"), ("P4", "em_tilted_spline", 128, +1, 12 / 16, "above the tilted spline")):
            ok, notes = True, []
            for c in cells:
                m = loglin[n].loc[c].risk
                o = comp[arm][n].reindex(loglin[n].index).loc[c].risk
                k = int((m < o).sum()) if sign < 0 else int((m > o).sum())
                ok &= k >= _need(ns, frac); notes.append(f"{c}: {k}/{len(m)}")
            rows.append(row("E43", pred, f"lab A, M = {n}: loglinear {label} in >= {_need(ns, frac)}/{ns} seeds on both cells", np.nan, None, len(cells), bool(ok), "; ".join(notes)))
    if not dB.empty:
        R = {a: {n: _risk_rows(dB, a, n) for n in (128, 2048)} for a in ("em_loglinear", "em_pair_quartic", "em_tilted_spline", "em_mixture")}
        ns = R["em_loglinear"][2048].index.get_level_values("seed").nunique()
        need = _need(ns, 7 / 8)
        ll, pq = R["em_loglinear"][2048].risk, R["em_pair_quartic"][2048].reindex(R["em_loglinear"][2048].index).risk
        k5, ratio = int((ll < pq).sum()), float(pq.mean() / ll.mean())
        rows.append(row("E43", "P5", f"lab B, M = 2048: loglinear below the site energy in >= {need}/{ns} seeds and mean ratio site energy / loglinear >= 1.25", ratio, k5, len(ll), bool(k5 >= need and ratio >= 1.25)))
        ok, notes = True, []
        for a in ("em_tilted_spline", "em_mixture"):
            o = R[a][2048].reindex(R["em_loglinear"][2048].index).risk
            k = int((ll < o).sum()); ok &= k >= need; notes.append(f"{a}: {k}/{len(ll)}")
        rows.append(row("E43", "P6", f"lab B, M = 2048: loglinear below the tilted spline and the mixture, each in >= {need}/{ns} seeds", np.nan, None, 2, bool(ok), "; ".join(notes)))
        r = {n: float(R["em_loglinear"][n].risk.mean() / R["em_pair_quartic"][n].reindex(R["em_loglinear"][n].index).risk.mean()) for n in (128, 2048)}
        rows.append(row("E43", "P7", "lab B: seed-mean loglinear / site energy above 1 at M = 128 and below 1 at M = 2048", r[2048], None, 2, bool(r[128] > 1 > r[2048]), f"M = 128: {r[128]:.2f}; M = 2048: {r[2048]:.2f}"))
    return rows


# --------------------------------------------------------------------------- E46
def read_e46(d46, d35):
    """d46: symmetry.csv rows (arms em_first_order_sym, em_first_order_2m); d35: E35x's frozen rows (b = 0)."""
    from make_replication_numbers import boot_ci
    rows = []
    for n in sorted(d46.n_chains.unique()):
        new = d46[d46.n_chains == n].pivot_table(index="seed", columns="arm", values="risk", dropna=False)
        old = d35[(d35.b == 0) & (d35.n_chains == n)].pivot_table(index="seed", columns="arm", values="risk")
        bund = d35[(d35.b == 0) & (d35.n_chains == n) & (d35.arm == "em_first_order")].set_index("seed").test_bundle
        mine = d46[(d46.n_chains == n) & (d46.arm == "em_first_order_sym")].set_index("seed").test_bundle
        if (bund.reindex(mine.index) != mine).any():
            raise ValueError("E46 and E35x were scored on different test bundles; the paired reading is not valid")
        p = new.join(old, how="inner")
        for pred, num, den, test in (("P1", "em_first_order_sym", "em_augmented", "contains0"),
                                     ("P2", "em_first_order_sym", "em_first_order", "below0"),
                                     ("P3", "em_first_order_sym", "em_first_order_2m", "below1.1")):
            x = np.log(p[num] / p[den])
            if x.isna().any():
                rows.append(row("E46", pred, f"M = {n}: mean log({num} / {den}); a failed fit counts against", np.nan, int(x.notna().sum()), len(x), False, "missing risks"))
                continue
            lo, hi = boot_ci(x.values)
            ok = {"contains0": lo <= 0 <= hi, "below0": hi < 0, "below1.1": hi < math.log(1.1)}[test]
            rows.append(row("E46", pred, f"M = {n}: mean log({num} / {den}), 95% interval {test}", float(x.mean()), int((x < 0).sum()), len(x), bool(ok),
                            f"CI [{lo:.3f}, {hi:.3f}]; geometric-mean ratio {math.exp(x.mean()):.3f}"))
    return rows


def load_e42_table(d33q, d42):
    sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "experiments")); sys.path.insert(0, str(ROOT / "src"))
    from make_field_potential_figs import _readings
    from markov_score.bp_grid import make_grid
    from frozen_config import FROZEN
    k = pd.concat([d for d in (d33q, d42) if not d.empty], ignore_index=True)
    k = k[(k.arm == "em_pair_quartic") & (k.selection == "risk") & k.cell.isin(TCELL)].copy()
    k["failed"] = k["failed"].fillna(0) if "failed" in k else 0
    grid, _ = make_grid(FROZEN.half_width, FROZEN.n_grid)
    tab, _ = _readings(k[k.failed == 0].assign(source="noisy"), grid)
    failed = {key: int(v) for key, v in k[k.failed == 1].groupby(["cell", "n_chains"]).size().items()}
    return tab, failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--e41", default="outputs/hpc_E41_s*/kernels.csv")
    ap.add_argument("--e42", default="outputs/hpc_E42_s*/kernels.csv")
    ap.add_argument("--e33k", default="outputs/hpc_E33k_s*/kernels.csv")
    ap.add_argument("--e33q", default="outputs/hpc_E33q_s*/kernels.csv")
    ap.add_argument("--e46", default="outputs/hpc_E46_s*/symmetry.csv")
    ap.add_argument("--e45", default="outputs/E45/inpainting.csv")
    ap.add_argument("--e44", default="outputs/hpc_E44_s*/quench.csv")
    ap.add_argument("--e47a", default="outputs/E47a/refinement.csv")
    ap.add_argument("--e47b", default="outputs/hpc_E47b_s*/kernels.csv")
    ap.add_argument("--e47t", default="outputs/hpc_E47t/scaling.csv")
    ap.add_argument("--e47c", default="outputs/E47c/refinement.csv")
    ap.add_argument("--e48", default="outputs/hpc_E48_s*/clean.csv")
    ap.add_argument("--e36x", default="outputs/hpc_E36x_s*/clean.csv")
    ap.add_argument("--e49", default="outputs/hpc_E49_s*/kernels.csv")
    ap.add_argument("--e43a", default="outputs/hpc_E43a_s*/kernels.csv")
    ap.add_argument("--e43b", default="outputs/hpc_E43b_s*/kernels.csv")
    ap.add_argument("--e35x", default="outputs/hpc_E35x_s*/augmented.csv")
    ap.add_argument("--out", default="reports/numbers")
    a = ap.parse_args()
    rows = []
    d41, d33k, d33q, d42 = _load(a.e41), _load(a.e33k), _load(a.e33q), _load(a.e42)
    d46, d35x = _load(a.e46), _load(a.e35x)
    d43a, d43b = _load(a.e43a), _load(a.e43b)
    d45 = _load(a.e45)
    d44 = _load(a.e44)
    d47b, d47t, d47a = _load(a.e47b), _load(a.e47t), _load(a.e47a)
    d47c = _load(a.e47c)
    d48, d36x = _load(a.e48), _load(a.e36x)
    d49 = _load(a.e49)
    if not d41.empty:
        rows += read_e41(d41, pd.concat([d33k, d33q], ignore_index=True))
    if not d42.empty:
        tab, failed = load_e42_table(d33q, d42)
        rows += read_e42(tab, failed)
    if not d46.empty:
        rows += read_e46(d46, d35x)
    if not d47a.empty:
        rows += read_e47_oracle(d47a)
    if not d47b.empty:
        rows += read_e47_learned(d47b, pd.concat([d33k, d33q], ignore_index=True))
    if not d47c.empty:
        rows += read_e47_headline(d47c)
    if not d47t.empty:
        rows += read_e47_time(d47t)
    if not d48.empty:
        rows += read_e48(d48, d36x)
    if not d49.empty:
        rows += read_e49(d49, d43b)
    if not d44.empty:
        rows += read_e44(d44)
    if not d45.empty:
        rows += read_e45(d45)
    if not d43a.empty or not d43b.empty:
        rows += read_e43(d43a, pd.concat([d33k, d33q], ignore_index=True), d43b)
    if not rows:
        print("nothing to read yet"); return 0
    R = pd.DataFrame(rows)
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / "extension_predictions.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 250, "display.max_colwidth", 90):
        print(R.to_string(index=False))
    print("wrote", out / "extension_predictions.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
