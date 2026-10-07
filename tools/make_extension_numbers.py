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
    ap.add_argument("--out", default="reports/numbers")
    a = ap.parse_args()
    rows = []
    d41, d33k, d33q, d42 = _load(a.e41), _load(a.e33k), _load(a.e33q), _load(a.e42)
    if not d41.empty:
        rows += read_e41(d41, pd.concat([d33k, d33q], ignore_index=True))
    if not d42.empty:
        tab, failed = load_e42_table(d33q, d42)
        rows += read_e42(tab, failed)
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
