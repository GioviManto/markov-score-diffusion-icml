#!/usr/bin/env python
"""E40's predeclared reading (registry E40.json).

    python tools/make_fft_numbers.py                                       # outputs/E40
    python tools/make_fft_numbers.py --src outputs/scratch/e40_quick --out outputs/scratch/e40_read

P1  (401, 8): FFT vs dense <= 1e-3 at every level; error(1601) <= error(401) / 10 at every level.
P2  N_g = 401: speed-up = dense / FFT <= 1.3.
P3  speed-up non-decreasing in N_g (strictly: each step up) and > 1 at the largest N_g.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def read(d):
    rows = []
    def row(p, clause, value, ok, note=""):
        rows.append(dict(experiment="E40", prediction=p, clause=clause, value=value,
                         verdict="UNRESOLVED" if ok is None else ("CONFIRMED" if ok else "FALSIFIED"), note=note))
    acc = d[(d.part == "accuracy") & (d.method == "fft")]
    sizes = sorted(acc.n_grid.unique())
    op = 401 if 401 in sizes else sizes[len(sizes) // 2]
    e_op = acc[acc.n_grid == op].set_index("t").err_vs_dense
    row("P1", f"({op}, 8): FFT vs dense <= 1e-3 at every level (worst)", float(e_op.max()), bool((e_op <= 1e-3).all()),
        "per level: " + ", ".join(f"{t:g}:{v:.1e}" for t, v in e_op.items()))
    hi = 1601 if 1601 in sizes else sizes[-1]
    if hi != op:
        e_hi = acc[acc.n_grid == hi].set_index("t").err_vs_dense
        ratio = (e_op / e_hi)
        row("P1", f"error({op}) / error({hi}) >= 10 at every level (smallest)", float(ratio.min()), bool((ratio >= 10).all()))
    tm = d[d.part == "timing"].pivot_table(index="n_grid", columns="method", values="wall")
    sp = (tm["dense"] / tm["fft"]).sort_index()
    if op in sp.index:
        row("P2", f"N_g = {op}: speed-up dense/FFT <= 1.3", float(sp.loc[op]), bool(sp.loc[op] <= 1.3),
            f"dense {tm.loc[op, 'dense']:.3f}s, FFT {tm.loc[op, 'fft']:.3f}s")
    mono = bool(np.all(np.diff(sp.values) > 0))
    row("P3", "speed-up increases at every refinement step", float(sp.values[-1]), mono, ", ".join(f"{n}:{v:.2f}" for n, v in sp.items()))
    row("P3", f"speed-up > 1 at N_g = {int(sp.index[-1])}", float(sp.values[-1]), bool(sp.values[-1] > 1))
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default="outputs/E40")
    ap.add_argument("--out", default="reports/numbers")
    a = ap.parse_args()
    d = pd.read_csv(ROOT / a.src / "fft.csv")
    R = read(d)
    with pd.option_context("display.width", 230, "display.max_colwidth", 110):
        print(R.to_string(index=False))
    o = ROOT / a.out; o.mkdir(parents=True, exist_ok=True)
    R.to_csv(o / "fft_predictions.csv", index=False, float_format="%.4g")
    return 0


if __name__ == "__main__":
    sys.exit(main())
