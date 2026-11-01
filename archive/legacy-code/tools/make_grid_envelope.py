#!/usr/bin/env python
"""Emit the quadrature-envelope macros from E11 (predeclared analysis, registry E11.json).

Usage: make_grid_envelope.py [--src outputs/E11] [--dest paper/sections]

For each candidate grid, the WORST case over families, correlations, seeds and levels of:
the drift of the posterior-mean field against the finest grid in the ladder, the belief
mass in the outermost 2% of the domain, and the column-mass defect of the discretised
kernel. Worst case, not mean: the point of the sweep is the envelope.

The drift is measured against the finest grid, so it is a convergence diagnostic and not
an error against the truth. The macros distinguish the operating grid AT THE FROZEN
CONFIGURATION (where the project's numbers live) from the operating grid's worst case
OVER THE SWEEP (which is a different and much larger number, and is the finding).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OP_N, OP_A = 401, 8.0
FROZEN_FAMILY, FROZEN_RHO = "laplace", 0.85


def _f(x: float) -> str:
    """Scientific notation LaTeX can set: 5.4\\times10^{-2}."""
    s = f"{x:.1e}"
    m, e = s.split("e")
    return f"{m}\\times 10^{{{int(e)}}}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "outputs" / "E11")
    ap.add_argument("--dest", type=Path, default=ROOT / "paper" / "sections")
    a = ap.parse_args()
    dest = a.dest if "dev" not in a.src.name else ROOT / "outputs" / "scratch"
    dest.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.src / "grid.csv")
    op = df[(df.n_grid == OP_N) & (df.half_width == OP_A)]
    if op.empty:
        raise SystemExit(f"E11: the operating grid ({OP_N}, {OP_A}) is not in the sweep")
    frozen = op[(op.family == FROZEN_FAMILY) & (op.rho == FROZEN_RHO)]
    if frozen.empty:
        raise SystemExit("E11: the frozen configuration is not in the sweep")

    m = {}
    # where the project's numbers actually live
    m["gridFrozenDrift"] = _f(frozen.drift_pooled.max())
    m["gridFrozenDriftWorstLevel"] = _f(frozen.drift_worst_level.max())
    m["gridFrozenBoundary"] = _f(frozen.boundary_belief_max.max())
    # the envelope over the whole sweep, at fixed half-width, as a function of n_grid
    fixedA = df[df.half_width == OP_A].groupby("n_grid").drift_pooled.max()
    m["gridFlatLo"] = _f(fixedA.iloc[0]); m["gridFlatHi"] = _f(fixedA.iloc[-1])
    m["gridFlatNlo"] = str(int(fixedA.index[0])); m["gridFlatNhi"] = str(int(fixedA.index[-1]))
    m["gridHalfWidth"] = f"{OP_A:g}"
    # what widening buys, at the two wider grids in the ladder
    for n, A, tag in ((401, 10.0, "widerA"), (801, 12.0, "widerB")):
        sub = df[(df.n_grid == n) & (df.half_width == A)]
        if not sub.empty:
            m[f"grid{tag[0].upper()+tag[1:]}Drift"] = _f(sub.drift_pooled.max())
            m[f"grid{tag[0].upper()+tag[1:]}N"] = str(n); m[f"grid{tag[0].upper()+tag[1:]}A"] = f"{A:g}"
    # the offending cell
    worst = op.loc[op.drift_pooled.idxmax()]
    m["gridWorstFamily"] = str(worst.family)
    m["gridWorstRho"] = f"{worst.rho:g}"
    m["gridWorstDrift"] = _f(worst.drift_pooled)
    m["gridWorstBoundary"] = f"{worst.boundary_belief_max:.2f}"
    # the column-mass defect, which grows with correlation and is renormalised away
    cm = df[df.half_width == OP_A].groupby("rho").col_mass_max.max()
    m["gridColMassLoRho"] = f"{cm.index.min():g}"; m["gridColMassHiRho"] = f"{cm.index.max():g}"
    m["gridColMassLo"] = _f(cm.iloc[0]); m["gridColMassHi"] = f"{cm.iloc[-1]:.2f}"
    m["gridColMassDriftHiRho"] = _f(df[(df.half_width == OP_A) & (df.rho == cm.index.max())].drift_pooled.max())
    m["gridFamilies"] = str(df.family.nunique()); m["gridRhos"] = str(df.rho.nunique())
    m["gridLevels"] = str(df.n_grid.nunique() and int(560 / max(1, df.n_grid.nunique())) and len(df))
    m["gridSeeds"] = str(df.seed.nunique())
    m["gridRefN"] = str(int(df.ref_n_grid.iloc[0])); m["gridRefA"] = f"{df.ref_half_width.iloc[0]:g}"
    m["gridOpN"] = str(OP_N); m["gridOpA"] = f"{OP_A:g}"
    del m["gridLevels"]
    # --- E11b: the same sweep at the longer chains the length study and E14 use. Reported as the
    # worst pooled drift on the operating grid per length (the envelope) and the frozen-configuration
    # drift per length, against the L = 32 values above; H11b is about their ratio.
    longer = ROOT / "outputs" / "E11b" / "grid.csv"
    if longer.exists() and "dev" not in a.src.name:
        dl = pd.read_csv(longer)
        opl = dl[(dl.n_grid == OP_N) & (dl.half_width == OP_A)]
        word = {64: "Sixtyfour", 128: "Onetwentyeight"}
        m["gridLenList"] = ", ".join(f"{int(L)}" for L in sorted(opl.n_sites.unique()))
        m["gridLenWorstDriftThirtytwo"] = _f(op.drift_pooled.max())
        m["gridLenFrozenDriftThirtytwo"] = _f(frozen.drift_pooled.max())
        ratios = []
        for L, sub in opl.groupby("n_sites"):
            tag = word[int(L)]
            m[f"gridLenWorstDrift{tag}"] = _f(sub.drift_pooled.max())
            fr = sub[(sub.family == FROZEN_FAMILY) & (sub.rho == FROZEN_RHO)]
            m[f"gridLenFrozenDrift{tag}"] = _f(fr.drift_pooled.max())
            m[f"gridLenFrozenBoundary{tag}"] = _f(fr.boundary_belief_max.max())
            ratios.append(sub.drift_pooled.max() / op.drift_pooled.max())
        m["gridLenWorstRatioMax"] = f"{max(ratios):.2f}"
        # the frozen configuration's drift ratio, which is the quantity H11b predicts to be O(1)
        m["gridLenFrozenRatioMax"] = f"{max(opl[(opl.family == FROZEN_FAMILY) & (opl.rho == FROZEN_RHO)].groupby('n_sites').drift_pooled.max() / frozen.drift_pooled.max()):.2f}"
        m["gridLenSeeds"] = str(dl.seed.nunique())

    (dest / "grid-envelope.tex").write_text(
        "%% GENERATED by tools/make_grid_envelope.py -- do not hand-edit.\n"
        + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()) + "\n")
    print(f"worst drift at ({OP_N}, {OP_A}) over the sweep: {worst.drift_pooled:.3e} "
          f"({worst.family}, rho={worst.rho:g}, outer mass {worst.boundary_belief_max:.2f})")
    print(f"at half-width {OP_A:g}, drift vs n_grid:\n{fixedA.to_string()}")
    print(f"wrote {dest/'grid-envelope.tex'} ({len(m)} macros)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
