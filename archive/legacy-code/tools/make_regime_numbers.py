#!/usr/bin/env python
"""Emit the regime-sweep table and macros from E13 (predeclared analysis, registry E13.json).

Usage: make_regime_numbers.py [--src outputs/E13] [--dest paper/sections]

Per (rho, L): the mean pooled relative score error of each arm over seeds, the paired
difference C=8 minus C=1 with a percentile bootstrap over seeds, and the apparatus check
|EM-BP(C=1) - LMMSE(true covariance)|, which Proposition 1 says is zero and which is
therefore a test of the whole pipeline at every point of the sweep rather than only at the
frozen configuration.

The trend in rho is reported ALONGSIDE the marginal's excess kurtosis 3(1-rho^2)/(1+rho^2),
as a direction. No exponent is fitted from it: the risk pools a twelve-level schedule whose
informative window moves with rho, so a log-log slope of the risk against kurtosis is not
the exponent of Theorem B and is not reported as one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RHO_TAG = {0.3: "three", 0.5: "five", 0.7: "seven", 0.85: "eightfive", 0.95: "ninefive"}
LEN_TAG = {16: "sixteen", 32: "thirtytwo", 64: "sixtyfour"}
B = 20_000


def marginal_kurtosis(rho: float, innovation_excess_kurtosis: float = 3.0) -> float:
    """Standardised excess kurtosis of the stationary marginal of the AR(1) chain.

    Cumulants add along X_t = sum_k rho^k eps_{t-k}, so kappa_4(X) = kappa_4(eps)/(1-rho^4);
    with unit marginal variance and innovation variance q = 1-rho^2 this is
    k_eps (1-rho^2)^2/(1-rho^4) = k_eps (1-rho^2)/(1+rho^2).
    """
    return innovation_excess_kurtosis * (1 - rho**2) / (1 + rho**2)


def _boot(d: np.ndarray, rng) -> tuple[float, float]:
    idx = rng.integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "outputs" / "E13")
    ap.add_argument("--dest", type=Path, default=ROOT / "paper" / "sections")
    ap.add_argument("--compare", type=Path, default=None,
                    help="an earlier run of the same design to check this one reproduces "
                         "(default: outputs/E13 when --src is a versioned re-run of it)")
    a = ap.parse_args()
    dest = a.dest if "dev" not in a.src.name else ROOT / "outputs" / "scratch"
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260918)

    df = pd.read_csv(a.src / "regimes.csv")
    piv = df.pivot_table(index=["rho", "n_sites", "seed"], columns=["arm", "components"], values="risk")
    piv.columns = ["_".join(str(c) for c in col) for col in piv.columns]
    piv = piv.reset_index()
    need = {"em_bp_1", "em_bp_8", "lmmse_true_0", "isotropic_0"}
    missing = need - set(piv.columns)
    if missing:
        raise SystemExit(f"E13: missing arms {sorted(missing)}")
    if piv[sorted(need)].isna().any().any():
        raise SystemExit("E13: a (rho, L, seed) cell is missing an arm; refusing to report a partial sweep")

    tstar = df.groupby(["rho", "n_sites"]).t_star.first()
    macros, rows = {}, []
    for (rho, L), g in piv.groupby(["rho", "n_sites"]):
        d = (g.em_bp_8 - g.em_bp_1).to_numpy()
        lo, hi = _boot(d, rng)
        ident = float(np.abs(g.em_bp_1 - g.lmmse_true_0).max())
        rows.append(dict(rho=rho, L=L, t_star=float(tstar[(rho, L)]),
                         kurt=marginal_kurtosis(rho),
                         em1=g.em_bp_1.mean(), em8=g.em_bp_8.mean(),
                         lmmse=g.lmmse_true_0.mean(), iso=g.isotropic_0.mean(),
                         gain=d.mean(), lo=lo, hi=hi, ident=ident, n_seeds=len(g)))
        tag = f"{RHO_TAG[rho]}{LEN_TAG[L]}"
        macros[f"regGain{tag}"] = f"{d.mean():.3f}"
        macros[f"regGainLo{tag}"] = f"{lo:.3f}"
        macros[f"regGainHi{tag}"] = f"{hi:.3f}"
        macros[f"regEmOne{tag}"] = f"{g.em_bp_1.mean():.3f}"
        macros[f"regEmEight{tag}"] = f"{g.em_bp_8.mean():.3f}"

    t = pd.DataFrame(rows).sort_values(["rho", "L"])
    if (t.hi >= 0).any():
        bad = t[t.hi >= 0][["rho", "L", "gain", "lo", "hi"]]
        print(f"NOTE: {len(bad)} cell(s) do not exclude zero:\n{bad.to_string(index=False)}")
    # The directory this fragment was generated from, so a checker can assert it is recomputing
    # from the same run. Without it audit_claims silently fell back to E13 while the fragment came
    # from E13x and compared two datasets (21 Sep).
    macros["regSource"] = a.src.name
    macros["regCells"] = str(len(t))
    macros["regSeeds"] = str(int(t.n_seeds.min()))
    macros["regGainStrongest"] = f"{t.gain.min():.3f}"
    macros["regGainWeakest"] = f"{t.gain.max():.3f}"
    macros["regAllExcludeZero"] = "all" if (t.hi < 0).all() else f"{int((t.hi < 0).sum())} of {len(t)}"
    macros["regIdentMax"] = f"{t.ident.max():.4f}"          # the Proposition 1 check, worst cell
    macros["regKurtHi"] = f"{marginal_kurtosis(t.rho.min()):.2f}"
    macros["regKurtLo"] = f"{marginal_kurtosis(t.rho.max()):.2f}"
    macros["regRhoLo"] = f"{t.rho.min():g}"
    macros["regRhoHi"] = f"{t.rho.max():g}"
    macros["regLenLo"] = f"{int(t.L.min())}"
    macros["regLenHi"] = f"{int(t.L.max())}"
    # the fitted innovation kurtosis, which the estimator should recover at 3 everywhere
    em8 = df[(df.arm == "em_bp") & (df.components == 8)]
    k = em8.groupby("rho").fit_kurt.agg(["mean", "std"])
    macros["regFitKurtLo"] = f"{k['mean'].min():.2f}"
    macros["regFitKurtHi"] = f"{k['mean'].max():.2f}"
    macros["regFitKurtSdLo"] = f"{k['std'].min():.2f}"
    macros["regFitKurtSdHi"] = f"{k['std'].max():.2f}"
    em1 = df[(df.arm == "em_bp") & (df.components == 1)]
    macros["regFitKurtOneMax"] = f"{float(np.abs(em1.fit_kurt).max()):.3f}"

    # Cross-run reproduction. A versioned re-run recomputes the earlier run's seeds in a new
    # directory, from a different deployment and often a different node. Every shared row should
    # agree exactly -- the generators are seeded from (tag, seed, L, rho) alone -- so the check is
    # a statement about the pipeline's determinism, not about the science, and it is emitted as
    # macros rather than asserted in prose.
    cmp_dir = a.compare
    if cmp_dir is None and a.src.name.startswith("E13") and a.src.name != "E13":
        cand = ROOT / "outputs" / "E13"
        cmp_dir = cand if (cand / "regimes.csv").exists() else None
    if cmp_dir is not None and (cmp_dir / "regimes.csv").exists():
        old_df = pd.read_csv(cmp_dir / "regimes.csv")
        on = ["rho", "n_sites", "seed", "arm", "components"]
        both = old_df.merge(df, on=on, suffixes=("_a", "_b"))
        dr = (both.risk_a - both.risk_b).abs()
        macros["regReproSource"] = cmp_dir.name
        macros["regReproRows"] = str(len(both))
        macros["regReproMaxDiff"] = f"{dr.max():.3g}" if dr.max() > 0 else "0"
        macros["regReproExact"] = "every" if (dr == 0).all() else f"{int((dr == 0).sum())} of {len(dr)}"
        if "fit_kurt" in old_df.columns:
            dk = (both.fit_kurt_a - both.fit_kurt_b).abs().dropna()
            macros["regReproMaxDiffKurt"] = f"{dk.max():.3g}" if len(dk) and dk.max() > 0 else "0"
        print(f"reproduction against {cmp_dir.name}: {len(both)} shared rows, "
              f"max |risk difference| {dr.max():.3g}, exact matches {int((dr == 0).sum())}/{len(dr)}")

    (dest / "regime-numbers.tex").write_text(
        "%% GENERATED by tools/make_regime_numbers.py -- do not hand-edit.\n"
        + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")

    # the table: capacity contrast down the correlations, one block per chain length
    lines = [r"%% GENERATED by tools/make_regime_numbers.py -- do not hand-edit.",
             r"\begin{tabular}{llrrrrl}", r"\toprule",
             r"$\corr$ & $L$ & $\tstar$ & $C=1$ & $C=8$ & \textsc{lmmse} & $C{=}8-C{=}1$ \\",
             r"\midrule"]
    for _, r in t.iterrows():
        lines.append(f"{r.rho:g} & {int(r.L)} & {r.t_star:.2f} & {r.em1:.3f} & {r.em8:.3f} & "
                     f"{r.lmmse:.3f} & ${r.gain:+.3f}$ [{r.lo:+.3f}, {r.hi:+.3f}] \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (dest / "regime-table.tex").write_text("\n".join(lines) + "\n")

    print(t.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nProposition 1 check (|EM-BP C=1 - LMMSE|), worst cell: {t.ident.max():.5f}")
    print(f"wrote {dest/'regime-numbers.tex'} ({len(macros)} macros) and {dest/'regime-table.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
