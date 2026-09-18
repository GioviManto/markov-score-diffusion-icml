#!/usr/bin/env python
"""Emit paper/sections/audit-numbers.tex from the audit experiments E15-E21 (18-19 September 2026).

One macro per number the body cites; a source that is missing is skipped (its macros are simply not
defined and the manuscript check reports them). Every macro is prefixed `aud`.
Usage: make_audit_numbers.py [--out paper/sections/audit-numbers.tex]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _mac(s):
    return "".join({"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}.get(c, c) for c in str(s)).replace(".", "p").replace("-", "m")


def sci(v):
    if v == 0:
        return "0"
    e = int(np.floor(np.log10(abs(v)))); m = v / 10**e
    return f"{m:.1f}\\times 10^{{{e}}}"


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="paper/sections/audit-numbers.tex"); args = ap.parse_args()
    M = {}
    # ---- E15 locality floor
    p = ROOT / "outputs/E15/locality.csv"
    if p.exists():
        d = pd.read_csv(p)
        for _, r in d[d.t < 0].iterrows():
            tag = _mac(f"{r.rho:g}")
            for c in d.columns:
                if c.startswith("floor_r") or c.startswith("gauss_r"):
                    M[f"audLoc{_mac(c.replace('_', '')).capitalize()}Rho{tag}"] = f"{r[c]:.3f}"
            M[f"audLocLmmseRho{tag}"] = f"{r.lmmse:.3f}"
        # level at which the radius-4 floor peaks at rho = 0.85
        s = d[(d.rho == 0.85) & (d.t > 0)]
        if len(s):
            j = s.floor_r4.idxmax(); M["audLocPeakLevel"] = f"{s.loc[j, 't']:.2f}"; M["audLocPeakFloorRfour"] = f"{s.loc[j, 'floor_r4']:.3f}"
    # ---- E16 Edgeworth oracle
    p = ROOT / "outputs/E16/edgeworth.csv"
    if p.exists():
        d = pd.read_csv(p)
        for fam, g in d[d.t > 0].groupby("family"):
            for _, r in g.iterrows():
                M[f"audEdgeClosed{fam.capitalize()}T{_mac(f'{r.t:.2f}')}"] = f"{r.gap_closed_log:.3f}"
                M[f"audEdgeLevelT{_mac(f'{r.t:.2f}')}"] = f"{r.t:.2f}"
            ok = g[g.gap_closed_log >= 0.9]
            M[f"audEdgeNinetyFrom{fam.capitalize()}"] = f"{ok.t.min():.2f}" if len(ok) else "never"
            neg = g[g.gap_closed_log < 0]
            M[f"audEdgeDivergesBelow{fam.capitalize()}"] = f"{neg.t.max():.2f}" if len(neg) else "never"
    # ---- E17 misspecification bias
    p = ROOT / "outputs/E17/bias.csv"
    if p.exists():
        d = pd.read_csv(p)
        for _, r in d[d.eps >= 0].iterrows():
            tag = f"{_mac(r.mechanism.capitalize())}E{_mac(f'{r.eps:g}')}"
            M[f"audBias{tag}"] = f"{r.bias_kl:.3f}"; M[f"audBiasLagone{tag}"] = f"{r.bias_lag1:.3f}"
            M[f"audBiasPredD{tag}"] = f"{r.pred_2048:.3f}"; M[f"audBiasPredC{tag}"] = f"{r.pred_512:.3f}"
            if np.isfinite(r.em_2048):
                M[f"audBiasEmD{tag}"] = f"{r.em_2048:.3f}"; M[f"audBiasEmC{tag}"] = f"{r.em_512:.3f}"
        for _, r in d[d.eps < 0].iterrows():
            M[f"audBiasOrder{_mac(r.mechanism.capitalize())}"] = f"{r.local_slope:.2f}"; M[f"audBiasCoef{_mac(r.mechanism.capitalize())}"] = f"{r.coefficient:.2f}"
        ok = d[(d.eps > 0) & np.isfinite(d.em_2048)]
        M["audBiasMaxAbsDevD"] = f"{(ok.bias_kl - ok.em_2048).abs().max():.3f}"
        M["audBiasMaxAbsDevLagoneD"] = f"{(ok.bias_lag1 - ok.em_2048).abs().max():.3f}"
    # ---- E18 wide-domain closure
    p = ROOT / "outputs/E18/closure_wide.csv"
    if p.exists():
        d = pd.read_csv(p)
        for (fam, ng, A), g in d.groupby(["family", "n_grid", "half_width"]):
            g = g.sort_values("t"); tag = _mac(f"{fam.capitalize()}N{int(ng)}")
            for _, r in g.iterrows():
                if np.isfinite(r.ratio_to_prediction):
                    M[f"audWideRatio{tag}T{_mac(f'{r.t:g}')}"] = f"{r.ratio_to_prediction:.2f}"
            sl = np.diff(np.log(g.sq_gap_per_chain.values)) / np.diff(g.t.values)
            for a, s in zip(g.t.values[:-1], sl):
                M[f"audWideSlope{tag}T{_mac(f'{a:g}')}"] = f"{s:.2f}"
    p = ROOT / "outputs/E18/constant.csv"
    if p.exists():
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            tag = _mac(f"{r.family.capitalize()}T{r.t:g}")
            M[f"audConstRatio{tag}"] = f"{r.ratio:.2f}"; M[f"audConstRatioLo{tag}"] = f"{r.ratio_lo:.2f}"; M[f"audConstRatioHi{tag}"] = f"{r.ratio_hi:.2f}"
            M[f"audConstRelSe{tag}"] = f"{100*r.rel_se:.0f}"; M[f"audConstMaxOverMedian{tag}"] = f"{r.max_gap/r.median_gap:.0f}"
        M["audConstChains"] = str(int(d.n_chains.iloc[0]))
    p = ROOT / "outputs/E18/scalar_gap.csv"
    if p.exists():
        d = pd.read_csv(p)
        for fam, g in d.groupby("family"):
            g = g.sort_values("t"); sl = np.diff(np.log(g.sq_gap.values)) / np.diff(g.t.values)
            for a, s in zip(g.t.values[:-1], sl):
                M[f"audScalarSlope{_mac(fam.capitalize())}T{_mac(f'{a:g}')}"] = f"{s:.2f}"
            for _, r in g.iterrows():
                if np.isfinite(r.ratio_to_prediction):
                    v = r.ratio_to_prediction
                    M[f"audScalarRatio{_mac(fam.capitalize())}T{_mac(f'{r.t:g}')}"] = f"{v:.2f}" if v < 100 else f"{v:.0f}"
    # ---- E19 moment estimators
    p = ROOT / "outputs/E19/moments.csv"
    if p.exists():
        d = pd.read_csv(p); g = d.groupby("t_obs")
        rr = g.apply(lambda x: np.sqrt(((x.fit_rho - x.true_rho) ** 2).mean()), include_groups=False)
        rk = g.apply(lambda x: np.sqrt(((x.fit_kurt - x.true_kurt) ** 2).mean()), include_groups=False)
        bk = g.apply(lambda x: (x.fit_kurt - x.true_kurt).mean(), include_groups=False)
        for t in rr.index:
            tag = _mac(f"{t:g}")
            M[f"audMomRmseRhoT{tag}"] = f"{rr[t]:.3f}"; M[f"audMomRmseKurtT{tag}"] = f"{rk[t]:.2f}" if rk[t] < 100 else f"{rk[t]:.0f}"; M[f"audMomBiasKurtT{tag}"] = f"{bk[t]:.2f}"
        tt = np.array(rr.index); hi = tt >= 1.0
        M["audMomSlopeRho"] = f"{np.polyfit(tt[hi], np.log(rr.values[hi]), 1)[0]:.2f}"; M["audMomSlopeKurt"] = f"{np.polyfit(tt[hi], np.log(rk.values[hi]), 1)[0]:.2f}"
        e06 = ROOT / "outputs/E06/levels.csv"
        if e06.exists():
            e = pd.read_csv(e06); ge = e[e.t_obs > 0].groupby("t_obs")
            er = ge.apply(lambda x: np.sqrt(((x.fit_rho - x.true_rho) ** 2).mean()), include_groups=False)
            ek = ge.apply(lambda x: np.sqrt(((x.fit_kurt - x.true_kurt) ** 2).mean()), include_groups=False)
            common = [t for t in rr.index if t in er.index and t <= 0.7]
            M["audMomOverEmRhoMin"] = f"{min(rr[t]/er[t] for t in common):.0f}"; M["audMomOverEmRhoMax"] = f"{max(rr[t]/er[t] for t in common):.0f}"
            M["audMomOverEmKurtMin"] = f"{min(rk[t]/ek[t] for t in common):.1f}"; M["audMomOverEmKurtMax"] = f"{max(rk[t]/ek[t] for t in common):.1f}"
    # ---- E20 selection
    p = ROOT / "outputs/E20/selection.csv"
    if p.exists():
        d = pd.read_csv(p)
        for Mn, g in d.groupby("n_chains"):
            tag = {32: "A", 128: "B", 512: "C", 2048: "D"}.get(int(Mn), str(int(Mn)))
            M[f"audSelOracle{tag}"] = f"{g.test_oracle.mean():.3f}"; M[f"audSelEvidence{tag}"] = f"{g.test_evidence.mean():.3f}"; M[f"audSelLast{tag}"] = f"{g.test_last.mean():.3f}"
            M[f"audSelDiff{tag}"] = f"{(g.test_evidence - g.test_oracle).mean():+.4f}"; M[f"audSelSame{tag}"] = str(int((g.ckpt_oracle == g.ckpt_evidence).sum())); M[f"audSelN{tag}"] = str(len(g))
    # ---- E21 information curve + scalar constant
    p = ROOT / "outputs/E21/information.csv"
    if p.exists():
        d = pd.read_csv(p).sort_values("t")
        for _, r in d.iterrows():
            tag = _mac(f"{r.t:g}")
            for Mn in (256, 512, 2048, 8192, 16384):
                if f"crb_kurt_M{Mn}" in d.columns:
                    M[f"audCrbKurtT{tag}M{_mac(Mn)}"] = f"{r[f'crb_kurt_M{Mn}']:.2f}"
            if np.isfinite(r.get("slope_logMstar_beta", np.nan)):
                M[f"audSlopeBetaT{tag}"] = f"{r.slope_logMstar_beta:.1f}"; M[f"audSlopeRhoT{tag}"] = f"{r.slope_logMstar_rho:.1f}"
        win = d[(d.t >= 0.6) & (d.t <= 1.0)]
        M["audSlopeBetaWindowLo"] = f"{win.slope_logMstar_beta.min():.1f}"; M["audSlopeBetaWindowHi"] = f"{win.slope_logMstar_beta.max():.1f}"
        M["audSlopeRhoWindowLo"] = f"{win.slope_logMstar_rho.min():.1f}"; M["audSlopeRhoWindowHi"] = f"{win.slope_logMstar_rho.max():.1f}"
    p = ROOT / "outputs/E21/scalar_constant.csv"
    if p.exists():
        d = pd.read_csv(p)
        for _, r in d.iterrows():
            M[f"audScalarConst{_mac(r.coordinate.replace('=', '').replace(' ', ''))}T{_mac(f'{r.t:g}')}"] = f"{r.ratio:.4f}"
        b = d[d.coordinate.str.startswith("beta")]
        M["audScalarConstWorst"] = f"{(b[b.t >= 1.5].ratio - 1).abs().max()*100:.1f}"
        M["audScalarBetaList"] = ", ".join(sorted({c.replace("beta=", "") for c in b.coordinate}, key=float))
        M["audScalarFromLevel"] = "1.5"
    # ---- E06 re-read: collapse fraction and cluster dispersion per level (from the frozen file)
    p = ROOT / "outputs/E06/levels.csv"
    if p.exists():
        e = pd.read_csv(p)
        for t, x in e[e.t_obs > 0].groupby("t_obs"):
            tag = _mac(f"{t:g}"); good = x[x.fit_kurt >= 1.0]
            M[f"audCollapsedT{tag}"] = str(int((x.fit_kurt < 1.0).sum())); M[f"audCollapsedNT{tag}"] = str(len(x))
            M[f"audSdKurtT{tag}"] = f"{x.fit_kurt.std(ddof=1):.2f}"; M[f"audSdKurtGoodT{tag}"] = f"{good.fit_kurt.std(ddof=1):.2f}" if len(good) > 1 else "--"
            M[f"audBiasKurtT{tag}"] = f"{(x.fit_kurt - 3).mean():+.2f}"; M[f"audSdRhoT{tag}"] = f"{x.fit_rho.std(ddof=1):.4f}"
    # ---- E11 re-read: exceedance counts for the flagged Student-t cell, regenerated from the tags
    try:
        sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
        from markov_score.priors import StudentTAR1
        from markov_score.utils import rng_for
        from scipy import stats
        counts, mx = [], 0.0
        for seed in range(4):
            rng = rng_for("e11-chains", "student", 0.0, 32, seed)
            A = np.stack([StudentTAR1(0.0).sample(rng, 32) for _ in range(256)])
            counts.append(int((np.abs(A) > 8).sum())); mx = max(mx, float(np.abs(A).max()))
        sc = np.sqrt(3 / 5)
        M["audExceedCounts"] = ", ".join(str(c) for c in counts); M["audExceedMax"] = f"{mx:.1f}"
        M["audExceedExpected"] = f"{8192*2*stats.t.sf(8/sc, 5):.1f}"
        M["audExceedExpectedTwelve"] = f"{8192*2*stats.t.sf(12/sc, 5):.2f}"; M["audExceedExpectedSixteen"] = f"{8192*2*stats.t.sf(16/sc, 5):.2f}"
    except Exception as e:  # pragma: no cover
        print("exceedance macros skipped:", e)
    out = ROOT / args.out
    out.write_text("%% GENERATED by tools/make_audit_numbers.py from outputs/E15-E21 and E06 -- do not hand-edit.\n" + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(M.items())) + "\n")
    print(f"wrote {out} ({len(M)} macros)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
