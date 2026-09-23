#!/usr/bin/env python
"""Independent re-derivation of every headline number, from the frozen CSVs, with plain pandas.

Usage: audit_claims.py            # prints a side-by-side of recomputed value vs. the macro the
                                  # paper reads, flags any disagreement beyond rounding

The generators (tools/make_*_numbers.py) are the only path by which a number reaches the
manuscript. That is a strength for provenance and a weakness for auditing: a bug in a generator
is invisible to the manuscript check, which only asks whether the macros match what the generator
emits. This script computes the same headline quantities a second way -- short, explicit pandas
on the frozen CSVs, no shared helpers -- and compares them with the macros in paper/sections/.

It is deliberately dumb. Each block is a few lines that a reader can check by eye against the
claim it audits (docs/CLAIM_EVIDENCE_MATRIX.md row in the comment). If a block and a macro
disagree, one of them is wrong and the audit says which quantity, not which side.
"""

from __future__ import annotations

import re
import sys
from itertools import combinations_with_replacement
from math import factorial
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SEC = ROOT / "paper" / "sections"

_MACROS: dict[str, str] = {}
for f in SEC.glob("*-numbers.tex"):
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", f.read_text()):
        _MACROS[m.group(1)] = m.group(2)

FINDINGS: list[tuple[str, str, str, str]] = []   # (claim, quantity, recomputed, macro)


def check(claim: str, quantity: str, value, macro: str | None, tol: float = 0.0015, fmt: str = "{:.4g}"):
    """Compare a recomputed value with a macro; tolerance covers the macro's own rounding."""
    mval = _MACROS.get(macro) if macro else None
    ok = None
    if mval is not None:
        try:
            ok = abs(float(mval) - float(value)) <= max(tol, 0.5 * 10 ** -_decimals(mval)) + 1e-12
        except ValueError:
            ok = str(mval).strip() == str(value).strip()
    flag = "   " if ok is None else ("OK " if ok else "!! ")
    shown = fmt.format(value) if isinstance(value, (int, float, np.floating)) else str(value)
    print(f"{flag}{claim:5s} {quantity:58s} {shown:>12s}   macro {macro or '-'}={mval}")
    if ok is False:
        FINDINGS.append((claim, quantity, shown, f"{macro}={mval}"))


def _decimals(s: str) -> int:
    return len(s.split(".")[1]) if "." in s else 0


def pooled(df, arm_col="arm", arm=None, **sel):
    d = df
    for k, v in sel.items():
        d = d[d[k] == v]
    if arm is not None:
        d = d[d[arm_col] == arm]
    return d


# ----------------------------------------------------------------------------- E02a capacity (E2/E13-cap)
def audit_capacity():
    df = pd.read_csv(OUT / "E02a" / "capacity.csv")
    d = df[df.region == "all"]
    for M, tag in ((32, "A"), (2048, "D")):
        c1 = d[(d.arm == "em_bp") & (d.components == 1) & (d.n_chains == M)].groupby("seed").risk.mean()
        c8 = d[(d.arm == "em_bp") & (d.components == 8) & (d.n_chains == M)].groupby("seed").risk.mean()
        lm = d[(d.arm == "lmmse_true") & (d.n_chains == M)].groupby("seed").risk.mean()
        check("E02a", f"C=1 mean risk at M={M}", c1.mean(), f"capEmone{tag}")
        check("E02a", f"LMMSE floor at M={M}", lm.mean(), f"capLmmse{tag}")
        check("E02a", f"C=8 mean risk at M={M}", c8.mean(), f"capEmeight{tag}")
        diff = (c1 - c8).dropna()
        check("E02a", f"paired C1-C8 at M={M}", diff.mean(), f"capDiff{tag}")
        check("E02a", f"seeds favouring C=8 at M={M}", int((diff > 0).sum()), f"capWins{tag}", fmt="{}")
    check("E02a", "seeds", int(d.seed.nunique()), "capSeeds", fmt="{}")


# ----------------------------------------------------------------------------- E03 baselines (E14-E17)
def audit_baselines():
    c = pd.read_csv(OUT / "E03" / "confirm.csv"); c = c[c.region == "all"]
    em = pd.read_csv(OUT / "E02a" / "capacity.csv"); em = em[(em.region == "all") & (em.arm == "em_bp") & (em.components == 8)]
    for M, tag in ((32, "A"), (2048, "D")):
        best = c[c.n_chains == M].groupby("arch").risk.mean().sort_values()
        check("E03", f"best network family at M={M}", best.index[0], None)
        check("E03", f"best network mean risk at M={M}", best.iloc[0], f"base{best.index[0].capitalize()}{tag}")
        e = em[em.n_chains == M].risk.mean()
        check("E03", f"EM-BP C=8 mean risk at M={M} (from E02a)", e, f"capEmeight{tag}")
    w = c[c.arch == "window"].groupby("n_chains").risk.mean()
    check("E03", "window head at M=512", w.get(512), "baseWindowC")
    check("E03", "window head at M=2048", w.get(2048), "baseWindowD")
    # matched interface -- the headline is interface C: implicit score matching on one noised
    # view per sequence, no clean chain, checkpoint selected on the implicit objective (claim E16
    # as corrected on 18 Sep); the fixed-corruption DSM arm (interface B, clean target) is reported
    # separately as matchedFixed*
    m = pd.read_csv(OUT / "E03" / "matched.csv"); m = m[(m.region == "all")]
    for arm, prefix, rprefix in (("ism_one_view", "matchedBest", "matchedRatioBest"), ("dsm_one_corruption", "matchedFixed", "matchedFixedRatio")):
        best = m[(m.n_chains == 2048) & (m.arm == arm)].groupby("arch").risk.mean().sort_values()
        arch, val = best.index[0], best.iloc[0]
        check("E03m", f"best {arm} arch at M=2048", arch, None)
        check("E03m", f"its mean risk ({arm})", val, f"{prefix}D")
        j = m[(m.arm == arm) & (m.arch == arch) & (m.n_chains == 2048)].merge(
            em[em.n_chains == 2048][["seed", "test_bundle", "risk"]].rename(columns={"risk": "risk_em"}),
            on=["seed", "test_bundle"])
        check("E03m", f"paired (seed, bundle) rows joined ({arm})", int(len(j)), None, fmt="{}")
        ratio = (j.risk / j.risk_em)
        check("E03m", f"paired factor {arm} / EM-BP at M=2048", ratio.mean(), f"{rprefix}D", tol=0.06)
        rng = np.random.default_rng(0); r = ratio.to_numpy()
        b = np.array([r[rng.integers(0, len(r), len(r))].mean() for _ in range(20000)])
        check("E03m", "  percentile-bootstrap 2.5%", np.percentile(b, 2.5), f"{rprefix}Dlo", tol=0.15)
        check("E03m", "  percentile-bootstrap 97.5%", np.percentile(b, 97.5), f"{rprefix}Dhi", tol=0.15)
        check("E03m", f"  wins ({arm})", int((ratio > 1).sum()), f"{prefix.replace('Best', 'Wins').replace('Fixed', 'FixedWins')}D", fmt="{}")
    ism_tr = m[(m.arm == "ism_one_view") & (m.arch == "transformer") & (m.n_chains == 2048)].risk.mean()
    check("E03m", "ISM transformer at M=2048", ism_tr, "matchedIsmoneviewTransformerD")
    # length
    l = pd.read_csv(OUT / "E03" / "length.csv")
    tr = l[(l.arch == "transformer") & (l.n_chains == 2048)].groupby("L_test").risk.mean()
    check("E03L", "transformer at L=32", tr.get(32), "lenTransformerDLthreetwo")
    check("E03L", "transformer at L=256", tr.get(256), "lenTransformerDLtwofivesix")
    el = pd.read_csv(OUT / "E03" / "em_length.csv")
    eL = el[el.n_chains == 2048].groupby("L_test").risk.mean()
    check("E03L", "EM-BP at L=32", eL.get(32), "lenEmbpDLthreetwo")
    check("E03L", "EM-BP at L=256", eL.get(256), "lenEmbpDLtwofivesix")


# ----------------------------------------------------------------------------- E05 generation (E9)
def audit_generation():
    g = pd.read_csv(OUT / "E05" / "generation.csv")
    d = g[g.n_chains == 2048]
    for arm, tag in (("exact", "Exact"), ("em_bp", "Em"), ("lmmse", "Lmmse"), ("isotropic", "Iso")):
        check("E05", f"log-lik gap, {arm}, M=2048", d[d.arm == arm].loglik_gap.mean(), f"genLoglikgapD{tag}")
    check("E05", "generated excess kurtosis, EM-BP, M=2048", d[d.arm == "em_bp"].kurt_gen.mean(), "genKurtgenDEm")
    check("E05", "reference excess kurtosis, M=2048", d.kurt_ref.mean(), "genKurtrefD", tol=0.006)
    ex = d[d.arm == "exact"].set_index("seed").loglik_gap; eb = d[d.arm == "em_bp"].set_index("seed").loglik_gap
    check("E05", "paired deficit EM-BP minus exact (nats)", (eb - ex).mean(), None)


# ----------------------------------------------------------------------------- E08 misspecification (E18)
def audit_misspec():
    df = pd.read_csv(OUT / "E08" / "misspecification.csv"); d = df[df.region == "all"]
    piv = d.pivot_table(index=["mechanism", "eps", "n_chains", "seed"], columns="arm", values="risk").reset_index()
    for mech, tag in (("ar2", "Artwo"), ("longrange", "Longrange"), ("sitevar", "Sitevar"), ("global", "Global")):
        g = piv[(piv.mechanism == mech) & (piv.n_chains == 2048)]
        eps = sorted(g.eps.unique())
        lr = [np.log(g[g.eps == e].window / g[g.eps == e].em_bp).mean() for e in eps]
        cross = [i for i in range(len(lr) - 1) if lr[i] > 0 >= lr[i + 1]]
        if cross:
            i = cross[0]; x = eps[i] + (eps[i + 1] - eps[i]) * lr[i] / (lr[i] - lr[i + 1])
            check("E08", f"{mech}: break-even bracket lo (M=2048)", eps[i], f"misBreakLo{tag}D", fmt="{:g}")
            check("E08", f"{mech}: break-even bracket hi (M=2048)", eps[i + 1], f"misBreakHi{tag}D", fmt="{:g}")
            check("E08", f"{mech}: interpolated crossing (M=2048)", x, f"misBreak{tag}D", tol=0.006)
        else:
            check("E08", f"{mech}: crossing (M=2048)", "not reached", f"misBreak{tag}D")
    g = piv[(piv.mechanism == "longrange") & (piv.n_chains == 2048) & (piv.eps == 0.4)]
    r = np.log(g.window / g.em_bp); se = r.std(ddof=1) / np.sqrt(len(r))
    check("E08", "longrange gamma=0.4 log-ratio (M=2048)", r.mean(), None)
    check("E08", "  its 95% CI lower", r.mean() - 1.96 * se, None); check("E08", "  its 95% CI upper", r.mean() + 1.96 * se, None)
    e0 = piv[(piv.mechanism == "ar2") & (piv.eps == 0.0)].groupby("n_chains").em_bp.mean()
    check("E08", "EM-BP risk at zero strength, M=512", e0.get(512), "misArtwoEmbpEzeropzeroC")
    check("E08", "EM-BP risk at zero strength, M=2048", e0.get(2048), "misArtwoEmbpEzeropzeroD")


# ----------------------------------------------------------------------------- E13 regimes (E20)
def audit_regimes():
    # E13x, the sixteen-seed run, is what paper/sections/regime-numbers.tex is generated from;
    # the frozen eight-seed E13 stays as history and is not what the paper reports. Recomputing
    # from E13 while the macros come from E13x compares two different datasets and disagrees by
    # the seed count (it did: weakest gain -0.0442 against -0.047).
    src = OUT / "E13x" if (OUT / "E13x" / "regimes.csv").exists() else OUT / "E13"
    df = pd.read_csv(src / "regimes.csv")
    piv = df.pivot_table(index=["rho", "n_sites", "seed"], columns=["arm", "components"], values="risk")
    piv.columns = ["_".join(str(c) for c in col) for col in piv.columns]; piv = piv.reset_index()
    wins = 0; ident = 0.0; gains = []
    for (r, L), g in piv.groupby(["rho", "n_sites"]):
        d = g.em_bp_8 - g.em_bp_1; se = d.std(ddof=1) / np.sqrt(len(d))
        wins += int(d.mean() + 1.96 * se < 0); gains.append(d.mean())
        ident = max(ident, float((g.em_bp_1 - g.lmmse_true_0).abs().max()))
    tag = src.name
    check(tag, "cells whose 95% normal CI excludes zero (of 15)", wins, None, fmt="{}")
    check(tag, "strongest paired gain", min(gains), "regGainStrongest")
    check(tag, "weakest paired gain", max(gains), "regGainWeakest")
    check(tag, "Prop. 1 check: max |C=1 - LMMSE| over cells", ident, "regIdentMax")


# ----------------------------------------------------------------------------- E11 grid envelope (E19)
def audit_grid():
    df = pd.read_csv(OUT / "E11" / "grid.csv")
    op = df[(df.n_grid == 401) & (df.half_width == 8.0)]
    fr = op[(op.family == "laplace") & (op.rho == 0.85)]
    check("E11", "frozen-config pooled drift at (401, 8)", fr.drift_pooled.max(), None, fmt="{:.2e}")
    fixed = df[df.half_width == 8.0].groupby("n_grid").drift_pooled.max()
    check("E11", "worst drift at A=8, N=201", fixed.get(201), None, fmt="{:.4e}")
    check("E11", "worst drift at A=8, N=1601", fixed.get(1601), None, fmt="{:.4e}")
    w = op.loc[op.drift_pooled.idxmax()]
    check("E11", "binding cell (family, rho)", f"{w.family}, rho={w.rho:g}", None)
    check("E11", "  its outer-2% belief mass", w.boundary_belief_max, None, fmt="{:.2f}")


# ----------------------------------------------------------------------------- E06 learning (E11/E12 rows)
def audit_learning():
    df = pd.read_csv(OUT / "E06" / "levels.csv")
    cap = df.groupby("t_obs").at_cap.mean()
    check("E06", "at-cap rate at t=1.4", cap.get(1.4), "learnAtCapTonepfour", tol=0.006)
    check("E06", "at-cap rate at t=2.0", cap.get(2.0), "learnAtCapTtwo", tol=0.006)
    check("E06", "max at-cap rate over t<=1.0", cap[cap.index <= 1.0].max(), "learnAtCapLowMax", tol=0.006)
    rm = df[df.t_obs > 0].groupby("t_obs").apply(lambda g: np.sqrt((g.rho_err ** 2).mean()) if "rho_err" in g else np.sqrt(((g.fit_rho - g.true_rho) ** 2).mean()), include_groups=False)
    check("E06", "RMSE of rho-hat at t=1.4", rm.get(1.4), "learnRmseRhoTonepfour")


# ----------------------------------------------------------------------------- E07 phase (E10 row)
def audit_phase():
    df = pd.read_csv(OUT / "E07" / "ladder.csv")
    # rows flagged pooled=True carry the schedule-pooled value in rel_lmmse (one per rep)
    lap = df[(df.family == "laplace") & (df.pooled.astype(bool))]
    p = lap.groupby("rho").rel_lmmse.mean()
    for r, tag in ((0.0, "zero"), (0.85, "zeropeightfive"), (0.95, "zeropninefive")):
        check("E07", f"Laplace pooled LMMSE floor at rho={r:g}", p.get(r), f"phaseLmmseLaplaceRho{tag}")
    # and the same value recomputed from the per-level rows: sqrt(sum e^2 E / sum E) over t
    per = df[(df.family == "laplace") & (~df.pooled.astype(bool)) & (df.rho == 0.85)]
    check("E07", "  rho=0.85 floor re-pooled from per-level rows", float(np.sqrt((per.rel_lmmse**2 * per.ref_energy).sum() / per.ref_energy.sum())), "phaseLmmseLaplaceRhozeropeightfive", tol=0.003)


# ----------------------------------------------------------------------------- E12 sample complexity (E21)
def audit_sample_complexity():
    df = pd.read_csv(OUT / "E12" / "ladder.csv")
    sizes = np.array(sorted(df.n_chains.unique())); seeds = np.array(sorted(df.seed.unique())); ts = np.array(sorted(df.t_obs.unique()))
    A = np.full((len(seeds), len(ts), len(sizes)), np.nan)
    for i, sd in enumerate(seeds):
        for j, t in enumerate(ts):
            for k, M in enumerate(sizes):
                v = df[(df.seed == sd) & (df.t_obs == t) & (df.n_chains == M)].rho_err.to_numpy()
                if len(v): A[i, j, k] = float(v[0]) ** 2
    def cross(y, thr=0.015):
        for i in range(len(y) - 1):
            if y[i] > thr >= y[i + 1]:
                f = (np.log(y[i]) - np.log(thr)) / (np.log(y[i]) - np.log(y[i + 1]))
                return float(np.exp(np.log(sizes[i]) + f * (np.log(sizes[i + 1]) - np.log(sizes[i]))))
        return np.nan
    def ols(y):
        ok = np.isfinite(y); return float(np.polyfit(ts[ok], np.log(y[ok]), 1)[0]) if ok.sum() >= 2 else np.nan
    n = len(seeds); vals, wts = [], []
    for ms in combinations_with_replacement(range(n), n):
        w = factorial(n) / np.prod([factorial(v) for v in Counter(ms).values()]) / n ** n
        r = np.sqrt(A[np.array(ms)].mean(axis=0)); v = ols(np.array([cross(r[j]) for j in range(len(ts))]))
        if np.isfinite(v): vals.append(v); wts.append(w)
    v, w = np.array(vals), np.array(wts); o = np.argsort(v); v, w = v[o], w[o]; cw = np.cumsum(w) / w.sum()
    check("E12", "covariance slope, point (OLS over 3 levels)", ols(np.array([cross(np.sqrt(A.mean(axis=0))[j]) for j in range(len(ts))])), "scSlopeCov")
    check("E12", "  exact-bootstrap 2.5%", float(v[np.searchsorted(cw, 0.025)]), "scSlopeCovLo")
    check("E12", "  exact-bootstrap 97.5%", float(v[np.searchsorted(cw, 0.975)]), "scSlopeCovHi")


def main() -> int:
    print(f"{'':3s}{'run':5s} {'quantity':58s} {'recomputed':>12s}   macro")
    for fn in (audit_capacity, audit_baselines, audit_generation, audit_misspec, audit_regimes,
               audit_grid, audit_learning, audit_phase, audit_sample_complexity):
        print(f"--- {fn.__name__} ---")
        try:
            fn()
        except Exception as e:  # a block that cannot run is itself a finding
            print(f"!! {fn.__name__} failed: {type(e).__name__}: {e}")
            FINDINGS.append((fn.__name__, "block failed", str(e), ""))
    print()
    if FINDINGS:
        print(f"{len(FINDINGS)} DISAGREEMENT(S) between an independent recomputation and a macro:")
        for c, q, v, m in FINDINGS:
            print(f"  {c}: {q}: recomputed {v} vs {m}")
        return 1
    print("every headline macro agrees with its independent recomputation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
