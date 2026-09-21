#!/usr/bin/env python
"""Emit the regime-baselines table and macros from E14 (predeclared analysis, registry E14.json).

Usage: make_regime_baselines.py [--src outputs/E14] [--dest paper/sections]

E14 re-runs the information-matched comparison across correlations and chain lengths. Per
(rho, L) this emits, for the eight confirmatory seeds:

  the mean pooled relative score error of every arm -- LMMSE of the true covariance, the
  isotropic map, EM-BP under the registered selection rule (held-out BP risk, which uses the
  true kernel) and under E20's oracle-free rule (held-out one-view log-evidence), and the two
  network arms per architecture: implicit score matching on one noised view per sequence
  (interface C, the matched arm) and denoising score matching with a clean target but one
  fixed corruption (interface B, reported separately, never as "matched");

  the paired per-seed factor best-network / EM-BP with a percentile bootstrap over seeds, the
  seed-wise win count and the ratio of means -- the same rule, resamples and seed as
  tools/make_baseline_numbers.py, so E14's factor at (0.85, 32) and E03's are one convention;

  the divergence count. A network row carries its whole held-out ISM curve; a fit whose
  curve never improves on its first checkpoint is a fit that diverged, and it is reported as
  divergence, not as the network's error. Factors are given with such cells included and
  with them excluded, so the prose can say which it quotes.

The trend in rho is reported beside the marginal's excess kurtosis 3(1-rho^2)/(1+rho^2) as a
direction. No exponent is fitted from it.

Refuses a partial sweep: every (rho, L, seed) must carry every arm, on one test bundle.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RHO_TAG = {0.5: "five", 0.85: "eightfive", 0.95: "ninefive"}
LEN_TAG = {32: "thirtytwo", 128: "onetwentyeight"}
ARCH_NAME = {"window": "window head", "transformer": "transformer"}
NET_ARMS = {"ism_one_view": "Ism", "dsm_one_corruption": "Fixed"}
EM_ARMS = {"em_bp": "", "em_bp_evidence": "Ev"}
B, SEED = 20_000, 20260915          # as make_baseline_numbers.boot_ci


def marginal_kurtosis(rho: float, k_eps: float = 3.0) -> float:
    return k_eps * (1 - rho**2) / (1 + rho**2)


def boot_ci(x):
    rng = np.random.default_rng(SEED); x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), size=(B, len(x)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def diverged(curve: str) -> bool:
    """A held-out ISM curve 'step:value;...' that never improves on its first checkpoint."""
    if not isinstance(curve, str) or not curve:
        return False
    vals = [float(p.split(":")[1]) for p in curve.split(";") if ":" in p]
    return len(vals) > 1 and min(vals) >= vals[0] and vals[-1] > vals[0]


def _tag(rho, L):
    return f"{RHO_TAG[rho]}{LEN_TAG[L]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "outputs" / "E14")
    ap.add_argument("--dest", type=Path, default=ROOT / "paper" / "sections")
    a = ap.parse_args()
    dest = a.dest if "dev" not in a.src.name and "smoke" not in a.src.name else ROOT / "outputs" / "scratch"
    dest.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.src / "regimes.csv")
    df["arch"] = df["arch"].fillna("")
    key = ["rho", "n_sites", "seed", "test_bundle"]

    # --- completeness: one row per arm per cell, one bundle per cell
    cells = df.groupby(["rho", "n_sites", "seed"])
    if (cells.test_bundle.nunique() > 1).any():
        raise SystemExit("E14: a cell scores arms on more than one test bundle; refusing")
    counts = cells.apply(lambda g: g.groupby(["arm", "arch"]).size().max(), include_groups=False)
    if (counts > 1).any():
        raise SystemExit("E14: duplicate (arm, arch) rows in a cell; refusing")
    archs = sorted(a for a in df.arch.unique() if a)
    expected = 2 + len(EM_ARMS) + len(NET_ARMS) * len(archs)
    short = cells.size()[cells.size() != expected]
    if len(short):
        raise SystemExit(f"E14: {len(short)} cell(s) missing an arm (expected {expected} rows):\n{short}")

    macros, rows = {}, []
    for (rho, L), g in df.groupby(["rho", "n_sites"]):
        tag = _tag(rho, L)
        rec = dict(rho=rho, L=L, t_star=float(g.t_star.iloc[0]), kurt=marginal_kurtosis(rho), n_seeds=g.seed.nunique())
        means = g.groupby(["arm", "arch"]).risk.mean()
        rec["lmmse"] = means[("lmmse_true", "")]; rec["iso"] = means[("isotropic", "")]
        for arm, sfx in EM_ARMS.items():
            rec[arm] = means[(arm, "")]
            macros[f"rbEm{sfx}{tag}"] = f"{rec[arm]:.3f}"
        macros[f"rbLmmse{tag}"] = f"{rec['lmmse']:.3f}"; macros[f"rbIso{tag}"] = f"{rec['iso']:.3f}"
        macros[f"rbTstar{tag}"] = f"{rec['t_star']:.2f}"; macros[f"rbKurt{tag}"] = f"{rec['kurt']:.2f}"

        em = g[g.arm == "em_bp"][key + ["risk"]].rename(columns={"risk": "risk_em"})
        emv = g[g.arm == "em_bp_evidence"][key + ["risk"]].rename(columns={"risk": "risk_emv"})
        for arm, pfx in NET_ARMS.items():
            sub = g[g.arm == arm]
            per = sub.groupby("arch").risk.mean().sort_values()
            best = per.index[0]
            rec[f"{arm}_arch"] = best; rec[f"{arm}_best"] = per.iloc[0]
            macros[f"rb{pfx}Best{tag}"] = f"{per.iloc[0]:.3f}"
            macros[f"rb{pfx}BestArch{tag}"] = ARCH_NAME.get(best, best)
            for arch in archs:
                macros[f"rb{pfx}{arch.capitalize()}{tag}"] = f"{per.get(arch, np.nan):.3f}"
            # divergence, per architecture, from the held-out ISM curve (ISM arm only carries it)
            sb = sub[sub.arch == best].copy()
            sb["diverged"] = sb.val_curve.map(diverged) if arm == "ism_one_view" else False
            n_div = int(sb.diverged.sum())
            rec[f"{arm}_diverged"] = n_div
            macros[f"rb{pfx}Diverged{tag}"] = str(n_div)
            j = sb.merge(em, on=key).merge(emv, on=key)
            if len(j) != rec["n_seeds"]:
                raise SystemExit(f"E14: pairing lost seeds at rho={rho} L={L} arm={arm}")
            for emcol, esfx in (("risk_em", ""), ("risk_emv", "Ev")):
                for subset, ssfx in ((j, ""), (j[~j.diverged], "Conv")):
                    if len(subset) == 0:
                        continue
                    ratio = subset.risk / subset[emcol]
                    lo, hi = boot_ci(ratio.values)
                    m = f"rb{pfx}Ratio{esfx}{ssfx}{tag}"
                    macros[m] = f"{ratio.mean():.1f}"; macros[m + "lo"] = f"{lo:.1f}"; macros[m + "hi"] = f"{hi:.1f}"
                    macros[f"rb{pfx}Wins{esfx}{ssfx}{tag}"] = str(int((ratio > 1).sum()))
                    macros[f"rb{pfx}N{esfx}{ssfx}{tag}"] = str(len(subset))
                    if emcol == "risk_em" and ssfx == "":
                        rec[f"{arm}_ratio"] = ratio.mean(); rec[f"{arm}_lo"] = lo; rec[f"{arm}_hi"] = hi
                        rec[f"{arm}_wins"] = int((ratio > 1).sum())
        rows.append(rec)

    t = pd.DataFrame(rows).sort_values(["rho", "L"])
    macros["rbCells"] = str(len(t)); macros["rbSeeds"] = str(int(t.n_seeds.min()))
    macros["rbRhoLo"] = f"{t.rho.min():g}"; macros["rbRhoHi"] = f"{t.rho.max():g}"
    macros["rbLenLo"] = str(int(t.L.min())); macros["rbLenHi"] = str(int(t.L.max()))
    macros["rbIsmRatioMin"] = f"{t.ism_one_view_ratio.min():.1f}"; macros["rbIsmRatioMax"] = f"{t.ism_one_view_ratio.max():.1f}"
    macros["rbIsmAllAboveOne"] = "all" if (t.ism_one_view_lo > 1).all() else f"{int((t.ism_one_view_lo > 1).sum())} of {len(t)}"
    macros["rbIsmDivergedTotal"] = str(int(t.ism_one_view_diverged.sum()))
    # the same cell E03 measured, so the two experiments can be read against each other
    fp = t[(t.rho == 0.85) & (t.L == 32)]
    if len(fp):
        macros["rbFrozenIsmRatio"] = f"{float(fp.ism_one_view_ratio.iloc[0]):.1f}"
        macros["rbFrozenEm"] = f"{float(fp.em_bp.iloc[0]):.3f}"

    (dest / "regime-baselines-numbers.tex").write_text(
        "%% GENERATED by tools/make_regime_baselines.py -- do not hand-edit.\n"
        + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")

    lines = [r"%% GENERATED by tools/make_regime_baselines.py -- do not hand-edit.",
             r"\begin{tabular}{llrrrrrl}", r"\toprule",
             r"$\corr$ & $L$ & \textsc{lmmse} & EM--BP & EM--BP (ev.) & best net (C) & best net (B) & factor C / EM--BP \\",
             r"\midrule"]
    for _, r in t.iterrows():
        lines.append(f"{r.rho:g} & {int(r.L)} & {r.lmmse:.3f} & {r.em_bp:.3f} & {r.em_bp_evidence:.3f} & "
                     f"{r.ism_one_view_best:.3f} & {r.dsm_one_corruption_best:.3f} & "
                     f"${r.ism_one_view_ratio:.1f}$ [{r.ism_one_view_lo:.1f}, {r.ism_one_view_hi:.1f}]"
                     + (f"$^{{\\dagger {int(r.ism_one_view_diverged)}}}$" if r.ism_one_view_diverged else "") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (dest / "regime-baselines-table.tex").write_text("\n".join(lines) + "\n")

    cols = ["rho", "L", "t_star", "kurt", "lmmse", "iso", "em_bp", "em_bp_evidence",
            "ism_one_view_arch", "ism_one_view_best", "ism_one_view_ratio", "ism_one_view_lo", "ism_one_view_hi",
            "ism_one_view_wins", "ism_one_view_diverged", "dsm_one_corruption_best", "dsm_one_corruption_ratio"]
    print(t[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if t.ism_one_view_diverged.sum():
        print(f"\nNOTE: {int(t.ism_one_view_diverged.sum())} diverged ISM fit(s); *Conv* macros exclude them.")
    print(f"wrote {dest/'regime-baselines-numbers.tex'} ({len(macros)} macros) and {dest/'regime-baselines-table.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
