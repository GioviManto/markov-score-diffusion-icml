#!/usr/bin/env python
"""Emit the capacity table and macros from E02 (predeclared analysis, registry E02.json).

Usage: make_capacity_numbers.py [--src outputs/E02a] [--dest paper/sections]
For a development pass (--src outputs/E02a_dev) the destination is forced to
outputs/scratch so a dev number can never reach the manuscript.

Per size M: mean +/- s.e. over seeds of the all-site pooled risk for the oracles
(isotropic, LMMSE with the true covariance) and for EM-BP at every capacity; the
screened window head of the archival confirmatory run (outputs/frozen/exp_31_confirm_merged)
joined on (seed, n_chains) ONLY where its recorded test-bundle hash equals ours (identical
data), otherwise omitted and said so; paired per-seed differences C=8 minus C=1 with a
20000-resample percentile bootstrap over seeds; per-seed ratios LMMSE/EM-BP(C).
Refuses if any confirmatory seed is missing a capacity, or if a cell fails the
two-grid-cell resolution gate (reported, not silently dropped).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIZE_TAG = {32: "A", 128: "B", 512: "C", 2048: "D"}
CAP_TAG = {0: "zero", 1: "one", 2: "two", 4: "four", 8: "eight", 16: "sixteen"}


def _mac(s: str) -> str:
    return "".join({"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}.get(c, c) for c in s)


def boot_ci(x, n=20000, seed=20260915):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/E02a")
    ap.add_argument("--dest", default="paper/sections")
    args = ap.parse_args()
    src = ROOT / args.src
    dev = "dev" in src.name or "quick" in src.name or "scratch" in src.parts
    dest = ROOT / ("outputs/scratch/sections_dev" if dev else args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src / "capacity.csv")
    allr = df[df.region == "all"].copy()
    seeds = sorted(allr.seed.unique())
    sizes = sorted(allr.n_chains.unique())
    caps = sorted(allr[allr.arm == "em_bp"].components.unique())
    # completeness
    for n in sizes:
        for C in caps:
            got = allr[(allr.arm == "em_bp") & (allr.n_chains == n) & (allr.components == C)].seed.nunique()
            if got != len(seeds):
                print(f"REFUSING: size {n} capacity {C} has {got}/{len(seeds)} seeds"); return 1
    unres = allr[(allr.arm == "em_bp") & (~allr.resolved.astype(bool))]
    print(f"seeds {seeds}; sizes {sizes}; capacities {caps}; unresolved cells {len(unres)}")

    # archival window head on identical bundles
    win = None
    conf = ROOT / "outputs/frozen/exp_31_confirm_merged/confirm.csv"
    if conf.exists():
        w = pd.read_csv(conf)
        w = w[(w.method == "window") & (w.region == "all")][["seed", "n_chains", "test_bundle", "risk", "checkpoint"]]
        ours = allr[allr.arm == "lmmse_true"][["seed", "n_chains", "test_bundle"]].drop_duplicates()
        j = ours.merge(w, on=["seed", "n_chains"], suffixes=("", "_w"))
        same = j[j.test_bundle == j.test_bundle_w]
        print(f"window head: {len(same)}/{len(j)} (seed,size) cells share the archival test bundle hash")
        win = same if len(same) == len(j) and len(j) > 0 else None

    macros, rows = {}, []
    for n in sizes:
        tag = SIZE_TAG.get(n, _mac(str(n)))
        sub = allr[allr.n_chains == n]
        rec = {"n": n}
        for arm in ("isotropic", "lmmse_true"):
            r = sub[sub.arm == arm].set_index("seed").risk
            rec[arm] = (r.mean(), r.std(ddof=1) / np.sqrt(len(r)))
            macros[f"cap{'Iso' if arm == 'isotropic' else 'Lmmse'}{tag}"] = f"{r.mean():.3f}"
        em = {}
        for C in caps:
            r = sub[(sub.arm == "em_bp") & (sub.components == C)].set_index("seed").risk
            em[C] = r
            rec[f"C{C}"] = (r.mean(), r.std(ddof=1) / np.sqrt(len(r)))
            macros[f"capEm{CAP_TAG.get(C, _mac(str(C)))}{tag}"] = f"{r.mean():.3f}"
            macros[f"capEm{CAP_TAG.get(C, _mac(str(C)))}{tag}se"] = f"{r.std(ddof=1) / np.sqrt(len(r)):.3f}"
            kurt = sub[(sub.arm == "em_bp") & (sub.components == C)].fit_kurt
            macros[f"capKurt{CAP_TAG.get(C, _mac(str(C)))}{tag}"] = f"{kurt.mean():.2f}"
        if 1 in em and 8 in em:
            d = (em[1] - em[8]).dropna()
            lo, hi = boot_ci(d.values)
            rec["d81"] = (d.mean(), lo, hi, int((d > 0).sum()), len(d))
            macros[f"capDiff{tag}"] = f"{d.mean():.3f}"; macros[f"capDiff{tag}lo"] = f"{lo:.3f}"; macros[f"capDiff{tag}hi"] = f"{hi:.3f}"
            macros[f"capWins{tag}"] = str(int((d > 0).sum()))
            ratio = (sub[sub.arm == "lmmse_true"].set_index("seed").risk / em[8]).dropna()
            macros[f"capFloorRatio{tag}"] = f"{ratio.mean():.1f}"
        if win is not None:
            wr = win[win.n_chains == n].set_index("seed").risk
            rec["window"] = (wr.mean(), wr.std(ddof=1) / np.sqrt(len(wr)))
            macros[f"capWindow{tag}"] = f"{wr.mean():.3f}"
        rows.append(rec)
    macros["capSeeds"] = str(len(seeds)); macros["capUnresolved"] = str(len(unres))
    macros["capSizes"] = ", ".join(str(n) for n in sizes)

    # table
    caps_cols = " & ".join(f"$\\ncomp={C}$" for C in caps)
    tex = ["%% GENERATED by tools/make_capacity_numbers.py from " + str(src.relative_to(ROOT)) + " -- do not hand-edit.",
           "\\begin{table}[t]", "\\centering", "\\small",
           "\\caption{Schedule-pooled relative score error on all sites, mean over seeds (s.e.\\ in the text macros). Oracles and EM--BP are scored on identical test bundles; the window head is the archival screened baseline on the same bundles where its hash matches.}",
           "\\label{tab:capacity}",
           "\\begin{tabular}{r" + "c" * (2 + len(caps) + (1 if win is not None else 0)) + "}", "\\toprule",
           "$\\nseq$ & isotropic & LMMSE floor & " + caps_cols + (" & window head" if win is not None else "") + " \\\\", "\\midrule"]
    for rec in rows:
        cells = [f"{rec['isotropic'][0]:.3f}", f"{rec['lmmse_true'][0]:.3f}"] + [f"{rec[f'C{C}'][0]:.3f}" for C in caps]
        if win is not None:
            cells.append(f"{rec['window'][0]:.3f}")
        tex.append(f"{rec['n']} & " + " & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (dest / "tab-capacity.tex").write_text("\n".join(tex) + "\n")
    lines = ["%% GENERATED by tools/make_capacity_numbers.py -- do not hand-edit."] + [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    (dest / "capacity-numbers.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {dest.relative_to(ROOT)}/tab-capacity.tex and capacity-numbers.tex ({len(macros)} macros)")
    for rec in rows:
        s = f"  M={rec['n']:5d}  iso {rec['isotropic'][0]:.3f}  lmmse {rec['lmmse_true'][0]:.3f}  " + "  ".join(f"C{C} {rec[f'C{C}'][0]:.3f}±{rec[f'C{C}'][1]:.3f}" for C in caps)
        if "d81" in rec:
            s += f"  | C1-C8 = {rec['d81'][0]:+.3f} [{rec['d81'][1]:+.3f},{rec['d81'][2]:+.3f}] wins {rec['d81'][3]}/{rec['d81'][4]}"
        if "window" in rec:
            s += f"  | window {rec['window'][0]:.3f}"
        print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
