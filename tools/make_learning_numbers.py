#!/usr/bin/env python
"""Emit paper/sections/learning-numbers.tex and paper/figures/fig_learning.pdf from E06.

Predeclared (registry E06.json): per observation level, the RMSE over seeds of the fitted
rho about 0.85 and of the fitted innovation excess kurtosis about 3.0 (with the bias
reported separately); OLS log-slope of each RMSE against t over the levels with
t >= 1.0; the ratio of the two; the clean-data arm (t = 0) as the floor; the mean
transition Hellinger distance and the twelve-level test risk per level.
Usage: make_learning_numbers.py [--src outputs/E06]; a dev source writes to scratch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
import figstyle as fs  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score import theory  # noqa: E402


def _mac(s):
    return "".join({"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}.get(c, c) for c in s).replace(".", "p")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/E06")
    args = ap.parse_args()
    src = ROOT / args.src
    dev = "dev" in src.name or "quick" in src.name or "scratch" in src.parts
    sec = ROOT / ("outputs/scratch/sections_dev" if dev else "paper/sections")
    figdir = ROOT / ("outputs/scratch/figures_dev" if dev else "paper/figures")
    sec.mkdir(parents=True, exist_ok=True); figdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src / "levels.csv")
    g = df.groupby("t_obs")
    hcol = next(c for c in ("hellinger_weighted_mean", "hellinger_hellinger_weighted_mean", "hellinger_mean", "hellinger_hellinger_mean") if c in df.columns)
    summ = pd.DataFrame({
        "rmse_rho": g.apply(lambda d: np.sqrt(((d.fit_rho - d.true_rho) ** 2).mean()), include_groups=False),
        "bias_rho": g.apply(lambda d: (d.fit_rho - d.true_rho).mean(), include_groups=False),
        "rmse_kurt": g.apply(lambda d: np.sqrt(((d.fit_kurt - d.true_kurt) ** 2).mean()), include_groups=False),
        "bias_kurt": g.apply(lambda d: (d.fit_kurt - d.true_kurt).mean(), include_groups=False),
        "hellinger": g[hcol].mean(),
        "risk": g.test_risk_schedule.mean(),
        "n_seeds": g.seed.nunique(),
    }).reset_index()
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    noisy = summ[summ.t_obs > 0]
    macros = {}
    rho = float(df.true_rho.iloc[0]); L = int(FROZEN.n_sites); M = int(df.n_chains.iloc[0])
    t_star = float(theory.crossover_time(rho, L=L))
    macros["learnTstar"] = f"{t_star:.2f}"; macros["learnRho"] = f"{rho:g}"
    # --- the information reference: per-sequence efficient information of the frozen E01 shape
    # table at beta = 1 (the Laplace member of the generalised-Gaussian family, same rho), read at
    # the E06 levels; the Cramer-Rao bound for rho is 1/sqrt(M I_rho) and for the excess kurtosis
    # |d kappa4 / d beta| / sqrt(M I_beta) (chain rule at beta = 1). It is the bound for an unbiased
    # estimator in that one-parameter family, quoted as a reference for the mixture estimator.
    crlb = None
    e01 = ROOT / "outputs/E01/shape_information.csv"
    if e01.exists() and abs(rho - float(FROZEN.rho)) < 1e-12:
        from scipy.special import gamma as G
        sh = pd.read_csv(e01); sh = sh[sh.beta == 1.0].groupby("t")[["eff_rho", "eff_beta"]].mean()
        K = lambda b: G(5 / b) * G(1 / b) / G(3 / b) ** 2 - 3
        dK = abs((K(1.0005) - K(0.9995)) / 0.001)
        tt = np.array(sh.index)
        crlb = pd.DataFrame({"t_obs": tt, "crlb_rho": 1 / np.sqrt(M * sh.eff_rho.values), "crlb_kurt": dK / np.sqrt(M * sh.eff_beta.values)})
        at = lambda col, t: float(np.exp(np.interp(t, tt, np.log(crlb[col].values))))
        ratios_rho, ratios_k = [], []
        for _, r in noisy.iterrows():
            rr, rk = r.rmse_rho / at("crlb_rho", r.t_obs), r.rmse_kurt / at("crlb_kurt", r.t_obs)
            tag = _mac(f"{r.t_obs:g}")
            macros[f"learnCrlbRhoT{tag}"] = f"{at('crlb_rho', r.t_obs):.4f}"; macros[f"learnCrlbKurtT{tag}"] = f"{at('crlb_kurt', r.t_obs):.3f}"
            macros[f"learnEffRhoT{tag}"] = f"{rr:.2f}"; macros[f"learnEffKurtT{tag}"] = f"{rk:.2f}"
            if r.rmse_kurt < 2.0:
                ratios_k.append(rk)
            if r.rmse_rho < 0.1:
                ratios_rho.append(rr)
        macros["learnEffRhoMean"] = f"{np.mean(ratios_rho):.2f}"; macros["learnEffKurtMean"] = f"{np.mean(ratios_k):.2f}"
        macros["learnEffRhoMax"] = f"{np.max(ratios_rho):.2f}"; macros["learnEffKurtMax"] = f"{np.max(ratios_k):.2f}"
        # the level at which the shape bound crosses the full range of the coordinate (3)
        cross = tt[np.argmax(crlb.crlb_kurt.values >= 3.0)] if (crlb.crlb_kurt.values >= 3.0).any() else np.nan
        macros["learnKurtBoundCross"] = f"{cross:.2f}"
        print(f"efficiency vs CRLB: rho {np.mean(ratios_rho):.2f} (max {np.max(ratios_rho):.2f}), kurt {np.mean(ratios_k):.2f} (max {np.max(ratios_k):.2f}); shape bound reaches 3 at t = {cross:.2f}")
    sat = noisy[noisy.rmse_kurt >= 2.0]
    macros["learnSatLevel"] = f"{float(sat.t_obs.min()):g}" if len(sat) else "none"
    win = noisy[(noisy.t_obs >= t_star) & (noisy.rmse_kurt < 2.0)]
    macros["learnWindowN"] = str(int(len(win)))
    if len(win) >= 3:
        s_rho = float(np.polyfit(win.t_obs, np.log(win.rmse_rho), 1)[0]); s_k = float(np.polyfit(win.t_obs, np.log(win.rmse_kurt), 1)[0])
        macros["learnSlopeRho"] = f"{s_rho:.2f}"; macros["learnSlopeKurt"] = f"{s_k:.2f}"; macros["learnSlopeRatio"] = f"{s_k - s_rho:.2f}"
        macros["learnWindowLo"] = f"{float(win.t_obs.min()):g}"; macros["learnWindowHi"] = f"{float(win.t_obs.max()):g}"
        print(f"slopes on the informative window t in [{win.t_obs.min():g}, {win.t_obs.max():g}] (t_* = {t_star:.2f}): rho {s_rho:.2f} (pred +2), kurt {s_k:.2f} (pred +4)")
    else:
        print(f"no informative window above t_* = {t_star:.2f} (kurtosis saturates at t = {macros['learnSatLevel']}): asymptotic slopes not fitted")
    pre = noisy[(noisy.t_obs < t_star) & (noisy.rmse_kurt < 2.0) & (noisy.t_obs >= 0.2)]
    if len(pre) >= 3:
        macros["learnPreSlopeRho"] = f"{float(np.polyfit(pre.t_obs, np.log(pre.rmse_rho), 1)[0]):.2f}"
        macros["learnPreSlopeKurt"] = f"{float(np.polyfit(pre.t_obs, np.log(pre.rmse_kurt), 1)[0]):.2f}"
        macros["learnPreLo"] = f"{float(pre.t_obs.min()):g}"; macros["learnPreHi"] = f"{float(pre.t_obs.max()):g}"
        print(f"pre-asymptotic slopes on [{pre.t_obs.min():g}, {pre.t_obs.max():g}]: rho {macros['learnPreSlopeRho']}, kurt {macros['learnPreSlopeKurt']}")
    for _, r in summ.iterrows():
        tag = _mac(f"{r.t_obs:g}")
        macros[f"learnRmseRhoT{tag}"] = f"{r.rmse_rho:.4f}"; macros[f"learnRmseKurtT{tag}"] = f"{r.rmse_kurt:.3f}"
        macros[f"learnHellT{tag}"] = f"{r.hellinger:.3f}"; macros[f"learnRiskT{tag}"] = f"{r.risk:.4f}"
        # the level itself, so prose never types a level next to a number computed at it
        macros[f"learnLevelT{tag}"] = f"{r.t_obs:g}"
    # Budget diagnostics. `at_cap` is the fraction of cells whose VALIDATION-selected checkpoint is
    # the last rung of the EM ladder -- i.e. validation was still improving when the ladder ran out,
    # so a longer ladder could have selected differently. It is not the same as the EM's own
    # convergence tolerance, and it is the one that bears on whether a reported error is
    # information or budget. Reported per level and for the two highest levels together.
    cap = df.groupby("t_obs").at_cap.mean()
    for t, v in cap.items():
        macros[f"learnAtCapT{_mac(f'{t:g}')}"] = f"{v:.2f}"
    lo = cap[cap.index <= 1.0]; hi = cap[cap.index > 1.0]
    macros["learnAtCapLowMax"] = f"{lo.max():.2f}"
    macros["learnAtCapHighMin"] = f"{hi.min():.2f}"
    macros["learnAtCapHighFrom"] = f"{hi.index.min():g}"
    macros["learnSeeds"] = str(int(summ.n_seeds.max())); macros["learnSize"] = str(int(df.n_chains.iloc[0]))
    # --- E06c, the budget control: the same seeds and data with the EM ladder extended to 8000
    # iterations. Paired by (seed, level) with E06; reported per level as its own RMSE, at_cap and
    # efficiency against the same reference, plus the largest movement at the informative levels.
    summc = None
    budget = ROOT / "outputs/E06c/levels.csv"
    if budget.exists() and not dev:
        import json
        dfc = pd.read_csv(budget)
        m = df.merge(dfc, on=["seed", "t_obs"], suffixes=("", "_c"))
        assert len(m) == len(df) == len(dfc), "E06c must pair one-to-one with E06"
        ladder = json.loads((budget.parent / "params_levels.json").read_text())["settings"]["em_checkpoints"]
        macros["learnBudgetLadderMax"] = str(int(max(ladder)))
        gc = m.groupby("t_obs")
        summc = pd.DataFrame({
            "rmse_rho": gc.apply(lambda d: np.sqrt(((d.fit_rho_c - d.true_rho) ** 2).mean()), include_groups=False),
            "rmse_kurt": gc.apply(lambda d: np.sqrt(((d.fit_kurt_c - d.true_kurt) ** 2).mean()), include_groups=False),
            "rmse_rho_base": gc.apply(lambda d: np.sqrt(((d.fit_rho - d.true_rho) ** 2).mean()), include_groups=False),
            "rmse_kurt_base": gc.apply(lambda d: np.sqrt(((d.fit_kurt - d.true_kurt) ** 2).mean()), include_groups=False),
            "at_cap": gc.at_cap_c.mean(), "risk": gc.test_risk_schedule_c.mean(),
            "same_ckpt": gc.apply(lambda d: (d.checkpoint == d.checkpoint_c).mean(), include_groups=False),
        }).reset_index()
        for _, r in summc.iterrows():
            tag = _mac(f"{r.t_obs:g}")
            macros[f"learnBudgetRmseRhoT{tag}"] = f"{r.rmse_rho:.4f}"; macros[f"learnBudgetRmseKurtT{tag}"] = f"{r.rmse_kurt:.3f}"
            macros[f"learnBudgetAtCapT{tag}"] = f"{r.at_cap:.2f}"; macros[f"learnBudgetRiskT{tag}"] = f"{r.risk:.4f}"
            if crlb is not None and r.t_obs > 0:
                macros[f"learnBudgetEffRhoT{tag}"] = f"{r.rmse_rho / at('crlb_rho', r.t_obs):.2f}"
        low = summc[(summc.t_obs > 0) & (summc.t_obs <= 1.0)]
        macros["learnBudgetLowHi"] = f"{float(low.t_obs.max()):g}"
        macros["learnBudgetMaxAbsDiffRhoLow"] = f"{(low.rmse_rho - low.rmse_rho_base).abs().max():.4f}"
        macros["learnBudgetMaxAbsDiffKurtLow"] = f"{(low.rmse_kurt - low.rmse_kurt_base).abs().max():.3f}"
        print("E06c budget control:"); print(summc.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    (sec / "learning-numbers.tex").write_text("%% GENERATED by tools/make_learning_numbers.py -- do not hand-edit.\n" + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")

    # figure: two panels, RMSE of rho and of kurtosis against t_obs, the estimator (points and mean)
    # against the information bound read from E01 (dashed), the clean-data floor (dotted) and, for the
    # shape coordinate, its full range (3): the estimator sits on the bound until the bound leaves the range
    fs.apply()
    fig, axes = plt.subplots(1, 2, figsize=(fs.COL_W, 1.95), gridspec_kw=dict(wspace=0.5))
    clean_labels = []
    for ax, col, lab, colr, bcol, tru in ((axes[0], "rmse_rho", r"RMSE of $\hat\rho$", fs.CAT[0], "crlb_rho", rho), (axes[1], "rmse_kurt", r"RMSE of $\hat\kappa_4$", fs.CAT[1], "crlb_kurt", 3.0)):
        for t, d in df[df.t_obs > 0].groupby("t_obs"):
            key = "fit_rho" if col == "rmse_rho" else "fit_kurt"
            ax.plot([t] * len(d), np.abs(d[key] - tru), ls="none", marker=".", ms=2, color=colr, alpha=0.25)
        ax.plot(noisy.t_obs, noisy[col], color=colr, marker="o", ms=3, lw=1.3, label="EM–BP")
        if summc is not None:
            # the budget control, only where it differs from the base run (the levels past t = 1)
            nc = summc[summc.t_obs >= 1.0]
            ax.plot(nc.t_obs, nc[col], color=colr, marker="o", ms=3.6, mfc="white", mew=0.9, lw=0, label="extended ladder")
        if crlb is not None:
            m = (crlb.t_obs >= noisy.t_obs.min() * 0.9) & (crlb.t_obs <= noisy.t_obs.max() * 1.05)
            ax.plot(crlb.t_obs[m], crlb[bcol][m], color=fs.INK2, lw=0.9, ls="--", label="information bound")
        clean = summ[summ.t_obs == 0]
        if len(clean):
            ax.axhline(float(clean[col].iloc[0]), color=fs.MUTED, lw=0.7, ls=":")
            # at the far right, just UNDER the floor: the left end of this line is buried in the
            # replicate cloud, and the region below the clean-data floor is empty by construction
            clean_labels.append((ax, float(clean[col].iloc[0])))
        ax.set_yscale("log")
        lo = float(df[df.t_obs > 0][("fit_rho" if col == "rmse_rho" else "fit_kurt")].sub(tru).abs().replace(0, np.nan).min()) * 0.5
        if col == "rmse_kurt":
            ax.axhline(3.0, color=fs.MUTED, lw=0.7, ls="-.")
            fs.halo(ax.text(noisy.t_obs.max(), 3.0 * 1.3, "range of $\\kappa_4$", fontsize=7,
                            color=fs.INK2, ha="right", va="bottom", zorder=6))
            ax.set_ylim(max(lo, 1e-2), max(30.0, float(noisy[col].max()) * 3))
        else:
            ax.set_ylim(max(lo, 1e-4), max(1.0, float(noisy[col].max()) * 3))
        ax.set_xlabel("observation level  $t$"); ax.set_ylabel(lab)
    # placed last, so the y-limits are final and the label can be tucked under its own line
    for ax, yv in clean_labels:
        x0, x1 = ax.get_xlim()
        fs.halo(ax.text(x1 - 0.02 * (x1 - x0), yv * 0.82, "clean data", fontsize=7,
                        color=fs.INK2, ha="right", va="top", zorder=6))
    axes[0].legend(loc="upper left", fontsize=6.5, frameon=False, handlelength=1.8, borderaxespad=0.2)
    fs.panel_label(axes[0], "(a)", x=-0.35); fs.panel_label(axes[1], "(b)", x=-0.35)
    fs.save(fig, figdir / "fig_learning.pdf")
    print("wrote", sec / "learning-numbers.tex", "and", figdir / "fig_learning.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
