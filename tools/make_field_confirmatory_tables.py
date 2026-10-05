#!/usr/bin/env python
"""Tables of the field laboratory's confirmatory results, generated from the landed CSVs.

Writes complete tabular environments (no captions; the report carries those) to reports/field-tables/:
    conf-pair.tex     E33, em_pair_quartic through noise (hpc_E33q_s*): Theta_hat/Theta and the barrier at
                      M = 2048 (P8), and the risk as a fraction of the linear floor (hpc_E33o) at every M
    conf-clean.tex    E34 clean (hpc_E34c): mixture/spline (P3), spline and Legendre risk at M = 2048 (P2,
                      clean half) and the parked and eigensolve-failure counts (amendment 2)
    conf-samples.tex  E34 physics_learned on the pair fits (E34_learned_pair): generated / true statistics
    conf-e35.tex      E35 (E35/augmented.csv): the global latent and the augmented state, per seed
    conf-score.tex    E33 complete (hpc_E33k_s*, hpc_E33q_s*, hpc_E33n_s*): every arm's risk over the linear floor,
                      per cell, at M = 128 and M = 2048 (failed fits excluded from the mean and counted)
    conf-p4.tex       E34 physics_learned on every arm (E34_learned, E34_learned_pair): generated / true kink
                      density and mean domain length, cell means over the four seeds

Every number in the report's confirmatory section comes from here; nothing is typed by hand.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "reports" / "field-tables"

ORDER = ["T0.3_h1_L32", "T0.5_iso_L32", "T1_iso_L32", "T2_iso_L32", "T0.3_h0.5_L64", "T0.3_h0.25_L128"]
LABEL = {"T0.3_h1_L32": r"$\Th=0.3$, $h=1$", "T0.5_iso_L32": r"$\Th=0.5$", "T1_iso_L32": r"$\Th=1$",
         "T2_iso_L32": r"$\Th=2$", "T0.3_h0.5_L64": r"$\Th=0.3$, $h=0.5$", "T0.3_h0.25_L128": r"$\Th=0.3$, $h=0.25$"}


def _load(pattern):
    paths = sorted(glob.glob(str(ROOT / pattern)))
    if not paths:
        raise SystemExit(f"missing: {pattern}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


HEAD = {
    "conf-pair.tex": [r"\begin{tabular}{@{}l c c c r r r r@{}}", r"\toprule",
                      r"& \multicolumn{3}{c}{$M=2048$} & \multicolumn{4}{c}{risk / linear floor} \\",
                      r"\cmidrule(lr){2-4}\cmidrule(l){5-8}",
                      r"cell & $\hat\Th/\Th$ & barrier & band & $32$ & $128$ & $512$ & $2048$ \\", r"\midrule"],
    "conf-samples.tex": [r"\begin{tabular}{@{}l c c c c c c@{}}", r"\toprule",
                         r"cell & kinks & domain & coarse kinks & coarse domain & bimodality & variance \\", r"\midrule"],
    "conf-clean.tex": [r"\begin{tabular}{@{}l r r r r r r@{}}", r"\toprule",
                       r"& \multicolumn{2}{c}{mixture / spline} & spline & \multicolumn{3}{c}{Legendre tilt} \\",
                       r"\cmidrule(lr){2-3}\cmidrule(l){5-7}",
                       r"cell & ratio & $>1$ & risk & risk & parked & eig. \\", r"\midrule"],
    "conf-score.tex": [r"\begin{tabular}{@{}l l r r r r r r@{}}", r"\toprule",
                       r"$M$ & arm & $\Th=0.3$ & $\Th=0.5$ & $\Th=1$ & $\Th=2$ & $h=0.5$ & $h=0.25$ \\", r"\midrule"],
    "conf-p4.tex": [r"\begin{tabular}{@{}l c c c c c c@{}}", r"\toprule",
                    r"arm & $\Th=0.3$ & $\Th=0.5$ & $\Th=1$ & $\Th=2$ & $h=0.5$ & $h=0.25$ \\", r"\midrule"],
    "conf-e35.tex": [r"\begin{tabular}{@{}r r c c c c@{}}", r"\toprule",
                     r"$b$ & $M$ & first-order & augmented & linear floor & first / augmented \\", r"\midrule"],
}


def _write(name, lines):
    OUT.mkdir(parents=True, exist_ok=True)
    body = HEAD[name] + lines + [r"\bottomrule", r"\end{tabular}"]
    (OUT / name).write_text("\n".join(body) + "\n")
    print("wrote", OUT / name)


def pair_table():
    from frozen_config import FROZEN
    from make_field_potential_figs import _readings
    from markov_score.bp_grid import make_grid
    grid, _ = make_grid(FROZEN.half_width, FROZEN.n_grid)
    Q = _load("outputs/hpc_E33q_s*/kernels.csv")
    Q = Q[Q.selection == "risk"]
    floor = _load("outputs/hpc_E33o/oracles.csv").query("arm == 'lmmse_true'").groupby("cell").risk.mean()
    q48 = Q[Q.n_chains == 2048].copy(); q48["source"] = "noisy"
    tab, _ = _readings(q48, grid)
    risk = Q.groupby(["cell", "n_chains"]).risk.mean().unstack()
    lines = []
    for c in ORDER:
        g = tab[tab.cell == c]
        T = float(g["T"].iloc[0])
        band = f"{int((np.abs(g.barrier_hat - 0.25) <= 0.05).sum())}/{len(g)}" if T <= 1 else "--"
        frac = [risk.loc[c, m] / floor[c] for m in (32, 128, 512, 2048)]
        lines.append(f"{LABEL[c]} & ${g.theta_ratio.mean():.3f}\\,({g.theta_ratio.std():.3f})$ & "
                     f"${g.barrier_hat.mean():.3f}\\,({g.barrier_hat.std():.3f})$ & {band} & "
                     + " & ".join(f"{v:.3f}" for v in frac) + r" \\")
    _write("conf-pair.tex", lines)


def clean_table():
    from field_diagnostics import annotate
    from markov_score.bp_grid import make_grid
    g, w = make_grid(8.0, 401)
    C = _load("outputs/hpc_E34c/clean.csv")
    C = C[(C.selection == "risk") & (C.n_chains == 2048)]
    piv = C.pivot_table(index=["cell", "seed"], columns="arm", values="risk")
    poly = annotate(C[C.arm == "em_tilted_poly"], g, w)
    lines = []
    for c in ORDER:
        p = piv.loc[c]
        ratio = p.em_mixture / p.em_tilted_spline
        pk = poly[poly.cell == c]
        lines.append(f"{LABEL[c]} & {ratio.mean():.1f} & {int((ratio > 1).sum())}/{len(ratio)} & "
                     f"{p.em_tilted_spline.mean():.4f} & {p.em_tilted_poly.mean():.4f} & "
                     f"{int(pk.parked.sum())}/{len(pk)} & {int(pk.eig_failed.sum())} \\\\")
    _write("conf-clean.tex", lines)


def samples_table():
    d = _load("outputs/E34_learned_pair/learned.csv")
    t = d[d.arm == "truth"].set_index(["cell", "seed"])
    a = d[d.arm == "em_pair_quartic"].set_index(["cell", "seed"])
    cols = ["kink_density", "mean_domain", "coarse_kink_density", "coarse_mean_domain", "bimodality_coefficient", "var"]
    r = a[cols] / t.loc[a.index, cols]
    lines = []
    for c in ORDER:
        rc = r.loc[c]
        lines.append(f"{LABEL[c]} & " + " & ".join(f"{rc[k].mean():.3f}\\,({np.abs(rc[k] - 1).max():.3f})" for k in cols) + r" \\")
    _write("conf-samples.tex", lines)


def e35_table():
    d = _load("outputs/E35/augmented.csv")
    p = d.pivot_table(index=["b", "n_chains", "seed"], columns="arm", values="risk")
    lines = []
    for (b, m), g in p.groupby(level=[0, 1]):
        def two(col, fmt="{:.4f}"):
            return " / ".join(fmt.format(v) for v in g[col].values)
        ratio = " / ".join(f"{v:.2f}" for v in (g.em_first_order / g.em_augmented).values)
        floor = "0 (exact)" if b == 0 else two("lmmse_true")
        lines.append(f"{b:g} & {m} & {two('em_first_order')} & {two('em_augmented')} & {floor} & {ratio} \\\\")
    _write("conf-e35.tex", lines)


ARM_LABEL = [("em_pair_quartic", "site energy (5)"), ("em_tilted_spline", "spline tilt (25)"), ("em_tilted_poly", "Legendre tilt (7)"),
             ("em_mdn", "MDN"), ("em_mixture", "mixture (paper)"), ("ism:window", "window, ISM"), ("ism:transformer", "transformer, ISM"),
             ("dsm:window", "window, DSM"), ("dsm:transformer", "transformer, DSM")]


def score_table():
    K = pd.concat([_load("outputs/hpc_E33k_s*/kernels.csv"), _load("outputs/hpc_E33q_s*/kernels.csv")], ignore_index=True)
    K = K[K.selection == "risk"][["cell", "n_chains", "arm", "risk"]]
    N = _load("outputs/hpc_E33n_s*/nets.csv")
    N = N.assign(arm=N.arm.str.replace("ism_one_view", "ism").str.replace("dsm_one_corruption", "dsm") + ":" + N.arch)[["cell", "n_chains", "arm", "risk"]]
    A = pd.concat([K, N], ignore_index=True)
    floor = _load("outputs/hpc_E33o/oracles.csv").query("arm == 'lmmse_true'").groupby("cell").risk.mean()
    mean = A.groupby(["n_chains", "arm", "cell"]).risk.mean()
    nfail = A.groupby(["n_chains", "arm", "cell"]).risk.apply(lambda r: int(r.isna().sum()))
    lines = []
    for m in (128, 2048):
        for i, (arm, lab) in enumerate(ARM_LABEL):
            vals = []
            for c in ORDER:
                f = nfail.get((m, arm, c), 0)
                v = mean.get((m, arm, c), np.nan)
                vals.append("failed" if f == 16 else (f"{v / floor[c]:.3f}" + (f"$^{{{f}}}$" if f else "")))
            lines.append(f"{m if i == 0 else ''} & {lab} & " + " & ".join(vals) + r" \\")
        if m == 128:
            lines.append(r"\midrule")
    _write("conf-score.tex", lines)


def p4_table():
    d = pd.concat([_load("outputs/E34_learned/learned.csv"), _load("outputs/E34_learned_pair/learned.csv")], ignore_index=True)
    t = d[d.arm == "truth"].drop_duplicates(["cell", "seed"]).set_index(["cell", "seed"])
    a = d[d.arm != "truth"].copy()
    for c in ("kink_density", "mean_domain"):
        a[c] = [r[c] / t.loc[(r.cell, r.seed), c] for _, r in a.iterrows()]
    lines = []
    for arm, lab in ARM_LABEL[:5]:
        for stat, sl in (("kink_density", "kinks"), ("mean_domain", "domain")):
            vals = []
            for c in ORDER:
                v = a[(a.arm == arm) & (a.cell == c)][stat]
                vals.append("failed" if v.isna().all() else f"{v.mean():.3f}")
            lines.append(f"{lab}, {sl} & " + " & ".join(vals) + r" \\")
    _write("conf-p4.tex", lines)


if __name__ == "__main__":
    pair_table()
    clean_table()
    samples_table()
    e35_table()
    score_table()
    p4_table()
