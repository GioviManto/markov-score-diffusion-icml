"""Read E51 against its registered predictions (experiments/registry/E51.json).

Prints the per-(arm, cell, M) medians and IQRs over seeds and a verdict line per prediction.
Writes nothing; the figure is drawn by tools/core/fig_readings.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
BASE = "T0.3_h1_L32"


def iqr(s):
    return float(np.nanpercentile(s, 75) - np.nanpercentile(s, 25))


def main() -> int:
    df = pd.read_csv(ROOT / "outputs/E51/readings.csv")
    floor = pd.read_csv(ROOT / "outputs/hpc_E33o/oracles.csv").query("arm == 'lmmse_true'").groupby("cell").risk.mean()
    df["risk_over_floor"] = df.risk / df.cell.map(floor)
    bad = df[(df.failed == 1) | ~np.isfinite(df.theta_ratio)]
    print(f"fits {len(df)}, failed or non-finite {len(bad)}")
    pd.set_option("display.width", 200)
    g = df.groupby(["arm", "cell", "n_chains"])
    tab = g.agg(theta_med=("theta_ratio", "median"), theta_iqr=("theta_ratio", iqr), barrier_med=("barrier_hat", "median"),
                cv_med=("curvature_cv", "median"), n=("seed", "size")).round(4)
    print(tab.to_string())

    v = {}
    cls = df[df.arm != "em_mdn"]
    d_theta = np.abs(cls.theta_ratio / cls.own_theta_ratio - 1).max()
    d_bar = np.abs(cls.barrier_hat - cls.own_barrier).max()
    v["P1"] = (d_theta < 1e-6 and d_bar < 1e-6, f"max rel dev Theta {d_theta:.2e}, max abs dev barrier {d_bar:.2e}")
    mdn = df[df.arm == "em_mdn"]
    m2048 = mdn[mdn.n_chains == 2048].groupby("cell")
    med = m2048.theta_ratio.median()
    v["P2"] = (0.9 <= med[BASE] <= 1.1, f"base-cell median Theta_hat/Theta = {med[BASE]:.3f}")
    n_ok = int(((med >= 0.8) & (med <= 1.2)).sum())
    v["P3"] = (n_ok >= 5, f"{n_ok}/6 cells in [0.8, 1.2]: " + ", ".join(f"{c} {x:.3f}" for c, x in med.items()))
    bar = m2048.barrier_hat.median()[BASE]
    v["P4"] = (0.175 <= bar <= 0.325, f"base-cell median barrier = {bar:.3f} (true 0.25)")
    wider = 0
    pairs = 0
    for (cell, n), sub in df.groupby(["cell", "n_chains"]):
        a = iqr(sub[sub.arm == "em_mdn"].theta_ratio)
        b = iqr(sub[sub.arm == "em_pair_quartic"].theta_ratio)
        pairs += 1
        wider += int(a > b)
    v["P5"] = (wider == pairs, f"MDN IQR wider in {wider}/{pairs} (cell, M) pairs")
    cv = mdn[mdn.cell == BASE].groupby("n_chains").curvature_cv.median()
    v["P6"] = (cv[2048] < cv[32], f"base-cell median curvature_cv: M=32 {cv[32]:.3f}, M=2048 {cv[2048]:.3f}")
    rho = spearmanr(mdn.curvature_cv, mdn.risk_over_floor).statistic
    v["P7"] = (rho > 0.3, f"Spearman(curvature_cv, risk/floor) over {len(mdn)} MDN fits = {rho:.3f}")
    print()
    for k, (ok, msg) in v.items():
        print(f"{k}: {'CONFIRMED' if ok else 'FALSIFIED'}  {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
