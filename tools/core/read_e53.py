"""Read E53 against its registered predictions (experiments/registry/E53.json)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
MIN_ABS_A = 0.5


def domain_table(df):
    d = df[(df.domain.notna()) & (df.domain != "") & (df.a.abs() >= MIN_ABS_A)]
    core, edge = d[d.role == "core"], d[d.role == "edge"]
    t = pd.DataFrame({"tau": core.groupby("domain").tau.median(), "within_sd": core.groupby("domain").tau.std(ddof=0),
                      "length": core.groupby("domain").domain_len.first(), "edge_tau": edge.groupby("domain").tau.median()})
    return t.dropna(subset=["tau"])


def main() -> int:
    df = pd.read_csv(ROOT / "outputs/E53/sites.csv", keep_default_na=False, na_values=[])
    for c in ("tau", "a"):
        df[c] = df[c].astype(float)
    df["domain_len"] = df["domain_len"].astype(int)
    verdicts = {"P1": [], "P2": [], "P3": []}
    for cell, sub in df.groupby("cell"):
        t = domain_table(sub)
        n = len(t)
        within = float(t.within_sd.median())
        across = float(t.tau.std(ddof=0))
        rho = float(spearmanr(t.tau, t.length).statistic)
        diff = (t.tau - t.edge_tau).dropna()
        share = float((diff > 0).mean())
        resolved = n >= 20
        print(f"{cell}: xi={sub.xi_sites.iloc[0]:.2f}, {n} domains; median within-domain sd {within:.3f}, across-domain sd "
              f"{across:.3f} (ratio {within / across:.3f}); Spearman(tau, length) {rho:.3f}; core - edge median "
              f"{diff.median():+.3f}, positive in {share:.0%}; never-committed sites {(sub.tau == 0).mean():.1%}")
        verdicts["P1"].append(resolved and within < 0.25 * across)
        verdicts["P2"].append(resolved and rho > 0.4)
        verdicts["P3"].append(resolved and diff.median() > 0 and share >= 0.75)
    print()
    for k, oks in verdicts.items():
        print(f"{k}: {'CONFIRMED' if all(oks) else 'FALSIFIED'}  per cell {oks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
