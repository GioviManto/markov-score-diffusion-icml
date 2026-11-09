"""Read E70 against its registered predictions (experiments/registry/E70.json).

    python tools/core/read_e70.py        # P1-P4 and the stall counts before/after the fix

Each refitted neural-kernel row is paired with its frozen original by (source, cell, seed, N). A stall is a fit that
ends with final_lr < 1e-10 at an error above 0.8 times the linear floor. A missing or failed row counts AGAINST.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
LAP_FLOOR = 0.1672                                   # E58 lmmse on the Laplace chain (seed mean)


def _cat(pattern):
    paths = sorted(glob.glob(str(ROOT / pattern)))
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else pd.DataFrame()


def _rows(df, source, cell, risk_col, arm=None, selection=None):
    if df.empty:
        return pd.DataFrame(columns=["source", "cell", "seed", "n_chains", "risk", "final_lr"])
    d = df.copy()
    if arm is not None:
        d = d[d.arm == arm]
    if selection is not None:
        d = d[d.selection == selection]
    if "cell" in d and cell != "laplace":
        d = d[d.cell == cell]
    d = d[d.failed == 0].drop_duplicates(["seed", "n_chains"])
    return pd.DataFrame({"source": source, "cell": cell, "seed": d.seed, "n_chains": d.n_chains,
                         "risk": d[risk_col], "final_lr": d.get("final_lr", np.nan)})


def load():
    pairs = []
    for cell, orig_pat, new_pat in (("laplace", "outputs/hpc_E58_kern_lap_s*/kernels.csv", "outputs/hpc_E70_clean_lap_s*/kernels.csv"),
                                    ("T0.3_h1_L32", "outputs/hpc_E58_kern_T0.3_h1_L32/kernels.csv", "outputs/hpc_E70_clean_T0.3_h1_L32/kernels.csv"),
                                    ("T1_iso_L32", "outputs/hpc_E58_kern_T1_iso_L32/kernels.csv", "outputs/hpc_E70_clean_T1_iso_L32/kernels.csv")):
        o, n = _rows(_cat(orig_pat), "E58", cell, "risk_pooled", "energy_h16"), _rows(_cat(new_pat), "E58", cell, "risk_pooled", "energy_h16")
        pairs.append((o, n))
    pairs.append((_rows(_cat("outputs/hpc_E56_s*/kernels.csv"), "E56", "laplace", "risk_pooled", "energy"),
                  _rows(_cat("outputs/hpc_E70_lap_em_s*/kernels.csv"), "E56", "laplace", "risk_pooled", "energy")))
    pairs.append((_rows(_cat("outputs/hpc_E55_kern_T0.3_h1_L32/kernels.csv"), "E55", "T0.3_h1_L32", "risk", "em_energy", "risk"),
                  _rows(_cat("outputs/hpc_E70_field_em/kernels.csv"), "E55", "T0.3_h1_L32", "risk", "em_energy", "risk")))
    orig = pd.concat([p[0] for p in pairs], ignore_index=True)
    new = pd.concat([p[1] for p in pairs], ignore_index=True)
    floors = _field_floors()
    for d in (orig, new):
        d["floor"] = [LAP_FLOOR if c == "laplace" else floors.get((c, s), np.nan) for c, s in zip(d.cell, d.seed)]
        d["stalled"] = (d.final_lr < 1e-10) & (d.risk > 0.8 * d.floor)
    return orig, new


def _field_floors():
    o = pd.read_csv(ROOT / "outputs/hpc_E33o/oracles.csv")
    o = o[o.arm == "lmmse_true"].drop_duplicates(["cell", "seed"])
    return {(c, s): r for c, s, r in zip(o.cell, o.seed, o.risk)}


def verdicts(orig, new) -> dict:
    key = ["source", "cell", "seed", "n_chains"]
    m = orig.merge(new, on=key, how="left", suffixes=("_orig", "_new"))
    p1 = bool(len(new) == len(orig) and not new.stalled.any())
    g = new.groupby(["source", "cell", "n_chains"]).risk
    p2 = bool(((g.mean() - g.median()).abs() / g.median() <= 0.15).all())
    ok = m[~m.stalled_orig]
    close = ((ok.risk_new - ok.risk_orig).abs() / ok.risk_orig <= 0.10)
    p3 = bool(ok.risk_new.notna().all() and close.mean() >= 0.8)
    p4 = _p4(new)
    out = lambda d: int((d.risk > 3 * d.groupby(["source", "cell", "n_chains"]).risk.transform("median")).sum())  # noqa: E731
    return {"P1": p1, "P2": p2, "P3": p3, "P4": p4, "D3_outliers_orig": out(orig), "D3_outliers_new": out(new),
            "stalls_orig": int(orig.stalled.sum()),
            "stalls_new": int(new.stalled.sum()), "P3_fraction_close": float(close.mean()) if len(close) else np.nan}


def _p4(new) -> bool:
    e02 = pd.read_csv(ROOT / "outputs/E02a/capacity.csv")
    e02 = e02[(e02.region == "all") & (e02.arm == "em_bp") & (e02.components == 8)].drop_duplicates(["seed", "n_chains"])
    e03 = pd.read_csv(ROOT / "outputs/E03/confirm.csv")
    e03 = e03[e03.region == "all"]
    nk = new[(new.source == "E56")].set_index(["seed", "n_chains"]).risk
    if len(nk) != 64:
        return False
    mix = e02.set_index(["seed", "n_chains"]).risk
    if nk.xs(2048, level=1).mean() > 1.15 * mix.xs(2048, level=1).mean():
        return False
    for arch in ("window", "mlp"):
        net = e03[e03.arch == arch].set_index(["seed", "n_chains"]).risk
        if not (nk < net.reindex(nk.index)).all():
            return False
    return True


def main() -> int:
    orig, new = load()
    v = verdicts(orig, new)
    print({k: ("CONFIRMED" if x is True else "FALSIFIED" if x is False else x) for k, x in v.items()})
    pd.set_option("display.width", 200)
    print(new.groupby(["source", "cell", "n_chains"]).risk.agg(["mean", "median", "count"]).round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
