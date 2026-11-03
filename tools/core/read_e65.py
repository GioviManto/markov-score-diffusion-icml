"""Read E65 against its registered predictions (experiments/registry/E65.json, amendment of 3 Nov before code).

    python tools/core/read_e65.py              # gates, tables, verdicts P1-P4, descriptive D1-D2

Inputs: E65's per-checkpoint curves (outputs/hpc_E65_k_s*/curves_kernels.csv, outputs/hpc_E65_nets/curves_nets.csv),
each row a (seed, N, arm, checkpoint) with the test error on all 32 sites (`test_all`), the flag of the checkpoint the
arm's ORIGINAL oracle rule selects (`sel_oracle`) and of the one its oracle-free criterion selects (`sel_free`).
Frozen comparators for the reproduction gate P1: E30's em_bp, E56's spline and energy, E58's clean energy_h16 (Laplace),
E03's six networks (confirm.csv, all sites).

A FAILED or missing fit counts AGAINST every prediction it enters: a seed mean needs all sixteen seeds, a paired win
needs both rows. An arm that fails P1 has its P2-P4 verdicts reported as VOID. The verdict functions take plain tables,
so tests/test_read_e65.py drives them with made-up data in both directions.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(16))
SIZES = (32, 128, 512, 2048)
EM_ARMS = ("em_bp", "spline", "energy")
KERNEL_ARMS = EM_ARMS + ("clean_energy",)
NETS = ("mlp", "window", "tcn", "bigru", "transformer", "unet")
P1_RTOL, P2_RTOL, P4_RTOL = 0.02, 0.10, 0.10


def _csvs(pattern):
    paths = sorted(glob.glob(str(ROOT / pattern)))
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else pd.DataFrame()


def load():
    c = pd.concat([_csvs("outputs/hpc_E65_k_s*/curves_kernels.csv"), _csvs("outputs/hpc_E65_nets/curves_nets.csv")],
                  ignore_index=True)
    if c.empty:
        raise SystemExit("no E65 curves on disk")
    frozen = []
    e30 = _csvs("outputs/hpc_E30_s*/joint.csv")
    frozen.append(e30[e30.arm == "em_bp"].drop_duplicates(["seed", "n_chains"]).assign(arm="em_bp", risk=lambda d: d.risk_pooled))
    e56 = _csvs("outputs/hpc_E56_s*/kernels.csv")
    frozen.append(e56[e56.arm.isin(["spline", "energy"]) & (e56.failed == 0)]
                  .drop_duplicates(["seed", "n_chains", "arm"]).assign(risk=lambda d: d.risk_pooled))
    e58 = _csvs("outputs/hpc_E58_kern_lap_s*/kernels.csv")
    frozen.append(e58[(e58.cell == "laplace") & (e58.arm == "energy_h16")].drop_duplicates(["seed", "n_chains"])
                  .assign(arm="clean_energy", risk=lambda d: d.risk_pooled))
    e03 = pd.read_csv(ROOT / "outputs/E03/confirm.csv")
    frozen.append(e03[e03.region == "all"].rename(columns={"arch": "arm"}))
    f = pd.concat([d[["arm", "seed", "n_chains", "risk", "test_bundle"]] for d in frozen], ignore_index=True)
    return c, f


def selected(curves, rule):
    """(arm, n_chains, seed) -> test error at the checkpoint `rule` ('oracle' or 'free') selects."""
    s = curves[(curves[f"sel_{rule}"] == 1) & (curves.failed.fillna(0) == 0)]
    return s.set_index(["arm", "n_chains", "seed"]).test_all


def seed_mean(series, arm, n):
    try:
        x = series.loc[(arm, n)].reindex(SEEDS)
    except KeyError:
        return np.nan
    return float(x.mean()) if np.isfinite(x.to_numpy(dtype=float)).all() else np.nan


def wins(a_series, b_series, a, b, n):
    """Seeds in which arm a's value is below arm b's, both finite."""
    try:
        xa = a_series.loc[(a, n)].reindex(SEEDS).to_numpy(dtype=float)
        xb = b_series.loc[(b, n)].reindex(SEEDS).to_numpy(dtype=float)
    except KeyError:
        return 0
    return int((np.isfinite(xa) & np.isfinite(xb) & (xa < xb)).sum())


def reproduction(curves, frozen):
    """P1 per arm: E65's oracle-selected seed mean within 2% of the frozen seed mean at every N."""
    o = selected(curves, "oracle")
    fz = frozen.set_index(["arm", "n_chains", "seed"]).risk
    out = {}
    for arm in KERNEL_ARMS + NETS:
        devs = []
        for n in SIZES:
            a, b = seed_mean(o, arm, n), seed_mean(fz, arm, n)
            devs.append(abs(a / b - 1) if np.isfinite(a) and np.isfinite(b) else np.inf)
        out[arm] = (max(devs) <= P1_RTOL, max(devs))
    return out


def verdicts(curves, frozen):
    rep = reproduction(curves, frozen)
    o, f = selected(curves, "oracle"), selected(curves, "free")
    p1_ok = all(ok for ok, _ in rep.values())
    p1_msg = "; ".join(f"{a} {d:.1%}" for a, (ok, d) in rep.items() if not ok) or "every arm within 2%"

    p2 = {a: max(abs(seed_mean(f, a, n) / seed_mean(o, a, n) - 1) if np.isfinite(seed_mean(f, a, n) / seed_mean(o, a, n))
                 else np.inf for n in (128, 512, 2048)) for a in EM_ARMS}
    best_net = {n: min(NETS, key=lambda a: (seed_mean(f, a, n) if np.isfinite(seed_mean(f, a, n)) else np.inf)) for n in SIZES}
    p3 = {a: min(wins(f, f, a, best_net[n], n) for n in SIZES) for a in EM_ARMS}
    p4 = max(abs(seed_mean(f, "bigru", n) / seed_mean(o, "bigru", n) - 1)
             if np.isfinite(seed_mean(f, "bigru", n) / seed_mean(o, "bigru", n)) else np.inf for n in SIZES)

    def void(arms, ok):
        return "VOID" if any(not rep[a][0] for a in arms) else ("CONFIRMED" if ok else "FALSIFIED")

    return {
        "P1": ("CONFIRMED" if p1_ok else "FALSIFIED", f"reproduction of the frozen rows: {p1_msg}"),
        "P2": (void(EM_ARMS, all(v <= P2_RTOL for v in p2.values())),
               "evidence vs oracle selection, max |ratio - 1| over N >= 128: " + ", ".join(f"{a} {v:.1%}" for a, v in p2.items())),
        "P3": (void(EM_ARMS + NETS, all(v >= 14 for v in p3.values())),
               "oracle-free both sides, min over N of seeds below the best network: " + ", ".join(f"{a} {v}/16" for a, v in p3.items())
               + f" (best network by N: {best_net})"),
        "P4": (void(("bigru",), p4 <= P4_RTOL), f"BiGRU val_dsm vs oracle selection, max |ratio - 1| over N: {p4:.1%}"),
    }


def gates(curves, frozen):
    k = curves[curves.failed.fillna(0) == 0]
    per = k.groupby(["arm", "n_chains", "seed"])
    assert (per.sel_oracle.sum() == 1).all() and (per.sel_free.sum() == 1).all(), "G1: not exactly one selection per rule"
    e03 = frozen[frozen.arm.isin(NETS)].drop_duplicates(["seed", "n_chains"]).set_index(["seed", "n_chains"]).test_bundle
    mine = k.drop_duplicates(["seed", "n_chains"]).set_index(["seed", "n_chains"]).test_bundle
    common = mine.index.intersection(e03.index)
    assert len(common) and (mine[common] == e03[common]).all(), "G2: E65's test bundles are not E03's"


def descriptive(curves):
    k = curves[curves.failed.fillna(0) == 0]
    sel = k[k.sel_oracle == 1].groupby(["arm", "n_chains"]).checkpoint.median().unstack()
    fre = k[k.sel_free == 1].groupby(["arm", "n_chains"]).checkpoint.median().unstack()
    rho = (k.groupby(["arm", "n_chains", "seed"])
           .apply(lambda d: d.val_all.rank().corr(d.criterion.rank()) if len(d) > 2 else np.nan)
           .groupby(level=[0, 1]).median().unstack())
    return sel, fre, rho


def main(argv) -> int:
    curves, frozen = load()
    gates(curves, frozen)
    o, f = selected(curves, "oracle"), selected(curves, "free")
    tab = pd.DataFrame({(rule, n): {a: seed_mean(s, a, n) for a in KERNEL_ARMS + NETS}
                        for rule, s in (("oracle", o), ("free", f)) for n in SIZES})
    print("D3  seed-mean test error, oracle and oracle-free selection")
    print(tab.to_string(float_format=lambda v: f"{v:.4f}"))
    sel, fre, rho = descriptive(curves)
    print("D1  median selected checkpoint, oracle rule\n" + sel.to_string())
    print("D1  median selected checkpoint, oracle-free rule\n" + fre.to_string())
    print("D2  median Spearman correlation across checkpoints, val_all vs criterion\n" + rho.to_string(float_format=lambda v: f"{v:.2f}"))
    for k, (verdict, msg) in verdicts(curves, frozen).items():
        print(f"{k}  {verdict}: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
