#!/usr/bin/env python
"""The predeclared readings of E35x and E36x (registered 5 Oct 2026, before submission).

    python tools/make_replication_numbers.py                       # the confirmatory shards
    python tools/make_replication_numbers.py --e35 outputs/E35/augmented.csv --e36 outputs/E36/clean.csv --out outputs/scratch/repl
                                                                   # dry run on the frozen two-seed designs

E35x (experiments/registry/E35x.json):
  P1  b = 0: the 95% bootstrap interval of the mean per-seed log(augmented / first-order) contains 0 at both M
  P2  b = 2: first-order seed mean changes by < 15% from M = 512 to 2048; augmented falls by > 30% and in >= 14/16 seeds
  P3  M = 2048: first/augmented > 2 in 16/16 seeds at b = 1 and 2, its seed mean rising over b = 0, 1, 2;
      at b = 2 the linear floor's seed mean is at least 3x the augmented arm's
  P4  first-order / linear floor > 1 in 16/16 seeds at b = 1 and 2, both M
E36x (experiments/registry/E36x.json):
  P1  stride 1: site energy and spline below 0.02 in 16/16; stride-4 / stride-1 >= 2 in 16/16, both arms
  P2  stride 4: MDN below the site energy in >= 12/16; UNRESOLVED if the MDN selected its last checkpoint in > 4 fits
  P3  the mixture's seed mean within 3% of E36's frozen floor at every stride
E38 (experiments/registry/E38.json):
  P1  grid BP's peak inference memory: log-log slope in [0.8, 1.2] over B = 128 -> 2048 (L = 32) and L = 32 -> 128 (B = 512)
  P2  at B = 512, L = 32, N_g = 401 grid BP's peak inference memory exceeds every network's
Bootstrap: Monte Carlo, 200,000 draws, seed 20260912 (sixteen seeds cannot be enumerated). The counts scale with the
seeds present, so a dry run on two seeds reads "2/2" where the confirmatory run needs "16/16".
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
B, SEED = 200_000, 20260912


def boot_ci(values, b=B, seed=SEED):
    v = np.asarray(values, float)
    idx = np.random.default_rng(seed).integers(0, v.size, size=(b, v.size))
    m = v[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _load(spec):
    paths = sorted(glob.glob(str(ROOT / spec))) if not Path(spec).is_absolute() else sorted(glob.glob(spec))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def row(exp, pred, clause, value, k, n, ok, note=""):
    return dict(experiment=exp, prediction=pred, clause=clause, value=value, k=k, n=n,
                verdict="UNRESOLVED" if ok is None else ("CONFIRMED" if ok else "FALSIFIED"), note=note)


def read_e35x(d):
    rows = []
    p = d.pivot_table(index=["b", "n_chains", "seed"], columns="arm", values="risk").sort_index()
    cap = d[d.arm == "em_augmented"].set_index(["b", "n_chains", "seed"]).at_cap.sort_index()
    ns = p.index.get_level_values("seed").nunique()
    ok1 = []
    for m in (512, 2048):
        g = p.loc[(0.0, m)]
        lr = np.log(g.em_augmented / g.em_first_order)
        lo, hi = boot_ci(lr)
        ok1.append(lo <= 0.0 <= hi)
        rows.append(row("E35x", "P1", f"b=0, M={m}: mean log(aug/first), 95% CI", float(lr.mean()), int((lr < 0).sum()), len(lr),
                        lo <= 0.0 <= hi, f"CI [{lo:.3f}, {hi:.3f}]"))
    g512, g2048 = p.loc[(2.0, 512)], p.loc[(2.0, 2048)]
    ch_first = g2048.em_first_order.mean() / g512.em_first_order.mean() - 1
    ch_aug = g2048.em_augmented.mean() / g512.em_augmented.mean() - 1
    k_fall = int((g2048.em_augmented < g512.em_augmented).sum())
    need = max(1, int(np.ceil(14 * ns / 16)))
    rows.append(row("E35x", "P2", "b=2: first-order change 512->2048 (< 15%)", float(ch_first), None, ns, abs(ch_first) < 0.15))
    rows.append(row("E35x", "P2", f"b=2: augmented change (< -30%) and falls in >= {need}/{ns}", float(ch_aug), k_fall, ns,
                    bool(ch_aug < -0.30 and k_fall >= need)))
    means = []
    for b in (0.0, 1.0, 2.0):
        g = p.loc[(b, 2048)]
        r = g.em_first_order / g.em_augmented
        means.append(float(r.mean()))
        if b > 0:
            rows.append(row("E35x", "P3", f"b={b:g}, M=2048: first/aug > 2 in all seeds", float(r.mean()), int((r > 2).sum()), len(r),
                            bool((r > 2).all()), f"augmented at cap in {int(cap.loc[(b, 2048)].sum())}"))
    rows.append(row("E35x", "P3", "M=2048: seed-mean first/aug rising over b = 0, 1, 2", means[-1], None, 3,
                    bool(means[0] < means[1] < means[2]), " < ".join(f"{v:.2f}" for v in means)))
    g = p.loc[(2.0, 2048)]
    fl = float(g.lmmse_true.mean() / g.em_augmented.mean())
    rows.append(row("E35x", "P3", "b=2, M=2048: floor / augmented (seed means) >= 3", fl, None, ns, fl >= 3))
    for b in (1.0, 2.0):
        for m in (512, 2048):
            g = p.loc[(b, m)]
            r = g.em_first_order / g.lmmse_true
            rows.append(row("E35x", "P4", f"b={b:g}, M={m}: first-order / floor > 1 in all seeds", float(r.mean()), int((r > 1).sum()), len(r),
                            bool((r > 1).all())))
    return rows


def read_e36x(d, floors):
    rows = []
    p = d.pivot_table(index=["stride", "seed"], columns="arm", values="risk")
    ns = p.index.get_level_values("seed").nunique()
    for arm in ("em_pair_quartic", "em_tilted_spline"):
        r1 = p.loc[1][arm]
        rows.append(row("E36x", "P1", f"stride 1: {arm} < 0.02 in all seeds", float(r1.mean()), int((r1 < 0.02).sum()), len(r1), bool((r1 < 0.02).all())))
        q = p.loc[4][arm] / r1
        rows.append(row("E36x", "P1", f"{arm}: stride-4 / stride-1 >= 2 in all seeds", float(q.mean()), int((q >= 2).sum()), len(q), bool((q >= 2).all())))
    s4 = p.loc[4]
    k = int((s4.em_mdn < s4.em_pair_quartic).sum())
    mdn4 = d[(d.stride == 4) & (d.arm == "em_mdn")]
    # at_cap is the driver's own flag (the selected checkpoint is the last one SAVED); the largest checkpoint seen in
    # the data would be wrong whenever no fit reached the ladder's end
    at_last = int(mdn4.at_cap.sum())
    need = max(1, int(np.ceil(12 * ns / 16)))
    unresolved = at_last > 4 * ns / 16
    rows.append(row("E36x", "P2", f"stride 4: MDN < site energy in >= {need}/{ns}", float((s4.em_mdn / s4.em_pair_quartic).mean()), k, ns,
                    None if unresolved else k >= need, f"MDN at its last checkpoint in {at_last}/{len(mdn4)} (largest selected: {int(mdn4.checkpoint.max()) if len(mdn4) else 0})"))
    for st, fl in sorted(floors.items()):
        mx = float(p.loc[st].em_mixture.mean())
        rows.append(row("E36x", "P3", f"stride {st}: mixture seed mean within 3% of the floor {fl:.4f}", mx / fl, None, ns, abs(mx / fl - 1) <= 0.03))
    return rows


def read_e38(d):
    rows = []
    inf = d[(d.kind == "infer") & (d.N_g == d[d.arm == "bp_grid"].N_g.mode().iloc[0])]
    bp = inf[inf.arm == "bp_grid"].set_index(["L", "B"]).increment_mb
    L0, B0 = int(inf.L.min()), int(sorted(inf.B.unique())[len(inf.B.unique()) // 2])
    for lab, sel in (("batch", bp.xs(L0, level="L")), ("length", bp.xs(B0, level="B"))):
        x, y = np.log(sel.index.values.astype(float)), np.log(sel.values)
        slope = float(np.polyfit(x, y, 1)[0]) if len(sel) > 1 and np.all(sel.values > 0) else float("nan")
        rows.append(row("E38", "P1", f"grid BP memory slope in the {lab}", slope, None, len(sel), None if np.isnan(slope) else bool(0.8 <= slope <= 1.2),
                        "MB: " + ", ".join(f"{v:.1f}" for v in sel.values)))
    op = inf[(inf.L == L0) & (inf.B == B0)].set_index("arm").increment_mb
    nets = op.drop("bp_grid")
    rows.append(row("E38", "P2", f"L={L0}, B={B0}: grid BP above every network", float(op["bp_grid"] / nets.max()), int((op["bp_grid"] > nets).sum()), len(nets),
                    bool((op["bp_grid"] > nets).all()), f"bp {op['bp_grid']:.1f} MB, largest network {nets.idxmax()} {nets.max():.1f} MB; method {d.method.iloc[0]}"))
    return rows


def e36_floors(path):
    o = pd.read_csv(path)
    o = o[(o.arm == "lmmse_true") & (o.level == -1)] if "level" in o else o[o.arm == "lmmse_true"]
    return {int(s): float(g.risk.mean()) for s, g in o.groupby("stride")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--e35", default="outputs/hpc_E35x_s*/augmented.csv")
    ap.add_argument("--e36", default="outputs/hpc_E36x_s*/clean.csv")
    ap.add_argument("--floors", default="outputs/E36/oracles.csv")
    ap.add_argument("--e38", default="outputs/hpc_E38/memory.csv")
    ap.add_argument("--out", default="reports/numbers")
    a = ap.parse_args()
    rows = []
    d35, d36 = _load(a.e35), _load(a.e36)
    if not d35.empty:
        rows += read_e35x(d35)
    if not d36.empty:
        rows += read_e36x(d36, e36_floors(ROOT / a.floors))
    d38 = _load(a.e38)
    if not d38.empty:
        rows += read_e38(d38)
    if not rows:
        print("nothing to read yet"); return 0
    R = pd.DataFrame(rows)
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / "replication_predictions.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 230, "display.max_colwidth", 70):
        print(R.to_string(index=False))
    print("wrote", out / "replication_predictions.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
