"""E33/E34's predeclared predictions, read off the frozen (or preview) tables, nothing else.

Every prediction in experiments/registry/E33.json (P1-P8) and E34.json (P2-P4) is a statement
about paired per-seed quantities on a cell at a size M. This tool computes exactly those
statements and nothing more: for each it prints the statistic the prediction names, the paired
count k/n of seeds on the predicted side, and a percentile bootstrap interval over seeds for the
paired ratio (20,000 draws, seed 20260912, as tools/make_tab_efficiency.paired_ci), and writes

    reports/numbers/field_predictions.csv      one row per (prediction, cell, M) with its verdict
    reports/numbers/field_pairs.csv            every paired ratio summarised (arm_a / arm_b per cell, M)

Verdicts are CONFIRMED / FALSIFIED / pending (no data yet) / partial (some cells missing). The
same reader runs on the laptop PREVIEW shards (`--preview`: two cells, two seeds, M <= 512) and
on the confirmatory shards; the output says which. Seeds are the resampling unit; the join is
on (cell, seed, n_chains, test_bundle), so a row that was scored on a different test bundle
cannot be paired by accident.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
B, SEED = 20_000, 20260912
TSWEEP = ["T0.3_h1_L32", "T0.5_iso_L32", "T1_iso_L32", "T2_iso_L32"]
REFINE = ["T0.3_h1_L32", "T0.3_h0.5_L64", "T0.3_h0.25_L128"]

CONFIRMATORY = dict(kernels="outputs/hpc_E33k_s*/kernels.csv", pair="outputs/hpc_E33q_s*/kernels.csv",
                    oracles="outputs/hpc_E33o/oracles.csv", nets="outputs/hpc_E33n_s*/nets.csv",
                    clean="outputs/hpc_E34c/clean.csv", learned="outputs/E34_learned*/learned.csv")  # physics_learned runs on the laptop
PREVIEW = dict(kernels="outputs/E33_preview_laptop/kernels.csv", pair="outputs/E33_preview_laptop_pair/kernels.csv",
               oracles="outputs/hpc_E33o/oracles.csv", nets="outputs/E33_preview_nets/nets.csv",
               clean="outputs/E34_preview_laptop*/clean.csv", learned="outputs/E34_preview_laptop_learned/learned.csv")


def load(pattern):
    paths = sorted(glob.glob(str(ROOT / pattern)))
    if not paths:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return mark_failed(df)


def mark_failed(df):
    """E33 amendment 3: a FAILED fit is a row with failed = 1 and risk NaN (shards written before
    the amendment carry no column and no NaN risk, so nothing in them is failed)."""
    if df.empty or "risk" not in df:
        return df
    df = df.copy()
    prior = df["failed"].fillna(0).astype(int) if "failed" in df else 0
    df["failed"] = ((prior == 1) | df["risk"].isna()).astype(int)
    return df


def nf(g, col="ratio"):
    """Failed fits in a group: a NaN ratio (either side failed) or a NaN risk."""
    return int(g[col].isna().sum())


def finalize_ok(R):
    """Amendment 3 (b): a failure can falsify a clause and never confirm one. A clause whose group
    holds a failed fit is not confirmed, whatever its mean says; pending stays pending."""
    R = R.copy()
    R["ok"] = [None if ok is None or (isinstance(ok, float) and np.isnan(ok)) else bool(ok and n == 0)
               for ok, n in zip(R["ok"], R["n_failed"])]
    return R


def paired_ci(values):
    values = np.asarray(values, float)
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, values.size, size=(B, values.size))
    boot = values[idx].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def risk_rows(df, arm, selection="risk", arch=None):
    if df.empty:
        return df
    m = (df.arm == arm) & (df.selection == selection) if "selection" in df else (df.arm == arm)
    if arch is not None:
        m &= df.arch == arch
    return df[m]


EMPTY_PAIR = pd.DataFrame({k: pd.Series(dtype=t) for k, t in (("cell", str), ("seed", int), ("n_chains", int), ("test_bundle", str),
                                                               ("risk_a", float), ("risk_b", float), ("ratio", float))})


def pair(a, b, keys=("cell", "seed", "n_chains", "test_bundle")):
    """Paired per-seed table of a.risk / b.risk, joined on the bundle so the test set is the same.
    An empty input gives an empty table WITH the columns, so every downstream groupby is a no-op."""
    if a.empty or b.empty:
        return EMPTY_PAIR.copy()
    keys = [k for k in keys if k in a and k in b]
    j = a[keys + ["risk"]].merge(b[keys + ["risk"]], on=keys, suffixes=("_a", "_b"))
    j["ratio"] = j.risk_a / j.risk_b  # NaN when either fit failed (amendment 3): counted, never a success
    return j


def summarise(j, label):
    out = []
    if j.empty:
        return out
    for (cell, n), g in j.groupby(["cell", "n_chains"]):
        lo, hi = paired_ci(g.ratio.dropna().values)
        out.append(dict(pair=label, cell=cell, n_chains=int(n), n_seeds=len(g), n_failed=nf(g), ratio=float(g.ratio.mean()), ci_lo=lo, ci_hi=hi,
                        k_below=int((g.ratio < 1).sum()), risk_a=float(g.risk_a.mean()), risk_b=float(g.risk_b.mean())))
    return out


def verdict(ok_flags):
    flags = [f for f in ok_flags if f is not None]
    if not flags:
        return "pending"
    if all(flags):
        return "CONFIRMED"
    if not any(flags):
        return "FALSIFIED"
    return "MIXED"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preview", action="store_true", help="read the laptop preview shards instead of the confirmatory ones")
    ap.add_argument("--out", default="reports/numbers")
    args = ap.parse_args()
    src = PREVIEW if args.preview else CONFIRMATORY
    tag = "preview" if args.preview else "confirmatory"
    K = load(src["kernels"]); Q = load(src["pair"]); O = load(src["oracles"]); N = load(src["nets"]); C = load(src["clean"]); Ld = load(src["learned"])
    allk = pd.concat([d for d in (K, Q) if not d.empty], ignore_index=True) if not (K.empty and Q.empty) else pd.DataFrame()
    print(f"[{tag}] kernels {len(K)} rows, pair {len(Q)}, oracles {len(O)}, nets {len(N)}, clean {len(C)}, learned {len(Ld)}")
    rows, pairs = [], []
    mix, poly, spl, mdn, pq = (risk_rows(allk, a) for a in ("em_mixture", "em_tilted_poly", "em_tilted_spline", "em_mdn", "em_pair_quartic"))
    # N-33 / E33 amendment 2: fits whose own invariant law sits outside the data range ("parked"),
    # counted per (arm, cell, size) and printed beside every verdict; nothing is excluded.
    parked = pd.DataFrame()
    tagged = [d.assign(source=src) for d, src in ((allk, "noisy"), (C, "clean")) if not d.empty]
    pk_src = pd.concat(tagged, ignore_index=True) if tagged else pd.DataFrame()
    if not pk_src.empty:
        from field_diagnostics import annotate
        from markov_score.bp_grid import make_grid
        gpk, wpk = make_grid(8.0, 401)
        sub = pk_src[(pk_src.selection == "risk") & pk_src.arm.isin(["em_tilted_poly", "em_tilted_spline", "em_pair_quartic"])]
        if not sub.empty:
            parked = annotate(sub, gpk, wpk).groupby(["source", "arm", "cell", "n_chains"]).agg(
                n=("parked", "size"), parked=("parked", "sum"), max_mass_out=("invariant_mass_out", "max"),
                eig_failed=("eig_failed", "sum"), max_gz_mass_out=("gz_mass_out", "max")).reset_index()

    # ---- P1: mixture / spline at 2048 above 1 in every seed; gap not closing from 512 to 2048
    j = pair(mix, spl); pairs += summarise(j, "em_mixture/em_tilted_spline")
    if not j.empty:
        for cell, g in j.groupby("cell"):
            g48, g512 = g[g.n_chains == 2048], g[g.n_chains == 512]
            if g48.empty:
                continue
            k, n = int((g48.ratio > 1).sum()), len(g48)
            nc = (g48.ratio.mean() >= 0.9 * g512.ratio.mean()) if not g512.empty else None
            rows.append(dict(prediction="P1", cell=cell, n_chains=2048, statistic="mixture/spline ratio; k(>1)/n; gap ratio 2048 vs 512",
                             value=float(g48.ratio.mean()), k=k, n=n, n_failed=nf(g48) + (nf(g512) if not g512.empty else 0), extra=float(g48.ratio.mean() / g512.ratio.mean()) if not g512.empty else np.nan,
                             ok=(k == n) and (nc if nc is not None else True)))
    # ---- P2: pair energy beats the mixture at every M >= 128; poly within 20% of spline at 2048
    for arm, d in (("em_tilted_poly", poly), ("em_tilted_spline", spl)):
        j = pair(d, mix); pairs += summarise(j, f"{arm}/em_mixture")
        for (cell, n), g in j[j.n_chains >= 128].groupby(["cell", "n_chains"]):
            k = int((g.ratio < 1).sum())
            rows.append(dict(prediction="P2", cell=cell, n_chains=int(n), statistic=f"{arm}/mixture; k(<1)/n", value=float(g.ratio.mean()), k=k, n=len(g), n_failed=nf(g), extra=np.nan, ok=(k == len(g))))
    j = pair(poly, spl); pairs += summarise(j, "em_tilted_poly/em_tilted_spline")
    for cell, g in j[j.n_chains == 2048].groupby("cell"):
        rows.append(dict(prediction="P2b", cell=cell, n_chains=2048, statistic="poly/spline at 2048 <= 1.2", value=float(g.ratio.mean()), k=int((g.ratio <= 1.2).sum()), n=len(g), n_failed=nf(g), extra=np.nan, ok=bool(g.ratio.mean() <= 1.2)))
    # ---- P3: mdn / mixture below 1.5 at 2048
    j = pair(mdn, mix); pairs += summarise(j, "em_mdn/em_mixture")
    for cell, g in j[j.n_chains == 2048].groupby("cell"):
        rows.append(dict(prediction="P3", cell=cell, n_chains=2048, statistic="mdn/mixture at 2048 < 1.5", value=float(g.ratio.mean()), k=int((g.ratio < 1.5).sum()), n=len(g), n_failed=nf(g), extra=np.nan, ok=bool(g.ratio.mean() < 1.5)))
    # ---- P4: no network reaches the spline at any M under either interface
    if not N.empty:
        for arm in ("ism_one_view", "dsm_one_corruption"):
            for arch in sorted(N.arch.unique()):
                nn = N[(N.arm == arm) & (N.arch == arch)]
                j = pair(nn, spl); pairs += summarise(j, f"{arm}:{arch}/em_tilted_spline")
                for (cell, n), g in j.groupby(["cell", "n_chains"]):
                    k = int((g.ratio > 1).sum())
                    rows.append(dict(prediction="P4", cell=cell, n_chains=int(n), statistic=f"{arm}:{arch}/spline; k(>1)/n", value=float(g.ratio.mean()), k=k, n=len(g), n_failed=nf(g), extra=np.nan, ok=(k == len(g))))
        # ---- P5: the window head's ISM risk at 2048 increases along the refinement cells; the spline's does not
        win = N[(N.arm == "ism_one_view") & (N.arch == "window") & (N.n_chains == 2048)].groupby("cell").risk.mean()
        sp = spl[spl.n_chains == 2048].groupby("cell").risk.mean()
        p5_failed = nf(spl[(spl.n_chains == 2048) & spl.cell.isin(REFINE)], "risk") if not spl.empty else 0
        if all(c in win.index for c in REFINE):
            w = [float(win[c]) for c in REFINE]
            s = [float(sp[c]) for c in REFINE] if all(c in sp.index for c in REFINE) else None
            rows.append(dict(prediction="P5", cell="refinement", n_chains=2048, statistic="window ISM risk h=1,0.5,0.25 increasing; spline not increasing",
                             value=w[-1] / w[0], k=int(all(a < b for a, b in zip(w, w[1:]))), n=1, n_failed=p5_failed, extra=(s[-1] / s[0]) if s else np.nan,
                             ok=bool(all(a < b for a, b in zip(w, w[1:])) and (s is None or not all(a < b for a, b in zip(s, s[1:]))))))
    # ---- P6: the floor at the base cell above 0.082, falling along the sweep
    if not O.empty:
        fl = O[O.arm == "lmmse_true"].groupby("cell").risk.mean()
        if all(c in fl.index for c in TSWEEP):
            v = [float(fl[c]) for c in TSWEEP]
            rows.append(dict(prediction="P6", cell="sweep", n_chains=2048, statistic="floor at base > 0.082 and monotone along Theta = 0.3, 0.5, 1, 2",
                             value=v[0], k=int(all(a > b for a, b in zip(v, v[1:]))), n=1, n_failed=0, extra=v[-1], ok=bool(v[0] > 0.082 and all(a > b for a, b in zip(v, v[1:])))))
    # ---- P7: pair energy within 10% of the spline at 2048; below it at 32 and 128
    j = pair(pq, spl); pairs += summarise(j, "em_pair_quartic/em_tilted_spline")
    for (cell, n), g in j.groupby(["cell", "n_chains"]):
        if n == 2048:
            rows.append(dict(prediction="P7", cell=cell, n_chains=2048, statistic="pair/spline at 2048 <= 1.1", value=float(g.ratio.mean()), k=int((g.ratio <= 1.1).sum()), n=len(g), n_failed=nf(g), extra=np.nan, ok=bool(g.ratio.mean() <= 1.1)))
        elif n in (32, 128):
            k = int((g.ratio < 1).sum())
            rows.append(dict(prediction="P7", cell=cell, n_chains=int(n), statistic="pair/spline; k(<1)/n", value=float(g.ratio.mean()), k=k, n=len(g), n_failed=nf(g), extra=np.nan, ok=(k == len(g))))
    # ---- P8: the temperature within 10% at 2048 (pair and spline); the barrier within 20% (pair, Theta <= 1)
    gc = pd.concat([d for d in (pq, spl) if not d.empty], ignore_index=True) if not (pq.empty and spl.empty) else pd.DataFrame()
    if not gc.empty and (gc.n_chains == 2048).any():
        from make_field_potential_figs import _readings
        from markov_score.bp_grid import make_grid
        sys.path.insert(0, str(ROOT / "experiments")); from frozen_config import FROZEN
        grid, _ = make_grid(FROZEN.half_width, FROZEN.n_grid)
        g48 = gc[gc.n_chains == 2048].copy(); g48["source"] = "noisy"
        failed48 = g48[g48.failed == 1].groupby(["cell", "arm"]).size()
        tab, _ = _readings(g48[g48.failed == 0], grid)
        for (cell, arm), g in tab.groupby(["cell", "arm"]):
            k = int((np.abs(g.theta_ratio - 1) <= 0.1).sum()); f8 = int(failed48.get((cell, arm), 0))
            rows.append(dict(prediction="P8", cell=cell, n_chains=2048, statistic=f"{arm}: |Theta_hat/Theta - 1| <= 0.1; k/n", value=float(g.theta_ratio.mean()), k=k, n=len(g) + f8, n_failed=f8, extra=float(g.theta_ratio.std()) if len(g) > 1 else np.nan, ok=(k == len(g))))
            if arm == "em_pair_quartic" and float(g["T"].iloc[0]) <= 1.0:
                k = int((np.abs(g.barrier_hat - 0.25) <= 0.05).sum())
                rows.append(dict(prediction="P8b", cell=cell, n_chains=2048, statistic="pair: |barrier - 1/4| <= 0.05; k/n", value=float(g.barrier_hat.mean()), k=k, n=len(g) + f8, n_failed=f8, extra=float(g.well_hat.mean()), ok=(k == len(g))))
    # ---- E34 P2/P3: the clean-data interface
    if not C.empty and not allk.empty:
        for arm in sorted(set(C.arm) & set(allk.arm)):
            j = pair(risk_rows(C, arm), risk_rows(allk, arm)); pairs += summarise(j, f"clean:{arm}/noisy:{arm}")
            for (cell, n), g in j.groupby(["cell", "n_chains"]):
                k = int((g.ratio < 1).sum())
                rows.append(dict(prediction="E34-P2", cell=cell, n_chains=int(n), statistic=f"clean/noisy {arm}; k(<1)/n", value=float(g.ratio.mean()), k=k, n=len(g), n_failed=nf(g), extra=float(g.risk_a.mean()), ok=(k == len(g))))
        for arm in ("em_tilted_poly", "em_tilted_spline", "em_pair_quartic"):
            c48 = risk_rows(C, arm); c48 = c48[c48.n_chains == 2048]
            for cell, g in c48.groupby("cell"):
                rows.append(dict(prediction="E34-P2b", cell=cell, n_chains=2048, statistic=f"clean {arm} at 2048 < 0.02", value=float(g.risk.mean()), k=int((g.risk < 0.02).sum()), n=len(g), n_failed=nf(g, "risk"), extra=np.nan, ok=bool(g.risk.mean() < 0.02)))
        j = pair(risk_rows(C, "em_mixture"), risk_rows(C, "em_tilted_spline")); pairs += summarise(j, "clean:em_mixture/clean:em_tilted_spline")
        for cell, g in j[j.n_chains == 2048].groupby("cell"):
            k = int((g.ratio > 1).sum())
            rows.append(dict(prediction="E34-P3", cell=cell, n_chains=2048, statistic="clean mixture/spline at 2048; k(>1)/n", value=float(g.ratio.mean()), k=k, n=len(g), n_failed=nf(g), extra=np.nan, ok=(k == len(g))))
    # ---- E34 P4: generated samples' kink density and mean domain within 15% (spline); mixture overshoots
    if not Ld.empty:
        # several physics_learned runs (the pair arm first, the kernel arms after) each carry the same
        # deterministic truth rows: keep one per (cell, seed), after checking that they agree
        T = Ld[Ld.arm == "truth"]
        spread = T.groupby(["cell", "seed"]).kink_density.agg(lambda v: float(v.max() - v.min()))
        if (spread > 1e-12).any():
            raise SystemExit(f"physics_learned runs disagree on the truth rows: {spread[spread > 1e-12].to_dict()}")
        Ld = pd.concat([Ld[Ld.arm != "truth"], T.drop_duplicates(["cell", "seed"])], ignore_index=True)
        tr = Ld[Ld.arm == "truth"].set_index(["cell", "seed"])
        for arm in ("em_tilted_spline", "em_mixture", "em_pair_quartic", "em_tilted_poly", "em_mdn"):
            a = Ld[(Ld.arm == arm) & (Ld.selection == "risk")]
            if a.empty:
                continue
            for cell, g in a.groupby("cell"):
                rk = np.array([float(r.kink_density) / float(tr.loc[(cell, r.seed)].kink_density) for r in g.itertuples()])
                rd = np.array([float(r.mean_domain) / float(tr.loc[(cell, r.seed)].mean_domain) for r in g.itertuples()])
                ok = bool(np.all(np.abs(rk - 1) <= 0.15) and np.all(np.abs(rd - 1) <= 0.15)) if arm == "em_tilted_spline" else (bool(np.all(rk > 1)) if arm == "em_mixture" else None)
                rows.append(dict(prediction="E34-P4", cell=cell, n_chains=int(g.n_chains.iloc[0]), statistic=f"{arm}: kink ratio, domain ratio to truth", value=float(np.nanmean(rk)) if np.isfinite(rk).any() else np.nan, k=int(np.isfinite(rk).sum()), n=len(g),
                                 n_failed=int(np.isnan(rk).sum()), extra=float(np.nanmean(rd)) if np.isfinite(rd).any() else np.nan, ok=ok))
    out = ROOT / args.out; out.mkdir(parents=True, exist_ok=True)
    R = pd.DataFrame(rows)
    if not R.empty:
        R = finalize_ok(R)
        R.insert(0, "data", tag)
        R["verdict"] = R.ok.map(lambda v: "pending" if v is None else ("CONFIRMED" if v else "FALSIFIED"))
        R.to_csv(out / f"field_predictions{'_preview' if args.preview else ''}.csv", index=False, float_format="%.4f")
        with pd.option_context("display.width", 220, "display.max_rows", 500):
            print(R.drop(columns=["ok"]).to_string(index=False))
        print("\nper prediction:", {p: verdict(list(g.ok)) for p, g in R.groupby("prediction")})
    if not parked.empty:
        parked.insert(0, "data", tag)
        parked.to_csv(out / f"field_parked{'_preview' if args.preview else ''}.csv", index=False, float_format="%.3g")
        tot = parked.groupby(["source", "arm"])[["n", "parked", "eig_failed"]].sum()
        print("\nparked fits (first-site law beyond |a| = 3.5), and fits whose eigensolve failed (estimator used N(0, 1)), per source and arm:")
        print(tot.to_string())
    fails = pk_src[pk_src.get("failed", pd.Series(0, index=pk_src.index)) == 1] if not pk_src.empty else pd.DataFrame()
    print(f"\nfailed fits (E33 amendment 3): {0 if fails.empty else len(fails[fails.selection == 'risk'])} risk-selected rows")
    if not fails.empty:
        ft = fails[fails.selection == "risk"].groupby(["source", "arm", "cell", "n_chains"]).size().rename("failed").reset_index()
        ft.insert(0, "data", tag)
        ft.to_csv(out / f"field_failed{'_preview' if args.preview else ''}.csv", index=False)
        print(ft.to_string(index=False))
    P = pd.DataFrame(pairs)
    if not P.empty:
        P.insert(0, "data", tag)
        P.to_csv(out / f"field_pairs{'_preview' if args.preview else ''}.csv", index=False, float_format="%.4f")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
