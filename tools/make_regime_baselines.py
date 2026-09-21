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

  divergence, by a rule fixed before the data (audit session, 21 Sep): an ISM cell is
  EXCLUDED from the *Conv* factors when its selected held-out ISM objective is not below the
  objective of the isotropic score s(x) = -x on the same held-out views, mean over the views
  of 1/2|x|^2 - L (expectation -L/2). That is the criterion the arm selects on, so a fit that
  cannot beat the trivial score on its own criterion is not a trained network; E03's healthy
  window curves sit near -27.6 at L = 32, the void E14 curves sat at -10 to +53. The views
  are regenerated here from the driver's own generators (deterministic in rho, L, seed), and
  the objective is computed on them, not from its expectation. A second, weaker signal -- a
  curve whose minimum is at its first checkpoint and whose last value exceeds its first --
  is REPORTED as a flag (dagger) but does not exclude, because it also fires on a fit that
  converged early and then overfitted, a legitimate optimum. The DSM arm selects on the
  held-out BP risk, not on an ISM objective, so only the flag applies to it;

  the agreement of the two EM selection rules per cell: the fraction of seeds where the
  held-out BP risk and the held-out log-evidence chose the same rung, and the largest
  difference in test risk between them. The regime statement is read on ism_one_view
  against em_bp_evidence (both oracle-free), with dsm_one_corruption as the
  network-favourable bracket.

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
import sys  # noqa: E402
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from e14_regime_baselines import FAMILIES, _one_view_groups  # noqa: E402  (numpy only)
from markov_score.utils import rng_for  # noqa: E402

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


def iso_ism_objective(family: str, rho: float, L: int, seed: int, n_val: int) -> float:
    """Mean over the run's held-out one-view samples of 1/2|x|^2 - L: the implicit
    score-matching objective of s(x) = -x, on exactly the views the network arm selected on
    (same innovation family, same generators, same tags, same split as the driver's _common
    and _one_view_groups). The family is read from the run's params, never assumed: E14b runs
    the bimodal member, and the threshold on Laplace views would be the wrong threshold."""
    prior = FAMILIES[family](rho)
    rng_va = rng_for("e14-valchains", seed, L, rho)
    Av = np.stack([prior.sample(rng_va, L) for _ in range(n_val)])
    tot, n = 0.0, 0
    for X, _t in _one_view_groups(Av, "e14-emval", seed, L):
        tot += float((0.5 * (X**2).sum(axis=1) - L).sum()); n += X.shape[0]
    return tot / n


def shape_flag(curve: str) -> bool:
    """A held-out curve 'step:value;...' whose minimum is its first checkpoint and whose last
    value exceeds its first. Reported, never used to exclude."""
    if not isinstance(curve, str) or not curve:
        return False
    vals = [float(p.split(":")[1].split("/")[0]) for p in curve.split(";") if ":" in p]
    return len(vals) > 1 and min(vals) >= vals[0] and vals[-1] > vals[0]


def _tag(rho, L):
    return f"{RHO_TAG[rho]}{LEN_TAG[L]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "outputs" / "E14")
    ap.add_argument("--dest", type=Path, default=ROOT / "paper" / "sections")
    ap.add_argument("--prefix", default=None, help="macro prefix; default rb for Laplace, rb<Family> otherwise")
    ap.add_argument("--stem", default=None, help="fragment stem; default regime-baselines, or regime-baselines-<family>")
    a = ap.parse_args()
    dest = a.dest if "dev" not in a.src.name and "smoke" not in a.src.name else ROOT / "outputs" / "scratch"
    dest.mkdir(parents=True, exist_ok=True)

    import json
    settings = json.load(open(a.src / "params_confirm.json"))["settings"]
    n_val = int(settings["n_val"])
    family = str(settings.get("family", "laplace"))       # drivers before eddf372 had no setting
    prefix = a.prefix or ("rb" if family == "laplace" else "rb" + family.capitalize())
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
            macros[f"{prefix}Em{sfx}{tag}"] = f"{rec[arm]:.3f}"
            # the ladder limit: seeds whose selected checkpoint is the last rung. Where it binds
            # the EM error is an upper bound on what a longer ladder would give, and every
            # factor against it is conservative.
            cap = g[g.arm == arm].at_cap.astype(bool)
            rec[f"{arm}_atcap"] = int(cap.sum())
            macros[f"{prefix}Em{sfx}AtCap{tag}"] = f"{int(cap.sum())} of {len(cap)}"
            macros[f"{prefix}Em{sfx}AtCapFrac{tag}"] = f"{cap.mean():.2f}"
        macros[f"{prefix}Lmmse{tag}"] = f"{rec['lmmse']:.3f}"; macros[f"{prefix}Iso{tag}"] = f"{rec['iso']:.3f}"
        macros[f"{prefix}Tstar{tag}"] = f"{rec['t_star']:.2f}"; macros[f"{prefix}Kurt{tag}"] = f"{rec['kurt']:.2f}"

        em = g[g.arm == "em_bp"][key + ["risk", "checkpoint"]].rename(columns={"risk": "risk_em", "checkpoint": "ck_em"})
        emv = g[g.arm == "em_bp_evidence"][key + ["risk", "checkpoint"]].rename(columns={"risk": "risk_emv", "checkpoint": "ck_emv"})
        ee = em.merge(emv, on=key)
        rec["em_same_rung"] = int((ee.ck_em == ee.ck_emv).sum()); rec["em_maxdiff"] = float((ee.risk_em - ee.risk_emv).abs().max())
        macros[f"{prefix}EmAgreeRung{tag}"] = f"{rec['em_same_rung']} of {len(ee)}"
        macros[f"{prefix}EmAgreeMaxDiff{tag}"] = f"{rec['em_maxdiff']:.3f}"
        # the isotropic score's ISM objective on this cell's own held-out views, per seed
        iso = {sd: iso_ism_objective(family, rho, L, int(sd), n_val) for sd in g.seed.unique()}
        rec["iso_obj"] = float(np.mean(list(iso.values())))
        macros[f"{prefix}IsoObj{tag}"] = f"{rec['iso_obj']:.1f}"
        for arm, pfx in NET_ARMS.items():
            sub = g[g.arm == arm]
            per = sub.groupby("arch").risk.mean().sort_values()
            best = per.index[0]
            rec[f"{arm}_arch"] = best; rec[f"{arm}_best"] = per.iloc[0]
            macros[f"{prefix}{pfx}Best{tag}"] = f"{per.iloc[0]:.3f}"
            macros[f"{prefix}{pfx}BestArch{tag}"] = ARCH_NAME.get(best, best)
            for arch in archs:
                macros[f"{prefix}{pfx}{arch.capitalize()}{tag}"] = f"{per.get(arch, np.nan):.3f}"
            sb = sub[sub.arch == best].copy()
            sb["flag"] = sb.val_curve.map(shape_flag)
            if arm == "ism_one_view":
                # exclusion: selected ISM objective not below the trivial score's, same views
                sb["excluded"] = sb.apply(lambda r: r.val_selected >= iso[r.seed], axis=1)
                macros[f"{prefix}IsmObjBest{tag}"] = f"{sb.val_selected.mean():.1f}"
            else:
                sb["excluded"] = False
            n_flag, n_exc = int(sb.flag.sum()), int(sb.excluded.sum())
            rec[f"{arm}_flag"] = n_flag; rec[f"{arm}_excluded"] = n_exc
            macros[f"{prefix}{pfx}Flag{tag}"] = str(n_flag); macros[f"{prefix}{pfx}Diverged{tag}"] = str(n_exc)
            j = sb.merge(em, on=key).merge(emv, on=key)
            if len(j) != rec["n_seeds"]:
                raise SystemExit(f"E14: pairing lost seeds at rho={rho} L={L} arm={arm}")
            for emcol, esfx in (("risk_em", ""), ("risk_emv", "Ev")):
                for subset, ssfx in ((j, ""), (j[~j.excluded], "Conv")):
                    if len(subset) == 0:
                        continue
                    ratio = subset.risk / subset[emcol]
                    lo, hi = boot_ci(ratio.values)
                    m = f"{prefix}{pfx}Ratio{esfx}{ssfx}{tag}"
                    macros[m] = f"{ratio.mean():.1f}"; macros[m + "lo"] = f"{lo:.1f}"; macros[m + "hi"] = f"{hi:.1f}"
                    if ssfx == "":     # the per-seed range, so the appendix can place a cell against E03's seeds
                        macros[f"{prefix}{pfx}Ratio{esfx}Min{tag}"] = f"{ratio.min():.1f}"
                        macros[f"{prefix}{pfx}Ratio{esfx}Max{tag}"] = f"{ratio.max():.1f}"
                    macros[f"{prefix}{pfx}Wins{esfx}{ssfx}{tag}"] = str(int((ratio > 1).sum()))
                    macros[f"{prefix}{pfx}N{esfx}{ssfx}{tag}"] = str(len(subset))
                    if emcol == "risk_emv" and ssfx == "":
                        rec[f"{arm}_ratio"] = ratio.mean(); rec[f"{arm}_lo"] = lo; rec[f"{arm}_hi"] = hi
                        rec[f"{arm}_wins"] = int((ratio > 1).sum())
                    if emcol == "risk_emv" and ssfx == "Conv":
                        rec[f"{arm}_ratio_conv"] = ratio.mean(); rec[f"{arm}_lo_conv"] = lo; rec[f"{arm}_hi_conv"] = hi
            # the best network against the LMMSE floor, paired per seed on the same test bundle:
            # "below" / "above" when the bootstrap interval of the paired difference clears zero,
            # "at" otherwise. A network above the floor at rho = 0.95 is a finding, not noise.
            fl = g[g.arm == "lmmse_true"][key + ["risk"]].rename(columns={"risk": "risk_floor"})
            jf = sb.merge(fl, on=key)
            d = (jf.risk - jf.risk_floor).values
            dlo, dhi = boot_ci(d)
            word = "below" if dhi < 0 else ("above" if dlo > 0 else "at")
            rec[f"{arm}_vs_floor"] = word
            macros[f"{prefix}{pfx}VsFloor{tag}"] = word
            macros[f"{prefix}{pfx}OverLmmse{tag}"] = f"{jf.risk.mean() / jf.risk_floor.mean():.2f}"
            macros[f"{prefix}{pfx}MinusFloor{tag}"] = f"{d.mean():+.3f}"
            macros[f"{prefix}{pfx}MinusFloor{tag}lo"] = f"{dlo:+.3f}"; macros[f"{prefix}{pfx}MinusFloor{tag}hi"] = f"{dhi:+.3f}"
        rows.append(rec)

    t = pd.DataFrame(rows).sort_values(["rho", "L"])
    macros[f"{prefix}Cells"] = str(len(t)); macros[f"{prefix}Seeds"] = str(int(t.n_seeds.min()))
    macros[f"{prefix}RhoLo"] = f"{t.rho.min():g}"; macros[f"{prefix}RhoHi"] = f"{t.rho.max():g}"
    macros[f"{prefix}LenLo"] = str(int(t.L.min())); macros[f"{prefix}LenHi"] = str(int(t.L.max()))
    macros[f"{prefix}IsmRatioMin"] = f"{t.ism_one_view_ratio.min():.1f}"; macros[f"{prefix}IsmRatioMax"] = f"{t.ism_one_view_ratio.max():.1f}"
    macros[f"{prefix}IsmAllAboveOne"] = "all" if (t.ism_one_view_lo > 1).all() else f"{int((t.ism_one_view_lo > 1).sum())} of {len(t)}"
    macros[f"{prefix}IsmDivergedTotal"] = str(int(t.ism_one_view_excluded.sum()))
    macros[f"{prefix}IsmFlagTotal"] = str(int(t.ism_one_view_flag.sum()))
    macros[f"{prefix}EmAgreeRungTotal"] = f"{int(t.em_same_rung.sum())} of {int((t.n_seeds).sum())}"
    macros[f"{prefix}EmAgreeMaxDiffAll"] = f"{t.em_maxdiff.max():.3f}"
    macros[f"{prefix}EmAtCapTotal"] = f"{int(t.em_bp_atcap.sum())} of {int(t.n_seeds.sum())}"
    macros[f"{prefix}EmEvAtCapTotal"] = f"{int(t.em_bp_evidence_atcap.sum())} of {int(t.n_seeds.sum())}"
    macros[f"{prefix}IsmAboveFloorCells"] = ", ".join(f"({r.rho:g}, {int(r.L)})" for _, r in t.iterrows() if r.ism_one_view_vs_floor == "above") or "none"
    macros[f"{prefix}IsmBelowFloorCells"] = ", ".join(f"({r.rho:g}, {int(r.L)})" for _, r in t.iterrows() if r.ism_one_view_vs_floor == "below") or "none"
    # the same cell E03 measured, so the two experiments can be read against each other
    fp = t[(t.rho == 0.85) & (t.L == 32)]
    if len(fp):
        macros[f"{prefix}FrozenIsmRatio"] = f"{float(fp.ism_one_view_ratio.iloc[0]):.1f}"
        macros[f"{prefix}FrozenEm"] = f"{float(fp.em_bp.iloc[0]):.3f}"

    stem = a.stem or ("regime-baselines" if family == "laplace" else f"regime-baselines-{family}")
    macros[f"{prefix}Family"] = family
    (dest / f"{stem}-numbers.tex").write_text(
        "%% GENERATED by tools/make_regime_baselines.py -- do not hand-edit.\n"
        + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")

    lines = [r"%% GENERATED by tools/make_regime_baselines.py -- do not hand-edit.",
             r"\begin{tabular}{llrrrrrl}", r"\toprule",
             r"$\corr$ & $L$ & \textsc{lmmse} & EM--BP (ev.) & EM--BP (bp) & best net (C) & best net (B) & factor C / EM--BP (ev.) \\",
             r"\midrule"]
    for _, r in t.iterrows():
        marks = ""
        if r.ism_one_view_excluded:
            marks += f"$^{{*{int(r.ism_one_view_excluded)}}}$"
        if r.ism_one_view_flag:
            marks += f"$^{{\\dagger {int(r.ism_one_view_flag)}}}$"
        lines.append(f"{r.rho:g} & {int(r.L)} & {r.lmmse:.3f} & {r.em_bp_evidence:.3f} & {r.em_bp:.3f} & "
                     f"{r.ism_one_view_best:.3f} & {r.dsm_one_corruption_best:.3f} & "
                     f"${r.ism_one_view_ratio:.1f}$ [{r.ism_one_view_lo:.1f}, {r.ism_one_view_hi:.1f}]{marks}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"%% * n: n ISM cells excluded from the Conv factors (selected ISM objective not below the trivial score's on the same views);",
              r"%% dagger n: n cells whose held-out curve never improved on its first checkpoint (reported, not excluded)."]
    (dest / f"{stem}-table.tex").write_text("\n".join(lines) + "\n")

    cols = ["rho", "L", "kurt", "lmmse", "em_bp", "em_bp_evidence", "em_same_rung", "em_maxdiff", "iso_obj",
            "ism_one_view_arch", "ism_one_view_best", "ism_one_view_ratio", "ism_one_view_lo", "ism_one_view_hi",
            "ism_one_view_wins", "ism_one_view_excluded", "ism_one_view_flag", "ism_one_view_vs_floor",
            "em_bp_atcap", "em_bp_evidence_atcap", "dsm_one_corruption_best", "dsm_one_corruption_ratio"]
    print(t[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if t.ism_one_view_excluded.sum():
        print(f"\nNOTE: {int(t.ism_one_view_excluded.sum())} ISM cell(s) excluded from the Conv factors (objective not below the trivial score's).")
    if t.ism_one_view_flag.sum():
        print(f"NOTE: {int(t.ism_one_view_flag.sum())} ISM cell(s) carry the curve-shape flag (reported, not excluded).")
    print("ratios in the table and rec fields are against em_bp_evidence; *Ev*-less macros are against em_bp")
    print(f"family={family} prefix={prefix}; wrote {dest/(stem+'-numbers.tex')} ({len(macros)} macros) and {dest/(stem+'-table.tex')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
