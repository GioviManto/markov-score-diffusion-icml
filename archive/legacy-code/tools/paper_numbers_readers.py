"""Readers for tools/make_paper_numbers.py: every raw statistic behind the field, compute and limits macros.

Each `read_*` function takes a `Sources` (a cached, frozen-checked loader of registered outputs) and returns
{macro name: raw value} (floats, ints or strings); the formatting into TeX lives in make_paper_numbers.py.
The pinned readings of E33, E42-E50 are IMPORTED from tools/make_field_numbers.py, tools/make_extension_numbers.py and
tools/make_field_potential_figs.py wherever they exist (pair, risk_rows, e50_crossings, e50_kink_fits,
load_e42_table, _readings); the rest are computed here from the frozen CSVs and nothing else.

A statistic over seeds is, unless a docstring says otherwise, "mean over seeds of the risk" for a level and "ratio of seed
means" for a ratio of two arms; where a registry clause fixes the other convention ("mean of per-seed ratios", paired by
seed) the function says so.
"""
from __future__ import annotations

import glob
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "tools", ROOT / "experiments", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import make_extension_numbers as X  # noqa: E402
import make_field_numbers as F  # noqa: E402

ORDER = ["T0.3_h1_L32", "T0.5_iso_L32", "T1_iso_L32", "T2_iso_L32", "T0.3_h0.5_L64", "T0.3_h0.25_L128"]
BASE, FINE = ORDER[0], ORDER[-1]
COLD3 = ["T0.3_h1_L32", "T0.3_h0.5_L64", "T0.3_h0.25_L128"]          # the three Theta = 0.3 cells
ARMS_SCORE = [("em_pair_quartic", "site energy (5)"), ("em_tilted_spline", "spline tilt (25)"), ("em_tilted_poly", "Legendre tilt (7)"),
              ("em_mdn", "MDN"), ("em_mixture", "mixture (paper)"), ("ism:window", "window, ISM"), ("ism:transformer", "transformer, ISM"),
              ("dsm:window", "window, DSM"), ("dsm:transformer", "transformer, DSM")]
NET_ARMS = [a for a, _ in ARMS_SCORE if ":" in a]
BARRIER, BAND = 0.25, 0.05            # the kink barrier 1/4 of the quartic double well, and E33's registered band (within 20% of it)


class Sources:
    """Loads registered outputs once, remembers every file it read (for the frozen check), memoises readers."""

    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        self.read_files: list[str] = []
        self._csv: dict[str, pd.DataFrame] = {}
        self._memo: dict[str, object] = {}

    def csv(self, pattern: str) -> pd.DataFrame:
        if pattern not in self._csv:
            paths = sorted(glob.glob(str(self.root / pattern)))
            if not paths:
                raise SystemExit(f"make_paper_numbers: missing registered output {pattern}")
            self.read_files += [str(Path(p).relative_to(self.root)) for p in paths]
            self._csv[pattern] = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
        return self._csv[pattern].copy()

    def memo(self, key, fn):
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]


# ------------------------------------------------------------------------------------------------ small helpers
def _seed_mean(d: pd.DataFrame, by, col="risk"):
    return d.groupby(by)[col].mean()


def _need_cols(d, *cols):
    miss = [c for c in cols if c not in d]
    if miss:
        raise SystemExit(f"make_paper_numbers: columns {miss} missing from a registered output")


# ------------------------------------------------------------------------------------------------ E32 (the cells)
def e32(S: Sources) -> dict:
    return S.memo("e32", lambda: dict(ladder=S.csv("outputs/E32/ladder.csv"), iso=S.csv("outputs/E32/iso.csv"),
                                      fixed=S.csv("outputs/E32/fixed_h.csv"), refine=S.csv("outputs/E32/refinement.csv")))


def cells_table_data(S: Sources) -> list[dict]:
    """One row per cell of E32's ladder: Theta, h (the solved h on the iso cells), L, xi in sites, excess kurtosis, the linear
    floor (pooled lmmse_true, level -1, mean of the two reps) and the isotropic risk (same)."""
    L = e32(S)["ladder"]
    iso = e32(S)["iso"]
    pooled = L[L.level == -1].groupby(["cell", "arm"]).risk.mean().unstack()
    rows = []
    for c in ORDER:
        g = L[L.cell == c].iloc[0]
        h = float(g.h)
        if "iso" in c:                       # the solved h is the iso table's, and must agree with the cell record
            h_iso = float(iso[np.isclose(iso["T"], g["T"])].h.iloc[0])
            assert abs(h_iso - h) < 1e-9, (c, h, h_iso)
        rows.append(dict(cell=c, T=float(g["T"]), h=h, L=int(g.n_sites), xi=float(g.corr_length_sites), kurt=float(g.excess_kurtosis),
                         floor=float(pooled.loc[c, "lmmse_true"]), iso=float(pooled.loc[c, "isotropic"]), solved=("iso" in c)))
    return rows


def read_e32(S: Sources) -> dict:
    d = e32(S)
    L, F_, R = d["ladder"], d["fixed"], d["refine"]
    out = {}
    out["fldOracleGridWorst"] = float(R[(R.grid_size == 401) & (R.half_width == 8.0)].rel_error.max())
    pr = L[L.arm == "projected"].groupby(["cell", "rep"]).edge_mass.max().groupby("cell").max()
    out["fldProjEdgeBase"] = float(pr[BASE])
    out["fldProjEdgeOthersMin"] = float(pr.drop(BASE).min())
    cells = {r["cell"]: r for r in cells_table_data(S)}
    out["fldXiHone"] = cells[BASE]["xi"]
    out["fldXiHzeroptwofive"] = cells[FINE]["xi"]
    fx = F_[F_.kind == "ladder"]
    phys = F_[F_.kind == "physics"].groupby("cell")
    xi = F_.groupby("cell").corr_length_sites.first()
    out["fldXiHot"], out["fldXiCold"] = float(xi["T2_h1_L32"]), float(xi["T0.2_h1_L32"])
    bim = phys.bimodality_coefficient.mean()
    out["fldBimodHot"], out["fldBimodCold"] = float(bim["T2_h1_L32"]), float(bim["T0.2_h1_L32"])
    fl = fx[(fx.level == -1) & (fx.arm == "lmmse_true")].groupby("cell").risk.mean()
    out["fldFloorFixedHHot"], out["fldFloorFixedHCold"] = float(fl["T2_h1_L32"]), float(fl["T0.2_h1_L32"])
    return out


# ------------------------------------------------------------------------------------------------ E33 (the score comparison)
def e33(S: Sources) -> dict:
    def build():
        K = F.mark_failed(S.csv("outputs/hpc_E33k_s*/kernels.csv"))
        Q = F.mark_failed(S.csv("outputs/hpc_E33q_s*/kernels.csv"))
        N = F.mark_failed(S.csv("outputs/hpc_E33n_s*/nets.csv"))
        O = S.csv("outputs/hpc_E33o/oracles.csv")
        floor = O[O.arm == "lmmse_true"].groupby("cell").risk.mean()
        allk = pd.concat([K, Q], ignore_index=True)
        return dict(K=K, Q=Q, N=N, O=O, floor=floor, allk=allk)
    return S.memo("e33", build)


def score_table_data(S: Sources) -> dict:
    """The conf-score.tex numbers (tools/make_field_confirmatory_tables.score_table): per (M, arm, cell) the seed-mean risk over the
    linear floor, and the number of failed fits (NaN risk) excluded from that mean."""
    def build():
        d = e33(S)
        K = d["allk"]
        K = K[K.selection == "risk"][["cell", "n_chains", "arm", "risk"]]
        N = d["N"]
        N = N.assign(arm=N.arm.str.replace("ism_one_view", "ism").str.replace("dsm_one_corruption", "dsm") + ":" + N.arch)[["cell", "n_chains", "arm", "risk"]]
        A = pd.concat([K, N], ignore_index=True)
        mean = A.groupby(["n_chains", "arm", "cell"]).risk.mean()
        nfail = A.groupby(["n_chains", "arm", "cell"]).risk.apply(lambda r: int(r.isna().sum()))
        ratio, fails = {}, {}
        for m in (128, 2048):
            for arm, _ in ARMS_SCORE:
                for c in ORDER:
                    f = int(nfail.get((m, arm, c), 0))
                    v = float(mean.get((m, arm, c), np.nan))
                    fails[(m, arm, c)] = f
                    ratio[(m, arm, c)] = np.nan if f == 16 else v / float(d["floor"][c])
        return dict(ratio=ratio, fails=fails)
    return S.memo("score_table", build)


def read_e33(S: Sources) -> dict:
    d = e33(S)
    T = score_table_data(S)["ratio"]
    allk, N = d["allk"], d["N"]
    out = {}
    site = [T[(2048, "em_pair_quartic", c)] for c in ORDER]
    mix = [T[(2048, "em_mixture", c)] for c in ORDER]
    out["fldSiteFloorMin"], out["fldSiteFloorMax"] = min(site), max(site)
    out["fldMixFloorMin"], out["fldMixFloorMax"] = min(mix), max(mix)
    site_r, spl, mix_r = (F.risk_rows(allk, a) for a in ("em_pair_quartic", "em_tilted_spline", "em_mixture"))
    j = F.pair(site_r, spl)                        # paired by (cell, seed, n_chains, test bundle); ratio = site / spline
    g = j[j.n_chains == 2048].groupby("cell").ratio.mean()
    out["fldSiteOverSplineMin"], out["fldSiteOverSplineMax"] = float(g.min()), float(g.max())
    jm = F.pair(mix_r, spl)
    gb = jm[jm.cell == BASE].groupby("n_chains").ratio.mean()
    out["fldMixSplineBaseA"], out["fldMixSplineBaseB"] = float(gb[512]), float(gb[2048])
    # P4: networks over the spline, paired, every cell, size, interface, architecture
    ratios = []
    for arm in ("ism_one_view", "dsm_one_corruption"):
        for arch in ("window", "transformer"):
            jj = F.pair(N[(N.arm == arm) & (N.arch == arch)], spl)
            ratios += list(jj.groupby(["cell", "n_chains"]).ratio.mean().values)
    out["fldNetOverSplineMin"] = float(min(ratios))
    # the introduction's range: best network over the site energy at M = 2048, per cell
    n48 = N[N.n_chains == 2048].groupby(["cell", "arm", "arch"]).risk.mean().groupby("cell").min()
    s48 = site_r[site_r.n_chains == 2048].groupby("cell").risk.mean()
    r = (n48 / s48).reindex(ORDER)
    out["fldNetRatioMin"], out["fldNetRatioMax"] = float(r.min()), float(r.max())
    # the window head along the refinement
    win = N[(N.arm == "ism_one_view") & (N.arch == "window") & (N.n_chains == 2048)].groupby("cell").risk.mean()
    spl_m = spl[spl.n_chains == 2048].groupby("cell").risk.mean()
    out["fldWindowRiskHone"], out["fldWindowRiskHzeroptwofive"] = float(win[BASE]), float(win[FINE])
    out["fldWindowSplineHone"], out["fldWindowSplineHzeroptwofive"] = float(win[BASE] / spl_m[BASE]), float(win[FINE] / spl_m[FINE])
    out["fldWindowFloorHone"], out["fldWindowFloorHzeroptwofive"] = T[(2048, "ism:window", BASE)], T[(2048, "ism:window", FINE)]
    return out


def e33_pair_readings(S: Sources) -> pd.DataFrame:
    """E33's site-energy fits at M = 2048 read as physics (make_field_potential_figs._readings), one row per good fit."""
    def build():
        from make_field_potential_figs import _readings
        from markov_score.bp_grid import make_grid
        from frozen_config import FROZEN
        Q = e33(S)["Q"]
        q = Q[(Q.selection == "risk") & (Q.n_chains == 2048)].copy()
        q["source"] = "noisy"
        grid, _ = make_grid(FROZEN.half_width, FROZEN.n_grid)
        tab, _ = _readings(q[q.failed == 0], grid)
        return tab
    return S.memo("pair_readings", build)


def read_e33_physics(S: Sources) -> dict:
    tab = e33_pair_readings(S)
    n_theta = tab.groupby("cell").size()
    assert (n_theta.values == 16).all() and len(n_theta) == 6, "E33 P8: every cell must have 16 good site-energy fits at M = 2048"
    cold = tab[tab["T"] <= 1.0]                                    # the five cells where the registered barrier clause reads
    out = {"fldThetaFits": len(tab), "fldThetaOkFits": int((np.abs(tab.theta_ratio - 1) <= 0.10).sum()),
           "fldBarrierFits": len(cold), "fldBarrierOkFits": int((np.abs(cold.barrier_hat - BARRIER) <= BAND).sum())}
    return out


def pair_table_data(S: Sources) -> list[dict]:
    """conf-pair.tex: per cell the mean and sd of Theta_hat/Theta and of the barrier at M = 2048, the count within 20% of 1/4 (cells
    with Theta <= 1), and the site energy's risk over the linear floor at every M."""
    d = e33(S)
    tab = e33_pair_readings(S)
    Q = d["Q"]
    Q = Q[Q.selection == "risk"]
    risk = Q.groupby(["cell", "n_chains"]).risk.mean().unstack()
    rows = []
    for c in ORDER:
        g = tab[tab.cell == c]
        T = float(g["T"].iloc[0])
        rows.append(dict(cell=c, T=T, theta_mean=float(g.theta_ratio.mean()), theta_sd=float(g.theta_ratio.std()),
                         barrier_mean=float(g.barrier_hat.mean()), barrier_sd=float(g.barrier_hat.std()),
                         band=(int((np.abs(g.barrier_hat - BARRIER) <= BAND).sum()), len(g)) if T <= 1 else None,
                         frac={m: float(risk.loc[c, m] / d["floor"][c]) for m in (32, 128, 512, 2048)}))
    return rows


def read_e42(S: Sources) -> dict:
    d42 = S.csv("outputs/hpc_E42_s*/kernels.csv")
    tab, _ = X.load_e42_table(e33(S)["Q"], d42)
    t = tab[tab.cell == "T2_iso_L32"].copy()
    t["err"] = (t.barrier_hat - BARRIER).abs()
    s = t[t.n_chains.isin((512, 2048, 8192, 32768))].groupby("n_chains").err.mean()
    slope = float(np.polyfit(np.log(s.index.values.astype(float)), np.log(s.values), 1)[0])
    ok = {int(n): int((g.err <= BAND).sum()) == 16 and len(g) == 16 for n, g in t.groupby("n_chains")}
    sizes = [n for n in sorted(ok) if n >= 2048 and ok[n]]
    return {"fldBarrierSlope": abs(slope), "fldBarrierBigM": sizes[0] if sizes else float("nan")}


# ------------------------------------------------------------------------------------------------ E34 / E37 (physical statistics of samples)
def read_e34(S: Sources) -> dict:
    P = S.csv("outputs/E34/physics.csv")
    P = P[P.source == "exact_samples"]
    out = {}
    raw = P.groupby("cell").kink_density_physical.mean()
    out["fldRawKinkA"], out["fldRawKinkB"], out["fldRawKinkC"] = float(raw[BASE]), float(raw["T0.3_h0.5_L64"]), float(raw[FINE])
    co = P.groupby("cell").coarse_kink_density_physical.mean()           # NaN rows are skipped: the h = 0.5 cell is rep 0 alone
    c3 = co[COLD3]
    out["fldCoarseKinkSpread"] = float((c3.max() - c3.min()) / c3.min())
    out["fldCorrZMax"] = float(P.corr_max_z.max())
    return out


def learned_ratios(S: Sources) -> pd.DataFrame:
    """E34 physics_learned: per (arm, cell, seed) generated / true for the coarse kink density and coarse mean domain length (the ratio is
    the same per site and per unit physical length: h is fixed within a cell), the raw kink density and mean domain, at M = 2048."""
    def build():
        d = pd.concat([S.csv("outputs/E34_learned/learned.csv"), S.csv("outputs/E34_learned_pair/learned.csv")], ignore_index=True)
        t = d[d.arm == "truth"].drop_duplicates(["cell", "seed"]).set_index(["cell", "seed"])
        a = d[d.arm != "truth"].copy()
        cols = ["coarse_kink_density", "coarse_mean_domain", "kink_density", "mean_domain"]
        for c in cols:
            a["r_" + c] = [r[c] / t.loc[(r.cell, r.seed), c] for _, r in a.iterrows()]
        return a
    return S.memo("learned_ratios", build)


def read_e34_learned(S: Sources) -> dict:
    a = learned_ratios(S)
    out = {}
    for arm, key in (("em_pair_quartic", "Site"), ("em_tilted_spline", "Spline")):
        m = a[a.arm == arm].groupby("cell").r_coarse_kink_density.mean().reindex(ORDER)
        out[f"fldKinkDev{key}"] = float((m - 1).abs().max())
    mx = a[a.arm == "em_mixture"].groupby("cell").r_coarse_kink_density.mean()[COLD3]
    out["fldMixKinkColdMin"], out["fldMixKinkColdMax"] = float(mx.min()), float(mx.max())
    return out


def read_e34_clean(S: Sources) -> dict:
    C = F.mark_failed(S.csv("outputs/hpc_E34c/clean.csv"))
    mix = F.risk_rows(e33(S)["allk"], "em_mixture")
    j = F.pair(F.risk_rows(C, "em_mixture"), mix)
    assert len(j) == 6 * 16 * 4, "E34 clean and E33 mixture must pair on every (cell, seed, size)"
    m = j.groupby("n_chains").ratio.mean()                   # clean / noisy, mean of per-seed ratios pooled over cells
    return {"fldCleanMixRatioMin": float(m.min()), "fldCleanMixRatioMax": float(m.max())}


def e37_ratios(S: Sources) -> pd.DataFrame:
    E = S.csv("outputs/E37/distribution.csv")
    t = E[E.arm == "truth"].set_index(["cell", "seed"])
    a = E[E.arm != "truth"].copy()
    for q in ("q10", "q50", "q90"):
        a["r_" + q] = [r[q] / t.loc[(r.cell, r.seed), q] for _, r in a.iterrows()]
    return a


def read_e37(S: Sources) -> dict:
    a = S.memo("e37", lambda: e37_ratios(S))
    out = {}
    for arm, key in (("em_pair_quartic", "Site"), ("em_tilted_spline", "Spline")):
        g = a[a.arm == arm]
        dev = max(float((g.groupby("cell")["r_" + q].mean() - 1).abs().max()) for q in ("q10", "q50", "q90"))
        out[f"fldDomQuantDev{key}"] = dev
    m = a[a.arm == "em_mixture"].groupby("cell").r_q50.mean()[COLD3]
    out["fldMixDomShortMin"], out["fldMixDomShortMax"] = float((1 - m).min()), float((1 - m).max())
    return out


def samples_table_data(S: Sources) -> dict:
    """Per arm (site energy, spline, mixture) and cell: coarse kink density and coarse mean domain length, generated / true (E34
    physics_learned, mean over the four seeds of the per-seed ratio); E37's seed-mean q10, q50, q90 ratios and KS statistic."""
    a = learned_ratios(S)
    e = S.memo("e37", lambda: e37_ratios(S))
    out = {}
    for arm in ("em_pair_quartic", "em_tilted_spline", "em_mixture"):
        g, h = a[a.arm == arm], e[e.arm == arm]
        rows = {"kinks": g.groupby("cell").r_coarse_kink_density.mean(), "domain": g.groupby("cell").r_coarse_mean_domain.mean(),
                "q10": h.groupby("cell").r_q10.mean(), "q50": h.groupby("cell").r_q50.mean(), "q90": h.groupby("cell").r_q90.mean(),
                "ks": h.groupby("cell").ks.mean()}
        out[arm] = {k: [float(v.get(c, np.nan)) for c in ORDER] for k, v in rows.items()}
    return out


# ------------------------------------------------------------------------------------------------ E24x (cost), E47 (coarse grid), E38 (memory)
def _median_ms(d, arm, col="sec_per_seq"):
    return float(d[d.arm == arm][col].median() * 1e3)


def read_e24x(S: Sources) -> dict:
    cpu, gpu = S.csv("outputs/hpc_E24x/cost.csv"), S.csv("outputs/hpc_E24x_gpu/cost.csv")
    out = {"cmpBpMsEightThreads": _median_ms(cpu, "bp_grid"), "cmpTfMsEightThreads": _median_ms(cpu, "transformer")}
    out["cmpRatioCpu"] = out["cmpBpMsEightThreads"] / out["cmpTfMsEightThreads"]
    g = gpu[gpu.device == "gpu"]
    out["cmpRatioGpu"] = _median_ms(g, "bp_grid") / _median_ms(g, "transformer")
    sc = S.csv("outputs/hpc_E24x_gpu/scaling.csv")
    sc = sc[(sc.device == "gpu") & (sc.factor == "batch") & (sc.batch == 2048)]
    per_seq = (sc.groupby("arm").wall_median.median() / 2048)
    out["cmpRatioGpuLarge"] = float(per_seq["bp_grid"] / per_seq["transformer"])
    sl = S.csv("outputs/hpc_E24x/slopes.csv")
    sl = sl[(sl.device == "cpu") & (sl.arm == "bp_grid")].set_index("factor").slope_full
    out["cmpSlopeBatch"], out["cmpSlopeLength"], out["cmpSlopeGrid"] = float(sl["batch"]), float(sl["n_sites"]), float(sl["grid_size"])
    an = S.csv("outputs/hpc_E24x/anatomy.csv")
    ag = S.csv("outputs/hpc_E24x_gpu/anatomy.csv")
    lo = float(an[(an.device == "cpu") & (an.grid_size == 401)].gemm_fraction.iloc[0])
    hi = float(ag[(ag.device == "cpu") & (ag.grid_size == 401)].gemm_fraction.iloc[0])
    out["cmpMatmulShareLo"], out["cmpMatmulShareHi"] = lo, hi
    out["cmpFftCapLo"], out["cmpFftCapHi"] = 1.0 / (1.0 - lo), 1.0 / (1.0 - hi)
    return out


def e47t(S: Sources) -> pd.DataFrame:
    d = S.csv("outputs/hpc_E47t/scaling.csv")
    d["ms"] = d.wall_median / d.batch * 1e3
    return d


def read_e47t(S: Sources) -> dict:
    d = e47t(S)
    bp = d[(d.arm == "bp_grid") & (d.factor == "grid_size")].set_index("value").ms
    net = d[(d.factor == "batch") & (d.arm != "bp_grid")].set_index("arm").ms
    out = {"cmpBpMsOneThread": float(bp[401]), "cmpBpMsOneFifty": float(bp[151]),
           "cmpWindowMsOne": float(net["window"]), "cmpGruMsOne": float(net["bigru"]), "cmpUnetMsOne": float(net["unet"]),
           "cmpTfMsOne": float(net["transformer"]), "cmpTcnMsOne": float(net["tcn"]),
           "cmpRatioOneTf": float(bp[401] / net["transformer"]), "cmpRatioOneGru": float(bp[401] / net["bigru"]),
           "cmpRatioFiftyGru": float(bp[151] / net["bigru"]),
           "cmpCoarseSpeedupHundred": float(bp[401] / bp[101]), "cmpCoarseSpeedupOneFifty": float(bp[401] / bp[151])}
    return out


def cost_time_data(S: Sources) -> dict:
    d = e47t(S)
    bp = d[(d.arm == "bp_grid") & (d.factor == "grid_size")].set_index("value").ms.sort_index()
    one = d[(d.factor == "batch") & (d.arm != "bp_grid")].set_index("arm").ms
    cpu = S.csv("outputs/hpc_E24x/cost.csv")
    eight = {a: _median_ms(cpu, a) for a in cpu.arm.unique()}
    return dict(bp_one={int(k): float(v) for k, v in bp.items()}, net_one={a: float(v) for a, v in one.items()}, eight=eight)


def read_e47(S: Sources) -> dict:
    out = {}
    a = S.csv("outputs/E47a/refinement.csv")
    out["cmpCoarseOracleHundred"] = float(a[(a.grid_size == 101) & (a.half_width == 8.0)].rel_error.max())
    # part (b): the learned kernels against E33's N_g = 401 fits (paired by seed, same test bundle)
    d, d33 = S.csv("outputs/hpc_E47b_s*/kernels.csv"), e33(S)["allk"]

    def ratio(arm, cell, n, ng):
        x = d[(d.arm == arm) & (d.selection == "risk") & (d.cell == cell) & (d.n_chains == n) & (d.grid_size == ng)].set_index("seed").sort_index()
        y = d33[(d33.arm == arm) & (d33.selection == "risk") & (d33.cell == cell) & (d33.n_chains == n)].set_index("seed").reindex(x.index)
        assert (x.test_bundle.values == y.test_bundle.values).all(), "E47 and E33 were scored on different test bundles"
        return float(x.risk.mean() / y.risk.mean()), float(x.wall.median() / y.wall.median())
    devs = [abs(ratio(arm, cell, 2048, 51)[0] - 1) for arm in ("em_pair_quartic", "em_tilted_spline") for cell in (BASE, FINE)]
    out["cmpCoarseLearnedFifty"] = max(devs)
    out["cmpCoarseFitSpeedup"] = 1.0 / ratio("em_pair_quartic", BASE, 2048, 101)[1]
    # part (d): the linear chain's headline kernels
    c = S.csv("outputs/E47c/refinement.csv")
    c = c[c.half_width == 8.0]

    def rr(col, n, ng):
        x = c[(c.n_chains == n) & (c.n_grid == ng)].set_index("seed")[col].sort_index()
        y = c[(c.n_chains == n) & (c.n_grid == 401)].set_index("seed")[col].reindex(x.index)
        return float(x.mean() / y.mean())
    out["cmpHeadlineAtOneFifty"] = max(abs(rr(col, n, 151) - 1) for col in ("risk_em_C1", "risk_em_C8") for n in (32, 2048))
    out["cmpHeadlineAtHundred"] = abs(rr("risk_em_C8", 2048, 101) - 1)
    out["cmpHeadlineAtFifty"] = abs(rr("risk_em_C8", 2048, 51) - 1)
    out["cmpResolvedHundred"] = float(c[c.n_grid == 101].resolved_C8.mean())
    return out


def read_e38(S: Sources) -> dict:
    def inc(path):
        m = S.csv(path)
        m = m[(m.kind == "infer") & (m.B == 512) & (m.L == 32) & (m.N_g == 401)].set_index("arm").increment_mb
        nets = m.drop(["bp_grid", "em_estep"], errors="ignore")
        return float(m["bp_grid"]), float(m["bp_grid"] / nets.max())
    mb, lo = inc("outputs/E38/memory.csv")
    _, hi = inc("outputs/hpc_E38/memory.csv")
    return {"cmpMemMb": mb, "cmpMemRatioMin": lo, "cmpMemRatioMax": hi}


# ------------------------------------------------------------------------------------------------ E22, E23, E40 (representations)
def read_e22(S: Sources) -> dict:
    d = S.csv("outputs/hpc_E22c/shapes_summary.csv")
    s = d[(d.rho == 0.85) & (d.n_sites == 32) & (d.grid_size == 801) & (d.level == 0.05) & (d.site == 31) & (d.family == "laplace")].set_index("ansatz").kl_median
    return {"cmpAnsatzOverGauss": float(s["gaussian"] / s["gauss_through_transition"])}


def fourier_table_data(S: Sources) -> dict:
    d = S.csv("outputs/hpc_E23/dstar.csv")
    d = d[d.family == "laplace"]
    out = {"_design": dict(rho=int(d.rho.nunique()), level=int(d.level.nunique()), site=int(d.site.nunique()))}
    for rep in ("grid", "fourier", "fourier_adapted", "cumulant"):
        g = d[d["repr"] == rep]
        assert len(g) == 144, (rep, len(g))
        row = {}
        for col in ("dstar_0.01", "dstar_0.0001", "dstar_1e-06"):
            row[col] = (float(g[col].median()), int(g[col].notna().sum()))        # NaN = the cell never reaches the threshold
        row["nonmono"] = int(g.non_monotone.sum())
        out[rep] = row
    return out


def read_e23(S: Sources) -> dict:
    t = fourier_table_data(S)
    assert len({k: v for k, v in t.items() if not k.startswith("_")}) == 4
    return {"cmpGridLo": t["grid"]["dstar_0.01"][0], "cmpGridHi": t["grid"]["dstar_0.0001"][0],
            "cmpFourierAdaptedLo": t["fourier_adapted"]["dstar_0.01"][0], "cmpFourierAdaptedHi": t["fourier_adapted"]["dstar_0.0001"][0],
            "cmpFourierReach": f"{t['fourier_adapted']['dstar_0.0001'][1]} of 144"}


def read_e40(S: Sources) -> dict:
    f = S.csv("outputs/E40/fft.csv")
    acc = f[(f.part == "accuracy") & (f.method == "fft") & (f.n_grid == 401) & (f.half_width == 8.0)]
    tm = f[f.part == "timing"].pivot_table(index="n_grid", columns="method", values="wall")
    return {"cmpFftError": float(acc.err_vs_dense.max()), "cmpFftSlower": float(tm.loc[401, "fft"] / tm.loc[401, "dense"]),
            "cmpFftSlowerLarge": float(tm.loc[3201, "fft"] / tm.loc[3201, "dense"])}


# ------------------------------------------------------------------------------------------------ E24x runtime of the field figure, training wall
def read_field_runtime(S: Sources) -> dict:
    sc = S.csv("outputs/hpc_E24x/scaling.csv")
    sc = sc[sc.factor == "n_sites"]
    lat = sc.groupby(["arm", "value"]).wall_median.median() / sc.batch.iloc[0] * 1e3        # ms per sequence (the runtime figure's)
    nets = [float(lat[(a, L)]) for a in ("window", "transformer") for L in (32, 64, 128)]
    return {"fldNetCostMin": min(nets), "fldNetCostMax": max(nets), "fldBpCostLone": float(lat[("bp_grid", 32)]), "fldBpCostLthree": float(lat[("bp_grid", 128)])}


def read_training(S: Sources) -> dict:
    d = e33(S)
    n = d["N"][d["N"].n_chains == 2048]
    net = n.groupby(["cell", "arm", "arch"]).wall.median()
    k = d["allk"]
    k = k[(k.n_chains == 2048) & (k.selection == "risk") & k.arm.isin(["em_pair_quartic", "em_tilted_spline", "em_mixture", "em_mdn"])]
    em = k.groupby(["cell", "arm"]).wall.median()
    return {"fldTrainNetMin": float(net.min() / 60), "fldTrainNetMax": float(net.max() / 60),
            "fldTrainEmMin": float(em.min() / 60), "fldTrainEmMax": float(em.max() / 3600)}


# ------------------------------------------------------------------------------------------------ E44, E45, E50 (beyond the score)
def read_e44(S: Sources) -> dict:
    d = S.csv("outputs/hpc_E44_s*/quench.csv")
    d = d[d.selection == "risk"]
    p = d.pivot_table(index=["tau", "seed"], columns="arm", values="risk")
    def ratio(tau: float) -> float:
        return float((p.loc[tau].em_pair_quartic / p.loc[tau].linear_empirical).mean())
    return {"fldQuenchEarly": ratio(1.5), "fldQuenchEq": ratio(float("inf"))}


def read_e45(S: Sources) -> dict:
    d = S.csv("outputs/E45/inpainting.csv")
    learned = ("em_pair_quartic", "em_tilted_spline")
    cov = d.groupby(["cell", "pattern", "arm"]).coverage90.mean()
    cl = cov.loc[(slice(None), slice(None), list(learned))]
    err = {k: g.pivot_table(index="seed", columns="arm", values="rel_err", dropna=False) for k, g in d.groupby(["cell", "pattern"])}
    wid = {k: g.pivot_table(index="seed", columns="arm", values="width90", dropna=False) for k, g in d.groupby(["cell", "pattern"])}
    er = [err[k][a].mean() / err[k]["oracle"].mean() for k in err for a in learned]
    wr = [wid[k][a].mean() / wid[k]["oracle"].mean() for k in err for a in learned]
    base = [k for k in err if k[0] == BASE]
    le = [err[k]["linear"].mean() / err[k]["oracle"].mean() - 1 for k in base]
    lw = [wid[k]["linear"].mean() / wid[k]["oracle"].mean() - 1 for k in base]
    ocov = float(d[d.arm == "oracle"].coverage90.mean())
    return {"fldInpaintCovMin": float(cl.min()), "fldInpaintCovMax": float(cl.max()), "fldInpaintOracleCov": ocov, "fldInpaintGridCost": 0.90 - ocov,
            "fldInpaintOracleDev": max(max(abs(x - 1) for x in er), max(abs(x - 1) for x in wr)),
            "fldInpaintOracleErrMin": min(er), "fldInpaintOracleErrMax": max(er), "fldInpaintOracleWidthMin": min(wr), "fldInpaintOracleWidthMax": max(wr),
            "fldInpaintLinErrMin": min(le), "fldInpaintLinErrMax": max(le), "fldInpaintLinWidthMin": min(lw), "fldInpaintLinWidthMax": max(lw)}


def read_e50(S: Sources) -> dict:
    c = S.csv("outputs/E50/commitment.csv")
    g, tc = X.e50_crossings(c)
    out = {}
    dev = (g.correct_all - 0.5 * (1.0 + g.commit_all)).abs()
    out["fldSpecCalibDev"] = float(dev.groupby(g.cell).mean().max())
    out["fldSpecTcCold"], out["fldSpecTcHot"] = tc["T0.2_h1_L32"]["all"], tc["T2_h1_L32"]["all"]
    out["fldSpecExcessCold"] = tc["T0.2_h1_L32"]["all"] - tc["T0.2_h1_L32"]["twin"]
    out["fldSpecExcessHot"] = tc["T2_h1_L32"]["all"] - tc["T2_h1_L32"]["twin"]
    out["fldSpecNoiseRatioExact"] = math.expm1(2.0 * tc["T0.2_h1_L32"]["all"])
    out["fldSpecNoiseRatioTwin"] = math.expm1(2.0 * tc["T0.2_h1_L32"]["twin"])
    for L, key in ((4, "Lfour"), (64, "Lsixtyfour"), (128, "Lonetwentyeight"), (256, "Ltwofiftysix")):
        out[f"fldSpecTc{key}"] = tc[f"T0.2_h1_L{L}"]["bulk"]
    below = [f"T0.2_h1_L{L}" for L in (4, 8, 16)]
    x = np.array([0.5 * np.log1p(tc[b]["lam"]) for b in below])
    y = np.array([tc[b]["bulk"] for b in below])
    out["fldSpecSlope"] = float(np.polyfit(x, y, 1)[0])
    k = S.csv("outputs/E50/kink.csv")
    fits = X.e50_kink_fits(k)
    out["fldKinkFit"] = float(fits[1 / 32][0])
    f = k[(k.n_fine == 2401) & np.isclose(k.h, 1 / 32)].set_index("T").xi_phys.sort_index()
    ln = np.log(f)

    def slope(t1, t2):
        return float((ln[t1] - ln[t2]) / (1.0 / t1 - 1.0 / t2))
    out["fldKinkSlopeHot"], out["fldKinkSlopeCold"] = slope(0.3, 0.275), slope(0.1, 0.125)
    out["fldKinkTHigh"], out["fldKinkTLow"] = (0.3 + 0.275) / 2, (0.1 + 0.125) / 2
    a = k[(k.n_fine == 2401) & np.isclose(k.h, 1 / 32)].set_index("T").xi_phys
    b = k[(k.n_fine == 2401) & np.isclose(k.h, 1 / 16)].set_index("T").xi_phys.reindex(a.index)
    out["fldKinkConvergence"] = float((np.log(a) - np.log(b)).abs().max())
    return out


# ------------------------------------------------------------------------------------------------ limits: E28, E27x, E30, E35x, E36x, E43, E46, E48, E49
def read_e27(S: Sources) -> dict:
    c = S.csv("outputs/hpc_E27x/cost.csv")
    s = c[c.batch == 2048].set_index("arm").sec_per_seq
    return {"genAncestralCostRatio": float(s["reverse_sde_200"] / s["ancestral_true"])}


def read_e28(S: Sources) -> dict:
    d = S.csv("outputs/hpc_E28/per_level.csv")
    out = {}

    def crossing(g):                       # first schedule level at which the LMMSE arm is better than EM-BP (no interpolation)
        p = g.pivot_table(index="t", columns="arm", values="risk_level").sort_index()
        below = p.index[(p.lmmse < p.em_bp).values]
        return float(below[0]) if len(below) else float("nan")
    for n, tag in ((128, "A"), (2048, "B")):
        s = d[d.n_chains == n]
        out[f"lvlCross{tag}"] = float(np.nanmedian([crossing(g) for _, g in s.groupby("seed")]))
        pool = s.groupby(["arm", "t"])[["sq_err", "sq_ref"]].sum()
        r = np.sqrt(pool.sq_err / pool.sq_ref).unstack("arm")
        out[f"lvlFirstLevels{tag}"] = int((r.idxmin(axis=1) == "em_bp").sum())
    return out


def read_e30(S: Sources) -> dict:
    d = S.csv("outputs/hpc_E30_s*/joint.csv")
    pool = d.groupby(["arm", "n_chains"])[["sq_err", "sq_ref"]].sum()          # pooled over the 16 seeds and the levels, as E30's registered reading
    r = np.sqrt(pool.sq_err / pool.sq_ref).unstack("arm")
    q = (r.mdn / r.em_bp)
    return {"limMdnLinearLo": float(q.min()), "limMdnLinearHi": float(q.max())}


def read_e35x(S: Sources) -> dict:
    d = S.csv("outputs/hpc_E35x_s*/augmented.csv")
    p = d.pivot_table(index=["b", "n_chains", "seed"], columns="arm", values="risk")
    g2, g1 = p.loc[(2.0, 2048)], p.loc[(2.0, 512)]
    fall = 1 - g2.em_augmented.mean() / g1.em_augmented.mean()
    flat = g2.em_first_order.mean() / g1.em_first_order.mean() - 1
    worse = [p.loc[(b, m)].em_first_order.mean() / p.loc[(b, m)].lmmse_true.mean() for b in (1.0, 2.0) for m in (512, 2048)]
    return {"limLatentAugFall": float(fall), "limLatentFlat": float(flat), "limLatentFirstOverAug": float((g2.em_first_order / g2.em_augmented).mean()),
            "limLatentFloorRatio": float(g2.lmmse_true.mean() / g2.em_augmented.mean()), "limLatentWorseMin": min(worse), "limLatentWorseMax": max(worse)}


def read_e36(S: Sources) -> dict:
    o = S.csv("outputs/E36/oracles.csv")
    p = S.csv("outputs/E36/physics.csv")
    c = S.csv("outputs/hpc_E36x_s*/clean.csv")
    pv = c.pivot_table(index=["seed", "stride"], columns="arm", values="risk")
    out = {"limStrideFloor": float(o[(o.arm == "lmmse_true") & (o.level == -1) & (o.stride == 4)].risk.mean())}
    m = p.groupby("stride")[["coarse_kink_density_physical", "coarse_mean_domain_physical"]].mean()
    out["limStrideAgree"] = float(((m.max() - m.min()) / m.min()).max())
    for arm, key in (("em_pair_quartic", "Site"), ("em_tilted_spline", "Spline")):
        s4, s1 = pv[arm].xs(4, level="stride"), pv[arm].xs(1, level="stride")
        out[f"limStrideGrowth{key}"] = float((s4 / s1).mean())
        out[f"limStride{key}Risk"] = float(s4.mean())
    return out


def read_e48(S: Sources) -> dict:
    d = S.csv("outputs/hpc_E48_s*/clean.csv")
    d36 = S.csv("outputs/hpc_E36x_s*/clean.csv")
    mdn = d[d.arm == "em_mdn"].set_index(["stride", "seed"]).sort_index()
    ref = d36[d36.arm == "em_pair_quartic"].set_index(["stride", "seed"]).sort_index().reindex(mdn.index)
    assert (mdn.test_bundle.values == ref.test_bundle.values).all(), "E48 and E36x were scored on different test bundles"
    m4, r4, m1, r1 = mdn.loc[4], ref.loc[4], mdn.loc[1], ref.loc[1]
    return {"limStrideMdnAtCap": int((m4.at_cap == 1).sum()), "limStrideMdnBelow": int((m4.risk < r4.risk).sum()),
            "limStrideMdnRatio": float((m4.risk / r4.risk).mean()), "limStrideOneRatio": float(m1.risk.mean() / r1.risk.mean())}


def read_e43_e49(S: Sources) -> dict:
    d33 = e33(S)["allk"]
    dA, dB, d49 = S.csv("outputs/hpc_E43a_s*/kernels.csv"), S.csv("outputs/hpc_E43b_s*/kernels.csv"), S.csv("outputs/hpc_E49_s*/kernels.csv")
    out = {}
    # lab A: the unrestricted kernel over the site energy, ratio of seed means, two cells, M = 128, 512, 2048
    rat = {}
    for n in (128, 512, 2048):
        ll = X._risk_rows(dA, "em_loglinear", n)
        pq = X._risk_rows(d33, "em_pair_quartic", n).reindex(ll.index)
        assert (ll.test_bundle.values == pq.test_bundle.values).all(), "E43 lab A and E33 were scored on different test bundles"
        for c in sorted(set(dA.cell)):
            rat[(c, n)] = float(ll.loc[c].risk.mean() / pq.loc[c].risk.mean())
    out["limUnrestrictedPriceMin"], out["limUnrestrictedPriceMax"] = min(rat.values()), max(rat.values())
    out["limUnrestrictedGrowthA"], out["limUnrestrictedGrowthB"] = rat[(BASE, 128)], rat[(BASE, 2048)]
    # lab B and the MDN
    R = {a: X._risk_rows(dB, a, 2048).loc["LAP_B_L32"] for a in ("em_loglinear", "em_pair_quartic", "em_tilted_spline")}
    mdn = X._risk_rows(d49, "em_mdn", 2048).loc["LAP_B_L32"]
    ref = {a: R[a].reindex(mdn.index) for a in R}
    assert (ref["em_pair_quartic"].test_bundle.values == mdn.test_bundle.values).all(), "E49 and E43 were scored on different test bundles"
    site, spl, ll = R["em_pair_quartic"].risk.mean(), R["em_tilted_spline"].risk.mean(), R["em_loglinear"].risk.mean()
    out["limLapSite"], out["limLapUnrestricted"], out["limLapStall"] = float(site), float(ll), float((site + spl) / 2)
    out["limLapUnrestrictedGain"] = float(site / ll)
    out["limLapMdn"] = float(mdn.risk.mean())
    out["limLapMdnAtCap"] = int((mdn.at_cap == 1).sum())
    out["limLapMdnBelowSite"] = int((mdn.risk < ref["em_pair_quartic"].risk).sum())
    out["limLapMdnBelowUnr"] = int((mdn.risk < ref["em_loglinear"].risk).sum())
    return out


def read_e46(S: Sources) -> dict:
    d46, d35 = S.csv("outputs/hpc_E46_s*/symmetry.csv"), S.csv("outputs/hpc_E35x_s*/augmented.csv")
    gains = []
    for n in (512, 2048):
        new = d46[d46.n_chains == n].pivot_table(index="seed", columns="arm", values="risk")
        old = d35[(d35.b == 0) & (d35.n_chains == n)].pivot_table(index="seed", columns="arm", values="risk")
        p = new.join(old, how="inner")
        assert len(p) == 16, "E46 and E35x must pair on 16 seeds"
        gains.append(1 - math.exp(float(np.log(p.em_first_order_sym / p.em_first_order).mean())))
    return {"limSymGainMin": min(gains), "limSymGainMax": max(gains)}


# ------------------------------------------------------------------------------------------------ everything
READERS = [read_e22, read_e23, read_e24x, read_e27, read_e28, read_e30, read_e32, read_e33, read_e33_physics, read_e34, read_e34_learned,
           read_e34_clean, read_e35x, read_e36, read_e37, read_e38, read_e40, read_e42, read_e43_e49, read_e44, read_e45, read_e46,
           read_e47, read_e47t, read_e48, read_e50, read_field_runtime, read_training]


def read_all(S: Sources) -> dict:
    out: dict = {}
    for fn in READERS:
        part = {k: v for k, v in fn(S).items() if not k.startswith("_")}
        dup = set(part) & set(out)
        if dup:
            raise SystemExit(f"make_paper_numbers: {fn.__name__} redefines {sorted(dup)}")
        out.update(part)
    return out
