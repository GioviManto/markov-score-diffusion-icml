#!/usr/bin/env python
"""E18 — the closure gap on a domain wide enough for the prior's tail, and the heavy-tailed regime.

E01 measured E||S - S^G||^2 on the (801, 8) grid and read the Student-t_5 curve's departure from
e^{-8t} near t = 3 as "the moment condition predicts". On that grid the prior itself is truncated at
|a| = 8, and beyond t ~ 3 the truncated prior's tail, not the true prior's, drives the reference:
every family departs from its constant there, heavy-tailed ones first. This experiment recomputes the
gap at (1601, 16) and (3201, 32) for the four E01 families plus Student-t with nu = 7 and 9 (finite
sixth and eighth moments), to t = 5, and adds an untruncated SCALAR quadrature (part "scalar") in
which the tail integral is actually performed, to separate the two mechanisms:

  (i)  the Hermite term (1/6)||kappa_4||^2 e^{-8t}, reached by every family on a wide enough domain;
  (ii) for a regularly varying tail of index nu, an additional e^{-nu t} term in EXPECTATION, carried
       by events of probability ~ alpha^nu that no sample of practical size contains (a sample-based
       chain measurement therefore follows (i); the scalar quadrature sees (ii)).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score import theory  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import GaussianMixtureAR1, LaplaceAR1, StudentTAR1, UniformAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

L = FROZEN.n_sites

SETTINGS = dict(rho=FROZEN.rho, levels=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0),
                grids=((801, 8.0), (1601, 16.0), (3201, 32.0)), n_chains=256, reps=2,
                families=("laplace", "bimodal", "uniform", "student5", "student7", "student9"),
                scalar_families=("t5", "t7", "t9", "t12", "laplace"), scalar_levels=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0),
                scalar_A=250.0, scalar_n=125_001, scalar_X=40.0, scalar_nx=4_001,
                const_families=("laplace", "bimodal", "uniform", "student5", "student7", "student9"), const_levels=(3.0, 4.0),
                const_grid=(1601, 16.0), const_chains=8192, const_chunk=512)
QUICK = dict(levels=(2.0, 3.0), grids=((201, 8.0), (401, 16.0)), n_chains=16, reps=1, families=("laplace", "student5"),
             scalar_families=("t5",), scalar_levels=(2.0, 3.0), scalar_n=20_001, scalar_nx=801)


def _prior(name, rho):
    return {"laplace": LaplaceAR1(rho), "bimodal": GaussianMixtureAR1(rho, 0.6), "uniform": UniformAR1(rho),
            "student5": StudentTAR1(rho, 5), "student7": StudentTAR1(rho, 7), "student9": StudentTAR1(rho, 9)}[name]


def part_chain(cfg, out):
    rho = cfg["rho"]; ts = cfg["levels"]
    sig0 = theory.chain_covariance(rho, 1 - rho**2, L)
    rows = []
    for name in cfg["families"]:
        prior = _prior(name, rho)
        k4 = prior.innovation_excess_kurtosis
        C4 = theory.score_gap_constant(theory.chain_kappa4(rho, theory.excess_kurtosis_to_kappa4(k4, prior.q), L)) if np.isfinite(k4) else np.nan
        for ng, A in cfg["grids"]:
            grid, w = make_grid(A, ng); logK = prior.log_transition_matrix(grid)
            gaps, nout = [], 0
            for rep in range(cfg["reps"]):
                rng = rng_for("e18-chain", name, rep)
                Am = np.stack([prior.sample(rng, L) for _ in range(cfg["n_chains"])]); Z = rng.standard_normal(Am.shape)
                nout += int((np.abs(Am) > A).sum())
                g = []
                for t in ts:
                    al, de = alpha_delta(t); X = al * Am + np.sqrt(de) * Z
                    m_bp, _ = grid_bp_batch(grid, w, logK, X, al, de)
                    s_bp = theory.score_from_mean(X, m_bp, t); s_lm = theory.score_from_mean(X, theory.lmmse_posterior_mean(X, sig0, t), t)
                    g.append(float(((s_lm - s_bp) ** 2).sum()) / Am.shape[0])
                gaps.append(g)
            g = np.mean(gaps, axis=0)
            for t, v in zip(ts, g):
                rows.append(dict(family=name, n_grid=ng, half_width=A, sites_outside=nout, t=t, sq_gap_per_chain=v, C4=C4,
                                 ratio_to_prediction=(v * np.exp(8 * t) / C4) if np.isfinite(C4) else np.nan))
            sl = np.diff(np.log(g)) / np.diff(ts)
            print(f"  [E18] {name:9s} ({ng},{A:g}) sites>|A|: {nout}  slopes " + " ".join(f"{s:.2f}" for s in sl)
                  + (f"  ratio at t=3,4,5: " + " ".join(f"{g[i]*np.exp(8*ts[i])/C4:.3g}" for i in (ts.index(3.0), ts.index(4.0), ts.index(5.0)) if all(x in ts for x in (3.0, 4.0, 5.0))) if np.isfinite(C4) else ""), flush=True)
    write_csv(out / "closure_wide.csv", rows)


def _scalar_grid(alpha, core=40.0, dcore=0.004, x_max=40.0, tail_factor=3.0):
    """Two-segment a-grid: a fine core on [-core, core] and coarse tails out to tail_factor * x_max / alpha,
    where the posterior for an x in the tail actually lives (a ~ x / alpha). A fixed domain truncates the
    prior's tail at large t and pins the tail posterior means, which is exactly the artefact this
    experiment exists to remove; the tail spacing 0.05 / alpha is a twentieth of the likelihood's width in a."""
    A = max(core * 2, tail_factor * x_max / alpha)
    a_core = np.arange(-core, core + dcore / 2, dcore)
    dtail = 0.05 / alpha
    a_right = np.arange(core + dtail, A + dtail / 2, dtail)
    a = np.concatenate([-a_right[::-1], a_core, a_right])
    w = np.gradient(a)  # trapezoid-like weights on the non-uniform grid
    return a, w


def _gap_scalar(logp0, ts, A_unused, n_unused, X, nx):
    x = np.linspace(-X, X, nx)
    out = []
    for t in ts:
        al, de = np.exp(-t), 1 - np.exp(-2 * t)
        a, w = _scalar_grid(al, x_max=X)
        p0 = np.exp(logp0(a)); p0 /= float(np.sum(p0 * w))
        Pt = np.empty(nx); Ea = np.empty(nx)
        for s in range(0, nx, 200):
            xx = x[s:s + 200, None]
            wgt = np.exp(-0.5 * (xx - al * a[None, :]) ** 2 / de) / np.sqrt(2 * np.pi * de) * p0[None, :] * w[None, :]
            Pt[s:s + 200] = wgt.sum(1); Ea[s:s + 200] = (wgt * a[None, :]).sum(1) / np.maximum(wgt.sum(1), 1e-300)
        S = -(x - al * Ea) / de
        out.append(float(np.trapezoid((S + x) ** 2 * Pt, x)))
    return out


def part_scalar(cfg, out):
    rows = []
    ts = list(cfg["scalar_levels"])
    for name in cfg["scalar_families"]:
        if name.startswith("t"):
            nu = float(name[1:]); sc = np.sqrt((nu - 2) / nu)
            logp0 = (lambda a, nu=nu, sc=sc: stats.t.logpdf(a / sc, nu) - np.log(sc)); k4 = 6 / (nu - 4) if nu > 4 else np.inf
        else:
            b = 1 / np.sqrt(2); logp0 = (lambda a, b=b: -np.abs(a) / b - np.log(2 * b)); k4 = 3.0
        g = np.array(_gap_scalar(logp0, ts, cfg["scalar_A"], cfg["scalar_n"], cfg["scalar_X"], cfg["scalar_nx"]))
        C4 = k4**2 / 6 if np.isfinite(k4) else np.nan
        for t, v in zip(ts, g):
            rows.append(dict(family=name, t=t, sq_gap=v, C4=C4, ratio_to_prediction=(v * np.exp(8 * t) / C4) if np.isfinite(C4) else np.nan))
        sl = np.diff(np.log(g)) / np.diff(ts)
        print(f"  [E18 scalar] {name:8s} slopes " + " ".join(f"{s:.2f}" for s in sl) + (f"  ratio " + " ".join(f"{r:.3g}" for r in g * np.exp(8 * np.array(ts)) / C4) if np.isfinite(C4) else ""), flush=True)
    write_csv(out / "scalar_gap.csv", rows)


def part_constant(cfg, out):
    """The constant to a stated precision: the per-chain gap is a degree-six polynomial of Gaussians with a
    heavy right tail (max/median ~ 100 over 256 chains), so a 512-chain mean carries a ~15 % standard error and
    the E01 agreement to 1 % was partly luck. 8192 chains per family and level, with a chain bootstrap s.e."""
    rho = cfg["rho"]; sig0 = theory.chain_covariance(rho, 1 - rho**2, L)
    ng, A = cfg["const_grid"]; grid, w = make_grid(A, ng)
    rows = []
    for name in cfg["const_families"]:
        prior = _prior(name, rho); logK = prior.log_transition_matrix(grid)
        C4 = theory.score_gap_constant(theory.chain_kappa4(rho, theory.excess_kurtosis_to_kappa4(prior.innovation_excess_kurtosis, prior.q), L))
        rng = rng_for("e18-constant", name)
        per = {t: [] for t in cfg["const_levels"]}
        for _ in range(cfg["const_chains"] // cfg["const_chunk"]):
            Am = np.stack([prior.sample(rng, L) for _ in range(cfg["const_chunk"])]); Z = rng.standard_normal(Am.shape)
            for t in cfg["const_levels"]:
                al, de = alpha_delta(t); X = al * Am + np.sqrt(de) * Z
                m_bp, _ = grid_bp_batch(grid, w, logK, X, al, de)
                s_bp = theory.score_from_mean(X, m_bp, t); s_lm = theory.score_from_mean(X, theory.lmmse_posterior_mean(X, sig0, t), t)
                per[t].append(((s_lm - s_bp) ** 2).sum(1))
        for t in cfg["const_levels"]:
            g = np.concatenate(per[t]); pred = C4 * np.exp(-8 * t)
            brng = np.random.default_rng(0); bs = np.array([g[brng.integers(0, len(g), len(g))].mean() for _ in range(2000)])
            rows.append(dict(family=name, t=t, n_chains=len(g), n_grid=ng, half_width=A, mean_gap=float(g.mean()), median_gap=float(np.median(g)),
                             max_gap=float(g.max()), predicted=pred, ratio=float(g.mean() / pred), ratio_lo=float(np.percentile(bs, 2.5) / pred),
                             ratio_hi=float(np.percentile(bs, 97.5) / pred), rel_se=float(bs.std() / g.mean())))
            print(f"  [E18 constant] {name:9s} t={t:g} n={len(g)}: ratio {g.mean()/pred:.3f} [{np.percentile(bs,2.5)/pred:.3f}, {np.percentile(bs,97.5)/pred:.3f}]  max/median {g.max()/np.median(g):.0f}", flush=True)
    write_csv(out / "constant.csv", rows)


PARTS = {"chain": part_chain, "scalar": part_scalar, "constant": part_constant}


def main():
    ap = experiment_parser("e18_closure_wide", "the closure gap on a wide domain; the heavy-tailed regime")
    args = ap.parse_args()
    if args.list_parts:
        print("\n".join(PARTS)); return
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    out = ensure_dir(Path(args.output_dir))
    for name, fn in select_parts(PARTS, args.only).items():
        fn(cfg, out)
        write_json(out / f"params_{name}.json", {**provenance(), "settings": cfg, "quick": bool(args.quick), "frozen": FROZEN.as_dict()})


if __name__ == "__main__":
    main()
