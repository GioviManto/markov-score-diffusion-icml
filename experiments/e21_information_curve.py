#!/usr/bin/env python
"""E21 — the exact information curve at rho = 0.5 (what E12b's M*(t) must be compared with), and the
scalar verification of the shape-information constant of Theorem 3, eq. (5).

Part "curve": per-sequence efficient Fisher information about beta (shape) and rho for the Laplace
member of the generalised-Gaussian chain at rho = 0.5, by Fisher-identity scores on the grid (E01's
method), on E12b's levels and beyond; the Cramer-Rao bound for the excess kurtosis at each M; the
local log-slope of 1/I per coordinate -- the slope M*(t) is predicted to have on those levels -- against
the asymptotic 8 and 4; and the exact Gaussian-chain rho information for comparison.

Part "constant": for the scalar unit-variance generalised Gaussian at beta in {1.5, 2, 3},
I_t(beta) e^{8t} against (1/24)(d kappa4/d beta)^2 by untruncated quadrature with the beta-derivative
taken by Richardson-extrapolated central differences; and the Gaussian variance coordinate's 1/2 as a
control. On the chain the outer-product estimate of this constant reaches only 0.8-0.9 of the prediction
(E01, beta = 2, 3) before its quadrature floor; the scalar computation has no such floor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.special import gamma as G

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score import theory  # noqa: E402
from markov_score.bp_grid import make_grid  # noqa: E402
from markov_score.information import outer_product_information, per_chain_scores, summarise  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.shape_kernel import GeneralizedGaussianKernel  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402
from exp_22_shape_information import sample_generalized_gaussian  # noqa: E402

L = FROZEN.n_sites
K = lambda b: G(5 / b) * G(1 / b) / G(3 / b) ** 2 - 3  # excess kurtosis of the generalised Gaussian
SETTINGS = dict(rho=0.5, levels=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.7, 2.0), n_chains=512, reps=4,
                grid_size=FROZEN.n_grid, half_width=FROZEN.half_width, sizes=(256, 512, 1024, 2048, 4096, 8192, 16384),
                betas=(2.0, 3.0, 1.5), scalar_levels=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5), scalar_a=40.0, scalar_n=40_001, scalar_x=14.0, scalar_nx=2_801, h=2e-2)
QUICK = dict(levels=(0.6, 1.0), n_chains=32, reps=1, grid_size=201, betas=(2.0,), scalar_levels=(1.0,), scalar_n=4_001, scalar_nx=401)


def part_curve(cfg, out):
    rho = cfg["rho"]; q = 1 - rho**2
    dK = abs((K(1.0005) - K(0.9995)) / 0.001)
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    k = GeneralizedGaussianKernel(rho=rho, q=q, beta=1.0)
    rows = []
    for t in cfg["levels"]:
        vb, vr = [], []
        for rep in range(cfg["reps"]):
            rng = rng_for("e12b-power", rep)
            A = np.stack([sample_generalized_gaussian(rng, k, L) for _ in range(cfg["n_chains"])])
            al, de = alpha_delta(t); X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
            s = summarise(outer_product_information(per_chain_scores(grid, w, k, X, al, de)))
            vb.append(s["efficient"][2]); vr.append(s["efficient"][0])
        gI = theory.gaussian_chain_fisher(rho, q, L, t)
        row = dict(t=t, I_beta_eff=float(np.mean(vb)), I_beta_eff_sd=float(np.std(vb)), I_rho_eff=float(np.mean(vr)), I_rho_gauss=float(gI["efficient"][0]), dkappa4_dbeta=dK)
        for M in cfg["sizes"]:
            row[f"crb_kurt_M{M}"] = dK / np.sqrt(M * row["I_beta_eff"]); row[f"crb_rho_M{M}"] = 1 / np.sqrt(M * row["I_rho_eff"])
        rows.append(row)
        print(f"  [E21] t={t:.2f} I_beta_eff={row['I_beta_eff']:.4e} I_rho_eff={row['I_rho_eff']:.4e} (Gaussian chain {row['I_rho_gauss']:.4e})", flush=True)
    ts = np.array([r["t"] for r in rows]); ib = np.array([r["I_beta_eff"] for r in rows]); ir = np.array([r["I_rho_eff"] for r in rows])
    sb = -np.diff(np.log(ib)) / np.diff(ts); sr = -np.diff(np.log(ir)) / np.diff(ts)
    for j, r in enumerate(rows):
        r["slope_logMstar_beta"] = float(sb[j]) if j < len(sb) else np.nan; r["slope_logMstar_rho"] = float(sr[j]) if j < len(sr) else np.nan
    print("  [E21] t_* =", theory.crossover_time(rho, q, L), " local slopes of log(1/I_beta):", np.round(sb, 2), " of log(1/I_rho):", np.round(sr, 2))
    write_csv(out / "information.csv", rows)


def _logp_gg(a, beta):
    s = np.sqrt(G(1 / beta) / G(3 / beta))
    return -np.abs(a / s) ** beta - np.log(2 * s * G(1 + 1 / beta))


def _info_scalar(theta_fn, theta, t, h, a, x, da):
    al, de = np.exp(-t), 1 - np.exp(-2 * t)
    def Pt(th):
        p0 = np.exp(theta_fn(a, th)); p0 /= np.trapezoid(p0, a)
        o = np.empty_like(x)
        for s in range(0, len(x), 500):
            xx = x[s:s + 500, None]
            o[s:s + 500] = (np.exp(-0.5 * (xx - al * a[None, :]) ** 2 / de) / np.sqrt(2 * np.pi * de) * p0[None, :] * da).sum(1)
        return o
    P0 = Pt(theta)
    d1 = (np.log(Pt(theta + h)) - np.log(Pt(theta - h))) / (2 * h); d2 = (np.log(Pt(theta + h / 2)) - np.log(Pt(theta - h / 2))) / h
    return float(np.trapezoid(((4 * d2 - d1) / 3) ** 2 * P0, x))


def part_constant(cfg, out):
    a = np.linspace(-cfg["scalar_a"], cfg["scalar_a"], cfg["scalar_n"]); da = a[1] - a[0]
    x = np.linspace(-cfg["scalar_x"], cfg["scalar_x"], cfg["scalar_nx"])
    rows = []
    for beta in cfg["betas"]:
        dK = (K(beta + 5e-4) - K(beta - 5e-4)) / 1e-3; pred = dK**2 / 24
        for t in cfg["scalar_levels"]:
            I = _info_scalar(_logp_gg, beta, t, cfg["h"], a, x, da)
            rows.append(dict(coordinate=f"beta={beta:g}", t=t, I_scaled=I * np.exp(8 * t), predicted=pred, ratio=I * np.exp(8 * t) / pred))
            print(f"  [E21 constant] beta={beta:g} t={t:.1f}: I e^8t = {I*np.exp(8*t):.5f}  predicted {pred:.5f}  ratio {I*np.exp(8*t)/pred:.4f}", flush=True)
    for t in cfg["scalar_levels"][:3]:
        I = _info_scalar(lambda a, v: -0.5 * a**2 / v - 0.5 * np.log(2 * np.pi * v), 1.0, t, cfg["h"], a, x, da)
        rows.append(dict(coordinate="gaussian variance", t=t, I_scaled=I * np.exp(4 * t), predicted=0.5, ratio=I * np.exp(4 * t) / 0.5))
        print(f"  [E21 constant] variance t={t:.1f}: I e^4t = {I*np.exp(4*t):.5f} (predicted 0.5)", flush=True)
    write_csv(out / "scalar_constant.csv", rows)


PARTS = {"curve": part_curve, "constant": part_constant}


def main():
    ap = experiment_parser("e21_information_curve", "exact information curve at rho = 0.5; scalar check of the shape constant")
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
