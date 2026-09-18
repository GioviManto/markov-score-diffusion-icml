#!/usr/bin/env python
"""E17 — the asymptotic bias of the chain estimator under each E08 misspecification, in closed form.

Every E08 truth is Gaussian and the C = 8 mixture fits a Gaussian innovation to it (fitted excess
kurtosis 0.04-0.34), so the M -> infinity limit of EM-BP is the population KL projection of the noised
truth onto the Gaussian AR(1) family under the one-view protocol (equal mass on the twelve levels),

    theta*(eps) = argmin_(rho, q) sum_t KL( N(0, Sigma_t(eps)) || N(0, Sigma_t(rho, q)) ),

and its relative score error against the truth's exact score is a trace formula. No Monte Carlo.
Reported per mechanism and strength: theta*, the bias b(eps), the lag-one plug-in AR(1) bias (E08's
lmmse_ar1 arm), the frozen E08 EM-BP errors at M = 512 and 2048 for comparison, and the prediction
sqrt(b^2 + v/M) with v read from the well-specified cells; plus the small-eps expansion (log-log slope
and coefficient), which says at what order each mechanism bites.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score import theory  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.utils import ensure_dir, write_csv, write_json  # noqa: E402
import e08_misspecification as e08  # noqa: E402

L = FROZEN.n_sites; T = tuple(FROZEN.t_grid); RHO0 = FROZEN.rho

SETTINGS = dict(mechanisms=dict(e08.SETTINGS["mechanisms"]), small=(1e-3, 3e-3, 1e-2, 3e-2), e08_dir="outputs/E08", v_const=0.2)
QUICK = dict(mechanisms={"ar2": (0.0, 0.2)}, small=(1e-2,))


def pooled_err(S_true, S_model):
    num = den = 0.0
    for t in T:
        al, de = alpha_delta(t)
        St = al**2 * S_true + de * np.eye(L); Sm = al**2 * S_model + de * np.eye(L)
        Pt = np.linalg.inv(St); D = np.linalg.inv(Sm) - Pt
        num += np.trace(D @ St @ D); den += np.trace(Pt)
    return float(np.sqrt(num / den))


def kl_project(S_true):
    def obj(p):
        rho, q = np.tanh(p[0]), np.exp(p[1])
        Sm0 = theory.chain_covariance(rho, q, L)
        v = 0.0
        for t in T:
            al, de = alpha_delta(t)
            St = al**2 * S_true + de * np.eye(L); Sm = al**2 * Sm0 + de * np.eye(L)
            _, ld = np.linalg.slogdet(Sm)
            v += 0.5 * (ld + np.trace(np.linalg.solve(Sm, St)))
        return v
    r = minimize(obj, x0=[np.arctanh(0.8), np.log(0.3)], method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-10, maxiter=4000))
    return float(np.tanh(r.x[0])), float(np.exp(r.x[1]))


def part_bias(cfg, out):
    emp = None
    src = Path(cfg["e08_dir"]) / "misspecification.csv"
    if src.exists():
        import pandas as pd
        E = pd.read_csv(src); E = E[E.region == "all"]
        emp = E.pivot_table(index=["mechanism", "eps", "n_chains"], columns="arm", values="risk", aggfunc="mean")
        rho_emp = E[E.arm == "em_bp"].groupby(["mechanism", "eps", "n_chains"]).fit_rho.mean()
    rows = []
    for mech, epss in cfg["mechanisms"].items():
        for eps in epss:
            S = e08._truth(mech, eps, L, RHO0).sigma0
            rho_h, q_h = kl_project(S)
            bias = pooled_err(S, theory.chain_covariance(rho_h, q_h, L))
            lag1 = float(np.mean(np.diag(S, 1)))
            bias_lag1 = pooled_err(S, lag1 ** np.abs(np.subtract.outer(np.arange(L), np.arange(L))))
            row = dict(mechanism=mech, eps=eps, rho_proj=rho_h, q_proj=q_h, bias_kl=bias, bias_lag1=bias_lag1,
                       pred_512=np.sqrt(bias**2 + cfg["v_const"] / 512), pred_2048=np.sqrt(bias**2 + cfg["v_const"] / 2048))
            if emp is not None:
                for n in (512, 2048):
                    row[f"em_{n}"] = float(emp.loc[(mech, eps, n), "em_bp"]) if (mech, eps, n) in emp.index else np.nan
                    row[f"rho_em_{n}"] = float(rho_emp.get((mech, eps, n), np.nan))
            rows.append(row)
            print(f"  [E17] {mech:9s} eps={eps:<5g} rho*={rho_h:.4f} q*={q_h:.4f} bias={bias:.4f} (lag1 plug-in {bias_lag1:.4f}) "
                  + (f"| EM-BP {row.get('em_512', np.nan):.4f} / {row.get('em_2048', np.nan):.4f}" if emp is not None else ""), flush=True)
        b = []
        for eps in cfg["small"]:
            S = e08._truth(mech, eps, L, RHO0).sigma0; rho_h, q_h = kl_project(S); b.append(pooled_err(S, theory.chain_covariance(rho_h, q_h, L)))
        b = np.array(b); sm = np.array(cfg["small"])
        sl = np.diff(np.log(b)) / np.diff(np.log(sm)) if len(b) > 1 else np.array([np.nan])
        rows.append(dict(mechanism=mech, eps=-1.0, bias_kl=np.nan, bias_lag1=np.nan, rho_proj=np.nan, q_proj=np.nan, pred_512=np.nan, pred_2048=np.nan,
                         small_eps=";".join(f"{x:g}" for x in sm), small_bias=";".join(f"{x:.4e}" for x in b),
                         local_slope=float(sl[-1]), coefficient=float(b[-1] / sm[-1] ** sl[-1]) if np.isfinite(sl[-1]) else np.nan))
        print(f"  [E17] {mech}: bias ~ eps^{sl[-1]:.2f}, coefficient {rows[-1]['coefficient']:.3f}", flush=True)
    write_csv(out / "bias.csv", rows)


PARTS = {"bias": part_bias}


def main():
    ap = experiment_parser("e17_misspec_bias", "closed-form asymptotic bias of the AR(1) estimator under misspecification")
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
        write_json(out / f"params_{name}.json", {**provenance(), "settings": {k: (v if not isinstance(v, dict) else str(v)) for k, v in cfg.items()}, "quick": bool(args.quick), "frozen": FROZEN.as_dict()})


if __name__ == "__main__":
    main()
