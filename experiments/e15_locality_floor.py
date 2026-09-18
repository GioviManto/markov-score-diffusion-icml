#!/usr/bin/env python
"""E15 — the locality floor: the error of the BEST estimator that sees only a window of radius r.

The manuscript reads the window head's plateau (0.121 -> 0.118 from M = 512 to 2048) as
"locality is a ceiling: a radius-4 window cannot represent a message that has crossed the
chain". That is a claim about information, and it is checkable exactly: the best radius-r
estimator of any kind is E[a_i | x_{i-r}, ..., x_{i+r}], which sum-product computes exactly on
the sub-chain once the exact marginal law of the window's first site is supplied as its initial
factor (mu_1 = N(0,1); mu_j = K mu_{j-1} on the grid). Its schedule-pooled relative score error is
a floor for every radius-r denoiser, learned or not. The Gaussian twin's windowed LMMSE is
computed alongside (closed form), and the decay |Sigma_t^{-1}|_{c,c+k} / |Sigma_t^{-1}|_{c,c} of
the exact Gaussian score's dependence on distance k (Proposition: influence radius r(rho, t)).

Reading: if the radius-4 floor at rho = 0.85 is far below the window head's 0.118, the plateau is
a property of that architecture and its training, not of the radius, and the sentence changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score import theory  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

L = FROZEN.n_sites
T = tuple(FROZEN.t_grid)

SETTINGS = dict(rhos=(0.5, 0.85, 0.95), radii=(0, 1, 2, 3, 4, 6, 8, 12, 16, 31), n_chains=512,
                grid_size=FROZEN.n_grid, half_width=FROZEN.half_width)
QUICK = dict(rhos=(0.85,), radii=(0, 2, 4), n_chains=32, grid_size=201)


def part_floor(cfg, out):
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    rows = []
    for rho in cfg["rhos"]:
        prior = LaplaceAR1(rho)
        logK = prior.log_transition_matrix(grid)
        K = np.exp(logK)
        mus = [np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi)]
        for _ in range(1, L):
            m = K @ (mus[-1] * w)
            mus.append(m / np.sum(m * w))
        logmus = [np.log(np.maximum(m, 1e-300)) for m in mus]
        rng = rng_for("locality", rho)
        A = np.stack([prior.sample(rng, L) for _ in range(cfg["n_chains"])])
        Z = rng.standard_normal(A.shape)
        sig0 = theory.chain_covariance(rho, 1 - rho**2, L)
        num = {r: 0.0 for r in cfg["radii"]}; numG = {r: 0.0 for r in cfg["radii"]}; den = 0.0; numL = 0.0
        for t in T:
            al, de = alpha_delta(t)
            X = al * A + np.sqrt(de) * Z
            m_full, _ = grid_bp_batch(grid, w, logK, X, al, de)
            s_ref = -(X - al * m_full) / de
            e_ref = float((s_ref**2).sum()); den += e_ref
            st = al**2 * sig0 + de * np.eye(L); P = np.linalg.inv(st)
            m_lm = theory.lmmse_posterior_mean(X, sig0, t)
            e_lm = float(((-(X - al * m_lm) / de - s_ref) ** 2).sum()); numL += e_lm
            lev = dict(rho=rho, t=t, lmmse=np.sqrt(e_lm / e_ref), tstar=float(theory.crossover_time(rho, L=L)))
            for r in cfg["radii"]:
                m_win = np.empty_like(X); m_winG = np.empty_like(X)
                for i in range(L):
                    lo, hi = max(0, i - r), min(L, i + r + 1)
                    mm, _ = grid_bp_batch(grid, w, logK, X[:, lo:hi], al, de, log_mu=logmus[lo])
                    m_win[:, i] = mm[:, i - lo]
                    sub = sig0[lo:hi, lo:hi]; stw = al**2 * sub + de * np.eye(hi - lo)
                    m_winG[:, i] = X[:, lo:hi] @ (al * sub[i - lo] @ np.linalg.inv(stw))
                e = float(((-(X - al * m_win) / de - s_ref) ** 2).sum()); num[r] += e
                eG = float(((-(X - al * m_winG) / de - s_ref) ** 2).sum()); numG[r] += eG
                lev[f"floor_r{r}"] = np.sqrt(e / e_ref); lev[f"gauss_r{r}"] = np.sqrt(eG / e_ref)
            c = L // 2
            for k in (1, 2, 4, 8):
                lev[f"influence_k{k}"] = abs(P[c, c + k]) / abs(P[c, c])
            rows.append(lev)
            print(f"  [E15] rho={rho} t={t:.3f} " + " ".join(f"r{r}:{lev[f'floor_r{r}']:.4f}" for r in cfg["radii"]) + f" lmmse:{lev['lmmse']:.4f}", flush=True)
        pooled = dict(rho=rho, t=-1.0, lmmse=np.sqrt(numL / den), tstar=float(theory.crossover_time(rho, L=L)))
        for r in cfg["radii"]:
            pooled[f"floor_r{r}"] = np.sqrt(num[r] / den); pooled[f"gauss_r{r}"] = np.sqrt(numG[r] / den)
        rows.append(pooled)
        print(f"  [E15] POOLED rho={rho}: " + " ".join(f"r{r}:{pooled[f'floor_r{r}']:.4f}" for r in cfg["radii"]) + f" lmmse:{pooled['lmmse']:.4f}", flush=True)
    write_csv(out / "locality.csv", rows)


PARTS = {"floor": part_floor}


def main():
    ap = experiment_parser("e15_locality_floor", "the exact best radius-r estimator against the full-chain score")
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
