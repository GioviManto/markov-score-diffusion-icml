#!/usr/bin/env python
"""E16 — the fourth-order (Edgeworth) correction to the covariance twin, evaluated exactly.

Theorem 3 says the covariance twin misses, to leading order, a single object: the degree-4 Hermite
term carried by kappa_4. If so, adding that term back should close the closure gap wherever the
expansion converges. This experiment adds it back exactly -- the first Edgeworth correction around
the twin, using the true Sigma_0 and the true kappa_4 tensor -- and measures how much of the gap it
closes at every level. It is the moment-method score estimator that the low-SNR literature suggests
(a Gram--Charlier score), here with oracle moments, so it bounds what any method built on
fourth-order moments can do.

    P_t ~ P_t^G (1 + eps),  eps = (1/24) sum kappa4^(t)[ijkl] H^{Sigma_t}_{ijkl}(x),  kappa4^(t) = alpha^4 kappa4(a)
    y = Sigma_t^{-1} x, A[m,l] = sum_{jk} kappa4^(t)[m,j,k,l] (Sigma_t^{-1})[j,k]
    eps = (K[y,y,y,y] - 6 y^T A y + 3 <A, Sigma_t^{-1}>) / 24
    grad eps = (1/6) Sigma_t^{-1} K[.,y,y,y] - (1/2) Sigma_t^{-1} A y
    S^E4 = S^G + grad eps / (1 + eps)   ("log" form)     or     S^G + grad eps   ("lin" form)

Relative score error per level and pooled, against tree-exact BP, E01's closure protocol.
Prediction: on families with an exponential moment the fraction of the gap closed -> 1 as t grows
and the correction diverges below the crossover; on Student-t it cannot close the tail term.
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
from markov_score.priors import GaussianMixtureAR1, LaplaceAR1, StudentTAR1, UniformAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

L = FROZEN.n_sites
T = tuple(sorted(set(FROZEN.t_grid) | {3.5, 4.0}))

SETTINGS = dict(families=("laplace", "bimodal", "uniform", "student"), n_chains=512, n_grid=801, half_width=8.0, rho=FROZEN.rho)
QUICK = dict(families=("laplace",), n_chains=32, n_grid=201)


def _families(rho):
    return {"laplace": LaplaceAR1(rho), "bimodal": GaussianMixtureAR1(rho, 0.6), "uniform": UniformAR1(rho), "student": StudentTAR1(rho, 5)}


def part_oracle(cfg, out):
    grid, w = make_grid(cfg["half_width"], cfg["n_grid"])
    rho = cfg["rho"]
    sig0 = theory.chain_covariance(rho, 1 - rho**2, L)
    fams = _families(rho)
    rows = []
    for name in cfg["families"]:
        prior = fams[name]
        logK = prior.log_transition_matrix(grid)
        k4e = theory.excess_kurtosis_to_kappa4(prior.innovation_excess_kurtosis, prior.q)
        K4 = theory.chain_kappa4(rho, k4e, L)
        rng = rng_for("e01-closure", name, 0)
        A = np.stack([prior.sample(rng, L) for _ in range(cfg["n_chains"])]); Z = rng.standard_normal(A.shape)
        pooled = dict(G=0.0, lin=0.0, log=0.0, ref=0.0)
        for t in T:
            al, de = alpha_delta(t)
            X = al * A + np.sqrt(de) * Z
            m_bp, _ = grid_bp_batch(grid, w, logK, X, al, de)
            s_bp = theory.score_from_mean(X, m_bp, t)
            st = al**2 * sig0 + de * np.eye(L); P = np.linalg.inv(st)
            sG = -(X @ P)
            Kt = al**4 * K4
            Y = X @ P
            T1 = np.einsum("mjkl,bj,bk,bl->bm", Kt, Y, Y, Y, optimize=True)
            Amat = np.einsum("mjkl,jk->ml", Kt, P, optimize=True)
            AP = float(np.einsum("ml,ml->", Amat, P))
            eps = (np.einsum("bm,bm->b", T1, Y) - 6 * np.einsum("bm,ml,bl->b", Y, Amat, Y) + 3 * AP) / 24.0
            grad = (T1 @ P) / 6.0 - 0.5 * (Y @ Amat.T) @ P
            s_lin = sG + grad
            s_log = sG + grad / np.maximum(1 + eps, 0.1)[:, None]
            ref = float((s_bp**2).sum())
            eG = float(((sG - s_bp) ** 2).sum()); elin = float(((s_lin - s_bp) ** 2).sum()); elog = float(((s_log - s_bp) ** 2).sum())
            if t in FROZEN.t_grid:
                pooled["G"] += eG; pooled["lin"] += elin; pooled["log"] += elog; pooled["ref"] += ref
            rows.append(dict(family=name, t=t, rel_lmmse=np.sqrt(eG / ref), rel_e4_lin=np.sqrt(elin / ref), rel_e4_log=np.sqrt(elog / ref),
                             gap_closed_lin=1 - elin / eG, gap_closed_log=1 - elog / eG, eps_min=float(eps.min()), eps_max=float(eps.max())))
            print(f"  [E16] {name:8s} t={t:.3f} lmmse {np.sqrt(eG/ref):.2e}  E4-lin {np.sqrt(elin/ref):.2e}  E4-log {np.sqrt(elog/ref):.2e}  gap closed (log) {1-elog/eG:+.3f}", flush=True)
        rows.append(dict(family=name, t=-1.0, rel_lmmse=np.sqrt(pooled["G"] / pooled["ref"]), rel_e4_lin=np.sqrt(pooled["lin"] / pooled["ref"]),
                         rel_e4_log=np.sqrt(pooled["log"] / pooled["ref"]), gap_closed_lin=np.nan, gap_closed_log=np.nan, eps_min=np.nan, eps_max=np.nan))
    write_csv(out / "edgeworth.csv", rows)


PARTS = {"oracle": part_oracle}


def main():
    ap = experiment_parser("e16_edgeworth_oracle", "the exact fourth-order correction to the covariance twin")
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
        write_json(out / f"params_{name}.json", {**provenance(), "settings": cfg, "quick": bool(args.quick), "t": T, "frozen": FROZEN.as_dict()})


if __name__ == "__main__":
    main()
