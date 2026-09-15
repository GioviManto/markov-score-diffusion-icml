#!/usr/bin/env python
"""E9 — approximation floors of the mixture-innovation family (infinite data, no noise).

Question (docs/RESEARCH_PLAN.md, capacity resolution): when the fitted C-component kernel of
E02a sits at some error, is it limited by the data or by the family? This experiment
computes, for each capacity C, the population-optimal mixture — the C-component Gaussian
mixture (with the true rho and innovation variance q) that minimises KL(Laplace || mixture)
over the innovation law, fitted deterministically by weighted EM on a fine quadrature grid —
and scores its tree-exact BP posterior against the true-kernel reference on the SAME test
bundles E02a uses (`exp31-test`, seeds 0-15, 2048 chains, twelve levels). The result is a
per-C infinite-data REFERENCE for the family, paired seed by seed with E02a's rows through
the bundle hash. It is not a bound: the KL-optimal innovation fit and the pooled-score-error
optimal C-component kernel are different objects, and a finite-sample fit made through the
noisy likelihood can score below the KL fit. What it supports is the reading "the fitted
C-component kernel is within a factor X of the family's infinite-data KL fit" and, through
the C = 1 row, a consistency check: the single-Gaussian population fit is the covariance
twin, so its error must equal the LMMSE floor of Proposition 1 on the same bundle.

Cells: one per seed (`common.run_cells`); the mixture fits are seed-independent and cached
in floor_fits.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from e02_capacity import REGIONS, T_SCHEDULE, _bundle, _bundle_hash, _risk, _sq  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.kernels import MixtureInnovationKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.utils import ensure_dir, write_csv, write_json  # noqa: E402

N_SITES = FROZEN.n_sites

SETTINGS = dict(
    components=(1, 2, 4, 8, 16), seeds=tuple(range(16)), n_test=2048,
    grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    fit_grid=4001, fit_halfwidth_b=14.0, em_iters=3000, em_tol=1e-12, inits=8,
)
QUICK = dict(components=(1, 2, 8), seeds=(10_000,), n_test=64, grid_size=201, fit_grid=801, em_iters=300, inits=2)


# ----------------------------------------------------------------------------- population fit


def _laplace_logpdf(u, b):
    return -np.abs(u) / b - np.log(2.0 * b)


def _mixture_logpdf(u, pi, mu, s2):
    z = -0.5 * (u[:, None] - mu[None]) ** 2 / s2[None] - 0.5 * np.log(2 * np.pi * s2[None]) + np.log(pi[None])
    m = z.max(axis=1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(z - m).sum(axis=1)))


def _weighted_em(u, w, pi, mu, s2, iters, tol):
    """EM for a Gaussian mixture on weighted points (w sums to 1): the population KL minimiser."""
    prev = -np.inf
    for it in range(iters):
        z = -0.5 * (u[:, None] - mu[None]) ** 2 / s2[None] - 0.5 * np.log(2 * np.pi * s2[None]) + np.log(pi[None])
        m = z.max(axis=1, keepdims=True)
        r = np.exp(z - m); ll = float((w * (m[:, 0] + np.log(r.sum(axis=1)))).sum()); r /= r.sum(axis=1, keepdims=True)
        wr = w[:, None] * r
        pi = wr.sum(axis=0); pi = np.maximum(pi, 1e-300)
        mu = (wr * u[:, None]).sum(axis=0) / pi
        s2 = (wr * (u[:, None] - mu[None]) ** 2).sum(axis=0) / pi
        s2 = np.maximum(s2, 1e-10); pi = pi / pi.sum()
        if ll - prev < tol:
            break
        prev = ll
    return pi, mu, s2, ll, it + 1


def population_fit(C, b, cfg, rng):
    """Best of several EM runs from symmetric scale-mixture and random-mean initialisations;
    returns (pi, mu, s2, kl, n_iter, init_kind). The Laplace law is a scale mixture of Gaussians,
    so zero-mean components with spread scales are the natural family; random means guard
    against that being a bad local optimum."""
    A = cfg["fit_halfwidth_b"] * b
    u = np.linspace(-A, A, cfg["fit_grid"]); du = u[1] - u[0]
    w = np.exp(_laplace_logpdf(u, b)) * du; w = w / w.sum()
    h_true = float((w * _laplace_logpdf(u, b)).sum())
    best = None
    inits = []
    q = 2 * b * b
    for k in range(cfg["inits"]):
        if k == 0:
            pi = np.full(C, 1.0 / C); mu = np.zeros(C); s2 = q * np.geomspace(0.15, 6.0, C) if C > 1 else np.array([q])
            kind = "scale-mixture"
        else:
            pi = rng.dirichlet(np.ones(C)); mu = rng.normal(0, 0.5 * np.sqrt(q), C) if C > 1 else np.zeros(1)
            s2 = q * rng.uniform(0.2, 3.0, C); kind = "random"
        pi, mu, s2, ll, n = _weighted_em(u, w, pi, mu, s2, cfg["em_iters"], cfg["em_tol"])
        kl = h_true - ll
        inits.append(dict(kind=kind, kl=kl, iters=n))
        if best is None or kl < best[3]:
            best = (pi, mu, s2, kl, n, kind)
    return best, inits


# ----------------------------------------------------------------------------- cells


def _cell(args):
    cfg, seed, fits = args
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    prior = LaplaceAR1(FROZEN.rho)
    log_k_true = prior.log_transition_matrix(grid)
    sampler = lambda rng: prior.sample(rng, N_SITES)
    _, test = _bundle(sampler, log_k_true, grid, weights, "exp31-test", seed, cfg["n_test"])
    base = dict(seed=seed, test_bundle=_bundle_hash(test))
    rows = []
    for C in cfg["components"]:
        f = fits[str(C)]
        kb = MixtureInnovationKernel(rho=FROZEN.rho, pi=np.array(f["pi"]), mu=np.array(f["mu"]), s2=np.array(f["s2"]))
        log_k = kb.log_transition_matrix(grid)
        per_region = {r: [] for r in REGIONS}
        for t in T_SCHEDULE:
            X, m_ref = test[t]
            al, de = alpha_delta(t)
            m, _ = grid_bp_batch(grid, weights, log_k, X, al, de)
            for r, sl in REGIONS.items():
                per_region[r].append(_sq(m, m_ref, X, t, sl))
        for r in REGIONS:
            pairs = per_region[r]
            rows.append({**base, "arm": "population_mixture", "components": C, "region": r, "kl": f["kl"],
                         "sq_err": sum(e for e, _ in pairs), "sq_ref": sum(x for _, x in pairs), "risk": _risk(pairs),
                         "fit_kurt": float(kb.innovation_moments["innovation_excess_kurtosis"]), "resolved": bool(kb.scale_diagnostics(grid)["resolved"])})
        print(f"  [E09] seed={seed} C={C} kl={f['kl']:.2e} risk(all)={rows[-3]['risk']:.4f} kurt={rows[-3]['fit_kurt']:.2f}", flush=True)
    return rows


def part_floor(cfg, out, workers):
    b = LaplaceAR1(FROZEN.rho).b
    rng = np.random.default_rng(20260915)
    fits = {}
    for C in cfg["components"]:
        (pi, mu, s2, kl, n, kind), inits = population_fit(C, b, cfg, rng)
        fits[str(C)] = dict(pi=pi.tolist(), mu=mu.tolist(), s2=s2.tolist(), kl=kl, iters=n, init=kind, all_inits=inits)
        print(f"  [E09 fit] C={C}: KL(Laplace||mixture)={kl:.3e} after {n} EM steps ({kind})", flush=True)
    write_json(out / "floor_fits.json", fits)
    cells = [(cfg, s, fits) for s in cfg["seeds"]]
    rows = run_cells(cells, _cell, workers, out, "floor.csv")
    write_csv(out / "floor.csv", rows)


PARTS = {"floor": part_floor}


def main():
    ap = experiment_parser("e09_capacity_floor", "population-optimal mixture floors per capacity on the E02a test bundles")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.list_parts:
        print("\n".join(PARTS)); return
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    out = ensure_dir(Path(args.output_dir))
    for name, fn in select_parts(PARTS, args.only).items():
        fn(cfg, out, args.workers)
        write_json(out / f"params_{name}.json", {**provenance(), "settings": cfg, "quick": bool(args.quick), "workers": args.workers})


if __name__ == "__main__":
    main()
