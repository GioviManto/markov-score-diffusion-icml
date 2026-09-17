#!/usr/bin/env python
"""E13 — does the headline result depend on the regime, or only on the estimator?

Every confirmatory number in this project is measured at one point of the model's parameter space:
a Laplace innovation, rho = 0.85, chain length 32. The capacity result -- a Gaussian-innovation
kernel pinned to the second-order floor while richer capacities go far below it -- is the load-
bearing one, and a reader is entitled to ask whether it is a property of the estimator or of that
particular corner.

This sweep re-runs the contrast that matters (C = 1 against C = 8, both against the LMMSE floor and
the isotropic reference on the same held-out bundles) across a grid of correlations and chain
lengths, with everything else at the frozen protocol. Two quantities are reported per cell: the
pooled relative score error of each arm, and the ratio of the C = 1 error to the floor, which
Proposition 1 says should be 1 at every point of the grid and is therefore a check on the whole
apparatus as much as on the result.

The correlation also moves the crossover t_* = 0.5 log lambda_max(Sigma_0), so the sweep doubles as
a test of whether the advantage tracks the theory's own scale rather than the schedule's.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score import theory  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.em import fit_em  # noqa: E402
from markov_score.kernels import MixtureInnovationKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402
from protocol import PROTOCOL  # noqa: E402

T_SCHEDULE = tuple(FROZEN.t_grid)

SETTINGS = dict(
    rhos=(0.3, 0.5, 0.7, 0.85, 0.95),
    lengths=(16, 32, 64),
    components=(1, 8),
    n_chains=512, n_val=512, n_test=1024,
    seeds=tuple(range(8)),
    grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    em_checkpoints=PROTOCOL.em_checkpoints,
    em_init_rho=PROTOCOL.em_init_rho, em_init_var=PROTOCOL.em_init_var,
)
QUICK = dict(rhos=(0.85,), lengths=(16,), components=(1, 8), n_chains=32, n_val=32, n_test=32,
             seeds=(10_000,), grid_size=201, em_checkpoints=(5, 10))


def _bundle(prior, log_k, grid, w, tag, seed, n_chains, L):
    rng = rng_for(tag, seed, L)
    A = np.stack([prior.sample(rng, L) for _ in range(n_chains)])
    out = {}
    for t in T_SCHEDULE:
        al, de = alpha_delta(t)
        X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
        m, _ = grid_bp_batch(grid, w, log_k, X, al, de)
        out[t] = (X, m)
    return A, out


def _risk(kernel_log_k, bundle, grid, w):
    num = den = 0.0
    for t in T_SCHEDULE:
        X, m_ref = bundle[t]
        al, de = alpha_delta(t)
        m, _ = grid_bp_batch(grid, w, kernel_log_k, X, al, de)
        s_hat = -(X - al * m) / de; s_ref = -(X - al * m_ref) / de
        num += float(((s_hat - s_ref) ** 2).sum()); den += float((s_ref ** 2).sum())
    return float(np.sqrt(num / den))


def _cell(args):
    cfg, rho, L, seed = args
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    prior = LaplaceAR1(rho)
    log_k_true = prior.log_transition_matrix(grid)
    _, val = _bundle(prior, log_k_true, grid, w, "e13-val", seed, cfg["n_val"], L)
    _, test = _bundle(prior, log_k_true, grid, w, "e13-test", seed, cfg["n_test"], L)
    rng_train = rng_for("e13-train", seed, L, rho)
    A = np.stack([prior.sample(rng_train, L) for _ in range(cfg["n_chains"])])
    sigma0 = theory.chain_covariance(rho, 1.0 - rho ** 2, L)
    base = dict(rho=rho, n_sites=L, seed=seed, n_chains=cfg["n_chains"],
                t_star=float(theory.crossover_time(rho, L=L)))

    rows = []
    # oracles on the same test bundle
    for arm, mean_fn in (("lmmse_true", lambda X, t: theory.lmmse_posterior_mean(X, sigma0, t)),
                         ("isotropic", lambda X, t: (X - (1 - np.exp(-2 * t)) * X) / np.exp(-t))):
        num = den = 0.0
        for t in T_SCHEDULE:
            X, m_ref = test[t]; al, de = alpha_delta(t)
            s_hat = -(X - al * mean_fn(X, t)) / de; s_ref = -(X - al * m_ref) / de
            num += float(((s_hat - s_ref) ** 2).sum()); den += float((s_ref ** 2).sum())
        rows.append({**base, "arm": arm, "components": 0, "risk": float(np.sqrt(num / den)),
                     "checkpoint": -1, "fit_kurt": np.nan})

    idx = np.array_split(rng_for("e13-split", seed, L).permutation(len(A)), len(T_SCHEDULE))
    noise = rng_for("e13-noise", seed, L)
    groups = []
    for t, ii in zip(T_SCHEDULE, idx):
        al, de = alpha_delta(t); sub = A[ii]
        groups.append((al * sub + np.sqrt(de) * noise.standard_normal(sub.shape), al, de))
    for C in cfg["components"]:
        k0 = MixtureInnovationKernel.init(C, rho=cfg["em_init_rho"], var=cfg["em_init_var"],
                                          rng=rng_for("e13-init", seed, L, C))
        ck = set(int(c) for c in cfg["em_checkpoints"])
        _, _, saved = fit_em(k0, grid, w, groups, n_iters=max(ck), tol=FROZEN.em_loglik_tol, checkpoints=ck)
        best, best_v = None, np.inf
        for it in sorted(saved):
            v = _risk(saved[it].log_transition_matrix(grid), val, grid, w)
            if v < best_v:
                best, best_v = it, v
        kb = saved[best]
        rows.append({**base, "arm": "em_bp", "components": C,
                     "risk": _risk(kb.log_transition_matrix(grid), test, grid, w),
                     "checkpoint": best,
                     "fit_kurt": float(kb.innovation_moments["innovation_excess_kurtosis"])})
        print(f"  [E13] rho={rho} L={L} seed={seed} C={C}: risk={rows[-1]['risk']:.4f} "
              f"kurt={rows[-1]['fit_kurt']:.2f} ckpt={best}", flush=True)
    return rows


def part_sweep(cfg, out, workers):
    cells = [(cfg, r, L, s) for r in cfg["rhos"] for L in cfg["lengths"] for s in cfg["seeds"]]
    print(f"  {len(cells)} cells", flush=True)
    rows = run_cells(cells, _cell, workers, out, "regimes.csv")
    write_csv(out / "regimes.csv", rows)


PARTS = {"sweep": part_sweep}


def main():
    ap = experiment_parser("e13_regime_sweep", "the capacity contrast across correlations and chain lengths")
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
