#!/usr/bin/env python
"""E6 — learning through noise: recovery of correlation and shape versus observation level.

Theorem B2 (docs/THEORY_NOTES.md) says one noised sequence at level t carries
information ∝ e^{-4t} about a covariance coordinate and ∝ e^{-8t} about a shape
coordinate (beyond the crossover t_*). So, at fixed M, the RMSE of an estimate of rho
from data observed at a single level t should grow like e^{2t} and that of the fitted
innovation excess kurtosis like e^{4t}; their ratio like e^{2t}.

Design: Laplace chain at the frozen configuration; M = `n_chains` sequences all noised
at ONE level t (so this is a single-level marginal likelihood, not the twelve-level
split of the headline protocol); EM-BP with the mixture kernel (C = `components`),
checkpoint ladder, the checkpoint selected on a validation bundle noised at the same
level (bulk region); for every seed and level the fitted rho, innovation variance and
excess kurtosis, the Hellinger distance of the fitted kernel to the truth, and the
score error of the fitted kernel on a twelve-level test bundle (the headline
estimand) are recorded. The clean-data maximum-likelihood fit (t = 0, no channel) is
included as the reference arm.

Predeclared analysis (tools/make_learning_numbers.py): RMSE over seeds of rho_hat and
kurt_hat per level; log-slope of RMSE against t on the levels above t_*; the ratio.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from protocol import PROTOCOL  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.em import fit_em  # noqa: E402
from markov_score.kernels import MixtureInnovationKernel  # noqa: E402
from markov_score.metrics import transition_hellinger  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

N_SITES = FROZEN.n_sites
T_SCHEDULE = tuple(FROZEN.t_grid)
BULK = slice(4, N_SITES - 4)

SETTINGS = dict(
    levels=(0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.0),
    rho=FROZEN.rho,
    n_chains=512, n_val=512, n_test=1024,
    seeds=tuple(range(16)),
    components=8,
    grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    em_checkpoints=PROTOCOL.em_checkpoints,
    em_init_rho=PROTOCOL.em_init_rho, em_init_var=PROTOCOL.em_init_var,
    include_clean=True,
)
DEV = dict(seeds=tuple(range(10_000, 10_004)))
QUICK = dict(levels=(0.1, 1.0), n_chains=32, n_val=32, n_test=32, seeds=(10_000,), grid_size=201, em_checkpoints=(5, 10))


def _sq(m_hat, m_ref, X, t, sl):
    al, de = alpha_delta(t)
    s_hat = -(X[:, sl] - al * m_hat[:, sl]) / de
    s_ref = -(X[:, sl] - al * m_ref[:, sl]) / de
    return float(((s_hat - s_ref) ** 2).sum()), float((s_ref**2).sum())


def _risk(pairs):
    num = sum(e for e, _ in pairs); den = sum(r for _, r in pairs)
    return float(np.sqrt(num / den)) if den > 0 else float("nan")


def _cell(args):
    cfg, seed, t_obs = args
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    prior = LaplaceAR1(cfg["rho"])
    log_k_true = prior.log_transition_matrix(grid)
    rng = rng_for("e06-train", seed, t_obs)
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(cfg["n_chains"])])
    ck = tuple(int(c) for c in cfg["em_checkpoints"])
    k0 = MixtureInnovationKernel.init(cfg["components"], rho=cfg["em_init_rho"], var=cfg["em_init_var"], rng=rng_for("e06-init", seed, t_obs))
    # validation bundle at the same level; test bundle on the twelve-level schedule
    rng_v = rng_for("e06-val", seed, t_obs)
    Av = np.stack([prior.sample(rng_v, N_SITES) for _ in range(cfg["n_val"])])
    rng_te = rng_for("e06-test", seed)
    At = np.stack([prior.sample(rng_te, N_SITES) for _ in range(cfg["n_test"])])
    test = {}
    for t in T_SCHEDULE:
        al, de = alpha_delta(t)
        X = al * At + np.sqrt(de) * rng_te.standard_normal(At.shape)
        m_ref, _ = grid_bp_batch(grid, weights, log_k_true, X, al, de)
        test[t] = (X, m_ref)
    if t_obs > 0:
        al, de = alpha_delta(t_obs)
        X = al * A + np.sqrt(de) * rng_for("e06-noise", seed, t_obs).standard_normal(A.shape)
        Xv = al * Av + np.sqrt(de) * rng_v.standard_normal(Av.shape)
        mv_ref, _ = grid_bp_batch(grid, weights, log_k_true, Xv, al, de)
        groups = [(X, al, de)]
    else:
        # clean data: observe a itself; the channel at t -> 0 is the identity, and the
        # E-step reduces to the empirical pair distribution. Implemented as the channel
        # at a tiny level, which keeps one code path and one M-step.
        al, de = alpha_delta(1e-3)
        X = al * A + np.sqrt(de) * rng_for("e06-noise", seed, t_obs).standard_normal(A.shape)
        Xv = al * Av + np.sqrt(de) * rng_v.standard_normal(Av.shape)
        mv_ref, _ = grid_bp_batch(grid, weights, log_k_true, Xv, al, de)
        groups = [(X, al, de)]
    _, trace, saved = fit_em(k0, grid, weights, groups, n_iters=max(ck), tol=FROZEN.em_loglik_tol, checkpoints=set(ck))

    def _val_risk(kernel):
        lk = kernel.log_transition_matrix(grid)
        m, _ = grid_bp_batch(grid, weights, lk, Xv, al, de)
        return _risk([_sq(m, mv_ref, Xv, t_obs if t_obs > 0 else 1e-3, BULK)])

    best, best_risk = None, np.inf
    for it in sorted(saved):
        r = _val_risk(saved[it])
        if r < best_risk:
            best, best_risk = it, r
    kb = saved[best]
    mom = kb.innovation_moments
    hd = transition_hellinger(kb.log_transition_matrix(grid), log_k_true, grid, weights)
    lk = kb.log_transition_matrix(grid)
    pairs = []
    for t in T_SCHEDULE:
        X, m_ref = test[t]
        a_, d_ = alpha_delta(t)
        m, _ = grid_bp_batch(grid, weights, lk, X, a_, d_)
        pairs.append(_sq(m, m_ref, X, t, slice(0, N_SITES)))
    row = dict(seed=seed, t_obs=t_obs, n_chains=cfg["n_chains"], components=cfg["components"],
               checkpoint=best, at_cap=best == max(saved), stop_reason=trace.stop_reason,
               fit_rho=float(kb.rho), fit_ivar=float(mom["innovation_var"]), fit_kurt=float(mom["innovation_excess_kurtosis"]),
               true_rho=cfg["rho"], true_ivar=1.0 - cfg["rho"] ** 2, true_kurt=3.0,
               resolved=bool(kb.scale_diagnostics(grid)["resolved"]),
               test_risk_schedule=_risk(pairs), val_risk=best_risk,
               **{k: float(v) for k, v in hd.items() if np.isscalar(v)})
    print(f"  [E06] seed={seed} t_obs={t_obs} ckpt={best} rho={row['fit_rho']:.4f} kurt={row['fit_kurt']:.3f} risk={row['test_risk_schedule']:.4f}", flush=True)
    return [row]


def _run(cells, fn, workers, out=None, csv_name=None):
    """Run cells serially or in a spawn pool; each cell's rows are written by the worker to
    out/cells/<csv stem>/<key>.json and the CSV is rebuilt after every completion
    (experiments/common.py: run_cells; resumable, no result travels through the pool pipe)."""
    return run_cells(cells, fn, workers, out, csv_name)


def part_levels(cfg, out, workers):
    levels = list(cfg["levels"]) + ([0.0] if cfg["include_clean"] else [])
    cells = [(cfg, s, t) for s in cfg["seeds"] for t in levels]
    rows = _run(cells, _cell, workers, out, "levels.csv")
    write_csv(out / "levels.csv", rows)


PARTS = {"levels": part_levels}


def main():
    ap = experiment_parser("e06_learning_through_noise", "recovery of rho and shape vs observation level")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.list_parts:
        print("\n".join(PARTS)); return
    cfg = dict(SETTINGS)
    if args.dev:
        cfg.update(DEV)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    out = ensure_dir(Path(args.output_dir))
    for name, fn in select_parts(PARTS, args.only).items():
        print(f"[e06] part {name}: {len(cfg['seeds'])} seeds x {len(cfg['levels'])} levels, workers={args.workers}", flush=True)
        fn(cfg, out, args.workers)
        write_json(out / f"params_{name}.json", {**provenance(), "settings": cfg, "dev": bool(args.dev), "quick": bool(args.quick),
                                                   "workers": args.workers, "frozen": FROZEN.as_dict()})


if __name__ == "__main__":
    main()
