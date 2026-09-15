#!/usr/bin/env python
"""E5 — does score accuracy transfer to generation?

Arms (each a score function s(x, t) fed to the same reverse SDE with the SAME Brownian
path and initialisation, so the comparison is paired):
    exact     true-kernel grid BP (the reference sampler; its distance to the true law
              is the sampler's own discretisation error)
    em_bp     the mixture kernel fitted in E02 at the requested (seed, M, C), read from
              outputs/E02a/capacity.csv (theta column); validation-selected checkpoint
    lmmse     the covariance twin: S = -Sigma_t^{-1} x
    isotropic S = -x
    net:<arch> a torch checkpoint from E03 confirm (path in --set net_ckpts)  [optional]

Step study (part `steps`): the exact arm integrated at 100/200/400/800 steps on NESTED
Brownian paths (markov_score.reverse.nested_brownian_path), so the differences between
resolutions measure discretisation, not Monte Carlo scatter — the fix the archival study
left unwired.

Metrics: markov_score.gen_metrics.compare — predeclared in docs/RESEARCH_PLAN.md (E5).
The reference sample R is drawn from the true law at t_min (clean chain through the
channel), never from clean p_0, which was the archival target error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score import gen_metrics, theory  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.kernels import MixtureInnovationKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.reverse import reverse_sde_with_increments, time_grid  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

N_SITES = FROZEN.n_sites

SETTINGS = dict(
    sizes=(128, 2048), components=8, seeds=tuple(range(4)),
    n_generate=2000, n_reference=4000,
    t_max=3.0, t_min=0.02, n_steps=400,
    steps_ladder=(100, 200, 400, 800),
    grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    em_source="outputs/E02a/capacity.csv",
    net_ckpts=None,
)
QUICK = dict(sizes=(32,), seeds=(10_000,), components=8, n_generate=64, n_reference=128, n_steps=20, steps_ladder=(10, 20), grid_size=201,
             em_source="outputs/scratch/e02_quick/capacity.csv")


def _theta_from_e02(path, seed, n, C):
    df = pd.read_csv(path)
    row = df[(df.arm == "em_bp") & (df.seed == seed) & (df.n_chains == n) & (df.components == C) & (df.region == "all")]
    if len(row) == 0:
        raise SystemExit(f"no E02 fit for seed={seed} n={n} C={C} in {path}")
    theta = np.array([float(v) for v in row.theta.iloc[0].split(";")])
    return _mixture_from_theta(theta, C)


def _mixture_from_theta(theta, C):
    rho, pi, mu, s2 = theta[0], theta[1:1 + C], theta[1 + C:1 + 2 * C], theta[1 + 2 * C:1 + 3 * C]
    return MixtureInnovationKernel(rho=float(rho), pi=pi, mu=mu, s2=s2)


def _score_bp(log_k, grid, weights):
    def f(X, t):
        al, de = alpha_delta(t)
        m, _ = grid_bp_batch(grid, weights, log_k, X, al, de)
        return -(X - al * m) / de
    return f


def _score_lmmse(sigma0):
    return lambda X, t: theory.lmmse_score(X, sigma0, t)


def _score_iso():
    return lambda X, t: -X


def _arms(cfg, seed, n, grid, weights, log_k_true, sigma0):
    arms = {"exact": _score_bp(log_k_true, grid, weights), "lmmse": _score_lmmse(sigma0), "isotropic": _score_iso()}
    try:
        kb = _theta_from_e02(Path(__file__).resolve().parents[1] / cfg["em_source"], seed, n, cfg["components"])
        arms["em_bp"] = _score_bp(kb.log_transition_matrix(grid), grid, weights)
    except SystemExit as e:
        print("  ", e)
    if cfg.get("net_ckpts"):
        import torch
        from markov_score import nn_torch as nt
        for name, spec in cfg["net_ckpts"].items():
            model = nt.build(spec["arch"], N_SITES, **spec["hp"]); model.load_state_dict(torch.load(spec["path"]))
            den = nt.Denoiser(model, spec["mode"])
            arms[f"net:{name}"] = (lambda d: (lambda X, t: -(X - np.exp(-t) * d.posterior_mean(X, t)) / (1 - np.exp(-2 * t))))(den)
    return arms


def part_generate(cfg, out):
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    prior = LaplaceAR1(FROZEN.rho)
    log_k_true = prior.log_transition_matrix(grid)
    sigma0 = theory.chain_covariance(FROZEN.rho, FROZEN.innovation_variance, N_SITES)
    times = time_grid(cfg["t_max"], cfg["t_min"], cfg["n_steps"])
    rows = []
    for seed in cfg["seeds"]:
        for n in cfg["sizes"]:
            rng = rng_for("e05-gen", seed, n)
            x_init = rng.standard_normal((cfg["n_generate"], N_SITES))
            dW = np.sqrt(np.diff(-times))[:, None, None] * rng.standard_normal((len(times) - 1, cfg["n_generate"], N_SITES))
            rng_ref = rng_for("e05-ref", seed)
            A = np.stack([prior.sample(rng_ref, N_SITES) for _ in range(cfg["n_reference"])])
            al, de = alpha_delta(cfg["t_min"])
            R = al * A + np.sqrt(de) * rng_ref.standard_normal(A.shape)
            for arm, score in _arms(cfg, seed, n, grid, weights, log_k_true, sigma0).items():
                G = reverse_sde_with_increments(x_init, score, times, dW)
                m = gen_metrics.compare(G, R, FROZEN.rho, sigma0, grid, weights, log_k_true, cfg["t_min"], seed=seed)
                rows.append(dict(seed=seed, n_chains=n, arm=arm, n_steps=cfg["n_steps"], **m))
                print(f"  [E05] seed={seed} n={n} {arm:10s} kurt {m['kurt_gen']:.3f} (ref {m['kurt_ref']:.3f})  loglik gap {m['loglik_gap']:.3f}  sw2 {m['sw2']:.4f}", flush=True)
    write_csv(out / "generation.csv", rows)


def part_steps(cfg, out):
    """Exact arm at nested resolutions on one Brownian path: strong error against the finest."""
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    prior = LaplaceAR1(FROZEN.rho)
    log_k_true = prior.log_transition_matrix(grid)
    score = _score_bp(log_k_true, grid, weights)
    ladder = sorted(cfg["steps_ladder"])
    fine = max(ladder)
    rng = rng_for("e05-steps", 0)
    n = min(cfg["n_generate"], 500)
    x_init = rng.standard_normal((n, N_SITES))
    times_fine = time_grid(cfg["t_max"], cfg["t_min"], fine)
    dW_fine = np.sqrt(np.diff(-times_fine))[:, None, None] * rng.standard_normal((fine, n, N_SITES))
    G_fine = reverse_sde_with_increments(x_init, score, times_fine, dW_fine)
    rows = [dict(n_steps=fine, strong_rmse_vs_finest=0.0, kurt=gen_metrics.excess_kurtosis(gen_metrics.ar_residuals(G_fine, FROZEN.rho)))]
    for k in ladder[:-1]:
        if fine % k:
            continue
        r = fine // k
        times = time_grid(cfg["t_max"], cfg["t_min"], k)
        dW = dW_fine.reshape(k, r, n, N_SITES).sum(axis=1)
        G = reverse_sde_with_increments(x_init, score, times, dW)
        rows.append(dict(n_steps=k, strong_rmse_vs_finest=float(np.sqrt(((G - G_fine) ** 2).mean())),
                         kurt=gen_metrics.excess_kurtosis(gen_metrics.ar_residuals(G, FROZEN.rho))))
        print(f"  [E05 steps] {k}: strong rmse vs {fine} = {rows[-1]['strong_rmse_vs_finest']:.4f}", flush=True)
    write_csv(out / "steps.csv", rows)


PARTS = {"steps": part_steps, "generate": part_generate}


def main():
    ap = experiment_parser("e05_generation", "reverse-SDE generation under exact, fitted, LMMSE, isotropic and network scores")
    args = ap.parse_args()
    if args.list_parts:
        print("\n".join(PARTS)); return
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    out = ensure_dir(Path(args.output_dir))
    for name, fn in select_parts(PARTS, args.only).items():
        print(f"[e05] part {name}", flush=True)
        fn(cfg, out)
        write_json(out / f"params_{name}.json", {**provenance(), "settings": {k: (v if not isinstance(v, dict) else str(v)) for k, v in cfg.items()}, "quick": bool(args.quick), "frozen": FROZEN.as_dict()})


if __name__ == "__main__":
    main()
