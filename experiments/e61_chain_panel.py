#!/usr/bin/env python
"""E61 — the same arms on seven chains with exactly known kernels.

Asked on 2 Nov 2026: every model so far was tested on the Laplace chain; test them on other chains whose kernels are known,
including "real" first-order Markov processes. Seven chains, all matched to the Laplace chain of E03 in variance (1) and
lag-one correlation (0.85), so a linear method sees the same covariance everywhere and every difference comes from shape
(calibration: tools/core/calibrate_e61.py, from the true kernels only):

    gauss     GaussianAR1(0.85)                                                        the control: linear is optimal
    laplace   LaplaceAR1(0.85)                                                         the anchor
    student   StudentTAR1(0.85, nu = 4)                                                heavy tails, in the shift class
    uniform   UniformAR1(0.85)                                                         bounded innovations
    skew      a sinh-arcsinh transformed Gaussian AR(1), marginal skewness 1           outside the shift class
    langevin  a double-well Langevin diffusion sampled every Delta = 0.3079            outside: nonlinear drift
    pearson   the Student (Pearson) diffusion, nu = 6, sampled every Delta = 0.1624    outside: state-dependent volatility

(markov_score.priors and markov_score.panel_chains). Arms. Kernels: mixture (C = 8), free4 (E60's free 2-D mixture,
C = 4) and energy (width 16, E56's screen winner, with the fresh-Adam-state rule), each by EM through exact grid BP from
one noisy view per training chain (E56's _fit_and_select: E55's ladder to 2400, the checkpoint with the lowest pooled
validation BP risk); energy_clean, the same energy kernel by clean-data maximum likelihood (E58's histogram Xi and the same
M-step, the same ladder and selection); the linear floor (lmmse). Networks: mlp and window, E03's screened winners trained
on the clean chains (E57's net cell).

E57's driver is reused IN PROCESS: its chain constructor, its data and its EM interface are replaced by E61's (E61's own
tags, e61-<chain>-...), so its network cell runs on E61's chains; e57_chain_families.py itself is not modified.

Parts
-----
kernels   chains x seeds x sizes x (lmmse + kernel arms) -> kernels.csv, one row per (chain, seed, n, arm, t)
nets      chains x seeds x sizes x networks -> nets.csv

PREDECLARED in experiments/registry/E61.json (P1-P5, D1-D5) before this file was written.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e57_chain_families as e57  # noqa: E402
import e58_clean_kernel as e58  # noqa: E402
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from markov_score.bp_grid import make_grid  # noqa: E402
from markov_score.em import clean_statistics  # noqa: E402
from markov_score.free_mixture_kernel import FreeMixtureKernel  # noqa: E402
from markov_score.kernels import MixtureInnovationKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.panel_chains import langevin_chain, pearson_chain, skew_chain  # noqa: E402
from markov_score.priors import GaussianAR1, LaplaceAR1, StudentTAR1, UniformAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

e03, e56 = e57.e03, e57.e56
N_SITES, T_SCHEDULE = e57.N_SITES, e57.T_SCHEDULE
CHAINS = ("gauss", "laplace", "student", "uniform", "skew", "langevin", "pearson")
KERNEL_ARMS = ("energy", "energy_clean", "free4", "mixture")       # slowest first, for the worker pool's tail
NETS = ("mlp", "window")
RHO = 0.85

SETTINGS = dict(e57.SETTINGS, chains=CHAINS, kernel_arms=KERNEL_ARMS, net_archs=NETS, energy_hidden=16,
                free_lr=1e-2, free_inner=10)
QUICK = dict(e57.QUICK, chains=CHAINS, kernel_arms=KERNEL_ARMS, net_archs=NETS, energy_hidden=8)


def chain_prior(chain: str):
    if chain == "gauss":
        return GaussianAR1(RHO)
    if chain == "laplace":
        return LaplaceAR1(RHO)
    if chain == "student":
        return StudentTAR1(RHO, 4.0)
    if chain == "uniform":
        return UniformAR1(RHO)
    if chain == "skew":
        return skew_chain()
    if chain == "langevin":
        return langevin_chain()
    if chain == "pearson":
        return pearson_chain()
    raise ValueError(f"unknown chain {chain!r}")


# ---------------------------------------------------------------------------------------- data (E57's, E61's tags)

def _cell_common(cfg, chain, seed, n):
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    prior = chain_prior(chain)
    log_k = prior.log_transition_matrix(grid)
    rng_train = rng_for(f"e61-{chain}-train", seed, n)
    A = np.stack([prior.sample(rng_train, N_SITES) for _ in range(n)])
    _, val = e03._bundle(prior, log_k, grid, weights, f"e61-{chain}-val", seed, cfg["n_val"])
    _, test = e03._bundle(prior, log_k, grid, weights, f"e61-{chain}-test", seed, cfg["n_test"])
    base = dict(chain=chain, seed=seed, n_chains=n, val_bundle=e03._bundle_hash(val), test_bundle=e03._bundle_hash(test))
    return A, val, test, base, grid, weights, prior, log_k


def _one_view_groups(chain, A, seed):
    idx = np.array_split(rng_for(f"e61-{chain}-em-split", seed).permutation(len(A)), len(T_SCHEDULE))
    noise = rng_for(f"e61-{chain}-em-noise", seed)
    out = []
    for t, ix in zip(T_SCHEDULE, idx):
        al, de = alpha_delta(t)
        sub = A[ix]
        out.append((al * sub + np.sqrt(de) * noise.standard_normal(sub.shape), al, de))
    return out


# In process only: E57's network cell reads its module globals at call time, so it now runs on E61's chains and tags.
e57.chain_prior = chain_prior
e57._cell_common = _cell_common
e57._one_view_groups = _one_view_groups


# ---------------------------------------------------------------------------------------- arms

def _init(cfg, chain, arm, seed, n):
    rho, var = cfg["em_init_rho"], cfg["em_init_var"]
    if arm == "mixture":
        return MixtureInnovationKernel.init(8, rho=rho, var=var, rng=rng_for("e61-mixture-init", chain, seed, n))
    if arm == "free4":
        return FreeMixtureKernel.init(4, rho=rho, var=var, lr=cfg["free_lr"], n_inner=cfg["free_inner"])
    if arm in ("energy", "energy_clean"):
        return replace(e56._init_kernel(cfg, "energy", seed, n, cfg["energy_hidden"]), fresh_on_reject=True)
    raise ValueError(f"unknown arm {arm!r}")


def _hp(cfg, arm):
    if arm == "mixture":
        return json.dumps({"C": 8})
    if arm == "free4":
        return json.dumps({"C": 4, "lr": cfg["free_lr"], "n_inner": cfg["free_inner"]})
    return json.dumps({"hidden": int(cfg["energy_hidden"]), "lr": cfg["energy_lr"], "n_inner": cfg["energy_inner"],
                       "fresh_on_reject": True, "data": "clean" if arm == "energy_clean" else "one noisy view"})


def _fit_clean_and_select(cfg, k0, A, grid, weights, val):
    """E58's clean-data maximum likelihood on E56's ladder, the checkpoint with the lowest pooled validation BP risk."""
    t0 = time.perf_counter()
    stats = clean_statistics(grid, A)
    saved, sec_step = e58.fit_clean_ladder(k0, grid, stats, set(cfg["ladder"]))
    wall = time.perf_counter() - t0
    curve = {it: e03._risk(e56._pairs(saved[it], val, grid, weights)) for it in sorted(saved)}
    best = min(curve, key=curve.get)
    k = saved[best]
    return k, {"checkpoint": best, "at_cap": int(best == max(saved)), "n_params": int(np.asarray(k.theta).size),
               "val_risk": curve[best], "val_curve": ";".join(f"{it}:{r:.6g}" for it, r in curve.items()),
               "wall": wall, "sec_per_iter": sec_step, "final_lr": float(getattr(k, "lr", float("nan")))}


def _kernel_cell(args):
    cfg, chain, seed, n, arm = args
    import torch
    torch.set_num_threads(int(cfg["torch_threads"]))
    A, val, test, base, grid, weights, prior, log_k = _cell_common(cfg, chain, seed, n)
    row0 = {**base, "arm": arm}
    if arm == "lmmse":
        mean, cov = e57.chain_covariance(log_k, grid, weights)
        pairs = e57._lmmse_pairs(cov, mean, test)
        pooled = e03._risk(pairs)
        return [{**row0, "t": t, "sq_err": e, "sq_ref": r, "risk_level": float(np.sqrt(e / r)) if r > 0 else np.nan,
                 "risk_pooled": pooled, "failed": 0} for t, (e, r) in zip(T_SCHEDULE, pairs)]
    k0 = _init(cfg, chain, arm, seed, n)
    try:
        if arm == "energy_clean":
            k, info = _fit_clean_and_select(cfg, k0, A, grid, weights, val)
        else:
            k, info = e56._fit_and_select(cfg, k0, _one_view_groups(chain, A, seed), grid, weights, val)
    except FloatingPointError as exc:       # E33 amendment 3: a failed fit is a row, and it counts against
        print(f"  [E61] {chain} seed={seed} n={n:>5} {arm:12s} FAILED: {exc}", flush=True)
        return [{**row0, "hp": _hp(cfg, arm), "t": t, "failed": 1, "stop_reason": f"FloatingPointError: {exc}",
                 "risk_pooled": np.nan} for t in T_SCHEDULE]
    pairs = e56._pairs(k, test, grid, weights)
    pooled = e03._risk(pairs)
    kh = e57.kernel_hellinger(k.log_transition_matrix(grid), log_k, grid, weights, prior)
    theta = ";".join(f"{v:.10g}" for v in np.asarray(k.theta, dtype=float))
    print(f"  [E61] {chain} seed={seed} n={n:>5} {arm:12s} pooled={pooled:.4f} ck={info['checkpoint']} H={kh:.4f}",
          flush=True)
    return [{**row0, "hp": _hp(cfg, arm), "t": t, "sq_err": e, "sq_ref": r,
             "risk_level": float(np.sqrt(e / r)) if r > 0 else np.nan, "risk_pooled": pooled, **info,
             "kernel_hellinger": kh, "theta": theta, "failed": 0} for t, (e, r) in zip(T_SCHEDULE, pairs)]


def _kernel_key(cell):
    _, chain, seed, n, arm = cell
    return f"{chain}_s{seed}_n{n}_{arm}"


# ---------------------------------------------------------------------------------------- parts

def part_kernels(cfg, out, workers):
    cells = [(cfg, c, s, n, a) for n in sorted(cfg["sizes"], reverse=True) for a in tuple(cfg["kernel_arms"]) + ("lmmse",)
             for c in cfg["chains"] for s in cfg["seeds"]]
    print(f"  kernels: {len(cells)} cells", flush=True)
    write_csv(Path(out) / "kernels.csv", run_cells(cells, _kernel_cell, workers, out, "kernels.csv", key=_kernel_key))


def part_nets(cfg, out, workers):
    cells = [(cfg, c, s, n, a) for n in sorted(cfg["sizes"], reverse=True) for c in cfg["chains"] for s in cfg["seeds"]
             for a in cfg["net_archs"]]
    write_csv(Path(out) / "nets.csv", run_cells(cells, e57._net_cell, workers, out, "nets.csv", key=e57._net_key))


def main() -> None:
    parser = experiment_parser("e61_chain_panel", "the same arms on seven chains with exactly known kernels")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    w = args.workers
    parts = {"kernels": ("kernel arms and the linear floor", lambda c, o: part_kernels(c, o, w)),
             "nets": ("the two networks", lambda c, o: part_nets(c, o, w))}
    if args.list_parts:
        print("\n".join(parts)); return
    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir)
    tag = "_".join(selected) if args.only else "all"
    resolved = {"n_sites": N_SITES, "t_schedule": list(T_SCHEDULE), "quick": args.quick,
                "parts": list(selected), "overrides": args.set, **cfg}
    write_json(out / f"params_{tag}.json", {**resolved, **provenance(resolved)})
    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(cfg, out)


if __name__ == "__main__":
    main()
