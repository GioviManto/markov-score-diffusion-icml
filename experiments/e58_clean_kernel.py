#!/usr/bin/env python
"""E58 — the neural kernel learned by ordinary maximum likelihood on CLEAN sequences, then BP for the score, against a
global MLP and a window CNN trained by standard DSM on the same clean sequences.

With clean chains nothing is hidden, so EM collapses to plain maximum likelihood of the transitions,

    max_theta  sum_{n,k} log K_theta(a_{k+1} | a_k),

computed here as the clean-data histogram Xi (em.clean_statistics, the t -> 0 limit of the E-step) followed by the SAME
M-step code the noisy-EM arms of E56 use: clean and noisy fits differ only in Xi. BP enters only at use time, turning the
learned kernel and a noisy sequence into E[a | x] and the score at every level.

Chains: the Laplace chain on E03's bundles (`laplace`, seeds 0-15) and the Allen-Cahn field on E33's bundles (cells
T0.3_h1_L32 and T1_iso_L32, seeds 0-7), N from 32 to 32768. Kernel arms: energy_h16/h32/h64 (NeuralEnergyKernel), the
ideal family (laplace_clean, or pair_clean = E33's site energy on the field) and mixture_clean (C = 8); the linear floor
(lmmse) from the exact covariance. Network arms: global MLP and window CNN, E03's recipe and configurations, every size.

Parts
-----
kernels   (cell, seed, n, arm) -> kernels.csv, one row per level
nets      (cell, seed, n, arch) -> nets.csv, one row per level

PREDECLARED in experiments/registry/E58.json (P1-P6, D1-D6) before this file was written and before E56 was read.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from protocol import PROTOCOL  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.em import clean_statistics  # noqa: E402
from markov_score.energy_kernel import NeuralEnergyKernel  # noqa: E402
from markov_score.kernels import LaplaceAR1Kernel, MixtureInnovationKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

# Imported, not copied: E03's Laplace bundles, estimand and network recipe; E33's field cells and bundles; E56's ladder.
import e03_baselines as e03  # noqa: E402
import e33_field_learning as e33  # noqa: E402
import e56_kernel_families as e56  # noqa: E402

N_SITES = FROZEN.n_sites
T_SCHEDULE = tuple(FROZEN.t_grid)
FIELD_CELLS = ("T0.3_h1_L32", "T1_iso_L32")
KERNEL_ARMS = ("energy_h16", "energy_h32", "energy_h64", "ideal", "mixture_clean")
NET_ARCHS = ("mlp", "window")

SETTINGS = dict(
    cells=("laplace",) + FIELD_CELLS,
    laplace_seeds=tuple(range(16)), field_seeds=tuple(range(8)),
    sizes=(32, 128, 512, 2048, 8192, 32768),
    n_val=PROTOCOL.n_val, n_test=PROTOCOL.n_test, grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    kernel_arms=KERNEL_ARMS, ladder=e56.LADDER, ideal_ladder=(1, 2, 5, 10),
    energy_lr=1e-2, energy_inner=10, components=8,
    em_init_rho=PROTOCOL.em_init_rho, em_init_var=PROTOCOL.em_init_var,
    net_archs=NET_ARCHS,
    # E03's screened winners (outputs/E03/winners.json), written out so a cluster job does not need E03's outputs
    nets={"mlp": {"hp": {"hidden": [128, 128]}, "lr": 1e-3, "mode": "eps"},
          "window": {"hp": {"radius": 4, "width": 64}, "lr": 1e-3, "mode": "eps"}},
    net_steps=PROTOCOL.net_steps, net_checkpoints=PROTOCOL.net_checkpoints, batch_size=PROTOCOL.batch_size,
    grad_clip=PROTOCOL.grad_clip, torch_threads=1,
)
QUICK = dict(cells=("laplace", "T0.3_h1_L32"), laplace_seeds=(0,), field_seeds=(0,), sizes=(32,), n_val=32, n_test=32,
             grid_size=101, ladder=(2, 4), ideal_ladder=(1, 2), net_checkpoints=(50, 100))


# ---------------------------------------------------------------------------------------- data

def context(cfg, cell, seed, n):
    """Everything a cell needs, from the bundles of the experiment it pairs with (E03 for laplace, E33 for the field)."""
    if cell == "laplace":
        A, val, test, base = e03._cell_common(cfg, seed, n)
        grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
        prior = LaplaceAR1(FROZEN.rho)
        log_k = prior.log_transition_matrix(grid)
        rng = rng_for("exp31-val", seed)                     # e03._bundle draws the validation chains first
        a_val = np.stack([prior.sample(rng, N_SITES) for _ in range(cfg["n_val"])])
        i = np.arange(N_SITES)
        cov = FROZEN.rho ** np.abs(i[:, None] - i[None, :]).astype(float)
        return dict(cell=cell, A=A, a_val=a_val, val=val, test=test, grid=grid, w=w, log_k=log_k, log_mu=None,
                    mu_fn=None, cov=cov, base={**base, "cell": cell})
    _, grid, w, log_k, log_mu, sigma0, A, a_val, val, test, base = e33._common(cfg, cell, seed, n)
    keep = {k: base[k] for k in ("seed", "n_chains", "val_bundle", "test_bundle")}
    return dict(cell=cell, A=A, a_val=a_val, val=val, test=test, grid=grid, w=w, log_k=log_k, log_mu=log_mu,
                mu_fn=e33._stationary_mu(grid, w), cov=sigma0, base={**keep, "cell": cell})


def score_pairs(mean_fn, bundle):
    """(sq_err, sq_ref) per level with E03's estimand on every site (= e14._pooled once summed)."""
    return [e03._sq(mean_fn(X, t), m_ref, X, t, e03.REGIONS["all"]) for t, (X, m_ref) in
            ((t, bundle[t]) for t in T_SCHEDULE)]


def kernel_mean_fn(ctx, kernel):
    lk = kernel.log_transition_matrix(ctx["grid"])
    mu = ctx["mu_fn"](kernel) if ctx["mu_fn"] is not None else None
    return lambda X, t: grid_bp_batch(ctx["grid"], ctx["w"], lk, X, *alpha_delta(t), log_mu=mu)[0]


def lmmse_mean_fn(cov):
    def mean(X, t):
        al, de = alpha_delta(t)
        gain = al * cov @ np.linalg.inv(al**2 * cov + de * np.eye(len(cov)))
        return X @ gain.T
    return mean


def clean_nll(kernel, grid, stats):
    """Held-out clean negative log-likelihood per transition (binned, as the fit)."""
    return float(-(stats.xi * kernel.log_transition_matrix(grid)).sum() / max(stats.n_edges, 1))


def kernel_hellinger(log_k_hat, ctx) -> float:
    """sum_j pi_j w_j H(K_hat(.|g_j), K(.|g_j)) / sum_j pi_j w_j, pi the chain's stationary law (E56's measure)."""
    grid, w = ctx["grid"], ctx["w"]
    lh, lt = e56._normalised(log_k_hat, w), e56._normalised(ctx["log_k"], w)
    h = np.sqrt(np.clip(1.0 - (np.exp(0.5 * (lh + lt)) * w[:, None]).sum(axis=0), 0.0, None))
    if ctx["log_mu"] is not None:
        pi = np.exp(ctx["log_mu"])
    else:
        from markov_score.stationary import invariant_log_density_eig
        pi = np.exp(invariant_log_density_eig(LaplaceAR1(FROZEN.rho), grid, w).log_density)
    return float((pi * w * h).sum() / (pi * w).sum())


# ---------------------------------------------------------------------------------------- kernel arms

def init_kernel(cfg, cell, arm, seed, n, grid):
    rho, var = cfg["em_init_rho"], cfg["em_init_var"]
    if arm.startswith("energy_h"):
        s = int(rng_for("e58-energy-init", cell, seed, n, arm).integers(0, 2**31 - 1))
        return NeuralEnergyKernel.init(hidden=int(arm[len("energy_h"):]), seed=s, half_width=cfg["half_width"],
                                       lr=cfg["energy_lr"], n_inner=cfg["energy_inner"])
    if arm == "ideal":
        if cell == "laplace":
            return LaplaceAR1Kernel(rho=rho, b=float(np.sqrt(var / 2.0)))
        return e33._init_kernel("em_pair_quartic", cfg, grid, seed, cell, n)
    if arm == "mixture_clean":
        return MixtureInnovationKernel.init(cfg["components"], rho=rho, var=var, rng=rng_for("e58-mix-init", cell, seed, n))
    raise ValueError(f"unknown arm {arm!r}")


def fit_clean_ladder(kernel, grid, stats, ladder):
    """Repeated M-steps on the clean histogram, snapshotting at the ladder; also the seconds per step."""
    saved, cur, t0 = {}, kernel, time.perf_counter()
    top = max(ladder)
    for it in range(top + 1):
        if it in ladder:
            saved[it] = cur
        if it < top:
            cur = cur.m_step(stats, grid)
    return saved, (time.perf_counter() - t0) / max(top, 1)


def _kernel_cell(args):
    cfg, cell, seed, n, arm = args
    import torch
    torch.set_num_threads(int(cfg["torch_threads"]))
    ctx = context(cfg, cell, seed, n)
    grid = ctx["grid"]
    row0 = {**ctx["base"], "arm": arm}
    if arm == "lmmse":
        pairs = score_pairs(lmmse_mean_fn(ctx["cov"]), ctx["test"])
        pooled = e03._risk(pairs)
        return [{**row0, "t": t, "sq_err": e, "sq_ref": r, "risk_level": float(np.sqrt(e / r)), "risk_pooled": pooled,
                 "failed": 0} for t, (e, r) in zip(T_SCHEDULE, pairs)]
    stats, held = clean_statistics(grid, ctx["A"]), clean_statistics(grid, ctx["a_val"])
    ladder = set(cfg["ideal_ladder"] if (arm == "ideal" and cell == "laplace") else cfg["ladder"])
    k0 = init_kernel(cfg, cell, arm, seed, n, grid)
    t0 = time.perf_counter()
    try:
        saved, sec_step = fit_clean_ladder(k0, grid, stats, ladder)
        curve = {}
        for it in sorted(saved):
            try:
                v = e03._risk(score_pairs(kernel_mean_fn(ctx, saved[it]), ctx["val"]))
            except FloatingPointError:
                v = np.inf                                   # a checkpoint that cannot be scored is not selectable
            curve[it] = (v, clean_nll(saved[it], grid, held))
        best = min(curve, key=lambda i: curve[i][0])
        k = saved[best]
        pairs = score_pairs(kernel_mean_fn(ctx, k), ctx["test"])
    except FloatingPointError as exc:                        # E33 amendment 3: a failed fit is a row and counts against
        print(f"  [E58] {cell} seed={seed} n={n:>6} {arm:13s} FAILED: {exc}", flush=True)
        return [{**row0, "t": t, "failed": 1, "stop_reason": f"FloatingPointError: {exc}", "risk_pooled": np.nan}
                for t in T_SCHEDULE]
    pooled = e03._risk(pairs)
    kh = kernel_hellinger(k.log_transition_matrix(grid), ctx)
    info = {"checkpoint": best, "at_cap": int(best >= max(ladder)), "n_params": int(np.asarray(k.theta).size),
            "val_risk": curve[best][0], "heldout_nll": curve[best][1], "wall": time.perf_counter() - t0,
            "sec_per_step": sec_step, "kept_steps": int(getattr(k, "kept_steps", -1)),
            "final_lr": float(getattr(k, "lr", np.nan)), "kernel_hellinger": kh,
            "val_curve": ";".join(f"{i}:{a:.6g}/{b:.8g}" for i, (a, b) in sorted(curve.items())),
            "theta": ";".join(f"{v:.10g}" for v in np.asarray(k.theta, dtype=float)), "failed": 0}
    print(f"  [E58] {cell} seed={seed} n={n:>6} {arm:13s} risk={pooled:.4f} ck={best} H={kh:.4f}", flush=True)
    return [{**row0, "t": t, "sq_err": e, "sq_ref": r, "risk_level": float(np.sqrt(e / r)) if r > 0 else np.nan,
             "risk_pooled": pooled, **info} for t, (e, r) in zip(T_SCHEDULE, pairs)]


def _net_cell(args):
    cfg, cell, seed, n, arch = args
    ctx = context(cfg, cell, seed, n)
    conf = cfg["nets"][arch]
    den, info = e03._train_select(cfg, arch, conf["hp"], conf["lr"], conf["mode"], ctx["A"], ctx["val"], seed, 0)
    pairs = score_pairs(den.posterior_mean, ctx["test"])
    pooled = e03._risk(pairs)
    print(f"  [E58 net] {cell} seed={seed} n={n:>6} {arch:6s} risk={pooled:.4f} ck={info['checkpoint']}", flush=True)
    return [{**ctx["base"], "arm": arch, "hp": json.dumps(conf["hp"]), "t": t, "sq_err": e, "sq_ref": r,
             "risk_level": float(np.sqrt(e / r)) if r > 0 else np.nan, "risk_pooled": pooled, **info, "failed": 0}
            for t, (e, r) in zip(T_SCHEDULE, pairs)]


def _seeds(cfg, cell):
    return cfg["laplace_seeds"] if cell == "laplace" else cfg["field_seeds"]


def _key(cell):
    _, c, seed, n, arm = cell
    return f"{c}_s{seed}_n{n}_{arm}"


def part_kernels(cfg, out, workers):
    cells = [(cfg, c, s, n, a) for c in cfg["cells"] for s in _seeds(cfg, c) for n in cfg["sizes"]
             for a in ("lmmse",) + tuple(cfg["kernel_arms"])]
    write_csv(Path(out) / "kernels.csv", run_cells(cells, _kernel_cell, workers, out, "kernels.csv", key=_key))


def part_nets(cfg, out, workers):
    cells = [(cfg, c, s, n, a) for c in cfg["cells"] for s in _seeds(cfg, c) for n in cfg["sizes"]
             for a in cfg["net_archs"]]
    write_csv(Path(out) / "nets.csv", run_cells(cells, _net_cell, workers, out, "nets.csv", key=_key))


def main() -> None:
    parser = experiment_parser("e58_clean_kernel", "the neural kernel by clean-data maximum likelihood, then BP")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    w = args.workers
    parts = {"kernels": ("clean-data kernel fits and the linear floor", lambda c, o: part_kernels(c, o, w)),
             "nets": ("global MLP and window CNN, standard DSM", lambda c, o: part_nets(c, o, w))}
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
