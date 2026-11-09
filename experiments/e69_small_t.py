#!/usr/bin/env python
"""E69 — score error at small noise, t from 0.005 to 0.05, on the Laplace chain.

The kernel does not depend on t, so E64's frozen kernels (experiments/data/e64_kernels.csv) are scored unchanged at
small t through BP on a finer grid (1201 points). The global MLP and the window CNN (E03's winners) are retrained with
the level drawn log-uniformly in [0.005, 3] (E64's recipe) on the same training sequences, and selected on validation
sequences at the E69 levels. References: BP with the true kernel on the same grid, the true kernel on the paper's
401-point grid (grid error), and the best linear denoiser.

Parts
-----
gate   G1: the true-kernel score on 1201 vs 2401 points at t = 0.005 (dev seed)   -> gate.json
cells  (seed, N) -> rows of (arm, t, sq_err, sq_ref)                            -> small_t.csv

PREDECLARED in experiments/registry/E69.json (G1-G2, P1-P5, D1-D2) before this file was written.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from protocol import PROTOCOL  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

import e03_baselines as e03  # noqa: E402
import e64_generation as e64  # noqa: E402
from e63_compute import NETS  # noqa: E402

N_SITES = FROZEN.n_sites
T69 = (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05)
KERNEL_ARMS = ("em_bp", "spline", "energy", "clean_energy")
NET_ARMS = ("mlp", "window")

SETTINGS = dict(
    seeds=tuple(range(8)), sizes=(128, 2048), levels=T69, grid_size=1201, grid_check=2401, paper_grid=FROZEN.n_grid,
    half_width=FROZEN.half_width, n_val=PROTOCOL.n_val, n_test=PROTOCOL.n_test, n_gate=PROTOCOL.n_test,
    kernel_arms=KERNEL_ARMS, net_arms=NET_ARMS, kernels_csv="experiments/data/e64_kernels.csv", shift_W=4.0,
    t_min=0.005, t_max=3.0, net_steps=PROTOCOL.net_steps, net_checkpoints=PROTOCOL.net_checkpoints,
    batch_size=PROTOCOL.batch_size, grad_clip=PROTOCOL.grad_clip, torch_threads=1, gate_seed=10000,
    grid_size_e03=FROZEN.n_grid,
)
QUICK = dict(seeds=(0,), sizes=(128,), levels=(0.005, 0.05), grid_size=301, grid_check=401, n_val=16, n_test=16,
             n_gate=16, net_steps=20, net_checkpoints=(10, 20), net_arms=("window",), kernel_arms=("em_bp",))


def _chains(prior, rng, n):
    return np.stack([prior.sample(rng, N_SITES) for _ in range(n)])


def _noisy(A, keys, levels):
    out = {}
    for t in levels:
        al, de = alpha_delta(t)
        out[t] = al * A + np.sqrt(de) * rng_for(*keys, t).standard_normal(A.shape)
    return out


def _bp_means(grid, w, log_k, X, t):
    al, de = alpha_delta(t)
    return grid_bp_batch(grid, w, log_k, X, al, de)[0]


def _sq(m_hat, m_ref, X, t):
    al, de = alpha_delta(t)
    s_hat, s_ref = -(X - al * m_hat) / de, -(X - al * m_ref) / de
    return float(((s_hat - s_ref) ** 2).sum()), float((s_ref ** 2).sum())


def _lmmse_means(X, t, sigma):
    al, de = alpha_delta(t)
    return al * (sigma @ np.linalg.solve(al * al * sigma + de * np.eye(len(sigma)), X.T)).T


# ---------------------------------------------------------------------------------------- networks

def train_small(cfg, arch, A, val, seed, n):
    """E64's continuous-level recipe; the checkpoint is selected on the validation score error pooled over E69's levels."""
    import torch
    from markov_score import nn_torch as nt
    torch.set_num_threads(int(cfg["torch_threads"]))
    hp, lr, mode = NETS[arch]
    s = int(rng_for("e69-net-seed", arch, seed, n).integers(0, 2**31 - 1))
    torch.manual_seed(s)
    rng = np.random.default_rng(s)
    model = nt.build(arch, N_SITES, **hp)
    tcfg = nt.TrainConfig(steps=1, batch_size=cfg["batch_size"], lr=lr, grad_clip=cfg["grad_clip"], parameterization=mode)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ck = sorted(int(c) for c in cfg["net_checkpoints"])
    lo, hi = np.log(cfg["t_min"]), np.log(cfg["t_max"])
    curve, best, best_state = {}, None, None
    model.train()
    for step in range(1, ck[-1] + 1):
        idx = rng.integers(0, len(A), size=min(cfg["batch_size"], len(A)))
        a = torch.as_tensor(A[idx], dtype=torch.float32)
        t = torch.as_tensor(np.exp(rng.uniform(lo, hi, size=len(idx))), dtype=torch.float32)
        al, de = torch.exp(-t)[:, None], 1.0 - torch.exp(-2.0 * t)[:, None]
        z = torch.as_tensor(rng.standard_normal(a.shape), dtype=torch.float32)
        loss = nt.dsm_loss(model, a, al * a + torch.sqrt(de) * z, z, t, al, de, tcfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        if step in ck:
            den = nt.Denoiser(model, mode)
            pairs = [_sq(den.posterior_mean(X, t_), m, X, t_) for t_, (X, m) in val.items()]
            curve[step] = e03._risk(pairs)
            if best is None or curve[step] < curve[best]:
                best, best_state = step, {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.train()
    den = nt.Denoiser(nt.load_checkpoint(model, best_state).eval(), mode)
    return den, dict(checkpoint=best, val_risk=curve[best], n_params=nt.n_params(model),
                     val_curve=";".join(f"{k}:{v:.6g}" for k, v in curve.items()))


# ---------------------------------------------------------------------------------------- one cell

def _cell(args):
    cfg, seed, n = args
    import pandas as pd
    t0 = time.perf_counter()
    prior = LaplaceAR1(FROZEN.rho)
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    log_k = prior.log_transition_matrix(grid)
    g401, w401 = make_grid(cfg["half_width"], cfg["paper_grid"])
    lk401 = prior.log_transition_matrix(g401)
    i = np.arange(N_SITES)
    sigma = FROZEN.rho ** np.abs(i[:, None] - i[None, :]).astype(float)
    levels = tuple(cfg["levels"])
    cfg03 = dict(cfg, grid_size=cfg["grid_size_e03"])
    A, _, test03, _ = e03._cell_common(cfg03, seed, n)               # E03's training sequences and test bundle
    Xt = _noisy(_chains(prior, rng_for("e69-test", seed, n), cfg["n_test"]), ("e69-test-noise", seed, n), levels)
    Xv = _noisy(_chains(prior, rng_for("e69-val", seed, n), cfg["n_val"]), ("e69-val-noise", seed, n), levels)
    ref = {t: _bp_means(grid, w, log_k, Xt[t], t) for t in levels}
    val = {t: (Xv[t], _bp_means(grid, w, log_k, Xv[t], t)) for t in levels}
    base = dict(seed=seed, n_chains=n, grid_size=cfg["grid_size"])
    rows = []

    def add(arm, means_fn, **extra):
        for t in levels:
            e, r = _sq(means_fn(Xt[t], t), ref[t], Xt[t], t)
            rows.append({**base, "arm": arm, "t": t, "sq_err": e, "sq_ref": r, "risk_level": np.sqrt(e / r),
                         "failed": 0, **extra})

    add("true_401", lambda X, t: _bp_means(g401, w401, lk401, X, t))
    add("lmmse", lambda X, t: _lmmse_means(X, t, sigma))
    table = pd.read_csv(Path(__file__).resolve().parents[1] / cfg["kernels_csv"])
    for arm in cfg["kernel_arms"]:
        r = table[(table.arm == arm) & (table.seed == seed) & (table.n_chains == n)]
        if len(r) != 1:
            rows.append({**base, "arm": arm, "failed": 1, "stop_reason": "no frozen kernel"}); continue
        kernel = e64.load_kernel(cfg, r.iloc[0])
        # G2: rebuilt on the paper's grid, the kernel scores what its frozen row says, on E03's test bundle
        lk_p = kernel.log_transition_matrix(g401)
        rebuilt = e03._risk([e03._sq(_bp_means(g401, w401, lk_p, X, t), m, X, t, e03.REGIONS["all"])
                             for t, (X, m) in test03.items()])
        lk = kernel.log_transition_matrix(grid)
        add(arm, lambda X, t, lk=lk: _bp_means(grid, w, lk, X, t), frozen_risk=float(r.iloc[0].risk_pooled),
            rebuilt_risk=rebuilt)
    for arch in cfg["net_arms"]:
        den, info = train_small(cfg, arch, A, val, seed, n)
        add(f"{arch}_small", den.posterior_mean, **info)
    print(f"  [E69] seed={seed} n={n:>5} done in {time.perf_counter() - t0:.0f} s", flush=True)
    return rows


def part_cells(cfg, out, workers):
    cells = [(cfg, s, n) for s in cfg["seeds"] for n in cfg["sizes"]]
    write_csv(Path(out) / "small_t.csv", run_cells(cells, _cell, workers, out, "small_t.csv",
                                                    key=lambda c: f"s{c[1]}_n{c[2]}"))


def part_gate(cfg, out):
    """G1: the reference grid is converged at the smallest level."""
    prior = LaplaceAR1(FROZEN.rho)
    t = min(cfg["levels"])
    X = _noisy(_chains(prior, rng_for("e69-gate", cfg["gate_seed"]), cfg["n_gate"]),
               ("e69-gate-noise", cfg["gate_seed"]), (t,))[t]
    means = {}
    for size in (cfg["grid_size"], cfg["grid_check"]):
        grid, w = make_grid(cfg["half_width"], size)
        means[size] = _bp_means(grid, w, prior.log_transition_matrix(grid), X, t)
    e, r = _sq(means[cfg["grid_size"]], means[cfg["grid_check"]], X, t)
    res = {"t": t, "grid": cfg["grid_size"], "check": cfg["grid_check"], "rel_diff": float(np.sqrt(e / r)),
           "pass": bool(np.sqrt(e / r) <= 1e-4)}
    write_json(Path(out) / "gate.json", res)
    print(f"  [E69] G1 {res}", flush=True)


def main() -> None:
    parser = experiment_parser("e69_small_t", "score error at small t on the Laplace chain")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    w = args.workers
    parts = {"gate": ("G1, the reference grid at the smallest t", part_gate),
             "cells": ("frozen kernels, references and retrained networks", lambda c, o: part_cells(c, o, w))}
    if args.list_parts:
        print("\n".join(parts)); return
    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir)
    tag = "_".join(selected) if args.only else "all"
    resolved = {"n_sites": N_SITES, "quick": args.quick, "parts": list(selected), "overrides": args.set, **cfg}
    write_json(out / f"params_{tag}.json", {**resolved, **provenance(resolved)})
    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(cfg, out)


if __name__ == "__main__":
    main()
