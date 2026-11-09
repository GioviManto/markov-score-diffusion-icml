#!/usr/bin/env python
"""E72 — E69's small-t study on the Allen-Cahn field (cell T0.3_h1_L32).

Frozen field kernels (experiments/data/e72_kernels.csv: E55's neural kernel by EM, E58's neural kernel and shift mixture
by clean maximum likelihood) are rebuilt as functions and scored through BP on a 1201-point grid at t from 0.005 to
0.05, each with its own stationary law for the first site (E33's convention). The global MLP and window CNN (E58's
recipe) are retrained with the level drawn log-uniformly in [0.005, 3] on E33's training sequences and selected on
validation sequences at the same levels (E69's train_small). Reference: BP with the exact kernel on the same grid.

Parts
-----
gate   G1: the exact-kernel score on 1201 vs 2401 points at t = 0.005 (dev seed)  -> gate.json
cells  (seed, N) -> rows of (arm, t, sq_err, sq_ref)                             -> small_t_field.csv

PREDECLARED in experiments/registry/E72.json (G1-G2, P1-P4, D1) before this file was written.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.energy_kernel import NeuralEnergyKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

import e03_baselines as e03  # noqa: E402
import e05_generation as e05  # noqa: E402
import e33_field_learning as e33  # noqa: E402
import e55_unet_energy as e55  # noqa: E402
import e58_clean_kernel as e58  # noqa: E402
import e69_small_t as e69  # noqa: E402

CELL = "T0.3_h1_L32"
KERNEL_ARMS = ("em_energy", "energy_h16", "mixture_clean")

SETTINGS = {**e58.SETTINGS, "cells": (CELL,), "seeds": tuple(range(8)), "sizes": (128, 2048), "levels": e69.T69,
            "fine_grid": 1201, "grid_check": 2401, "kernel_arms": KERNEL_ARMS, "net_arms": ("mlp", "window"),
            "kernels_csv": "experiments/data/e72_kernels.csv", "t_min": 0.005, "t_max": 3.0, "gate_seed": 10000,
            "n_gate": 2048}
QUICK = {"seeds": (0,), "sizes": (128,), "levels": (0.005, 0.05), "fine_grid": 301, "grid_check": 401, "n_val": 16,
         "n_test": 16, "n_gate": 16, "net_steps": 20, "net_checkpoints": (10, 20), "net_arms": ("window",),
         "kernel_arms": ("energy_h16",)}


def _chain():
    chain, _ = e55.resolve_cell(e55.find_cell(CELL))
    return chain, e55.find_cell(CELL)["L"]


def _rebuild(row, half_width):
    theta = np.array([float(v) for v in str(row["theta"]).split(";")])
    hp = json.loads(row["hp"])
    if row["arm"] == "mixture_clean":
        return e05._mixture_from_theta(theta, int(hp["components"]))
    return NeuralEnergyKernel.from_theta(theta, int(hp["hidden"]), half_width)


def _bp(grid, w, log_k, log_mu, X, t):
    al, de = alpha_delta(t)
    return grid_bp_batch(grid, w, log_k, X, al, de, log_mu=log_mu)[0]


def _cell(args):
    cfg, seed, n = args
    import pandas as pd
    t0 = time.perf_counter()
    chain, L = _chain()
    ctx = e58.context(cfg, CELL, seed, n)                          # E33's training sequences and test bundle (401 points)
    grid, w = make_grid(cfg["half_width"], cfg["fine_grid"])
    log_k, log_mu = chain.log_transition_matrix(grid), chain.log_stationary(grid)
    levels = tuple(cfg["levels"])
    Xt = e69._noisy(chain.sample_batch(rng_for("e72-test", seed, n), cfg["n_test"], L), ("e72-test-noise", seed, n), levels)
    Xv = e69._noisy(chain.sample_batch(rng_for("e72-val", seed, n), cfg["n_val"], L), ("e72-val-noise", seed, n), levels)
    ref = {t: _bp(grid, w, log_k, log_mu, Xt[t], t) for t in levels}
    val = {t: (Xv[t], _bp(grid, w, log_k, log_mu, Xv[t], t)) for t in levels}
    base = {"seed": seed, "n_chains": n, "cell": CELL, "grid_size": cfg["fine_grid"]}
    rows = []

    def add(arm, means_fn, **extra):
        for t in levels:
            e, r = e69._sq(means_fn(Xt[t], t), ref[t], Xt[t], t)
            rows.append({**base, "arm": arm, "t": t, "sq_err": e, "sq_ref": r, "risk_level": np.sqrt(e / r),
                         "failed": 0, **extra})

    cov = ctx["cov"]
    add("lmmse", lambda X, t: e69._lmmse_means(X, t, cov))
    table = pd.read_csv(Path(__file__).resolve().parents[1] / cfg["kernels_csv"])
    mu_fine = e33._stationary_mu(grid, w)
    for arm in cfg["kernel_arms"]:
        r = table[(table.arm == arm) & (table.seed == seed) & (table.n_chains == n)]
        if len(r) != 1:
            rows.append({**base, "arm": arm, "failed": 1, "stop_reason": "no frozen kernel"}); continue
        kernel = _rebuild(r.iloc[0], cfg["half_width"])
        rebuilt = e03._risk(e58.score_pairs(e58.kernel_mean_fn(ctx, kernel), ctx["test"]))   # G2, on the 401 grid
        lk, mu = kernel.log_transition_matrix(grid), mu_fine(kernel)
        add(arm, lambda X, t, lk=lk, mu=mu: _bp(grid, w, lk, mu, X, t), frozen_risk=float(r.iloc[0].frozen_risk),
            rebuilt_risk=rebuilt, source=r.iloc[0].source)
    for arch in cfg["net_arms"]:
        den, info = e69.train_small(cfg, arch, ctx["A"], val, seed, n)
        add(f"{arch}_small", den.posterior_mean, **info)
    print(f"  [E72] seed={seed} n={n:>5} done in {time.perf_counter() - t0:.0f} s", flush=True)
    return rows


def part_cells(cfg, out, workers):
    cells = [(cfg, s, n) for s in cfg["seeds"] for n in cfg["sizes"]]
    write_csv(Path(out) / "small_t_field.csv", run_cells(cells, _cell, workers, out, "small_t_field.csv",
                                                          key=lambda c: f"s{c[1]}_n{c[2]}"))


def part_gate(cfg, out):
    chain, L = _chain()
    t = min(cfg["levels"])
    X = e69._noisy(chain.sample_batch(rng_for("e72-gate", cfg["gate_seed"]), cfg["n_gate"], L),
                   ("e72-gate-noise", cfg["gate_seed"]), (t,))[t]
    means = {}
    for size in (cfg["fine_grid"], cfg["grid_check"]):
        grid, w = make_grid(cfg["half_width"], size)
        means[size] = _bp(grid, w, chain.log_transition_matrix(grid), chain.log_stationary(grid), X, t)
    e, r = e69._sq(means[cfg["fine_grid"]], means[cfg["grid_check"]], X, t)
    res = {"t": t, "grid": cfg["fine_grid"], "check": cfg["grid_check"], "rel_diff": float(np.sqrt(e / r)),
           "pass": bool(np.sqrt(e / r) <= 1e-4)}
    write_json(Path(out) / "gate.json", res)
    print(f"  [E72] G1 {res}", flush=True)


def main() -> None:
    parser = experiment_parser("e72_small_t_field", "score error at small t on the Allen-Cahn field")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    parts = {"gate": ("G1, the reference grid at the smallest t", part_gate),
             "cells": ("frozen field kernels, references and retrained networks",
                       lambda c, o: part_cells(c, o, args.workers))}
    if args.list_parts:
        print("\n".join(parts)); return
    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir)
    tag = "_".join(selected) if args.only else "all"
    resolved = {"quick": args.quick, "parts": list(selected), "overrides": args.set, **cfg}
    write_json(out / f"params_{tag}.json", {**resolved, **provenance(resolved)})
    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(cfg, out)


if __name__ == "__main__":
    main()
