#!/usr/bin/env python
"""E74 — generation on harder chains and on the Allen-Cahn field, every model through E64's sampler, samples kept.

Cells: E61's double-well Langevin, skewed and Pearson chains, and the field T0.3_h1_L32. Frozen kernels (no refit) go
through BP in the reverse sampler with their own stationary law for the first site; MLP, window CNN and U-Net are
retrained on log-uniform continuous levels (E64's recipe) on the same training sequences. Samples of seed 0 are saved
for images, and the oracle and the two neural kernels also generate long chains (L = 128).

Part
----
gen   (cell, seed, N) -> generation.csv (one row per arm) and samples/<cell>_n<N>.npz, long/<cell>.npz (seed 0)

PREDECLARED in experiments/registry/E74.json (P1-P4, D1-D2) before this file was written.
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
from markov_score import ancestral as an  # noqa: E402
from markov_score import gen_metrics as gm  # noqa: E402
from markov_score.bp_grid import grid_bp_batch  # noqa: E402
from markov_score.energy_kernel import NeuralEnergyKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

import e05_generation as e05  # noqa: E402
import e33_field_learning as e33  # noqa: E402
import e55_unet_energy as e55  # noqa: E402
import e58_clean_kernel as e58  # noqa: E402
import e61_chain_panel as e61  # noqa: E402
import e64_generation as e64  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIELD = "T0.3_h1_L32"
KERNEL_ARMS = ("energy", "energy_clean", "mixture")
FIELD_NAMES = {"energy": "em_energy", "energy_clean": "energy_h16", "mixture": "mixture_clean"}

SETTINGS = {**e64.SETTINGS, **e61.SETTINGS, "cells": ("langevin", "skew", "pearson", FIELD), "seeds": tuple(range(8)),
            "sizes": (128, 2048), "kernel_arms": KERNEL_ARMS, "net_arms": ("mlp", "window", "unet"), "n_gen": 2048,
            "gen_steps": 400, "t_min": 0.005, "t_max": 3.0, "n_image": 64, "long_L": 128, "image_seed": 0}
QUICK = {"cells": ("langevin", FIELD), "seeds": (0,), "sizes": (128,), "n_gen": 64, "gen_steps": 10, "grid_size": 101,
         "n_val": 16, "n_test": 16, "net_steps": 20, "net_checkpoints": (10, 20), "net_arms": ("window",),
         "kernel_arms": ("energy_clean",), "n_image": 8, "long_L": 40}


def _setup(cfg, cell, seed, n):
    """(training sequences, 12-level validation bundle, grid, weights, true log-kernel, true log-mu, sampler of exact draws)."""
    if cell == FIELD:
        ctx = e58.context(dict(e58.SETTINGS, **{k: cfg[k] for k in ("grid_size", "n_val", "n_test", "half_width")}),
                          FIELD, seed, n)
        chain, _ = e55.resolve_cell(e55.find_cell(FIELD))
        grid, w = ctx["grid"], ctx["w"]
        return (ctx["A"], ctx["val"], grid, w, chain.log_transition_matrix(grid), chain.log_stationary(grid),
                lambda rng, m, L: chain.sample_batch(rng, m, L))
    A, val, _, _, grid, w, prior, log_k = e61._cell_common(cfg, cell, seed, n)
    mu = e33._stationary_mu(grid, w)(prior)
    return A, val, grid, w, log_k, mu, lambda rng, m, L: np.stack([prior.sample(rng, L) for _ in range(m)])


def _kernel(cell, arm, seed, n, half_width):
    import pandas as pd
    if cell == FIELD:
        t = pd.read_csv(ROOT / "experiments/data/e72_kernels.csv")
        r = t[(t.arm == FIELD_NAMES[arm]) & (t.seed == seed) & (t.n_chains == n)]
    else:
        t = pd.read_csv(ROOT / "experiments/data/e74_kernels.csv")
        r = t[(t.chain == cell) & (t.arm == arm) & (t.seed == seed) & (t.n_chains == n)]
    if len(r) != 1:
        return None
    row = r.iloc[0]
    theta = np.array([float(v) for v in str(row["theta"]).split(";")])
    hp = json.loads(row["hp"])
    if arm == "mixture":
        return e05._mixture_from_theta(theta, int(hp.get("C", hp.get("components", 8))))
    return NeuralEnergyKernel.from_theta(theta, int(hp["hidden"]), half_width)


def _mean_fn(grid, w, log_k, log_mu):
    def mean(X, t):
        al, de = alpha_delta(t)
        return grid_bp_batch(grid, w, log_k, X, al, de, log_mu=log_mu)[0]
    return mean


def switch_rate(S: np.ndarray) -> float:
    """Sign changes per link: switches between the wells (double well), kinks between domains (field)."""
    return float((np.sign(S[:, 1:]) != np.sign(S[:, :-1])).mean())


def _cell(args):
    cfg, cell, seed, n, out = args
    t0 = time.perf_counter()
    A, val, grid, w, log_k, log_mu, draw = _setup(cfg, cell, seed, n)
    L, N = FROZEN.n_sites, cfg["n_gen"]
    ladder = np.geomspace(cfg["t_max"], cfg["t_min"], cfg["gen_steps"] + 1)
    R = draw(rng_for("e74-ref", cell, seed, n), N, L)
    noise = lambda: rng_for("e74-gen-noise", cell, seed, n)  # noqa: E731   the same sampler noise for every arm
    rows, keep, long = [], {}, {}
    rate_ref = switch_rate(R)

    def add(arm, G, **extra):
        # the registered metrics only (gm.compare's likelihood term is not used by E74 and fails on samples that leave
        # the grid's support)
        m = {"sw2": gm.sliced_wasserstein2(G, R, seed=seed), "w1_marginal": gm.wasserstein1_1d(G, R),
             "energy": gm.energy_distance(G, R, seed=seed)}
        rate = switch_rate(G)
        rows.append({"cell": cell, "seed": seed, "n_chains": n, "arm": arm, **m, "switch_gen": rate, "switch_ref": rate_ref,
                     "switch_err": abs(rate - rate_ref), "var_gen": float(G.var()), "failed": 0, **extra})
        keep[arm] = G[:cfg["n_image"]]
        print(f"  [E74] {cell} seed={seed} n={n:>5} {arm:13s} sw2={m['sw2']:.4f} switch={rate:.3f} (ref {rate_ref:.3f})",
              flush=True)

    add("exact", draw(rng_for("e74-ref2", cell, seed, n), N, L))
    oracle = _mean_fn(grid, w, log_k, log_mu)
    add("oracle", an.generate(oracle, N, L, ladder, noise()))
    image = seed == cfg["image_seed"]
    if image and n == max(cfg["sizes"]):
        long["exact"] = draw(rng_for("e74-long-ref", cell), cfg["n_image"], cfg["long_L"])
        long["oracle"] = an.generate(oracle, cfg["n_image"], cfg["long_L"], ladder, rng_for("e74-long-noise", cell))
    mu_fn = e33._stationary_mu(grid, w)
    for arm in cfg["kernel_arms"]:
        k = _kernel(cell, arm, seed, n, cfg["half_width"])
        if k is None:
            rows.append({"cell": cell, "seed": seed, "n_chains": n, "arm": arm, "failed": 1}); continue
        lk = k.log_transition_matrix(grid)
        fn = _mean_fn(grid, w, lk, mu_fn(k))
        add(arm, an.generate(fn, N, L, ladder, noise()))
        if image and n == max(cfg["sizes"]) and arm != "mixture":
            long[arm] = an.generate(fn, cfg["n_image"], cfg["long_L"], ladder, rng_for("e74-long-noise", cell))
    for arch in cfg["net_arms"]:
        den, info = e64.train_continuous(cfg, arch, A, val, seed, n)
        add(f"{arch}_cont", an.generate(den.posterior_mean, N, L, ladder, noise()),
            **{k_: info[k_] for k_ in ("checkpoint", "n_params") if k_ in info})
    keep["reference"] = R[:cfg["n_image"]]
    if image:
        d = ensure_dir(Path(out) / "samples")
        np.savez_compressed(d / f"{cell}_n{n}.npz", **keep)
        if long:
            np.savez_compressed(ensure_dir(Path(out) / "long") / f"{cell}.npz", **long)
    print(f"  [E74] {cell} seed={seed} n={n:>5} done in {time.perf_counter() - t0:.0f} s", flush=True)
    return rows


def part_gen(cfg, out, workers):
    cells = [(cfg, c, s, n, str(out)) for n in sorted(cfg["sizes"], reverse=True) for c in cfg["cells"] for s in cfg["seeds"]]
    write_csv(Path(out) / "generation.csv", run_cells(cells, _cell, workers, out, "generation.csv",
                                                       key=lambda c: f"{c[1]}_s{c[2]}_n{c[3]}"))


def main() -> None:
    parser = experiment_parser("e74_generation_hard", "generation on harder chains and the field, samples kept")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    parts = {"gen": ("generation, metrics and samples", lambda c, o: part_gen(c, o, args.workers))}
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
