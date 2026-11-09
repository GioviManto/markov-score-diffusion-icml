#!/usr/bin/env python
"""E71 — EM with several noisy copies of each training sequence, Laplace chain.

EM normally sees one noisy copy of each training sequence, at one of the twelve levels (E03's interface). Here each
sequence is noised K times at K distinct levels (K = 1, 4, 12; K = 1 is E03's interface exactly), and the mixture
(E30's fit) and the neural energy kernel (E56's fit, width 16, optimiser restart on) are learned from all copies.

Part
----
fit   (seed, N, K, arm) -> rows per level on E03's test bundle -> views.csv

PREDECLARED in experiments/registry/E71.json (P1-P4, D1-D2) before this file was written.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score.bp_grid import make_grid  # noqa: E402
from markov_score.kernels import MixtureInnovationKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

import e03_baselines as e03  # noqa: E402
import e30_joint_ladder as e30  # noqa: E402
import e56_kernel_families as e56  # noqa: E402

T_SCHEDULE = tuple(FROZEN.t_grid)

SETTINGS = {**e30.SETTINGS, **e56.SETTINGS,
            "seeds": tuple(range(8)), "sizes": (32, 128, 512), "views": (1, 4, 12), "arms": ("em_bp", "energy"),
            "energy_hidden": 16, "energy_fresh": True}
QUICK = {"seeds": (0,), "sizes": (32,), "views": (1, 4), "n_val": 32, "n_test": 32, "grid_size": 101,
         "em_checkpoints": (2, 4), "ladder": (2, 4), "energy_hidden": 8}


def view_groups(A, seed, n, k):
    """K noisy copies of every sequence at K distinct levels, as fit_em's (X, alpha, delta) groups."""
    if k == 1:
        return [(X, *alpha_delta(t)) for X, t in e03._one_view_groups(A, seed)]
    n_lv = len(T_SCHEDULE)
    pick = rng_for("e71-levels", seed, n, k)
    levels = np.stack([pick.permutation(n_lv)[:k] for _ in range(len(A))])        # (n, k) level indices
    noise = rng_for("e71-noise", seed, n, k)
    groups = []
    for j, t in enumerate(T_SCHEDULE):
        rows = np.where((levels == j).any(axis=1))[0]
        if len(rows) == 0:
            continue
        al, de = alpha_delta(t)
        sub = A[rows]
        groups.append((al * sub + np.sqrt(de) * noise.standard_normal(sub.shape), al, de))
    return groups


def _cell(args):
    cfg, seed, n, k, arm = args
    import torch
    torch.set_num_threads(int(cfg["torch_threads"]))
    t0 = time.perf_counter()
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    A, val, test, base = e03._cell_common(cfg, seed, n)
    groups = view_groups(A, seed, n, k)
    n_noisy = int(sum(len(g[0]) for g in groups))
    if arm == "em_bp":
        k0 = MixtureInnovationKernel.init(cfg["em_components"], rho=0.3, var=0.8, rng=rng_for("e30-em-init", seed, n))
        _, pairs, info = e30._fit_and_select(cfg, k0, groups, grid, weights, val, test, cfg["em_checkpoints"])
    else:
        k0 = e56._init_kernel(cfg, "energy", seed, n, cfg["energy_hidden"])
        kern, info = e56._fit_and_select(cfg, k0, groups, grid, weights, val)
        pairs = e56._pairs(kern, test, grid, weights)
    pooled = e03._risk(pairs)
    wall = time.perf_counter() - t0
    print(f"  [E71] seed={seed} n={n:>4} K={k:>2} {arm:6s} pooled={pooled:.4f} ck={info['checkpoint']} {wall:.0f}s",
          flush=True)
    keep = {key: info[key] for key in ("checkpoint", "at_cap", "val_risk", "stop_reason") if key in info}
    return [{**base, "arm": arm, "views": k, "n_noisy": n_noisy, "t": t, "sq_err": e, "sq_ref": r,
             "risk_level": float(np.sqrt(e / r)) if r > 0 else np.nan, "risk_pooled": pooled, "wall": wall,
             "failed": 0, **keep} for t, (e, r) in zip(T_SCHEDULE, pairs)]


def part_fit(cfg, out, workers):
    cells = [(cfg, s, n, k, a) for n in sorted(cfg["sizes"], reverse=True) for k in sorted(cfg["views"], reverse=True)
             for a in cfg["arms"] for s in cfg["seeds"]]
    write_csv(Path(out) / "views.csv", run_cells(cells, _cell, workers, out, "views.csv",
                                                  key=lambda c: f"s{c[1]}_n{c[2]}_K{c[3]}_{c[4]}"))


def main() -> None:
    parser = experiment_parser("e71_views", "EM with several noisy copies of each training sequence")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    parts = {"fit": ("EM fits for every K", lambda c, o: part_fit(c, o, args.workers))}
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
