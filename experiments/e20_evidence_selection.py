#!/usr/bin/env python
"""E20 — does EM-BP's checkpoint selection depend on oracle access?

Every EM-BP number in the paper selects its iteration count on a validation bundle scored against the
TRUE-kernel score (so do the standard-recipe networks and the fixed-corruption arm; only the implicit-
score-matching arm selects without an oracle). This control replicates E02a cells exactly (same bundle
tags, same initialisation, same ladder, C = 8) and records, at every checkpoint, (i) the oracle
validation risk, (ii) the held-out one-view marginal log-evidence per edge (no clean chain, no true
score), and (iii) the test risk; then the test risk under each selection rule, and at the last rung.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from protocol import PROTOCOL  # noqa: E402
from markov_score.em import e_step_multi, fit_em  # noqa: E402
from markov_score.bp_grid import make_grid  # noqa: E402
from markov_score.kernels import MixtureInnovationKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402
import e02_capacity as e02  # noqa: E402

N = FROZEN.n_sites; T = tuple(FROZEN.t_grid)
SETTINGS = dict(sizes=(32, 512), seeds=tuple(range(16)), components=8, n_val=PROTOCOL.n_val, n_test=PROTOCOL.n_test,
                grid_size=FROZEN.n_grid, half_width=FROZEN.half_width, em_checkpoints=PROTOCOL.em_checkpoints,
                em_init_rho=PROTOCOL.em_init_rho, em_init_var=PROTOCOL.em_init_var)
QUICK = dict(sizes=(32,), seeds=(0,), n_val=32, n_test=32, grid_size=201, em_checkpoints=(5, 10))


def _cell(args):
    cfg, M, seed = args
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    prior = LaplaceAR1(FROZEN.rho); logK = prior.log_transition_matrix(grid)
    sampler = lambda rng: prior.sample(rng, N)
    rng_train = rng_for("exp31-train", seed, M)
    A = np.stack([prior.sample(rng_train, N) for _ in range(M)])
    _, val = e02._bundle(sampler, logK, grid, w, "exp31-val", seed, cfg["n_val"])
    _, test = e02._bundle(sampler, logK, grid, w, "exp31-test", seed, cfg["n_test"])
    rng_v = rng_for("exp31-val", seed)
    Av = np.stack([prior.sample(rng_v, N) for _ in range(cfg["n_val"])])
    idx = np.array_split(rng_for("e20-split", seed).permutation(len(Av)), len(T)); noise_v = rng_for("e20-noise", seed)
    held = []
    for t, ix in zip(T, idx):
        al, de = alpha_delta(t); sub = Av[ix]; held.append((al * sub + np.sqrt(de) * noise_v.standard_normal(sub.shape), al, de))
    idx = np.array_split(rng_for("exp31-em-split", seed).permutation(len(A)), len(T)); noise = rng_for("exp31-em-noise", seed)
    groups = []
    for t, ix in zip(T, idx):
        al, de = alpha_delta(t); sub = A[ix]; groups.append((al * sub + np.sqrt(de) * noise.standard_normal(sub.shape), al, de))
    ck = tuple(int(c) for c in cfg["em_checkpoints"])
    k0 = MixtureInnovationKernel.init(cfg["components"], rho=cfg["em_init_rho"], var=cfg["em_init_var"], rng=rng_for("exp31-em-init", seed))
    _, trace, saved = fit_em(k0, grid, w, groups, n_iters=max(ck), tol=FROZEN.em_loglik_tol, checkpoints=set(ck))
    curve = {}
    for it in sorted(saved):
        kb = saved[it]
        vr = e02._risk(e02._eval_kernel(kb, val, grid, w, "bulk"))
        ev = e_step_multi(grid, w, kb.log_transition_matrix(grid), held).log_evidence / (cfg["n_val"] * (N - 1))
        curve[it] = [vr, ev, np.nan]
    b_or = min(curve, key=lambda i: curve[i][0]); b_ev = max(curve, key=lambda i: curve[i][1]); last = max(curve)
    # the test risk is needed only at the three checkpoints the rules pick (the full ladder's test
    # curve costs four times the fit and adds nothing to the comparison)
    for it in {b_or, b_ev, last}:
        curve[it][2] = e02._risk(e02._eval_kernel(saved[it], test, grid, w, "all"))
    row = dict(n_chains=M, seed=seed, components=cfg["components"], ckpt_oracle=b_or, ckpt_evidence=b_ev, ckpt_last=last,
               test_oracle=curve[b_or][2], test_evidence=curve[b_ev][2], test_last=curve[last][2], stop_reason=trace.stop_reason,
               curve=";".join(f"{i}:{v[0]:.4f}/{v[1]:.5f}/{v[2]:.4f}" for i, v in sorted(curve.items())))
    print(f"  [E20] M={M} seed={seed}: oracle ckpt {b_or} -> {curve[b_or][2]:.4f}; evidence ckpt {b_ev} -> {curve[b_ev][2]:.4f}; last {last} -> {curve[last][2]:.4f}", flush=True)
    return [row]


def part_selection(cfg, out, workers):
    cells = [(cfg, M, s) for M in cfg["sizes"] for s in cfg["seeds"]]
    rows = run_cells(cells, _cell, workers, out, "selection.csv")
    write_csv(out / "selection.csv", rows)


PARTS = {"selection": part_selection}


def main():
    ap = experiment_parser("e20_evidence_selection", "EM-BP checkpoint selection with and without the true score")
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
        write_json(out / f"params_{name}.json", {**provenance(), "settings": cfg, "quick": bool(args.quick), "workers": args.workers, "frozen": FROZEN.as_dict()})


if __name__ == "__main__":
    main()
