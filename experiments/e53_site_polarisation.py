"""E53: per-site polarisation along diffusion time. Do the sites of the field commit one after another or at once?

One exact OU path per clean chain (read backwards it is a generation with the exact score); at each level the
exact posterior sign probability of every site by grid BP. Writes the per-site commitment times and, for the
figure, the full polarisation map of the first chain of each cell.
Predictions: experiments/registry/E53.json. Reading: tools/core/read_e53.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import experiment_parser, provenance  # noqa: E402
from e32_field_oracles import resolve_cell  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

CELLS = (dict(id="T0.2_h1_L256", T=0.2, h=1.0, L=256, role="e53"), dict(id="T0.3_h1_L256", T=0.3, h=1.0, L=256, role="e53"))
SETTINGS = dict(n_chains=32, levels=tuple(np.geomspace(0.02, 4.0, 80)), grid_size=FROZEN.n_grid,
                half_width=FROZEN.half_width, commit=0.5, min_abs_a=0.5, min_domain=8, core_margin=2)
QUICK = dict(n_chains=4, levels=tuple(np.geomspace(0.02, 4.0, 12)), grid_size=201)


def ou_path(A, levels, rng):
    """X[k] = exact OU state at time levels[k] started from A at time 0 (successive exact transitions)."""
    X, x, t_prev = [], A, 0.0
    for t in levels:
        dt = t - t_prev
        x = np.exp(-dt) * x + np.sqrt(1 - np.exp(-2 * dt)) * rng.standard_normal(A.shape)
        X.append(x); t_prev = t
    return X


def commitment_times(M, sign_a, levels, thr):
    """tau_i = largest level t_k such that m_i sign(a_i) >= thr at every level t_j <= t_k (0 if never)."""
    ok = (M * sign_a[None]) >= thr                     # (levels, B, L), levels ascending
    run = np.cumprod(ok, axis=0).astype(bool)           # committed at every level up to k
    n_ok = run.sum(axis=0)                              # how many of the smallest levels
    lv = np.concatenate([[0.0], np.asarray(levels)])
    return lv[n_ok]


def domains(a, min_len, margin):
    """Interior constant-sign runs of length >= min_len: list of (start, stop) and role per site."""
    s = np.sign(a)
    cuts = np.flatnonzero(s[1:] != s[:-1]) + 1
    bounds = np.concatenate([[0], cuts, [len(a)]])
    dom_id = np.full(len(a), -1)
    role = np.array(["other"] * len(a), dtype=object)
    k = 0
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if lo == 0 or hi == len(a) or hi - lo < min_len:
            continue
        dom_id[lo:hi] = k
        role[lo + margin:hi - margin] = "core"
        role[lo] = role[hi - 1] = "edge"
        k += 1
    return dom_id, role


def main():
    ap = experiment_parser("e53_site_polarisation", "per-site commitment along exact OU paths on the phi^4 field")
    args = ap.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    out = ensure_dir(Path(args.output_dir))
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    rows, maps = [], {}
    for cell in CELLS:
        chain, info = resolve_cell(cell)
        log_k, log_mu = chain.log_transition_matrix(grid), chain.log_stationary(grid)
        A = chain.sample_batch(rng_for("e53-chains", cell["id"]), cfg["n_chains"], cell["L"])
        X = ou_path(A, cfg["levels"], rng_for("e53-path", cell["id"]))
        M = []
        for t, x in zip(cfg["levels"], X):
            al, de = np.exp(-t), 1 - np.exp(-2 * t)
            _, _, p = grid_bp_batch(grid, w, log_k, x, al, de, log_mu=log_mu, return_sign_prob=True)
            M.append(2 * p - 1)
        M = np.array(M)                                  # (levels, B, L)
        tau = commitment_times(M, np.sign(A), cfg["levels"], cfg["commit"])
        maps[cell["id"]] = M[:, 0, :]
        for b in range(cfg["n_chains"]):
            dom, role = domains(A[b], cfg["min_domain"], cfg["core_margin"])
            lengths = {d: int((dom == d).sum()) for d in set(dom.tolist()) if d >= 0}
            for i in range(cell["L"]):
                rows.append(dict(cell=cell["id"], T=info["T"], h=info["h"], xi_sites=info["corr_length_sites"], chain=b, site=i,
                                 a=float(A[b, i]), tau=float(tau[b, i]), domain=(f"{b}_{dom[i]}" if dom[i] >= 0 else ""),
                                 domain_len=lengths.get(int(dom[i]), 0), role=role[i]))
        print(f"  [E53] {cell['id']}: xi={info['corr_length_sites']:.2f}, {len(set(r['domain'] for r in rows if r['cell'] == cell['id'])) - 1} domains", flush=True)
    write_csv(out / "sites.csv", rows)
    np.savez_compressed(out / "maps.npz", levels=np.asarray(cfg["levels"]), **{k: v for k, v in maps.items()})
    write_json(out / "params_sites.json", {**provenance(), "settings": {**cfg, "cells": CELLS}, "quick": bool(args.quick)})


if __name__ == "__main__":
    main()
