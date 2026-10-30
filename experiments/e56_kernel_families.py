#!/usr/bin/env python
"""E56 — kernel families on the Laplace chain: does EM-BP's advantage belong to the mixture or to the structure?

Jerome's step 1 after the call of 29 Oct 2026: thesis Fig. 6.3 for other kernel families than the C = 8 Gaussian mixture,
and the neural kernel on the chain the paper calls its largest section. Six arms, all learned by EM through exact grid BP
from one noisy view per chain and selected on the validation bundle's BP risk (E30's rule):

    spline, rbf, fourier, legendre   shift kernels phi(a' - alpha a), log phi in a 23-function basis plus a learned
                                     quadratic envelope, 25 parameters each (markov_score.shift_kernel)
    loglinear2d                      E43's log-linear 2-D spline kernel, 100 parameters, exact concave M-step
    energy                           E55's neural energy kernel exp(-E(a, a'))/Z, uniform start, width from the screen

The data are E03's bundles, IMPORTED as E30 imports them (e03._cell_common, e03._one_view_groups, e03._sq, e03._risk), so
every row joins E03's six networks, E30's em_bp and mdn and E02a's em_bp on (seed, n_chains, test_bundle).

Parts
-----
timing     dev seed, N = timing_n: seconds per EM iteration for every arm (the budget amendment's input)
screen     dev seeds x screen_sizes x energy_widths: pooled validation risk -> winners.json {"energy_hidden": w}
fit        confirmatory seeds x sizes x arms -> kernels.csv, one row per (seed, n, arm, t)
densities  from kernels.csv: learned log phi on [-eps_half, eps_half] (shift arms) and log K(.|a) at slice_a (every arm)
verify     THE GATE, post-fetch: sq_ref per (seed, n, t) equals E30's within relative 1e-12, else exit 1 (2: cannot verify)

PREDECLARED in experiments/registry/E56.json (P1-P5, D1-D4) before this file was written.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from protocol import PROTOCOL  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.em import fit_em  # noqa: E402
from markov_score.energy_kernel import NeuralEnergyKernel  # noqa: E402
from markov_score.loglinear_kernel import LogLinearKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.shift_kernel import BASES, ShiftKernel  # noqa: E402
from markov_score.stationary import invariant_log_density_eig  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

# E03 is imported, not copied: the same bundle objects the nets were scored on.
import e03_baselines as e03  # noqa: E402

N_SITES = FROZEN.n_sites
T_SCHEDULE = tuple(FROZEN.t_grid)
LADDER = (10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200, 1600, 2400)   # E55's EM_LADDER (E56.json)
SHIFT_ARMS = BASES
ARMS = SHIFT_ARMS + ("loglinear2d", "energy")

SETTINGS = dict(
    seeds=PROTOCOL.seeds, dev_seeds=(10_000, 10_001), sizes=(32, 128, 512, 2048),
    n_val=PROTOCOL.n_val, n_test=PROTOCOL.n_test,
    grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    arms=ARMS, em_init_rho=PROTOCOL.em_init_rho, em_init_var=PROTOCOL.em_init_var,
    shift_W=4.0, shift_lr=1e-2, shift_inner=10,
    loglinear_basis=10,
    energy_lr=1e-2, energy_inner=10, energy_widths=(16, 32, 64),
    energy_hidden=0,          # 0: read <out>/winners.json, written by the screen part
    screen_sizes=(32, 2048), ladder=LADDER,
    timing_n=2048, timing_iters=3,
    density_sizes=(128, 512, 2048), eps_half=3.0, eps_points=161, slice_a=(-1.0, 0.0, 1.0),
    device="cpu", torch_threads=1,
)
QUICK = dict(seeds=(0,), dev_seeds=(10_000,), sizes=(32,), n_val=64, n_test=64, grid_size=101, ladder=(2, 4),
             screen_sizes=(32,), energy_widths=(8,), timing_n=32, timing_iters=2, density_sizes=(32,))


# ---------------------------------------------------------------------------------------- arms

def _init_kernel(cfg, arm, seed, n, hidden=0):
    rho, var = cfg["em_init_rho"], cfg["em_init_var"]
    if arm in SHIFT_ARMS:
        return ShiftKernel.init(arm, rho=rho, var=var, half_width_basis=cfg["shift_W"], lr=cfg["shift_lr"],
                                n_inner=cfg["shift_inner"])
    if arm == "loglinear2d":
        return LogLinearKernel.init(cfg["loglinear_basis"], rho=rho, var=var, half_width=cfg["half_width"],
                                    grid_size=cfg["grid_size"])
    if arm == "energy":
        if not hidden:
            raise ValueError("the energy arm needs a width: run the screen part or --set energy_hidden=<w>")
        s = int(rng_for("e56-energy-init", seed, n).integers(0, 2**31 - 1))
        return NeuralEnergyKernel.init(hidden=int(hidden), seed=s, half_width=cfg["half_width"], lr=cfg["energy_lr"],
                                       n_inner=cfg["energy_inner"], device=cfg["device"])
    raise ValueError(f"unknown arm {arm!r}")


def _rebuild(arm, theta, hp, cfg):
    """A fitted kernel from its kernels.csv row (theta written to 10 significant digits)."""
    theta = np.asarray(theta, dtype=float)
    if arm in SHIFT_ARMS:
        return ShiftKernel.from_theta(arm, theta, half_width_basis=cfg["shift_W"])
    if arm == "loglinear2d":
        return LogLinearKernel(coef=theta, half_width=cfg["half_width"])
    return NeuralEnergyKernel.from_theta(theta, int(json.loads(hp)["hidden"]), cfg["half_width"])


def _hidden(cfg, out):
    if cfg["energy_hidden"]:
        return int(cfg["energy_hidden"])
    p = Path(out) / "winners.json"
    return int(json.loads(p.read_text())["energy_hidden"]) if p.is_file() else 0


# ---------------------------------------------------------------------------------------- scoring

def _pairs(kernel, bundle, grid, weights):
    log_k = kernel.log_transition_matrix(grid)
    out = []
    for t in T_SCHEDULE:
        X, m_ref = bundle[t]
        al, de = alpha_delta(t)
        m, _ = grid_bp_batch(grid, weights, log_k, X, al, de)
        # e03._sq, not a local copy: same score-space definition, same region slice.
        out.append(e03._sq(m, m_ref, X, t, e03.REGIONS["all"]))
    return out


def _normalised(log_k, weights):
    return log_k - np.log(np.exp(log_k).T @ weights)[None, :]


def kernel_hellinger(log_k_hat, log_k_true, grid, weights) -> float:
    """sum_j pi(a_j) w_j H(K_hat(.|a_j), K(.|a_j)): the Hellinger distance between the learned and the true conditional
    laws, averaged over the true chain's invariant law on the grid."""
    lh, lt = _normalised(log_k_hat, weights), _normalised(log_k_true, weights)
    bc = (np.exp(0.5 * (lh + lt)) * weights[:, None]).sum(axis=0)
    h = np.sqrt(np.clip(1.0 - bc, 0.0, None))
    pi = np.exp(invariant_log_density_eig(LaplaceAR1(FROZEN.rho), grid, weights).log_density)
    return float((pi * weights * h).sum() / (pi * weights).sum())


def _fit_and_select(cfg, kernel0, groups, grid, weights, val):
    t0 = time.perf_counter()
    _, trace, saved = fit_em(kernel0, grid, weights, groups, n_iters=max(cfg["ladder"]), tol=FROZEN.em_loglik_tol,
                             checkpoints=set(cfg["ladder"]))
    wall = time.perf_counter() - t0
    curve = {it: e03._risk(_pairs(saved[it], val, grid, weights)) for it in sorted(saved)}
    best = min(curve, key=curve.get)
    ev = trace.log_evidence
    k = saved[best]
    return k, {
        "checkpoint": best, "at_cap": int(best == max(saved)), "n_params": int(np.asarray(k.theta).size),
        "val_risk": curve[best], "val_curve": ";".join(f"{it}:{r:.6g}" for it, r in curve.items()),
        "evidence_curve": ";".join(f"{it}:{ev[it] / max(trace.n_edges, 1):.10g}" for it in sorted(saved) if it < len(ev)),
        "n_iters": len(ev), "converged": bool(trace.converged), "stop_reason": trace.stop_reason,
        "monotone_violation": float(trace.monotone_violation), "wall": wall,
        "sec_per_iter": float(np.mean(trace.seconds)) if trace.seconds else float("nan"),
        "kept_steps": int(getattr(k, "kept_steps", -1)), "final_lr": float(getattr(k, "lr", float("nan"))),
    }


def _hp(arm, cfg, hidden):
    if arm in SHIFT_ARMS:
        return json.dumps({"W": cfg["shift_W"], "lr": cfg["shift_lr"], "n_inner": cfg["shift_inner"]})
    if arm == "loglinear2d":
        return json.dumps({"n_basis": cfg["loglinear_basis"]})
    return json.dumps({"hidden": int(hidden), "lr": cfg["energy_lr"], "n_inner": cfg["energy_inner"]})


def _cell(args):
    cfg, seed, n, arm, hidden, role = args
    import torch
    torch.set_num_threads(int(cfg["torch_threads"]))
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    A, val, test, base = e03._cell_common(cfg, seed, n)
    # E03's own corruption interface, so every arm learns from the SAME corrupted sample the networks' EM-BP partners
    # did (tags exp31-em-split / exp31-em-noise); alpha_delta(t) is the change of variable to fit_em's (X, alpha, delta).
    groups = [(X, *alpha_delta(t)) for X, t in e03._one_view_groups(A, seed)]
    k0 = _init_kernel(cfg, arm, seed, n, hidden)
    row0 = {**base, "role": role, "arm": arm, "hp": _hp(arm, cfg, hidden)}
    try:
        k, info = _fit_and_select(cfg, k0, groups, grid, weights, val)
    except FloatingPointError as exc:       # E33 amendment 3: a failed fit is a row, and it counts against the prediction
        print(f"  [E56] seed={seed} n={n:>5} {arm:11s} FAILED: {exc}", flush=True)
        return [{**row0, "t": t, "failed": 1, "stop_reason": f"FloatingPointError: {exc}", "risk_pooled": np.nan}
                for t in T_SCHEDULE]
    if role == "screen":
        print(f"  [E56 screen] seed={seed} n={n:>5} {arm} H={hidden} val={info['val_risk']:.4f}", flush=True)
        return [{**row0, **info, "t": np.nan, "failed": 0}]
    pairs = _pairs(k, test, grid, weights)
    pooled = e03._risk(pairs)
    kh = kernel_hellinger(k.log_transition_matrix(grid), LaplaceAR1(FROZEN.rho).log_transition_matrix(grid), grid, weights)
    theta = ";".join(f"{v:.10g}" for v in np.asarray(k.theta, dtype=float))
    alpha = float(k.alpha) if arm in SHIFT_ARMS else np.nan
    print(f"  [E56] seed={seed} n={n:>5} {arm:11s} pooled={pooled:.4f} ck={info['checkpoint']} "
          f"at_cap={info['at_cap']} H={kh:.4f} {info['stop_reason']}", flush=True)
    return [{**row0, "t": t, "sq_err": e, "sq_ref": r, "risk_level": float(np.sqrt(e / r)) if r > 0 else np.nan,
             "risk_pooled": pooled, **info, "kernel_hellinger": kh, "alpha": alpha, "theta": theta, "failed": 0}
            for t, (e, r) in zip(T_SCHEDULE, pairs)]


def _key(cell):
    _, seed, n, arm, hidden, role = cell
    return f"{role}_s{seed}_n{n}_{arm}" + (f"_H{hidden}" if arm == "energy" else "")


# ---------------------------------------------------------------------------------------- parts

def part_timing(cfg, out, workers):
    """Seconds per EM iteration at N = timing_n on the first dev seed, every arm (energy at every screened width)."""
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    seed, n = cfg["dev_seeds"][0], cfg["timing_n"]
    A, _, _, _ = e03._cell_common(cfg, seed, n)
    groups = [(X, *alpha_delta(t)) for X, t in e03._one_view_groups(A, seed)]
    rows = []
    for arm in cfg["arms"]:
        for hidden in (cfg["energy_widths"] if arm == "energy" else (0,)):
            k0 = _init_kernel(cfg, arm, seed, n, hidden)
            _, trace = fit_em(k0, grid, weights, groups, n_iters=cfg["timing_iters"], tol=0.0)
            sec = float(np.median(trace.seconds))
            budget = max(cfg["ladder"]) * sec
            rows.append({"arm": arm, "hidden": hidden, "n_chains": n, "grid_size": cfg["grid_size"],
                         "sec_per_iter": sec, "sec_full_ladder": budget})
            print(f"  [timing] {arm:11s} H={hidden:<3} {sec:.3f} s/iteration -> {budget / 3600:.2f} h for the ladder",
                  flush=True)
    write_csv(Path(out) / "timing.csv", rows)


def part_screen(cfg, out, workers):
    cells = [(cfg, s, n, "energy", h, "screen") for s in cfg["dev_seeds"] for n in cfg["screen_sizes"]
             for h in cfg["energy_widths"]]
    rows = run_cells(cells, _cell, workers, out, "screening.csv", key=_key)
    d = pd.DataFrame(rows)
    d = d[d.failed == 0]
    score = d.groupby(d.hp.map(lambda s: json.loads(s)["hidden"])).val_risk.mean()
    win = int(score.idxmin())
    write_json(Path(out) / "winners.json", {"energy_hidden": win, "pooled_val_risk": score.to_dict(),
                                            "rule": "lowest mean validation risk over dev seeds x screen_sizes"})
    print(f"  [screen] energy width {win} (mean validation risk {score.to_dict()})", flush=True)


def part_fit(cfg, out, workers):
    hidden = _hidden(cfg, out) if "energy" in cfg["arms"] else 0
    cells = [(cfg, s, n, a, hidden if a == "energy" else 0, "fit")
             for s in cfg["seeds"] for n in cfg["sizes"] for a in cfg["arms"]]
    rows = run_cells(cells, _cell, workers, out, "kernels.csv", key=_key)
    write_csv(Path(out) / "kernels.csv", rows)


def part_densities(cfg, out, workers):
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    k = pd.read_csv(Path(out) / "kernels.csv")
    k = k[(k.failed == 0) & k.n_chains.isin(cfg["density_sizes"])].drop_duplicates(["seed", "n_chains", "arm"])
    eps = np.linspace(-cfg["eps_half"], cfg["eps_half"], cfg["eps_points"])
    j = [int(np.argmin(np.abs(grid - a))) for a in cfg["slice_a"]]
    rows = []
    for r in k.itertuples():
        kern = _rebuild(r.arm, [float(v) for v in r.theta.split(";")], r.hp, cfg)
        lk = _normalised(kern.log_transition_matrix(grid), weights)
        key = dict(seed=r.seed, n_chains=r.n_chains, arm=r.arm)
        if r.arm in SHIFT_ARMS:
            rows += [{**key, "what": "log_phi", "a": np.nan, "x": float(e), "value": float(v)}
                     for e, v in zip(eps, kern.log_phi(eps))]
        for a, jj in zip(cfg["slice_a"], j):
            rows += [{**key, "what": "log_K", "a": float(grid[jj]), "x": float(x), "value": float(v)}
                     for x, v in zip(grid, lk[:, jj])]
    true = _normalised(LaplaceAR1(FROZEN.rho).log_transition_matrix(grid), weights)
    b = LaplaceAR1(FROZEN.rho).b
    rows += [{"seed": -1, "n_chains": 0, "arm": "true", "what": "log_phi", "a": np.nan, "x": float(e),
              "value": float(-abs(e) / b - np.log(2 * b))} for e in eps]
    for a, jj in zip(cfg["slice_a"], j):
        rows += [{"seed": -1, "n_chains": 0, "arm": "true", "what": "log_K", "a": float(grid[jj]), "x": float(x),
                  "value": float(v)} for x, v in zip(grid, true[:, jj])]
    write_csv(Path(out) / "densities.csv", rows)
    print(f"  [densities] {len(k)} fitted kernels -> densities.csv ({len(rows)} rows)", flush=True)


# sq_ref depends only on the test bundle and the schedule; through a CSV it agrees to float64 round-off, not bit for
# bit. E30's measured contrast: same bundles 4e-17 relative, a different bundle ~0.17 relative.
SQ_REF_RTOL = 1e-12


def part_verify(cfg, out, workers=1, reference=None):
    """THE GATE. If E56 is not on E30's (= E03's) bundles, its rows cannot be joined with the networks: exit 1."""
    root = Path(__file__).resolve().parents[1]
    mine = pd.read_csv(Path(out) / "kernels.csv")
    mine = mine[mine.failed == 0]
    paths = [Path(reference)] if reference else sorted(root.glob("outputs/hpc_E30_s*/joint.csv"))
    if not paths:
        print("[verify] CANNOT VERIFY: no E30 joint.csv found (fetch outputs/hpc_E30_s* first)", flush=True)
        raise SystemExit(2)
    ref = pd.concat([pd.read_csv(p) for p in paths]).groupby(["seed", "n_chains", "t"])["sq_ref"].first()
    m = mine.groupby(["seed", "n_chains", "t"])["sq_ref"].first()
    common = ref.index.intersection(m.index)
    if len(common) == 0:
        print("[verify] CANNOT VERIFY: no (seed, n, t) cell in common with E30", flush=True)
        raise SystemExit(2)
    worst = float(((ref.loc[common] - m.loc[common]).abs() / ref.loc[common].abs()).max())
    write_json(Path(out) / "verify.json", {"checked": len(common), "worst_rel_sq_ref_diff": worst,
                                           "rtol": SQ_REF_RTOL, "reference": [str(p) for p in paths]})
    if worst > SQ_REF_RTOL:
        print(f"[verify] FAIL: sq_ref differs by {worst:.3e} relative over {len(common)} cells (rtol {SQ_REF_RTOL:.0e});"
              f" E56 is NOT on E03's bundles and cannot be joined with the networks", flush=True)
        raise SystemExit(1)
    print(f"[verify] OK: {len(common)} cells, sq_ref agrees with E30 to {worst:.3e} relative", flush=True)


def main() -> None:
    parser = experiment_parser("e56_kernel_families", "kernel families on the Laplace chain, on E03's bundles")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    w = args.workers
    parts = {"timing": ("seconds per EM iteration, every arm", lambda c, o: part_timing(c, o, w)),
             "screen": ("energy width on the dev seeds", lambda c, o: part_screen(c, o, w)),
             "fit": ("confirmatory fits", lambda c, o: part_fit(c, o, w)),
             "densities": ("learned laws for Fig. 6.3", lambda c, o: part_densities(c, o, w)),
             "verify": ("sq_ref identical to E30's", lambda c, o: part_verify(c, o, w))}
    if args.list_parts:
        print("\n".join(parts)); return
    selected = select_parts(parts, args.only)
    # As in E30 (N-30): the reference outputs are local, fetched artefacts, absent on the cluster, so the gate is
    # dropped from a DEFAULT run and must be asked for with --only verify after a fetch. It never passes vacuously.
    if not args.only:
        selected = {k: v for k, v in selected.items() if k != "verify"}
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
