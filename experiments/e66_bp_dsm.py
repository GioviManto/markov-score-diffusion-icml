#!/usr/bin/env python
"""E66 — the neural energy kernel trained by the networks' own objective: DSM on clean chains through a differentiable BP.

The model is BP with a learned kernel K_theta = exp(-E_theta(a, a')) / Z (E56's NeuralEnergyKernel, width 16, uniform
start); the denoiser is its posterior mean, computed by `markov_score.bp_torch` (grid_bp_batch's recursion in torch).
Training is E03's recipe for the networks: clean training chains, batch 64, a level drawn from the twelve for each
example, fresh noise every step, the eps-prediction loss with eps_hat = (x - alpha m_hat) / sqrt(Delta), Adam, gradient
clipping at 1, 32000 steps, checkpoints 500 ... 32000, selection by the validation score error on the bulk (E03's rule).
Training runs on a 201-point grid; every evaluation uses the 401-point grid (the kernel is a function, not a table).

Parts
-----
screen   learning rate over (1e-3, 3e-3, 1e-2) on dev seeds 10000-10001 at N = 32 and 2048, 8000 steps -> screen.csv,
         winners.json (the lowest pooled validation error)
fit      confirmatory seeds 0-15, N in (32, 128, 512, 2048), the screened learning rate -> kernels.csv (one row per level)

PREDECLARED in experiments/registry/E66.json (P1-P3, D1-D3) before this file was written.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from protocol import PROTOCOL  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.energy_kernel import NeuralEnergyKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

import e03_baselines as e03  # noqa: E402
import e58_clean_kernel as e58  # noqa: E402

N_SITES = FROZEN.n_sites
T_SCHEDULE = tuple(FROZEN.t_grid)

SETTINGS = dict(
    seeds=PROTOCOL.seeds, sizes=(32, 128, 512, 2048), dev_seeds=(10_000, 10_001), screen_sizes=(32, 2048),
    lrs=(1e-3, 3e-3, 1e-2), screen_steps=8000, lr=0.0,               # lr 0: read the screen's winners.json
    winners="", hidden=16, train_grid=201, grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    n_val=PROTOCOL.n_val, n_test=PROTOCOL.n_test, steps=PROTOCOL.net_steps, checkpoints=PROTOCOL.net_checkpoints,
    batch_size=PROTOCOL.batch_size, grad_clip=PROTOCOL.grad_clip, torch_threads=1, device="cpu",
)
QUICK = dict(seeds=(0,), sizes=(32,), dev_seeds=(10_000,), screen_sizes=(32,), lrs=(3e-3,), screen_steps=20, steps=20,
             checkpoints=(10, 20), n_val=16, n_test=16, grid_size=101, train_grid=51)


# ---------------------------------------------------------------------------------------- training

def train(cfg, A, seed, n, lr, steps, checkpoints):
    """DSM through BP; returns {step: NeuralEnergyKernel} at the checkpoints and the seconds per step."""
    import torch
    from markov_score.bp_torch import posterior_mean
    torch.set_num_threads(int(cfg["torch_threads"]))
    dev = cfg["device"]
    g_np, w_np = make_grid(cfg["half_width"], cfg["train_grid"])
    G, W = torch.as_tensor(g_np, device=dev), torch.as_tensor(w_np, device=dev)
    k0 = NeuralEnergyKernel.init(hidden=cfg["hidden"], half_width=cfg["half_width"], device=dev,
                                 seed=int(rng_for("e66-init", seed, n).integers(0, 2**31 - 1)))
    net = k0._net()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    rng = rng_for("e66-train", seed, n)
    torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
    ts = np.asarray(T_SCHEDULE)
    saved, t0 = {}, time.perf_counter()
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(A), size=min(cfg["batch_size"], len(A)))      # nn_torch._draw_batch's rule
        t = rng.choice(ts, size=len(idx))
        al, de = np.exp(-t), 1.0 - np.exp(-2.0 * t)
        z = rng.standard_normal((len(idx), A.shape[1]))
        x = al[:, None] * A[idx] + np.sqrt(de)[:, None] * z
        X, AL, DE = (torch.as_tensor(v, device=dev) for v in (x, al, de))
        m = posterior_mean(G, W, k0._log_k(net, g_np), X, AL, DE)
        eps_hat = (X - AL[:, None] * m) / torch.sqrt(DE)[:, None]
        loss = ((eps_hat - torch.as_tensor(z, device=dev)) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
        opt.step()
        if step in checkpoints:
            saved[step] = replace(k0, state={k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, device="cpu")
    return saved, (time.perf_counter() - t0) / steps


def _val_dsm(lk, grid, w, a_val, seed):
    """The eps-prediction DSM loss on the clean validation chains, every level, one fixed noise draw (as E65)."""
    rng = rng_for("e66-valdsm", seed)
    tot, n = 0.0, 0
    for t in T_SCHEDULE:
        al, de = alpha_delta(t)
        z = rng.standard_normal(a_val.shape)
        x = al * a_val + np.sqrt(de) * z
        m, _ = grid_bp_batch(grid, w, lk, x, al, de)
        tot += float((((x - al * m) / np.sqrt(de) - z) ** 2).mean(axis=1).sum()); n += len(a_val)
    return tot / n


def _eval(lk, bundle, grid, w, region):
    pairs = []
    for t in T_SCHEDULE:
        X, m_ref = bundle[t]
        m, _ = grid_bp_batch(grid, w, lk, X, *alpha_delta(t))
        pairs.append(e03._sq(m, m_ref, X, t, e03.REGIONS[region]))
    return pairs


def _a_val(cfg, seed):
    prior, rng = LaplaceAR1(FROZEN.rho), rng_for("exp31-val", seed)
    return np.stack([prior.sample(rng, N_SITES) for _ in range(cfg["n_val"])])


def _fit_select(cfg, seed, n, lr, steps, checkpoints):
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    A, val, test, base = e03._cell_common(cfg, seed, n)
    saved, sec = train(cfg, A, seed, n, lr, steps, set(checkpoints))
    a_val = _a_val(cfg, seed)
    curve = {}
    for step, k in sorted(saved.items()):
        lk = k.log_transition_matrix(grid)
        try:
            curve[step] = (e03._risk(_eval(lk, val, grid, w, "bulk")), e03._risk(_eval(lk, val, grid, w, "all")),
                           _val_dsm(lk, grid, w, a_val, seed))
        except FloatingPointError:
            curve[step] = (np.inf, np.inf, np.inf)
    best = min(curve, key=lambda s: (curve[s][0], s))
    k = saved[best]
    lk = k.log_transition_matrix(grid)
    info = dict(checkpoint=best, at_cap=int(best == max(saved)), val_risk=curve[best][0], val_all=curve[best][1],
                val_dsm=curve[best][2], sec_per_step=sec, lr=lr, n_params=k.n_params,
                val_curve=";".join(f"{s}:{a:.6g}/{b:.6g}/{c:.6g}" for s, (a, b, c) in sorted(curve.items())),
                theta=";".join(f"{v:.10g}" for v in k.theta))
    return k, lk, grid, w, val, test, base, info


# ---------------------------------------------------------------------------------------- parts

def _screen_cell(args):
    cfg, seed, n, lr = args
    _, lk, grid, w, val, _, base, info = _fit_select(cfg, seed, n, lr, cfg["screen_steps"],
                                                     tuple(c for c in cfg["checkpoints"] if c <= cfg["screen_steps"])
                                                     or (cfg["screen_steps"],))
    pairs = _eval(lk, val, grid, w, "bulk")
    print(f"  [E66 screen] seed={seed} n={n:>5} lr={lr:g}: val(bulk)={info['val_risk']:.4f} ck={info['checkpoint']}", flush=True)
    return [{**base, "lr": lr, "sq_err": sum(e for e, _ in pairs), "sq_ref": sum(r for _, r in pairs), **info}]


def _fit_cell(args):
    cfg, seed, n, lr = args
    t0 = time.perf_counter()
    try:
        k, lk, grid, w, val, test, base, info = _fit_select(cfg, seed, n, lr, cfg["steps"], cfg["checkpoints"])
        pairs = _eval(lk, test, grid, w, "all")
    except FloatingPointError as exc:                        # a failed fit is a row and counts against (E33 amendment 3)
        print(f"  [E66] seed={seed} n={n:>5} FAILED: {exc}", flush=True)
        _, _, _, base = e03._cell_common(cfg, seed, n)
        return [{**base, "arm": "bp_dsm", "t": t, "failed": 1, "stop_reason": str(exc), "risk_pooled": np.nan} for t in T_SCHEDULE]
    pooled = e03._risk(pairs)
    ctx = dict(grid=grid, w=w, log_k=LaplaceAR1(FROZEN.rho).log_transition_matrix(grid), log_mu=None)
    kh = e58.kernel_hellinger(lk, ctx)
    print(f"  [E66] seed={seed} n={n:>5} bp_dsm test={pooled:.4f} ck={info['checkpoint']} H={kh:.4f} "
          f"({info['sec_per_step'] * 1e3:.0f} ms/step, {time.perf_counter() - t0:.0f} s)", flush=True)
    return [{**base, "arm": "bp_dsm", "t": t, "sq_err": e, "sq_ref": r, "risk_level": float(np.sqrt(e / r)) if r > 0 else np.nan,
             "risk_pooled": pooled, **info, "kernel_hellinger": kh, "wall": time.perf_counter() - t0, "failed": 0}
            for t, (e, r) in zip(T_SCHEDULE, pairs)]


def _winner(screen_rows):
    tot = {}
    for r in screen_rows:
        e, s = tot.get(r["lr"], (0.0, 0.0))
        tot[r["lr"]] = (e + float(r["sq_err"]), s + float(r["sq_ref"]))
    risk = {lr: float(np.sqrt(e / s)) for lr, (e, s) in tot.items()}
    best = min(risk, key=risk.get)
    return {"lr": float(best), "screen_risk": risk[best], "all": {str(k): v for k, v in risk.items()}}


def part_screen(cfg, out, workers):
    cells = [(cfg, s, n, lr) for s in cfg["dev_seeds"] for n in cfg["screen_sizes"] for lr in cfg["lrs"]]
    rows = run_cells(cells, _screen_cell, workers, out, "screen.csv", key=lambda c: f"s{c[1]}_n{c[2]}_lr{c[3]:g}")
    write_csv(Path(out) / "screen.csv", rows)
    win = _winner(rows)
    write_json(Path(out) / "winners.json", win)
    print(f"  [E66 screen] winner lr = {win['lr']:g} (pooled validation error {win['screen_risk']:.4f})", flush=True)


def part_fit(cfg, out, workers):
    lr = float(cfg["lr"])
    if lr <= 0:
        path = Path(cfg["winners"] or Path(out) / "winners.json")
        lr = float(json.loads(path.read_text())["lr"])
    cells = [(cfg, s, n, lr) for s in cfg["seeds"] for n in cfg["sizes"]]
    write_csv(Path(out) / "kernels.csv", run_cells(cells, _fit_cell, workers, out, "kernels.csv",
                                                    key=lambda c: f"s{c[1]}_n{c[2]}"))


def main() -> None:
    parser = experiment_parser("e66_bp_dsm", "the neural kernel trained by DSM through differentiable BP")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    w = args.workers
    parts = {"screen": ("learning-rate screen on the dev seeds", lambda c, o: part_screen(c, o, w)),
             "fit": ("confirmatory fits", lambda c, o: part_fit(c, o, w))}
    if args.list_parts:
        print("\n".join(parts)); return
    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir)
    tag = "_".join(selected) if args.only else "all"
    resolved = {"n_sites": N_SITES, "t_schedule": list(T_SCHEDULE), "quick": args.quick, "parts": list(selected),
                "overrides": args.set, **cfg}
    write_json(out / f"params_{tag}.json", {**resolved, **provenance(resolved)})
    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(cfg, out)


if __name__ == "__main__":
    main()
