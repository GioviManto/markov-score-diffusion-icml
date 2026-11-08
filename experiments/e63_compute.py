#!/usr/bin/env python
"""E63 — compute, not update counts: seconds per update for every arm on the same hardware.

The paper's step-efficiency claim counted parameter updates, EM iterations against SGD steps. One EM iteration is a
full BP pass over the data plus an M-step; one SGD step is one minibatch. E63 times both on one CPU thread (dev seed
10000 only, never a confirmatory seed), so that validation and test curves can be put on a wall-clock axis.

Parts
-----
timing      (arm, N) -> timing.csv: seconds per EM iteration (em_bp, spline, energy; noisy one-view data), per clean
            M-step (clean_energy, plus the one-off histogram), per SGD step (E03's six networks, batch 64), and the
            use-time seconds per test sequence over the twelve levels (BP on the grid; each network's forward pass)
timing_gpu  the networks' SGD steps and forward passes, and BP, on one GPU (if one is visible)

The curves themselves come from E65's refits (amendment of 3 Nov, before code) and are joined in tools/core/read_e63.py.
PREDECLARED in experiments/registry/E63.json (P1-P4, D1-D4) before this file was written.
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
from markov_score.em import clean_statistics, fit_em  # noqa: E402
from markov_score.kernels import MixtureInnovationKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv  # noqa: E402

import e03_baselines as e03  # noqa: E402
import e56_kernel_families as e56  # noqa: E402

N_SITES = FROZEN.n_sites
T_SCHEDULE = tuple(FROZEN.t_grid)
EM_ARMS = ("em_bp", "spline", "energy")
CLEAN_ARMS = ("clean_energy",)
# E03's screened winners (outputs/E03/winners.json), written out so a cluster job does not need E03's outputs
NETS = {"mlp": ({"hidden": (128, 128)}, 1e-3, "eps"), "window": ({"radius": 4, "width": 64}, 1e-3, "eps"),
        "tcn": ({"channels": 32, "dilations": (1, 2, 4, 8)}, 1e-3, "eps"), "bigru": ({"hidden": 32}, 1e-3, "eps"),
        "transformer": ({"d": 32, "heads": 4, "layers": 2}, 3e-3, "eps"), "unet": ({"channels": 16}, 1e-3, "eps")}

SETTINGS = dict(
    seed=10_000, sizes=(32, 128, 512, 2048), n_val=PROTOCOL.n_val, n_test=PROTOCOL.n_test,
    grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    em_init_rho=PROTOCOL.em_init_rho, em_init_var=PROTOCOL.em_init_var, components=8,
    shift_W=4.0, shift_lr=1e-2, shift_inner=10, energy_lr=1e-2, energy_inner=10, energy_hidden=16,
    em_arms=EM_ARMS, clean_arms=CLEAN_ARMS, nets=tuple(NETS),
    em_reps=10, clean_reps=10, net_warmup=20, net_steps=200, batch_size=PROTOCOL.batch_size, grad_clip=PROTOCOL.grad_clip,
    use_batch=2048, use_reps=3, torch_threads=1, device="cpu",
)
QUICK = dict(sizes=(32,), n_val=16, n_test=16, grid_size=101, em_reps=1, clean_reps=1, net_warmup=2, net_steps=5,
             use_batch=16, use_reps=1)


def _threads(n):
    import torch
    torch.set_num_threads(int(n))
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(int(n))
    except ImportError:                                   # the job also exports OMP/MKL/OPENBLAS_NUM_THREADS=1
        pass


def _em_kernel(cfg, arm, n):
    if arm == "em_bp":
        return MixtureInnovationKernel.init(cfg["components"], rho=0.3, var=0.8, rng=rng_for("e30-em-init", cfg["seed"], n))
    return e56._init_kernel({**cfg, "device": cfg["device"]}, arm, cfg["seed"], n, hidden=cfg["energy_hidden"])


def _median_tail(xs, k):
    return float(np.median(xs[-k:])) if xs else float("nan")


def _row(cfg, what, arm, n, sec, **extra):
    return dict(what=what, arm=arm, n_chains=n, device=cfg["device"], threads=cfg["torch_threads"],
                grid_size=cfg["grid_size"], seconds=sec, **extra)


def _em_cell(cfg, arm, n):
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    A, _, _, _ = e03._cell_common(cfg, cfg["seed"], n)
    groups = [(X, *alpha_delta(t)) for X, t in e03._one_view_groups(A, cfg["seed"])]
    trace = fit_em(_em_kernel(cfg, arm, n), grid, w, groups, n_iters=cfg["em_reps"] + 1, tol=0.0)[1]
    sec = _median_tail(trace.seconds, cfg["em_reps"])               # the first iteration is the warm-up
    print(f"  [E63] {arm:12s} N={n:>5}: {sec:.3f} s per EM iteration", flush=True)
    return [_row(cfg, "update", arm, n, sec, unit="em_iteration", n_measured=min(cfg["em_reps"], len(trace.seconds)))]


def _clean_cell(cfg, arm, n):
    import e58_clean_kernel as e58
    grid, _ = make_grid(cfg["half_width"], cfg["grid_size"])
    A, _, _, _ = e03._cell_common(cfg, cfg["seed"], n)
    t0 = time.perf_counter()
    stats = clean_statistics(grid, A)
    t_hist = time.perf_counter() - t0
    k = e58.init_kernel({**cfg, "energy_lr": cfg["energy_lr"], "energy_inner": cfg["energy_inner"]}, "laplace",
                        f"energy_h{cfg['energy_hidden']}", cfg["seed"], n, grid)
    secs = []
    for _ in range(cfg["clean_reps"] + 1):
        t0 = time.perf_counter()
        k = k.m_step(stats, grid)
        secs.append(time.perf_counter() - t0)
    sec = _median_tail(secs, cfg["clean_reps"])
    print(f"  [E63] {arm:12s} N={n:>5}: {sec:.3f} s per clean M-step (+{t_hist:.3f} s histogram once)", flush=True)
    return [_row(cfg, "update", arm, n, sec, unit="clean_m_step", n_measured=cfg["clean_reps"]),
            _row(cfg, "setup", arm, n, t_hist, unit="clean_histogram", n_measured=1)]


def _net_cell(cfg, arch, n):
    import torch
    from markov_score import nn_torch as nt
    hp, lr, mode = NETS[arch]
    A, _, test, _ = e03._cell_common(cfg, cfg["seed"], n)
    dev = nt.pick_device("cuda" if cfg["device"] == "cuda" else "cpu")
    model = nt.build(arch, N_SITES, **hp).to(dev)
    tcfg = nt.TrainConfig(steps=1, batch_size=cfg["batch_size"], lr=lr, grad_clip=cfg["grad_clip"], parameterization=mode,
                          t_levels=T_SCHEDULE, seed=0, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(0)
    sync = (lambda: torch.cuda.synchronize()) if dev.startswith("cuda") else (lambda: None)
    secs = []
    for i in range(cfg["net_warmup"] + cfg["net_steps"]):
        t0 = time.perf_counter()
        a, x, z, t, al, de = nt._draw_batch(A, tcfg, rng)
        loss = nt.dsm_loss(model, a, x, z, t, al, de, tcfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        sync()
        if i >= cfg["net_warmup"]:
            secs.append(time.perf_counter() - t0)
    sec = float(np.median(secs))
    den = nt.Denoiser(model.eval(), mode)
    use = _use_time(cfg, lambda X, t: den.posterior_mean(X, t), test, sync)
    print(f"  [E63] {arch:12s} N={n:>5}: {sec * 1e3:.2f} ms per SGD step, {use * 1e6:.1f} us per test sequence", flush=True)
    return [_row(cfg, "update", arch, n, sec, unit="sgd_step", n_measured=len(secs), n_params=nt.n_params(model)),
            _row(cfg, "use", arch, n, use, unit="test_sequence_12_levels", n_measured=cfg["use_reps"])]


def _use_time(cfg, mean_fn, test, sync=lambda: None):
    """Median over repetitions of the seconds per test sequence for the posterior mean at all twelve levels."""
    B = cfg["use_batch"]
    out = []
    for _ in range(cfg["use_reps"]):
        t0 = time.perf_counter()
        for t in T_SCHEDULE:
            mean_fn(test[t][0][:B], t)
        sync()
        out.append((time.perf_counter() - t0) / B)
    return float(np.median(out))


def _bp_use_cell(cfg, n):
    from markov_score.backend import get_xp
    from markov_score.priors import LaplaceAR1
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    _, _, test, _ = e03._cell_common(cfg, cfg["seed"], n)
    lk = LaplaceAR1(FROZEN.rho).log_transition_matrix(grid)
    xp = get_xp("gpu" if cfg["device"] == "cuda" else "cpu")
    use = _use_time(cfg, lambda X, t: grid_bp_batch(grid, w, lk, X, *alpha_delta(t), xp=xp)[0], test)
    print(f"  [E63] {'bp':12s}: {use * 1e6:.1f} us per test sequence", flush=True)
    return [_row(cfg, "use", "bp", n, use, unit="test_sequence_12_levels", n_measured=cfg["use_reps"])]


def _cell(args):
    cfg, kind, arm, n = args
    _threads(cfg["torch_threads"])
    if kind == "em":
        return _em_cell(cfg, arm, n)
    if kind == "clean":
        return _clean_cell(cfg, arm, n)
    if kind == "net":
        return _net_cell(cfg, arm, n)
    return _bp_use_cell(cfg, n)


def _cells(cfg, gpu=False):
    if gpu:
        return [(cfg, "net", a, n) for a in cfg["nets"] for n in cfg["sizes"]] + [(cfg, "bp", "bp", max(cfg["sizes"]))]
    return ([(cfg, "em", a, n) for a in cfg["em_arms"] for n in cfg["sizes"]]
            + [(cfg, "clean", a, n) for a in cfg["clean_arms"] for n in cfg["sizes"]]
            + [(cfg, "net", a, n) for a in cfg["nets"] for n in cfg["sizes"]]
            + [(cfg, "bp", "bp", max(cfg["sizes"]))])


def _key(cell):
    _, kind, arm, n = cell
    return f"{kind}_{arm}_n{n}_{cell[0]['device']}"


def part_timing(cfg, out, workers, gpu=False):
    c = dict(cfg, device="cuda") if gpu else cfg
    name = "timing_gpu.csv" if gpu else "timing.csv"
    # one worker: timings are only comparable when nothing else runs on the node
    write_csv(Path(out) / name, run_cells(_cells(c, gpu), _cell, 1, out, name, key=_key))


def main() -> None:
    parser = experiment_parser("e63_compute", "seconds per update for every arm on the same hardware")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    parts = {"timing": ("one CPU thread: EM iterations, clean M-steps, SGD steps, use time", lambda c, o: part_timing(c, o, 1)),
             "timing_gpu": ("one GPU: SGD steps, network and BP use time", lambda c, o: part_timing(c, o, 1, gpu=True))}
    if args.list_parts:
        print("\n".join(parts)); return
    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir)
    tag = "_".join(selected) if args.only else "all"
    resolved = {"n_sites": N_SITES, "t_schedule": list(T_SCHEDULE), "quick": args.quick, "parts": list(selected),
                "overrides": args.set, **{k: v for k, v in cfg.items()}}
    from markov_score.utils import write_json
    write_json(out / f"params_{tag}.json", {**resolved, **provenance(resolved)})
    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(cfg, out)


if __name__ == "__main__":
    main()
