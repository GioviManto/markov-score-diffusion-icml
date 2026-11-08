#!/usr/bin/env python
"""E64 — generation on the Laplace chain down to t = 0.005, every denoiser through the same sampler.

E55's ancestral sampler (x ~ N(0, 1) at t = 3, 400 geometric bridge steps down to t_min = 0.005, posterior-mean readout
at t_min), 2048 generated sequences per arm, the SAME sampler noise for every arm of a cell. Arms:

    exact_ref / exact        two independent direct draws of the chain (the reference R and the Monte Carlo floor)
    oracle_sampler           the true kernel through BP in the sampler
    em_bp, spline, energy,   frozen fitted kernels (experiments/data/e64_kernels.csv: E02a, E56, E58 at the same seed
    clean_energy             and N) through BP in the sampler; em_bp and clean_energy also sampled DIRECTLY as Markov
                             chains on the grid (`<arm>_direct`)
    unet, bigru, window      E03's winners RETRAINED for generation on log-uniform continuous levels in [t_min, 3],
                             E03's recipe otherwise, selected by the validation score error on the twelve levels (bulk)

Metrics: markov_score.gen_metrics.compare against R (sliced W2, energy distance, marginal W1, covariance error, AR(0.85)
residual kurtosis and autocorrelation, log-likelihood under the true chain at t_min).

Part
----
gen   (seed, N) -> generation.csv, one row per arm

PREDECLARED in experiments/registry/E64.json (P1-P4, D1-D4) before this file was written.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from protocol import PROTOCOL  # noqa: E402
from markov_score import ancestral as an  # noqa: E402
from markov_score import gen_metrics as gm  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.energy_kernel import NeuralEnergyKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

import e03_baselines as e03  # noqa: E402
import e05_generation as e05  # noqa: E402
import e56_kernel_families as e56  # noqa: E402
from e63_compute import NETS  # noqa: E402

N_SITES = FROZEN.n_sites
T_SCHEDULE = tuple(FROZEN.t_grid)
KERNEL_ARMS = ("em_bp", "spline", "energy", "clean_energy")
DIRECT_ARMS = ("em_bp", "clean_energy")
NET_ARMS = ("unet", "bigru", "window")

SETTINGS = dict(
    seeds=tuple(range(8)), sizes=(128, 2048), n_gen=2048, gen_steps=400, t_min=0.005, t_max=3.0,
    grid_size=FROZEN.n_grid, half_width=FROZEN.half_width, n_val=PROTOCOL.n_val, n_test=PROTOCOL.n_test,
    kernel_arms=KERNEL_ARMS, direct_arms=DIRECT_ARMS, net_arms=NET_ARMS,
    kernels_csv="experiments/data/e64_kernels.csv", shift_W=4.0,
    net_steps=PROTOCOL.net_steps, net_checkpoints=PROTOCOL.net_checkpoints, batch_size=PROTOCOL.batch_size,
    grad_clip=PROTOCOL.grad_clip, torch_threads=1,
)
QUICK = dict(seeds=(0,), sizes=(128,), n_gen=64, gen_steps=10, grid_size=101, n_val=16, n_test=16,
             net_steps=20, net_checkpoints=(10, 20), net_arms=("window",))


# ---------------------------------------------------------------------------------------- denoisers

def bp_mean_fn(grid, w, log_k):
    def mean(X, t):
        al, de = alpha_delta(t)
        return grid_bp_batch(grid, w, log_k, X, al, de)[0]
    return mean


def load_kernel(cfg, row):
    theta = np.array([float(v) for v in str(row["theta"]).split(";")])
    hp = json.loads(row["hp"])
    if row["arm"] == "em_bp":
        return e05._mixture_from_theta(theta, int(hp["components"]))
    if row["arm"] == "spline":
        return e56._rebuild("spline", theta, row["hp"], {**cfg, "shift_W": hp.get("W", cfg["shift_W"])})
    return NeuralEnergyKernel.from_theta(theta, int(hp["hidden"]), cfg["half_width"])


def train_continuous(cfg, arch, A, val, seed, n):
    """E03's recipe with the level drawn log-uniformly in [t_min, t_max]; selection by the bulk validation error."""
    import torch
    from markov_score import nn_torch as nt
    torch.set_num_threads(int(cfg["torch_threads"]))
    hp, lr, mode = NETS[arch]
    s = int(rng_for("e64-net-seed", arch, seed, n).integers(0, 2**31 - 1))
    torch.manual_seed(s)
    rng = np.random.default_rng(s)
    model = nt.build(arch, N_SITES, **hp)
    tcfg = nt.TrainConfig(steps=1, batch_size=cfg["batch_size"], lr=lr, grad_clip=cfg["grad_clip"], parameterization=mode)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ck = set(int(c) for c in cfg["net_checkpoints"])
    lo, hi = np.log(cfg["t_min"]), np.log(cfg["t_max"])
    saved = {}
    model.train()
    for step in range(1, max(ck) + 1):
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
            saved[step] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    curve = {}
    for step, state in sorted(saved.items()):
        den = nt.Denoiser(nt.load_checkpoint(model, state), mode)
        curve[step] = e03._risk(e03._eval(den, val, "bulk"))
    best = min(curve, key=lambda k: (curve[k], k))
    den = nt.Denoiser(nt.load_checkpoint(model, saved[best]).eval(), mode)
    return den, dict(checkpoint=best, val_risk=curve[best], n_params=nt.n_params(model))


# ---------------------------------------------------------------------------------------- one cell

def _cell(args):
    cfg, seed, n = args
    import pandas as pd
    t0 = time.perf_counter()
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    prior = LaplaceAR1(FROZEN.rho)
    log_k = prior.log_transition_matrix(grid)
    i = np.arange(N_SITES)
    sigma = FROZEN.rho ** np.abs(i[:, None] - i[None, :]).astype(float)
    N, L = cfg["n_gen"], N_SITES
    ladder = np.geomspace(cfg["t_max"], cfg["t_min"], cfg["gen_steps"] + 1)
    R = _draws(prior, rng_for("e64-gen-exact", seed, n), N, L)
    noise = lambda: rng_for("e64-gen-noise", seed, n)  # noqa: E731   the same sampler noise for every arm
    base = dict(seed=seed, n_chains=n, n_gen=N, gen_steps=cfg["gen_steps"], t_min=cfg["t_min"],
                ref_hash=hashlib.sha256(np.ascontiguousarray(R).tobytes()).hexdigest()[:12])
    A, val, test, _ = e03._cell_common(cfg, seed, n)
    rows = []

    def add(arm, G, **extra):
        m = gm.compare(G, R, FROZEN.rho, sigma, grid, w, log_k, cfg["t_min"], seed=seed)
        rows.append({**base, "arm": arm, **m, **extra, "failed": 0})
        print(f"  [E64] seed={seed} n={n:>5} {arm:20s} sw2={m['sw2']:.4f} kurt={m['kurt_gen']:.2f} "
              f"(ref {m['kurt_ref']:.2f}) llgap={m['loglik_gap']:.3f}", flush=True)

    add("exact", _draws(prior, rng_for("e64-gen-exact2", seed, n), N, L))
    add("oracle_sampler", an.generate(bp_mean_fn(grid, w, log_k), N, L, ladder, noise()))
    table = pd.read_csv(Path(__file__).resolve().parents[1] / cfg["kernels_csv"])
    for arm in cfg["kernel_arms"]:
        r = table[(table.arm == arm) & (table.seed == seed) & (table.n_chains == n)]
        if len(r) != 1:
            rows.append({**base, "arm": arm, "failed": 1, "stop_reason": "no frozen kernel"}); continue
        lk = load_kernel(cfg, r.iloc[0]).log_transition_matrix(grid)
        # gate: the rebuilt kernel scores what its frozen row says it scored, on the same test bundle
        rebuilt = e03._risk([e03._sq(bp_mean_fn(grid, w, lk)(X, t), m, X, t, e03.REGIONS["all"])
                             for t, (X, m) in ((t, test[t]) for t in T_SCHEDULE)])
        add(arm, an.generate(bp_mean_fn(grid, w, lk), N, L, ladder, noise()), frozen_risk=float(r.iloc[0].risk_pooled),
            rebuilt_risk=rebuilt)
        if arm in cfg["direct_arms"]:
            log_mu = -0.5 * grid ** 2                              # the first-site law BP uses (grid_bp_batch's default)
            add(f"{arm}_direct", an.sample_grid_chain(lk, log_mu, grid, w, N, L, rng_for("e64-gen-direct", seed, n)))
    for arch in cfg["net_arms"]:
        den, info = train_continuous(cfg, arch, A, val, seed, n)
        add(f"{arch}_cont", an.generate(den.posterior_mean, N, L, ladder, noise()), **info)
    print(f"  [E64] seed={seed} n={n:>5} done in {time.perf_counter() - t0:.0f} s", flush=True)
    return rows


def _draws(prior, rng, n, L):
    return np.stack([prior.sample(rng, L) for _ in range(n)])


def part_gen(cfg, out, workers):
    cells = [(cfg, s, n) for s in cfg["seeds"] for n in cfg["sizes"]]
    write_csv(Path(out) / "generation.csv", run_cells(cells, _cell, workers, out, "generation.csv",
                                                       key=lambda c: f"s{c[1]}_n{c[2]}"))


def main() -> None:
    parser = experiment_parser("e64_generation", "generation on the Laplace chain, every denoiser through one sampler")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    w = args.workers
    parts = {"gen": ("generation and its metrics", lambda c, o: part_gen(c, o, w))}
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
