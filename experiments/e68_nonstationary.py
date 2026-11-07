#!/usr/bin/env python
"""E68 — a chain whose kernel changes along the sequence: one shared kernel, per-link kernels, or one kernel of the site?

The user's ablation (7 Nov 2026). The chains (markov_score.nonstationary.NonstationaryLaplaceChain, d_rho = 0.1,
d_logsd = 0.35) have a Laplace kernel K_k at each link k, its lag-one correlation rho_k and the sites' standard
deviations s_k drifting (drift05, drift10) or switching at mid-chain (switch025, switch05); `stationary` is E03's chain.
The noise channel and the twelve levels are unchanged. The reference is per-link BP (markov_score.site_bp) with the
true kernels and first-site law on 601 points (amendment 2).

Arms (kernels by EM through per-link BP from one noisy view per training chain, chains split evenly over the twelve
levels; 201 points; every kernel arm learns a Gaussian first-site law N(m, v) by its closed-form EM update; the
checkpoint of its ladder with the lowest pooled validation risk; networks by E03's recipe on the clean chains):
    shared_mixture    the thesis mixture (C = 8) at every link, statistics summed over links      E56 ladder to 1200
    persite_mixture   31 independent C = 8 mixtures, one per link, each on its own link's statistics   ladder to 300
    shared_energy     NeuralEnergyKernel (width 16, fresh_on_reject) at every link                      ladder to 1200
    site_energy       SiteEnergyKernel: the same network with the link position as a third input        ladder to 600
    mlp, window       E03's screened winners
    window_pos        the window winner plus position features (nn_torch.WindowPosHead), not screened
    lmmse             the exact covariance
The mixture arms stop early when the mean log-evidence per chain rises by less than `em_tol` in one iteration (later
checkpoints are then the final iterate); the energy arms run to their cap (amendment of 7 Nov, before any run: a
rejected generalised M-step leaves an energy kernel unchanged and the evidence flat for one iteration, which is not
convergence; amendment 1). The networks are selected on E03's validation region (the bulk sites), the kernels on all sites.

Parts
-----
timing    one dev seed at N = 2048: seconds per EM iteration of each kernel arm, per 1000 steps of each network
bundles   per (cell, seed): validation and test bundles with the reference, cached under bundles/; for seed 0 the
          test reference is recomputed at 1201 points (`ref_check_rel`, gated by the reader)
arms      cells x seeds x sizes x arms -> arms.csv, one row per (cell, seed, n, arm, t)

PREDECLARED in experiments/registry/E68.json (P1-P5, D1-D4) before this file was written.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import e03_baselines as e03
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts
from frozen_config import FROZEN
from protocol import PROTOCOL

from markov_score.bp_grid import make_grid
from markov_score.em import ExpectedStatistics
from markov_score.energy_kernel import NeuralEnergyKernel
from markov_score.kernels import MixtureInnovationKernel
from markov_score.noising import alpha_delta
from markov_score.nonstationary import NonstationaryLaplaceChain
from markov_score.site_bp import site_bp_batch
from markov_score.site_energy_kernel import SiteEnergyKernel
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json

N_SITES, T_SCHEDULE = FROZEN.n_sites, tuple(FROZEN.t_grid)
CELLS = {"stationary": ("drift", 0.0), "drift05": ("drift", 0.5), "drift10": ("drift", 1.0),
         "switch025": ("switch", 0.25), "switch05": ("switch", 0.5)}
KERNEL_ARMS = ("persite_mixture", "site_energy", "shared_energy", "shared_mixture")
NET_ARMS = ("window_pos", "window", "mlp")
ARMS = KERNEL_ARMS + NET_ARMS + ("lmmse",)                     # slowest first, for the worker pool's tail
LADDER = (10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200)   # E56's ladder, to 1200
WINDOW = {"hp": {"radius": 4, "width": 64}, "lr": 1e-3, "mode": "eps"}

SETTINGS = dict(
    cells=tuple(CELLS), seeds=tuple(range(8)), dev_seeds=(10_000,), sizes=(32, 128, 512, 2048), arms=ARMS,
    d_rho=0.1, d_logsd=0.35, grid_size=201, ref_grid=601, check_grid=1201, half_width=8.0,
    n_val=PROTOCOL.n_val, n_test=PROTOCOL.n_test, components=8,
    em_init_rho=PROTOCOL.em_init_rho, em_init_var=PROTOCOL.em_init_var, em_tol=1e-9,
    ladders={"shared_mixture": LADDER, "persite_mixture": tuple(c for c in LADDER if c <= 300),
             "shared_energy": LADDER, "site_energy": tuple(c for c in LADDER if c <= 600)},
    energy_hidden=16, energy_lr=1e-2, energy_inner=10,
    nets={"mlp": {"hp": {"hidden": [128, 128]}, "lr": 1e-3, "mode": "eps"}, "window": WINDOW, "window_pos": WINDOW},
    net_steps=PROTOCOL.net_steps, net_checkpoints=PROTOCOL.net_checkpoints, batch_size=PROTOCOL.batch_size,
    grad_clip=PROTOCOL.grad_clip, torch_threads=1, timing_n=2048, timing_iters=3, timing_net_steps=200,
)
QUICK = dict(cells=("stationary", "switch05"), seeds=(0,), sizes=(32,), grid_size=101, ref_grid=161, check_grid=241,
             n_val=32, n_test=32, energy_hidden=8, net_checkpoints=(100, 200),
             ladders={"shared_mixture": (2, 4), "persite_mixture": (2, 4), "shared_energy": (2, 4), "site_energy": (2, 4)},
             timing_n=64, timing_iters=1, timing_net_steps=20)


def chain_of(cfg, cell):
    schedule, delta = CELLS[cell]
    return NonstationaryLaplaceChain(schedule, delta, n_sites=N_SITES, d_rho=cfg["d_rho"], d_logsd=cfg["d_logsd"])


def log_gauss(grid, m, v):
    return -0.5 * (grid - m) ** 2 / v - 0.5 * np.log(2 * np.pi * v)


# ---------------------------------------------------------------------------------------- bundles (cached)

def _bundle_path(out, kind, cell, seed):
    return Path(out) / "bundles" / f"{kind}_{cell}_s{seed}.npz"


def _reference(chain, grid, w, X, t):
    al, de = alpha_delta(t)
    return site_bp_batch(grid, w, chain.log_kernels(grid), X, al, de, chain.log_mu(grid)).means


def make_bundle(cfg, kind, cell, seed):
    chain = chain_of(cfg, cell)
    grid, w = make_grid(cfg["half_width"], cfg["ref_grid"])
    rng = rng_for(f"e68-{kind}", cell, seed)
    A = chain.sample(rng, cfg["n_val"] if kind == "val" else cfg["n_test"])
    out = {}
    for t in T_SCHEDULE:
        al, de = alpha_delta(t)
        X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
        out[t] = (X, _reference(chain, grid, w, X, t))
    return out


def load_bundle(out, kind, cell, seed):
    z = np.load(_bundle_path(out, kind, cell, seed))
    return {t: (z[f"X_{i}"], z[f"m_{i}"]) for i, t in enumerate(T_SCHEDULE)}


def _ref_check(cfg, cell, bundle):
    """Relative change of the reference score from ref_grid to check_grid points, pooled over levels and sites."""
    chain = chain_of(cfg, cell)
    grid, w = make_grid(cfg["half_width"], cfg["check_grid"])
    num = den = 0.0
    for t, (X, m_ref) in bundle.items():
        al, de = alpha_delta(t)
        s_ref, s_fine = (al * m_ref - X) / de, (al * _reference(chain, grid, w, X, t) - X) / de
        num += float(((s_ref - s_fine) ** 2).sum()); den += float((s_fine**2).sum())
    return float(np.sqrt(num / den))


def _bundle_cell(args):
    cfg, out, kind, cell, seed = args
    p = _bundle_path(out, kind, cell, seed)
    t0 = time.perf_counter()
    b = make_bundle(cfg, kind, cell, seed)
    ensure_dir(p.parent)
    np.savez(p.with_suffix(".tmp.npz"), **{f"X_{i}": b[t][0] for i, t in enumerate(T_SCHEDULE)},
             **{f"m_{i}": b[t][1] for i, t in enumerate(T_SCHEDULE)})
    p.with_suffix(".tmp.npz").replace(p)
    check = _ref_check(cfg, cell, b) if (kind == "test" and seed == cfg["seeds"][0]) else np.nan
    print(f"  [E68 bundle] {kind} {cell} seed={seed} {time.perf_counter() - t0:.0f} s"
          + ("" if np.isnan(check) else f" ref_check={check:.2e}"), flush=True)
    return [{"kind": kind, "cell": cell, "seed": seed, "bundle": e03._bundle_hash(b), "ref_check_rel": check,
             "seconds": time.perf_counter() - t0}]


# ---------------------------------------------------------------------------------------- kernel arms

def training_views(A, cell, seed):
    """One noisy view per chain, the chains split evenly over the twelve levels (E62's protocol); stacked."""
    idx = np.array_split(rng_for("e68-em-split", cell, seed, len(A)).permutation(len(A)), len(T_SCHEDULE))
    noise = rng_for("e68-em-noise", cell, seed, len(A))
    X, al, de = np.empty_like(A), np.empty(len(A)), np.empty(len(A))
    for t, ix in zip(T_SCHEDULE, idx):
        a_, d_ = alpha_delta(t)
        X[ix] = a_ * A[ix] + np.sqrt(d_) * noise.standard_normal(A[ix].shape)
        al[ix], de[ix] = a_, d_
    return X, al, de


class KernelArm:
    """One kernel arm's parameters and its log kernels per link; `step` is one M-step from per-link statistics."""

    def __init__(self, arm, cfg, grid, seed_key):
        self.arm, self.grid, n_links = arm, grid, N_SITES - 1
        if arm == "shared_mixture":
            self.k = MixtureInnovationKernel.init(cfg["components"], rho=cfg["em_init_rho"], var=cfg["em_init_var"],
                                                  rng=rng_for("e68-mix-init", *seed_key))
        elif arm == "persite_mixture":
            self.k = [MixtureInnovationKernel.init(cfg["components"], rho=cfg["em_init_rho"], var=cfg["em_init_var"],
                                                   rng=rng_for("e68-mix-init", *seed_key, link)) for link in range(n_links)]
        elif arm == "shared_energy":
            self.k = NeuralEnergyKernel.init(cfg["energy_hidden"], int(rng_for("e68-energy-init", *seed_key).integers(2**31)),
                                             cfg["half_width"], lr=cfg["energy_lr"], n_inner=cfg["energy_inner"],
                                             fresh_on_reject=True)
        elif arm == "site_energy":
            self.k = SiteEnergyKernel.init(cfg["energy_hidden"], int(rng_for("e68-energy-init", *seed_key).integers(2**31)),
                                           cfg["half_width"], n_links, lr=cfg["energy_lr"], n_inner=cfg["energy_inner"])
        else:
            raise ValueError(arm)

    def log_kernels(self):
        if self.arm == "persite_mixture":
            return np.stack([k.log_transition_matrix(self.grid) for k in self.k])
        if self.arm == "site_energy":
            return self.k.log_transition_matrices(self.grid)
        return self.k.log_transition_matrix(self.grid)

    def step(self, xi, n_chains):
        def stats(x, n_edges):
            return ExpectedStatistics(xi=x, site1=np.zeros(len(self.grid)), log_evidence=0.0, n_edges=n_edges,
                                      n_chains=n_chains)
        new = object.__new__(KernelArm)
        new.arm, new.grid = self.arm, self.grid
        if self.arm == "persite_mixture":
            new.k = [k.m_step(stats(x, n_chains), self.grid) for k, x in zip(self.k, xi)]
        elif self.arm == "site_energy":
            new.k = self.k.m_step(stats(xi, n_chains * len(xi)), self.grid)
        else:
            new.k = self.k.m_step(stats(xi.sum(axis=0), n_chains * len(xi)), self.grid)
        return new

    def theta(self):
        if self.arm == "persite_mixture":
            return np.concatenate([np.asarray(k.theta) for k in self.k])
        return np.asarray(self.k.theta)

    def describe(self):
        if self.arm == "persite_mixture":
            return {"fit_alpha": ";".join(f"{k.rho:.5f}" for k in self.k),
                    "fit_sd": ";".join(f"{np.sqrt(k.innovation_moments['innovation_var']):.5f}" for k in self.k)}
        if self.arm == "shared_mixture":
            return {"fit_alpha": f"{self.k.rho:.5f}", "fit_sd": f"{np.sqrt(self.k.innovation_moments['innovation_var']):.5f}"}
        return {"theta": ";".join(f"{v:.10g}" for v in self.theta())}


def _pairs(lks, mu, grid, w, bundle):
    out = []
    for t, (X, m_ref) in bundle.items():
        al, de = alpha_delta(t)
        out.append(e03._sq(site_bp_batch(grid, w, lks, X, al, de, log_gauss(grid, *mu)).means, m_ref, X, t,
                           e03.REGIONS["all"]))
    return out


def fit_kernel(arm, cfg, A, cell, seed, val):
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    X, al, de = training_views(A, cell, seed)
    ladder = tuple(sorted(cfg["ladders"][arm]))
    model, mu = KernelArm(arm, cfg, grid, (cell, seed, len(A))), (0.0, 1.0)
    saved, evidence, stop, t0 = {}, [], "cap", time.perf_counter()
    for it in range(1, max(ladder) + 1):
        r = site_bp_batch(grid, w, model.log_kernels(), X, al, de, log_gauss(grid, *mu), pairs=True)
        evidence.append(float(r.log_evidence.mean()))
        model = model.step(r.xi, len(A))
        m1 = float(r.site1 @ grid) / len(A)
        mu = (m1, max(float(r.site1 @ grid**2) / len(A) - m1**2, (grid[1] - grid[0]) ** 2))
        if it in ladder:
            saved[it] = (model, mu)
        if arm.endswith("mixture") and it > 1 and evidence[-1] - evidence[-2] < cfg["em_tol"]:
            stop = "tol"
            for c in ladder:
                if c > it:
                    saved[c] = (model, mu)
            break
    wall = time.perf_counter() - t0
    curve = {c: e03._risk(_pairs(m.log_kernels(), mu_, grid, w, val)) for c, (m, mu_) in saved.items()}
    best = min(curve, key=curve.get)
    model, mu = saved[best]
    drops = -np.diff(evidence) if len(evidence) > 1 else np.zeros(1)
    info = {"checkpoint": best, "at_cap": int(best == max(ladder)), "wall": wall, "n_iters": len(evidence),
            "sec_per_iter": wall / len(evidence), "stop_reason": stop, "val_risk": curve[best],
            "val_curve": ";".join(f"{c}:{v:.5f}" for c, v in sorted(curve.items())),
            "monotone_violation": float(max(0.0, drops.max())), "n_params": int(model.theta().size),
            "mu_mean": mu[0], "mu_var": mu[1], **model.describe()}
    return model, mu, grid, w, info


# ---------------------------------------------------------------------------------------- one arm cell

def _rows(base, arm, pairs, info):
    pooled = e03._risk(pairs)
    return [{**base, "arm": arm, "t": t, "sq_err": e, "sq_ref": r, "risk_level": float(np.sqrt(e / r)) if r > 0 else np.nan,
             "risk_pooled": pooled, "failed": 0, **info} for t, (e, r) in zip(T_SCHEDULE, pairs)]


def _arm_cell(args):
    cfg, out, cell, seed, n, arm = args
    import torch
    torch.set_num_threads(int(cfg["torch_threads"]))
    val, test = load_bundle(out, "val", cell, seed), load_bundle(out, "test", cell, seed)
    base = {"cell": cell, "seed": seed, "n_chains": n, "val_bundle": e03._bundle_hash(val),
            "test_bundle": e03._bundle_hash(test)}
    A = chain_of(cfg, cell).sample(rng_for("e68-train", cell, seed, n), n)
    try:
        if arm == "lmmse":
            cov = chain_of(cfg, cell).covariance()
            pairs = []
            for t, (X, m_ref) in test.items():
                al, de = alpha_delta(t)
                m = X @ (al * cov @ np.linalg.inv(al**2 * cov + de * np.eye(N_SITES))).T
                pairs.append(e03._sq(m, m_ref, X, t, e03.REGIONS["all"]))
            info = {}
        elif arm in KERNEL_ARMS:
            model, mu, grid, w, info = fit_kernel(arm, cfg, A, cell, seed, val)
            pairs = _pairs(model.log_kernels(), mu, grid, w, test)
        else:
            net = cfg["nets"][arm]
            den, info = e03._train_select(cfg, arm, net["hp"], net["lr"], net["mode"], A, val, seed, 0)
            pairs = e03._eval(den, test, "all")
    except FloatingPointError as exc:       # E33 amendment 3: a failed fit is a row, and it counts against
        print(f"  [E68] {cell} seed={seed} n={n:>5} {arm:15s} FAILED: {exc}", flush=True)
        return [{**base, "arm": arm, "t": t, "failed": 1, "stop_reason": f"FloatingPointError: {exc}",
                 "risk_pooled": np.nan} for t in T_SCHEDULE]
    rows = _rows(base, arm, pairs, info)
    print(f"  [E68] {cell} seed={seed} n={n:>5} {arm:15s} pooled={rows[0]['risk_pooled']:.4f}", flush=True)
    return rows


# ---------------------------------------------------------------------------------------- parts

def part_timing(cfg, out, workers):
    import torch
    from markov_score import nn_torch as nt
    torch.set_num_threads(int(cfg["torch_threads"]))
    cell, seed, n = "drift10", cfg["dev_seeds"][0], cfg["timing_n"]
    A = chain_of(cfg, cell).sample(rng_for("e68-timing", seed), n)
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    X, al, de = training_views(A, cell, seed)
    rows = []
    for arm in KERNEL_ARMS:
        model, mu = KernelArm(arm, cfg, grid, ("timing", seed)), (0.0, 1.0)
        t_e = t_m = 0.0
        for _ in range(cfg["timing_iters"]):
            t0 = time.perf_counter()
            r = site_bp_batch(grid, w, model.log_kernels(), X, al, de, log_gauss(grid, *mu), pairs=True)
            t1 = time.perf_counter()
            model = model.step(r.xi, n)
            t_e += t1 - t0; t_m += time.perf_counter() - t1
        rows.append({"what": arm, "n": n, "e_step_s": t_e / cfg["timing_iters"], "m_step_s": t_m / cfg["timing_iters"],
                     "per_iter_s": (t_e + t_m) / cfg["timing_iters"], "ladder_cap": max(cfg["ladders"][arm])})
        print(f"  [E68 timing] {arm:15s} E-step {rows[-1]['e_step_s']:.2f} s  M-step {rows[-1]['m_step_s']:.2f} s", flush=True)
    for arm in NET_ARMS:
        net = cfg["nets"][arm]
        model = nt.build(arm, N_SITES, **net["hp"])
        tcfg = nt.TrainConfig(steps=cfg["timing_net_steps"], batch_size=cfg["batch_size"], lr=net["lr"],
                              grad_clip=cfg["grad_clip"], parameterization=net["mode"],
                              checkpoints=(cfg["timing_net_steps"],), t_levels=T_SCHEDULE, seed=0)
        t0 = time.perf_counter()
        nt.train(model, A, tcfg)
        rows.append({"what": arm, "n": n, "per_1000_steps_s": 1000 * (time.perf_counter() - t0) / cfg["timing_net_steps"],
                     "n_params": nt.n_params(model)})
        print(f"  [E68 timing] {arm:15s} {rows[-1]['per_1000_steps_s']:.1f} s per 1000 steps", flush=True)
    write_csv(Path(out) / "timing.csv", rows)


def part_bundles(cfg, out, workers):
    cells = [(cfg, str(out), kind, c, s) for c in cfg["cells"] for s in cfg["seeds"] for kind in ("test", "val")
             if not _bundle_path(out, kind, c, s).exists()]
    print(f"  bundles: {len(cells)} to build", flush=True)
    write_csv(Path(out) / "bundles.csv", run_cells(cells, _bundle_cell, workers, out, "bundles.csv",
                                                   key=lambda c: f"{c[2]}_{c[3]}_s{c[4]}"))


def part_arms(cfg, out, workers):
    cells = [(cfg, str(out), c, s, n, a) for n in sorted(cfg["sizes"], reverse=True) for a in cfg["arms"]
             for c in cfg["cells"] for s in cfg["seeds"]]
    print(f"  arms: {len(cells)} cells", flush=True)
    write_csv(Path(out) / "arms.csv", run_cells(cells, _arm_cell, workers, out, "arms.csv",
                                                key=lambda c: f"{c[2]}_s{c[3]}_n{c[4]}_{c[5]}"))


def main() -> None:
    parser = experiment_parser("e68_nonstationary", "non-stationary chains: shared, per-link and site-conditioned kernels")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    w = args.workers
    parts = {"timing": ("seconds per EM iteration and per network step, one dev seed", lambda c, o: part_timing(c, o, w)),
             "bundles": ("validation and test bundles with the per-link BP reference", lambda c, o: part_bundles(c, o, w)),
             "arms": ("the arms on the non-stationary chains", lambda c, o: part_arms(c, o, w))}
    if args.list_parts:
        print("\n".join(parts)); return
    selected = select_parts(parts, args.only)
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
