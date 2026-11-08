#!/usr/bin/env python
"""E67 — a vector state (d = 2): EM through BP on a 2-D state grid against two-channel networks.

The chain is markov_score.vector_chain.RotatedLaplaceVAR (b_{k+1} = 0.85 R(pi/6) b_k + Laplace noise, stationary
covariance I), L = 32, each component noised independently. BP runs on the 41 x 41 product grid on [-5, 5] (1681 states,
a 1681 x 1681 kernel), at eight log-spaced levels from t = 0.3 to 3 (the resolution rule sqrt(2t) >= 3h, h = 0.25).

Arms: lmmse (the true covariance); var1 (a Gaussian VAR(1) by EM, closed-form M-step); energy2d (a neural energy kernel
on (b, b') in R^4, width 16, uniform start, EM); window2 and bigru2 (two input channels, E03's recipe: eps-DSM on the
clean chains, batch 64, Adam, 32000 steps). EM learns from one noisy view per training sequence, split over the levels.
Every arm is selected by its validation score error (all sites, both components) and scored on the test bundle against
BP with the true kernel on the same grid.

Parts
-----
bundles     per seed: validation and test chains, their noisy views and the reference means -> bundles/s<seed>.npz
gridcheck   the reference on 41 against 61 points per axis, 256 test sequences of seed 0 -> gridcheck.csv
kernels     (seed, n, arm in lmmse, var1, energy2d) -> arms.csv
timing      P3: seconds per EM iteration at the largest N on a dev seed, with the job's BLAS threads -> timing.csv
nets        (seed, n, arm in window2, bigru2) -> nets.csv

PREDECLARED in experiments/registry/E67.json (P1-P3, D1-D2) before this file was written.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from protocol import PROTOCOL  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402
from markov_score.vector_chain import (GaussianVAR1, NeuralEnergyKernel2D, RotatedLaplaceVAR, bp2d, fit_em2d,  # noqa: E402
                                       state_grid)

import e03_baselines as e03  # noqa: E402   (_sq and _risk: the estimand, with both components summed)

L = 32
LEVELS = tuple(float(t) for t in np.geomspace(0.3, 3.0, 8))
SETTINGS = dict(
    seeds=tuple(range(8)), sizes=(128, 512, 2048), n_axis=41, n_axis_check=61, half_width=5.0, levels=LEVELS,
    n_val=PROTOCOL.n_val, n_test=PROTOCOL.n_test, n_check=256, chunk=128,
    kernel_arms=("lmmse", "var1", "energy2d"), net_arms=("window2", "bigru2"),
    ladder=(10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600), energy_hidden=16, energy_lr=1e-2, energy_inner=10,
    net_steps=PROTOCOL.net_steps, net_checkpoints=PROTOCOL.net_checkpoints, batch_size=PROTOCOL.batch_size,
    grad_clip=PROTOCOL.grad_clip, net_lr=1e-3, torch_threads=1, timing_iters=3,
)
QUICK = dict(seeds=(0,), sizes=(32,), n_axis=11, n_axis_check=15, n_val=16, n_test=16, n_check=8, levels=(0.5, 2.0),
             ladder=(2, 4), net_steps=20, net_checkpoints=(10, 20), timing_iters=2)
CHAIN = RotatedLaplaceVAR()


# ---------------------------------------------------------------------------------------- data

def _grid(cfg, n_axis=None):
    _, _, states, w2 = state_grid(cfg["half_width"], n_axis or cfg["n_axis"])
    return states, w2


def _views(B, levels, rng):
    """{t: X} one noisy view of every chain at every level."""
    out = {}
    for t in levels:
        al, de = alpha_delta(t)
        out[t] = al * B + math.sqrt(de) * rng.standard_normal(B.shape)
    return out


def build_bundles(cfg, seed, out):
    path = Path(out) / "bundles" / f"s{seed}.npz"
    if path.exists():
        return path
    states, w2 = _grid(cfg)
    lk = CHAIN.log_transition_matrix(states, w2)
    arrays = {}
    for name, n in (("val", cfg["n_val"]), ("test", cfg["n_test"])):
        rng = rng_for(f"e67-{name}", seed)
        B = CHAIN.sample(rng, n, L)
        arrays[f"{name}_B"] = B
        for i, (t, X) in enumerate(_views(B, cfg["levels"], rng).items()):
            al, de = alpha_delta(t)
            arrays[f"{name}_X{i}"] = X
            arrays[f"{name}_M{i}"] = bp2d(states, w2, lk, X, al, de, chunk=cfg["chunk"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.rename(path)
    return path


def load_bundles(cfg, seed, out):
    z = np.load(build_bundles(cfg, seed, out))
    def b(name):
        return {t: (z[f"{name}_X{i}"], z[f"{name}_M{i}"]) for i, t in enumerate(cfg["levels"])}
    return b("val"), b("test"), z["val_B"]


def train_chains(cfg, seed, n):
    return CHAIN.sample(rng_for("e67-train", seed, n), n, L)


def one_view_groups(cfg, A, seed):
    idx = np.array_split(rng_for("e67-em-split", seed).permutation(len(A)), len(cfg["levels"]))
    noise = rng_for("e67-em-noise", seed)
    out = []
    for t, ix in zip(cfg["levels"], idx):
        al, de = alpha_delta(t)
        out.append((al * A[ix] + math.sqrt(de) * noise.standard_normal(A[ix].shape), al, de))
    return out


def risk(mean_fn, bundle, levels):
    pairs = [e03._sq(mean_fn(X, t), m, X, t, slice(0, L)) for t, (X, m) in ((t, bundle[t]) for t in levels)]
    return e03._risk(pairs), pairs


# ---------------------------------------------------------------------------------------- kernel arms

def lmmse_mean_fn():
    C = CHAIN.covariance(L)

    def mean(X, t):
        al, de = alpha_delta(t)
        gain = al * C @ np.linalg.inv(al ** 2 * C + de * np.eye(2 * L))
        return (X.reshape(len(X), -1) @ gain.T).reshape(X.shape)
    return mean


def kernel_mean_fn(cfg, states, w2, lk):
    return lambda X, t: bp2d(states, w2, lk, X, *alpha_delta(t), chunk=cfg["chunk"])


def _kernel_cell(args):
    cfg, seed, n, arm, out = args
    import torch
    torch.set_num_threads(int(cfg["torch_threads"]))
    states, w2 = _grid(cfg)
    val, test, _ = load_bundles(cfg, seed, out)
    base = dict(seed=seed, n_chains=n, arm=arm, n_axis=cfg["n_axis"])
    t0 = time.perf_counter()
    if arm == "lmmse":
        r, _ = risk(lmmse_mean_fn(), test, cfg["levels"])
        return [{**base, "risk_pooled": r, "failed": 0}]
    groups = one_view_groups(cfg, train_chains(cfg, seed, n), seed)
    if arm == "var1":
        k0 = GaussianVAR1.init()
    else:
        k0 = NeuralEnergyKernel2D.init(hidden=cfg["energy_hidden"], half_width=cfg["half_width"], lr=cfg["energy_lr"],
                                       n_inner=cfg["energy_inner"],
                                       seed=int(rng_for("e67-energy-init", seed, n).integers(0, 2**31 - 1)))
    try:
        saved, ev, secs = fit_em2d(k0, states, w2, groups, set(cfg["ladder"]), chunk=cfg["chunk"])
        curve = {it: risk(kernel_mean_fn(cfg, states, w2, k.log_transition_matrix(states, w2)), val, cfg["levels"])[0]
                 for it, k in saved.items()}
        best = min(curve, key=lambda i: (curve[i], i))
        r, _ = risk(kernel_mean_fn(cfg, states, w2, saved[best].log_transition_matrix(states, w2)), test, cfg["levels"])
    except FloatingPointError as exc:                         # a failed fit is a row and counts against
        print(f"  [E67] seed={seed} n={n:>5} {arm} FAILED: {exc}", flush=True)
        return [{**base, "risk_pooled": np.nan, "failed": 1, "stop_reason": str(exc)}]
    sec = float(np.median(secs)) if secs else np.nan
    print(f"  [E67] seed={seed} n={n:>5} {arm:9s} test={r:.4f} ck={best} {sec:.1f} s/iter ({time.perf_counter() - t0:.0f} s)",
          flush=True)
    return [{**base, "risk_pooled": r, "checkpoint": best, "val_risk": curve[best], "sec_per_iter": sec,
             "n_params": int(np.asarray(saved[best].theta).size), "monotone": int(np.all(np.diff(ev) >= -1e-6 * np.abs(ev[1:]))),
             "val_curve": ";".join(f"{i}:{v:.6g}" for i, v in sorted(curve.items())),
             "theta": ";".join(f"{v:.10g}" for v in np.asarray(saved[best].theta, dtype=float)), "failed": 0}]


# ---------------------------------------------------------------------------------------- two-channel networks

def build_net(arch):
    import torch
    import torch.nn as nn
    from markov_score.nn_torch import TimeFeatures

    class Window2(nn.Module):
        """E03's window head with two input and two output channels (radius 4, width 64)."""

        def __init__(self, radius=4, width=64):
            super().__init__()
            self.tf = TimeFeatures(4)
            self.conv = nn.Conv1d(2, width, 2 * radius + 1, padding=radius)
            self.tproj = nn.Linear(self.tf.dim, width)
            self.mlp = nn.Sequential(nn.Tanh(), nn.Conv1d(width, width, 1), nn.Tanh(), nn.Conv1d(width, 2, 1))

        def forward(self, x, t):                               # x (B, L, 2)
            h = self.conv(x.transpose(1, 2)) + self.tproj(self.tf(t))[:, :, None]
            return self.mlp(h).transpose(1, 2)

    class BiGRU2(nn.Module):
        """E03's BiGRU with two input and two output channels (hidden 32)."""

        def __init__(self, hidden=32):
            super().__init__()
            self.tf = TimeFeatures(4)
            self.inp = nn.Linear(2 + self.tf.dim, hidden)
            self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.Linear(2 * hidden + 2, hidden), nn.Tanh(), nn.Linear(hidden, 2))

        def forward(self, x, t):
            B, Ls, _ = x.shape
            tfeat = self.tf(t)[:, None, :].expand(B, Ls, -1)
            h, _ = self.gru(torch.tanh(self.inp(torch.cat([x, tfeat], dim=2))))
            return self.head(torch.cat([h, x], dim=2))

    return {"window2": Window2, "bigru2": BiGRU2}[arch]()


def _net_cell(args):
    cfg, seed, n, arch, out = args
    import torch
    torch.set_num_threads(int(cfg["torch_threads"]))
    val, test, _ = load_bundles(cfg, seed, out)
    A = train_chains(cfg, seed, n)
    s = int(rng_for("e67-net-seed", arch, seed, n).integers(0, 2**31 - 1))
    torch.manual_seed(s)
    rng = np.random.default_rng(s)
    model = build_net(arch)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["net_lr"])
    ck = set(int(c) for c in cfg["net_checkpoints"])
    levels = np.asarray(cfg["levels"])
    saved, t0 = {}, time.perf_counter()
    for step in range(1, max(ck) + 1):
        idx = rng.integers(0, len(A), size=min(cfg["batch_size"], len(A)))
        a = torch.as_tensor(A[idx], dtype=torch.float32)
        t = torch.as_tensor(rng.choice(levels, size=len(idx)), dtype=torch.float32)
        al, de = torch.exp(-t)[:, None, None], 1.0 - torch.exp(-2.0 * t)[:, None, None]
        z = torch.as_tensor(rng.standard_normal(a.shape), dtype=torch.float32)
        loss = ((model(al * a + torch.sqrt(de) * z, t) - z) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        if step in ck:
            saved[step] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    sec = (time.perf_counter() - t0) / max(ck)

    def mean_fn(state):
        model.load_state_dict(state)
        model.eval()

        @torch.no_grad()
        def mean(X, t):
            x = torch.as_tensor(X, dtype=torch.float32)
            out = model(x, torch.full((len(X),), float(t)))
            al, de = math.exp(-t), 1.0 - math.exp(-2.0 * t)
            return ((x - math.sqrt(de) * out) / al).numpy().astype(np.float64)
        return mean

    curve = {st: risk(mean_fn(state), val, cfg["levels"])[0] for st, state in sorted(saved.items())}
    best = min(curve, key=lambda i: (curve[i], i))
    r, _ = risk(mean_fn(saved[best]), test, cfg["levels"])
    n_par = sum(p.numel() for p in model.parameters())
    print(f"  [E67 net] seed={seed} n={n:>5} {arch:8s} test={r:.4f} ck={best}", flush=True)
    return [dict(seed=seed, n_chains=n, arm=arch, risk_pooled=r, checkpoint=best, val_risk=curve[best], sec_per_step=sec,
                 n_params=int(n_par), val_curve=";".join(f"{i}:{v:.6g}" for i, v in sorted(curve.items())), failed=0)]


# ---------------------------------------------------------------------------------------- parts

def part_bundles(cfg, out, workers):
    for seed in cfg["seeds"]:
        t0 = time.perf_counter()
        build_bundles(cfg, seed, out)
        print(f"  [E67 bundles] seed={seed} ({time.perf_counter() - t0:.0f} s)", flush=True)


def part_gridcheck(cfg, out, workers):
    rng = rng_for("e67-gridcheck", 0)
    B = CHAIN.sample(rng, cfg["n_check"], L)
    rows = []
    s1, w1 = _grid(cfg)
    s2, w2 = _grid(cfg, cfg["n_axis_check"])
    lk1, lk2 = CHAIN.log_transition_matrix(s1, w1), CHAIN.log_transition_matrix(s2, w2)
    for t, X in _views(B, cfg["levels"], rng).items():
        al, de = alpha_delta(t)
        m1 = bp2d(s1, w1, lk1, X, al, de, chunk=cfg["chunk"])
        m2 = bp2d(s2, w2, lk2, X, al, de, chunk=cfg["chunk"])
        e, r = e03._sq(m1, m2, X, t, slice(0, L))
        rows.append(dict(t=t, n_axis=cfg["n_axis"], n_axis_check=cfg["n_axis_check"], sq_diff=e, sq_ref=r,
                         rel=float(np.sqrt(e / r))))
        print(f"  [E67 gridcheck] t={t:.3f}: {cfg['n_axis']} vs {cfg['n_axis_check']} per axis, relative {rows[-1]['rel']:.2e}", flush=True)
    pooled = float(np.sqrt(sum(r["sq_diff"] for r in rows) / sum(r["sq_ref"] for r in rows)))
    rows.append(dict(t=-1.0, n_axis=cfg["n_axis"], n_axis_check=cfg["n_axis_check"], rel=pooled))
    print(f"  [E67 gridcheck] pooled relative difference {pooled:.2e} (gate: < 1e-2)", flush=True)
    write_csv(Path(out) / "gridcheck.csv", rows)


def part_timing(cfg, out, workers):
    import os
    states, w2 = _grid(cfg)
    n, seed = max(cfg["sizes"]), 10_000
    groups = one_view_groups(cfg, train_chains(cfg, seed, n), seed)
    rows = []
    for arm in ("var1", "energy2d"):
        k0 = GaussianVAR1.init() if arm == "var1" else NeuralEnergyKernel2D.init(
            hidden=cfg["energy_hidden"], seed=0, half_width=cfg["half_width"], lr=cfg["energy_lr"], n_inner=cfg["energy_inner"])
        _, _, secs = fit_em2d(k0, states, w2, groups, {cfg["timing_iters"]}, chunk=cfg["chunk"])
        sec = float(np.median(secs[1:] if len(secs) > 1 else secs))
        rows.append(dict(arm=arm, n_chains=n, n_axis=cfg["n_axis"], blas_threads=os.environ.get("OMP_NUM_THREADS", ""),
                         sec_per_iter=sec, n_measured=len(secs)))
        print(f"  [E67 timing] {arm:9s} N={n}: {sec:.1f} s per EM iteration (OMP_NUM_THREADS={rows[-1]['blas_threads']})", flush=True)
    write_csv(Path(out) / "timing.csv", rows)


def part_kernels(cfg, out, workers):
    part_bundles(cfg, out, workers)
    cells = [(cfg, s, n, a, str(out)) for s in cfg["seeds"] for n in cfg["sizes"] for a in cfg["kernel_arms"]]
    write_csv(Path(out) / "arms.csv", run_cells(cells, _kernel_cell, workers, out, "arms.csv",
                                                key=lambda c: f"s{c[1]}_n{c[2]}_{c[3]}"))


def part_nets(cfg, out, workers):
    part_bundles(cfg, out, workers)
    cells = [(cfg, s, n, a, str(out)) for s in cfg["seeds"] for n in cfg["sizes"] for a in cfg["net_arms"]]
    write_csv(Path(out) / "nets.csv", run_cells(cells, _net_cell, workers, out, "nets.csv",
                                                key=lambda c: f"s{c[1]}_n{c[2]}_{c[3]}"))


def main() -> None:
    parser = experiment_parser("e67_vector_state", "a two-dimensional state: EM through 2-D BP against two-channel networks")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    w = args.workers
    parts = {"bundles": ("validation and test bundles with the reference", lambda c, o: part_bundles(c, o, w)),
             "gridcheck": ("the reference on two grids", lambda c, o: part_gridcheck(c, o, w)),
             "timing": ("seconds per EM iteration at the largest N", lambda c, o: part_timing(c, o, w)),
             "kernels": ("lmmse, var1 and energy2d", lambda c, o: part_kernels(c, o, w)),
             "nets": ("two-channel window CNN and BiGRU", lambda c, o: part_nets(c, o, w))}
    if args.list_parts:
        print("\n".join(parts)); return
    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir)
    tag = "_".join(selected) if args.only else "all"
    resolved = {"n_sites": L, "quick": args.quick, "parts": list(selected), "overrides": args.set, **cfg}
    write_json(out / f"params_{tag}.json", {**resolved, **provenance(resolved)})
    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(cfg, out)


if __name__ == "__main__":
    main()
