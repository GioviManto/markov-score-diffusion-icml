#!/usr/bin/env python
"""E57 — kernel families on chains inside and outside the shift class, against four networks.

X2 of the 29 Oct plan. Three chains matched to the Laplace chain in variance (1) and lag-one correlation (0.85), so a
linear method sees the same covariance everywhere and every difference comes from shape (calibration:
tools/core/calibrate_e57.py, from the true kernels only):

    bimodal   a' = 0.85 a + eps, eps an equal two-Gaussian mixture (GaussianMixtureAR1, kappa 0.6)   inside the class
    tanh      a' = 1.303746 tanh(a) + 0.497509 eps                                                  outside: nonlinear mean
    hetero    a' = 0.85 a + 0.471169 sqrt(1 + 0.25 a^2) eps                                         outside: a-dependent scale

Arms. Kernels, by EM through exact grid BP from one noisy copy per chain, selected on the validation BP risk (E56's code
and settings): the four shift families, loglinear2d, energy (width = E56's screen winner) and the C = 8 mixture; the
linear floor (the LMMSE of each chain's exact covariance) as a reference. Networks: global MLP, window CNN, BiGRU and
U-Net with E03's recipe and E03's screened configurations (SETTINGS["nets"], equal to outputs/E03/winners.json, checked
by tests/test_e57_driver.py), trained on the clean chains, selected on the validation bundle.

Parts
-----
timing    dev seed, N = timing_n: seconds per EM iteration for every kernel arm and per training step for every net
kernels   chains x seeds x sizes x kernel arms (+ the linear floor) -> kernels.csv, one row per (chain, seed, n, arm, t)
nets      chains x seeds x sizes x networks -> nets.csv, the same columns where they apply

PREDECLARED in experiments/registry/E57.json (P1-P4, D1-D4) before this file was written.
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
from protocol import PROTOCOL  # noqa: E402
from markov_score.bp_grid import make_grid  # noqa: E402
from markov_score.em import fit_em  # noqa: E402
from markov_score.kernels import MixtureInnovationKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import GaussianMixtureAR1, HeteroscedasticAR1, TanhDriftChain  # noqa: E402
from markov_score.stationary import invariant_log_density_eig  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

# E03 for the bundles, the estimand and the network recipe; E56 for the kernel arms and their EM. Imported, not copied.
import e03_baselines as e03  # noqa: E402
import e56_kernel_families as e56  # noqa: E402

N_SITES = FROZEN.n_sites
T_SCHEDULE = tuple(FROZEN.t_grid)
CHAINS = ("bimodal", "tanh", "hetero")
KERNEL_ARMS = e56.SHIFT_ARMS + ("loglinear2d", "energy", "mixture")
NETS = ("mlp", "window", "bigru", "unet")

SETTINGS = dict(
    e56.SETTINGS,
    chains=CHAINS, seeds=tuple(range(8)), dev_seeds=(10_000,), sizes=(32, 128, 512, 2048),
    kernel_arms=KERNEL_ARMS, net_archs=NETS,
    energy_hidden=0,                 # 0: read `energy_winners` (E56's screen)
    energy_winners="",
    # E03's screened winners (outputs/E03/winners.json), written out so a cluster job does not need E03's outputs
    nets={"mlp": {"hp": {"hidden": [128, 128]}, "lr": 1e-3, "mode": "eps"},
          "window": {"hp": {"radius": 4, "width": 64}, "lr": 1e-3, "mode": "eps"},
          "bigru": {"hp": {"hidden": 32}, "lr": 1e-3, "mode": "eps"},
          "unet": {"hp": {"channels": 16}, "lr": 1e-3, "mode": "eps"}},
    net_steps=PROTOCOL.net_steps, net_checkpoints=PROTOCOL.net_checkpoints, batch_size=PROTOCOL.batch_size,
    grad_clip=PROTOCOL.grad_clip, timing_net_steps=200,
)
QUICK = dict(e56.QUICK, chains=CHAINS, seeds=(0,), sizes=(32,), energy_hidden=8, net_checkpoints=(100, 200),
             timing_net_steps=50)


def chain_prior(chain: str):
    if chain == "bimodal":
        return GaussianMixtureAR1(rho=0.85, kappa=0.6)
    if chain == "tanh":
        return TanhDriftChain()
    if chain == "hetero":
        return HeteroscedasticAR1(rho=0.85, c=0.25)
    raise ValueError(f"unknown chain {chain!r}")


# ---------------------------------------------------------------------------------------- data

def _cell_common(cfg, chain, seed, n):
    """E03's cell, for one of E57's chains, under E57's own tags."""
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    prior = chain_prior(chain)
    log_k = prior.log_transition_matrix(grid)
    rng_train = rng_for(f"e57-{chain}-train", seed, n)
    A = np.stack([prior.sample(rng_train, N_SITES) for _ in range(n)])
    _, val = e03._bundle(prior, log_k, grid, weights, f"e57-{chain}-val", seed, cfg["n_val"])
    _, test = e03._bundle(prior, log_k, grid, weights, f"e57-{chain}-test", seed, cfg["n_test"])
    base = dict(chain=chain, seed=seed, n_chains=n, val_bundle=e03._bundle_hash(val), test_bundle=e03._bundle_hash(test))
    return A, val, test, base, grid, weights, prior, log_k


def _one_view_groups(chain, A, seed):
    """EM-BP's interface, as E03's: one corruption per chain, chains split across the twelve levels; E57's tags."""
    idx = np.array_split(rng_for(f"e57-{chain}-em-split", seed).permutation(len(A)), len(T_SCHEDULE))
    noise = rng_for(f"e57-{chain}-em-noise", seed)
    out = []
    for t, ix in zip(T_SCHEDULE, idx):
        al, de = alpha_delta(t)
        sub = A[ix]
        out.append((al * sub + np.sqrt(de) * noise.standard_normal(sub.shape), *alpha_delta(t)))
    return out


def chain_covariance(log_k, grid, weights, n_sites=N_SITES):
    """The exact covariance of the chain on the grid, started from a_1 ~ N(0, 1) as every prior samples it."""
    k = np.exp(log_k - log_k.max(0, keepdims=True))
    k /= (k * weights[:, None]).sum(0, keepdims=True)
    p = k * weights[:, None]                                  # column-stochastic on the grid's probability vectors
    pi = [np.exp(-0.5 * grid**2) * weights]
    pi[0] = pi[0] / pi[0].sum()
    for _ in range(n_sites - 1):
        pi.append(p @ pi[-1])
    mean = np.array([q @ grid for q in pi])
    cov = np.empty((n_sites, n_sites))
    for i in range(n_sites):
        h = grid * pi[i]                                     # E[a_i 1{a_j = g}] propagated forward
        for j in range(i, n_sites):
            cov[i, j] = cov[j, i] = h @ grid - mean[i] * mean[j]
            h = p @ h
    return mean, cov


def _lmmse_pairs(cov, mean, bundle):
    """The best linear estimator of a from x at each level, scored with E03's estimand."""
    out = []
    for t in T_SCHEDULE:
        X, m_ref = bundle[t]
        al, de = alpha_delta(t)
        gain = al * cov @ np.linalg.inv(al**2 * cov + de * np.eye(len(cov)))
        m = mean + (X - al * mean) @ gain.T
        out.append(e03._sq(m, m_ref, X, t, e03.REGIONS["all"]))
    return out


def kernel_hellinger(log_k_hat, log_k_true, grid, weights, prior) -> float:
    """E56's kernel error with the chain's own truth: sum_j pi_j w_j H(K_hat(.|g_j), K(.|g_j)) / sum_j pi_j w_j."""
    lh, lt = e56._normalised(log_k_hat, weights), e56._normalised(log_k_true, weights)
    bc = (np.exp(0.5 * (lh + lt)) * weights[:, None]).sum(axis=0)
    h = np.sqrt(np.clip(1.0 - bc, 0.0, None))
    pi = np.exp(invariant_log_density_eig(prior, grid, weights).log_density)
    return float((pi * weights * h).sum() / (pi * weights).sum())


# ---------------------------------------------------------------------------------------- arms

def _hidden(cfg):
    if cfg["energy_hidden"]:
        return int(cfg["energy_hidden"])
    p = Path(cfg["energy_winners"]) if cfg["energy_winners"] else None
    if p is None or not p.is_file():
        raise SystemExit("the energy arm needs E56's width: --set energy_winners='<E56_dev/winners.json>' "
                         "or --set energy_hidden=<w>")
    return int(json.loads(p.read_text())["energy_hidden"])


def _init(cfg, chain, arm, seed, n, hidden):
    if arm == "mixture":
        return MixtureInnovationKernel.init(8, rho=cfg["em_init_rho"], var=cfg["em_init_var"],
                                            rng=rng_for("e57-mixture-init", chain, seed, n))
    return e56._init_kernel(cfg, arm, seed, n, hidden)


def _kernel_cell(args):
    cfg, chain, seed, n, arm, hidden = args
    import torch
    torch.set_num_threads(int(cfg["torch_threads"]))
    A, val, test, base, grid, weights, prior, log_k = _cell_common(cfg, chain, seed, n)
    row0 = {**base, "arm": arm}
    if arm == "lmmse":
        mean, cov = chain_covariance(log_k, grid, weights)
        pairs = _lmmse_pairs(cov, mean, test)
        pooled = e03._risk(pairs)
        return [{**row0, "t": t, "sq_err": e, "sq_ref": r, "risk_level": float(np.sqrt(e / r)), "risk_pooled": pooled,
                 "failed": 0} for t, (e, r) in zip(T_SCHEDULE, pairs)]
    groups = _one_view_groups(chain, A, seed)
    k0 = _init(cfg, chain, arm, seed, n, hidden)
    hp = json.dumps({"hidden": int(hidden)}) if arm == "energy" else (e56._hp(arm, cfg, 0) if arm != "mixture"
                                                                       else json.dumps({"C": 8}))
    try:
        k, info = e56._fit_and_select(cfg, k0, groups, grid, weights, val)
    except FloatingPointError as exc:       # E33 amendment 3: a failed fit is a row, and it counts against
        print(f"  [E57] {chain} seed={seed} n={n:>5} {arm:11s} FAILED: {exc}", flush=True)
        return [{**row0, "hp": hp, "t": t, "failed": 1, "stop_reason": f"FloatingPointError: {exc}",
                 "risk_pooled": np.nan} for t in T_SCHEDULE]
    pairs = e56._pairs(k, test, grid, weights)
    pooled = e03._risk(pairs)
    kh = kernel_hellinger(k.log_transition_matrix(grid), log_k, grid, weights, prior)
    theta = ";".join(f"{v:.10g}" for v in np.asarray(k.theta, dtype=float))
    print(f"  [E57] {chain} seed={seed} n={n:>5} {arm:11s} pooled={pooled:.4f} ck={info['checkpoint']} H={kh:.4f}",
          flush=True)
    return [{**row0, "hp": hp, "t": t, "sq_err": e, "sq_ref": r, "risk_level": float(np.sqrt(e / r)) if r > 0 else np.nan,
             "risk_pooled": pooled, **info, "kernel_hellinger": kh, "theta": theta, "failed": 0}
            for t, (e, r) in zip(T_SCHEDULE, pairs)]


def _net_cell(args):
    cfg, chain, seed, n, arch = args
    A, val, test, base, *_ = _cell_common(cfg, chain, seed, n)
    w = cfg["nets"][arch]
    den, info = e03._train_select(cfg, arch, w["hp"], w["lr"], w["mode"], A, val, seed, 0)
    pairs = e03._eval(den, test, "all")
    pooled = e03._risk(pairs)
    print(f"  [E57 net] {chain} seed={seed} n={n:>5} {arch:6s} pooled={pooled:.4f} ck={info['checkpoint']}", flush=True)
    return [{**base, "arm": arch, "hp": json.dumps(w["hp"]), "lr": w["lr"], "mode": w["mode"], "t": t, "sq_err": e,
             "sq_ref": r, "risk_level": float(np.sqrt(e / r)) if r > 0 else np.nan, "risk_pooled": pooled, **info,
             "failed": 0} for t, (e, r) in zip(T_SCHEDULE, pairs)]


def _kernel_key(cell):
    _, chain, seed, n, arm, _ = cell
    return f"{chain}_s{seed}_n{n}_{arm}"


def _net_key(cell):
    _, chain, seed, n, arch = cell
    return f"{chain}_s{seed}_n{n}_{arch}"


# ---------------------------------------------------------------------------------------- parts

def part_timing(cfg, out, workers):
    rows = []
    seed, n = cfg["dev_seeds"][0], cfg["timing_n"]
    hidden = _hidden(cfg) if "energy" in cfg["kernel_arms"] else 0
    for chain in cfg["chains"]:
        A, val, _, _, grid, weights, _, _ = _cell_common(cfg, chain, seed, n)
        groups = _one_view_groups(chain, A, seed)
        for arm in cfg["kernel_arms"]:
            _, trace = fit_em(_init(cfg, chain, arm, seed, n, hidden), grid, weights, groups,
                              n_iters=cfg["timing_iters"], tol=0.0)
            sec = float(np.median(trace.seconds))
            rows.append({"chain": chain, "arm": arm, "n_chains": n, "sec": sec, "unit": "EM iteration",
                         "sec_full_budget": sec * max(cfg["ladder"])})
            print(f"  [timing] {chain:8s} {arm:11s} {sec:.3f} s per EM iteration", flush=True)
        for arch in cfg["net_archs"]:
            c = dict(cfg, net_checkpoints=(cfg["timing_net_steps"],))
            w = cfg["nets"][arch]
            t0 = time.perf_counter()
            e03._train_select(c, arch, w["hp"], w["lr"], w["mode"], A, val, seed, 0)
            sec = (time.perf_counter() - t0) / cfg["timing_net_steps"]
            rows.append({"chain": chain, "arm": arch, "n_chains": n, "sec": sec, "unit": "training step",
                         "sec_full_budget": sec * cfg["net_steps"]})
            print(f"  [timing] {chain:8s} {arch:11s} {sec * 1e3:.2f} ms per training step (validation included)", flush=True)
    write_csv(Path(out) / "timing.csv", rows)


def part_kernels(cfg, out, workers):
    hidden = _hidden(cfg) if "energy" in cfg["kernel_arms"] else 0
    cells = [(cfg, c, s, n, a, hidden if a == "energy" else 0)
             for c in cfg["chains"] for s in cfg["seeds"] for n in cfg["sizes"] for a in ("lmmse",) + tuple(cfg["kernel_arms"])]
    write_csv(Path(out) / "kernels.csv", run_cells(cells, _kernel_cell, workers, out, "kernels.csv", key=_kernel_key))


def part_nets(cfg, out, workers):
    cells = [(cfg, c, s, n, a) for c in cfg["chains"] for s in cfg["seeds"] for n in cfg["sizes"] for a in cfg["net_archs"]]
    write_csv(Path(out) / "nets.csv", run_cells(cells, _net_cell, workers, out, "nets.csv", key=_net_key))


def main() -> None:
    parser = experiment_parser("e57_chain_families", "kernel families on chains inside and outside the shift class")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    w = args.workers
    parts = {"timing": ("seconds per EM iteration and per training step", lambda c, o: part_timing(c, o, w)),
             "kernels": ("kernel arms and the linear floor", lambda c, o: part_kernels(c, o, w)),
             "nets": ("the four networks", lambda c, o: part_nets(c, o, w))}
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
