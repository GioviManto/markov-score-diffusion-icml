#!/usr/bin/env python
"""E62 — how long first-order EM-BP stays good when the data are second order, and the pair-state repair.

Jerome's "how long does it stay good?". The truth is a Laplace AR(2) at E03's lag-one correlation and variance,

    a_t = phi1 a_{t-1} + phi2 a_{t-2} + eps,   phi1 = 0.85 (1 - phi2),   Var eps = (1 - 0.85^2)(1 - phi2^2),

for phi2 in (-0.3, 0, 0.1, 0.2, 0.3, 0.4, 0.6), started from its stationary law; misspecification is measured by delta,
the KL rate to the best first-order chain (tools/core/calibrate_e62.py). The reference is exact pair-state BP
(markov_score.pair_bp) with the true kernel on 151 points.

Arms (one noisy view per training chain for the two kernels, the clean chains for the networks):
    first   the thesis mixture (C = 8) by EM through first-order grid BP (401 points), its own invariant law as initial
            law; the checkpoint of E33's ladder with the lowest pooled validation risk
    pair    the two-lag mixture (markov_score.ar2_kernel, C = 8) by EM through pair-state BP (101 points), its own
            stationary pair law as initial law; the ladder to 300, the same selection
    mlp, window   E03's screened winners (DSM on the clean chains)
    lmmse   the LMMSE of the true AR(2) covariance
Wind (P5): E59's Horns Rev windows, both kernels by clean-data maximum likelihood, the networks as above; scored against
the clean value (E03's estimand with the clean value in place of the posterior mean: the score error plus a term common
to every method, so paired comparisons rank methods as the score error does).

Parts
-----
bundles   per (phi2, seed): the validation and test bundles with their pair-state reference, cached under bundles/
arms      phi2 x seeds x sizes x arms -> arms.csv, one row per (phi2, seed, n, arm, t)
wind      seeds x wind sizes x arms -> wind.csv

PREDECLARED in experiments/registry/E62.json (P1-P5, D1-D5) before this file was written.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e03_baselines as e03  # noqa: E402
import e33_field_learning as e33  # noqa: E402
import e57_chain_families as e57  # noqa: E402
import e59_forecasting as e59  # noqa: E402
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score.ar2_kernel import Ar2MixtureKernel, clean_triples, fit_pair_em  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.em import clean_statistics, fit_em  # noqa: E402
from markov_score.kernels import MixtureInnovationKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.pair_bp import pair_bp_batch, stationary_pair  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

N_SITES, T_SCHEDULE = FROZEN.n_sites, tuple(FROZEN.t_grid)
RHO1 = 0.85
PHI2 = (-0.3, 0.0, 0.1, 0.2, 0.3, 0.4, 0.6)
ARMS = ("pair", "first", "window", "mlp", "lmmse")                  # slowest first, for the worker pool's tail
WIND_ARMS = ("pair", "first", "window", "mlp")

SETTINGS = dict(
    e57.SETTINGS, phi2=PHI2, seeds=tuple(range(8)), sizes=(128, 512, 2048), arms=ARMS,
    grid_size=401, half_width=8.0, pair_grid=101, ref_grid=151, components=8, burn=600,
    first_ladder=tuple(e33.SETTINGS["em_checkpoints"]), pair_ladder=(10, 20, 40, 80, 120, 160, 220, 300),
    wind_sizes=(32, 128, 512, 1917), wind_arms=WIND_ARMS, wind_test_stride=e59.SETTINGS["wind_test_stride"],
    nets={a: e57.SETTINGS["nets"][a] for a in ("mlp", "window")},
)
QUICK = dict(e57.QUICK, phi2=(0.0, 0.4), seeds=(0,), sizes=(32,), grid_size=101, pair_grid=41, ref_grid=61,
             first_ladder=(2, 4), pair_ladder=(2, 4), wind_sizes=(32,), n_val=32, n_test=32, burn=200)


# ---------------------------------------------------------------------------------------- the chain

def coefficients(phi2):
    return RHO1 * (1 - phi2), (1 - RHO1 ** 2) * (1 - phi2 ** 2)


def true_k2(grid, w, phi2):
    """log K2[k, j, i] of the Laplace AR(2), normalised over k on the grid."""
    phi1, s2 = coefficients(phi2)
    b = np.sqrt(s2 / 2)
    lk = -np.abs(grid[:, None, None] - phi1 * grid[None, :, None] - phi2 * grid[None, None, :]) / b
    return lk - np.log(np.einsum("kji,k->ji", np.exp(lk), w))[None]


def sample(phi2, rng, n, burn):
    phi1, s2 = coefficients(phi2)
    a = np.zeros((n, burn + N_SITES))
    e = rng.laplace(0.0, np.sqrt(s2 / 2), size=a.shape)
    for t in range(2, a.shape[1]):
        a[:, t] = phi1 * a[:, t - 1] + phi2 * a[:, t - 2] + e[:, t]
    return a[:, burn:]


def ar2_cov(phi2):
    phi1, _ = coefficients(phi2)
    rho = [1.0, RHO1]
    for _ in range(2, N_SITES):
        rho.append(phi1 * rho[-1] + phi2 * rho[-2])
    i = np.arange(N_SITES)
    return np.array(rho)[np.abs(i[:, None] - i[None, :])]


def _pair_means(grid, w, lk2, mu2, X, t):
    al, de = alpha_delta(t)
    return pair_bp_batch(grid, w, lk2, X, al, de, np.log(np.maximum(mu2, 1e-300)))[0]


# ---------------------------------------------------------------------------------------- bundles (cached)

def _bundle_path(out, kind, phi2, seed):
    return Path(out) / "bundles" / f"{kind}_phi{phi2:+.2f}_s{seed}.npz"


def make_bundle(cfg, kind, phi2, seed):
    grid, w = make_grid(cfg["half_width"], cfg["ref_grid"])
    lk2 = true_k2(grid, w, phi2)
    mu2 = stationary_pair(lk2, w)
    rng = rng_for(f"e62-{kind}", phi2, seed)
    A = sample(phi2, rng, cfg["n_val"] if kind == "val" else cfg["n_test"], cfg["burn"])
    out = {}
    for t in T_SCHEDULE:
        al, de = alpha_delta(t)
        X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
        out[t] = (X, _pair_means(grid, w, lk2, mu2, X, t))
    return out


def load_bundle(out, kind, phi2, seed):
    z = np.load(_bundle_path(out, kind, phi2, seed))
    return {t: (z[f"X_{i}"], z[f"m_{i}"]) for i, t in enumerate(T_SCHEDULE)}


def _bundle_cell(args):
    cfg, out, kind, phi2, seed = args
    p = _bundle_path(out, kind, phi2, seed)
    t0 = time.perf_counter()
    b = make_bundle(cfg, kind, phi2, seed)
    ensure_dir(p.parent)
    np.savez(p.with_suffix(".tmp.npz"), **{f"X_{i}": b[t][0] for i, t in enumerate(T_SCHEDULE)},
             **{f"m_{i}": b[t][1] for i, t in enumerate(T_SCHEDULE)})
    p.with_suffix(".tmp.npz").replace(p)
    print(f"  [E62 bundle] {kind} phi2={phi2:+.2f} seed={seed} {time.perf_counter() - t0:.0f} s", flush=True)
    return [{"kind": kind, "phi2": phi2, "seed": seed, "bundle": e03._bundle_hash(b), "seconds": time.perf_counter() - t0}]


# ---------------------------------------------------------------------------------------- arms

def _groups(A, tag, phi2, seed):
    idx = np.array_split(rng_for(f"e62-{tag}-em-split", phi2, seed).permutation(len(A)), len(T_SCHEDULE))
    noise = rng_for(f"e62-{tag}-em-noise", phi2, seed)
    out = []
    for t, ix in zip(T_SCHEDULE, idx):
        al, de = alpha_delta(t)
        out.append((al * A[ix] + np.sqrt(de) * noise.standard_normal(A[ix].shape), al, de))
    return out


def _first_pairs(kernel, grid, w, mu_fn, bundle):
    lk, mu = kernel.log_transition_matrix(grid), mu_fn(kernel)
    return [e03._sq(grid_bp_batch(grid, w, lk, X, *alpha_delta(t), log_mu=mu)[0], m_ref, X, t, e03.REGIONS["all"])
            for t, (X, m_ref) in bundle.items()]


def _pair_pairs(kernel, grid, w, bundle):
    lk2 = kernel.log_k2(grid, w)
    mu2 = stationary_pair(lk2, w)
    return [e03._sq(_pair_means(grid, w, lk2, mu2, X, t), m_ref, X, t, e03.REGIONS["all"]) for t, (X, m_ref) in bundle.items()]


def fit_first(cfg, A_or_stats, grid, w, val, seed_key, clean=False):
    """The thesis mixture, by EM through first-order BP (or by clean ML), selected on the validation risk."""
    mu_fn = e33._stationary_mu(grid, w)
    k0 = MixtureInnovationKernel.init(cfg["components"], rho=cfg["em_init_rho"], var=cfg["em_init_var"],
                                      rng=rng_for("e62-first-init", *seed_key))
    ladder = set(cfg["first_ladder"])
    t0 = time.perf_counter()
    if clean:
        saved, cur = {}, k0
        for it in range(max(ladder) + 1):
            if it in ladder:
                saved[it] = cur
            if it < max(ladder):
                cur = cur.m_step(A_or_stats, grid)
    else:
        _, _, saved = fit_em(k0, grid, w, A_or_stats, n_iters=max(ladder), tol=FROZEN.em_loglik_tol, checkpoints=ladder,
                             log_mu=mu_fn)
    wall = time.perf_counter() - t0
    curve = {it: e03._risk(_first_pairs(saved[it], grid, w, mu_fn, val)) for it in sorted(saved)}
    best = min(curve, key=curve.get)
    return saved[best], mu_fn, {"checkpoint": best, "at_cap": int(best == max(saved)), "wall": wall,
                                "val_risk": curve[best], "n_params": int(np.asarray(saved[best].theta).size),
                                "fit_rho": float(saved[best].rho)}


def fit_pair(cfg, groups_or_xi, grid, w, val, seed_key, clean=False):
    """The two-lag mixture, by EM through pair-state BP (or by clean ML), selected on the validation risk."""
    k0 = Ar2MixtureKernel.init(cfg["components"], rho=cfg["em_init_rho"], var=cfg["em_init_var"],
                               rng=rng_for("e62-pair-init", *seed_key))
    ladder = set(cfg["pair_ladder"])
    t0 = time.perf_counter()
    if clean:
        saved, cur = {}, k0
        for it in range(max(ladder) + 1):
            if it in ladder:
                saved[it] = cur
            if it < max(ladder):
                cur = cur.m_step(groups_or_xi, grid, w)
        sec = (time.perf_counter() - t0) / max(ladder)
    else:
        saved, sec = fit_pair_em(k0, grid, w, groups_or_xi, ladder)
    wall = time.perf_counter() - t0
    curve = {it: e03._risk(_pair_pairs(saved[it], grid, w, val)) for it in sorted(saved)}
    best = min(curve, key=curve.get)
    k = saved[best]
    return k, {"checkpoint": best, "at_cap": int(best == max(saved)), "wall": wall, "sec_per_iter": sec,
               "val_risk": curve[best], "n_params": int(k.theta.size), "fit_phi1": float(k.phi[0]),
               "fit_phi2": float(k.phi[1])}


def _rows(base, arm, pairs, info):
    pooled = e03._risk(pairs)
    return [{**base, "arm": arm, "t": t, "sq_err": e, "sq_ref": r, "risk_level": float(np.sqrt(e / r)) if r > 0 else np.nan,
             "risk_pooled": pooled, "failed": 0, **info} for t, (e, r) in zip(T_SCHEDULE, pairs)]


def _arm_cell(args):
    cfg, out, phi2, seed, n, arm = args
    import torch
    torch.set_num_threads(int(cfg["torch_threads"]))
    val, test = load_bundle(out, "val", phi2, seed), load_bundle(out, "test", phi2, seed)
    base = {"phi2": phi2, "seed": seed, "n_chains": n, "val_bundle": e03._bundle_hash(val),
            "test_bundle": e03._bundle_hash(test)}
    A = sample(phi2, rng_for("e62-train", phi2, seed, n), n, cfg["burn"])
    try:
        if arm == "lmmse":
            cov = ar2_cov(phi2)
            pairs = []
            for t, (X, m_ref) in test.items():
                al, de = alpha_delta(t)
                m = X @ (al * cov @ np.linalg.inv(al ** 2 * cov + de * np.eye(N_SITES))).T
                pairs.append(e03._sq(m, m_ref, X, t, e03.REGIONS["all"]))
            info = {}
        elif arm == "first":
            grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
            k, mu_fn, info = fit_first(cfg, _groups(A, "train", phi2, seed), grid, w, val, (phi2, seed, n))
            pairs = _first_pairs(k, grid, w, mu_fn, test)
            info["theta"] = ";".join(f"{v:.10g}" for v in k.theta)
        elif arm == "pair":
            grid, w = make_grid(cfg["half_width"], cfg["pair_grid"])
            k, info = fit_pair(cfg, _groups(A, "train", phi2, seed), grid, w, val, (phi2, seed, n))
            pairs = _pair_pairs(k, grid, w, test)
            info["theta"] = ";".join(f"{v:.10g}" for v in k.theta)
        else:
            net = cfg["nets"][arm]
            den, info = e03._train_select(cfg, arm, net["hp"], net["lr"], net["mode"], A, val, seed, 0)
            pairs = e03._eval(den, test, "all")
    except FloatingPointError as exc:       # E33 amendment 3: a failed fit is a row, and it counts against
        print(f"  [E62] phi2={phi2:+.2f} seed={seed} n={n:>5} {arm:6s} FAILED: {exc}", flush=True)
        return [{**base, "arm": arm, "t": t, "failed": 1, "stop_reason": f"FloatingPointError: {exc}",
                 "risk_pooled": np.nan} for t in T_SCHEDULE]
    rows = _rows(base, arm, pairs, info)
    print(f"  [E62] phi2={phi2:+.2f} seed={seed} n={n:>5} {arm:6s} pooled={rows[0]['risk_pooled']:.4f}", flush=True)
    return rows


# ---------------------------------------------------------------------------------------- wind (P5)

def wind_bundles(cfg, seed):
    """Noisy validation and test bundles of E59's windows, scored against the clean value."""
    _, V, T = e59.data(dict(e59.SETTINGS, wind_test_stride=cfg["wind_test_stride"]), "wind", seed, 32)
    out = {}
    for kind, C in (("val", V), ("test", T)):
        rng = rng_for(f"e62-wind-{kind}", seed)
        b = {}
        for t in T_SCHEDULE:
            al, de = alpha_delta(t)
            b[t] = (al * C + np.sqrt(de) * rng.standard_normal(C.shape), C)
        out[kind] = b
    return out["val"], out["test"]


def _wind_cell(args):
    cfg, seed, n, arm = args
    import torch
    torch.set_num_threads(int(cfg["torch_threads"]))
    train, _, _ = e59.data(dict(e59.SETTINGS, wind_test_stride=cfg["wind_test_stride"]), "wind", seed, n)
    val, test = wind_bundles(cfg, seed)
    base = {"seed": seed, "n_chains": n, "test_bundle": e03._bundle_hash(test)}
    if arm == "first":
        grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
        k, mu_fn, info = fit_first(cfg, clean_statistics(grid, train), grid, w, val, ("wind", seed, n), clean=True)
        pairs = _first_pairs(k, grid, w, mu_fn, test)
    elif arm == "pair":
        grid, w = make_grid(cfg["half_width"], cfg["pair_grid"])
        k, info = fit_pair(cfg, clean_triples(grid, train), grid, w, val, ("wind", seed, n), clean=True)
        pairs = _pair_pairs(k, grid, w, test)
    else:
        net = cfg["nets"][arm]
        den, info = e03._train_select(cfg, arm, net["hp"], net["lr"], net["mode"], train, val, seed, 0)
        pairs = e03._eval(den, test, "all")
    rows = _rows(base, arm, pairs, info)
    print(f"  [E62 wind] seed={seed} n={n:>5} {arm:6s} pooled={rows[0]['risk_pooled']:.4f}", flush=True)
    return rows


# ---------------------------------------------------------------------------------------- parts

def part_bundles(cfg, out, workers):
    cells = [(cfg, str(out), kind, p, s) for p in cfg["phi2"] for s in cfg["seeds"] for kind in ("test", "val")
             if not _bundle_path(out, kind, p, s).exists()]
    print(f"  bundles: {len(cells)} to build", flush=True)
    rows = run_cells(cells, _bundle_cell, workers, out, "bundles.csv", key=lambda c: f"{c[2]}_{c[3]:+.2f}_s{c[4]}")
    write_csv(Path(out) / "bundles.csv", rows)


def part_arms(cfg, out, workers):
    cells = [(cfg, str(out), p, s, n, a) for n in sorted(cfg["sizes"], reverse=True) for a in cfg["arms"]
             for p in cfg["phi2"] for s in cfg["seeds"]]
    print(f"  arms: {len(cells)} cells", flush=True)
    write_csv(Path(out) / "arms.csv", run_cells(cells, _arm_cell, workers, out, "arms.csv",
                                                key=lambda c: f"{c[2]:+.2f}_s{c[3]}_n{c[4]}_{c[5]}"))


def part_wind(cfg, out, workers):
    cells = [(cfg, s, n, a) for n in sorted(cfg["wind_sizes"], reverse=True) for a in cfg["wind_arms"] for s in cfg["seeds"]]
    write_csv(Path(out) / "wind.csv", run_cells(cells, _wind_cell, workers, out, "wind.csv",
                                                key=lambda c: f"s{c[1]}_n{c[2]}_{c[3]}"))


def main() -> None:
    parser = experiment_parser("e62_second_order", "first-order EM-BP against second-order data, and the pair-state repair")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    w = args.workers
    parts = {"bundles": ("validation and test bundles with the pair-state reference", lambda c, o: part_bundles(c, o, w)),
             "arms": ("the arms on the AR(2) chains", lambda c, o: part_arms(c, o, w)),
             "wind": ("the Horns Rev wind windows", lambda c, o: part_wind(c, o, w))}
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
