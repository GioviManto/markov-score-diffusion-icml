#!/usr/bin/env python
"""E73 — a hidden-regime Markov chain, and two real series (T-bill rate, Milan temperature).

regime   the noise of a first-order chain switches between a calm and a stormy regime that persists; BP on the doubled
         grid (value x regime) is exact. Arms: the regime kernel by EM on the doubled grid, the first-order mixture and
         neural kernel by EM (regime-blind), the global MLP and window CNN (DSM on clean chains), the linear denoiser.
real     DTB3 (3-month T-bill rate, standardised level) and Milan (daily mean temperature anomaly), windows of 32 days;
         kernels by clean maximum likelihood (first-order mixture, pair-state mixture, neural kernel) and the two
         networks, scored against the score of the clean value (no exact score: paired differences only).

PREDECLARED in experiments/registry/E73.json (P1-P7, D1-D2) before this file was written.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.ar2_kernel import clean_triples  # noqa: E402
from markov_score.em import clean_statistics, fit_em  # noqa: E402
from markov_score.energy_kernel import NeuralEnergyKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.regime_kernel import RegimeKernel, doubled, sample_regime_chain  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

import e03_baselines as e03  # noqa: E402
import e33_field_learning as e33  # noqa: E402
import e56_kernel_families as e56  # noqa: E402
import e58_clean_kernel as e58  # noqa: E402
import e62_second_order as e62  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
T_SCHEDULE = tuple(FROZEN.t_grid)
L = FROZEN.n_sites
S0 = float(np.sqrt(2 * (1 - 0.85 ** 2) / 10.0))                   # s1 = 3 s0 and (s0^2 + s1^2)/2 = 1 - 0.85^2
TRUE_REGIME = RegimeKernel(0.85, S0, 3 * S0, 0.95, 0.95)
REGIME_ARMS = ("regime", "energy", "first", "window", "mlp", "lmmse")
REAL_ARMS = ("pair", "energy", "first", "window", "mlp")

SETTINGS = {**e62.SETTINGS, "seeds": tuple(range(8)), "regime_sizes": (128, 512, 2048), "regime_arms": REGIME_ARMS,
            "real_series": ("dtb3", "milan"), "real_arms": REAL_ARMS, "real_sizes": {"dtb3": (32, 128, 398),
            "milan": (32, 128, 512, 913)}, "energy_hidden": 16, "energy_fresh": True, "ladder": e56.LADDER,
            "regime_ladder": (10, 20, 40, 80, 120, 200, 300, 400), "device": "cpu", "burn_regime": 300,
            "real_test_stride": 8}
QUICK = {"seeds": (0,), "regime_sizes": (32,), "real_sizes": {"dtb3": (32,), "milan": (32,)}, "n_val": 32, "n_test": 32,
         "grid_size": 101, "pair_grid": 41, "first_ladder": (2, 4), "pair_ladder": (2, 4), "ladder": (2, 4),
         "regime_ladder": (2, 4), "net_checkpoints": (10, 20), "net_steps": 20, "energy_hidden": 8,
         "regime_arms": ("regime", "first", "window", "lmmse"), "real_arms": ("first", "window")}


def _rows(base, arm, pairs, info):
    pooled = e03._risk(pairs)
    return [{**base, "arm": arm, "t": t, "sq_err": e, "sq_ref": r, "risk_level": float(np.sqrt(e / r)) if r > 0 else np.nan,
             "risk_pooled": pooled, "failed": 0, **info} for t, (e, r) in zip(T_SCHEDULE, pairs)]


def _net(cfg, arm, A, val, test, seed):
    net = cfg["nets"][arm]
    den, info = e03._train_select(cfg, arm, net["hp"], net["lr"], net["mode"], A, val, seed, 0)
    keep = {k: info[k] for k in ("checkpoint", "val_risk", "n_params") if k in info}
    return e03._eval(den, test, "all"), keep


def _energy_kernel(cfg, key):
    s = int(rng_for("e73-energy-init", *key).integers(0, 2**31 - 1))
    return NeuralEnergyKernel.init(hidden=int(cfg["energy_hidden"]), seed=s, half_width=cfg["half_width"],
                                   lr=cfg["energy_lr"], n_inner=cfg["energy_inner"], fresh_on_reject=bool(cfg["energy_fresh"]))


def _select(saved, score):
    curve = {it: score(k) for it, k in sorted(saved.items())}
    best = min(curve, key=curve.get)
    return saved[best], {"checkpoint": best, "at_cap": int(best == max(saved)), "val_risk": curve[best],
                         "n_params": int(np.asarray(saved[best].theta).size)}


# ---------------------------------------------------------------------------------------- hidden regime

def _regime_bundle(tag, seed, n, G, W, log_k, log_mu):
    A = sample_regime_chain(TRUE_REGIME, rng_for(f"e73-regime-{tag}", seed), n, L, burn=300)
    rng = rng_for(f"e73-regime-{tag}-noise", seed)
    out = {}
    for t in T_SCHEDULE:
        al, de = alpha_delta(t)
        X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
        out[t] = (X, grid_bp_batch(G, W, log_k, X, al, de, log_mu=log_mu)[0])
    return out


def _regime_cell(args):
    cfg, seed, n, arm = args
    import torch
    torch.set_num_threads(int(cfg["torch_threads"]))
    t0 = time.perf_counter()
    g, w = make_grid(cfg["half_width"], cfg["grid_size"])
    G, W = doubled(g, w)
    mu2 = e33._stationary_mu(G, W)
    lk_true = TRUE_REGIME.log_transition_matrix(G)
    mu_true = mu2(TRUE_REGIME)
    val = _regime_bundle("val", seed, cfg["n_val"], G, W, lk_true, mu_true)
    test = _regime_bundle("test", seed, cfg["n_test"], G, W, lk_true, mu_true)
    A = sample_regime_chain(TRUE_REGIME, rng_for("e73-regime-train", seed, n), n, L, burn=cfg["burn_regime"])
    idx = np.array_split(rng_for("e73-regime-split", seed, n).permutation(n), len(T_SCHEDULE))
    noise = rng_for("e73-regime-em-noise", seed, n)
    groups = []
    for t, ix in zip(T_SCHEDULE, idx):
        al, de = alpha_delta(t)
        groups.append((al * A[ix] + np.sqrt(de) * noise.standard_normal(A[ix].shape), al, de))
    base = {"part": "regime", "seed": seed, "n_chains": n, "test_bundle": e03._bundle_hash(test)}
    info = {}

    def pairs_on(grid_, w_, lk, mu, bundle):
        return [e03._sq(grid_bp_batch(grid_, w_, lk, X, *alpha_delta(t), log_mu=mu)[0], m, X, t, e03.REGIONS["all"])
                for t, (X, m) in bundle.items()]

    if arm == "regime":
        ladder = set(cfg["regime_ladder"])
        _, _, saved = fit_em(RegimeKernel.init(0.3, 0.4, 1.2, 0.9, 0.9), G, W, groups, n_iters=max(ladder),
                             tol=FROZEN.em_loglik_tol, checkpoints=ladder, log_mu=mu2)
        k, info = _select(saved, lambda k: e03._risk(pairs_on(G, W, k.log_transition_matrix(G), mu2(k), val)))
        pairs = pairs_on(G, W, k.log_transition_matrix(G), mu2(k), test)
        info.update({"fit_alpha": k.alpha, "fit_s0": k.s0, "fit_s1": k.s1, "fit_p00": k.p00, "fit_p11": k.p11})
    elif arm == "first":
        k, mu_fn, info = e62.fit_first(cfg, groups, g, w, val, ("regime", seed, n))
        pairs = e62._first_pairs(k, g, w, mu_fn, test)
    elif arm == "energy":
        k0 = _energy_kernel(cfg, ("regime", seed, n))
        k, info = e56._fit_and_select(cfg, k0, groups, g, w, val)
        info = {key: info[key] for key in ("checkpoint", "at_cap", "val_risk", "n_params", "final_lr")}
        pairs = e56._pairs(k, test, g, w)
    elif arm == "lmmse":
        i = np.arange(L)
        cov = 0.85 ** np.abs(i[:, None] - i[None, :]).astype(float)
        mean = e58.lmmse_mean_fn(cov)
        pairs = [e03._sq(mean(X, t), m, X, t, e03.REGIONS["all"]) for t, (X, m) in test.items()]
    else:
        pairs, info = _net(cfg, arm, A, val, test, seed)
    rows = _rows(base, arm, pairs, {**info, "wall": time.perf_counter() - t0})
    print(f"  [E73 regime] seed={seed} n={n:>5} {arm:7s} pooled={rows[0]['risk_pooled']:.4f}", flush=True)
    return rows


# ---------------------------------------------------------------------------------------- real series

def series(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(standardised series, split label 0 train / 1 val / 2 test), NaN where a day is missing."""
    import pandas as pd
    if name == "dtb3":
        d = pd.read_csv(ROOT / "experiments/data/raw_dtb3/DTB3.csv")
        d["DTB3"] = pd.to_numeric(d.DTB3, errors="coerce")
        d = d.dropna()
        yr = pd.to_datetime(d.observation_date).dt.year.to_numpy()
        x = d.DTB3.to_numpy(float)
        split = np.where(yr <= 2004, 0, np.where(yr <= 2010, 1, 2))
    elif name == "milan":
        d = pd.read_csv(ROOT / "experiments/data/raw_ghcn/ITE00100554.csv", usecols=["DATE", "TMAX", "TMIN"], low_memory=False)
        d["DATE"] = pd.to_datetime(d.DATE)
        d = d[d.DATE.dt.year >= 1900].set_index("DATE").asfreq("D")
        temp = ((d.TMAX + d.TMIN) / 20.0).to_numpy(float)          # tenths of a degree -> degrees
        yr, doy = d.index.year.to_numpy(), np.minimum(d.index.dayofyear.to_numpy(), 365)
        train = yr <= 1979
        clim = np.array([np.nanmean(temp[train & (doy == k)]) for k in range(1, 366)])
        x = temp - clim[doy - 1]
        split = np.where(train, 0, np.where(yr <= 1989, 1, 2))
    else:
        raise ValueError(name)
    tr = split == 0
    z = (x - np.nanmean(x[tr])) / np.nanstd(x[tr])
    return z, split


def _windows(z, stride):
    starts = np.arange(0, len(z) - L + 1, stride)
    W = np.stack([z[s:s + L] for s in starts])
    return W[np.isfinite(W).all(axis=1)]


def real_data(cfg, name, seed, n):
    z, split = series(name)
    train = _windows(z[split == 0], L)
    idx = rng_for("e73-real-sub", name, seed, n).choice(len(train), size=min(n, len(train)), replace=False)
    return train[idx], _windows(z[split == 1], L), _windows(z[split == 2], cfg["real_test_stride"])


def _real_bundle(C, name, kind, seed):
    rng = rng_for(f"e73-real-{name}-{kind}", seed)
    out = {}
    for t in T_SCHEDULE:
        al, de = alpha_delta(t)
        out[t] = (al * C + np.sqrt(de) * rng.standard_normal(C.shape), C)    # target: the clean value
    return out


def _real_cell(args):
    cfg, name, seed, n, arm = args
    import torch
    torch.set_num_threads(int(cfg["torch_threads"]))
    t0 = time.perf_counter()
    train, V, T = real_data(cfg, name, seed, n)
    val, test = _real_bundle(V, name, "val", seed), _real_bundle(T, name, "test", seed)
    base = {"part": "real", "series": name, "seed": seed, "n_chains": n, "n_train": len(train),
            "test_bundle": e03._bundle_hash(test)}
    key = ("real", name, seed, n)
    if arm == "first":
        g, w = make_grid(cfg["half_width"], cfg["grid_size"])
        k, mu_fn, info = e62.fit_first(cfg, clean_statistics(g, train), g, w, val, key, clean=True)
        pairs = e62._first_pairs(k, g, w, mu_fn, test)
    elif arm == "pair":
        g, w = make_grid(cfg["half_width"], cfg["pair_grid"])
        k, info = e62.fit_pair(cfg, clean_triples(g, train), g, w, val, key, clean=True)
        pairs = e62._pair_pairs(k, g, w, test)
    elif arm == "energy":
        g, w = make_grid(cfg["half_width"], cfg["grid_size"])
        mu_fn = e33._stationary_mu(g, w)
        saved, _ = e58.fit_clean_ladder(_energy_kernel(cfg, key), g, clean_statistics(g, train), set(cfg["ladder"]))
        k, info = _select(saved, lambda k: e03._risk(e62._first_pairs(k, g, w, mu_fn, val)))
        info["final_lr"] = float(getattr(k, "lr", np.nan))
        pairs = e62._first_pairs(k, g, w, mu_fn, test)
    else:
        pairs, info = _net(cfg, arm, train, val, test, seed)
    rows = _rows(base, arm, pairs, {**info, "wall": time.perf_counter() - t0})
    print(f"  [E73 {name}] seed={seed} n={n:>4} {arm:7s} pooled={rows[0]['risk_pooled']:.4f}", flush=True)
    return rows


# ---------------------------------------------------------------------------------------- parts

def part_regime(cfg, out, workers):
    cells = [(cfg, s, n, a) for n in sorted(cfg["regime_sizes"], reverse=True) for a in cfg["regime_arms"]
             for s in cfg["seeds"]]
    write_csv(Path(out) / "regime.csv", run_cells(cells, _regime_cell, workers, out, "regime.csv",
                                                   key=lambda c: f"s{c[1]}_n{c[2]}_{c[3]}"))


def part_real(cfg, out, workers):
    cells = [(cfg, name, s, n, a) for name in cfg["real_series"] for n in sorted(cfg["real_sizes"][name], reverse=True)
             for a in cfg["real_arms"] for s in cfg["seeds"]]
    write_csv(Path(out) / "real.csv", run_cells(cells, _real_cell, workers, out, "real.csv",
                                                 key=lambda c: f"{c[1]}_s{c[2]}_n{c[3]}_{c[4]}"))


def part_describe(cfg, out):
    """D1: lag-1 and lag-2 partial autocorrelations of each real series on its training years."""
    res = {}
    for name in cfg["real_series"]:
        z, split = series(name)
        x = z[split == 0]
        x = x[np.isfinite(x)]
        r1 = np.corrcoef(x[:-1], x[1:])[0, 1]
        r2 = np.corrcoef(x[:-2], x[2:])[0, 1]
        res[name] = {"n_train_days": int(len(x)), "pacf1": float(r1), "pacf2": float((r2 - r1 ** 2) / (1 - r1 ** 2))}
    write_json(Path(out) / "describe.json", res)
    print(f"  [E73] {res}", flush=True)


def main() -> None:
    parser = experiment_parser("e73_real_and_hidden", "hidden-regime chain and two real series")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    w = args.workers
    parts = {"describe": ("partial autocorrelations of the real series", part_describe),
             "regime": ("the hidden-regime chain", lambda c, o: part_regime(c, o, w)),
             "real": ("the two real series", lambda c, o: part_real(c, o, w))}
    if args.list_parts:
        print("\n".join(parts)); return
    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir)
    tag = "_".join(selected) if args.only else "all"
    resolved = {"quick": args.quick, "parts": list(selected), "overrides": args.set, **cfg}
    write_json(out / f"params_{tag}.json", {**resolved, **provenance(resolved)})
    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(cfg, out)


if __name__ == "__main__":
    main()
