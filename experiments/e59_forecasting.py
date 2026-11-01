#!/usr/bin/env python
"""E59 — probabilistic forecasting with a learned Markov kernel, on SETAR and on hourly offshore wind speed.

A first-order chain's h-step predictive law is the kernel applied h times to the filtered law of the present:
    p(a_{k+h} | past) = K^h p(a_k | past),
with the filtered law a point mass at the last value when the past is clean and a BP forward filter when it is noisy.
The kernel is the neural energy kernel fitted by clean-data maximum likelihood (E58's code), selected on the held-out
likelihood (no oracle exists for real data). Against it, on the same training windows: the Gaussian AR(1) kernel (its
filter is the Kalman filter), quantile MLP and GRU networks (pinball loss), persistence and climatology; on SETAR also
the oracle (the true kernel). Every method is scored by the same quantile CRPS on the same test targets.

Benchmarks: `setar` (standardised SETAR(2; 1, 1), priors.SETARChain) and `wind` (Horns Rev, ERA5 2015-2024,
data/e59/wind_hornsrev_2015_2024.json; square root, minus the training month-by-hour mean, over the training sd).
Windows of 32; forecast origin at site 24; horizons 1, 2, 4, 8; scenarios `clean` and `noisy` (past observed at t = 0.2).

PREDECLARED in experiments/registry/E59.json (P1-P5, D1-D4) before this file was written.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from markov_score import forecast as fc  # noqa: E402
from markov_score import quantile_nets as qn  # noqa: E402
from markov_score.bp_grid import make_grid  # noqa: E402
from markov_score.em import clean_statistics  # noqa: E402
from markov_score.energy_kernel import NeuralEnergyKernel  # noqa: E402
from markov_score.kernels import GaussianAR1Kernel  # noqa: E402
from markov_score.priors import SETARChain  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

import e58_clean_kernel as e58  # noqa: E402   (fit_clean_ladder and clean_nll, imported not copied)

ROOT = Path(__file__).resolve().parents[1]
WIND = ROOT / "data" / "e59" / "wind_hornsrev_2015_2024.json"
WIND_SHA256 = "7d55be6177241d07b01ba6afe18d1958104e48b953431ac05dbb90a42daf3f7e"
L, ORIGIN, WINDOW, HORIZONS = 32, 23, 16, (1, 2, 4, 8)      # ORIGIN: zero-based index of site 24

SETTINGS = dict(
    benchmarks=("setar", "wind"), seeds=tuple(range(8)),
    sizes={"setar": (32, 128, 512, 2048), "wind": (32, 128, 512, 1917)},
    grids={"setar": (8.0, 401), "wind": (6.0, 601)},
    n_val_setar=1024, n_test_setar=4096, wind_test_stride=8, max_test=0,
    noisy_t=0.2, ladder=(10, 20, 40, 80, 160, 300, 600, 1200),
    energy_hidden=16, energy_lr=1e-2, energy_inner=10, init_rho=0.3, init_var=0.8,
    nets=("qmlp", "qgru"), net_checkpoints=(1000, 2000, 4000, 8000), net_lr=1e-3, net_batch=256, torch_threads=1,
)
QUICK = dict(seeds=(0,), sizes={"setar": (32,), "wind": (32,)}, n_val_setar=64, n_test_setar=64, max_test=64,
             ladder=(2, 4), net_checkpoints=(50, 100))


# ---------------------------------------------------------------------------------------- data

def wind_series() -> tuple[np.ndarray, np.ndarray]:
    """The registered transform of the wind series, and the split label of every hour (0 train, 1 val, 2 test)."""
    raw = WIND.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == WIND_SHA256, "the wind file is not the registered one"
    d = json.loads(raw)
    v = np.array(d["hourly"]["wind_speed_100m"], dtype=float)
    ts = [dt.datetime.fromisoformat(x) for x in d["hourly"]["time"]]
    yr = np.array([x.year for x in ts]); mo = np.array([x.month for x in ts]); hr = np.array([x.hour for x in ts])
    r = np.sqrt(v)
    train = yr <= 2021
    clim = np.zeros((13, 24))
    for m in range(1, 13):
        for h in range(24):
            clim[m, h] = r[train & (mo == m) & (hr == h)].mean()
    z = r - clim[mo, hr]
    z = (z - z[train].mean()) / z[train].std()
    split = np.where(train, 0, np.where(yr == 2022, 1, 2))
    return z, split


def _windows(x, stride):
    starts = np.arange(0, len(x) - L + 1, stride)
    return np.stack([x[s:s + L] for s in starts])


def data(cfg, bench, seed, n):
    """(train windows, validation windows, test windows), each (count, 32), clean."""
    if bench == "setar":
        prior = SETARChain()
        rt, rv, rs = rng_for("e59-setar-train", seed, n), rng_for("e59-setar-val", seed), rng_for("e59-setar-test", seed)
        A = np.stack([prior.sample(rt, L) for _ in range(n)])
        V = np.stack([prior.sample(rv, L) for _ in range(cfg["n_val_setar"])])
        T = np.stack([prior.sample(rs, L) for _ in range(cfg["n_test_setar"])])
        return A, V, T
    z, split = wind_series()
    train = _windows(z[split == 0], L)                       # 1917 non-overlapping windows
    idx = rng_for("e59-wind-sub", seed, n).choice(len(train), size=min(n, len(train)), replace=False)
    T = _windows(z[split == 2], cfg["wind_test_stride"])
    if cfg["max_test"]:
        T = T[:cfg["max_test"]]
    return train[idx], _windows(z[split == 1], L), T


def noisy(cfg, A, bench, split, seed, n):
    t = cfg["noisy_t"]
    al, de = np.exp(-t), 1 - np.exp(-2 * t)
    xi = rng_for(f"e59-{bench}-noise-{split}", seed, n).standard_normal(A.shape)
    return al * A + np.sqrt(de) * xi, al, de


def net_examples(A, P):
    """Every origin with 16 past and 8 future values inside a window: inputs from P (clean or noisy), targets from A."""
    X, Y = [], []
    for o in range(WINDOW - 1, L - max(HORIZONS)):
        X.append(P[:, o - WINDOW + 1:o + 1])
        Y.append(np.stack([A[:, o + h] for h in HORIZONS], axis=1))
    return np.concatenate(X), np.concatenate(Y)


# ---------------------------------------------------------------------------------------- scoring

def score(q, y):
    """Per-sample quantile CRPS, coverage and the median's squared error."""
    return fc.crps_from_quantiles(q, y), fc.coverage(q, y), (q[:, len(fc.LEVELS) // 2] - y) ** 2


def grid_rows(name, log_k, grid, w, past, scen, al, de, Y):
    rows = []
    f = fc.forward_filter(log_k, grid, w, past, al, de) if scen == "noisy" else None
    for i, h in enumerate(HORIZONS):
        p = fc.propagate(f, log_k, w, h) if scen == "noisy" else fc.predictive_from_point(log_k, grid, w, past[:, -1], h)
        q = fc.quantiles(p, grid, w)
        crps, cov, se = score(q, Y[:, i])
        rows.append(dict(method=name, h=h, crps=float(crps.mean()), crps_exact=float(fc.crps_exact(p, grid, w, Y[:, i]).mean()),
                         cov90=float(cov.mean()), rmse=float(np.sqrt(se.mean())),
                         log_score=float(fc.log_score(p, grid, Y[:, i]).mean())))
    return rows


def quantile_rows(name, Q, Y):
    rows = []
    for i, h in enumerate(HORIZONS):
        crps, cov, se = score(Q[:, i, :], Y[:, i])
        rows.append(dict(method=name, h=h, crps=float(crps.mean()), crps_exact=np.nan, cov90=float(cov.mean()),
                         rmse=float(np.sqrt(se.mean())), log_score=np.nan))
    return rows


# ---------------------------------------------------------------------------------------- one cell

def _cell(args):
    cfg, bench, seed, n = args
    import torch
    torch.set_num_threads(int(cfg["torch_threads"]))
    half, size = cfg["grids"][bench]
    grid, w = make_grid(half, size)
    A, V, T = data(cfg, bench, seed, n)
    Y = np.stack([T[:, ORIGIN + h] for h in HORIZONS], axis=1)
    base = dict(benchmark=bench, seed=seed, n_chains=n, n_test=len(T),
                target_hash=hashlib.sha256(np.ascontiguousarray(Y).tobytes()).hexdigest()[:12])
    rows = []
    # kernels: one fit each, from the clean training windows, used in both scenarios
    stats, held = clean_statistics(grid, A), clean_statistics(grid, V)
    t0 = time.perf_counter()
    k0 = NeuralEnergyKernel.init(hidden=cfg["energy_hidden"], seed=int(rng_for("e59-energy-init", bench, seed, n).integers(0, 2**31 - 1)),
                                 half_width=half, lr=cfg["energy_lr"], n_inner=cfg["energy_inner"])
    saved, _ = e58.fit_clean_ladder(k0, grid, stats, set(cfg["ladder"]))
    nll = {it: e58.clean_nll(k, grid, held) for it, k in saved.items()}
    best = min(nll, key=nll.get)
    kernels = {"kernel": (saved[best].log_transition_matrix(grid), dict(checkpoint=best, heldout_nll=nll[best],
                                                                        wall=time.perf_counter() - t0))}
    ar1 = GaussianAR1Kernel(rho=cfg["init_rho"], q=cfg["init_var"]).m_step(stats, grid)
    kernels["gauss_ar1"] = (ar1.log_transition_matrix(grid), dict(checkpoint=1, heldout_nll=e58.clean_nll(ar1, grid, held)))
    if bench == "setar":
        kernels["oracle"] = (SETARChain().log_transition_matrix(grid), dict(checkpoint=0, heldout_nll=np.nan))
    # naive baselines from the clean training windows
    clim = np.quantile(A.ravel(), fc.LEVELS)
    changes = {h: np.quantile(np.concatenate([A[:, o + h] - A[:, o] for o in range(WINDOW - 1, L - max(HORIZONS))]), fc.LEVELS)
               for h in HORIZONS}
    for scen in ("clean", "noisy"):
        if scen == "clean":
            past, al, de = T[:, :ORIGIN + 1], 1.0, 0.0
            Ptr, Pv, Pte = A, V, T
        else:
            Pte, al, de = noisy(cfg, T, bench, "test", seed, n)
            past = Pte[:, :ORIGIN + 1]
            Ptr, _, _ = noisy(cfg, A, bench, "train", seed, n)
            Pv, _, _ = noisy(cfg, V, bench, "val", seed, n)
        for name, (lk, info) in kernels.items():
            for r in grid_rows(name, lk, grid, w, past, scen, al, de, Y):
                rows.append({**base, "scenario": scen, **r, **info})
        last = past[:, -1] / (al if scen == "noisy" else 1.0)
        rows += [{**base, "scenario": scen, **r} for r in
                 quantile_rows("climatology", np.broadcast_to(clim, (len(T), len(HORIZONS), len(fc.LEVELS))), Y)]
        rows += [{**base, "scenario": scen, **r} for r in
                 quantile_rows("persistence", np.stack([last[:, None] + changes[h][None, :] for h in HORIZONS], axis=1), Y)]
        X, Yx = net_examples(A, Ptr)
        Xv, Yv = net_examples(V, Pv)
        Xt = Pte[:, ORIGIN - WINDOW + 1:ORIGIN + 1]
        for arch in cfg["nets"]:
            t0 = time.perf_counter()
            model, info = qn.train(arch, X, Yx, Xv, Yv, fc.LEVELS, steps=max(cfg["net_checkpoints"]),
                                   checkpoints=set(cfg["net_checkpoints"]), lr=cfg["net_lr"], batch=cfg["net_batch"],
                                   seed=int(rng_for("e59-net", arch, bench, scen, seed, n).integers(0, 2**31 - 1)))
            info["wall"] = time.perf_counter() - t0
            rows += [{**base, "scenario": scen, **r, **info} for r in quantile_rows(arch, qn.predict(model, Xt), Y)]
    print(f"  [E59] {bench} seed={seed} n={n:>5} done ({len(rows)} rows; kernel checkpoint {best})", flush=True)
    return rows


def part_forecast(cfg, out, workers):
    cells = [(cfg, b, s, n) for b in cfg["benchmarks"] for s in cfg["seeds"] for n in cfg["sizes"][b]]
    key = lambda c: f"{c[1]}_s{c[2]}_n{c[3]}"  # noqa: E731
    write_csv(Path(out) / "forecasts.csv", run_cells(cells, _cell, workers, out, "forecasts.csv", key=key))


def main() -> None:
    parser = experiment_parser("e59_forecasting", "probabilistic forecasting with a learned Markov kernel")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    parts = {"forecast": ("every method, both benchmarks, both scenarios", lambda c, o: part_forecast(c, o, args.workers))}
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
