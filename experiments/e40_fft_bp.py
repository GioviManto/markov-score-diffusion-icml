#!/usr/bin/env python
"""E40 -- BP with the transition applied by FFT convolution: what it costs in accuracy, what it buys in time.

The consolidated note, section 29 step 5 ("implement FFT-based Gaussian convolution for the current Markov model")
and section 9.2. For K(a'|a) = p_eps(a' - rho a) the transition is a convolution after rescaling a -> rho a
(`markov_score.bp_fft`); the rescaling needs interpolation, which the dense recursion does not. E24's anatomy put a cap
on the payoff before any implementation: at N_g = 401 the N_g^2 matmul is 24% of grid BP's time.

PREDECLARED (registry E40.json, after an accuracy pilot recorded there, before any timing):
  P1  at (401, 8) FFT-BP vs dense grid BP <= 1e-3 at every level; error at N_g = 1601 at least 10x smaller than at 401.
  P2  at N_g = 401 FFT-BP is at most 1.3x faster than dense grid BP (possibly slower).
  P3  the speed-up increases monotonically over N_g = 201 .. 3201 and exceeds 1 at 3201.

Single-threaded: the thread variables are set before numpy loads, so BLAS (dense) and pocketfft (FFT) both use one
core and the comparison is per core.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score.bp_fft import FFTTransition, fft_bp_batch, residual_log_density  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

SETTINGS = dict(rho=FROZEN.rho, n_sites=FROZEN.n_sites, n_chains=256, levels=tuple(FROZEN.t_grid), half_width=8.0,
                grid_sizes=(201, 401, 801, 1601, 3201), reference=(1601, 8.0), time_batch=512, time_level=1.0, repeats=5)
QUICK = dict(n_chains=16, levels=(0.05, 1.0), grid_sizes=(101, 201), reference=(201, 8.0), time_batch=32, repeats=2)


def _transition(prior, grid, w):
    lags = np.arange(-(len(grid) - 1), len(grid)) * (grid[1] - grid[0])
    return FFTTransition(grid, w, residual_log_density(prior, lags), prior.rho)


def _score(X, m, alpha, delta):
    return -(X - alpha * np.asarray(m)) / delta


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def part_accuracy(cfg, out):
    prior = LaplaceAR1(cfg["rho"])
    rng = rng_for("e40-truth")
    A = np.stack([prior.sample(rng, cfg["n_sites"]) for _ in range(cfg["n_chains"])])
    views = {}
    n_ref, a_ref = cfg["reference"]
    g_ref, w_ref = make_grid(a_ref, n_ref)
    lk_ref = prior.log_transition_matrix(g_ref)
    for t in cfg["levels"]:
        al, de = alpha_delta(float(t))
        X = al * A + np.sqrt(de) * rng_for("e40-noise", float(t)).standard_normal(A.shape)
        views[float(t)] = (X, al, de, _score(X, grid_bp_batch(g_ref, w_ref, lk_ref, X, al, de)[0], al, de))
    rows = []
    for n in cfg["grid_sizes"]:
        grid, w = make_grid(cfg["half_width"], n)
        lk = prior.log_transition_matrix(grid)
        tr = _transition(prior, grid, w)
        for t, (X, al, de, s_ref) in views.items():
            s_dense = _score(X, grid_bp_batch(grid, w, lk, X, al, de)[0], al, de)
            s_fft = _score(X, fft_bp_batch(grid, w, tr, X, al, de)[0], al, de)
            rows.append(dict(part="accuracy", n_grid=n, half_width=cfg["half_width"], t=t, method="fft",
                             err_vs_dense=_rel(s_fft, s_dense), err_vs_reference=_rel(s_fft, s_ref), wall=np.nan))
            rows.append(dict(part="accuracy", n_grid=n, half_width=cfg["half_width"], t=t, method="dense",
                             err_vs_dense=0.0, err_vs_reference=_rel(s_dense, s_ref), wall=np.nan))
        worst = max(r["err_vs_dense"] for r in rows if r["n_grid"] == n and r["method"] == "fft")
        print(f"  [E40] accuracy N_g={n}: worst FFT vs dense {worst:.2e}", flush=True)
    return rows


def _median_time(fn, repeats):
    fn()                                                   # warm-up
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), ts


def part_timing(cfg, out):
    prior = LaplaceAR1(cfg["rho"])
    rng = rng_for("e40-time")
    A = np.stack([prior.sample(rng, cfg["n_sites"]) for _ in range(cfg["time_batch"])])
    al, de = alpha_delta(float(cfg["time_level"]))
    X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
    rows = []
    for n in cfg["grid_sizes"]:
        grid, w = make_grid(cfg["half_width"], n)
        lk = prior.log_transition_matrix(grid)
        tr = _transition(prior, grid, w)
        td, ts_d = _median_time(lambda: grid_bp_batch(grid, w, lk, X, al, de, chunk=None), cfg["repeats"])
        tf, ts_f = _median_time(lambda: fft_bp_batch(grid, w, tr, X, al, de), cfg["repeats"])
        for method, med, ts in (("dense", td, ts_d), ("fft", tf, ts_f)):
            rows.append(dict(part="timing", n_grid=n, half_width=cfg["half_width"], t=cfg["time_level"], method=method,
                             err_vs_dense=np.nan, err_vs_reference=np.nan, wall=med,
                             walls=";".join(f"{x:.5f}" for x in ts), batch=cfg["time_batch"]))
        print(f"  [E40] timing N_g={n}: dense {td:.3f}s, fft {tf:.3f}s, speed-up {td / tf:.2f}", flush=True)
    return rows


PARTS = {"accuracy": part_accuracy, "timing": part_timing}


def main():
    ap = experiment_parser("e40_fft_bp", "BP with an FFT transition: accuracy and time against dense grid BP")
    args = ap.parse_args()
    if args.list_parts:
        print("\n".join(PARTS)); return
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    out = ensure_dir(Path(args.output_dir))
    rows = []
    for name, fn in select_parts(PARTS, args.only).items():
        rows += fn(cfg, out)
        write_json(out / f"params_{name}.json", {**provenance(), "settings": cfg, "quick": bool(args.quick),
                                                 "threads": {v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    write_csv(out / "fft.csv", rows)


if __name__ == "__main__":
    main()
