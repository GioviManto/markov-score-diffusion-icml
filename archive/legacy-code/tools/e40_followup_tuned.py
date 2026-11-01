#!/usr/bin/env python
"""E40 follow-up, UNREGISTERED (information only): how much of the FFT path's deficit is implementation slack?

The registered E40 run used markov_score.bp_fft as committed: the FFT length is the next power of two >= 3 N_g - 2, and
the cubic interpolation is a fancy-indexed gather. Neither is minimal. The outputs needed, n-1 .. 2n-2 of the linear
convolution, are free of wrap-around for any circular length >= 2 N_g - 1, and the gather is a sparse matrix with four
entries per row. This script times a tuned transition (scipy.fft.next_fast_len(2 N_g - 1), CSR gather) against the
same dense grid BP under E40's timing design (B = 512, t = 1, single thread, median of 5), checks it returns the same
posterior means as the registered FFT path, and writes outputs/E40_followup/tuned.csv. It reads nothing into E40's
verdicts.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.fft as sfft  # noqa: E402
import scipy.sparse as sp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import e40_fft_bp as e40  # noqa: E402
from markov_score.bp_fft import FFTTransition, fft_bp_batch  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv  # noqa: E402


class TunedTransition(FFTTransition):
    def __init__(self, grid, weights, log_p_eps_lags, rho):
        super().__init__(grid, weights, log_p_eps_lags, rho)
        n = self.n
        k = np.exp(np.asarray(log_p_eps_lags, dtype=float))
        self.nfft = sfft.next_fast_len(2 * n - 1, real=True)
        self.k_hat = sfft.rfft(k, self.nfft)
        self.k_rev_hat = sfft.rfft(k[::-1], self.nfft)
        rows = np.repeat(np.arange(n), 4)
        self.Gf = sp.csr_matrix((self.f_w.ravel(), (rows, self.f_idx.ravel())), shape=(n, n)).T.tocsr()
        self.Gb = sp.csr_matrix((self.b_w.ravel(), (rows, self.b_idx.ravel())), shape=(n, n)).T.tocsr()

    def _conv(self, g, kernel_hat):
        full = sfft.irfft(sfft.rfft(g, self.nfft, axis=1) * kernel_hat[None, :], self.nfft, axis=1)
        # circular length >= 2n - 1: outputs n-1 .. 2n-2 are free of wrap-around
        return full[:, self.n - 1: 2 * self.n - 1]

    def forward(self, f):
        g = (f @ self.Gf) / self.rho
        return self._conv(g * self.weights[None, :], self.k_hat)

    def backward(self, h):
        return self._conv(h * self.weights[None, :], self.k_rev_hat) @ self.Gb


def main() -> int:
    cfg = dict(e40.SETTINGS)
    prior = LaplaceAR1(cfg["rho"])
    rng = rng_for("e40-time")
    A = np.stack([prior.sample(rng, cfg["n_sites"]) for _ in range(cfg["time_batch"])])
    al, de = alpha_delta(float(cfg["time_level"]))
    X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
    rows = []
    for n in cfg["grid_sizes"]:
        grid, w = make_grid(cfg["half_width"], n)
        lags = np.arange(-(n - 1), n) * (grid[1] - grid[0])
        from markov_score.bp_fft import residual_log_density
        lp = residual_log_density(prior, lags)
        reg, tuned = FFTTransition(grid, w, lp, prior.rho), TunedTransition(grid, w, lp, prior.rho)
        same = float(np.max(np.abs(fft_bp_batch(grid, w, reg, X[:64], al, de)[0] - fft_bp_batch(grid, w, tuned, X[:64], al, de)[0])))
        lk = prior.log_transition_matrix(grid)
        td, _ = e40._median_time(lambda: grid_bp_batch(grid, w, lk, X, al, de, chunk=None), cfg["repeats"])
        tt, _ = e40._median_time(lambda: fft_bp_batch(grid, w, tuned, X, al, de), cfg["repeats"])
        rows.append(dict(n_grid=n, dense=td, fft_tuned=tt, speedup=td / tt, nfft=tuned.nfft, max_mean_diff_vs_registered=same))
        print(f"  [E40 follow-up] N_g={n}: dense {td:.3f}s, tuned FFT {tt:.3f}s (nfft {tuned.nfft}), speed-up {td / tt:.2f}; "
              f"means differ from the registered FFT path by {same:.1e}", flush=True)
    out = ensure_dir(ROOT / "outputs/E40_followup")
    write_csv(out / "tuned.csv", rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
