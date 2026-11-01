"""Calibrate E59's SETAR chain from its TRUE kernel on a wide grid: no sample is drawn and nothing is fitted.

    .venv/bin/python tools/core/calibrate_e59.py

A two-regime self-exciting threshold autoregression (Tong's SETAR(2; 1, 1)), the textbook nonlinear time-series model:

    a' = phi_L a + c_L + s_L eps   if a <= r       (low regime: persistent, quiet)
    a' = phi_H a + c_H + s_H eps   if a >  r       (high regime: mean-reverting, noisy)

The raw chain is then standardised, b = (a - loc) / scale, with loc and scale its stationary mean and standard deviation,
so that every method in E59 works on a series of mean 0 and variance 1 like the other chains of the project. The kernel
is not a shift kernel phi(b' - rho b): both its mean and its scale change at the threshold.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigs

RAW = dict(phi_l=0.95, c_l=0.0, s_l=0.4, phi_h=0.5, c_h=0.4, s_h=0.8, r=0.0)
G = np.linspace(-15.0, 15.0, 3001)
W = np.full(G.size, G[1] - G[0])
W[[0, -1]] *= 0.5


def raw_log_k(p=RAW, g=G):
    low = g[None, :] <= p["r"]
    mean = np.where(low, p["phi_l"] * g[None, :] + p["c_l"], p["phi_h"] * g[None, :] + p["c_h"])
    sd = np.where(low, p["s_l"], p["s_h"])
    return -0.5 * ((g[:, None] - mean) / sd) ** 2 - np.log(sd) - 0.5 * np.log(2 * np.pi)


def stats(log_k):
    k = np.exp(log_k - log_k.max(0, keepdims=True))
    k /= (k * W[:, None]).sum(0, keepdims=True)
    p = k * W[:, None]
    _, vec = eigs(p, k=1, which="LM")
    pi = np.abs(np.real(vec[:, 0]))
    pi /= pi.sum()
    m = pi @ G
    v = pi @ (G - m) ** 2
    c1 = (pi * (G - m) * ((p * G[:, None]).sum(0) - m)).sum() / v
    return m, v, c1, pi


def main() -> None:
    m, v, c1, pi = stats(raw_log_k())
    s = np.sqrt(v)
    z = (G - m) / s
    skew = pi @ z**3
    kurt = pi @ z**4 - 3
    frac_low = pi[G <= RAW["r"]].sum()
    print(f"SETAR raw parameters {RAW}")
    print(f"stationary: loc {m:.6f}, scale {s:.6f}, lag-1 corr {c1:.6f}, skew {skew:.3f}, excess kurtosis {kurt:.3f}, "
          f"time in the low regime {frac_low:.3f}, threshold in standard units {(RAW['r'] - m) / s:.6f}")


if __name__ == "__main__":
    main()
