"""E62's appendix (D5): what longer lags cost on a grid, and what they buy. Laptop, true kernels only.

    python tools/core/appendix_e62_arp.py

An order-p chain is first order on the p-tuple state; because consecutive tuples share p - 1 coordinates, exact BP costs
n^(p+1) per chain and site (n^2, n^3, n^4 for p = 1, 2, 3), and the forward messages n^p per site. Part 1 times exact BP for
p = 1, 2, 3 at n = 51 and 101 (one thread, the cluster's setting). Part 2 builds a Laplace AR(3) at lag-one correlation
0.85 and variance 1 from its partial autocorrelations (pi_2, pi_3) = (0.3, 0.2) (Durbin-Levinson), and scores, against
the exact triple-state reference on 51 points, the BP of its best first-order and best second-order projections (the
stationary law's conditionals): how much of the misspecification the pair state already removes.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
import e03_baselines as e03  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.pair_bp import pair_bp_batch  # noqa: E402
from markov_score.utils import rng_for  # noqa: E402

L, RHO1, PACF = 32, 0.85, (0.3, 0.2)


def _lik(grid, X, al, de):
    ll = -0.5 * (X[:, :, None] - al * grid[None, None, :]) ** 2 / de
    return np.exp(ll - ll.max(axis=2, keepdims=True))


def triple_bp_means(grid, w, k3, X, al, de, mu3):
    """Exact BP on (a_{t-2}, a_{t-1}, a_t). k3[l, k, j, i] = K3(u_l | u_k, u_j, u_i) (densities); mu3[i, j, k] the law of
    (a_1, a_2, a_3). Forward-backward, n^4 per chain and site; returns the posterior means (B, L)."""
    B, n = len(X), len(grid)
    phi = _lik(grid, X, al, de)
    kw = np.transpose(k3, (2, 1, 3, 0)) * w[None, None, :, None]              # [j, k, i, l] = w_i K3(l | k, j, i)
    kb = np.transpose(k3, (2, 1, 0, 3)) * w[None, None, :, None]              # [j, k, l, i] = w_l K3(l | k, j, i)
    fw = np.empty((L, B, n, n, n))
    f = mu3[None] * phi[:, 0, :, None, None] * phi[:, 1, None, :, None] * phi[:, 2, None, None, :]
    f /= f.sum(axis=(1, 2, 3), keepdims=True)
    fw[2] = f
    for t in range(3, L):
        g = np.matmul(np.transpose(f, (2, 3, 0, 1)), kw)                       # (j, k, b, l)
        f = np.transpose(g, (2, 0, 1, 3)) * phi[:, t, None, None, :]           # [b, j, k, l]
        f /= f.sum(axis=(1, 2, 3), keepdims=True)
        fw[t] = f
    bk = np.ones((B, n, n, n))
    w3 = w[None, :, None, None] * w[None, None, :, None] * w[None, None, None, :]
    means = np.empty((B, L))
    for t in range(L - 1, 1, -1):
        post = fw[t] * bk * w3
        post /= post.sum(axis=(1, 2, 3), keepdims=True)
        means[:, t] = post.sum(axis=(1, 2)) @ grid
        if t == 2:
            means[:, 1] = post.sum(axis=(1, 3)) @ grid
            means[:, 0] = post.sum(axis=(2, 3)) @ grid
            break
        h = bk * phi[:, t, None, None, :]                                      # [b, j, k, l]
        g = np.matmul(np.transpose(h, (1, 2, 0, 3)), kb)                       # (j, k, b, i)
        bk = np.transpose(g, (2, 3, 0, 1))                                     # [b, i, j, k]
        bk /= bk.max(axis=(1, 2, 3), keepdims=True)
    return means


def ar3():
    """(phi1, phi2, phi3) from (pi_1, pi_2, pi_3) = (0.85, 0.3, 0.2) by Durbin-Levinson, and the innovation variance."""
    phi = np.zeros(0)
    var = 1.0
    for p in (RHO1,) + PACF:
        phi = np.concatenate([phi - p * phi[::-1], [p]])
        var *= 1 - p ** 2
    return phi, var


def laplace_k3(grid, w, phi, var):
    b = np.sqrt(var / 2)
    lk = -np.abs(grid[:, None, None, None] - phi[0] * grid[None, :, None, None] - phi[1] * grid[None, None, :, None]
                 - phi[2] * grid[None, None, None, :]) / b
    k = np.exp(lk)
    return k / np.einsum("lkji,l->kji", k, w)[None]


def stationary_triple(k3, w, iters=4000, tol=1e-12):
    n = len(w)
    kw = np.transpose(k3, (2, 1, 3, 0)) * w[None, None, :, None]
    f = np.ones((n, n, n))                                                     # f[i, j, k]
    for _ in range(iters):
        g = np.matmul(np.transpose(f, (1, 2, 0))[:, :, None, :], kw)[:, :, 0, :]   # g[j, k, l]
        g /= np.sum(g * w[:, None, None] * w[None, :, None] * w[None, None, :])
        done = np.max(np.abs(g - f)) < tol * g.max()
        f = g
        if done:
            break
    return f


def sample(phi, var, rng, n, burn=800):
    a = np.zeros((n, burn + L))
    e = rng.laplace(0.0, np.sqrt(var / 2), size=a.shape)
    for t in range(3, a.shape[1]):
        a[:, t] = phi[0] * a[:, t - 1] + phi[1] * a[:, t - 2] + phi[2] * a[:, t - 3] + e[:, t]
    return a[:, burn:]


def timing():
    rng = np.random.default_rng(0)
    print("part 1: exact BP, seconds per chain of L = 32 (one thread)")
    for n in (51, 101):
        grid, w = make_grid(8.0, n)
        X = rng.standard_normal((32, L))
        lk1 = -0.5 * (grid[:, None] - 0.85 * grid[None, :]) ** 2 / 0.2775
        t0 = time.perf_counter(); grid_bp_batch(grid, w, lk1, X, 0.9, 0.19); t1 = (time.perf_counter() - t0) / 32
        lk2 = np.log(np.maximum(np.repeat(np.exp(lk1)[:, :, None], n, 2), 1e-300))
        mu2 = np.full((n, n), 1.0 / (16.0 ** 2))
        t0 = time.perf_counter(); pair_bp_batch(grid, w, lk2, X, 0.9, 0.19, np.log(mu2)); t2 = (time.perf_counter() - t0) / 32
        if n <= 51:
            k3 = np.repeat(np.repeat(np.exp(lk1)[:, :, None, None], n, 2), n, 3)
            Xs = X[:4]
            t0 = time.perf_counter(); triple_bp_means(grid, w, k3, Xs, 0.9, 0.19, np.full((n, n, n), 1.0 / 16 ** 3))
            t3 = f"{(time.perf_counter() - t0) / 4:.3f}"
        else:
            t3 = f"~{t2 * n:.1f} (n^4 / n^3 scaling from the pair timing; the n^3 forward messages are 4 GB per chain)"
        print(f"  n = {n}: p = 1 {t1:.5f}, p = 2 {t2:.4f}, p = 3 {t3}", flush=True)


def ar3_projections():
    grid, w = make_grid(8.0, 51)
    phi, var = ar3()
    k3 = laplace_k3(grid, w, phi, var)
    mu3 = stationary_triple(k3, w)                                             # [i, j, k] = (a_{t-2}, a_{t-1}, a_t)
    m3 = mu3 * w[:, None, None] * w[None, :, None] * w[None, None, :]
    p2 = m3.sum(0)                                                             # (a_{t-1}, a_t) mass
    p1 = p2.sum(0)
    # second-order projection K2(a_t | a_{t-1}, a_{t-2}) = p(a_{t-2}, a_{t-1}, a_t) / p(a_{t-2}, a_{t-1}), as [k, j, i]
    k2 = np.transpose(m3 / np.maximum(m3.sum(2, keepdims=True), 1e-300), (2, 1, 0)) / w[:, None, None]
    lk2 = np.log(np.maximum(k2, 1e-300))
    mu2 = (p2 / (w[:, None] * w[None, :])).T                                    # [j = a_t, i = a_{t-1}] density
    k1 = (p2 / np.maximum(p2.sum(1, keepdims=True), 1e-300)).T / w[:, None]     # [k = a_t, j = a_{t-1}]
    mu1 = p1 / w
    rng = rng_for("e62-appendix-ar3", 0)
    A = sample(phi, var, rng, 48)
    errs = {"first": [], "second": []}
    for t in (0.05, 0.3, 1.0):
        al, de = alpha_delta(t)
        X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
        ref = np.concatenate([triple_bp_means(grid, w, k3, X[s:s + 8], al, de, mu3) for s in range(0, len(X), 8)])
        m1 = grid_bp_batch(grid, w, np.log(np.maximum(k1, 1e-300)), X, al, de, log_mu=np.log(np.maximum(mu1, 1e-300)))[0]
        m2 = pair_bp_batch(grid, w, lk2, X, al, de, np.log(np.maximum(mu2, 1e-300)))[0]
        errs["first"].append(e03._sq(m1, ref, X, t, slice(None)))
        errs["second"].append(e03._sq(m2, ref, X, t, slice(None)))
    dg = -0.5 * sum(np.log(1 - p ** 2) for p in PACF)
    print(f"part 2: Laplace AR(3), phi = {np.round(phi, 4).tolist()}, innovation var {var:.4f}, delta_gauss "
          f"(asymptotic) {dg:.4f}, of which the lag-3 part {-0.5 * np.log(1 - PACF[1] ** 2):.4f}; n = 51, 48 chains, "
          f"t in (0.05, 0.3, 1.0)")
    print(f"  best first-order projection:  {e03._risk(errs['first']):.4f}")
    print(f"  best second-order projection: {e03._risk(errs['second']):.4f}")


if __name__ == "__main__":
    timing()
    ar3_projections()
