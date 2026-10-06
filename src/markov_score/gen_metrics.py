"""Distributional metrics for generated sequences against a known law.

All metrics compare a generated sample G (n_g, L) with a reference sample R (n_r, L) drawn
from the true law at the same level t_min, except `log_likelihood_true`, which scores
samples under the true noised law directly (no reference sample needed) by the exact
evidence of tree-exact BP under the true kernel: the one metric a reviewer cannot argue
with, because the law is known and the evidence is exact for the discretised model.

Predeclared set (docs/RESEARCH_PLAN.md, E5):
    driving-noise excess kurtosis (with bootstrap s.e.)
    marginal 1-D Wasserstein-1 (pooled over sites)
    sliced Wasserstein-2 in R^L (random projections, common seed)
    energy distance in R^L
    covariance Frobenius relative error and worst-lag absolute error
    lag-1..5 autocorrelation error of the AR residuals
    fourth-order lag cumulant cum(r_i, r_i, r_{i+1}, r_{i+1}) of the residuals
    histogram KL of the residuals (true || generated)
    mean log-likelihood under the true P_{t_min} (BP evidence), and its gap to the
    reference sample's own mean log-likelihood
"""

from __future__ import annotations

import numpy as np

from markov_score.bp_grid import grid_bp_batch


def ar_residuals(a: np.ndarray, rho: float) -> np.ndarray:
    return a[:, 1:] - rho * a[:, :-1]


def excess_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, float).ravel()
    m2 = np.mean((x - x.mean()) ** 2); m4 = np.mean((x - x.mean()) ** 4)
    return float(m4 / m2**2 - 3.0)


def bootstrap_se(rows: np.ndarray, stat, n_boot: int = 200, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    vals = [stat(rows[rng.integers(0, len(rows), len(rows))]) for _ in range(n_boot)]
    return float(np.std(vals, ddof=1))


def wasserstein1_1d(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.sort(np.asarray(x, float).ravel()), np.sort(np.asarray(y, float).ravel())
    q = np.linspace(0, 1, 2001)[1:-1]
    return float(np.mean(np.abs(np.quantile(x, q) - np.quantile(y, q))))


def sliced_wasserstein2(G: np.ndarray, R: np.ndarray, n_proj: int = 256, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    L = G.shape[1]
    P = rng.standard_normal((n_proj, L)); P /= np.linalg.norm(P, axis=1, keepdims=True)
    g, r = G @ P.T, R @ P.T
    n = min(len(g), len(r))
    q = np.linspace(0, 1, n + 1)[1:-1] if n > 2 else np.array([0.5])
    tot = 0.0
    for k in range(n_proj):
        tot += np.mean((np.quantile(g[:, k], q) - np.quantile(r[:, k], q)) ** 2)
    return float(np.sqrt(tot / n_proj))


def energy_distance(G: np.ndarray, R: np.ndarray, max_n: int = 1000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    G = G[rng.permutation(len(G))[:max_n]]; R = R[rng.permutation(len(R))[:max_n]]
    def md(A, B):
        d = np.sqrt(np.maximum(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1), 0.0)); return d.mean()
    return float(2 * md(G, R) - md(G, G) - md(R, R))


def covariance_error(G: np.ndarray, sigma_true: np.ndarray) -> tuple[float, float]:
    S = np.cov(G.T)
    L = S.shape[0]
    fro = float(np.linalg.norm(S - sigma_true) / np.linalg.norm(sigma_true))
    lags = [abs(np.mean(np.diag(S, k)) - np.mean(np.diag(sigma_true, k))) for k in range(0, min(6, L))]
    return fro, float(max(lags))


def residual_autocorr(r: np.ndarray, max_lag: int = 5) -> np.ndarray:
    r = r - r.mean()
    v = np.mean(r * r)
    return np.array([np.mean(r[:, :-k] * r[:, k:]) / v for k in range(1, max_lag + 1)])


def lag_fourth_cumulant(r: np.ndarray) -> float:
    """cum(r_i, r_i, r_{i+1}, r_{i+1}) pooled over sites: zero for Gaussian driving noise,
    and, for an AR(1) residual at t_min > 0, a known nonzero value that shape-blind samplers miss."""
    a, b = r[:, :-1].ravel(), r[:, 1:].ravel()
    a = a - a.mean(); b = b - b.mean()
    return float(np.mean(a * a * b * b) - np.mean(a * a) * np.mean(b * b) - 2 * np.mean(a * b) ** 2)


def histogram_kl(p_samples: np.ndarray, q_samples: np.ndarray, n_bins: int = 60, q_hi: float = 0.999) -> float:
    lo, hi = np.quantile(p_samples, 1 - q_hi), np.quantile(p_samples, q_hi)
    edges = np.linspace(lo, hi, n_bins + 1)
    p, _ = np.histogram(p_samples, edges); q, _ = np.histogram(q_samples, edges)
    p = p / p.sum(); q = (q + 1.0) / (q.sum() + n_bins)
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def log_likelihood_true(X: np.ndarray, grid, weights, log_k_true, t: float, chunk: int = 512) -> np.ndarray:
    """Per-sample log P_t(x) under the TRUE kernel by the exact evidence of grid BP."""
    alpha, delta = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
    out = []
    for i in range(0, len(X), chunk):
        _, _, ev = grid_bp_batch(grid, weights, log_k_true, X[i:i + chunk], alpha, delta, return_evidence=True)
        out.append(np.asarray(ev))
    return np.concatenate(out)


def compare(G: np.ndarray, R: np.ndarray, rho: float, sigma_true: np.ndarray, grid, weights, log_k_true, t_min: float, seed: int = 0) -> dict:
    rg, rr = ar_residuals(G, rho), ar_residuals(R, rho)
    out = {
        "kurt_gen": excess_kurtosis(rg), "kurt_ref": excess_kurtosis(rr),
        "kurt_gen_se": bootstrap_se(rg, excess_kurtosis, seed=seed),
        "w1_marginal": wasserstein1_1d(G, R),
        "sw2": sliced_wasserstein2(G, R, seed=seed),
        "energy": energy_distance(G, R, seed=seed),
        "cov_fro": covariance_error(G, sigma_true)[0], "cov_worst_lag": covariance_error(G, sigma_true)[1],
        "acf_err_max": float(np.max(np.abs(residual_autocorr(rg) - residual_autocorr(rr)))),
        "cum4_lag_gen": lag_fourth_cumulant(rg), "cum4_lag_ref": lag_fourth_cumulant(rr),
        "hist_kl_resid": histogram_kl(rr.ravel(), rg.ravel()),
        "var_gen": float(G.var()), "var_ref": float(R.var()),
    }
    llg = log_likelihood_true(G, grid, weights, log_k_true, t_min)
    llr = log_likelihood_true(R, grid, weights, log_k_true, t_min)
    out.update({"loglik_gen": float(llg.mean()), "loglik_ref": float(llr.mean()),
                "loglik_gap": float(llr.mean() - llg.mean()),
                "loglik_gen_se": float(llg.std(ddof=1) / np.sqrt(len(llg))), "loglik_ref_se": float(llr.std(ddof=1) / np.sqrt(len(llr)))})
    return out
