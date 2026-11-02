"""E62's calibration, from the TRUE kernels only: no training data, no noisy view, no fit to data.

    python tools/core/calibrate_e62.py

E62 asks how long first-order EM-BP stays good when the data are second order. The chain is an AR(2) with Laplace
innovations, a_t = phi1 a_{t-1} + phi2 a_{t-2} + eps, with the lag-one correlation held at E03's 0.85 and the variance
at 1 (Yule-Walker: phi1 = 0.85 (1 - phi2), Var eps = (1 - 0.85^2)(1 - phi2^2)), so a first-order method sees the same
lag-one covariance at every phi2 and only the memory beyond one step changes. Started from its stationary law.

For each phi2:
  delta_gauss  the Gaussian KL rate to the best first-order chain, -(L-2)/(2L) log(1 - phi2^2) (the partial
               autocorrelation at lag 2 is phi2; the trace term of the Gaussian KL is exactly zero)
  delta_exact  the exact KL rate of the Laplace AR(2) to its best first-order chain, (L-2)/L I(a_t; a_{t-2} | a_{t-1}),
               on the pair grid
  bias         the BP score error of the best first-order kernel (the stationary pair law's conditional, which EM's
               M-projection approaches as M grows) against the pair-state reference: first-order EM-BP's floor
  linear       the LMMSE of the true AR(2) covariance
on a calibration-only bundle (tag e62-calibration, 256 chains, L = 32, E03's twelve levels); errors in E03's estimand.
The reference is pair-state BP (markov_score.pair_bp) at n = 101 on [-8, 8], checked against n = 151 and 201.

Last, wind: the partial autocorrelations of E59's transformed Horns Rev series (training years) and its delta_gauss
from them, -1/2 sum_{j = 2}^{24} log(1 - pi_j^2): where a real series sits on E62's axis.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
import e03_baselines as e03  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.pair_bp import pair_bp_batch  # noqa: E402
from markov_score.utils import rng_for  # noqa: E402

RHO1, L = 0.85, 32
PHI2 = (-0.3, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
T_SCHEDULE = e03.T_SCHEDULE


def coefficients(phi2):
    return RHO1 * (1 - phi2), (1 - RHO1 ** 2) * (1 - phi2 ** 2)


def laplace_k2(grid, w, phi2):
    """log K2[k, j, i] = log Laplace(u_k - phi1 u_j - phi2 u_i; b), normalised over k on the grid."""
    phi1, s2 = coefficients(phi2)
    b = np.sqrt(s2 / 2)
    lk = -np.abs(grid[:, None, None] - phi1 * grid[None, :, None] - phi2 * grid[None, None, :]) / b
    return lk - np.log(np.einsum("kji,k->ji", np.exp(lk), w))[None]


def stationary_pair(lk2, w, iters=3000, tol=1e-14):
    """mu2[j, i] = p(a_t = u_j, a_{t-1} = u_i), the fixed point of the pair operator (w (x) w normalised)."""
    k2 = np.exp(lk2)
    n = len(w)
    f = np.full((n, n), 1.0)                                   # f[i, j] = p(a_{t-1} = i, a_t = j)
    for _ in range(iters):
        g = np.einsum("ij,kji,i->jk", f, k2, w)
        g /= np.sum(g * w[:, None] * w[None, :])
        if np.max(np.abs(g - f)) < tol:
            f = g
            break
        f = g
    return f.T                                                 # [j = a_t, i = a_{t-1}]


def sample(phi2, rng, n_chains, burn=600):
    phi1, s2 = coefficients(phi2)
    b = np.sqrt(s2 / 2)
    a = np.zeros((n_chains, burn + L))
    e = rng.laplace(0.0, b, size=a.shape)
    for t in range(2, burn + L):
        a[:, t] = phi1 * a[:, t - 1] + phi2 * a[:, t - 2] + e[:, t]
    return a[:, burn:]


def ar2_cov(phi2):
    phi1, _ = coefficients(phi2)
    rho = [1.0, RHO1]
    for _ in range(2, L):
        rho.append(phi1 * rho[-1] + phi2 * rho[-2])
    i = np.arange(L)
    return np.array(rho)[np.abs(i[:, None] - i[None, :])]


def cmi(lk2, mu2, w):
    """I(a_t; a_{t-2} | a_{t-1}) under the stationary law, in nats."""
    k2 = np.exp(lk2)                                           # [k, j, i]
    p3 = k2 * mu2[None, :, :] * w[:, None, None] * w[None, :, None] * w[None, None, :]   # mass of (a_t, a_{t-1}, a_{t-2})
    p3 /= p3.sum()
    pji = p3.sum(0)                                             # (a_{t-1}, a_{t-2})
    pkj = p3.sum(2)                                             # (a_t, a_{t-1})
    pj = p3.sum((0, 2))
    on = p3 > 0
    num = p3 * pj[None, :, None]
    den = pkj[:, :, None] * pji[None, :, :]
    return float(np.sum(p3[on] * np.log(num[on] / den[on])))


def scores(grid, w, lk2, mu2, phi2, A, rng):
    """(sq_err, sq_ref) of the first-order projection and of the linear floor against the pair reference, all levels."""
    mu1 = (mu2 * w[:, None]).sum(0)                             # p(a_{t-1} = u_i) = sum_j w_j mu2[j, i]
    lk1 = np.log(np.maximum(mu2, 1e-300)) - np.log(np.maximum(mu1, 1e-300))[None, :]      # p(a_t | a_{t-1})
    lk1 -= np.log(np.einsum("kj,k->j", np.exp(lk1), w))[None, :]
    cov = ar2_cov(phi2)
    out = {"bias": [], "linear": []}
    for t in T_SCHEDULE:
        al, de = alpha_delta(t)
        X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
        ref, _ = pair_bp_batch(grid, w, lk2, X, al, de, np.log(np.maximum(mu2, 1e-300)))
        m1, _ = grid_bp_batch(grid, w, lk1, X, al, de, log_mu=np.log(np.maximum(mu1, 1e-300)))
        ml = X @ (al * cov @ np.linalg.inv(al ** 2 * cov + de * np.eye(L))).T
        for key, m in (("bias", m1), ("linear", ml)):
            out[key].append(e03._sq(m, ref, X, t, e03.REGIONS["all"]))
    return {k: e03._risk(v) for k, v in out.items()}


def refinement(phi2):
    """Relative score difference of the pair reference between n = 101 and n = 151, 201 (64 chains, three levels)."""
    rng = rng_for("e62-calibration-refine", phi2)
    A = sample(phi2, rng, 64)
    res = {}
    means = {}
    for n in (101, 151, 201):
        grid, w = make_grid(8.0, n)
        lk2 = laplace_k2(grid, w, phi2)
        mu2 = stationary_pair(lk2, w)
        means[n] = []
        noise = rng_for("e62-calibration-refine-noise", phi2)
        for t in (T_SCHEDULE[0], T_SCHEDULE[len(T_SCHEDULE) // 2], T_SCHEDULE[-1]):
            al, de = alpha_delta(t)
            X = al * A + np.sqrt(de) * noise.standard_normal(A.shape)
            means[n].append((X, al, de, pair_bp_batch(grid, w, lk2, X, al, de, np.log(np.maximum(mu2, 1e-300)))[0]))
    for n in (101, 151):
        num = sum(((m - m2) ** 2).sum() * (al / de) ** 2 for (X, al, de, m), (_, _, _, m2) in zip(means[n], means[201]))
        den = sum((((X - al * m2) / de) ** 2).sum() for X, al, de, m2 in means[201])
        res[n] = float(np.sqrt(num / den))
    return res


def wind_delta(max_lag=24):
    import e59_forecasting as e59
    z, split = e59.wind_series()
    x = z[split == 0]
    x = x - x.mean()
    acf = np.array([np.dot(x[:-k or None], x[k:]) / np.dot(x, x) for k in range(max_lag + 1)])
    pacf = [1.0]
    phi = np.zeros(0)
    for k in range(1, max_lag + 1):                             # Durbin-Levinson
        num = acf[k] - np.dot(phi, acf[1:k][::-1])
        den = 1 - np.dot(phi, acf[1:k])
        pk = num / den
        phi = np.concatenate([phi - pk * phi[::-1], [pk]])
        pacf.append(pk)
    pacf = np.array(pacf)
    delta = -0.5 * np.sum(np.log(1 - pacf[2:] ** 2))
    return acf, pacf, delta


def main() -> int:
    grid, w = make_grid(8.0, 101)
    print(f"{'phi2':>5s} {'phi1':>6s} {'rho2':>6s} {'delta_gauss':>11s} {'delta_exact':>11s} {'bias':>8s} {'linear':>8s}")
    for phi2 in PHI2:
        lk2 = laplace_k2(grid, w, phi2)
        mu2 = stationary_pair(lk2, w)
        dg = -(L - 2) / (2 * L) * np.log(1 - phi2 ** 2)
        de = (L - 2) / L * cmi(lk2, mu2, w)
        rng = rng_for("e62-calibration", phi2)
        A = sample(phi2, rng, 256)
        s = scores(grid, w, lk2, mu2, phi2, A, rng)
        print(f"{phi2:5.2f} {coefficients(phi2)[0]:6.3f} {ar2_cov(phi2)[0, 2]:6.4f} {dg:11.5f} {de:11.5f} "
              f"{s['bias']:8.4f} {s['linear']:8.4f}", flush=True)
    print("reference grid refinement (relative score difference to n = 201):")
    for phi2 in (0.0, 0.4, 0.6):
        r = refinement(phi2)
        print(f"  phi2 {phi2:.1f}: n = 101 {r[101]:.2e}, n = 151 {r[151]:.2e}", flush=True)
    acf, pacf, delta = wind_delta()
    print(f"wind (E59's transformed Horns Rev series, training years): lag-1 corr {acf[1]:.4f}, "
          f"partial autocorrelations 2-6 {np.round(pacf[2:7], 4).tolist()}, delta_gauss (lags 2-24) {delta:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
