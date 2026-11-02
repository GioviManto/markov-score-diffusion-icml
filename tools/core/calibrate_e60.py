"""E60's calibration, from the TRUE kernels only: no training data, no noise, no E33 fit is read.

For each of the four slide-15 cells, the best member of each family is fitted to the exact pair law
P(a, a') = mu(a) K(a' | a) on E33's grid by maximising sum P log K_theta, the population M-step:

    free C   C free 2-D Gaussians on the pair (a, a'), read as K(a' | a) = p(a, a') / p(a); C = 2, 4, 8
    shift    the thesis mixture, a' = rho a + eps with eps a C = 8 mixture (MixtureInnovationKernel's own M-step)

Each best member is scored two ways: the KL per edge from the true kernel, and the BP score error on E33's seed-0 test
bundle divided by the linear floor's (the unit of the slide-15 table). A best member bounds what EM from noisy data can
reach; it says nothing about how close EM gets.

    python tools/core/calibrate_e60.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
import e33_field_learning as e33  # noqa: E402
from markov_score import theory  # noqa: E402
from markov_score.bp_grid import grid_bp_batch  # noqa: E402
from markov_score.em import ExpectedStatistics  # noqa: E402
from markov_score.kernels import MixtureInnovationKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.utils import rng_for  # noqa: E402

CELLS = ("T0.3_h1_L32", "T0.5_iso_L32", "T1_iso_L32", "T2_iso_L32")
FREE_C = (2, 4, 8)
N_EDGES = 10**6
LOG_2PI = float(np.log(2 * np.pi))


class Fixed:
    """A log-transition matrix posing as a kernel, for E33's invariant-law helper."""

    def __init__(self, name, lk):
        self.name, self.lk = name, lk

    def log_transition_matrix(self, grid):
        return self.lk


def column_normalised(lk, w):
    z = lk + np.log(w)[:, None]
    return lk - np.logaddexp.reduce(z, axis=0)[None, :]


def free_log_k(logit, mu, chol, g):
    """log K(a' | a) of C free 2-D Gaussians; rows a' (out), columns a (in). chol = (log l11, l21, log l22)."""
    l11, l21, l22 = chol[:, 0].exp(), chol[:, 1], chol[:, 2].exp()
    s11 = l11 ** 2                                           # variance of a in component c
    a, ap = g[None, None, :], g[None, :, None]
    logw = logit - torch.logsumexp(logit, 0)
    gate = logw[:, None, None] - 0.5 * (a - mu[:, 0, None, None]) ** 2 / s11[:, None, None] - 0.5 * (LOG_2PI + s11.log())[:, None, None]
    m = mu[:, 1, None, None] + (l21 / l11)[:, None, None] * (a - mu[:, 0, None, None])   # conditional mean of a'
    v = l22 ** 2                                              # conditional variance of a'
    comp = -0.5 * (ap - m) ** 2 / v[:, None, None] - 0.5 * (LOG_2PI + v.log())[:, None, None]
    return torch.logsumexp(gate + comp, 0) - torch.logsumexp(gate, 0)


def weighted_gmm(pts, p, C, rng, iters=200):
    """Closed-form EM of a 2-D Gaussian mixture on weighted points: the starting point of the conditional fit."""
    mu = pts[rng.choice(len(pts), C, replace=False, p=p / p.sum())]
    S = np.repeat(np.cov(pts.T, aweights=p)[None] / C, C, 0)
    w = np.full(C, 1.0 / C)
    for _ in range(iters):
        d = pts[None] - mu[:, None]
        Si = np.linalg.inv(S)
        ll = -0.5 * np.einsum("cni,cij,cnj->cn", d, Si, d) - 0.5 * np.log(np.linalg.det(S))[:, None] + np.log(w)[:, None]
        r = np.exp(ll - np.logaddexp.reduce(ll, 0)) * p[None]
        nk = r.sum(1)
        w = nk / nk.sum()
        mu = (r @ pts) / nk[:, None]
        d = pts[None] - mu[:, None]
        S = np.einsum("cn,cni,cnj->cij", r, d, d) / nk[:, None, None] + 1e-6 * np.eye(2)
    return w, mu, S


def fit_free(P, grid, w, C, seed, restarts=3):
    """Best member of the free family: weighted 2-D GMM EM, then L-BFGS on sum P log K (column-normalised on the grid)."""
    rng = rng_for("e60-calibration", C, seed)
    jj, kk = np.nonzero(P.T > 1e-12 * P.max())
    pts, p = np.stack([grid[jj], grid[kk]], 1), P[kk, jj]
    g, Pt, lw = (torch.as_tensor(x) for x in (grid, P, np.log(w)))
    best = (-np.inf, None)
    for _ in range(restarts):
        w0, mu0, S0 = weighted_gmm(pts, p, C, rng)
        L0 = np.linalg.cholesky(S0)
        theta = [torch.tensor(np.log(w0), requires_grad=True), torch.tensor(mu0, requires_grad=True),
                 torch.tensor(np.stack([np.log(L0[:, 0, 0]), L0[:, 1, 0], np.log(L0[:, 1, 1])], 1), requires_grad=True)]
        opt = torch.optim.LBFGS(theta, max_iter=400, line_search_fn="strong_wolfe", tolerance_grad=1e-10, tolerance_change=1e-14)

        def objective():
            lk = free_log_k(*theta, g)
            return (Pt * (lk - torch.logsumexp(lk + lw[:, None], 0)[None, :])).sum()

        def closure():
            opt.zero_grad()
            loss = -objective()
            loss.backward()
            return loss
        for _ in range(5):
            opt.step(closure)
        with torch.no_grad():
            q = float(objective())
        if q > best[0]:
            best = (q, [t.detach().clone() for t in theta])
    with torch.no_grad():
        lk = free_log_k(*best[1], g).numpy()
    logit = best[1][0].numpy()
    pi = np.exp(logit - np.logaddexp.reduce(logit))
    return lk, dict(eff_components=float(np.exp(-(pi * np.log(pi)).sum())), slopes=(best[1][2][:, 1] / best[1][2][:, 0].exp()).numpy())


def fit_shift(P, grid, cell):
    stats = ExpectedStatistics(xi=P * N_EDGES, site1=P.sum(0), log_evidence=0.0, n_edges=N_EDGES, n_chains=1)
    k = MixtureInnovationKernel.init(8, rho=0.3, var=0.8, rng=rng_for("e60-calibration-shift", cell))
    for _ in range(400):
        k = k.m_step(stats, grid)
    return k.log_transition_matrix(grid), dict(rho=float(k.rho))


def main() -> int:
    cfg = dict(e33.SETTINGS)
    print("best member per family, fitted to the exact pair law; error = BP score error / linear floor (seed-0 test bundle)")
    print(f"{'cell':14s} {'family':8s} {'KL/edge':>10s} {'error':>7s}  notes")
    for cell in CELLS:
        chain, grid, w, log_k, log_mu, sigma0, _, _, _, test, _ = e33._common(cfg, cell, 0, 32)
        lk_true = column_normalised(log_k, w)
        mu_true = np.exp(log_mu) * w
        P = np.exp(lk_true + np.log(w)[:, None]) * (mu_true / mu_true.sum())[None, :]
        P /= P.sum()
        floor = e33.e14._pooled(lambda X, t: theory.lmmse_posterior_mean(X, sigma0, t), test)
        mu_fn = e33._stationary_mu(grid, w)

        def report(name, lk, notes):
            lk = column_normalised(lk, w)
            on = P > 0                                    # the true kernel underflows to log 0 where P is 0
            kl = float((P[on] * (lk_true[on] - lk[on])).sum())
            mu = mu_fn(Fixed(name, lk))
            err = e33.e14._pooled(lambda X, t: grid_bp_batch(grid, w, lk, X, *alpha_delta(t), log_mu=mu)[0], test)
            print(f"{cell:14s} {name:8s} {kl:10.2e} {err / floor:7.3f}  {notes}", flush=True)

        lk, d = fit_shift(P, grid, cell)
        report("shift8", lk, f"rho {d['rho']:.3f}")
        for C in FREE_C:
            lk, d = fit_free(P, grid, w, C, 0)
            report(f"free{C}", lk, f"effective components {d['eff_components']:.2f}, slopes {np.round(np.sort(d['slopes']), 2).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
