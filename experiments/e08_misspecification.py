#!/usr/bin/env python
"""E8 — controlled misspecification: the advantage of the chain estimator versus a
measurable departure from the chain, against a locality-aware learned baseline.

Truths (all with exact reference scores):
    global    Gaussian AR(1) plus a rank-one global latent of strength beta
              (markov_score.priors.GaussianAR1PlusGlobal); exact score by linear algebra
    longrange Gaussian AR(1) plus long-range precision coupling gamma
              (markov_score.priors.GaussianLongRange); exact score by linear algebra
    ar2       Gaussian AR(2): a_i = rho1 a_{i-1} + rho2 a_{i-2} + eps, parameterised by the
              second coefficient rho2 at fixed lag-one correlation; exact score by
              linear algebra from its covariance
    sitevar   Gaussian AR(1) with site-dependent rho_i = rho + delta * cos(2 pi i / L);
              exact score by linear algebra
Every truth is Gaussian so that its score is exact without a grid; the misspecification
parameter eps is the strength, and eps = 0 is the well-specified chain. (A Laplace
version of the global-latent truth exists in the archival record via Gauss-Hermite
marginalisation; it is not needed to locate the boundary and is left out.)

Arms: EM-BP with the mixture-innovation AR(1) kernel (the structured estimator, now
misspecified), checkpoint selected on validation; the window head (torch, E03 recipe,
hyperparameters given by --set window_hp or the E03 winner); the LMMSE oracle of the
misspecified AR(1) covariance (the best a Gaussian AR(1) can do); the isotropic score.
Estimand: schedule-pooled relative score error against the exact score, all sites;
sizes (512, 2048) to separate the statistical gain (small M) from the asymptotic bias
(large M); 8 seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from protocol import PROTOCOL  # noqa: E402
from markov_score import theory  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.em import fit_em  # noqa: E402
from markov_score.kernels import MixtureInnovationKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import GaussianAR1PlusGlobal, GaussianLongRange  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

N_SITES = FROZEN.n_sites
T_SCHEDULE = tuple(FROZEN.t_grid)
BULK = slice(4, N_SITES - 4)
REGIONS = {"all": slice(0, N_SITES), "bulk": BULK}

SETTINGS = dict(
    mechanisms={"global": (0.0, 0.1, 0.25, 0.5, 1.0), "longrange": (0.0, 0.05, 0.1, 0.2, 0.4),
                "ar2": (0.0, 0.1, 0.2, 0.3, 0.4), "sitevar": (0.0, 0.05, 0.1, 0.14)},
    sizes=(512, 2048), seeds=tuple(range(8)),
    n_val=512, n_test=1024, grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    em_components=8, em_checkpoints=PROTOCOL.em_checkpoints, em_init_rho=PROTOCOL.em_init_rho, em_init_var=PROTOCOL.em_init_var,
    window_hp=dict(radius=2, width=64), window_lr=1e-3, window_mode="eps", window_steps=16_000,
    window_checkpoints=(500, 1000, 2000, 4000, 8000, 16000), batch_size=64, torch_threads=1,
    arms=("em_bp", "window", "lmmse_ar1", "isotropic"),
)
QUICK = dict(mechanisms={"global": (0.0, 1.0)}, sizes=(64,), seeds=(10_000,), n_val=32, n_test=32, grid_size=201,
             em_checkpoints=(5, 10), window_steps=100, window_checkpoints=(50, 100))


# ------------------------------------------------------------------ Gaussian truths with exact scores


class GaussianCov:
    """A zero-mean Gaussian prior given by its covariance; exact posterior mean by linear algebra."""

    def __init__(self, sigma0):
        self.sigma0 = np.asarray(sigma0, float)
        self._chol = np.linalg.cholesky(self.sigma0 + 1e-12 * np.eye(len(sigma0)))

    def sample(self, rng, n):
        return (self._chol @ rng.standard_normal((len(self.sigma0), n))).T

    def posterior_mean(self, X, t):
        return theory.lmmse_posterior_mean(X, self.sigma0, t)


def _ar2_cov(rho1, rho2, L):
    """Covariance of a stationary AR(2) with unit variance, by the Yule–Walker recursion."""
    g = np.zeros(L)
    g[0] = 1.0
    g[1] = rho1 / (1 - rho2)
    for k in range(2, L):
        g[k] = rho1 * g[k - 1] + rho2 * g[k - 2]
    return g[np.abs(np.subtract.outer(np.arange(L), np.arange(L)))]


def _sitevar_cov(rho, delta, L):
    """Cov of a_1 ~ N(0,1), a_i = rho_i a_{i-1} + eps_i with Var eps_i = 1 - rho_i^2 (unit variances).

    The construction needs every site coefficient inside (-1, 1): at rho = 0.85 a strength above
    0.15 pushes rho_k past 1, the innovation variance 1 - rho_k^2 turns negative and the matrix is
    not a covariance at all. That is a property of the mechanism, not of a particular draw, so it
    is refused here rather than surfacing later as a Cholesky failure (which is how the
    delta = 0.2 cells of 15 September died)."""
    rhos = rho + delta * np.cos(2 * np.pi * np.arange(L) / L)
    if np.abs(rhos).max() >= 1.0:
        raise ValueError(f"sitevar strength {delta} at rho {rho} gives |rho_k| = {np.abs(rhos).max():.3f} >= 1: "
                         "not a valid chain (innovation variance would be negative)")
    # Cov(a_i, a_j) = prod_{k=i+1}^{j} rho_k  (unit variances at every site)
    S = np.eye(L)
    for i in range(L):
        for j in range(i + 1, L):
            S[i, j] = S[j, i] = np.prod(rhos[i + 1:j + 1])
    return S


def _truth(mech, eps, L, rho):
    if mech == "global":
        return GaussianCov(GaussianAR1PlusGlobal(rho, eps).covariance(L))
    if mech == "longrange":
        return GaussianCov(GaussianLongRange(rho, eps).covariance(L))
    if mech == "ar2":
        # keep the lag-one correlation at rho: rho1 = rho (1 - rho2)
        rho1 = rho * (1 - eps)
        return GaussianCov(_ar2_cov(rho1, eps, L))
    if mech == "sitevar":
        return GaussianCov(_sitevar_cov(rho, eps, L))
    raise ValueError(mech)


# ------------------------------------------------------------------ scoring


def _sq(m_hat, m_ref, X, t, sl):
    al, de = alpha_delta(t)
    s_hat = -(X[:, sl] - al * m_hat[:, sl]) / de
    s_ref = -(X[:, sl] - al * m_ref[:, sl]) / de
    return float(((s_hat - s_ref) ** 2).sum()), float((s_ref**2).sum())


def _risk(pairs):
    num = sum(e for e, _ in pairs); den = sum(r for _, r in pairs)
    return float(np.sqrt(num / den)) if den > 0 else float("nan")


def _bundle(truth, tag, seed, n):
    rng = rng_for(tag, seed)
    A = truth.sample(rng, n)
    out = {}
    for t in T_SCHEDULE:
        al, de = alpha_delta(t)
        X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
        out[t] = (X, truth.posterior_mean(X, t))
    return A, out


def _cell(args):
    cfg, mech, eps, seed, n = args
    L, rho = N_SITES, FROZEN.rho
    truth = _truth(mech, eps, L, rho)
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    A = truth.sample(rng_for(("e08-train", mech, eps), seed, n), n)
    _, val = _bundle(truth, ("e08-val", mech, eps), seed, cfg["n_val"])
    _, test = _bundle(truth, ("e08-test", mech, eps), seed, cfg["n_test"])
    base = dict(mechanism=mech, eps=eps, seed=seed, n_chains=n)
    rows = []

    DEFAULTS = {"checkpoint": -1, "at_cap": False, "fit_rho": np.nan, "fit_kurt": np.nan, "n_params": 0}

    def score_rows(arm, mean_fn, extra=None):
        for region, sl in REGIONS.items():
            pairs = [_sq(mean_fn(test[t][0], t), test[t][1], test[t][0], t, sl) for t in T_SCHEDULE]
            rows.append({**base, "arm": arm, "region": region, "sq_err": sum(e for e, _ in pairs), "sq_ref": sum(r for _, r in pairs),
                         "risk": _risk(pairs), **DEFAULTS, **(extra or {})})

    # oracles of the wrong model: AR(1) covariance with the truth's lag-one correlation, unit variance
    lag1 = float(np.mean(np.diag(truth.sigma0, 1)))
    sig_ar1 = lag1 ** np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
    if "lmmse_ar1" in cfg["arms"]:
        score_rows("lmmse_ar1", lambda X, t: theory.lmmse_posterior_mean(X, sig_ar1, t), {"fit_rho": lag1})
    if "isotropic" in cfg["arms"]:
        score_rows("isotropic", lambda X, t: (X - (1 - np.exp(-2 * t)) * X) / np.exp(-t))
    if "em_bp" in cfg["arms"]:
        idx = np.array_split(rng_for(("e08-split", mech, eps), seed).permutation(n), len(T_SCHEDULE))
        noise = rng_for(("e08-noise", mech, eps), seed)
        groups = []
        for t, ix in zip(T_SCHEDULE, idx):
            al, de = alpha_delta(t)
            sub = A[ix]; groups.append((al * sub + np.sqrt(de) * noise.standard_normal(sub.shape), al, de))
        ck = tuple(int(c) for c in cfg["em_checkpoints"])
        k0 = MixtureInnovationKernel.init(cfg["em_components"], rho=cfg["em_init_rho"], var=cfg["em_init_var"], rng=rng_for(("e08-init", mech, eps), seed))
        _, trace, saved = fit_em(k0, grid, weights, groups, n_iters=max(ck), tol=FROZEN.em_loglik_tol, checkpoints=set(ck))
        best, best_r = None, np.inf
        for it in sorted(saved):
            lk = saved[it].log_transition_matrix(grid)
            pairs = []
            for t in T_SCHEDULE:
                X, m_ref = val[t]; al, de = alpha_delta(t)
                m, _ = grid_bp_batch(grid, weights, lk, X, al, de); pairs.append(_sq(m, m_ref, X, t, BULK))
            r = _risk(pairs)
            if r < best_r:
                best, best_r = it, r
        kb = saved[best]; lk = kb.log_transition_matrix(grid)
        cache = {}
        def em_mean(X, t):
            if t not in cache:
                al, de = alpha_delta(t); cache[t], _ = grid_bp_batch(grid, weights, lk, X, al, de)
            return cache[t]
        score_rows("em_bp", em_mean, {"checkpoint": best, "at_cap": best == max(saved), "fit_rho": float(kb.rho), "fit_kurt": float(kb.innovation_moments["innovation_excess_kurtosis"])})
    if "window" in cfg["arms"]:
        import torch
        from markov_score import nn_torch as nt
        torch.set_num_threads(cfg["torch_threads"])
        ck = tuple(int(c) for c in cfg["window_checkpoints"])
        model = nt.build("window", L, **cfg["window_hp"])
        tcfg = nt.TrainConfig(steps=max(ck), batch_size=cfg["batch_size"], lr=cfg["window_lr"], parameterization=cfg["window_mode"], checkpoints=ck, t_levels=T_SCHEDULE,
                              seed=int(rng_for(("e08-win", mech, eps), seed).integers(0, 2**31 - 1)))
        saved = nt.train(model, A, tcfg)
        best, best_r = None, np.inf
        for step in sorted(saved):
            den = nt.Denoiser(nt.load_checkpoint(model, saved[step]), cfg["window_mode"])
            r = _risk([_sq(den.posterior_mean(val[t][0], t), val[t][1], val[t][0], t, BULK) for t in T_SCHEDULE])
            if r < best_r:
                best, best_r = step, r
        den = nt.Denoiser(nt.load_checkpoint(model, saved[best]), cfg["window_mode"])
        score_rows("window", den.posterior_mean, {"checkpoint": best, "at_cap": best == max(saved), "n_params": nt.n_params(model)})
    r_all = {r["arm"]: r["risk"] for r in rows if r["region"] == "all"}
    print(f"  [E08] {mech} eps={eps} seed={seed} n={n}: " + " ".join(f"{k} {v:.4f}" for k, v in r_all.items()), flush=True)
    return rows


def _run(cells, fn, workers, out=None, csv_name=None):
    """Run cells serially or in a spawn pool; each cell's rows are written by the worker to
    out/cells/<csv stem>/<key>.json and the CSV is rebuilt after every completion
    (experiments/common.py: run_cells; resumable, no result travels through the pool pipe)."""
    return run_cells(cells, fn, workers, out, csv_name)


def part_sweep(cfg, out, workers):
    cells = [(cfg, m, e, s, n) for m, epss in cfg["mechanisms"].items() for e in epss for s in cfg["seeds"] for n in cfg["sizes"]]
    print(f"  {len(cells)} cells", flush=True)
    rows = _run(cells, _cell, workers, out, "misspecification.csv")
    write_csv(out / "misspecification.csv", rows)


PARTS = {"sweep": part_sweep}


def main():
    ap = experiment_parser("e08_misspecification", "advantage versus a measurable departure from the chain")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.list_parts:
        print("\n".join(PARTS)); return
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    out = ensure_dir(Path(args.output_dir))
    for name, fn in select_parts(PARTS, args.only).items():
        print(f"[e08] part {name}, workers={args.workers}", flush=True)
        fn(cfg, out, args.workers)
        write_json(out / f"params_{name}.json", {**provenance(), "settings": {k: (v if not isinstance(v, dict) else str(v)) for k, v in cfg.items()},
                                                   "quick": bool(args.quick), "workers": args.workers, "frozen": FROZEN.as_dict()})


if __name__ == "__main__":
    main()
