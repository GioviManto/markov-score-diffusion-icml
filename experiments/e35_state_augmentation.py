#!/usr/bin/env python
"""E35 -- a global latent and its repair by state augmentation (the consolidated note, section 8).

Truth: X_i = G b + U_i with G in {-1, +1} a fair coin per sequence and U the Gaussian AR(1) chain
at the laboratory's correlation rho = 0.85 (unit variance). Conditioned on G the sequence is
first-order Markov; marginalised over G it is not. The exact score is in closed form (a mixture
of two Gaussian chains, `augmented.two_phase_gaussian_posterior_mean`), so no grid enters the
reference. b = 0 is the control (no latent); at b = 2 the marginal of X is bimodal and the
whole sequence sits in one phase.

Arms, per (b, M, seed), on the registered interface (one noised view per sequence spread over
the twelve levels; validation 1024 and test 2048 chains; ladder and risk selection as E33):
    em_first_order    the paper's estimator: EM through exact BP around an innovation-mixture
                      kernel (C = 8) with its own invariant law. MISSPECIFIED for b > 0: a
                      first-order kernel with a state-independent innovation cannot hold a
                      global phase, so it converges to the KL-closest first-order law.
    em_augmented      the repair: the same kernel family on the AUGMENTED state (X, G), i.e. a
                      mixture of two chains tied by reflection (K_- = K_+ reflected), the mixture
                      weight inferred per sequence from the evidence; EM by
                      `augmented.fit_em_augmented`. Exactly specified at every b, with the SAME
                      parameter count as the first-order arm.
    lmmse_true        the linear floor of the true covariance b^2 1 1^T + Sigma_U (the covariance
                      twin; this is E08's rank-one latent in second-order terms).
    isotropic         the structure-free map.
The exact score (closed form) is what every arm is scored against; its own risk is zero.

The window head is not run here: E08 already measured it against the first-order arm on the
Gaussian rank-one latent, and this experiment is about the model class, not the network.

PREDECLARED (registry E35.json, written before any cell ran):
  P1  at b = 0 the augmented arm costs nothing: its risk is within 10% of the first-order
      arm's at every M (paired over seeds).
  P2  the first-order arm's deficit is a bias, not a data problem: at b = 2 its risk changes by
      less than 15% from M = 512 to M = 2048, while the augmented arm's falls by more than 30%.
  P3  the ratio first-order / augmented at M = 2048 rises monotonically with b (0, 1, 2) and
      exceeds 2 at b = 2 in every seed; at b = 2 the augmented arm is below the linear floor by
      a factor of at least 3 at M = 2048.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score import theory  # noqa: E402
from markov_score.augmented import augmented_posterior, fit_em_augmented, two_phase_gaussian_posterior_mean  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.em import fit_em  # noqa: E402
from markov_score.kernels import MixtureInnovationKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import GaussianAR1  # noqa: E402
from markov_score.stationary import invariant_log_density_eig  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402
from protocol import PROTOCOL  # noqa: E402

import e14_regime_baselines as e14  # noqa: E402

T_SCHEDULE = tuple(FROZEN.t_grid)
RHO = FROZEN.rho

SETTINGS = dict(
    offsets=(0.0, 1.0, 2.0), sizes=(512, 2048), seeds=(0, 1), n_sites=32,
    n_val=PROTOCOL.n_val, n_test=PROTOCOL.n_test,
    grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    components=8, em_checkpoints=(10, 20, 40, 60, 80, 120, 160, 220, 300, 400),
    em_init_rho=PROTOCOL.em_init_rho, em_init_var=PROTOCOL.em_init_var,
    arms=("em_first_order", "em_augmented"),
)
QUICK = dict(offsets=(1.5,), sizes=(32,), seeds=(10_000,), n_sites=12, n_val=32, n_test=32, grid_size=201,
             components=2, em_checkpoints=(5, 10))


def _two_phase(rng, n_chains, L, b):
    prior = GaussianAR1(RHO)
    U = np.stack([prior.sample(rng, L) for _ in range(n_chains)])
    G = rng.choice([-1.0, 1.0], size=n_chains)
    return G[:, None] * b + U


def _sigma_u(L):
    """Covariance of U, the chain WITHIN one phase: what the two-phase posterior needs."""
    i = np.arange(L)
    return RHO ** np.abs(i[:, None] - i[None, :])


def _sigma_true(L, b):
    """Covariance of X = G b 1 + U across phases: what the LMMSE (linear floor) needs."""
    return b**2 * np.ones((L, L)) + _sigma_u(L)


def _bundle(tag, seed, b, n, L):
    """Noised views with the CLOSED-FORM posterior mean of the truth as the reference.

    The reference takes the WITHIN-phase covariance. Until 2 Oct 2026 it was given the total
    covariance `_sigma_true`, which is the same thing only at b = 0: at b > 0 the "exact score"
    was wrong by up to 1.08 in the posterior mean (N-34; tests/test_augmented.py pins it)."""
    rng = rng_for(tag, seed, b, L)
    A = _two_phase(rng, n, L, b)
    sigma_u = _sigma_u(L)
    out = {}
    for t in T_SCHEDULE:
        al, de = alpha_delta(t)
        X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
        out[t] = (X, two_phase_gaussian_posterior_mean(X, sigma_u, b, t))
    return out


def _mu_fn(grid, w):
    def mu(kernel):
        return invariant_log_density_eig(kernel, grid, w).log_density
    return mu


def _cell(args):
    cfg, b, seed, n = args
    L = cfg["n_sites"]
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    val = _bundle("e35-val", seed, b, cfg["n_val"], L)
    test = _bundle("e35-test", seed, b, cfg["n_test"], L)
    A = _two_phase(rng_for("e35-train", seed, b, L, n), n, L, b)
    Av = _two_phase(rng_for("e35-valchains", seed, b, L), cfg["n_val"], L, b)
    groups = [(X, *alpha_delta(t)) for X, t in e14._one_view_groups(A, f"e35-em-b{b:g}-M{n}", seed, L)]
    held = [(X, *alpha_delta(t)) for X, t in e14._one_view_groups(Av, f"e35-emval-b{b:g}", seed, L)]
    sigma0 = _sigma_true(L, b)
    base = dict(b=b, rho=RHO, n_sites=L, seed=seed, n_chains=n,
                val_bundle=e14._bundle_hash(val), test_bundle=e14._bundle_hash(test))
    rows = [dict(base, arm="lmmse_true", checkpoint=-1, at_cap=0, val_selected=np.nan, wall=np.nan, n_params=0,
                 converged=False, stop_reason="", monotone_violation=np.nan, theta="",
                 risk=e14._pooled(lambda X, t: theory.lmmse_posterior_mean(X, sigma0, t), test)),
            dict(base, arm="isotropic", checkpoint=-1, at_cap=0, val_selected=np.nan, wall=np.nan, n_params=0,
                 converged=False, stop_reason="", monotone_violation=np.nan, theta="",
                 risk=e14._pooled(lambda X, t: (X - (1 - np.exp(-2 * t)) * X) / np.exp(-t), test))]
    mu_fn = _mu_fn(grid, w)
    ck = set(int(c) for c in cfg["em_checkpoints"])
    for arm in cfg["arms"]:
        k0 = MixtureInnovationKernel.init(cfg["components"], rho=cfg["em_init_rho"], var=cfg["em_init_var"],
                                          rng=rng_for("e35-init", arm, seed, b, n))
        t0 = time.time()
        if arm == "em_first_order":
            _, trace, saved = fit_em(k0, grid, w, groups, n_iters=max(ck), tol=FROZEN.em_loglik_tol, checkpoints=ck, log_mu=mu_fn)
            conv, reason, viol = bool(trace.converged), trace.stop_reason, float(trace.monotone_violation)

            def mean_of(kb, X, t):
                return grid_bp_batch(grid, w, kb.log_transition_matrix(grid), X, *alpha_delta(t), log_mu=mu_fn(kb))[0]
        else:
            _, trace, saved = fit_em_augmented(k0, grid, w, groups, n_iters=max(ck), log_mu=mu_fn, tol=FROZEN.em_loglik_tol, checkpoints=ck)
            conv, reason, viol = bool(trace["converged"]), trace["stop_reason"], float(trace["monotone_violation"])

            def mean_of(kb, X, t):
                return augmented_posterior(grid, w, kb.log_transition_matrix(grid), X, *alpha_delta(t), mu_fn(kb))[0]
        wall = time.time() - t0
        best, best_v = None, np.inf
        for it in sorted(saved):
            v = e14._pooled(lambda X, t, kb=saved[it]: mean_of(kb, X, t), val)
            if v < best_v:
                best, best_v = it, v
        kb = saved[best]
        risk = e14._pooled(lambda X, t: mean_of(kb, X, t), test)
        rows.append(dict(base, arm=arm, checkpoint=best, at_cap=int(best == max(saved)), val_selected=best_v, wall=wall,
                         n_params=int(len(np.asarray(kb.theta))), converged=conv, stop_reason=reason, monotone_violation=viol,
                         theta=";".join(f"{v:.8g}" for v in np.asarray(kb.theta, dtype=float)), risk=risk))
    print(f"  [E35] b={b:g} seed={seed} M={n}: " + " ".join(f"{r['arm']}={r['risk']:.4f}" for r in rows), flush=True)
    return rows


def part_augmented(cfg, out, workers):
    cells = [(cfg, b, s, n) for n in sorted(cfg["sizes"], reverse=True) for b in cfg["offsets"] for s in cfg["seeds"]]
    print(f"  augmented: {len(cells)} cells", flush=True)
    write_csv(out / "augmented.csv", run_cells(cells, _cell, workers, out, "augmented.csv",
                                               key=lambda c: f"b{c[1]:g}_s{c[2]}_M{c[3]}"))


PARTS = {"augmented": part_augmented}


def main():
    ap = experiment_parser("e35_state_augmentation", "a global sign latent: first-order EM-BP against the augmented state")
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
        fn(cfg, out, args.workers)
        write_json(out / f"params_{name}.json", {**provenance(), "settings": cfg, "quick": bool(args.quick), "workers": args.workers})


if __name__ == "__main__":
    main()
