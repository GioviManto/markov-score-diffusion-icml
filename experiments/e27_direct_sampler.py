#!/usr/bin/env python
"""E27 — the STRONG form of Q8: generation with no score and no diffusion.

Jerome, 25 September 2026 (transcript 01:19:08–01:19:20): "The exact score, or say the
exact BP, so not even a score, I think we don't even need the [diffusion] to generate.
We can just start from a flat marginal and run BP once."

E26 answered the WEAK form (does score accuracy transfer to generation?) by driving a
400-step reverse SDE with each arm's score. It did not test the claim Jerome made, which
is that the reverse SDE is unnecessary: on a chain, exact BP is a *sampler*, not only a
score oracle, because sum-product on a tree gives the exact conditional laws that
forward-filtering--backward-sampling needs. One sweep, exact draws, no score, no
integrator, no discretisation in t.

FOUR ARMS, all producing samples AT t_min so they are directly comparable with E26's
reference sample R (clean chain through the channel at t_min):

  ancestral_true  a_1 ~ mu, a_{i+1} ~ K_true(. | a_i) on the grid. Jerome's claim in its
                  purest form: the "flat marginal" limit, where the observation carries
                  no information and one forward pass IS the generator.
  ffbs_true       the faithful reading: draw x ~ p_{t_max} exactly as the reverse SDE's
                  initialisation does, run ONE BP forward filter conditioned on that x,
                  and sample backward. This is "run BP once".
  ancestral_em    the same forward pass with the EM-FITTED mixture kernel (E02a, the
                  validation-selected checkpoint E26's em_bp arm used). This is the arm
                  that MATTERS: it separates kernel error from integrator error, which
                  no experiment in this project has done. E26's em_bp reached excess
                  kurtosis 1.18 against a reference 1.87, and that number confounds
                  "the fitted kernel is wrong" with "400 steps is not enough".
  mc_floor        i.i.d. draws from the TRUE law in continuous space (the prior's own
                  recursion, no grid). Distance to R is pure Monte Carlo at this sample
                  size. E26 had no such arm, so its `exact` arm's kurtosis deficit
                  (1.66 vs 1.87) could not be split into bias and noise. Every other
                  arm's excess over this floor is the part that is real.

PREDECLARED, written before the script was executed (see experiments/registry/E27.json):
  P1  mc_floor and ancestral_true are indistinguishable on every metric: the grid at
      N_g = 401 is not what limits generation. Falsified if ancestral_true's
      hist_kl_resid exceeds mc_floor's by more than the seed-to-seed spread.
  P2  ancestral_true BEATS E26's 400-step `exact` arm on excess kurtosis, because the
      reverse SDE carries a discretisation bias that one exact sweep does not. E26's
      step study already showed a residual at 400 steps (strong_rmse 0.0033 against 800).
  P3  ffbs_true and ancestral_true agree: at t_max = 3 the observation is nearly
      uninformative (alpha = e^-3 = 0.0498), so conditioning on it changes nothing
      measurable. This is the quantitative content of "flat marginal".
  P4  ancestral_em reproduces E26's em_bp gap to within the seed spread. If instead it
      is MUCH closer to the reference than E26's em_bp was, then a material part of
      E26's em_bp deficit was the integrator and not the kernel, and E26's reading has
      to be weakened. This is the risky prediction.
  P5  cost: one sweep is O(L N_g^2) against the reverse SDE's n_steps * O(L N_g^2), so
      the speed-up is of order n_steps. Part `cost` measures it; part `steps` asks the
      converse -- how many reverse-SDE steps buy the accuracy one sweep gives for free.

QUANTISATION. A grid sampler draws grid POINTS, and a lattice-valued sample has a biased
kurtosis. The trapezoid rule defines a piecewise-constant density, and the correct
sampler for it resamples uniformly inside the chosen cell; `jitter` does that and is on
by default. `jitter=False` is kept as a control so the size of the artefact is reported
rather than assumed small.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from protocol import PROTOCOL  # noqa: E402
from markov_score import gen_metrics, theory  # noqa: E402
from markov_score.bp_grid import make_grid  # noqa: E402
from markov_score.kernels import MixtureInnovationKernel  # noqa: E402
from markov_score.noising import alpha_delta, log_likelihood_matrix  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.reverse import reverse_sde_with_increments, time_grid  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

N_SITES = FROZEN.n_sites

SETTINGS = dict(
    sizes=(128, 2048), components=8, seeds=PROTOCOL.seeds,
    n_generate=2000, n_reference=4000,
    t_max=3.0, t_min=0.02,
    steps_ladder=(25, 50, 100, 200, 400, 800),
    n_steps_reference=400,          # the resolution E26 reported, for the paired contrast
    grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    em_source="outputs/E02a/capacity.csv",
    jitter=True,
    cost_batch=(128, 512, 2048),
    cost_repeat=5,
)
QUICK = dict(
    sizes=(32,), seeds=(10_000,), n_generate=256, n_reference=512,
    grid_size=201, steps_ladder=(25, 100), cost_batch=(64,), cost_repeat=2,
    em_source="outputs/scratch/e02_quick/capacity.csv",
)


# --------------------------------------------------------------------------- samplers

def _sample_rows(P: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Inverse-CDF draw of one index per row of `P` (rows need not be normalised)."""
    c = np.cumsum(P, axis=1)
    c /= c[:, -1:]
    return np.array([np.searchsorted(c[b], u[b]) for b in range(len(u))])


def _dejitter(idx, grid, rng_jit, on):
    """Grid points, or a uniform draw inside the chosen cell of the piecewise-constant
    density the trapezoid rule defines.

    `rng_jit` is a SEPARATE stream from the one that draws the indices. That is what makes
    the jitter contrast paired: the jittered and unjittered rows of a cell come from
    identical grid indices, so their difference is the quantisation artefact and nothing
    else. Keying the index stream on `jitter` instead -- the first version of this script --
    gave two independent draws, and at n_generate = 256 the Monte Carlo scatter in excess
    kurtosis was 0.48, which is larger than any artefact it was meant to expose.
    """
    a = grid[idx]
    h = float(grid[1] - grid[0])
    u = rng_jit.random(a.shape)          # drawn either way, so the streams stay aligned
    return a + h * (u - 0.5) if on else a


def ancestral_grid(K, grid, weights, n, rng, rng_jit, *, jitter=True, log_mu=None):
    """a_1 ~ mu, a_{i+1} ~ K(. | a_i), entirely on the grid. ONE forward pass, no BP
    messages at all: with a flat likelihood the backward message of a normalised kernel
    is constant, so sum-product degenerates to exactly this."""
    m = len(grid)
    mu = np.exp(-0.5 * grid**2) if log_mu is None else np.exp(log_mu - log_mu.max())
    idx = _sample_rows(np.broadcast_to(mu * weights, (n, m)).copy(), rng.random(n))
    out = np.empty((n, N_SITES))
    out[:, 0] = _dejitter(idx, grid, rng_jit, jitter)
    for i in range(1, N_SITES):
        # K[out, in]: row `out` given column `in`, so the conditional on a_i = grid[j]
        # is the COLUMN K[:, j]. The index, not the jittered value, carries the chain
        # forward -- jitter must not change the conditioning, or the two rows of the
        # jitter contrast would no longer be the same draw.
        idx = _sample_rows(K[:, idx].T * weights[None, :], rng.random(n))
        out[:, i] = _dejitter(idx, grid, rng_jit, jitter)
    return out


def ffbs_grid(K, grid, weights, X, alpha, delta, rng, rng_jit, *, jitter=True, log_mu=None):
    """Forward filtering -- backward sampling on the chain posterior given ONE noisy
    observation X at (alpha, delta). Exact conditional draws; one forward sweep."""
    n, m = len(X), len(grid)
    log_mu = (-0.5 * grid**2 - 0.5 * np.log(2 * np.pi)) if log_mu is None else log_mu
    # Batched over sequences, one (n, M) @ (M, M) gemm per site, exactly as grid_bp_batch
    # does. The first version looped `for b in range(n)` in Python and came out 5x slower
    # than the ancestral arm -- which would have made the cost comparison in part `cost`
    # a statement about this loop rather than about the algorithm.
    z = X[:, :, None] - alpha * grid[None, None, :]
    log_ell = -0.5 * z**2 / delta
    ell = np.exp(log_ell - log_ell.max(axis=2, keepdims=True))     # (n, L, M)
    F = np.empty((n, N_SITES, m))
    f = np.exp(log_mu - log_mu.max())[None, :] * ell[:, 0]
    F[:, 0] = f / (f * weights).sum(axis=1, keepdims=True)
    for i in range(1, N_SITES):
        f = (F[:, i - 1] * weights) @ K.T * ell[:, i]
        F[:, i] = f / (f * weights).sum(axis=1, keepdims=True)
    out = np.empty((n, N_SITES))
    idx = _sample_rows(F[:, -1] * weights[None, :], rng.random(n))
    out[:, -1] = _dejitter(idx, grid, rng_jit, jitter)
    for i in range(N_SITES - 2, -1, -1):
        # p(a_i | a_{i+1}, x) ~ F_i(a_i) K(a_{i+1} | a_i): row K[idx_{i+1}, :].
        idx = _sample_rows(F[:, i] * K[idx, :] * weights[None, :], rng.random(len(X)))
        out[:, i] = _dejitter(idx, grid, rng_jit, jitter)
    return out


# --------------------------------------------------------------------------- plumbing

def _theta_from_e02(path, seed, n, C):
    df = pd.read_csv(path)
    df = df[(df.arm == "em_bp") & (df.seed == seed) & (df.n_chains == n)
            & (df.components == C) & (df.region == "all")]
    if df.empty:
        raise SystemExit(f"no E02 fit for seed={seed} n={n} C={C} in {path}")
    theta = np.array([float(v) for v in df.theta.iloc[0].split(";")])
    rho, pi, mu, s2 = theta[0], theta[1:1 + C], theta[1 + C:1 + 2 * C], theta[1 + 2 * C:1 + 3 * C]
    return MixtureInnovationKernel(rho=float(rho), pi=pi, mu=mu, s2=s2)


def _score_bp(log_k, grid, weights):
    from markov_score.bp_grid import grid_bp_batch

    def s(X, t):
        al, de = alpha_delta(t)
        m, _ = grid_bp_batch(grid, weights, log_k, X, al, de)
        return -(X - al * m) / de
    return s


def _cell_setup(cfg, seed, n):
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    prior = LaplaceAR1(FROZEN.rho)
    log_k_true = prior.log_transition_matrix(grid)
    sigma0 = theory.chain_covariance(FROZEN.rho, FROZEN.innovation_variance, N_SITES)
    rng_ref = rng_for("e05-ref", seed)      # E26's reference stream, verbatim, so the
    A = np.stack([prior.sample(rng_ref, N_SITES)   # comparison with E26 is paired
                  for _ in range(cfg["n_reference"])])
    al, de = alpha_delta(cfg["t_min"])
    R = al * A + np.sqrt(de) * rng_ref.standard_normal(A.shape)
    return grid, weights, prior, log_k_true, sigma0, R


def _sample_cell(args):
    cfg, seed, n = args
    grid, weights, prior, log_k_true, sigma0, R = _cell_setup(cfg, seed, n)
    K_true = np.exp(log_k_true)
    al_min, de_min = alpha_delta(cfg["t_min"])
    al_max, de_max = alpha_delta(cfg["t_max"])
    N = cfg["n_generate"]

    try:
        kb = _theta_from_e02(Path(__file__).resolve().parents[1] / cfg["em_source"],
                             seed, n, cfg["components"])
        K_em = np.exp(kb.log_transition_matrix(grid))
    except SystemExit as e:
        print("  ", e, flush=True)
        K_em = None

    # x at t_max drawn from the TRUE law through the channel -- the distribution the reverse
    # SDE's N(0,1) initialisation approximates, and what "flat marginal" means here. Drawn
    # ONCE, outside the jitter loop, so both jitter rows condition on the same observation.
    rng_hi = rng_for("e27-ffbs-x", seed, n)
    A_hi = np.stack([prior.sample(rng_hi, N_SITES) for _ in range(N)])
    X_hi = al_max * A_hi + np.sqrt(de_max) * rng_hi.standard_normal(A_hi.shape)

    rows = []
    for jitter in ((True, False) if cfg["jitter"] else (False,)):
        # Index streams keyed WITHOUT `jitter`; offset streams keyed with it. Both jitter
        # rows therefore draw identical grid indices (see _dejitter).
        arms = {}
        arms["ancestral_true"] = ancestral_grid(
            K_true, grid, weights, N, rng_for("e27-anc-true", seed, n),
            rng_for("e27-anc-true-jit", seed, n), jitter=jitter)
        arms["ffbs_true"] = ffbs_grid(
            K_true, grid, weights, X_hi, al_max, de_max,
            rng_for("e27-ffbs-true", seed, n),
            rng_for("e27-ffbs-true-jit", seed, n), jitter=jitter)
        if K_em is not None:
            arms["ancestral_em"] = ancestral_grid(
                K_em, grid, weights, N, rng_for("e27-anc-em", seed, n),
                rng_for("e27-anc-em-jit", seed, n), jitter=jitter)
        if jitter:
            # Continuous-space draws: no grid, so the jitter control does not apply to it.
            # Recorded once, with jitter = -1, rather than duplicated as two identical rows.
            rng_mc = rng_for("e27-mcfloor", seed, n)   # hoisted: see tests/test_no_degenerate_bundles.py
            arms["mc_floor"] = np.stack([prior.sample(rng_mc, N_SITES) for _ in range(N)])

        for arm, A_gen in arms.items():
            jit_col = -1 if arm == "mc_floor" else int(jitter)
            rg = rng_for("e27-noise", seed, n, arm)      # same channel draw in both rows
            G = al_min * A_gen + np.sqrt(de_min) * rg.standard_normal(A_gen.shape)
            m = gen_metrics.compare(G, R, FROZEN.rho, sigma0, grid, weights,
                                    log_k_true, cfg["t_min"], seed=seed)
            rows.append(dict(seed=seed, n_chains=n, arm=arm, jitter=jit_col,
                             n_sweeps=1, **m))
            print(f"  [E27] seed={seed} n={n} jit={jit_col:>2} {arm:15s} "
                  f"kurt {m['kurt_gen']:.3f} (ref {m['kurt_ref']:.3f})  "
                  f"kl {m['hist_kl_resid']:.5f}  sw2 {m['sw2']:.4f}", flush=True)
    return rows


def part_sample(cfg, out, workers):
    cells = [(cfg, s, n) for s in cfg["seeds"] for n in cfg["sizes"]]
    rows = run_cells(cells, _sample_cell, workers, out, "direct.csv")
    write_csv(out / "direct.csv", rows)


def _steps_cell(args):
    """How many reverse-SDE steps buy the accuracy one exact sweep gives for free?
    Same reference sample and same metric as part `sample`, so the two are comparable."""
    cfg, seed, n = args
    grid, weights, prior, log_k_true, sigma0, R = _cell_setup(cfg, seed, n)
    al_min, de_min = alpha_delta(cfg["t_min"])
    score = _score_bp(log_k_true, grid, weights)
    rows = []
    for ns in cfg["steps_ladder"]:
        times = time_grid(cfg["t_max"], cfg["t_min"], ns)
        rng = rng_for("e27-steps", seed, n, ns)
        x0 = rng.standard_normal((cfg["n_generate"], N_SITES))
        dW = np.sqrt(np.diff(-times))[:, None, None] * rng.standard_normal(
            (len(times) - 1, cfg["n_generate"], N_SITES))
        t0 = time.perf_counter()
        G = reverse_sde_with_increments(x0, score, times, dW)
        sec = time.perf_counter() - t0
        m = gen_metrics.compare(G, R, FROZEN.rho, sigma0, grid, weights,
                                log_k_true, cfg["t_min"], seed=seed)
        rows.append(dict(seed=seed, n_chains=n, arm="reverse_sde_exact", n_steps=ns,
                         sec=sec, **m))
        print(f"  [E27/steps] seed={seed} n={n} steps={ns:>4} kurt {m['kurt_gen']:.3f} "
              f"(ref {m['kurt_ref']:.3f}) kl {m['hist_kl_resid']:.5f} {sec:.2f}s", flush=True)
    return rows


def part_steps(cfg, out, workers):
    cells = [(cfg, s, n) for s in cfg["seeds"] for n in cfg["sizes"]]
    rows = run_cells(cells, _steps_cell, workers, out, "steps.csv")
    write_csv(out / "steps.csv", rows)


def part_cost(cfg, out):
    """Wall-clock per generated sequence: one sweep against the reverse SDE, same machine,
    same grid. `_timed` discards a warm-up and reports the median of `cost_repeat` calls."""
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    prior = LaplaceAR1(FROZEN.rho)
    log_k_true = prior.log_transition_matrix(grid)
    K_true = np.exp(log_k_true)
    al_max, de_max = alpha_delta(cfg["t_max"])
    score = _score_bp(log_k_true, grid, weights)

    def timed(fn):
        fn()
        ts = []
        for _ in range(cfg["cost_repeat"]):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts))

    rows = []
    for B in cfg["cost_batch"]:
        rng = rng_for("e27-cost", B)
        A = np.stack([prior.sample(rng, N_SITES) for _ in range(B)])
        X_hi = al_max * A + np.sqrt(de_max) * rng.standard_normal(A.shape)
        cases = {
            "ancestral_true": lambda: ancestral_grid(
                K_true, grid, weights, B, rng_for("c", B), rng_for("cj", B), jitter=True),
            "ffbs_true": lambda: ffbs_grid(
                K_true, grid, weights, X_hi, al_max, de_max,
                rng_for("c", B), rng_for("cj", B), jitter=True),
        }
        for ns in cfg["steps_ladder"]:
            def mk(ns=ns):
                times = time_grid(cfg["t_max"], cfg["t_min"], ns)
                r = rng_for("c2", B, ns)
                x0 = r.standard_normal((B, N_SITES))
                dW = np.sqrt(np.diff(-times))[:, None, None] * r.standard_normal(
                    (len(times) - 1, B, N_SITES))
                return lambda: reverse_sde_with_increments(x0, score, times, dW)
            cases[f"reverse_sde_{ns}"] = mk()
        for name, fn in cases.items():
            sec = timed(fn)
            rows.append(dict(batch=B, arm=name, sec=sec, sec_per_seq=sec / B,
                             n_steps=int(name.rsplit("_", 1)[1]) if name.startswith("reverse") else 1))
            print(f"  [E27/cost] B={B:>5} {name:20s} {sec:8.4f}s  {1e3*sec/B:8.4f} ms/seq", flush=True)
    write_csv(out / "cost.csv", rows)


def main() -> None:
    parser = experiment_parser(
        "e27_direct_sampler",
        "Q8 strong form: one BP sweep generates, with no score and no reverse SDE.",
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)

    parts = {
        "sample": ("one-sweep samplers against the true law", lambda c, o: part_sample(c, o, args.workers)),
        "steps": ("reverse SDE at 25..800 steps, same metric", lambda c, o: part_steps(c, o, args.workers)),
        "cost": ("wall-clock: one sweep vs the reverse SDE", part_cost),
    }
    if args.list_parts:
        print("\n".join(parts))
        return

    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir)
    tag = "_".join(selected) if args.only else "all"
    resolved = {"n_sites": N_SITES, "quick": args.quick, "parts": list(selected),
                "overrides": args.set, **cfg}
    write_json(out / f"params_{tag}.json",
               {**resolved, **provenance(resolved)})
    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(cfg, out)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
