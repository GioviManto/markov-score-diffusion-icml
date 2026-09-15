#!/usr/bin/env python
"""E2 — does the innovation mixture's capacity buy anything on the headline task?

The previous manuscript's Appendix L said no (exp_32). The audit found that verdict to
be a stopping-rule artefact and that the LMMSE floor (0.168 pooled, Laplace chain)
sits far above the reported C = 8 errors. This experiment settles it with the
checkpoint discipline of exp_31 applied identically to every capacity.

Parts
-----
capacity  Laplace chain at the frozen configuration. For each (seed, M) the SAME
          bundles as exp_31 (tags `exp31-train/val/test`, so the test-bundle hash can
          be checked against outputs/frozen/exp_31_confirm_merged/confirm.csv and the
          screened window head's errors can be placed on identical data). For each
          C in `components`: EM-BP from the exp_31 initialisation stream, checkpoint
          ladder to `em_cap`, checkpoint selected on the validation bundle (bulk
          region, as exp_31), scored on the test bundle in every region. Oracles on the
          same test bundle: LMMSE with the true covariance, the isotropic score.

family    the one-parameter generalised-Gaussian family at fixed rho and variance:
          beta in `betas` (excess kurtosis +3 ... -1.1). M = `family_size`,
          C in `family_components`; same ladder and selection; scored against
          true-kernel BP of that family; oracle ladder; Hellinger distance of the
          fitted kernel to the true one.

Every row records seed, size, capacity, selected checkpoint, at-cap flag, stop
reason, region, unreduced sq_err / sq_ref, the fitted rho, innovation variance and
excess kurtosis, the grid-resolution flag, and the bundle hashes. Provenance in
params_<part>.json. `--workers N` runs (seed, size) cells in N processes.
"""

from __future__ import annotations

import os
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
from markov_score.metrics import transition_hellinger  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.shape_kernel import GeneralizedGaussianKernel, generalized_gaussian_excess_kurtosis  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

N_SITES = FROZEN.n_sites
T_SCHEDULE = tuple(FROZEN.t_grid)
BULK = slice(4, N_SITES - 4)
REGIONS = {"all": slice(0, N_SITES), "bulk": BULK, "centre": slice(N_SITES // 2, N_SITES // 2 + 1)}

SETTINGS = dict(
    sizes=PROTOCOL.sizes,
    components=PROTOCOL.capacities,
    seeds=PROTOCOL.seeds,
    n_val=PROTOCOL.n_val, n_test=PROTOCOL.n_test,
    grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    em_checkpoints=PROTOCOL.em_checkpoints,
    em_init_rho=PROTOCOL.em_init_rho, em_init_var=PROTOCOL.em_init_var,
    betas=(1.0, 1.25, 1.5, 2.0, 3.0, 6.0),
    family_size=512, family_components=(1, 8),
)
DEV = dict(seeds=tuple(range(10_000, 10_004)))
QUICK = dict(sizes=(32,), components=(1, 8), seeds=(10_000,), n_val=64, n_test=64, grid_size=201,
             em_checkpoints=(5, 10, 20), betas=(1.0, 2.0), family_size=32)


# ----------------------------------------------------------------------------- shared


def _bundle(sampler, log_k, grid, weights, tag, seed, n_chains):
    rng = rng_for(tag, seed)
    A = np.stack([sampler(rng) for _ in range(n_chains)])
    out = {}
    for t in T_SCHEDULE:
        al, de = alpha_delta(t)
        X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
        m_ref, _ = grid_bp_batch(grid, weights, log_k, X, al, de)
        out[t] = (X, m_ref)
    return A, out


def _bundle_hash(bundle) -> str:
    import hashlib
    h = hashlib.sha256()
    for t in sorted(bundle):
        X, m = bundle[t]
        h.update(np.ascontiguousarray(X, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(m, dtype=np.float64).tobytes())
    return h.hexdigest()[:12]


def _sq(m_hat, m_ref, X, t, sl):
    al, de = alpha_delta(t)
    s_hat = -(X[:, sl] - al * m_hat[:, sl]) / de
    s_ref = -(X[:, sl] - al * m_ref[:, sl]) / de
    return float(((s_hat - s_ref) ** 2).sum()), float((s_ref**2).sum())


def _risk(pairs):
    num = sum(e for e, _ in pairs); den = sum(r for _, r in pairs)
    return float(np.sqrt(num / den)) if den > 0 else float("nan")


def _eval_kernel(kernel, bundle, grid, weights, region):
    log_k = kernel.log_transition_matrix(grid)
    pairs = []
    for t in T_SCHEDULE:
        X, m_ref = bundle[t]
        al, de = alpha_delta(t)
        m, _ = grid_bp_batch(grid, weights, log_k, X, al, de)
        pairs.append(_sq(m, m_ref, X, t, REGIONS[region]))
    return pairs


def _oracle_rows(bundle, grid, weights, sigma0, base):
    rows = []
    for region, sl in REGIONS.items():
        pairs_lm, pairs_iso = [], []
        for t in T_SCHEDULE:
            X, m_ref = bundle[t]
            al, de = alpha_delta(t)
            m_lm = theory.lmmse_posterior_mean(X, sigma0, t)
            pairs_lm.append(_sq(m_lm, m_ref, X, t, sl))
            # isotropic score -x  <=>  posterior mean m = (x - Delta x)/alpha
            m_iso = (X - de * X) / al
            pairs_iso.append(_sq(m_iso, m_ref, X, t, sl))
        for arm, pairs in (("lmmse_true", pairs_lm), ("isotropic", pairs_iso)):
            rows.append({**base, "arm": arm, "components": 0, "checkpoint": -1, "at_cap": False,
                         "stop_reason": "oracle", "region": region,
                         "sq_err": sum(e for e, _ in pairs), "sq_ref": sum(r for _, r in pairs), "risk": _risk(pairs),
                         "n_params": 0, "fit_rho": np.nan, "fit_ivar": np.nan, "fit_kurt": np.nan, "resolved": True,
                         "val_risk_selected": np.nan, "val_curve": "", "theta": ""})
    return rows


def _em_fit_and_score(cfg, C, A, val, test, grid, weights, seed_tags, base):
    rng_init = rng_for(seed_tags["init"][0], *seed_tags["init"][1:])
    idx = np.array_split(rng_for(seed_tags["split"][0], *seed_tags["split"][1:]).permutation(len(A)), len(T_SCHEDULE))
    noise = rng_for(seed_tags["noise"][0], *seed_tags["noise"][1:])
    groups = []
    for t, ix in zip(T_SCHEDULE, idx):
        al, de = alpha_delta(t)
        sub = A[ix]
        groups.append((al * sub + np.sqrt(de) * noise.standard_normal(sub.shape), al, de))
    ck = tuple(int(c) for c in cfg["em_checkpoints"])
    k0 = MixtureInnovationKernel.init(C, rho=cfg["em_init_rho"], var=cfg["em_init_var"], rng=rng_init)
    _, trace, saved = fit_em(k0, grid, weights, groups, n_iters=max(ck), tol=FROZEN.em_loglik_tol, checkpoints=set(ck))
    best, best_risk = None, np.inf
    val_curve = {}
    for it in sorted(saved):
        r = _risk(_eval_kernel(saved[it], val, grid, weights, "bulk"))
        val_curve[it] = r
        if r < best_risk:
            best, best_risk = it, r
    kb = saved[best]
    mom = kb.innovation_moments
    diag = kb.scale_diagnostics(grid)
    rows = []
    # one BP pass per test level, scored on every region
    log_kb = kb.log_transition_matrix(grid)
    means = {}
    for t in T_SCHEDULE:
        X, m_ref = test[t]
        al, de = alpha_delta(t)
        means[t], _ = grid_bp_batch(grid, weights, log_kb, X, al, de)
    for region in REGIONS:
        pairs = [_sq(means[t], test[t][1], test[t][0], t, REGIONS[region]) for t in T_SCHEDULE]
        rows.append({**base, "arm": "em_bp", "components": C, "checkpoint": best, "at_cap": best == max(saved),
                     "stop_reason": trace.stop_reason, "region": region,
                     "sq_err": sum(e for e, _ in pairs), "sq_ref": sum(r for _, r in pairs), "risk": _risk(pairs),
                     "n_params": int(len(kb.theta)), "fit_rho": float(kb.rho),
                     "fit_ivar": float(mom["innovation_var"]), "fit_kurt": float(mom["innovation_excess_kurtosis"]),
                     "resolved": bool(diag["resolved"]), "val_risk_selected": best_risk,
                     "theta": ";".join(f"{v:.10g}" for v in np.asarray(kb.theta).ravel()),
                     "val_curve": ";".join(f"{k}:{v:.5f}" for k, v in sorted(val_curve.items()))})
    return rows, kb


# ----------------------------------------------------------------------------- capacity


def _capacity_cell(args):
    cfg, seed, n = args
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    prior = LaplaceAR1(FROZEN.rho)
    log_k = prior.log_transition_matrix(grid)
    sampler = lambda rng: prior.sample(rng, N_SITES)
    A = np.stack([sampler(rng_for("exp31-train", seed, n)) for _ in range(n)]) if False else None
    rng_train = rng_for("exp31-train", seed, n)
    A = np.stack([prior.sample(rng_train, N_SITES) for _ in range(n)])
    _, val = _bundle(sampler, log_k, grid, weights, "exp31-val", seed, cfg["n_val"])
    _, test = _bundle(sampler, log_k, grid, weights, "exp31-test", seed, cfg["n_test"])
    sigma0 = theory.chain_covariance(FROZEN.rho, FROZEN.innovation_variance, N_SITES)
    base = dict(seed=seed, n_chains=n, val_bundle=_bundle_hash(val), test_bundle=_bundle_hash(test))
    rows = _oracle_rows(test, grid, weights, sigma0, base)
    tags = {"init": ("exp31-em-init", seed), "split": ("exp31-em-split", seed), "noise": ("exp31-em-noise", seed)}
    for C in cfg["components"]:
        r, _ = _em_fit_and_score(cfg, C, A, val, test, grid, weights, tags, base)
        rows.extend(r)
        print(f"  [E02 capacity] seed={seed} n={n} C={C} ckpt={r[0]['checkpoint']} risk(all)={r[0]['risk']:.4f} kurt={r[0]['fit_kurt']:.2f}", flush=True)
    return rows


def part_capacity(cfg, out, workers):
    cells = [(cfg, s, n) for s in cfg["seeds"] for n in cfg["sizes"]]
    rows = _run(cells, _capacity_cell, workers, out, "capacity.csv")
    write_csv(out / "capacity.csv", rows)


# ----------------------------------------------------------------------------- family


def _family_cell(args):
    cfg, seed, beta = args
    from exp_22_shape_information import sample_generalized_gaussian
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    q = FROZEN.innovation_variance
    ktrue = GeneralizedGaussianKernel(rho=FROZEN.rho, q=q, beta=beta)
    log_k = ktrue.log_transition_matrix(grid)
    sampler = lambda rng: sample_generalized_gaussian(rng, ktrue, N_SITES)
    n = cfg["family_size"]
    rng_train = rng_for("e02-fam-train", beta, seed, n)
    A = np.stack([sampler(rng_train) for _ in range(n)])
    _, val = _bundle(sampler, log_k, grid, weights, ("e02-fam-val", beta), seed, cfg["n_val"])
    _, test = _bundle(sampler, log_k, grid, weights, ("e02-fam-test", beta), seed, cfg["n_test"])
    sigma0 = theory.chain_covariance(FROZEN.rho, q, N_SITES)
    base = dict(seed=seed, n_chains=n, beta=beta, excess_kurtosis=generalized_gaussian_excess_kurtosis(beta),
                val_bundle=_bundle_hash(val), test_bundle=_bundle_hash(test))
    rows = _oracle_rows(test, grid, weights, sigma0, base)
    # uniform row schema: the oracle rows carry the Hellinger keys as NaN
    hell_keys = [k for k, v in transition_hellinger(log_k, log_k, grid, weights).items() if np.isscalar(v)]
    for r in rows:
        r.update({k: np.nan for k in hell_keys})
    tags = {"init": ("e02-fam-init", beta, seed), "split": ("e02-fam-split", beta, seed), "noise": ("e02-fam-noise", beta, seed)}
    for C in cfg["family_components"]:
        r, kb = _em_fit_and_score(cfg, C, A, val, test, grid, weights, tags, base)
        hd = transition_hellinger(kb.log_transition_matrix(grid), log_k, grid, weights)
        for row in r:
            row.update({k: float(v) for k, v in hd.items() if np.isscalar(v)})
        rows.extend(r)
        print(f"  [E02 family] beta={beta} seed={seed} C={C} ckpt={r[0]['checkpoint']} risk(all)={r[0]['risk']:.4f} kurt={r[0]['fit_kurt']:.2f}", flush=True)
    return rows


def part_family(cfg, out, workers):
    cells = [(cfg, s, b) for b in cfg["betas"] for s in cfg["seeds"]]
    rows = _run(cells, _family_cell, workers, out, "family.csv")
    write_csv(out / "family.csv", rows)


# ----------------------------------------------------------------------------- driver


def _run(cells, fn, workers, out=None, csv_name=None):
    """Run cells serially or in a spawn pool; each cell's rows are written by the worker to
    out/cells/<csv stem>/<key>.json and the CSV is rebuilt after every completion
    (experiments/common.py: run_cells; resumable, no result travels through the pool pipe)."""
    return run_cells(cells, fn, workers, out, csv_name)


PARTS = {"capacity": part_capacity, "family": part_family}


def main():
    ap = experiment_parser("e02_capacity", "mixture capacity on the headline task; the tunable non-Gaussian family")
    ap.add_argument("--dev", action="store_true", help="development seeds 10000-10003 instead of the confirmatory 0-15")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.list_parts:
        print("\n".join(PARTS)); return
    cfg = dict(SETTINGS)
    if args.dev:
        cfg.update(DEV)
    cfg["protocol_version"] = PROTOCOL.version
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    out = ensure_dir(Path(args.output_dir))
    for name, fn in select_parts(PARTS, args.only).items():
        print(f"[e02] part {name}: {len(cfg['seeds'])} seeds, workers={args.workers}", flush=True)
        fn(cfg, out, args.workers)
        write_json(out / f"params_{name}.json", {**provenance(), "settings": cfg, "dev": bool(args.dev),
                                                   "quick": bool(args.quick), "workers": args.workers,
                                                   "frozen": FROZEN.as_dict()})


if __name__ == "__main__":
    main()
