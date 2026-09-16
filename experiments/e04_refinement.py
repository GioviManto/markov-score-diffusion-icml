#!/usr/bin/env python
"""E4 — numerical refinement of the reported estimands on the ACTUAL fitted models.

Bundle size. The movement of an estimand between two grid resolutions is computed on the SAME
chains at both, so it is a paired quantity and 512 held-out sequences resolve it far more finely
than the effects it has to bound. The first attempt used 2048 (matching E02a's test bundle) and the
Ng = 1601 pass over that bundle -- sixteen times the cost of the Ng = 401 pass, six arms deep, on
twenty-four workers sharing memory bandwidth -- had produced no completed cell after nineteen hours
of a twenty-three hour allocation. Amended to 512 on 16 September, before any cell completed and
before any number was read.

For the E02 cells at the requested sizes, re-score the validation-selected EM-BP kernel
(theta from outputs/E02a/capacity.csv), the LMMSE oracle and the isotropic score on the
same held-out test bundle (same chains, same noise) at every (N_g, A) in the ladder,
with the reference posterior means recomputed at that resolution. Three numbers per
(seed, M, C, N_g, A): the pooled risk of each arm, and the movement of the reference
itself (relative RMS change of the reference score between resolutions). The strongest
statement this supports is "the reported effect moves by less than X under refinement".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score import theory  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.kernels import MixtureInnovationKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

N_SITES = FROZEN.n_sites
T_SCHEDULE = tuple(FROZEN.t_grid)

SETTINGS = dict(
    sizes=(32, 2048), components=(1, 8), seeds=tuple(range(16)),
    ladder=((401, 8.0), (801, 8.0), (1601, 8.0), (801, 10.0)),
    n_test=512, em_source="outputs/E02a/capacity.csv", chunk=256,
)
QUICK = dict(sizes=(32,), components=(1, 8), seeds=(10_000,), ladder=((201, 8.0), (401, 8.0)), n_test=64, em_source="outputs/scratch/e02_quick/capacity.csv")


def _kernel(path, seed, n, C):
    df = pd.read_csv(path)
    row = df[(df.arm == "em_bp") & (df.seed == seed) & (df.n_chains == n) & (df.components == C) & (df.region == "all")]
    if len(row) == 0:
        return None
    th = np.array([float(v) for v in row.theta.iloc[0].split(";")])
    return MixtureInnovationKernel(rho=float(th[0]), pi=th[1:1 + C], mu=th[1 + C:1 + 2 * C], s2=th[1 + 2 * C:1 + 3 * C])


def _bp_means(log_k, grid, w, X, t, chunk):
    al, de = alpha_delta(t)
    out = []
    for i in range(0, len(X), chunk):
        m, _ = grid_bp_batch(grid, w, log_k, X[i:i + chunk], al, de)
        out.append(m)
    return np.concatenate(out)


def _cell(args):
    """One (seed, size): the whole ladder on that seed's test bundle, so the reference at each
    resolution and the movement between resolutions are computed from the same chains."""
    cfg, seed, n = args
    prior = LaplaceAR1(FROZEN.rho)
    sigma0 = theory.chain_covariance(FROZEN.rho, FROZEN.innovation_variance, N_SITES)
    src = Path(__file__).resolve().parents[1] / cfg["em_source"]
    rows = []
    # the same test bundle as E02/exp_31 (chains and noise), regenerated from its stream
    rng = rng_for("exp31-test", seed)
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(cfg["n_test"])])
    X = {}
    for t in T_SCHEDULE:
        al, de = alpha_delta(t)
        X[t] = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
    kernels = {C: _kernel(src, seed, n, C) for C in cfg["components"]}
    ref_prev = None
    for ng, hw in cfg["ladder"]:
        grid, w = make_grid(hw, ng)
        log_k_true = prior.log_transition_matrix(grid)
        ref = {t: _bp_means(log_k_true, grid, w, X[t], t, cfg["chunk"]) for t in T_SCHEDULE}
        def pooled(mean_fn):
            num = den = 0.0
            for t in T_SCHEDULE:
                al, de = alpha_delta(t)
                s_ref = -(X[t] - al * ref[t]) / de
                s_hat = -(X[t] - al * mean_fn(t)) / de
                num += ((s_hat - s_ref) ** 2).sum(); den += (s_ref**2).sum()
            return float(np.sqrt(num / den))
        rec = dict(seed=seed, n_chains=n, n_grid=ng, half_width=hw,
                   risk_lmmse=pooled(lambda t: theory.lmmse_posterior_mean(X[t], sigma0, t)),
                   risk_iso=pooled(lambda t: (X[t] - (1 - np.exp(-2 * t)) * X[t]) / np.exp(-t)))
        if ref_prev is not None:
            num = den = 0.0
            for t in T_SCHEDULE:
                al, de = alpha_delta(t)
                s1 = -(X[t] - al * ref[t]) / de; s0 = -(X[t] - al * ref_prev[t]) / de
                num += ((s1 - s0) ** 2).sum(); den += (s1**2).sum()
            rec["ref_move_vs_prev"] = float(np.sqrt(num / den))
        else:
            rec["ref_move_vs_prev"] = 0.0
        for C, kb in kernels.items():
            if kb is None:
                continue
            lk = kb.log_transition_matrix(grid)
            rec[f"risk_em_C{C}"] = pooled(lambda t, lk=lk: _bp_means(lk, grid, w, X[t], t, cfg["chunk"]))
            rec[f"resolved_C{C}"] = bool(kb.scale_diagnostics(grid)["resolved"])
        ref_prev = ref
        rows.append(rec)
        print(f"  [E04] seed={seed} n={n} Ng={ng} A={hw}: " + " ".join(f"{k}={v:.4f}" for k, v in rec.items() if k.startswith("risk") or k.startswith("ref_")), flush=True)
    return rows


def part_refine(cfg, out, workers):
    cells = [(cfg, s, n) for s in cfg["seeds"] for n in cfg["sizes"]]
    rows = run_cells(cells, _cell, workers, out, "refinement.csv")
    write_csv(out / "refinement.csv", rows)


PARTS = {"refine": part_refine}


def main():
    ap = experiment_parser("e04_refinement", "grid refinement on the actual fitted models and held-out bundles")
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
        print(f"[e04] part {name}", flush=True)
        fn(cfg, out, args.workers)
        write_json(out / f"params_{name}.json", {**provenance(), "settings": cfg, "quick": bool(args.quick), "frozen": FROZEN.as_dict()})


if __name__ == "__main__":
    main()
