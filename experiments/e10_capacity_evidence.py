#!/usr/bin/env python
"""E10 — capacity in the thesis's own estimand: held-out log-evidence per edge.

Why this exists. Chapter 10 of the thesis concluded that no mixture capacity improves on a
single Gaussian innovation, scoring HELD-OUT LOG-EVIDENCE per edge, and flagged its own
caveat: the share of fits stopped at the iteration cap and with a component under the grid's
resolution floor both climbed steeply with C, so the sweep compared convergence states rather
than estimators. E02a re-ran the capacity sweep with the checkpoint selected identically for
every capacity and found the opposite -- but in a DIFFERENT estimand (relative score error).
Two differences at once is not an explanation, so this experiment scores the E02a fits in the
thesis's estimand: the mean log p_t(x) per transition under the fitted kernel on the same
held-out bundles, paired within seed against C = 1.

It introduces no new fit. The kernels are the validation-selected ones already frozen in
outputs/E02a/capacity.csv (theta column); only the scoring is new, so any difference from the
thesis is attributable to the protocol and not to a second change of estimator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from e02_capacity import REGIONS, T_SCHEDULE, _bundle, _bundle_hash  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.kernels import MixtureInnovationKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.utils import ensure_dir, write_csv, write_json  # noqa: E402
from protocol import PROTOCOL  # noqa: E402

N_SITES = FROZEN.n_sites

SETTINGS = dict(
    sizes=PROTOCOL.sizes, components=PROTOCOL.capacities, seeds=PROTOCOL.seeds,
    n_test=PROTOCOL.n_test, grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    em_source="outputs/E02a/capacity.csv",
)
QUICK = dict(sizes=(32,), components=(1, 8), seeds=(10_000,), n_test=64, grid_size=201,
             em_source="outputs/scratch/e02_quick3/capacity.csv")


def _kernel(df, seed, n, C):
    row = df[(df.arm == "em_bp") & (df.seed == seed) & (df.n_chains == n)
             & (df.components == C) & (df.region == "all")]
    if len(row) == 0 or not isinstance(row.theta.iloc[0], str):
        return None
    th = np.array([float(v) for v in row.theta.iloc[0].split(";")])
    return MixtureInnovationKernel(rho=float(th[0]), pi=th[1:1 + C], mu=th[1 + C:1 + 2 * C], s2=th[1 + 2 * C:1 + 3 * C])


def _cell(args):
    cfg, seed, n = args
    src = Path(__file__).resolve().parents[1] / cfg["em_source"]
    df = pd.read_csv(src)
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    prior = LaplaceAR1(FROZEN.rho)
    log_k_true = prior.log_transition_matrix(grid)
    sampler = lambda rng: prior.sample(rng, N_SITES)
    _, test = _bundle(sampler, log_k_true, grid, weights, "exp31-test", seed, cfg["n_test"])
    base = dict(seed=seed, n_chains=n, test_bundle=_bundle_hash(test))
    n_edges = N_SITES - 1
    rows = []

    def evidence_per_edge(log_k):
        """Mean over the schedule and the bundle of log p_t(x) per transition, in nats."""
        tot = 0.0
        for t in T_SCHEDULE:
            X, _ = test[t]
            al, de = alpha_delta(t)
            _, _, ev = grid_bp_batch(grid, weights, log_k, X, al, de, return_evidence=True)
            tot += float(np.mean(ev)) / n_edges
        return tot / len(T_SCHEDULE)

    ev_true = evidence_per_edge(log_k_true)
    for C in cfg["components"]:
        kb = _kernel(df, seed, n, C)
        if kb is None:
            continue
        ev = evidence_per_edge(kb.log_transition_matrix(grid))
        rows.append({**base, "arm": "em_bp", "components": C, "logev_per_edge": ev,
                     "logev_true": ev_true, "logev_gap_to_true": ev_true - ev,
                     "fit_kurt": float(kb.innovation_moments["innovation_excess_kurtosis"]),
                     "resolved": bool(kb.scale_diagnostics(grid)["resolved"])})
        print(f"  [E10] seed={seed} n={n} C={C}: logev/edge {ev:.5f} (true {ev_true:.5f}, gap {ev_true - ev:.5f})", flush=True)
    return rows


def part_evidence(cfg, out, workers):
    cells = [(cfg, s, n) for s in cfg["seeds"] for n in cfg["sizes"]]
    rows = run_cells(cells, _cell, workers, out, "evidence.csv")
    write_csv(out / "evidence.csv", rows)


PARTS = {"evidence": part_evidence}


def main():
    ap = experiment_parser("e10_capacity_evidence", "held-out log-evidence per edge of the E02a fits")
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
