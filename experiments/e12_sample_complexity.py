#!/usr/bin/env python
"""E12 — sample complexity by order: how much data each coordinate costs at each noise level.

The corollary predicts that recovering a covariance coordinate from sequences observed at level t
needs M proportional to e^{4t} of them, and a shape coordinate M proportional to e^{8t}, so their
ratio grows as e^{4t}.

E06 and E06b tried to see this in the ERROR at fixed M and could not, for a reason that is a
property of the question rather than of those runs: hold M fixed and raise t, and the shape
estimate stops being an estimate. Its bound crosses the coordinate's own range, the fit collapses
to a Gaussian innovation, and the error saturates at the width of the parameter rather than
growing at any rate (docs/NEGATIVE_RESULTS.md).

The sample-complexity statement does not have that problem, because it is the statement that
compensates: it asks how much data buys back a FIXED accuracy as the level rises. Nothing
saturates as long as the ladder of M reaches far enough, and the quantity read off --- the M at
which the error curve crosses a fixed threshold --- is exactly what the corollary is about.

Design. rho = 0.5, so the crossover sits at t_* = 0.54 and the levels swept lie above it, in the
regime where the exponents are predicted rather than approached. Each cell is one (level, M, seed):
M sequences drawn from the chain, all noised at that one level, one view each, fitted by EM-BP with
the protocol ladder, and scored by the error of the two fitted coordinates against the truth. The
analysis interpolates M*(t) at fixed accuracy for each coordinate and fits the log-slope.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
import e06_learning_through_noise as e06  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score.utils import ensure_dir, write_csv, write_json  # noqa: E402
from protocol import PROTOCOL  # noqa: E402

SETTINGS = dict(
    rho=0.5,
    levels=(0.6, 0.8, 1.0),
    sizes=(256, 512, 1024, 2048, 4096, 8192),
    seeds=tuple(range(8)),
    components=8,
    n_val=512, n_test=512,
    grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    em_checkpoints=PROTOCOL.em_checkpoints,
    em_init_rho=PROTOCOL.em_init_rho, em_init_var=PROTOCOL.em_init_var,
)
QUICK = dict(levels=(0.6,), sizes=(64, 128), seeds=(10_000,), n_val=32, grid_size=201,
             em_checkpoints=(5, 10))


def _cell(args):
    """One (level, size, seed). The fit is E06's cell verbatim, with the training-set size as the
    swept quantity, so the estimator compared across levels here is the same object E06 measured
    against the information bound -- no second implementation to drift."""
    cfg, t_obs, n, seed = args
    sub = dict(e06.SETTINGS)
    sub.update({k: cfg[k] for k in ("rho", "components", "n_val", "grid_size", "half_width",
                                    "em_checkpoints", "em_init_rho", "em_init_var") if k in cfg})
    sub["n_chains"] = n
    sub["n_test"] = cfg.get("n_test", 512)
    rows = e06._cell((sub, seed, t_obs))
    for r in rows:
        r["n_chains"] = n
        r["rho_err"] = abs(r["fit_rho"] - r["true_rho"])
        r["kurt_err"] = abs(r["fit_kurt"] - r["true_kurt"])
        print(f"  [E12] t={t_obs} M={n} seed={seed}: rho_err={r['rho_err']:.4f} "
              f"kurt_err={r['kurt_err']:.3f} ckpt={r['checkpoint']}", flush=True)
    return rows


def part_ladder(cfg, out, workers):
    cells = [(cfg, t, n, s) for t in cfg["levels"] for n in cfg["sizes"] for s in cfg["seeds"]]
    print(f"  {len(cells)} cells", flush=True)
    rows = run_cells(cells, _cell, workers, out, "ladder.csv")
    write_csv(out / "ladder.csv", rows)


PARTS = {"ladder": part_ladder}


def main():
    ap = experiment_parser("e12_sample_complexity", "sequences needed per coordinate against observation level")
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
