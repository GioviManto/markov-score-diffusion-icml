"""E47: how coarse can the BP grid be before the estimator notices? (Jerome's bottleneck: "the grid updates are a pain".)

Grid BP costs L B N_g^2 in the transition and L B N_g elsewhere; every experiment so far runs at N_g = 401 on [-8, 8] (spacing 0.04),
a setting chosen for convergence safety and checked only from ABOVE (E04, E11: 801 and 1601 change nothing). On the exclusive node the
measured time per sequence is 0.52 ms at N_g = 401 and 0.114 ms at 101 (E24x scaling: 4.6 times faster), against 0.03-0.04 ms for the
transformer, the GRU and the U-Net: coarsening alone would move BP from 12-18 times slower than the best networks to 3-4 times. What was
never measured is the accuracy side, for the oracle and for the LEARNED estimator.

Part `learned` (this file): the E33 fit-and-score path with the BP grid coarsened. Data, noised training views, validation and test bundles
and the REFERENCE posterior means (the true kernel on the registered 401-point grid) are E33's own, built by e33._common; only the grid
the kernel is FITTED and EVALUATED on changes, so a risk here is the score error of the coarse-grid estimator against the fine-grid truth,
discretisation included. Cells: the base cell (T0.3_h1_L32) and the finest lattice (T0.3_h0.25_L128, whose Gaussian bond is half as wide, so
it should need the finest grid). Arms em_pair_quartic and em_tilted_spline; M in (128, 2048); N_g in (51, 101, 201); seeds 0-7. The comparators
are E33's frozen fits at N_g = 401 on the same seeds (test-bundle hashes compared at read time).

The oracle part is E32's refinement part, unmodified, with a coarser list of grids (registry E47.json).

Predictions (experiments/registry/E47.json):
  Oracle  P1  at N_g = 101 the oracle's relative score error against (1601, 8) is below 1e-2, worst level and replicate, on all six phi^4 cells
          P2  at N_g = 201 it is below 1e-3 on all six
          P3  the worst-level error falls at least as fast as N_g^-1.5 between 51 and 201 on every cell (log-log slope <= -1.5)
          P4  the finest lattice (T0.3_h0.25_L128) has the largest error of the six cells at N_g = 51 and at 101
  Learned P5  base cell, M = 2048: at N_g = 201 and 101 the seed-mean risk of both arms is within 5% of E33's N_g = 401 fit (ratio in [0.95, 1.05])
          P6  base cell, M = 2048, N_g = 51: the site energy is within 15% of its N_g = 401 risk
          P7  finest lattice, M = 2048: at N_g = 201 both arms are within 10% of N_g = 401, and at N_g = 101 at least one arm is MORE than 10% worse
          P8  the site energy's median EM fit time at N_g = 101 is at most one quarter of E33's at N_g = 401 (base cell, M = 2048)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e33_field_learning as e33  # noqa: E402
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from markov_score.bp_grid import make_grid  # noqa: E402
from markov_score.utils import ensure_dir, write_csv, write_json  # noqa: E402

e14 = e33.e14

SETTINGS = dict(e33.SETTINGS, cells=("T0.3_h1_L32", "T0.3_h0.25_L128"), sizes=(128, 2048), seeds=tuple(range(8)),
                arms=("em_pair_quartic", "em_tilted_spline"), coarse_grids=(51, 101, 201))
QUICK = dict(e33.QUICK, arms=("em_pair_quartic",), coarse_grids=(51,), grid_size=401, cells=("T0.3_h1_L32",))


def _cell(args):
    cfg, cell_id, seed, n, ng = args
    # E33's own data and REFERENCE (the true kernel on the registered fine grid): built exactly as in E33
    chain, grid_f, w_f, log_k, log_mu, sigma0, A, Av, val, test, base = e33._common(cfg, cell_id, seed, n)
    L = base["n_sites"]
    groups = e14._one_view_groups(A, f"e33-em-{cell_id}-M{n}", seed, L)
    val_groups = e14._one_view_groups(Av, f"e33-emval-{cell_id}", seed, L)
    grid, w = make_grid(cfg["half_width"], ng)       # the grid the kernel is fitted and evaluated on
    rows = []
    for arm in cfg["arms"]:
        for r in e33._fit_arm(cfg, arm, grid, w, groups, val, val_groups, test, seed, cell_id, L, n):
            rows.append({**base, **e33.ROW_DEFAULTS, **r, "grid_size": ng, "reference_grid": cfg["grid_size"]})
    print(f"  [E47] {cell_id} seed={seed} M={n} N_g={ng}: " + " ".join(
        f"{r['arm']}={r['risk']:.4f}" for r in rows if r["selection"] == "risk"), flush=True)
    return rows


def part_learned(cfg, out, workers):
    cells = [(cfg, c, s, n, ng) for n in sorted(cfg["sizes"], reverse=True) for c in cfg["cells"]
             for ng in sorted(cfg["coarse_grids"], reverse=True) for s in cfg["seeds"]]
    print(f"  learned: {len(cells)} cells x {len(cfg['arms'])} arms", flush=True)
    write_csv(out / "kernels.csv", run_cells(cells, _cell, workers, out, "kernels.csv",
                                             key=lambda c: f"{c[1]}_s{c[2]}_M{c[3]}_N{c[4]}"))


PARTS = {"learned": part_learned}


def main():
    ap = experiment_parser("e47_coarse_grid", "the BP grid coarsened: the learned estimator's accuracy and fit time")
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
