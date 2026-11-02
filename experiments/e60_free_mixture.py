"""E60: a kernel with no form. A free Gaussian mixture on the pair (a, a'), learned by EM through BP on the Allen-Cahn field.

Asked on 1 Nov 2026: do not give the kernel a form, fit a Gaussian mixture without the shift. E60 fits
markov_score.free_mixture_kernel (K(a' | a) = p(a, a') / p(a) with p a free C-component 2-D Gaussian mixture, 6 C
parameters) by EM through exact grid BP from noisy data, on the four Allen-Cahn cells of the slide-15 table, with every one
of E33's data, validation and test bundles, seeds and EM settings, so that each row pairs with E33's frozen em_mixture,
em_mdn, em_pair_quartic and network rows on the same seed (bundle hashes compared at read time).

The calibration (tools/core/calibrate_e60.py, from the TRUE kernels only) is why: the best shift mixture sits AT the linear
floor on these cells (the field's drift depends on the state through the double well, which a' = rho a + eps cannot
represent), while the best free mixture with C = 4 (components gated at mirror-image centres) reaches about 0.02 of it.

This is the frozen e33 driver with one addition applied IN PROCESS (e33_field_learning.py itself is not modified): the arms
`em_free_C<C>` in the arm constructor. Rows have E33's schema, so the same readings apply (tools/core/read_e60.py).

Predictions (experiments/registry/E60.json), risk selection, in units of the cell's linear floor, paired by seed:
  P1  M = 512, 2048: em_free_C4 below em_mixture in >= 14/16 seeds on every cell; its seed-mean at 2048 below 0.5
  P2  M = 128: em_free_C4 below em_mdn in >= 12/16 seeds on >= 3 of the 4 cells
  P3  the seed-mean ratio em_free_C8 / em_free_C4 is smaller at M = 2048 than at M = 128 on every cell
  P4  M = 2048: em_pair_quartic below em_free_C4 in >= 12/16 seeds on >= 3 of the 4 cells
  P5  M = 128, 2048: em_free_C4 below the best of E33's networks in >= 14/16 seeds on every cell
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e33_field_learning as e33  # noqa: E402
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from markov_score.free_mixture_kernel import FreeMixtureKernel  # noqa: E402
from markov_score.utils import ensure_dir, write_csv, write_json  # noqa: E402

CELLS = ("T0.3_h1_L32", "T0.5_iso_L32", "T1_iso_L32", "T2_iso_L32")
ARM_PREFIX = "em_free_C"

_init_e33 = e33._init_kernel


def _init(arm, cfg, grid, seed, cell_id, n):
    if arm.startswith(ARM_PREFIX):
        return FreeMixtureKernel.init(int(arm[len(ARM_PREFIX):]), rho=cfg["em_init_rho"], var=cfg["em_init_var"],
                                      lr=cfg["free_lr"], n_inner=cfg["free_inner"])
    return _init_e33(arm, cfg, grid, seed, cell_id, n)


# The addition, in process only: E33's file is the frozen script and stays byte-identical.
e33._init_kernel = _init

SETTINGS = dict(e33.SETTINGS, arms=("em_free_C4", "em_free_C8"), sizes=(128, 512, 2048), cells=CELLS,
                free_lr=1e-2, free_inner=10)
QUICK = dict(e33.QUICK, arms=("em_free_C4",), free_lr=1e-2, free_inner=10)


def part_kernels(cfg, out, workers):
    cells = [(cfg, c, s, n) for n in sorted(cfg["sizes"], reverse=True) for c in cfg["cells"] for s in cfg["seeds"]]
    print(f"  kernels: {len(cells)} cells x {len(cfg['arms'])} arms", flush=True)
    write_csv(out / "kernels.csv", run_cells(cells, e33._kernels_cell, workers, out, "kernels.csv"))


PARTS = {"kernels": part_kernels}


def main():
    ap = experiment_parser("e60_free_mixture", "a free 2-D Gaussian-mixture kernel by EM through BP on the Allen-Cahn field")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.list_parts:
        print("\n".join(PARTS)); return
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    out = ensure_dir(Path(args.output_dir))
    cells_info = [e33.resolve_cell(c)[1] for c in e33.CELLS if c["id"] in cfg["cells"]]
    write_json(out / "cells.json", cells_info)
    for name, fn in select_parts(PARTS, args.only).items():
        fn(cfg, out, args.workers)
        write_json(out / f"params_{name}.json", {**provenance(), "settings": cfg, "cells": cells_info,
                                                 "quick": bool(args.quick), "workers": args.workers})


if __name__ == "__main__":
    main()
