#!/usr/bin/env python
"""E37 -- the domain-length DISTRIBUTION of exact and generated fields (the consolidated note, section 24.5).

E34's physics_learned reported the mean and median domain length of samples drawn from the learned kernels. The note
asks to "report at least" the distribution. E37 draws E34's own samples -- the same frozen E33 fits and the same random
streams (rng tags e34-learned-truth, e34-learned, e34-learned-jit) -- and compares, per (cell, seed, arm), the whole
distribution of interior COARSE domain lengths (N-31's observable: every run shorter than one kink width, sqrt(2)/h
sites, merged into its neighbours) in physical units: quantiles, the two-sample Kolmogorov-Smirnov distance to the
truth's, and histogram counts on common bins.

PREDECLARED (registry E37.json, written before any cell ran):
  P1  spline and site energy: on every cell the seed means of q10, q50, q90 are within 10% of the truth's, and the
      seed-mean KS distance to the truth is below 0.05.
  P2  mixture: on the three T = 0.3 cells its seed-mean KS distance exceeds 0.05 and its q90 is more than 15% below
      the truth's.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score.bp_grid import make_grid  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

import e33_field_learning as e33  # noqa: E402
import e34_field_physics as e34  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BINS = np.arange(0.0, 64.5, 1.0)          # physical units; the coarsest cell's domains are a few to tens of units

SETTINGS = dict(
    cells=tuple(c["id"] for c in e33.CELLS), seeds=(0, 1, 2, 3), n_chains=2048, n_learned=2048,
    grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    components=8, mdn_components=4, mdn_hidden=32,
    kernels_csv=("outputs/hpc_E33k_s0/kernels.csv", "outputs/hpc_E33q_s0/kernels.csv"),
)
QUICK = dict(cells=("T0.3_h1_L32",), seeds=(0,), n_learned=128, grid_size=201)


def coarse_domains(A: np.ndarray, h: float) -> np.ndarray:
    """Interior coarse domain lengths of a batch of standardised chains, in physical units."""
    min_len = max(1, int(np.ceil(np.sqrt(2.0) / h)))
    runs = [e34._coarse_runs(np.sign(row), min_len) for row in A]
    return np.concatenate(runs) * h if runs else np.empty(0)


def _row(base, arm, d, truth_d, failed=0):
    if failed or d.size == 0:
        return dict(base, arm=arm, failed=int(failed), n_domains=int(d.size), q10=np.nan, q50=np.nan, q90=np.nan,
                    mean=np.nan, ks=np.nan, ks_p=np.nan)
    ks = ks_2samp(d, truth_d) if arm != "truth" else None
    return dict(base, arm=arm, failed=0, n_domains=int(d.size), q10=float(np.quantile(d, 0.1)), q50=float(np.quantile(d, 0.5)),
                q90=float(np.quantile(d, 0.9)), mean=float(d.mean()),
                ks=float(ks.statistic) if ks else 0.0, ks_p=float(ks.pvalue) if ks else 1.0)


def _cell(args):
    cfg, cell_id, seed, rows_for_cell = args
    cell = next(c for c in e34.CELLS if c["id"] == cell_id)
    chain, info = e34.resolve_cell(cell)
    L, h = cell["L"], info["h"]
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    import e27_direct_sampler as e27
    e27.N_SITES = L
    truth = chain.sample_batch(rng_for("e34-learned-truth", cell_id, seed), cfg["n_learned"], L)
    td = coarse_domains(truth, h)
    base = dict(cell=cell_id, T=info.get("T", np.nan), h=h, n_sites=L, seed=seed)
    out, hist = [_row(base, "truth", td, td)], [dict(base, arm="truth", **{f"b{int(b)}": int(c) for b, c in zip(BINS[:-1], np.histogram(td, BINS)[0])})]
    mu_fn = e33._stationary_mu(grid, w)
    for row in rows_for_cell:
        theta = row.get("theta")
        if row.get("failed") == 1 or not isinstance(theta, str) or not theta:
            out.append(_row(base, row["arm"], np.empty(0), td, failed=1))
            continue
        k = e34.kernel_from_row(row, cfg, grid)
        K = np.exp(k.log_transition_matrix(grid))
        S = e27.ancestral_grid(K, grid, w, cfg["n_learned"], rng_for("e34-learned", cell_id, seed, row["arm"]),
                               rng_for("e34-learned-jit", cell_id, seed, row["arm"]), jitter=True, log_mu=mu_fn(k))
        d = coarse_domains(S, h)
        out.append(_row(base, row["arm"], d, td))
        hist.append(dict(base, arm=row["arm"], **{f"b{int(b)}": int(c) for b, c in zip(BINS[:-1], np.histogram(d, BINS)[0])}))
    print(f"  [E37] {cell_id} seed={seed}: " + " ".join(f"{r['arm']}:ks={r['ks']:.3f}/q90={r['q90']:.2f}" for r in out), flush=True)
    return [dict(r, kind="stats") for r in out] + [dict(r, kind="hist") for r in hist]


def part_distribution(cfg, out, workers):
    df = pd.concat([pd.read_csv(ROOT / p) for p in cfg["kernels_csv"]], ignore_index=True)
    df = df[(df.selection == "risk") & (df.n_chains == cfg["n_chains"]) & df.seed.isin(cfg["seeds"]) & df.cell.isin(cfg["cells"])]
    cells = [(cfg, c, int(s), sub.to_dict("records")) for (c, s), sub in df.groupby(["cell", "seed"])]
    print(f"  distribution: {len(cells)} cells from {len(df)} fitted kernels", flush=True)
    rows = run_cells(cells, _cell, workers, out, "all.csv", key=lambda c: f"{c[1]}_s{c[2]}")
    write_csv(out / "distribution.csv", [r for r in rows if r["kind"] == "stats"])
    write_csv(out / "domains.csv", [r for r in rows if r["kind"] == "hist"])


PARTS = {"distribution": part_distribution}


def main():
    ap = experiment_parser("e37_domain_distribution", "the domain-length distribution of exact and generated fields")
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
