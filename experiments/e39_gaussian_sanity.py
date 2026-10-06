#!/usr/bin/env python
"""E39 -- the Gaussian sanity check: grid BP against the closed-form score and the Kalman (RTS) smoother.

The consolidated note asks for it twice: as a short proposition (section 29 step 3; appendix F item 4) and as
figure 2 of its figures-first plan (section 28). On the Gaussian linear chain three computations of one
conditional law must agree: the closed form S = -Sigma_t^{-1} x, the Rauch-Tung-Striebel smoother
(`markov_score.kalman`, written without the BP code), and sum-product, which grid BP computes by trapezoidal
quadrature on N_g points of [-A, A]. E39 refines the grid at A = 8 (part `grid`) and widens the domain at the
operating spacing 0.04 (part `domain`), at every level of the paper's schedule, and records the pooled relative
score error of each arm against each reference.

PREDECLARED (registry E39.json, written before any run):
  P1  RTS against the closed form <= 1e-12 at all twelve levels.
  P2  at (N_g, A) = (401, 8): grid BP against both references <= 1e-10, and moment-projected BP too.
  P3  at A = 8 the error is non-increasing in N_g until the floating floor, and the N_g reaching 1e-10 is at
      least as large at t = 0.05 as at t = 3.
  P4  at fixed spacing the error is non-increasing in A, and at A = 3 it exceeds 1e-6 at t = 0.05.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, grid_bp_projected_batch, make_grid  # noqa: E402
from markov_score.kalman import closed_form_gaussian, kalman_rts  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import GaussianAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

SETTINGS = dict(
    rho=FROZEN.rho, n_sites=FROZEN.n_sites, n_chains=256, levels=tuple(FROZEN.t_grid),
    grid_sizes=(11, 21, 41, 81, 161, 321, 401, 801, 1601), grid_half_width=8.0,
    domain_half_widths=(2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0), domain_spacing=0.04,
    operating=(401, 8.0),
)
QUICK = dict(n_chains=16, levels=(0.05, 1.0, 3.0), grid_sizes=(21, 101, 401), domain_half_widths=(3.0, 8.0))


def _rel(est, ref):
    return float(np.linalg.norm(est - ref) / np.linalg.norm(ref))


def _bundle(cfg):
    prior = GaussianAR1(cfg["rho"])
    rng = rng_for("e39-truth")
    A = np.stack([prior.sample(rng, cfg["n_sites"]) for _ in range(cfg["n_chains"])])
    views = {}
    for t in cfg["levels"]:
        alpha, delta = alpha_delta(float(t))
        X = alpha * A + np.sqrt(delta) * rng_for("e39-noise", float(t)).standard_normal(A.shape)
        _, s_cf = closed_form_gaussian(X, prior.covariance(cfg["n_sites"]), alpha, delta)
        k = kalman_rts(X, prior.rho, prior.q, alpha, delta)
        s_rts = -(X - alpha * k.smoothed_means) / delta
        views[float(t)] = (X, alpha, delta, s_cf, s_rts)
    return prior, views


def _rows(prior, views, part, n_grid, half_width, projected=False):
    grid, w = make_grid(half_width, n_grid)
    log_K = prior.log_transition_matrix(grid)
    rows = []
    for t, (X, alpha, delta, s_cf, s_rts) in views.items():
        if projected:
            means, _ = grid_bp_projected_batch(grid, w, log_K, X, alpha, delta)
        else:
            means, _ = grid_bp_batch(grid, w, log_K, X, alpha, delta)
        s = -(X - alpha * np.asarray(means)) / delta
        rows.append(dict(part=part, arm="projected" if projected else "grid_bp", n_grid=n_grid, half_width=half_width,
                         spacing=2 * half_width / (n_grid - 1), t=t, err_closed_form=_rel(s, s_cf), err_rts=_rel(s, s_rts),
                         rts_vs_closed_form=_rel(s_rts, s_cf)))
    return rows


def part_grid(cfg, out):
    prior, views = _bundle(cfg)
    rows = []
    for n_grid in cfg["grid_sizes"]:
        rows += _rows(prior, views, "grid", int(n_grid), float(cfg["grid_half_width"]))
        print(f"  [E39] grid N_g={n_grid}: max err {max(r['err_closed_form'] for r in rows[-len(views):]):.2e}", flush=True)
    n_op, a_op = cfg["operating"]
    rows += _rows(prior, views, "grid", int(n_op), float(a_op), projected=True)
    return rows


def part_domain(cfg, out):
    prior, views = _bundle(cfg)
    rows = []
    for a in cfg["domain_half_widths"]:
        n_grid = int(round(2 * a / cfg["domain_spacing"])) + 1
        rows += _rows(prior, views, "domain", n_grid, float(a))
        print(f"  [E39] domain A={a} (N_g={n_grid}): max err {max(r['err_closed_form'] for r in rows[-len(views):]):.2e}", flush=True)
    return rows


PARTS = {"grid": part_grid, "domain": part_domain}


def main():
    ap = experiment_parser("e39_gaussian_sanity", "grid BP against the closed-form score and the RTS smoother")
    args = ap.parse_args()
    if args.list_parts:
        print("\n".join(PARTS)); return
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    out = ensure_dir(Path(args.output_dir))
    rows = []
    for name, fn in select_parts(PARTS, args.only).items():
        rows += fn(cfg, out)
        write_json(out / f"params_{name}.json", {**provenance(), "settings": cfg, "quick": bool(args.quick)})
    write_csv(out / "sanity.csv", rows)


if __name__ == "__main__":
    main()
