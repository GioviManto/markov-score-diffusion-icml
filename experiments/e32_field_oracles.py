#!/usr/bin/env python
"""E32 -- the phi^4 chain as a laboratory: its spectrum, its oracle ladder, and the theory on it.

Jerome's email of 1 October 2026 (after talking to Marc): test the approach on "real" Markov
data from a simple physical model, the stochastic Allen-Cahn / phi^4 field, whose stationary
law is a nearest-neighbour Gibbs measure. `markov_score.field` builds that chain exactly
(transfer operator, h-transform, two exact samplers). This experiment is DETERMINISTIC and
cheap, and it does three things BEFORE any learning run is registered:

spectrum    the (T, h) map of the chain: lambda_1/lambda_0 (the rho analogue), correlation
            length in sites, marginal scale and excess kurtosis, and the resolution of the
            operating grid (step width in grid cells). Plus the ISO-CORRELATION line: for each
            T, the spacing h at which lambda_1/lambda_0 = 0.85, the laboratory's frozen
            correlation, so the second laboratory can be compared to the first at matched
            second-order structure.
ladder      the oracle ladder on the confirmatory cells: exact BP (the reference), the LMMSE of
            the true covariance (the linear floor = the Gaussian closure of Proposition 1),
            the isotropic map, and the MOMENT-PROJECTED recursion as a diagnostic with its
            edge-mass record (`bp_grid.grid_bp_projected_batch`: on this chain the backward
            messages are nearly flat and the projection is not a well-posed estimator).
refinement  the exact score at (401, 8) against (801, 8) and (1601, 8) on the same data: the
            quadrature error of the operating grid on this family, before anything is fitted
            on it (the E04 / E11 gate, applied to a new family).
theory      Theorem 2 on the second laboratory: kappa_4 of the chain by transfer-matrix
            four-point functions, the predicted closure-gap constant (1/6)||kappa_4||_F^2, and
            the measured gap E||S - S^G||^2 e^{8t} from the ladder's data.

Cells. The T sweep sits on the iso-correlation line (rho_1 = 0.85): T in (0.3, 0.5, 1.0, 2.0)
with h solved per T (at h = 1 the line passes through T = 0.30, which is why the base cell is
there). The h sweep refines the base cell at FIXED physical length: (h, L) = (1, 32), (0.5, 64),
(0.25, 128), the reading of Jerome's prediction that the advantage of structure grows under
refinement because a fixed window's receptive field does not.

PREDECLARED (registry E32.json):
  P1  the operating grid's exact score is within 1e-3 relative of the (1601, 8) one at every
      level of every cell (FALSIFIED by any cell above 1e-2, which would move that cell to a
      finer grid in E33).
  P2  the moment-projected recursion is NOT well-posed on this chain: at some level of every
      cell its backward messages carry more than 10% of their mass in the outer 5% of the
      operating grid (a nearly flat message, see `bp_grid.grid_bp_projected_batch`), so the
      Gaussian-closure arm of the field chain is its covariance twin (the LMMSE of the true
      covariance), as Proposition 1 identifies the two on linear transitions. FALSIFIED if the
      edge mass stays below 10% at every level of some cell, in which case the projected arm
      is well-posed there and must be reported beside the floor.
  P3  the closure-gap ratio (measured / predicted) approaches 1 above t_* on every cell, as it
      does for the AR(1) families (E01).
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
from markov_score.bp_grid import grid_bp_batch, grid_bp_projected_batch, make_grid  # noqa: E402
from markov_score.field import Phi4Chain, h_for_correlation  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

T_SCHEDULE = tuple(FROZEN.t_grid)
RHO_TARGET = FROZEN.rho

# The confirmatory cells, shared with E33 (which imports CELLS from here). "iso" means the
# spacing is solved so that lambda_1 / lambda_0 = RHO_TARGET; the achieved value is recorded.
CELLS = (
    dict(id="T0.3_h1_L32",   T=0.3, h=1.0,   L=32,  role="base"),
    dict(id="T0.5_iso_L32",  T=0.5, h="iso", L=32,  role="tsweep"),
    dict(id="T1_iso_L32",    T=1.0, h="iso", L=32,  role="tsweep"),
    dict(id="T2_iso_L32",    T=2.0, h="iso", L=32,  role="tsweep"),
    dict(id="T0.3_h0.5_L64", T=0.3, h=0.5,   L=64,  role="refine"),
    dict(id="T0.3_h0.25_L128", T=0.3, h=0.25, L=128, role="refine"),
)


# Amendment 1 (1 Oct 2026, afternoon): the note's own axis. The registered T sweep sits on the
# iso-correlation line, which holds the correlation length in sites FIXED by construction; the
# consolidated note (section 24.6) reads the temperature at FIXED spacing, where lowering T
# lengthens the domains and sharpens the bimodality at the same time. Both axes are reported;
# the base cell belongs to both. `CELLS` itself is untouched (E33 imports it).
FIXED_H_TEMPERATURES = (0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0)
FIXED_H_CELLS = tuple(dict(id=f"T{T:g}_h1_L32", T=T, h=1.0, L=32, role="fixed_h") for T in FIXED_H_TEMPERATURES)
ALL_CELLS = CELLS + tuple(c for c in FIXED_H_CELLS if c["id"] not in {d["id"] for d in CELLS})


def find_cell(cell_id: str) -> dict:
    return next(c for c in ALL_CELLS if c["id"] == cell_id)


def resolve_cell(cell: dict) -> tuple[Phi4Chain, dict]:
    """The chain of a cell, with the iso-correlation spacing solved if asked for."""
    h = cell["h"]
    if h == "iso":
        h = h_for_correlation(cell["T"], RHO_TARGET)
    chain = Phi4Chain(cell["T"], float(h))
    info = dict(cell=cell["id"], T=cell["T"], h=float(h), n_sites=cell["L"], role=cell["role"],
                rho1=chain.rho, corr_length_sites=chain.spectrum.correlation_length_sites,
                scale=chain.scale, excess_kurtosis=chain.innovation_excess_kurtosis)
    return chain, info


SETTINGS = dict(
    temperatures=(0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0),
    spacings=(0.125, 0.25, 0.5, 1.0, 2.0),
    iso_temperatures=(0.3, 0.5, 0.7, 1.0, 1.5, 2.0),
    cells=tuple(c["id"] for c in CELLS),
    n_chains=512, reps=(0, 1),
    grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    refinement_grids=((401, 8.0), (801, 8.0), (1601, 8.0)),
    theory_cells=("T0.3_h1_L32", "T0.5_iso_L32", "T1_iso_L32", "T2_iso_L32"),
    theory_modes=256,
    fixed_h_cells=tuple(c["id"] for c in FIXED_H_CELLS), fixed_h_n_physics=4096,
)
QUICK = dict(temperatures=(0.3, 1.0), spacings=(0.5, 1.0), iso_temperatures=(0.5,),
             cells=("T0.3_h1_L32",), n_chains=16, reps=(0,), grid_size=201,
             refinement_grids=((201, 8.0), (401, 8.0)), theory_cells=("T0.3_h1_L32",), theory_modes=64,
             fixed_h_cells=("T0.3_h1_L32", "T1_h1_L32"), fixed_h_n_physics=128)


# ------------------------------------------------------------------ parts


def part_spectrum(cfg, out, workers):
    rows = []
    g = np.linspace(-cfg["half_width"], cfg["half_width"], cfg["grid_size"])
    dx = float(g[1] - g[0])
    for T in cfg["temperatures"]:
        for h in cfg["spacings"]:
            sp = Phi4Chain(T, h).spectrum
            lo, hi = sp.resolved
            rows.append(dict(T=T, h=h, rho1=sp.rho1, corr_length_sites=sp.correlation_length_sites,
                             scale=sp.scale, excess_kurtosis=sp.excess_kurtosis, lambda0=sp.lambda0,
                             resolved_half_width_phys=float(sp.u[hi - 1]),
                             step_sd_over_grid=float(np.sqrt(T * h) / sp.scale / dx)))
            print(f"  [spectrum] T={T} h={h}: rho1={sp.rho1:.4f} xi={sp.correlation_length_sites:.2f} "
                  f"kurt={sp.excess_kurtosis:+.3f}", flush=True)
    write_csv(out / "spectrum.csv", rows)
    iso = []
    for T in cfg["iso_temperatures"]:
        h = h_for_correlation(T, RHO_TARGET)
        sp = Phi4Chain(T, h).spectrum
        iso.append(dict(T=T, h=h, rho_target=RHO_TARGET, rho1=sp.rho1, scale=sp.scale,
                        excess_kurtosis=sp.excess_kurtosis, corr_length_sites=sp.correlation_length_sites,
                        step_sd_over_grid=float(np.sqrt(T * h) / sp.scale / dx)))
        print(f"  [iso] T={T}: h*={h:.4f} rho1={sp.rho1:.4f} kurt={sp.excess_kurtosis:+.3f}", flush=True)
    write_csv(out / "iso.csv", iso)


def _data(chain, L, n, rep, tag):
    rng = rng_for(tag, chain.T, chain.h, L, rep)
    A = chain.sample_batch(rng, n, L)
    noise = rng.standard_normal((len(T_SCHEDULE),) + A.shape)
    return A, noise


def _rel(m_hat, m_ref, X, al, de):
    s_hat = -(X - al * m_hat) / de
    s_ref = -(X - al * m_ref) / de
    return float(((s_hat - s_ref) ** 2).sum()), float((s_ref ** 2).sum())


def _ladder_cell(args):
    cfg, cell_id, rep = args
    cell = find_cell(cell_id)
    chain, info = resolve_cell(cell)
    L = cell["L"]
    g, w = make_grid(cfg["half_width"], cfg["grid_size"])
    log_k = chain.log_transition_matrix(g)
    log_mu = chain.log_stationary(g)
    sigma0 = chain.covariance(L)
    A, noise = _data(chain, L, cfg["n_chains"], rep, "e32-ladder")
    rows = []
    pooled = {a: [0.0, 0.0] for a in ("lmmse_true", "projected", "isotropic")}
    edge_max = [0.0]
    t0 = time.time()
    for li, t in enumerate(T_SCHEDULE):
        al, de = alpha_delta(t)
        X = al * A + np.sqrt(de) * noise[li]
        m_ref, _ = grid_bp_batch(g, w, log_k, X, al, de, log_mu=log_mu)
        m_pr, _, edge = grid_bp_projected_batch(g, w, log_k, X, al, de, log_mu=log_mu, return_diagnostics=True)
        arms = {
            "lmmse_true": theory.lmmse_posterior_mean(X, sigma0, t),
            "projected": m_pr,
            "isotropic": (X - de * X) / al,
        }
        gap = float(((m_ref - arms["lmmse_true"]) ** 2).sum()) * (al / de) ** 2 / len(A)
        for arm, m in arms.items():
            e, r = _rel(m, m_ref, X, al, de)
            pooled[arm][0] += e; pooled[arm][1] += r
            rows.append({**info, "rep": rep, "arm": arm, "level": t, "risk": float(np.sqrt(e / r)),
                         "score_gap_sq_per_seq": gap if arm == "lmmse_true" else np.nan,
                         "edge_mass": edge if arm == "projected" else np.nan,
                         "n_chains": len(A)})
        edge_max[0] = max(edge_max[0], edge)
    for arm, (e, r) in pooled.items():
        rows.append({**info, "rep": rep, "arm": arm, "level": -1.0, "risk": float(np.sqrt(e / r)),
                     "score_gap_sq_per_seq": np.nan, "edge_mass": edge_max[0] if arm == "projected" else np.nan,
                     "n_chains": len(A)})
    print(f"  [E32 ladder] {cell_id} rep={rep}: " + " ".join(
        f"{a}={np.sqrt(e / r):.4f}" for a, (e, r) in pooled.items())
        + f" edge_mass={edge_max[0]:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return rows


def part_ladder(cfg, out, workers):
    cells = [(cfg, c, r) for c in cfg["cells"] for r in cfg["reps"]]
    write_csv(out / "ladder.csv", run_cells(cells, _ladder_cell, workers, out, "ladder.csv"))


def _refinement_cell(args):
    cfg, cell_id, rep = args
    cell = next(c for c in CELLS if c["id"] == cell_id)
    chain, info = resolve_cell(cell)
    L = cell["L"]
    A, noise = _data(chain, L, cfg["n_chains"], rep, "e32-refine")
    grids = list(cfg["refinement_grids"])
    fine = grids[-1]
    gf, wf = make_grid(fine[1], fine[0])
    lkf, lmf = chain.log_transition_matrix(gf), chain.log_stationary(gf)
    rows = []
    for li, t in enumerate(T_SCHEDULE):
        al, de = alpha_delta(t)
        X = al * A + np.sqrt(de) * noise[li]
        m_fine, _ = grid_bp_batch(gf, wf, lkf, X, al, de, log_mu=lmf)
        for n_g, hw in grids[:-1]:
            g, w = make_grid(hw, n_g)
            m, _ = grid_bp_batch(g, w, chain.log_transition_matrix(g), X, al, de, log_mu=chain.log_stationary(g))
            e, r = _rel(m, m_fine, X, al, de)
            rows.append({**info, "rep": rep, "level": t, "grid_size": n_g, "half_width": hw,
                         "fine_grid_size": fine[0], "rel_error": float(np.sqrt(e / r)),
                         "max_abs_mean_diff": float(np.max(np.abs(m - m_fine)))})
    worst = max(r["rel_error"] for r in rows if r["grid_size"] == cfg["grid_size"]) if any(
        r["grid_size"] == cfg["grid_size"] for r in rows) else float("nan")
    print(f"  [E32 refine] {cell_id} rep={rep}: worst rel error of the operating grid {worst:.2e}", flush=True)
    return rows


def part_refinement(cfg, out, workers):
    cells = [(cfg, c, r) for c in cfg["cells"] for r in cfg["reps"]]
    write_csv(out / "refinement.csv", run_cells(cells, _refinement_cell, workers, out, "refinement.csv"))


def _theory_cell(args):
    cfg, cell_id = args
    cell = next(c for c in CELLS if c["id"] == cell_id)
    chain, info = resolve_cell(cell)
    L = cell["L"]
    t0 = time.time()
    sigma0 = chain.covariance(L)
    lam_max = float(np.linalg.eigvalsh(sigma0).max())
    kap = chain.fourth_cumulant_tensor(L, cfg["theory_modes"])
    c4 = float((kap ** 2).sum()) / 6.0
    import pandas as pd
    ladder = pd.read_csv(Path(cfg["_out"]) / "ladder.csv")
    sub = ladder[(ladder.cell == cell_id) & (ladder.arm == "lmmse_true") & (ladder.level > 0)]
    rows = []
    for t, grp in sub.groupby("level"):
        gap = float(grp.score_gap_sq_per_seq.mean())
        rows.append({**info, "level": float(t), "measured_gap": gap, "predicted_gap": c4 * np.exp(-8.0 * t),
                     "ratio": gap / (c4 * np.exp(-8.0 * t)), "c4": c4, "kappa4_frob": float(np.sqrt((kap ** 2).sum())),
                     "lambda_max": lam_max, "t_star": 0.5 * np.log(lam_max), "n_reps": int(grp.rep.nunique())})
    print(f"  [E32 theory] {cell_id}: C4={c4:.4g} t*={0.5 * np.log(lam_max):.3f} "
          + " ".join(f"t={r['level']:.2f}:{r['ratio']:.3f}" for r in rows[-4:]) + f" ({time.time() - t0:.0f}s)", flush=True)
    return rows


def part_theory(cfg, out, workers):
    if not (out / "ladder.csv").exists():
        raise SystemExit("theory needs ladder.csv: run --only ladder first")
    cfg = {**cfg, "_out": str(out)}
    cells = [(cfg, c) for c in cfg["theory_cells"]]
    write_csv(out / "theory.csv", run_cells(cells, _theory_cell, workers, out, "theory.csv"))


def _fixed_h_cell(args):
    """The oracle ladder of a fixed-spacing cell plus the physical statistics of 4096 exact
    chains (E34's `physical_statistics`, imported at call time: e34 imports this module)."""
    cfg, cell_id, rep = args
    cell = find_cell(cell_id)
    chain, info = resolve_cell(cell)
    from e34_field_physics import physical_statistics
    A = chain.sample_batch(rng_for("e32-fixed-h-physics", chain.T, chain.h, cell["L"], rep), cfg["fixed_h_n_physics"], cell["L"])
    st = physical_statistics(A, chain.scale, cell["L"], chain.h)
    # scalars only (C(r) is in the spectrum); every row carries every column, the CSV writer needs that
    st = {k: v for k, v in st.items() if np.isscalar(v)}
    st.update(kink_density_physical=st["kink_density"] / chain.h, mean_domain_physical=st["mean_domain"] * chain.h,
              coarse_kink_density_physical=st["coarse_kink_density"] / chain.h,
              coarse_mean_domain_physical=st["coarse_mean_domain"] * chain.h)
    blank = {k: np.nan for k in st}
    rows = [{**r, "kind": "ladder", **blank} for r in _ladder_cell(args)]
    rows.append({**info, "rep": rep, "arm": "physics", "level": np.nan, "risk": np.nan, "score_gap_sq_per_seq": np.nan,
                 "edge_mass": np.nan, "n_chains": len(A), "kind": "physics", **st})
    print(f"  [E32 fixed_h] {cell_id} rep={rep}: xi={info['corr_length_sites']:.2f} kinks/bond={st['kink_density']:.4f} "
          f"coarse={st['coarse_kink_density']:.4f} bimod={st['bimodality_coefficient']:.3f}", flush=True)
    return rows


def part_fixed_h(cfg, out, workers):
    cells = [(cfg, c, r) for c in cfg["fixed_h_cells"] for r in cfg["reps"]]
    write_csv(out / "fixed_h.csv", run_cells(cells, _fixed_h_cell, workers, out, "fixed_h.csv"))


PARTS = {"spectrum": part_spectrum, "ladder": part_ladder, "refinement": part_refinement, "theory": part_theory,
         "fixed_h": part_fixed_h}


def main():
    ap = experiment_parser("e32_field_oracles", "the phi^4 chain: spectrum, oracle ladder, refinement, theory")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.list_parts:
        print("\n".join(PARTS)); return
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    out = ensure_dir(Path(args.output_dir))
    cells_info = [resolve_cell(c)[1] for c in CELLS if c["id"] in cfg["cells"]]
    write_json(out / "cells.json", cells_info)
    for name, fn in select_parts(PARTS, args.only).items():
        fn(cfg, out, args.workers)
        write_json(out / f"params_{name}.json", {**provenance(), "settings": {k: v for k, v in cfg.items() if not k.startswith("_")},
                                                 "cells": cells_info, "quick": bool(args.quick), "workers": args.workers})


if __name__ == "__main__":
    main()
