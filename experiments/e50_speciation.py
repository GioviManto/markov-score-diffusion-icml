"""E50: speciation on the phi^4 field. When does the exact denoiser commit to the sign of a domain, and is
the lattice chain the discretised SPDE as far as its kinks are concerned?

Jerome's email of 30 Sep 2026 (item 10) and the consolidated note (sections 18-20) ask to use the
temperature sweep to watch the denoiser commit to whole domains as the corruption falls, with Theta (the
physical temperature) kept apart from t (the corruption level), and state that there is no 1D phase
transition but a sharp crossover with xi ~ exp(E_kink / Theta). Never measured before E50. Two parts,
both deterministic in their inputs and run on the laptop with the TRUE kernel (no learning):

commitment  per cell, 2 x 2048 exact chains noised at 48 levels t in [0.02, 4]; the exact posterior's
            sign probability p_i = P(a_i > 0 | x_t) by grid BP (`return_sign_prob`), the commitment
            |2 p_i - 1| over all sites and over the central sites (`commitment.bulk_sites`), the share of
            signs guessed right, the posterior-mean MSE against the clean field (the MMSE, the note's
            section 24.5) and the linear floor's MSE on the same data; and the GAUSSIAN TWIN (same
            covariance) in closed form (`commitment.gaussian_twin`). Cells: E32's fixed-spacing axis
            (h = 1, L = 32, Theta = 0.2 ... 2), a length series at Theta = 0.2 (xi = 19.5 sites; L = 4,
            8, 16 below xi, 64, 128, 256 above it; the note's section 20), and E32's other five cells.
kink_energy the correlation length of the transfer operator on a (Theta, h) ladder down to h = 1/32 and
            Theta = 0.1 (eigensolve only, `field.solve_spectrum`). The dilute-kink-gas (instanton)
            result for the continuum SPDE: xi_phys = Theta / Delta E with Delta E ~ Theta^{1/2}
            exp(-E_kink / Theta), so ln xi_phys = E_kink / Theta + (1/2) ln Theta + c + O(Theta), with the
            kink energy E_kink = int_{-1}^{1} sqrt(2 V) = 2 sqrt(2) / 3 for V = (phi^2 - 1)^2 / 4.

Predictions (experiments/registry/E50.json), read by tools/make_extension_numbers.py (read_e50_*):
  P1  calibration: on every cell the share of signs guessed right equals (1 + commitment) / 2 to within
      0.01 on average over levels, and within 0.05 at every level
  P2  fixed-spacing axis: the commitment time t_c (all sites) falls strictly as Theta rises (7/7 pairs)
  P3  the exact posterior commits EARLIER (at larger t) than its Gaussian twin at Theta <= 0.5 (4/4 cells),
      and the excess t_c - t_c^twin is larger at Theta = 0.2 than at Theta = 2
  P4  length series (central sites): t_c rises with L from 4 to 64 (each step > 0.02) and moves by less
      than 0.03 from L = 128 to 256
  P5  below xi (L = 4, 8, 16): the slope of the central t_c against (1/2) ln(1 + Lambda_max(L)), Lambda_max
      the largest eigenvalue of the L x L covariance (the speciation scale of Biroli et al. 2024), lies in
      [0.8, 1.2]
  P6  at h = 1/16 and 1/32 the fit ln xi_phys = a / Theta + b ln Theta + c + d Theta over Theta in [0.1, 0.3]
      gives a within 2% of 2 sqrt(2) / 3 = 0.9428
  P7  and b in [0.35, 0.65]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from e32_field_oracles import CELLS, FIXED_H_CELLS, resolve_cell  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score import theory  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.commitment import bulk_sites, commitment, gaussian_twin, sign_correct  # noqa: E402
from markov_score.field import solve_spectrum  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

LSERIES_T = 0.2
LSERIES = tuple(dict(id=f"T0.2_h1_L{L}", T=LSERIES_T, h=1.0, L=L, role="lseries") for L in (4, 8, 16, 64, 128, 256))
EXTRA = tuple(c for c in CELLS if c["id"] not in {d["id"] for d in FIXED_H_CELLS})
ALL = FIXED_H_CELLS + LSERIES + EXTRA

SETTINGS = dict(cells=tuple(c["id"] for c in ALL), levels=tuple(float(f"{t:.4g}") for t in np.geomspace(0.02, 4.0, 48)),
                n_chains=2048, reps=(0, 1), grid_size=FROZEN.n_grid, half_width=FROZEN.half_width, bulk_half_max=8,
                kink_temperatures=(0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3),
                kink_spacings=(1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125), kink_n_fine=2401, kink_half_width=12.0,
                kink_check_spacings=(0.0625, 0.03125), kink_check_n_fine=4801)
QUICK = dict(cells=("T0.3_h1_L32", "T0.2_h1_L4"), levels=(0.1, 0.5, 1.0, 2.0), n_chains=64, reps=(0,), grid_size=201,
             kink_temperatures=(0.2, 0.3), kink_spacings=(1.0, 0.5), kink_n_fine=801, kink_check_spacings=(0.5,),
             kink_check_n_fine=1201)


def _cell(args):
    cfg, cell_id, rep = args
    cell = next(c for c in ALL if c["id"] == cell_id)
    chain, info = resolve_cell(cell)
    L = cell["L"]
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    log_k, log_mu = chain.log_transition_matrix(grid), chain.log_stationary(grid)
    cov = chain.covariance(L)
    lam_max = float(np.linalg.eigvalsh(cov)[-1])
    bulk = bulk_sites(L, cfg["bulk_half_max"])
    A = chain.sample_batch(rng_for("e50-chains", cell_id, rep), cfg["n_chains"], L)
    rows = []
    for t in cfg["levels"]:
        al, de = alpha_delta(t)
        X = al * A + np.sqrt(de) * rng_for("e50-noise", cell_id, rep, t).standard_normal(A.shape)
        mean, var, p = grid_bp_batch(grid, w, log_k, X, al, de, log_mu=log_mu, return_sign_prob=True)
        c, ok = commitment(p), sign_correct(p, A)
        c_tw, v_tw = gaussian_twin(cov, al, de)
        lin = theory.lmmse_posterior_mean(X, cov, t)
        rows.append(dict(info, rep=rep, n_chains=cfg["n_chains"], lambda_max=lam_max, level=t, alpha=al, delta=de,
                         commit_all=float(c.mean()), commit_bulk=float(c[:, bulk].mean()),
                         correct_all=float(ok.mean()), correct_bulk=float(ok[:, bulk].mean()),
                         twin_commit_all=float(c_tw.mean()), twin_commit_bulk=float(c_tw[bulk].mean()),
                         mse_exact=float(((mean - A) ** 2).mean()), post_var_exact=float(var.mean()),
                         mse_linear=float(((lin - A) ** 2).mean()), post_var_twin=float(v_tw.mean()),
                         n_bulk=len(bulk)))
    r = {x["level"]: x for x in rows}
    print(f"  [E50] {cell_id} rep={rep}: xi={info['corr_length_sites']:.2f} Lambda={lam_max:.2f} commit(t=0.5)="
          f"{min(r.items(), key=lambda kv: abs(kv[0] - 0.5))[1]['commit_all']:.3f}", flush=True)
    return rows


def part_commitment(cfg, out, workers):
    # largest cells first so the pool's tail is short
    size = {c["id"]: c["L"] for c in ALL}
    cells = [(cfg, c, r) for c in sorted(cfg["cells"], key=lambda c: -size[c]) for r in cfg["reps"]]
    print(f"  commitment: {len(cells)} cells x {len(cfg['levels'])} levels", flush=True)
    write_csv(out / "commitment.csv", run_cells(cells, _cell, workers, out, "commitment.csv", key=lambda c: f"{c[1]}_r{c[2]}"))


def _kink(args):
    cfg, T, h, n_fine = args
    sp = solve_spectrum(float(T), float(h), n_fine, cfg["kink_half_width"])
    xi = sp.correlation_length_sites
    row = dict(T=T, h=h, rho1=sp.rho1, xi_sites=xi, xi_phys=h * xi, scale=sp.scale, excess_kurtosis=sp.excess_kurtosis,
               lambda0=sp.lambda0, lambda1=float(sp.lam[1]), n_fine=n_fine, half_width=cfg["kink_half_width"],
               step_sd_over_dx=float(np.sqrt(T * h) / (sp.u[1] - sp.u[0])))
    return [row]


def part_kink_energy(cfg, out, workers):
    # the registered ladder on the registered eigensolve grid, plus the two finest spacings on a grid twice as fine
    # (a check of the eigensolve's own discretisation, read descriptively)
    cells = [(cfg, T, h, cfg["kink_n_fine"]) for h in cfg["kink_spacings"] for T in cfg["kink_temperatures"]]
    cells += [(cfg, T, h, cfg["kink_check_n_fine"]) for h in cfg["kink_check_spacings"] for T in cfg["kink_temperatures"]]
    write_csv(out / "kink.csv", run_cells(cells, _kink, workers, out, "kink.csv", key=lambda c: f"T{c[1]:g}_h{c[2]:g}_n{c[3]}"))


PARTS = {"commitment": part_commitment, "kink_energy": part_kink_energy}


def main():
    ap = experiment_parser("e50_speciation", "commitment of the exact denoiser to domains; the kink energy of the lattice chain")
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
