"""E44: non-equilibrium snapshots. What does an estimator built for stationary Markov data do on a quench?

The paper's estimator assumes a stationary Markov chain in space. The physical model's stationary states are Gibbs, but a physicist's
snapshots are often taken during coarsening after a quench from high temperature, when they are NOT: domains of +-1 whose size grows
with time (markov_score.allen_cahn: the lattice Langevin dynamics of the same Hamiltonian, validated against the exact chain). E44
trains and tests on snapshots at several times after the quench, with the exact Gibbs chain at tau = infinity as the control.

The estimand. There is no exact score for a non-equilibrium law, so the risk uses the CLEAN SNAPSHOT as the reference: E33's pooled
relative score error with `m_ref` = the clean sequence, i.e. the relative denoising-score error (the score a corruption-based loss
targets). It is not zero for a perfect denoiser, so the readings are RATIOS against a comparator fitted on the same data: the
empirical linear estimator (the Gaussian posterior mean with the training snapshots' own sample covariance), which needs no model of
the dynamics. Fit protocol, EM ladder, checkpoint selection and the arms' constructors are E33's (e33._fit_arm, imported).

Cell: T = 0.3, h = 1, L = 32 (E33's base cell), standardised by the equilibrium scale. Arms em_pair_quartic (the site energy: exactly
specified at equilibrium, 5 parameters) and em_tilted_spline (25 parameters), and linear_empirical. M = 512; seeds 0-7.

Predictions (experiments/registry/E44.json), R = arm risk / linear_empirical risk, paired per seed:
  P1  tau = infinity: the site energy beats the empirical linear estimator in 8/8 seeds (R < 1)
  P2  the advantage survives the quench: R < 1 for the site energy in >= 7/8 seeds at EVERY tau
  P3  the order flips off equilibrium: at tau = 1.5, 3 and 6 the spline's risk is below the site energy's in >= 7/8 seeds
  P4  at equilibrium the site energy's risk is below the spline's in >= 7/8 seeds
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e33_field_learning as e33  # noqa: E402
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from e32_field_oracles import CELLS, resolve_cell  # noqa: E402
from markov_score import theory  # noqa: E402
from markov_score.allen_cahn import quench, windows  # noqa: E402
from markov_score.bp_grid import make_grid  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

e14 = e33.e14
CELL = "T0.3_h1_L32"
STRIDE = 64                                   # windows start every 64 sites of the 256-site ring: 32 sites apart, beyond the correlation length

SETTINGS = dict(e33.SETTINGS, cells=(CELL,), taus=(1.5, 3.0, 6.0, 12.0, float("inf")), sizes=(512,), seeds=tuple(range(8)),
                n_val=512, n_test=1024, arms=("em_pair_quartic", "em_tilted_spline"), sim_sites=256, sim_dt=0.02)
QUICK = dict(e33.QUICK, taus=(3.0,), n_val=32, n_test=32, sim_sites=128, arms=("em_pair_quartic",))


def _draw(cfg, chain, scale, tau, tag, seed, m, L):
    """m sequences of length L from the process at time tau after the quench (tau = inf: the exact Gibbs chain), standardised."""
    rng = rng_for("e44", tag, tau, seed)
    if np.isinf(tau):
        return chain.sample_batch(rng, m, L)
    per_run = len(range(0, cfg["sim_sites"] - L + 1, STRIDE))
    phi = quench(rng, 0.3, 1.0, tau, int(np.ceil(m / per_run)), cfg["sim_sites"], cfg["sim_dt"])
    return windows(phi, L, STRIDE)[:m] / scale


def _bundle(A, tag, tau, seed):
    """Noised views at every level of the schedule, the CLEAN sequence as the reference mean."""
    rng = rng_for("e44-noise", tag, tau, seed)
    out = {}
    for t in e33.T_SCHEDULE:
        al, de = alpha_delta(t)
        out[t] = (al * A + np.sqrt(de) * rng.standard_normal(A.shape), A)
    return out


def _cell(args):
    cfg, tau, seed, n = args
    cell = next(c for c in CELLS if c["id"] == CELL)
    chain, info = resolve_cell(cell)
    L, s = cell["L"], info["scale"]
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    A = _draw(cfg, chain, s, tau, "train", seed, n, L)
    Av = _draw(cfg, chain, s, tau, "valchains", seed, cfg["n_val"], L)
    val = _bundle(_draw(cfg, chain, s, tau, "valbundle", seed, cfg["n_val"], L), "val", tau, seed)
    test = _bundle(_draw(cfg, chain, s, tau, "testbundle", seed, cfg["n_test"], L), "test", tau, seed)
    var = float(A.var())
    base = dict(cell=CELL, tau=tau, seed=seed, n_chains=n, T=info["T"], h=info["h"], n_sites=L,
                var_a=var, rho1_emp=float((A[:, :-1] * A[:, 1:]).mean() / var), kurt_emp=float((A**4).mean() / var**2 - 3.0),
                test_bundle=e14._bundle_hash(test))
    cid = f"{CELL}_tau{tau:g}"
    groups = e14._one_view_groups(A, f"e44-em-{cid}-M{n}", seed, L)
    val_groups = e14._one_view_groups(Av, f"e44-emval-{cid}", seed, L)
    rows = []
    for arm in cfg["arms"]:
        for r in e33._fit_arm(cfg, arm, grid, w, groups, val, val_groups, test, seed, cid, L, n):
            rows.append({**base, **e33.ROW_DEFAULTS, **r})
    sigma_hat = (A.T @ A) / len(A)
    rows.append({**base, **e33.ROW_DEFAULTS, "arm": "linear_empirical", "selection": "risk", "failed": 0,
                 "risk": e14._pooled(lambda X, t: theory.lmmse_posterior_mean(X, sigma_hat, t), test)})
    print(f"  [E44] tau={tau:g} seed={seed} M={n} (var {var:.2f}, rho1 {base['rho1_emp']:.2f}): " + " ".join(
        f"{r['arm']}={r['risk']:.4f}" for r in rows if r["selection"] == "risk"), flush=True)
    return rows


def part_quench(cfg, out, workers):
    cells = [(cfg, tau, s, n) for n in sorted(cfg["sizes"], reverse=True) for tau in cfg["taus"] for s in cfg["seeds"]]
    print(f"  quench: {len(cells)} cells", flush=True)
    write_csv(out / "quench.csv", run_cells(cells, _cell, workers, out, "quench.csv", key=lambda c: f"tau{c[1]:g}_s{c[2]}_M{c[3]}"))


PARTS = {"quench": part_quench}


def main():
    ap = experiment_parser("e44_quench", "non-equilibrium snapshots: the estimator built for stationary Markov data on a quench")
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
