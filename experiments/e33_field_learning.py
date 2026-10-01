#!/usr/bin/env python
"""E33 -- learning the phi^4 chain through noise: Jerome's four arms, and the ones that bracket them.

The email of 1 October 2026 asked for a concrete test on "real" Markov data from a physical
model, with four arms tracked across temperature and lattice spacing: EM with a learned pair
energy, the Gaussian closure, the window CNN, and the exact score. This driver runs them on
the cells E32 fixed (`e32_field_oracles.CELLS`: the iso-correlation temperature sweep and the
fixed-physical-length refinement), under the registered protocol (sizes, seeds, bundles,
ladder, selection rules), and adds the arms that make the four readable:

    exact score          the reference every row is scored against (E32: BP with the true
                         kernel and the true stationary law, validated by refinement)
    linear floor         LMMSE of the true covariance -- what second-order structure buys
    gaussian closure     = the linear floor, by Proposition 1's identification; the
                         moment-projected recursion is recorded only as a diagnostic, because
                         on this chain its backward messages are nearly flat (E32, P2)
    em_mixture           the paper's estimator, C = 8 innovation mixture, 25 parameters.
                         MISSPECIFIED here: the true drift is nonlinear.
    em_tilted_poly       the learned pair energy, parsimonious: Gaussian step + Legendre tilt
                         of degree 6 (7 parameters). Exactly specified up to the basis.
    em_tilted_spline     the learned pair energy, flexible: 24 cubic B-splines (25 parameters,
                         the mixture's count). Exactly specified.
    em_mdn               the neural kernel (E25/E30): a state-dependent mixture from a small
                         network, 1,516 parameters. Expressive enough for a nonlinear drift.
    window / transformer the window head (Jerome's CNN) and the global attention model, each
                         re-screened per cell and trained two ways, exactly E14's arms:
                         ISM on the one-view bundle (the matched interface) and DSM on clean
                         targets with one fixed corruption (the network-favourable bracket).

Every kernel arm: EM through exact BP on ONE noised view per sequence (the sequences split
over the twelve levels), the same ladder to 1200 for every kernel (E30's lesson: a mismatched
cap decides the comparison), the kernel's OWN invariant density as the initial law (the model
is a stationary chain, as the data are), and both selection rules of E14 -- held-out BP risk
(registered) and held-out one-view evidence (oracle-free). The four kernels start from the
same (rho, var) = (0.3, 0.8) Gaussian embedding where the family allows it.

Parts (disjoint CSVs; the cluster cases shard `kernels` and `nets` by seed into their own
output directories, N-27):
    oracles   per (cell, seed): floor, closure, isotropic on the test bundle, with the
              closure's edge-mass diagnostic
    kernels   per (cell, seed, M): the four kernel arms
    screen    per (cell, dev seed, arch, hp): the networks' hyper-parameters, 8k ISM steps
    nets      per (cell, seed, M in net_sizes): the per-cell winners, ISM and DSM, 32k steps

PREDECLARED (registry E33.json, written before any confirmatory cell ran):
  P1  em_mixture keeps an asymptotic bias on every cell: its risk at M = 2048 is above
      em_tilted_spline's in 16/16 seeds, and the gap does NOT close from M = 512 to 2048.
  P2  the learned pair energy beats the mixture at every M >= 128 on every cell (paired,
      16/16), and the parsimonious tilt is within 20% of the flexible one at M = 2048.
  P3  em_mdn improves relative to em_mixture compared with E30 (where it lost 2.1-2.9x on
      the AR(1) chain): on the field chain the ratio mdn / mixture is below 1.5 at M = 2048.
  P4  no network reaches the learned pair energy at any M on any cell (16/16 paired), under
      either interface.
  P5  the window head's deficit relative to exact BP GROWS along the refinement cells
      (h = 1 -> 0.5 -> 0.25 at fixed physical length) while em_tilted_spline's does not.
  P6  the linear floor is HIGHER at the base cell (T = 0.3, marginal kurtosis -1.35) than
      E14b's bimodal floor 0.082 (kurtosis -0.72) at the same correlation, and it falls
      monotonically along the temperature sweep as the marginal approaches a Gaussian.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score import theory  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, grid_bp_projected_batch, make_grid  # noqa: E402
from markov_score.em import e_step_multi, fit_em  # noqa: E402
from markov_score.kernels import MDNKernel, MixtureInnovationKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.stationary import invariant_log_density_eig  # noqa: E402
from markov_score.tilted_kernel import TiltedWalkKernel  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402
from protocol import PROTOCOL  # noqa: E402

# E14 is imported, not copied: the network arms are the same estimators, trained by the same
# functions, as in the regime comparison the paper reports.
import e14_regime_baselines as e14  # noqa: E402
from e32_field_oracles import CELLS, resolve_cell  # noqa: E402

T_SCHEDULE = tuple(FROZEN.t_grid)
KERNEL_ARMS = ("em_mixture", "em_tilted_poly", "em_tilted_spline", "em_mdn")

SETTINGS = dict(
    cells=tuple(c["id"] for c in CELLS),
    sizes=PROTOCOL.sizes, net_sizes=(2048, 128),
    n_chains=2048, n_val=PROTOCOL.n_val, n_test=PROTOCOL.n_test,
    seeds=PROTOCOL.seeds, dev_seeds=PROTOCOL.dev_seeds[:2],
    grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    arms=KERNEL_ARMS, components=8, poly_degree=6, spline_basis=24, mdn_components=4, mdn_hidden=32,
    em_checkpoints=PROTOCOL.em_checkpoints, em_init_rho=PROTOCOL.em_init_rho, em_init_var=PROTOCOL.em_init_var,
    archs=tuple(e14.GRIDS), net_steps=PROTOCOL.net_steps, net_checkpoints=PROTOCOL.net_checkpoints,
    screen_steps=8_000, mode=PROTOCOL.parameterizations[0],
    lrs_by_arch={"window": 1e-3, "transformer": 3e-3},
    trace_by_arch={"window": "coloured", "transformer": "hutchinson"},
    batch_size=PROTOCOL.batch_size, grad_clip=PROTOCOL.grad_clip,
    winners=None, torch_threads=1, device="auto",
)
QUICK = dict(cells=("T0.3_h1_L32",), sizes=(32,), net_sizes=(32,), n_chains=64, n_val=32, n_test=32,
             seeds=(10_000,), dev_seeds=(10_000,), grid_size=201, em_checkpoints=(5, 10),
             net_steps=100, net_checkpoints=(50, 100), screen_steps=100, archs=("window",),
             arms=("em_mixture", "em_tilted_poly"))


# ------------------------------------------------------------------ bundles


def _bundle(chain, log_k, log_mu, grid, w, tag, seed, cell_id, n_chains, L):
    rng = rng_for(tag, seed, cell_id)
    A = chain.sample_batch(rng, n_chains, L)      # one generator, every chain a fresh draw
    out = {}
    for t in T_SCHEDULE:
        al, de = alpha_delta(t)
        X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
        m, _ = grid_bp_batch(grid, w, log_k, X, al, de, log_mu=log_mu)
        out[t] = (X, m)
    return A, out


def _common(cfg, cell_id, seed, n):
    cell = next(c for c in CELLS if c["id"] == cell_id)
    chain, info = resolve_cell(cell)
    L = cell["L"]
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    log_k = chain.log_transition_matrix(grid)
    log_mu = chain.log_stationary(grid)
    _, val = _bundle(chain, log_k, log_mu, grid, w, "e33-val", seed, cell_id, cfg["n_val"], L)
    _, test = _bundle(chain, log_k, log_mu, grid, w, "e33-test", seed, cell_id, cfg["n_test"], L)
    A = chain.sample_batch(rng_for("e33-train", seed, cell_id, n), n, L)
    Av = chain.sample_batch(rng_for("e33-valchains", seed, cell_id), cfg["n_val"], L)
    assert len(np.unique(A[:, 0])) > 1 and len(np.unique(Av[:, 0])) > 1, "degenerate training set"
    sigma0 = chain.covariance(L)
    base = dict(**info, seed=seed, n_chains=n,
                t_star=float(0.5 * np.log(np.linalg.eigvalsh(sigma0).max())),
                val_bundle=e14._bundle_hash(val), test_bundle=e14._bundle_hash(test))
    return chain, grid, w, log_k, log_mu, sigma0, A, Av, val, test, base


# ------------------------------------------------------------------ kernel arms


def _stationary_mu(grid, w):
    """The kernel's own invariant density as the initial law (direct eigensolve), with a loud
    fallback that is also recorded: a run whose fallback fired is a run to read with care."""
    def mu(kernel):
        try:
            return invariant_log_density_eig(kernel, grid, w).log_density
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            warnings.warn(f"invariant law unavailable for {kernel.name}: {exc}; standard normal used", RuntimeWarning)
            return -0.5 * grid**2 - 0.5 * np.log(2.0 * np.pi)
    return mu


def _init_kernel(arm, cfg, grid, seed, cell_id, n):
    rng = rng_for("e33-init", arm, seed, cell_id, n)
    rho, var = cfg["em_init_rho"], cfg["em_init_var"]
    if arm == "em_mixture":
        return MixtureInnovationKernel.init(cfg["components"], rho=rho, var=var, rng=rng)
    if arm == "em_tilted_poly":
        return TiltedWalkKernel.init("legendre", cfg["poly_degree"], grid, rho=rho, var=var)
    if arm == "em_tilted_spline":
        return TiltedWalkKernel.init("bspline", cfg["spline_basis"], grid, rho=rho, var=var)
    if arm == "em_mdn":
        return MDNKernel.init(cfg["mdn_components"], cfg["mdn_hidden"], rng)
    raise ValueError(arm)


def _kernel_diagnostics(kernel, grid):
    d = {}
    if isinstance(kernel, MixtureInnovationKernel):
        d["fit_rho"] = float(kernel.rho)
        d["fit_kurt"] = float(kernel.innovation_moments["innovation_excess_kurtosis"])
        d["inner_converged"] = bool(kernel.inner_converged)
    elif isinstance(kernel, TiltedWalkKernel):
        td = kernel.tilt_diagnostics(grid)
        d["fit_gamma"] = td["gamma"]; d["tilt_range"] = td["tilt_range"]
        d["step_resolved"] = bool(td["resolved"]); d["inner_converged"] = bool(kernel.inner_converged)
    return d


def _fit_arm(cfg, arm, grid, w, groups, val, val_groups, test, seed, cell_id, L, n):
    gs = [(X, *alpha_delta(t)) for X, t in groups]
    held = [(X, *alpha_delta(t)) for X, t in val_groups]
    n_edges = sum(len(X) for X, _ in val_groups) * (L - 1)
    mu_fn = _stationary_mu(grid, w)
    k0 = _init_kernel(arm, cfg, grid, seed, cell_id, n)
    ck = set(int(c) for c in cfg["em_checkpoints"])
    t0 = time.time()
    _, trace, saved = fit_em(k0, grid, w, gs, n_iters=max(ck), tol=FROZEN.em_loglik_tol,
                             checkpoints=ck, log_mu=mu_fn)
    wall = time.time() - t0
    curve, mus = {}, {}
    best, best_v, best_e, best_ev = None, np.inf, None, -np.inf
    for it in sorted(saved):
        mu = mu_fn(saved[it]); mus[it] = mu
        lk = saved[it].log_transition_matrix(grid)
        v = e14._pooled(lambda X, t, lk=lk, mu=mu: grid_bp_batch(grid, w, lk, X, *alpha_delta(t), log_mu=mu)[0], val)
        ev = float(e_step_multi(grid, w, lk, held, mu).log_evidence) / n_edges
        curve[it] = (v, ev)
        if v < best_v:
            best, best_v = it, v
        if ev > best_ev:
            best_e, best_ev = it, ev
    rows = []
    for selection, it, sel in (("risk", best, best_v), ("evidence", best_e, best_ev)):
        kb = saved[it]
        lk = kb.log_transition_matrix(grid)
        risk = e14._pooled(lambda X, t, lk=lk, mu=mus[it]: grid_bp_batch(grid, w, lk, X, *alpha_delta(t), log_mu=mu)[0], test)
        rows.append(dict(arm=arm, selection=selection, risk=risk, checkpoint=it, at_cap=int(it == max(saved)),
                         val_selected=sel, n_params=int(len(np.asarray(kb.theta))), wall=wall,
                         converged=bool(trace.converged), stop_reason=trace.stop_reason,
                         monotone_violation=float(trace.monotone_violation),
                         n_iters=len(trace.log_evidence), **_kernel_diagnostics(kb, grid),
                         theta=";".join(f"{v:.8g}" for v in np.asarray(kb.theta, dtype=float)),
                         val_curve=";".join(f"{i}:{a:.5f}/{b:.5f}" for i, (a, b) in sorted(curve.items()))))
    return rows


ROW_DEFAULTS = dict(arm="", selection="", arch="", hp="", checkpoint=-1, at_cap=0, val_selected=np.nan,
                    wall=np.nan, n_params=0, converged=False, stop_reason="", monotone_violation=np.nan,
                    n_iters=0, fit_rho=np.nan, fit_kurt=np.nan, inner_converged=False, fit_gamma=np.nan,
                    tilt_range=np.nan, step_resolved=False, theta="", val_curve="", edge_mass=np.nan)


def _kernels_cell(args):
    cfg, cell_id, seed, n = args
    chain, grid, w, log_k, log_mu, sigma0, A, Av, val, test, base = _common(cfg, cell_id, seed, n)
    L = chain and base["n_sites"]
    groups = e14._one_view_groups(A, f"e33-em-{cell_id}-M{n}", seed, L)
    val_groups = e14._one_view_groups(Av, f"e33-emval-{cell_id}", seed, L)
    rows = []
    for arm in cfg["arms"]:
        for r in _fit_arm(cfg, arm, grid, w, groups, val, val_groups, test, seed, cell_id, L, n):
            rows.append({**base, **ROW_DEFAULTS, **r})
    print(f"  [E33 kernels] {cell_id} seed={seed} M={n}: " + " ".join(
        f"{r['arm']}={r['risk']:.4f}" for r in rows if r["selection"] == "risk"), flush=True)
    return rows


# ------------------------------------------------------------------ oracles


def _oracles_cell(args):
    cfg, cell_id, seed = args
    chain, grid, w, log_k, log_mu, sigma0, A, Av, val, test, base = _common(cfg, cell_id, seed, cfg["n_chains"])
    rows = []
    rows.append({**base, **ROW_DEFAULTS, "arm": "lmmse_true",
                 "risk": e14._pooled(lambda X, t: theory.lmmse_posterior_mean(X, sigma0, t), test)})
    edge = 0.0

    def closure(X, t):
        nonlocal edge
        m, _, e = grid_bp_projected_batch(grid, w, log_k, X, *alpha_delta(t), log_mu=log_mu, return_diagnostics=True)
        edge = max(edge, e)
        return m
    rows.append({**base, **ROW_DEFAULTS, "arm": "projected_diagnostic", "risk": e14._pooled(closure, test), "edge_mass": edge})
    rows.append({**base, **ROW_DEFAULTS, "arm": "isotropic",
                 "risk": e14._pooled(lambda X, t: (X - (1 - np.exp(-2 * t)) * X) / np.exp(-t), test)})
    print(f"  [E33 oracles] {cell_id} seed={seed}: " + " ".join(f"{r['arm']}={r['risk']:.4f}" for r in rows), flush=True)
    return rows


# ------------------------------------------------------------------ networks


def _screen_cell(args):
    cfg, cell_id, seed, arch, hp = args
    chain, grid, w, log_k, log_mu, sigma0, A, Av, val, test, base = _common(cfg, cell_id, seed, cfg["n_chains"])
    L = base["n_sites"]
    groups = e14._one_view_groups(A, f"e33-em-{cell_id}-M{cfg['n_chains']}", seed, L)
    val_groups = e14._one_view_groups(Av, f"e33-emval-{cell_id}", seed, L)
    den, info = e14._train_ism(cfg, arch, hp, L, groups, val_groups, seed, cfg["screen_steps"])
    risk = e14._pooled(lambda X, t: den.posterior_mean(X, t), test)
    print(f"  [E33 screen] {cell_id} seed={seed} {arch} {hp}: risk={risk:.4f} ckpt={info['checkpoint']} {info['wall']:.0f}s", flush=True)
    return [{**base, **ROW_DEFAULTS, "arm": "ism_one_view", "arch": arch, "hp": str(hp), "risk": risk, **info}]


def winners_per_cell(path) -> dict:
    """Per (cell, arch), the hp with the lowest risk pooled over development seeds."""
    import pandas as pd
    df = pd.read_csv(path)
    out = {}
    for (cell, arch), sub in df.groupby(["cell", "arch"]):
        g = sub.groupby("hp").risk.mean().sort_values()
        out[f"{cell}|{arch}"] = dict(hp=eval(g.index[0]), screen_risk=float(g.iloc[0]))
    return out


def _nets_cell(args):
    cfg, cell_id, seed, n, winners = args
    chain, grid, w, log_k, log_mu, sigma0, A, Av, val, test, base = _common(cfg, cell_id, seed, n)
    L = base["n_sites"]
    groups, idx = e14._one_view_groups(A, f"e33-em-{cell_id}-M{n}", seed, L, return_index=True)
    val_groups = e14._one_view_groups(Av, f"e33-emval-{cell_id}", seed, L)
    rows = []
    for arch, ww in winners.items():
        den, ninfo = e14._train_ism(cfg, arch, ww["hp"], L, groups, val_groups, seed, cfg["net_steps"])
        rows.append({**base, **ROW_DEFAULTS, "arm": "ism_one_view", "arch": arch, "hp": str(ww["hp"]),
                     "risk": e14._pooled(lambda X, t: den.posterior_mean(X, t), test), **ninfo})
        den, ninfo = e14._train_dsm_one_corruption(cfg, arch, ww["hp"], L, A, groups, idx, val, seed, cfg["net_steps"])
        rows.append({**base, **ROW_DEFAULTS, "arm": "dsm_one_corruption", "arch": arch, "hp": str(ww["hp"]),
                     "risk": e14._pooled(lambda X, t: den.posterior_mean(X, t), test), **ninfo})
    print(f"  [E33 nets] {cell_id} seed={seed} M={n}: " + " ".join(f"{r['arm']}/{r['arch']}={r['risk']:.4f}" for r in rows), flush=True)
    return rows


# ------------------------------------------------------------------ parts


def part_oracles(cfg, out, workers):
    cells = [(cfg, c, s) for c in cfg["cells"] for s in cfg["seeds"]]
    write_csv(out / "oracles.csv", run_cells(cells, _oracles_cell, workers, out, "oracles.csv"))


def part_kernels(cfg, out, workers):
    # largest M first, so the longest cells start first and the pool drains evenly
    cells = [(cfg, c, s, n) for n in sorted(cfg["sizes"], reverse=True) for c in cfg["cells"] for s in cfg["seeds"]]
    print(f"  kernels: {len(cells)} cells x {len(cfg['arms'])} arms", flush=True)
    write_csv(out / "kernels.csv", run_cells(cells, _kernels_cell, workers, out, "kernels.csv"))


def part_screen(cfg, out, workers):
    cells = [(cfg, c, s, arch, hp) for c in cfg["cells"] for s in cfg["dev_seeds"]
             for arch in cfg["archs"] for hp in e14.GRIDS[arch]]
    print(f"  screen: {len(cells)} cells", flush=True)
    write_csv(out / "screening.csv", run_cells(cells, _screen_cell, workers, out, "screening.csv"))


def part_nets(cfg, out, workers):
    w = cfg["winners"]
    if isinstance(w, str) and w.startswith("auto:"):
        w = winners_per_cell(w[5:])
    if not w:
        raise SystemExit("nets needs --set winners=\"'auto:<screening.csv>'\"")
    write_json(out / "winners.json", w)
    cells = []
    for n in cfg["net_sizes"]:
        for c in cfg["cells"]:
            sel = {a: w[f"{c}|{a}"] for a in cfg["archs"] if f"{c}|{a}" in w}
            if len(sel) != len(cfg["archs"]):
                raise SystemExit(f"no screened winner for cell {c} (have {sorted(w)})")
            cells += [(cfg, c, s, n, sel) for s in cfg["seeds"]]
    print(f"  nets: {len(cells)} cells", flush=True)
    write_csv(out / "nets.csv", run_cells(cells, _nets_cell, workers, out, "nets.csv"))


PARTS = {"oracles": part_oracles, "kernels": part_kernels, "screen": part_screen, "nets": part_nets}


def main():
    ap = experiment_parser("e33_field_learning", "learning the phi^4 chain through noise: kernels, closure, networks")
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
        write_json(out / f"params_{name}.json", {**provenance(), "settings": cfg, "cells": cells_info,
                                                 "quick": bool(args.quick), "workers": args.workers})


if __name__ == "__main__":
    main()
