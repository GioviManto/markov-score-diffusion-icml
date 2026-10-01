#!/usr/bin/env python
"""E36 -- the subsampling protocol: one fine field observed at three strides (the note, section 21.1).

The registered refinement cells (E32/E33) are three LATTICE MODELS at h = 1, 0.5, 0.25 on a fixed
physical length. The consolidated note recommends instead to "generate at fine resolution and
subsample systematically". Both are exact. `field.StridedChain` observes the fine chain
(Theta = 0.3, h = 0.25, 128 sites) every s-th site, s in (1, 2, 4): the observed chain is exactly
Markov with the s-step kernel K^s (Chapman-Kolmogorov on the grid), the same stationary law,
effective spacing s h = 0.25, 0.5, 1 and lambda_1/lambda_0 to the power s. So the refinement
question is asked of ONE physical object rather than three.

The point that makes this more than a protocol variant: for s >= 2 the kernel K^s is NOT a
tilted walk (marginalising the intermediate sites destroys the Gaussian-bond form), so the
gradient-class kernels (the quartic site energy, the tilted bases) are misspecified on the
subsampled chain, while a state-dependent mixture (the MDN kernel) is not. Which kernel
families survive subsampling is a question the paper must answer if the stride protocol is
adopted.

Parts (laptop, deterministic data):
    oracles   per (stride, rep): the oracle ladder on 512 strided chains, exact BP with K^s as the
              reference, the LMMSE of the strided covariance, the isotropic map, the projected
              diagnostic with its edge mass. Comparable with E32's lattice cells at the same
              effective spacing.
    physics   per (stride, rep): E34's physical statistics on 4096 strided chains (the same fine
              samples subsampled), in physical units.
    clean     per (stride, seed): clean-data maximum likelihood of em_pair_quartic, em_tilted_spline,
              em_mixture and em_mdn on 2048 strided chains, scored on a strided test bundle against
              exact BP with K^s (E34's clean ladder; the MDN's to 400).

PREDECLARED (registry E36.json, written before any cell ran):
  P1  the COARSE kink density per unit length and the coarse mean domain length in physical units
      agree across the three strides within 10% (one physical object; N-31's statistic).
  P2  the linear floor at stride 4 (effective spacing 1, 32 sites) is within 15% of E32's floor at
      the lattice cell T0.3_h1_L32 (0.269): the two refinement protocols agree at second order.
  P3  the gradient-class kernels are exactly specified at stride 1 (clean risk below 0.02 at
      M = 2048) and misspecified at stride 4: their clean risk at stride 4 is at least twice their
      stride-1 risk (both seeds, both kernels), while the MDN's clean risk at stride 4 is below the
      quartic site energy's.
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
from markov_score.em import clean_statistics  # noqa: E402
from markov_score.field import Phi4Chain, StridedChain  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

import e14_regime_baselines as e14  # noqa: E402
import e33_field_learning as e33  # noqa: E402
import e34_field_physics as e34  # noqa: E402

T_SCHEDULE = tuple(FROZEN.t_grid)

SETTINGS = dict(
    T=0.3, h_fine=0.25, L_fine=128, strides=(1, 2, 4), reps=(0, 1), seeds=(0, 1),
    n_oracle=512, n_physics=4096, n_clean=2048, n_val=1024, n_test=2048,
    grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    arms=("em_pair_quartic", "em_tilted_spline", "em_mixture", "em_mdn"),
    components=8, poly_degree=6, spline_basis=24, mdn_components=4, mdn_hidden=32,
    em_init_rho=0.3, em_init_var=0.8,
    clean_checkpoints=(1, 2, 4, 8, 16, 32, 64, 120, 220, 400),
)
QUICK = dict(strides=(1, 4), reps=(0,), seeds=(10_000,), n_oracle=16, n_physics=128, n_clean=32, n_val=32, n_test=32,
             grid_size=201, arms=("em_pair_quartic", "em_mixture"), clean_checkpoints=(1, 2, 4))


def _chain(cfg, stride):
    ch = StridedChain(Phi4Chain(cfg["T"], cfg["h_fine"]), int(stride))
    L = cfg["L_fine"] // int(stride)
    info = dict(stride=int(stride), T=cfg["T"], h_fine=cfg["h_fine"], h=ch.h, n_sites=L, rho1=ch.rho,
                corr_length_sites=-1.0 / np.log(ch.rho), scale=ch.scale, excess_kurtosis=ch.innovation_excess_kurtosis)
    return ch, L, info


def _oracle_cell(args):
    cfg, stride, rep = args
    ch, L, info = _chain(cfg, stride)
    g, w = make_grid(cfg["half_width"], cfg["grid_size"])
    log_k, log_mu = ch.log_transition_matrix(g), ch.log_stationary(g)
    sigma0 = ch.covariance(L)
    rng = rng_for("e36-oracle", stride, rep)
    A = ch.sample_batch(rng, cfg["n_oracle"], L)
    noise = rng.standard_normal((len(T_SCHEDULE),) + A.shape)
    pooled = {a: [0.0, 0.0] for a in ("lmmse_true", "projected", "isotropic")}
    rows, edge_max = [], 0.0
    for li, t in enumerate(T_SCHEDULE):
        al, de = alpha_delta(t)
        X = al * A + np.sqrt(de) * noise[li]
        m_ref, _ = grid_bp_batch(g, w, log_k, X, al, de, log_mu=log_mu)
        m_pr, _, edge = grid_bp_projected_batch(g, w, log_k, X, al, de, log_mu=log_mu, return_diagnostics=True)
        edge_max = max(edge_max, edge)
        arms = {"lmmse_true": theory.lmmse_posterior_mean(X, sigma0, t), "projected": m_pr, "isotropic": (X - de * X) / al}
        for arm, m in arms.items():
            s_hat = -(X - al * m) / de; s_ref = -(X - al * m_ref) / de
            e, r = float(((s_hat - s_ref) ** 2).sum()), float((s_ref ** 2).sum())
            pooled[arm][0] += e; pooled[arm][1] += r
            rows.append({**info, "rep": rep, "arm": arm, "level": t, "risk": float(np.sqrt(e / r)), "edge_mass": edge if arm == "projected" else np.nan})
    for arm, (e, r) in pooled.items():
        rows.append({**info, "rep": rep, "arm": arm, "level": -1.0, "risk": float(np.sqrt(e / r)), "edge_mass": edge_max if arm == "projected" else np.nan})
    print(f"  [E36 oracles] stride={stride} rep={rep}: " + " ".join(f"{a}={np.sqrt(e / r):.4f}" for a, (e, r) in pooled.items()), flush=True)
    return rows


def _physics_cell(args):
    cfg, stride, rep = args
    ch, L, info = _chain(cfg, stride)
    A = ch.sample_batch(rng_for("e36-physics", stride, rep), cfg["n_physics"], L)
    st = {k: v for k, v in e34.physical_statistics(A, ch.scale, L, ch.h).items() if np.isscalar(v)}
    st.update(kink_density_physical=st["kink_density"] / ch.h, mean_domain_physical=st["mean_domain"] * ch.h,
              coarse_kink_density_physical=st["coarse_kink_density"] / ch.h, coarse_mean_domain_physical=st["coarse_mean_domain"] * ch.h)
    print(f"  [E36 physics] stride={stride} rep={rep}: coarse kinks/unit={st['coarse_kink_density_physical']:.4f} "
          f"coarse domain={st['coarse_mean_domain_physical']:.2f} bimod={st['bimodality_coefficient']:.3f}", flush=True)
    return [{**info, "rep": rep, "n_chains": len(A), **st}]


def _clean_cell(args):
    cfg, stride, seed = args
    ch, L, info = _chain(cfg, stride)
    g, w = make_grid(cfg["half_width"], cfg["grid_size"])
    log_k, log_mu = ch.log_transition_matrix(g), ch.log_stationary(g)
    _, val = e33._bundle(ch, log_k, log_mu, g, w, "e36-val", seed, f"stride{stride}", cfg["n_val"], L)
    _, test = e33._bundle(ch, log_k, log_mu, g, w, "e36-test", seed, f"stride{stride}", cfg["n_test"], L)
    A = ch.sample_batch(rng_for("e36-train", seed, stride, cfg["n_clean"]), cfg["n_clean"], L)
    stats = clean_statistics(g, A)
    mu_fn = e33._stationary_mu(g, w)
    base = dict(**info, seed=seed, n_chains=cfg["n_clean"], val_bundle=e14._bundle_hash(val), test_bundle=e14._bundle_hash(test))
    rows = []
    for arm in cfg["arms"]:
        t0 = time.time()
        k0 = e33._init_kernel(arm, cfg, g, seed, f"stride{stride}", cfg["n_clean"])
        ck = tuple(int(c) for c in cfg["clean_checkpoints"])
        if arm.startswith("em_tilted") or arm.startswith("em_pair"):
            ck = tuple(c for c in ck if c <= 4) or (1,)
        saved = e34._fit_clean_ladder(k0, g, stats, set(ck))
        best, best_v, mus = None, np.inf, {}
        for it in sorted(saved):
            mu = mu_fn(saved[it]); mus[it] = mu
            lk = saved[it].log_transition_matrix(g)
            v = e14._pooled(lambda X, t, lk=lk, mu=mu: grid_bp_batch(g, w, lk, X, *alpha_delta(t), log_mu=mu)[0], val)
            if v < best_v:
                best, best_v = it, v
        kb = saved[best]; lk = kb.log_transition_matrix(g)
        risk = e14._pooled(lambda X, t, lk=lk, mu=mus[best]: grid_bp_batch(g, w, lk, X, *alpha_delta(t), log_mu=mu)[0], test)
        rows.append({**base, **e33.ROW_DEFAULTS, "arm": arm, "selection": "risk", "risk": risk, "checkpoint": best,
                     "at_cap": int(best == max(saved)), "val_selected": best_v, "n_params": int(len(np.asarray(kb.theta))),
                     "wall": time.time() - t0, "n_iters": max(saved), **e33._kernel_diagnostics(kb, g),
                     "theta": ";".join(f"{v:.8g}" for v in np.asarray(kb.theta, dtype=float))})
    print(f"  [E36 clean] stride={stride} seed={seed}: " + " ".join(f"{r['arm']}={r['risk']:.4f}" for r in rows), flush=True)
    return rows


def part_oracles(cfg, out, workers):
    cells = [(cfg, s, r) for s in cfg["strides"] for r in cfg["reps"]]
    write_csv(out / "oracles.csv", run_cells(cells, _oracle_cell, workers, out, "oracles.csv", key=lambda c: f"s{c[1]}_r{c[2]}"))


def part_physics(cfg, out, workers):
    cells = [(cfg, s, r) for s in cfg["strides"] for r in cfg["reps"]]
    write_csv(out / "physics.csv", run_cells(cells, _physics_cell, workers, out, "physics.csv", key=lambda c: f"s{c[1]}_r{c[2]}"))


def part_clean(cfg, out, workers):
    cells = [(cfg, s, sd) for s in cfg["strides"] for sd in cfg["seeds"]]
    write_csv(out / "clean.csv", run_cells(cells, _clean_cell, workers, out, "clean.csv", key=lambda c: f"s{c[1]}_seed{c[2]}"))


PARTS = {"oracles": part_oracles, "physics": part_physics, "clean": part_clean}


def main():
    ap = experiment_parser("e36_stride_protocol", "one fine phi^4 field observed at three strides: oracles, physics, clean fits")
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
