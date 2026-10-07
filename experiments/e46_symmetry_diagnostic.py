"""E46: why does the augmented kernel beat the first-order one when there is no latent?

E35x (frozen, sixteen seeds): at b = 0 the augmented arm's risk is LOWER than the first-order arm's, by 18% at M = 512 and
27% at M = 2048 (geometric mean; the interval of the mean log ratio excludes 0), although the data have no latent and the
two models have the same number of parameters. The augmented model is a mixture of the kernel and its REFLECTION
(K_-(x'|x) = K_+(-x'|-x), shared parameters; markov_score.augmented), so its E-step feeds the shared kernel every sequence
and its mirror image. The truth at b = 0 is reflection-symmetric. The hypothesis: the gain is the symmetry, not the latent.

This driver tests it with two new arms on E35's own data (same training chains, same validation and test bundles, same
initialisation as the first-order arm; the bundle hashes are compared at read time against E35x's frozen rows):

    em_first_order_sym   the FIRST-ORDER kernel fitted on the training sequences and their mirror images (x and -x, each noised
                         view reflected with it), no mixture, no latent state: the symmetry is used and nothing else.
    em_first_order_2m    the first-order kernel fitted on twice as many INDEPENDENT sequences (the original n and n fresh ones):
                         the control for "it is only more data".

b = 0 only; sizes (512, 2048); seeds 0-15. The comparators, em_first_order and em_augmented, are E35x's frozen fits.

Predictions (experiments/registry/E46.json):
  P1  the mean per-seed log(sym / augmented) has a 95% interval containing 0 at both M (the symmetrised first-order kernel matches it)
  P2  the mean log(sym / first-order) has an interval entirely below 0 at both M (the mirror data alone give the gain)
  P3  the mean log(sym / 2m) has an interval whose upper end is below log 1.1 at both M (reflection is at least nearly as good as
      doubling the data)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e35_state_augmentation as e35  # noqa: E402
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.em import fit_em  # noqa: E402
from markov_score.kernels import MixtureInnovationKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

e14 = e35.e14

SETTINGS = dict(e35.SETTINGS, offsets=(0.0,), sizes=(512, 2048), seeds=tuple(range(16)),
                arms=("em_first_order_sym", "em_first_order_2m"))
QUICK = dict(e35.QUICK, offsets=(0.0,))


def mirror(groups):
    """Every noised view and its mirror image: reflecting the observation reflects the chain, and the noise is symmetric."""
    return list(groups) + [(-X, al, de) for X, al, de in groups]


def _fit_arm(cfg, arm, grid, w, groups, val, seed, b, n):
    mu_fn = e35._mu_fn(grid, w)
    ck = set(int(c) for c in cfg["em_checkpoints"])
    # the initialisation is the first-order arm's own (E35 tag), so the arms differ in their data and nothing else
    k0 = MixtureInnovationKernel.init(cfg["components"], rho=cfg["em_init_rho"], var=cfg["em_init_var"],
                                      rng=rng_for("e35-init", "em_first_order", seed, b, n))
    t0 = time.time()
    _, trace, saved = fit_em(k0, grid, w, groups, n_iters=max(ck), tol=FROZEN.em_loglik_tol, checkpoints=ck, log_mu=mu_fn)
    wall = time.time() - t0

    def mean_of(kb, X, t):
        return grid_bp_batch(grid, w, kb.log_transition_matrix(grid), X, *alpha_delta(t), log_mu=mu_fn(kb))[0]

    best, best_v = None, np.inf
    for it in sorted(saved):
        v = e14._pooled(lambda X, t, kb=saved[it]: mean_of(kb, X, t), val)
        if v < best_v:
            best, best_v = it, v
    return saved[best], best, best_v, wall, trace, mean_of, saved


def _cell(args):
    cfg, b, seed, n = args
    L = cfg["n_sites"]
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    val = e35._bundle("e35-val", seed, b, cfg["n_val"], L)
    test = e35._bundle("e35-test", seed, b, cfg["n_test"], L)
    A = e35._two_phase(rng_for("e35-train", seed, b, L, n), n, L, b)
    groups = [(X, *alpha_delta(t)) for X, t in e14._one_view_groups(A, f"e35-em-b{b:g}-M{n}", seed, L)]
    base = dict(b=b, rho=e35.RHO, n_sites=L, seed=seed, n_chains=n,
                val_bundle=e14._bundle_hash(val), test_bundle=e14._bundle_hash(test))
    rows = []
    for arm in cfg["arms"]:
        if arm == "em_first_order_sym":
            data = mirror(groups)
        elif arm == "em_first_order_2m":
            A2 = e35._two_phase(rng_for("e46-train-extra", seed, b, L, n), n, L, b)
            data = groups + [(X, *alpha_delta(t)) for X, t in e14._one_view_groups(A2, f"e46-em-extra-b{b:g}-M{n}", seed, L)]
        else:
            raise ValueError(arm)
        kb, best, best_v, wall, trace, mean_of, saved = _fit_arm(cfg, arm, grid, w, data, val, seed, b, n)
        risk = e14._pooled(lambda X, t: mean_of(kb, X, t), test)
        rows.append(dict(base, arm=arm, checkpoint=best, at_cap=int(best == max(saved)), val_selected=best_v, wall=wall,
                         n_params=int(len(np.asarray(kb.theta))), converged=bool(trace.converged), stop_reason=trace.stop_reason,
                         monotone_violation=float(trace.monotone_violation),
                         theta=";".join(f"{v:.8g}" for v in np.asarray(kb.theta, dtype=float)), risk=risk))
    print(f"  [E46] b={b:g} seed={seed} M={n}: " + " ".join(f"{r['arm']}={r['risk']:.4f}" for r in rows), flush=True)
    return rows


def part_symmetry(cfg, out, workers):
    cells = [(cfg, b, s, n) for n in sorted(cfg["sizes"], reverse=True) for b in cfg["offsets"] for s in cfg["seeds"]]
    print(f"  symmetry: {len(cells)} cells", flush=True)
    write_csv(out / "symmetry.csv", run_cells(cells, _cell, workers, out, "symmetry.csv",
                                              key=lambda c: f"b{c[1]:g}_s{c[2]}_M{c[3]}"))


PARTS = {"symmetry": part_symmetry}


def main():
    ap = experiment_parser("e46_symmetry_diagnostic", "is the b = 0 gain of the augmented kernel the reflection symmetry?")
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
