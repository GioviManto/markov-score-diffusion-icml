#!/usr/bin/env python
"""E14 — does the structured estimator's advantage over networks survive a change of regime?

The paper's headline is a comparison at one point: a Laplace AR(1) chain with rho = 0.85 and
length 32, where EM-BP on one noised view per sequence reaches 0.023 against the best network's
0.155 at M = 2048 (claim E16). Everything in that sentence except the estimator is a choice, and
the obvious reviewer question is whether the factor is a property of the method or of the corner.

This experiment re-runs that comparison -- the information-matched one, which is the fair and the
harder of the two -- across three correlations and two chain lengths, and reports the paired factor
in each. The correlation moves both the crossover t_* and the non-Gaussianity of the marginal
(excess kurtosis 3(1-rho^2)/(1+rho^2)), so if the advantage is really about representing a message
that has crossed the chain, it should narrow where the chain carries less and widen where the
posterior is longer-ranged. The length moves the receptive-field argument directly: the window head
is given the same radius it won with at L = 32, which is the point -- a bounded window is a
bounded window whatever the chain does around it.

Two deliberate departures from E03, both in the direction of affordability and both applied to
every arm alike: the divergence in the training objective is estimated by Hutchinson for all
architectures (E03 used the exact trace for the window head, which costs L backward passes per
step and is prohibitive at L = 128), and the screen trains to 8k steps where the confirmatory run
trains to 32k. Checkpoint selection uses the exact divergence throughout.

Fairness. The networks are re-screened inside each regime, on development seeds disjoint from the
confirmatory ones, over the same architecture grid E03 used; the winner per family is then run on
the confirmatory seeds. So nothing is transferred from the frozen point except the grid of
candidates itself. Both sides see exactly one corruption of each training sequence, drawn once:
the networks train by implicit score matching on those views and select on the implicit objective
of a held-out noisy bundle, never on the true score; EM-BP fits the mixture kernel on the same
views and selects on the same held-out bundle's BP risk. The LMMSE of the true covariance and the
isotropic map are computed on the test bundle as the two references.

Reported per cell: the pooled relative score error of every arm on a common test bundle, and the
paired ratio best-network / EM-BP over seeds.
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
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.em import fit_em  # noqa: E402
from markov_score.kernels import MixtureInnovationKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402
from protocol import PROTOCOL  # noqa: E402

T_SCHEDULE = tuple(FROZEN.t_grid)

# the same candidate grid E03 screened, restricted to the two families that carried the matched
# comparison there: the local head and the global attention model.
GRIDS = {
    "window": [dict(radius=r, width=w) for r in (2, 4, 8) for w in (64, 128)],
    "transformer": [dict(d=d, heads=4, layers=l) for d, l in ((32, 2), (64, 2), (64, 4))],
}

SETTINGS = dict(
    rhos=(0.5, 0.85, 0.95),
    lengths=(32, 128),
    n_chains=2048, n_val=PROTOCOL.n_val, n_test=PROTOCOL.n_test,
    seeds=tuple(range(8)),
    dev_seeds=PROTOCOL.dev_seeds[:2],
    components=8,
    grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    em_checkpoints=PROTOCOL.em_checkpoints,
    em_init_rho=PROTOCOL.em_init_rho, em_init_var=PROTOCOL.em_init_var,
    archs=tuple(GRIDS),
    net_steps=PROTOCOL.net_steps, net_checkpoints=PROTOCOL.net_checkpoints,
    screen_steps=8_000, lr=PROTOCOL.lrs[0], mode=PROTOCOL.parameterizations[0],
    batch_size=PROTOCOL.batch_size, grad_clip=PROTOCOL.grad_clip,
    winners=None,                       # confirm: "auto:<screening.csv>", or a dict
    torch_threads=1,
)
QUICK = dict(rhos=(0.85,), lengths=(32,), n_chains=64, n_val=32, n_test=32, seeds=(10_000,),
             dev_seeds=(10_000,), grid_size=201, em_checkpoints=(5, 10),
             net_steps=100, net_checkpoints=(50, 100), screen_steps=100)


# ------------------------------------------------------------------ bundles and scoring


def _bundle(prior, log_k, grid, w, tag, seed, n_chains, L):
    rng = rng_for(tag, seed, L)
    A = np.stack([prior.sample(rng, L) for _ in range(n_chains)])
    out = {}
    for t in T_SCHEDULE:
        al, de = alpha_delta(t)
        X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
        m, _ = grid_bp_batch(grid, w, log_k, X, al, de)
        out[t] = (X, m)
    return A, out


def _bundle_hash(bundle):
    import hashlib
    h = hashlib.sha256()
    for t in sorted(bundle):
        X, m = bundle[t]
        h.update(np.ascontiguousarray(X, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(m, dtype=np.float64).tobytes())
    return h.hexdigest()[:12]


def _pooled(mean_fn, bundle):
    """Pooled relative score error of a posterior-mean map over the schedule."""
    num = den = 0.0
    for t in T_SCHEDULE:
        X, m_ref = bundle[t]
        al, de = alpha_delta(t)
        s_hat = -(X - al * mean_fn(X, t)) / de
        s_ref = -(X - al * m_ref) / de
        num += float(((s_hat - s_ref) ** 2).sum()); den += float((s_ref ** 2).sum())
    return float(np.sqrt(num / den))


def _one_view_groups(A, tag, seed, L):
    """The shared information interface: one corruption per chain, chains split over the levels."""
    idx = np.array_split(rng_for(tag + "-split", seed, L).permutation(len(A)), len(T_SCHEDULE))
    noise = rng_for(tag + "-noise", seed, L)
    out = []
    for t, ii in zip(T_SCHEDULE, idx):
        al, de = alpha_delta(t)
        sub = A[ii]
        out.append((al * sub + np.sqrt(de) * noise.standard_normal(sub.shape), t))
    return out


# ------------------------------------------------------------------ arms


def _ism_objective(den, groups):
    """Mean implicit-score-matching objective on held-out noisy views, exact divergence."""
    import torch
    tot, n = 0.0, 0
    for X, t in groups:
        x = torch.as_tensor(X, dtype=torch.float32).requires_grad_(True)
        tt = torch.full((x.shape[0],), float(t))
        al, de = float(np.exp(-t)), float(1 - np.exp(-2 * t))
        out = den.model(x, tt)
        m = out if den.parameterization == "x0" else (x - np.sqrt(de) * out) / al
        s = -(x - al * m) / de
        div = torch.zeros(x.shape[0])
        for i in range(x.shape[1]):
            g = torch.autograd.grad(s[:, i].sum(), x, retain_graph=True)[0]
            div = div + g[:, i]
        tot += float((0.5 * (s ** 2).sum(dim=1) + div).detach().sum()); n += x.shape[0]
    return tot / n


def _train_ism(cfg, arch, hp, L, groups, val_groups, seed, steps):
    """Train by ISM on the one-view groups; select the checkpoint on the held-out ISM objective."""
    import torch
    from markov_score import nn_torch as nt
    torch.set_num_threads(cfg["torch_threads"])
    ck = tuple(int(c) for c in cfg["net_checkpoints"] if c <= steps) or (steps,)
    model = nt.build(arch, L, **hp)
    tcfg = nt.TrainConfig(steps=max(ck), batch_size=cfg["batch_size"], lr=cfg["lr"],
                          grad_clip=cfg["grad_clip"], parameterization=cfg["mode"], loss="ism",
                          checkpoints=ck, t_levels=T_SCHEDULE,
                          seed=int(rng_for("e14-net", arch, str(hp), L, seed).integers(0, 2**31 - 1)),
                          # Hutchinson for EVERY family here, the window head included. E03 could
                          # afford the exact Jacobian trace for the window at L = 32; at L = 128 it
                          # is L backward passes per step and dominates the run. Using one estimator
                          # for all arms also makes the comparison internally uniform, which is what
                          # this experiment is about. Checkpoint selection below stays exact.
                          hutchinson=True)
    t0 = time.time()
    saved = nt.train(model, None, tcfg, X_noisy_groups=groups)
    best, best_v = None, np.inf
    for step in sorted(saved):
        den = nt.Denoiser(nt.load_checkpoint(model, saved[step]), cfg["mode"])
        v = _ism_objective(den, val_groups)
        if v < best_v:
            best, best_v = step, v
    den = nt.Denoiser(nt.load_checkpoint(model, saved[best]), cfg["mode"])
    return den, dict(checkpoint=best, at_cap=best == max(saved), val_selected=best_v,
                     wall=time.time() - t0, n_params=nt.n_params(model))


def _em_arm(cfg, grid, w, groups, val, seed, rho, L):
    """EM-BP on the same one-view groups, checkpoint selected on the held-out BP risk."""
    gs = [(X, *alpha_delta(t)) for X, t in groups]
    k0 = MixtureInnovationKernel.init(cfg["components"], rho=cfg["em_init_rho"], var=cfg["em_init_var"],
                                      rng=rng_for("e14-init", seed, L, rho))
    ck = set(int(c) for c in cfg["em_checkpoints"])
    _, _, saved = fit_em(k0, grid, w, gs, n_iters=max(ck), tol=FROZEN.em_loglik_tol, checkpoints=ck)
    best, best_v = None, np.inf
    for it in sorted(saved):
        lk = saved[it].log_transition_matrix(grid)
        v = _pooled(lambda X, t, lk=lk: grid_bp_batch(grid, w, lk, X, *alpha_delta(t))[0], val)
        if v < best_v:
            best, best_v = it, v
    kb = saved[best]
    return kb, dict(checkpoint=best, at_cap=best == max(saved), val_selected=best_v,
                    fit_kurt=float(kb.innovation_moments["innovation_excess_kurtosis"]),
                    fit_rho=float(kb.rho))


# ------------------------------------------------------------------ cells


def _common(cfg, rho, L, seed, n):
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    prior = LaplaceAR1(rho)
    log_k = prior.log_transition_matrix(grid)
    _, val = _bundle(prior, log_k, grid, w, "e14-val", seed, cfg["n_val"], L)
    _, test = _bundle(prior, log_k, grid, w, "e14-test", seed, cfg["n_test"], L)
    A = np.stack([prior.sample(rng_for("e14-train", seed, L, rho), L) for _ in range(n)])
    Av = np.stack([prior.sample(rng_for("e14-valchains", seed, L, rho), L) for _ in range(cfg["n_val"])])
    base = dict(rho=rho, n_sites=L, seed=seed, n_chains=n,
                t_star=float(theory.crossover_time(rho, L=L)),
                val_bundle=_bundle_hash(val), test_bundle=_bundle_hash(test))
    return grid, w, A, Av, val, test, base


def _screen_cell(args):
    cfg, rho, L, seed, arch, hp = args
    grid, w, A, Av, val, test, base = _common(cfg, rho, L, seed, cfg["n_chains"])
    groups = _one_view_groups(A, "e14-em", seed, L)
    val_groups = _one_view_groups(Av, "e14-emval", seed, L)
    den, info = _train_ism(cfg, arch, hp, L, groups, val_groups, seed, cfg["screen_steps"])
    risk = _pooled(lambda X, t: den.posterior_mean(X, t), test)
    print(f"  [E14 screen] rho={rho} L={L} seed={seed} {arch} {hp}: risk={risk:.4f} "
          f"ckpt={info['checkpoint']} {info['wall']:.0f}s", flush=True)
    return [{**base, "arch": arch, "hp": str(hp), "risk": risk, **info}]


# every row carries the same columns; an arm that has no fitted parameter leaves them blank rather
# than shortening the record, so the CSV has one schema and a missing value means "not applicable".
ROW_DEFAULTS = dict(arch="", hp="", checkpoint=-1, at_cap=False, val_selected=np.nan,
                    wall=np.nan, n_params=0, fit_kurt=np.nan, fit_rho=np.nan)


def _confirm_cell(args):
    cfg, rho, L, seed, winners = args
    grid, w, A, Av, val, test, base = _common(cfg, rho, L, seed, cfg["n_chains"])
    sigma0 = theory.chain_covariance(rho, 1.0 - rho ** 2, L)
    rows = []

    for arm, fn in (("lmmse_true", lambda X, t: theory.lmmse_posterior_mean(X, sigma0, t)),
                    ("isotropic", lambda X, t: (X - (1 - np.exp(-2 * t)) * X) / np.exp(-t))):
        rows.append({**base, **ROW_DEFAULTS, "arm": arm, "risk": _pooled(fn, test)})

    groups = _one_view_groups(A, "e14-em", seed, L)
    val_groups = _one_view_groups(Av, "e14-emval", seed, L)

    kb, info = _em_arm(cfg, grid, w, groups, val, seed, rho, L)
    lk = kb.log_transition_matrix(grid)
    rows.append({**base, **ROW_DEFAULTS, "arm": "em_bp",
                 "risk": _pooled(lambda X, t: grid_bp_batch(grid, w, lk, X, *alpha_delta(t))[0], test),
                 **info})

    for arch, ww in winners.items():
        den, ninfo = _train_ism(cfg, arch, ww["hp"], L, groups, val_groups, seed, cfg["net_steps"])
        rows.append({**base, **ROW_DEFAULTS, "arm": "ism_one_view", "arch": arch,
                     "hp": str(ww["hp"]),
                     "risk": _pooled(lambda X, t: den.posterior_mean(X, t), test), **ninfo})
    print(f"  [E14] rho={rho} L={L} seed={seed}: em={rows[2]['risk']:.4f} "
          + " ".join(f"{r['arch']}={r['risk']:.4f}" for r in rows[3:])
          + f"  lmmse={rows[0]['risk']:.4f}", flush=True)
    return rows


# ------------------------------------------------------------------ parts


def part_screen(cfg, out, workers):
    cells = [(cfg, r, L, s, arch, hp)
             for r in cfg["rhos"] for L in cfg["lengths"] for s in cfg["dev_seeds"]
             for arch in cfg["archs"] for hp in GRIDS[arch]]
    print(f"  screen: {len(cells)} cells", flush=True)
    write_csv(out / "screening.csv", run_cells(cells, _screen_cell, workers, out, "screening.csv"))


def winners_per_regime(path) -> dict:
    """Per (rho, L, arch), the hp with the lowest risk pooled over development seeds."""
    import pandas as pd
    df = pd.read_csv(path)
    out = {}
    for (rho, L, arch), sub in df.groupby(["rho", "n_sites", "arch"]):
        g = sub.groupby("hp").risk.mean().sort_values()
        out[f"{rho}|{L}|{arch}"] = dict(hp=eval(g.index[0]), screen_risk=float(g.iloc[0]))
    return out


def part_confirm(cfg, out, workers):
    w = cfg["winners"]
    if isinstance(w, str) and w.startswith("auto:"):
        w = winners_per_regime(w[5:])
    if not w:
        raise SystemExit("confirm needs --set winners=\"'auto:<screening.csv>'\"")
    write_json(out / "winners.json", w)
    cells = []
    for r in cfg["rhos"]:
        for L in cfg["lengths"]:
            sel = {a: w[f"{r}|{L}|{a}"] for a in cfg["archs"] if f"{r}|{L}|{a}" in w}
            if len(sel) != len(cfg["archs"]):
                raise SystemExit(f"no screened winner for rho={r} L={L} (have {sorted(w)})")
            cells += [(cfg, r, L, s, sel) for s in cfg["seeds"]]
    print(f"  confirm: {len(cells)} cells", flush=True)
    write_csv(out / "regimes.csv", run_cells(cells, _confirm_cell, workers, out, "regimes.csv"))


PARTS = {"screen": part_screen, "confirm": part_confirm}


def main():
    ap = experiment_parser("e14_regime_baselines",
                           "the information-matched comparison across correlations and chain lengths")
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
        write_json(out / f"params_{name}.json", {**provenance(), "settings": cfg,
                                                 "quick": bool(args.quick), "workers": args.workers})


if __name__ == "__main__":
    main()
