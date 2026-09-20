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

Each network arm trains exactly as its E03 winner did (outputs/E03/winners.json): the window
head with the EXACT divergence and learning rate 1e-3, the transformer with Hutchinson's estimator
and learning rate 3e-3, parameterisation eps for both. The exact divergence of the window head is
computed with 2r+1 coloured probes (markov_score.nn_torch.divergence), which is the exact trace to
round-off at the cost of Hutchinson, so it is affordable at L = 128. The screen trains to 8k steps
where the confirmatory run trains to 32k; checkpoint selection uses the exact divergence throughout.

Amendments of 20 September 2026, before any valid confirmatory cell ran. The first screen (job
663143) gave the window head 0.88 and 1.65 at the frozen point (rho = 0.85, L = 32, M = 2048,
E03's radius and width) against 0.166 for the same network in E03's matched arm. The cause was
not the estimator but the data: _common() re-created the random generator inside its sampling
comprehension, so every training chain was the same draw and both arms fitted 2048 copies of
one chain (docs/NEGATIVE_RESULTS.md, N-22). It is fixed below, with an assertion. The per-arm
estimators above are E03's and were adopted in the same pass; they stay because they are what
E03 used, so that the two experiments' network arms are the same estimator.

Two arms were added at the same time, both cheap and both there to bracket the comparison rather
than to change it: (i) the network-favourable interface of E03, denoising score matching on CLEAN
targets with one fixed corruption per chain (dsm_one_corruption), so the factor is reported under
the interface that helps the network most as well as under the matched one; (ii) EM-BP with its
checkpoint chosen on the held-out one-view log-evidence (em_bp_evidence), the oracle-free rule of
E20, so the matched comparison can be read with no true score anywhere in either arm.

Fairness. The networks are re-screened inside each regime, on development seeds disjoint from the
confirmatory ones, over the same architecture grid E03 used; the winner per family is then run on
the confirmatory seeds. So nothing is transferred from the frozen point except the grid of
candidates itself. Both sides see exactly one corruption of each training sequence, drawn once:
the networks train by implicit score matching on those views and select on the implicit objective
of a held-out noisy bundle, never on the true score; EM-BP fits the mixture kernel on the same
views and selects on the same held-out bundle's BP risk (and, in a second row, on the held-out
evidence). The LMMSE of the true covariance and the isotropic map are computed on the test bundle
as the two references.

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
from markov_score.em import e_step_multi, fit_em  # noqa: E402
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
    screen_steps=8_000, mode=PROTOCOL.parameterizations[0],
    # per-arm training as E03's matched winners (outputs/E03/winners.json): learning rate and
    # divergence estimator by architecture; "coloured" is the exact trace of a windowed map
    lrs_by_arch={"window": 1e-3, "transformer": 3e-3},
    trace_by_arch={"window": "coloured", "transformer": "hutchinson"},
    batch_size=PROTOCOL.batch_size, grad_clip=PROTOCOL.grad_clip,
    winners=None,                       # confirm: "auto:<screening.csv>", or a dict
    torch_threads=1,
    device="auto",                      # nn_torch.pick_device: CUDA when the node has it, else CPU
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


def _one_view_groups(A, tag, seed, L, return_index=False):
    """The shared information interface: one corruption per chain, chains split over the levels.
    With `return_index` the per-level index arrays come back too, so an arm that pairs each view
    with its clean chain (the dsm_one_corruption bracket) uses the very same views."""
    idx = np.array_split(rng_for(tag + "-split", seed, L).permutation(len(A)), len(T_SCHEDULE))
    noise = rng_for(tag + "-noise", seed, L)
    out = []
    for t, ii in zip(T_SCHEDULE, idx):
        al, de = alpha_delta(t)
        sub = A[ii]
        out.append((al * sub + np.sqrt(de) * noise.standard_normal(sub.shape), t))
    return (out, idx) if return_index else out


# ------------------------------------------------------------------ arms


def _train_ism(cfg, arch, hp, L, groups, val_groups, seed, steps):
    """Train by ISM on the one-view groups; select the checkpoint on the held-out ISM objective
    (exact divergence). Learning rate and divergence estimator per architecture, as E03's winners."""
    import torch
    from markov_score import nn_torch as nt
    torch.set_num_threads(cfg["torch_threads"])
    ck = tuple(int(c) for c in cfg["net_checkpoints"] if c <= steps) or (steps,)
    model = nt.build(arch, L, **hp)
    tcfg = nt.TrainConfig(steps=max(ck), batch_size=cfg["batch_size"], lr=cfg["lrs_by_arch"][arch],
                          grad_clip=cfg["grad_clip"], parameterization=cfg["mode"], loss="ism",
                          checkpoints=ck, t_levels=T_SCHEDULE,
                          seed=int(rng_for("e14-net", arch, str(hp), L, seed).integers(0, 2**31 - 1)),
                          trace=cfg["trace_by_arch"][arch], device=nt.pick_device(cfg["device"]))
    t0 = time.time()
    saved = nt.train(model, None, tcfg, X_noisy_groups=groups)
    # Selection on the held-out ISM objective with the EXACT divergence for every arm: the coloured
    # probes are exact for the window head, the L-pass loop is used for the dense transformer.
    sel_trace = "coloured" if nt.receptive_radius(model) is not None else "exact"
    # The whole held-out curve is recorded, not only its minimum: a fit that diverged (E03 saw ISM
    # risks near 1.2 at M = 32) shows a curve that never turns over, and the reader of the CSV must
    # be able to call that divergence rather than "the network's error".
    best, best_v, curve = None, np.inf, {}
    for step in sorted(saved):
        den = nt.Denoiser(nt.load_checkpoint(model, saved[step]), cfg["mode"])
        v = nt.ism_objective(den, val_groups, sel_trace)
        curve[step] = v
        if v < best_v:
            best, best_v = step, v
    den = nt.Denoiser(nt.load_checkpoint(model, saved[best]), cfg["mode"])
    return den, dict(checkpoint=best, at_cap=best == max(saved), val_selected=best_v,
                     wall=time.time() - t0, n_params=nt.n_params(model),
                     val_curve=";".join(f"{k}:{v:.5f}" for k, v in sorted(curve.items())))


def _train_dsm_one_corruption(cfg, arch, hp, L, A, groups, idx, val, seed, steps):
    """E03's network-favourable interface: denoising score matching on CLEAN targets, each chain
    paired with its one fixed noised view (the same views the matched arms see), checkpoint on the
    validation BP risk as in E03. Same recipe otherwise (batch, Adam, clip, ladder)."""
    import torch
    from markov_score import nn_torch as nt
    torch.set_num_threads(cfg["torch_threads"])
    ck = tuple(int(c) for c in cfg["net_checkpoints"] if c <= steps) or (steps,)
    dev = torch.device(nt.pick_device(cfg["device"]))
    model = nt.build(arch, L, **hp).to(dev)
    Xs = np.concatenate([X for X, _ in groups]); Ts = np.concatenate([np.full(len(X), t) for X, t in groups])
    As = np.concatenate([A[ix] for ix in idx])
    seed_net = int(rng_for("e14-dsm", arch, str(hp), L, seed).integers(0, 2**31 - 1))
    torch.manual_seed(seed_net); rng = np.random.default_rng(seed_net)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lrs_by_arch"][arch])
    saved, t0 = {}, time.time()
    model.train()
    for step in range(1, max(ck) + 1):
        b = rng.integers(0, len(Xs), size=min(cfg["batch_size"], len(Xs)))
        x = torch.as_tensor(Xs[b], dtype=torch.float32, device=dev); a = torch.as_tensor(As[b], dtype=torch.float32, device=dev)
        t = torch.as_tensor(Ts[b], dtype=torch.float32, device=dev)
        al = torch.exp(-t)[:, None]; de = 1 - torch.exp(-2 * t)[:, None]
        z = (x - al * a) / torch.sqrt(de)
        out = model(x, t)
        target = z if cfg["mode"] == "eps" else a
        loss = ((out - target) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]); opt.step()
        if step in ck:
            saved[step] = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    best, best_v, curve = None, np.inf, {}
    for step in sorted(saved):
        den = nt.Denoiser(nt.load_checkpoint(model, saved[step]), cfg["mode"])
        v = _pooled(lambda X, t: den.posterior_mean(X, t), val)
        curve[step] = v
        if v < best_v:
            best, best_v = step, v
    den = nt.Denoiser(nt.load_checkpoint(model, saved[best]), cfg["mode"])
    return den, dict(checkpoint=best, at_cap=best == max(saved), val_selected=best_v,
                     wall=time.time() - t0, n_params=nt.n_params(model),
                     val_curve=";".join(f"{k}:{v:.5f}" for k, v in sorted(curve.items())))


def _em_arm(cfg, grid, w, groups, val, val_groups, seed, rho, L):
    """EM-BP on the same one-view groups. One ladder, two selection rules: the held-out BP risk
    (the registered rule, which uses the true kernel's posterior means on the validation bundle)
    and the held-out one-view log-evidence per edge (E20's oracle-free rule, computed on the
    noised views of the validation chains, the same views the network arm selects on)."""
    gs = [(X, *alpha_delta(t)) for X, t in groups]
    held = [(X, *alpha_delta(t)) for X, t in val_groups]
    n_edges = sum(len(X) for X, _ in val_groups) * (L - 1)
    k0 = MixtureInnovationKernel.init(cfg["components"], rho=cfg["em_init_rho"], var=cfg["em_init_var"],
                                      rng=rng_for("e14-init", seed, L, rho))
    ck = set(int(c) for c in cfg["em_checkpoints"])
    _, trace, saved = fit_em(k0, grid, w, gs, n_iters=max(ck), tol=FROZEN.em_loglik_tol, checkpoints=ck)
    best, best_v, best_e, best_ev, curve = None, np.inf, None, -np.inf, {}
    for it in sorted(saved):
        lk = saved[it].log_transition_matrix(grid)
        v = _pooled(lambda X, t, lk=lk: grid_bp_batch(grid, w, lk, X, *alpha_delta(t))[0], val)
        ev = float(e_step_multi(grid, w, lk, held).log_evidence) / n_edges
        curve[it] = (v, ev)
        if v < best_v:
            best, best_v = it, v
        if ev > best_ev:
            best_e, best_ev = it, ev
    out = []
    for it, sel in ((best, best_v), (best_e, best_ev)):
        kb = saved[it]
        out.append((kb, dict(checkpoint=it, at_cap=it == max(saved), val_selected=sel,
                             fit_kurt=float(kb.innovation_moments["innovation_excess_kurtosis"]),
                             fit_rho=float(kb.rho),
                             val_curve=";".join(f"{i}:{a:.5f}/{b:.5f}" for i, (a, b) in sorted(curve.items())))))
    return out


# ------------------------------------------------------------------ cells


def _common(cfg, rho, L, seed, n):
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    prior = LaplaceAR1(rho)
    log_k = prior.log_transition_matrix(grid)
    _, val = _bundle(prior, log_k, grid, w, "e14-val", seed, cfg["n_val"], L)
    _, test = _bundle(prior, log_k, grid, w, "e14-test", seed, cfg["n_test"], L)
    # ONE generator per set, created outside the comprehension. The first version of this line
    # re-created the generator inside it, so every chain of the training set was the same draw
    # (and every validation chain likewise); both arms then fitted 2048 copies of one chain, which
    # is what the superseded screen of 19 Sep measured (registry amendment of 20 Sep, N-22).
    rng_tr = rng_for("e14-train", seed, L, rho)
    A = np.stack([prior.sample(rng_tr, L) for _ in range(n)])
    rng_va = rng_for("e14-valchains", seed, L, rho)
    Av = np.stack([prior.sample(rng_va, L) for _ in range(cfg["n_val"])])
    assert len(np.unique(A[:, 0])) > 1 and len(np.unique(Av[:, 0])) > 1, "degenerate training set"
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
                    wall=np.nan, n_params=0, fit_kurt=np.nan, fit_rho=np.nan, val_curve="")


def _confirm_cell(args):
    cfg, rho, L, seed, winners = args
    grid, w, A, Av, val, test, base = _common(cfg, rho, L, seed, cfg["n_chains"])
    sigma0 = theory.chain_covariance(rho, 1.0 - rho ** 2, L)
    rows = []

    for arm, fn in (("lmmse_true", lambda X, t: theory.lmmse_posterior_mean(X, sigma0, t)),
                    ("isotropic", lambda X, t: (X - (1 - np.exp(-2 * t)) * X) / np.exp(-t))):
        rows.append({**base, **ROW_DEFAULTS, "arm": arm, "risk": _pooled(fn, test)})

    groups, idx = _one_view_groups(A, "e14-em", seed, L, return_index=True)
    val_groups = _one_view_groups(Av, "e14-emval", seed, L)

    for arm, (kb, info) in zip(("em_bp", "em_bp_evidence"), _em_arm(cfg, grid, w, groups, val, val_groups, seed, rho, L)):
        lk = kb.log_transition_matrix(grid)
        rows.append({**base, **ROW_DEFAULTS, "arm": arm,
                     "risk": _pooled(lambda X, t, lk=lk: grid_bp_batch(grid, w, lk, X, *alpha_delta(t))[0], test),
                     **info})

    for arch, ww in winners.items():
        den, ninfo = _train_ism(cfg, arch, ww["hp"], L, groups, val_groups, seed, cfg["net_steps"])
        rows.append({**base, **ROW_DEFAULTS, "arm": "ism_one_view", "arch": arch, "hp": str(ww["hp"]),
                     "risk": _pooled(lambda X, t: den.posterior_mean(X, t), test), **ninfo})
        den, ninfo = _train_dsm_one_corruption(cfg, arch, ww["hp"], L, A, groups, idx, val, seed, cfg["net_steps"])
        rows.append({**base, **ROW_DEFAULTS, "arm": "dsm_one_corruption", "arch": arch, "hp": str(ww["hp"]),
                     "risk": _pooled(lambda X, t: den.posterior_mean(X, t), test), **ninfo})
    print(f"  [E14] rho={rho} L={L} seed={seed}: "
          + " ".join(f"{r['arm']}{('/' + r['arch']) if r['arch'] else ''}={r['risk']:.4f}" for r in rows), flush=True)
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
