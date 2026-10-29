"""E55: the standard diffusion network against an agnostic neural kernel inside BP, on the phi^4 field; then longer chains
and more data.

Why (the user, 26 Oct 2026, after Jerome's call of 19 Oct): "the fight of a standard neural network with denoising score
matching -- usually a U-Net is used in diffusion models -- against a small network that just recovers the kernel and gives
it to BP, and BP gives the score. Still in a first-order Markov chain." And: "give no information to the small network, so
that it is comparable with the network"; "try longer Markov chains and bigger amounts of data, we have GPUs".

The arms, each told only what its method needs:
    unet          the DDPM U-Net (markov_score.unet_ddpm), standard denoising score matching on the CLEAN training
                  sequences: a fresh sequence, level and noise for every example of every step, EMA weights
    window        E54's window CNN (E33's per-cell radius and width) under the SAME recipe as the U-Net, so that its
                  numbers are no longer E54's 64k-step upper bounds
    em_energy     the neural-energy kernel (markov_score.energy_kernel): K(a'|a) = exp(-E(a, a'))/Z(a), E a small MLP,
                  uniform at the start; told only "first-order Markov, the same kernel at every link"; learned by
                  generalised EM through BP from ONE noisy copy of each training sequence (E33's interface)
    em_pair_quartic  the 5-parameter site energy (the ideal case, the physics known), E33's code and protocol, fitted only
                  where E33 has no frozen fit (M = 8192, 32768 and the two longer chains)
The comparison favours the networks: they see every clean sequence at every level with fresh noise; the kernels see one
noisy copy of each. Every arm's checkpoint is chosen by the same rule (validation risk against the exact BP score), every
arm's size by the same screen (dev seed, M = 32 and 2048, mean log validation risk), and every arm is scored on E33's
test bundles where E33 has them (same generator keys; the oracle part is the same-draw gate, as tools/core/
check_e54_pairing.py for E54).

Cells. E33's six (the T sweep and the two refinements), plus two LONGER chains of the base field (T = 0.3, h = 1, L = 128
and 512 sites). Sizes M = 32 ... 8192 sequences everywhere, 32768 on the base cell, 2048 at most at L = 512, so that the
largest fits all have about one million transitions.

Parts (independent, each its own CSV; a shard is one cell and/or arm through --set):
    oracles   linear floor (LMMSE) and isotropic risk per (cell, seed), bundles built ON THE CPU: the pairing gate
    timing    dev-seed throughput probe that sets the budgets (never a result)
    screen    size/learning-rate screen on the dev seed at M = 32 and 2048 (screening.csv)
    winners   the registered rule applied to the screening CSVs (winners.json)
    nets      unet and window, confirmatory seeds (nets.csv)
    energy    em_energy, confirmatory seeds (kernels.csv)
    pair      em_pair_quartic where E33 has no fit (pair.csv)

Predictions: experiments/registry/E55.json. Reading: tools/core/read_e55.py.
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e14_regime_baselines as e14  # noqa: E402
import e33_field_learning as e33  # noqa: E402
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from e32_field_oracles import CELLS as E32_CELLS, resolve_cell  # noqa: E402
from markov_score import theory  # noqa: E402
from markov_score.backend import get_xp, to_host  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.em import e_step_multi, fit_em  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
T_SCHEDULE = e33.T_SCHEDULE
E33_CELLS = tuple(e33.SETTINGS["cells"])
NEW_CELLS = (dict(id="T0.3_h1_L128", T=0.3, h=1.0, L=128, role="length"),
             dict(id="T0.3_h1_L512", T=0.3, h=1.0, L=512, role="length"))
CELLS = E33_CELLS + tuple(c["id"] for c in NEW_CELLS)
SIZES = {**{c: (32, 128, 512, 2048, 8192) for c in E33_CELLS}, "T0.3_h1_L32": (32, 128, 512, 2048, 8192, 32768),
         "T0.3_h1_L128": (32, 128, 512, 2048, 8192), "T0.3_h1_L512": (32, 128, 512, 2048)}
E33_SIZES = (32, 128, 512, 2048)          # where E33 has frozen site-energy fits
# E33's per-cell window radius and width (outputs/hpc_E33n_s0/winners.json, frozen). Copied here because outputs/ is not
# deployed to the cluster (timing job 720363 failed on the file); tests/test_e55_driver.py holds the copy equal to the file.
E33_WINDOW_HP = {"T0.3_h1_L32": dict(radius=2, width=64), "T0.3_h0.5_L64": dict(radius=4, width=64),
                 "T0.3_h0.25_L128": dict(radius=8, width=64), "T0.5_iso_L32": dict(radius=4, width=128),
                 "T1_iso_L32": dict(radius=4, width=64), "T2_iso_L32": dict(radius=4, width=64)}
WINDOW_CELL = {"T0.3_h1_L128": "T0.3_h1_L32", "T0.3_h1_L512": "T0.3_h1_L32"}   # same field, same window
EM_LADDER = (10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200, 1600, 2400)

SETTINGS = dict(
    cells=CELLS, sizes=SIZES, seeds=tuple(range(8)), dev_seed=10_000, screen_sizes=(32, 2048),
    n_val=e33.SETTINGS["n_val"], n_test=e33.SETTINGS["n_test"],
    grid_size=e33.SETTINGS["grid_size"], half_width=e33.SETTINGS["half_width"],
    em_init_rho=e33.SETTINGS["em_init_rho"], em_init_var=e33.SETTINGS["em_init_var"],
    em_checkpoints=e33.SETTINGS["em_checkpoints"],        # the pair arm: E33's ladder, unchanged
    energy_checkpoints=EM_LADDER, energy_lr=1e-2, energy_inner=10,
    net_arms=("unet", "window"),
    screen=dict(unet=[dict(base=b, lr=lr) for b in (32, 64, 128) for lr in (2e-4, 1e-3)],
                window=[dict(lr=lr) for lr in (2e-4, 1e-3)],
                em_energy=[dict(hidden=h) for h in (16, 32, 64)]),
    winners=None,
    # half-octave ladder from 16 steps (amendments 2026-10-27 and 2026-10-29: in the screen every U-Net at M = 32 chose the
    # then-first rung, 256); the window CNN trains to 2^20 steps (2026-10-29: at M = 2048 it chose the last rung, 2^18, on
    # every cell); each net uses the rungs up to its own budget
    steps=2**18, window_steps=2**20, net_checkpoints=tuple(sorted({int(round(2 ** (k / 2))) for k in range(8, 41)})),
    batch_size=128, warmup=1000,
    ema_decay=0.999, grad_clip=1.0, torch_threads=2, device="auto", eval_chunk=512, compile=True, compile_mode="reduce-overhead",
    timing_steps=2000, timing_em_iters=5,
    timing_probes=None,          # None = the registered probe list in part_timing; e55-timing8 passes its own
    cost_lengths=(32, 64, 128, 512), cost_unet_bases=(32, 64, 128), cost_batches_gpu=(1, 64, 1024), cost_batches_cpu=(1, 64),
    cost_reps=7, cost_gpu=True,
    # generation (amendment 2026-10-28): networks retrained on log-uniform levels in [gen_tmin, gen_tmax], all arms through
    # the same 400-level ancestral sampler (tests/test_ancestral.py: 0.98 of the variance with the exact denoiser)
    gen_cells=("T0.3_h1_L32", "T1_iso_L32", "T0.3_h0.25_L128", "T0.3_h1_L512"), gen_sizes=(128, 2048), gen_seeds=(0, 1, 2, 3),
    gen_n=2048, gen_steps=400, gen_tmin=0.005, gen_tmax=3.0, gen_kernel_csvs=(), gen_e33_csv="data/e55_gen_e33_site_energy.csv",
    screen_arms=("unet", "window", "em_energy"), screen_glob=None,   # winners: None = this directory's screening.csv
)
QUICK = dict(cells=("T0.3_h1_L32",), sizes={"T0.3_h1_L32": (32,)}, seeds=(10_001,), screen_sizes=(32,),
             n_val=32, n_test=32, grid_size=101, em_checkpoints=(2, 4), energy_checkpoints=(2, 4),
             screen=dict(unet=[dict(base=8, lr=1e-3)], window=[dict(lr=1e-3)], em_energy=[dict(hidden=8)]),
             steps=64, window_steps=64, net_checkpoints=(32, 64), batch_size=16, warmup=8, timing_steps=20, timing_em_iters=2,
             cost_lengths=(32,), cost_unet_bases=(8,), cost_batches_cpu=(1, 4), cost_reps=2, cost_gpu=False,
             gen_cells=("T0.3_h1_L32",), gen_sizes=(32,), gen_seeds=(10_001,), gen_n=128, gen_steps=20,
             timing_probes=[("unet", dict(base=8, lr=1e-3), "T0.3_h1_L32", 32), ("em_energy", dict(hidden=8), "T0.3_h1_L32", 32),
                            ("em_pair_quartic", {}, "T0.3_h1_L32", 32), ("parity", dict(n_val=24, n_test=24), "T0.3_h1_L32", 32)])


# ------------------------------------------------------------------ cells and bundles


def find_cell(cell_id: str) -> dict:
    return next(c for c in tuple(E32_CELLS) + NEW_CELLS if c["id"] == cell_id)


def _bundle(chain, log_k, log_mu, grid, w, tag, seed, cell_id, n_chains, L, xp):
    """e33._bundle with the reference BP on `xp`: the same draws, the reference means to round-off."""
    rng = rng_for(tag, seed, cell_id)
    A = chain.sample_batch(rng, n_chains, L)
    out = {}
    for t in T_SCHEDULE:
        al, de = alpha_delta(t)
        X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
        out[t] = (X, to_host(grid_bp_batch(grid, w, log_k, X, al, de, log_mu=log_mu, xp=xp)[0]))
    return A, out


def common(cfg, cell_id, seed, n, xp=np):
    """e33._common for any cell of E55, with the same stream tags (keyed by the cell id) and the reference BP on `xp`.
    tests/test_e55_driver.py holds it equal to e33._common, array for array, on an E33 cell."""
    cell = find_cell(cell_id)
    chain, info = resolve_cell(cell)
    L = cell["L"]
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    log_k = chain.log_transition_matrix(grid)
    log_mu = chain.log_stationary(grid)
    _, val = _bundle(chain, log_k, log_mu, grid, w, "e33-val", seed, cell_id, cfg["n_val"], L, xp)
    _, test = _bundle(chain, log_k, log_mu, grid, w, "e33-test", seed, cell_id, cfg["n_test"], L, xp)
    A = chain.sample_batch(rng_for("e33-train", seed, cell_id, n), n, L)
    Av = chain.sample_batch(rng_for("e33-valchains", seed, cell_id), cfg["n_val"], L)
    assert len(np.unique(A[:, 0])) > 1 and len(np.unique(Av[:, 0])) > 1, "degenerate training set"
    sigma0 = chain.covariance(L)
    base = dict(**info, seed=seed, n_chains=n,
                t_star=float(0.5 * np.log(np.linalg.eigvalsh(sigma0).max())),
                val_bundle=e14._bundle_hash(val), test_bundle=e14._bundle_hash(test),
                val_draws=draws_hash(val), test_draws=draws_hash(test))
    return chain, grid, w, log_k, log_mu, sigma0, A, Av, val, test, base


def draws_hash(bundle) -> str:
    """Hash of the noisy draws X alone. The bundle hash also covers the reference means, whose last bits depend on the
    device and the BLAS library (E54's pairing amendment); the draws come from numpy's generator and do not. Equal draw
    hashes are the pairing gate between E55's CPU-built (oracles) and GPU-built (fits) bundles."""
    import hashlib
    h = hashlib.sha256()
    for t in sorted(bundle):
        h.update(np.ascontiguousarray(bundle[t][0], dtype=np.float64).tobytes())
    return h.hexdigest()[:12]


def _private_compile_caches():
    """Each worker process compiles into its OWN cache directories on the node's local disk. With one shared cache, eight
    workers compiling the same graph at once corrupted a Triton cache JSON (JSONDecodeError: Extra data) and killed three
    screen jobs in their first minute (723257, 723260, 723261, 28 Oct 2026). Set before anything is compiled."""
    import os
    import tempfile
    root = Path(os.environ.get("TMPDIR") or tempfile.gettempdir()) / f"e55_{os.environ.get('SLURM_JOB_ID', 'local')}_{os.getpid()}"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(root / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(root / "triton")
    if not getattr(_private_compile_caches, "registered", False):     # the node's /tmp is shared: clean up at exit
        import atexit
        import shutil
        atexit.register(shutil.rmtree, root, True)
        _private_compile_caches.registered = True
    return root


def pooled_levels(mean_fn, bundle):
    """e14._pooled -- the same sums in the same order, so the pooled value is identical -- and the relative score error at
    each of the twelve levels, sqrt(sum (s_hat - s*)^2 / sum s*^2) over the test sequences and sites of that level."""
    num = den = 0.0
    per = {}
    for t in T_SCHEDULE:
        X, m_ref = bundle[t]
        al, de = alpha_delta(t)
        s_hat = -(X - al * mean_fn(X, t)) / de
        s_ref = -(X - al * m_ref) / de
        n_t = float(((s_hat - s_ref) ** 2).sum()); d_t = float((s_ref ** 2).sum())
        num += n_t; den += d_t
        per[t] = float(np.sqrt(n_t / d_t))
    return float(np.sqrt(num / den)), per


def _levels(per) -> str:
    return ";".join(f"{t:g}:{v:.6f}" for t, v in sorted(per.items()))


def dsm_validation_set(Av, cell_id, seed):
    """The networks' own selection criterion: clean validation sequences (E33's validation chains) with one FIXED noise
    draw per level, so every checkpoint is scored on the same examples."""
    rng = rng_for("e55-dsmval", cell_id, seed)
    out = []
    for t in T_SCHEDULE:
        al, de = alpha_delta(t)
        eps = rng.standard_normal(Av.shape)
        out.append((t, al * Av + np.sqrt(de) * eps, eps))
    return out


def dsm_loss(mean_fn, dsm_set) -> float:
    """Validation DSM loss, mean over levels of mean (eps_hat - eps)^2, eps_hat = (x - alpha m(x)) / sqrt(Delta): the
    training loss of the networks on held-out sequences (uniform over the levels, as in training)."""
    tot = 0.0
    for t, X, eps in dsm_set:
        al, de = alpha_delta(t)
        tot += float(np.mean(((X - al * mean_fn(X, t)) / np.sqrt(de) - eps) ** 2))
    return tot / len(dsm_set)


def _setup(cfg):
    _private_compile_caches()
    import torch
    torch.set_num_threads(cfg["torch_threads"])
    from markov_score import nn_torch as nt
    dev = nt.pick_device(cfg["device"])
    return torch, dev


# ------------------------------------------------------------------ networks


def _window_hp(cell_id):
    return dict(E33_WINDOW_HP[WINDOW_CELL.get(cell_id, cell_id)])


def _fit_net(cfg, arm, conf, cell_id, seed, n, dev, steps=None, checkpoints=None, t_range=None, return_model=False):
    import torch
    from markov_score import nn_torch as nt
    from markov_score import unet_ddpm as ud
    chain, grid, w, log_k, log_mu, sigma0, A, Av, val, test, base = common(cfg, cell_id, seed, n, get_xp())
    L = base["n_sites"]
    init_seed = int(rng_for("e55-init", arm, cell_id, seed, n).integers(0, 2**31 - 1))
    if arm == "unet":
        hp = dict(base=conf["base"])
        model = ud.build(init_seed, **hp)
    else:
        hp = _window_hp(cell_id)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(init_seed)
            model = nt.build("window", L, **hp)
    budget = int(steps or (cfg.get("window_steps", cfg["steps"]) if arm == "window" else cfg["steps"]))
    ck = tuple(checkpoints or tuple(c for c in cfg["net_checkpoints"] if c <= budget))
    dcfg = ud.DSMConfig(steps=budget, batch_size=cfg["batch_size"], lr=float(conf["lr"]),
                        warmup=cfg["warmup"], grad_clip=cfg["grad_clip"], ema_decay=cfg["ema_decay"], checkpoints=ck,
                        seed=int(rng_for("e55-train", arm, cell_id, seed, n).integers(0, 2**31 - 1)), device=dev,
                        compile=bool(conf.get("compile", cfg["compile"])),
                        compile_mode=str(conf.get("compile_mode", cfg["compile_mode"])), t_range=t_range)
    t0 = time.time()
    clock = {}
    saved = ud.train_dsm(model, A, T_SCHEDULE, dcfg, clock=clock)
    t_train = time.time() - t0
    mean_fn = lambda X, t: ud.posterior_mean(model, X, t, cfg["eval_chunk"])  # noqa: E731
    dsm_set = dsm_validation_set(Av, cell_id, seed)
    best, best_v, curve, dsm_best, dsm_v, dsm_curve = None, np.inf, {}, None, np.inf, {}
    for step in sorted(saved):
        model.load_state_dict(saved[step])
        v = e14._pooled(mean_fn, val)
        curve[step] = v
        if v < best_v:
            best, best_v = step, v
        d = dsm_loss(mean_fn, dsm_set)          # realistic selection: the network's own loss, no exact score
        dsm_curve[step] = d
        if d < dsm_v:
            dsm_best, dsm_v = step, d
    model.load_state_dict(saved[best])
    risk, per = pooled_levels(mean_fn, test)
    if dsm_best == best:
        risk_dsm, per_dsm = risk, per
    else:
        model.load_state_dict(saved[dsm_best])
        risk_dsm, per_dsm = pooled_levels(mean_fn, test)
        model.load_state_dict(saved[best])
    row = {**base, "arm": arm, "hp": json.dumps(hp), "lr": float(conf["lr"]), "steps": dcfg.steps, "compiled": int(dcfg.compile),
            "checkpoint": best, "at_cap": int(best == max(saved)), "val_selected": best_v, "risk": risk,
            "risk_levels": _levels(per), "dsm_checkpoint": dsm_best, "dsm_at_cap": int(dsm_best == max(saved)),
            "dsm_val_selected": dsm_v, "risk_dsm": risk_dsm, "risk_dsm_levels": _levels(per_dsm),
            "dsm_curve": ";".join(f"{k}:{v:.6f}" for k, v in sorted(dsm_curve.items())),
            "n_params": ud.n_params(model), "train_seconds": t_train, "wall": time.time() - t0, "device": str(dev),
            "ckpt_seconds": ";".join(f"{k}:{v:.2f}" for k, v in sorted(clock.items())),
            "val_curve": ";".join(f"{k}:{v:.6f}" for k, v in sorted(curve.items())),
            "t_range": "" if t_range is None else f"{t_range[0]:g}-{t_range[1]:g}"}
    return (row, model) if return_model else row


# ------------------------------------------------------------------ kernels


def _mean_fn(grid, w, lk, mu, xp):
    return lambda X, t: to_host(grid_bp_batch(grid, w, lk, X, *alpha_delta(t), log_mu=mu, xp=xp)[0])


def fit_kernel(cfg, arm, k0, grid, w, groups, val, val_groups, test, L, ladder, xp=np):
    """e33._fit_arm with the kernel passed in and the BP scoring on `xp` (tests/test_e55_driver.py holds the two equal
    on the CPU): generalised EM on the one-view noisy groups, the kernel's own stationary law as the initial law,
    checkpoints chosen by validation risk ("risk") and by held-out evidence ("evidence"); a FloatingPointError fails
    this fit only (E33 amendment 3)."""
    gs = [(X, *alpha_delta(t)) for X, t in groups]
    held = [(X, *alpha_delta(t)) for X, t in val_groups]
    n_edges = sum(len(X) for X, _ in val_groups) * (L - 1)
    mu_fn = e33._stationary_mu(grid, w)
    ck = set(int(c) for c in ladder)
    t0 = time.time()
    try:
        _, trace, saved = fit_em(k0, grid, w, gs, n_iters=max(ck), tol=e33.FROZEN.em_loglik_tol, checkpoints=ck,
                                 log_mu=mu_fn)
    except FloatingPointError as exc:
        return [e33._failed_row(arm, sel, k0, time.time() - t0, f"FloatingPointError: {exc}") for sel in ("risk", "evidence")]
    wall = time.time() - t0
    curve, mus = {}, {}
    best, best_v, best_e, best_ev = None, np.inf, None, -np.inf
    for it in sorted(saved):
        mu = mu_fn(saved[it]); mus[it] = mu
        lk = saved[it].log_transition_matrix(grid)
        try:
            v = e14._pooled(_mean_fn(grid, w, lk, mu, xp), val)
        except FloatingPointError:
            v = np.inf
        try:
            ev = float(e_step_multi(grid, w, lk, held, mu).log_evidence) / n_edges
        except FloatingPointError:
            ev = -np.inf
        curve[it] = (v, ev)
        if v < best_v:
            best, best_v = it, v
        if ev > best_ev:
            best_e, best_ev = it, ev
    rows = []
    for selection, it, sel in (("risk", best, best_v), ("evidence", best_e, best_ev)):
        if it is None:
            rows.append(e33._failed_row(arm, selection, k0, wall, "no checkpoint of the ladder could be scored"))
            continue
        kb = saved[it]
        try:
            risk, per = pooled_levels(_mean_fn(grid, w, kb.log_transition_matrix(grid), mus[it], xp), test)
        except FloatingPointError as exc:
            rows.append(e33._failed_row(arm, selection, k0, wall, f"test scoring of checkpoint {it}: FloatingPointError: {exc}"))
            continue
        rows.append(dict(arm=arm, selection=selection, risk=risk, risk_levels=_levels(per), failed=0, checkpoint=it, at_cap=int(it == max(saved)),
                         val_selected=sel, n_params=int(len(np.asarray(kb.theta))), wall=wall,
                         converged=bool(trace.converged), stop_reason=trace.stop_reason,
                         monotone_violation=float(trace.monotone_violation), n_iters=len(trace.log_evidence),
                         **e33._kernel_diagnostics(kb, grid), final_lr=float(getattr(kb, "lr", np.nan)),
                         theta=";".join(f"{v:.8g}" for v in np.asarray(kb.theta, dtype=float)),
                         val_curve=";".join(f"{i}:{a:.6f}/{b:.6f}" for i, (a, b) in sorted(curve.items()))))
    return rows


def _kernel0(cfg, arm, conf, grid, seed, cell_id, n, dev):
    if arm == "em_pair_quartic":
        return e33._init_kernel(arm, cfg, grid, seed, cell_id, n)
    from markov_score.energy_kernel import NeuralEnergyKernel
    return NeuralEnergyKernel.init(hidden=int(conf["hidden"]), seed=int(rng_for("e55-init", arm, cell_id, seed, n).integers(0, 2**31 - 1)),
                                   half_width=cfg["half_width"], lr=cfg["energy_lr"], n_inner=cfg["energy_inner"], device=dev)


def _fit_kernel_cell(cfg, arm, conf, cell_id, seed, n, dev, ladder=None):
    xp = get_xp()
    chain, grid, w, log_k, log_mu, sigma0, A, Av, val, test, base = common(cfg, cell_id, seed, n, xp)
    L = base["n_sites"]
    groups = e14._one_view_groups(A, f"e33-em-{cell_id}-M{n}", seed, L)        # E33's views, for E33's cells and sizes
    val_groups = e14._one_view_groups(Av, f"e33-emval-{cell_id}", seed, L)
    k0 = _kernel0(cfg, arm, conf, grid, seed, cell_id, n, dev)
    lad = ladder or (cfg["em_checkpoints"] if arm == "em_pair_quartic" else cfg["energy_checkpoints"])
    rows = fit_kernel(cfg, arm, k0, grid, w, groups, val, val_groups, test, L, lad, xp)
    return [{**base, **e33.ROW_DEFAULTS, "hp": json.dumps(conf), "device": str(dev), **r} for r in rows]


# ------------------------------------------------------------------ parts


def _shard_cells(cfg):
    size = {c: find_cell(c)["L"] for c in cfg["cells"]}
    return sorted(cfg["cells"], key=lambda c: -size[c])


def _oracles_cell(args):
    cfg, cell_id, seed = args
    chain, grid, w, log_k, log_mu, sigma0, A, Av, val, test, base = common(cfg, cell_id, seed, 32, np)
    rows = [{**base, "arm": "lmmse_true", "risk": e14._pooled(lambda X, t: theory.lmmse_posterior_mean(X, sigma0, t), test)},
            {**base, "arm": "isotropic", "risk": e14._pooled(lambda X, t: (X - (1 - np.exp(-2 * t)) * X) / np.exp(-t), test)}]
    print(f"  [E55 oracles] {cell_id} seed={seed}: floor {rows[0]['risk']:.4f}", flush=True)
    return rows


def part_oracles(cfg, out, workers):
    cells = [(cfg, c, s) for c in _shard_cells(cfg) for s in cfg["seeds"]]
    write_csv(out / "oracles.csv", run_cells(cells, _oracles_cell, workers, out, "oracles.csv", key=lambda c: f"{c[1]}_s{c[2]}"))


def _screen_cell(args):
    cfg, arm, conf, cell_id, n = args
    torch, dev = _setup(cfg)
    seed = cfg["dev_seed"]
    if arm == "em_energy":
        rows = [r for r in _fit_kernel_cell(cfg, arm, conf, cell_id, seed, n, dev) if r["selection"] == "risk"]
    else:
        rows = [_fit_net(cfg, arm, conf, cell_id, seed, n, dev)]
    for r in rows:
        r.update(conf_key=json.dumps(conf, sort_keys=True), screen_arm=arm)
        print(f"  [E55 screen] {arm} {conf} {cell_id} M={n}: val {r['val_selected']:.4f} (ckpt {r['checkpoint']})", flush=True)
    return rows


def part_screen(cfg, out, workers):
    arms = list(cfg["screen_arms"])
    cells = [(cfg, a, conf, c, n) for c in _shard_cells(cfg) for a in arms for conf in cfg["screen"][a]
             for n in cfg["screen_sizes"]]
    write_csv(out / "screening.csv", run_cells(cells, _screen_cell, workers, out, "screening.csv",
                                               key=lambda c: f"{c[1]}_{c[3]}_M{c[4]}_" + "_".join(f"{k}{v}" for k, v in sorted(c[2].items()))))


def choose_winners(screen_rows) -> dict:
    """The registered rule: per (cell, arm), the configuration with the lowest mean over the screen sizes of
    log(validation risk at its selected checkpoint); a configuration missing a size, or with a non-finite risk, cannot win;
    ties go to the configuration listed first in SETTINGS['screen'] (the smaller one)."""
    import pandas as pd
    df = pd.DataFrame(screen_rows)
    out = {}
    for (cell, arm), g in df.groupby(["cell", "screen_arm"]):
        n_sizes = g.n_chains.nunique()
        score = {}
        for key, h in g.groupby("conf_key", sort=False):
            v = h.val_selected.astype(float)
            if h.n_chains.nunique() == n_sizes and np.isfinite(v).all() and (v > 0).all():
                score[key] = float(np.log(v).mean())
        if score:
            order = [json.dumps(c, sort_keys=True) for c in SETTINGS["screen"][arm]]
            best = min(score, key=lambda k: (score[k], order.index(k) if k in order else len(order)))
            out[f"{cell}|{arm}"] = dict(conf=json.loads(best), score=score[best], scores=score)
    return out


def part_winners(cfg, out, workers):
    paths = sorted(glob.glob(str(Path(cfg["screen_glob"])))) if cfg.get("screen_glob") else [str(out / "screening.csv")]
    import pandas as pd
    rows = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True).to_dict("records")
    w = choose_winners(rows)
    write_json(out / "winners.json", dict(sources=paths, winners=w))
    for k, v in sorted(w.items()):
        print(f"  [E55 winners] {k}: {v['conf']}  (mean log val {v['score']:.4f})", flush=True)


def _winner(cfg, cell_id, arm):
    w = cfg["winners"]
    if isinstance(w, str):
        w = json.loads(Path(w).read_text())["winners"]
    if not w or f"{cell_id}|{arm}" not in w:
        raise SystemExit(f"no screen winner for {cell_id}|{arm}: pass --set winners=\"'<winners.json>'\"")
    return w[f"{cell_id}|{arm}"]["conf"]


def _nets_cell(args):
    cfg, arm, cell_id, seed, n = args
    torch, dev = _setup(cfg)
    row = _fit_net(cfg, arm, _winner(cfg, cell_id, arm), cell_id, seed, n, dev)
    print(f"  [E55 nets] {arm} {cell_id} seed={seed} M={n}: risk {row['risk']:.4f} (ckpt {row['checkpoint']}, "
          f"{row['wall']:.0f}s)", flush=True)
    return [row]


def part_nets(cfg, out, workers):
    cells = [(cfg, a, c, s, n) for c in _shard_cells(cfg) for n in sorted(cfg["sizes"][c], reverse=True)
             for a in cfg["net_arms"] for s in cfg["seeds"]]
    write_csv(out / "nets.csv", run_cells(cells, _nets_cell, workers, out, "nets.csv", key=lambda c: f"{c[1]}_{c[2]}_s{c[3]}_M{c[4]}"))


def _kernel_cell(args):
    cfg, arm, cell_id, seed, n = args
    torch, dev = _setup(cfg)
    conf = _winner(cfg, cell_id, arm) if arm == "em_energy" else {}
    rows = _fit_kernel_cell(cfg, arm, conf, cell_id, seed, n, dev)
    print(f"  [E55 {arm}] {cell_id} seed={seed} M={n}: " + " ".join(f"{r['selection']}={r['risk']:.4f}" for r in rows), flush=True)
    return rows


def part_energy(cfg, out, workers):
    cells = [(cfg, "em_energy", c, s, n) for c in _shard_cells(cfg) for n in sorted(cfg["sizes"][c], reverse=True)
             for s in cfg["seeds"]]
    write_csv(out / "kernels.csv", run_cells(cells, _kernel_cell, workers, out, "kernels.csv", key=lambda c: f"{c[2]}_s{c[3]}_M{c[4]}"))


def pair_targets(cfg):
    """(cell, M) where the site energy needs a fit: everything E33 did not fit (E33: its six cells at M <= 2048)."""
    return [(c, n) for c in _shard_cells(cfg) for n in sorted(cfg["sizes"][c], reverse=True)
            if not (c in E33_CELLS and n in E33_SIZES)]


def part_pair(cfg, out, workers):
    cells = [(cfg, "em_pair_quartic", c, s, n) for c, n in pair_targets(cfg) for s in cfg["seeds"]]
    if not cells:
        print("  [E55 pair] nothing to fit: E33 has every (cell, M) of this shard", flush=True); return
    write_csv(out / "pair.csv", run_cells(cells, _kernel_cell, workers, out, "pair.csv", key=lambda c: f"{c[2]}_s{c[3]}_M{c[4]}"))


def _timing_cell(args):
    """Throughput on the dev seed: seconds per network step and per EM iteration at the given cell and size."""
    cfg, what, conf, cell_id, n = args
    torch, dev = _setup(cfg)
    seed = cfg["dev_seed"]
    t0 = time.time()
    if what == "parity":
        # the reference means of one test bundle built on the CPU and on the BP device, on THIS hardware
        pcfg = {**cfg, **conf}                       # a smaller bundle: the CPU build of a full L = 512 one takes ~30 min
        cpu = common(pcfg, cell_id, seed, n, np)[9]
        devb = common(pcfg, cell_id, seed, n, get_xp())[9]
        dev_max = max(float(np.max(np.abs(cpu[t][1] - devb[t][1]))) for t in cpu)
        same = all(np.array_equal(cpu[t][0], devb[t][0]) for t in cpu)
        print(f"  [E55 parity] {cell_id}: max |m_dev - m_cpu| = {dev_max:.2e}, draws equal: {same}", flush=True)
        return [dict(what=what, conf="{}", cell=cell_id, n_chains=n, units=1, seconds_per_unit=time.time() - t0,
                     total_seconds=time.time() - t0, device=str(get_xp().__name__), risk=dev_max, draws_equal=int(same))]
    if what in ("unet", "window"):
        k = cfg["timing_steps"]
        row = _fit_net(cfg, what, conf, cell_id, seed, n, dev, steps=k, checkpoints=(k // 2, k))
        c = dict(kv.split(":") for kv in row["ckpt_seconds"].split(";"))
        per = (float(c[str(k)]) - float(c[str(k // 2)])) / (k - k // 2)     # steady state: compilation excluded
    else:
        k = cfg["timing_em_iters"]
        rows = _fit_kernel_cell(cfg, what, conf, cell_id, seed, n, dev, ladder=(k,))
        row = next(r for r in rows if r["selection"] == "risk")
        per = row["wall"] / k
    r = dict(what=what, conf=json.dumps(conf), cell=cell_id, n_chains=n, units=k, seconds_per_unit=per,
             total_seconds=time.time() - t0, device=str(dev), risk=row["risk"])
    print(f"  [E55 timing] {what} {conf} {cell_id} M={n}: {per:.4f} s/unit", flush=True)
    return [r]


def part_timing(cfg, out, workers):
    probes = cfg.get("timing_probes") or [("unet", dict(base=128, lr=2e-4), c, 2048) for c in ("T0.3_h1_L32", "T0.3_h1_L512")] + \
        [("unet", dict(base=32, lr=2e-4), "T0.3_h1_L32", 2048), ("window", dict(lr=2e-4), "T0.3_h1_L512", 2048)] + \
        [("em_energy", dict(hidden=64), "T0.3_h1_L32", 32768), ("em_energy", dict(hidden=64), "T0.3_h1_L512", 2048),
         ("em_pair_quartic", {}, "T0.3_h1_L512", 2048), ("parity", dict(n_val=24, n_test=240), "T0.3_h1_L512", 32)]
    cells = [(cfg, *p) for p in probes]
    write_csv(out / "timing.csv", run_cells(cells, _timing_cell, workers, out, "timing.csv",
                                            key=lambda c: f"{c[1]}_{c[3]}_M{c[4]}_" + "_".join(f"{k}{v}" for k, v in sorted(c[2].items()))))


COST_WINDOW = {32: "T0.3_h1_L32", 64: "T0.3_h0.5_L64", 128: "T0.3_h0.25_L128", 512: "T0.3_h1_L32"}   # the cell's window


def _timed(fn, sync, reps):
    """Median seconds per call after two warm-up calls; `sync` waits for the device."""
    for _ in range(2):
        fn(); sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.min(ts))


def _cost_cell(args):
    """Inference cost: seconds to produce the score of B sequences of L sites at one level, on the GPU or one CPU thread.
    The cost of a network depends on its size only, and the cost of BP on the grid (N_g, L, B) only, never on the fitted
    values, so untrained models are timed; 'build' rows time the one-off construction of a learned kernel's grid matrix
    and stationary law."""
    cfg, what, conf, L, B, device = args
    import torch
    torch.set_num_threads(1)
    _private_compile_caches()
    from markov_score import nn_torch as nt
    from markov_score import unet_ddpm as ud
    t = 0.3215
    al, de = alpha_delta(t)
    X = rng_for("e55-cost", L, B).standard_normal((B, L))
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    chain, _ = resolve_cell(find_cell("T0.3_h1_L32"))
    lk, lmu = chain.log_transition_matrix(grid), chain.log_stationary(grid)
    peak = np.nan
    if what in ("unet", "window"):
        dev = torch.device(device)
        model = (ud.build(0, base=conf["base"]) if what == "unet" else nt.build("window", L, **_window_hp(COST_WINDOW[L])))
        model = model.to(dev).eval()
        sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        med, best = _timed(lambda: ud.posterior_mean(model, X, t, chunk=max(B, 1)), sync, cfg["cost_reps"])
        if dev.type == "cuda":
            peak = float(torch.cuda.max_memory_allocated())
        n_par = ud.n_params(model)
    elif what == "bp":
        xp = get_xp() if device == "cuda" else np
        sync = (lambda: xp.cuda.Device().synchronize()) if xp is not np else (lambda: None)
        med, best = _timed(lambda: grid_bp_batch(grid, w, lk, X, al, de, log_mu=lmu, xp=xp), sync, cfg["cost_reps"])
        if xp is not np:
            peak = float(xp.get_default_memory_pool().total_bytes())
        n_par = 0
    else:                                   # build: a learned kernel's grid matrix and its stationary law, once per model
        if what == "build_energy":
            from markov_score.energy_kernel import NeuralEnergyKernel
            k = NeuralEnergyKernel.init(hidden=int(conf["hidden"]), seed=0, half_width=cfg["half_width"], device=device)
        else:
            k = e33._init_kernel("em_pair_quartic", cfg, grid, 0, "T0.3_h1_L32", 32)
        mu_fn = e33._stationary_mu(grid, w)
        med, best = _timed(lambda: (k.log_transition_matrix(grid), mu_fn(k)), lambda: None, cfg["cost_reps"])
        n_par = int(len(np.asarray(k.theta)))
    row = dict(what=what, conf=json.dumps(conf), n_sites=L, batch=B, device=device, threads=1, grid_size=cfg["grid_size"],
               seconds_per_call=med, seconds_min=best, seconds_per_sequence=med / max(B, 1), peak_bytes=peak, n_params=n_par)
    print(f"  [E55 cost] {what} {conf} L={L} B={B} {device}: {med * 1e3:.2f} ms/call", flush=True)
    return [row]


def part_cost(cfg, out, workers):
    devices = ("cuda", "cpu") if cfg["cost_gpu"] else ("cpu",)
    cells = []
    for L in cfg["cost_lengths"]:
        for device in devices:
            for B in (cfg["cost_batches_gpu"] if device == "cuda" else cfg["cost_batches_cpu"]):
                cells += [(cfg, "unet", dict(base=b), L, B, device) for b in cfg["cost_unet_bases"]]
                cells += [(cfg, "window", {}, L, B, device), (cfg, "bp", {}, L, B, device)]
    for device in devices:
        cells += [(cfg, "build_energy", dict(hidden=h), 0, 0, device) for h in (16, 32, 64)]
    cells += [(cfg, "build_pair", {}, 0, 0, "cpu")]
    write_csv(out / "cost.csv", run_cells(cells, _cost_cell, 1, out, "cost.csv",
                                          key=lambda c: f"{c[1]}_L{c[3]}_B{c[4]}_{c[5]}_" + "_".join(f"{k}{v}" for k, v in sorted(c[2].items()))))


# ------------------------------------------------------------------ generation (amendment 2026-10-28, option A)


def generation_stats(G, R, chain, info, L):
    """Distances of a generated batch G to exact samples R of the same cell, and G's physical statistics (E34's)."""
    from e34_field_physics import physical_statistics
    from markov_score import gen_metrics as gm
    st = physical_statistics(G, chain.scale, L, info["h"])
    corr = np.array([float(v) for v in st["corr"].split(";")])
    corr_true = chain.covariance(L)[0]
    return dict(swd=gm.sliced_wasserstein2(G, R, n_proj=256, seed=0), w1_marginal=gm.wasserstein1_1d(G, R),
                corr_max_err=float(np.max(np.abs(corr - corr_true))), corr_rms_err=float(np.sqrt(np.mean((corr - corr_true) ** 2))),
                var=st["var"], excess_kurtosis=st["excess_kurtosis"], kink_density=st["kink_density"],
                coarse_kink_density=st["coarse_kink_density"], coarse_mean_domain=st["coarse_mean_domain"],
                barrier_mass=st["barrier_mass"], bimodality=st["bimodality_coefficient"])


def _kernel_row(paths, arm, cell_id, n, seed):
    """The risk-selected fit of `arm` at (cell, M, seed) from the CSVs that hold it, or None."""
    import pandas as pd
    for path in paths:
        if not Path(path).exists():
            continue
        d = pd.read_csv(path)
        d = d[(d.arm == arm) & (d.selection == "risk") & (d.cell == cell_id) & (d.n_chains == n) & (d.seed == seed)]
        d = d[d.theta.notna() & (d.failed.fillna(0) == 0)] if "failed" in d else d[d.theta.notna()]
        if len(d):
            return d.iloc[0]
    return None


def _gen_cell(args):
    """One (cell, M, seed): exact references, the exact denoiser through the sampler, the networks retrained on continuous
    levels, and the fitted kernels (from the main fits) through the sampler and by direct sampling of their chain."""
    cfg, cell_id, n, seed = args
    torch, dev = _setup(cfg)
    from markov_score import ancestral as an
    from markov_score import unet_ddpm as ud
    from markov_score.energy_kernel import NeuralEnergyKernel
    from markov_score.pair_energy_kernel import PairEnergyKernel
    xp = get_xp()
    chain, grid, w, log_k, log_mu, sigma0, A, Av, val, test, base = common(cfg, cell_id, seed, n, xp)
    info = resolve_cell(find_cell(cell_id))[1]
    L, N = base["n_sites"], cfg["gen_n"]
    ladder = np.geomspace(cfg["gen_tmax"], cfg["gen_tmin"], cfg["gen_steps"] + 1)
    R = chain.sample_batch(rng_for("e55-gen-exact", cell_id, seed), N, L)
    noise = lambda: rng_for("e55-gen-noise", cell_id, seed)  # noqa: E731   the same sampler noise for every arm
    keep = {k: base[k] for k in ("cell", "T", "h", "n_sites", "role", "seed", "n_chains", "test_bundle", "test_draws")}
    rows = []

    def add(arm, G, **extra):
        rows.append({**keep, "arm": arm, "n_gen": N, "gen_steps": cfg["gen_steps"], "gen_tmin": cfg["gen_tmin"],
                     **generation_stats(G, R, chain, info, L), **extra})
        print(f"  [E55 gen] {cell_id} M={n} seed={seed} {arm}: swd {rows[-1]['swd']:.4f} kinks {rows[-1]['coarse_kink_density']:.4f}", flush=True)

    add("exact_ref", R)
    add("exact", chain.sample_batch(rng_for("e55-gen-exact2", cell_id, seed), N, L))          # Monte Carlo noise floor
    add("oracle_sampler", an.generate(_mean_fn(grid, w, log_k, log_mu, xp), N, L, ladder, noise()))
    for arm in ("unet", "window"):
        row, model = _fit_net(cfg, arm, _winner(cfg, cell_id, arm), cell_id, seed, n, dev,
                              t_range=(cfg["gen_tmin"], cfg["gen_tmax"]), return_model=True)
        G = an.generate(lambda X, t: ud.posterior_mean(model, X, t, cfg["eval_chunk"]), N, L, ladder, noise())
        add(f"{arm}_cont", G, risk=row["risk"], risk_levels=row["risk_levels"], checkpoint=row["checkpoint"],
            at_cap=row["at_cap"], hp=row["hp"])
    paths = [str(Path(p_)) for p_ in cfg["gen_kernel_csvs"]] + [str(Path(__file__).resolve().parent / cfg["gen_e33_csv"])]
    for arm, label in (("em_energy", "em_energy"), ("em_pair_quartic", "site_energy")):
        r = _kernel_row(paths, arm, cell_id, n, seed)
        if r is None:
            rows.append({**keep, "arm": label, "failed": 1, "stop_reason": "no fitted kernel found"}); continue
        theta = np.array([float(v) for v in str(r["theta"]).split(";")])
        if arm == "em_energy":
            k = NeuralEnergyKernel.from_theta(theta, int(json.loads(r["hp"])["hidden"]), cfg["half_width"])
        else:
            k = PairEnergyKernel(log_gamma=float(theta[0]), coef=theta[1:], half_width=float(cfg["half_width"]))
        lk = k.log_transition_matrix(grid)
        mu = e33._stationary_mu(grid, w)(k)
        add(label, an.generate(_mean_fn(grid, w, lk, mu, xp), N, L, ladder, noise()), risk=float(r["risk"]))
        add(f"{label}_direct", an.sample_grid_chain(lk, mu, grid, w, N, L, rng_for("e55-gen-direct", cell_id, seed)),
            risk=float(r["risk"]))
    return rows


def part_gen(cfg, out, workers):
    cells = [(cfg, c, n, s) for c in _shard_cells({**cfg, "cells": [c for c in cfg["cells"] if c in cfg["gen_cells"]]})
             for n in cfg["gen_sizes"] for s in cfg["gen_seeds"]]
    if not cells:
        print("  [E55 gen] no generation cell in this shard", flush=True); return
    write_csv(out / "generation.csv", run_cells(cells, _gen_cell, workers, out, "generation.csv",
                                                key=lambda c: f"{c[1]}_M{c[2]}_s{c[3]}"))


PARTS = {"oracles": part_oracles, "timing": part_timing, "screen": part_screen, "winners": part_winners,
         "nets": part_nets, "energy": part_energy, "pair": part_pair, "cost": part_cost, "gen": part_gen}


def main():
    ap = experiment_parser("e55_unet_energy", "DDPM U-Net (standard DSM) vs a neural-energy kernel inside BP, phi^4 field")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.list_parts:
        print("\n".join(PARTS)); return
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    if isinstance(cfg["sizes"], dict):
        cfg["sizes"] = {c: tuple(cfg["sizes"][c]) for c in cfg["cells"]}
    out = ensure_dir(Path(args.output_dir))
    for name, fn in select_parts(PARTS, args.only).items():
        fn(cfg, out, args.workers)
        write_json(out / f"params_{name}.json", {**provenance(cfg), "settings": cfg, "quick": bool(args.quick),
                                                  "workers": args.workers, "bp_device": str(get_xp().__name__)})


if __name__ == "__main__":
    main()
