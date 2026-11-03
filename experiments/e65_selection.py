#!/usr/bin/env python
"""E65 — checkpoint selection without the exact score.

Every learned arm of the paper chooses its checkpoint by the score error on validation data, which needs the exact
score. E65 refits with the frozen recipes and records, at EVERY checkpoint, the oracle criterion (validation error, on
all 32 sites and on the bulk), a practitioner's criterion, and the test error on all 32 sites (never used to select;
it is the per-checkpoint curve E63 puts on a wall-clock axis):

    em_bp, spline, energy   noisy EM (E30's mixture; E56's spline and energy, width 16) on E03's one-view data;
                            criterion: held-out evidence, the mean log p(x) per validation view under BP (no clean data)
    clean_energy            E58's clean-data maximum likelihood (energy_h16); criterion: held-out clean NLL per transition
    E03's six networks      E03's recipe and seed stream; criterion: the validation DSM loss on the clean validation
                            chains at every level with a fixed noise draw (standard practice)

Each arm is selected by its ORIGINAL oracle rule (E30/E56/E58 on all sites, E03's networks on the bulk) and by its
oracle-free criterion; both flags are written on the curve.

Parts
-----
kernels   (seed, n, arm) -> curves_kernels.csv, one row per checkpoint
nets      (seed, n, arch) -> curves_nets.csv, one row per checkpoint

PREDECLARED in experiments/registry/E65.json (P1-P4, D1-D3; amendment of 3 Nov before code) before this file was written.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from protocol import PROTOCOL  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.em import clean_statistics, fit_em  # noqa: E402
from markov_score.kernels import MixtureInnovationKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

import e03_baselines as e03  # noqa: E402
import e56_kernel_families as e56  # noqa: E402
import e58_clean_kernel as e58  # noqa: E402
from e63_compute import NETS  # noqa: E402   E03's winners, hp as tuples so E03's seed stream (str(hp)) is reproduced

N_SITES = FROZEN.n_sites
T_SCHEDULE = tuple(FROZEN.t_grid)
KERNEL_ARMS = ("em_bp", "spline", "energy", "clean_energy")
E30_LADDER = (10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200)
CRITERION = {"em_bp": "heldout_evidence", "spline": "heldout_evidence", "energy": "heldout_evidence",
             "clean_energy": "heldout_clean_nll", **{a: "val_dsm" for a in NETS}}
MAXIMISE = {"heldout_evidence"}                     # every other criterion is minimised

SETTINGS = dict(
    seeds=PROTOCOL.seeds, sizes=(32, 128, 512, 2048), n_val=PROTOCOL.n_val, n_test=PROTOCOL.n_test,
    grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    em_init_rho=PROTOCOL.em_init_rho, em_init_var=PROTOCOL.em_init_var, components=8,
    shift_W=4.0, shift_lr=1e-2, shift_inner=10, energy_lr=1e-2, energy_inner=10, energy_hidden=16, device="cpu",
    kernel_arms=KERNEL_ARMS, em_ladder=E30_LADDER, ladder=e56.LADDER,
    nets=tuple(NETS), net_steps=PROTOCOL.net_steps, net_checkpoints=PROTOCOL.net_checkpoints,
    batch_size=PROTOCOL.batch_size, grad_clip=PROTOCOL.grad_clip, torch_threads=1,
)
QUICK = dict(seeds=(0,), sizes=(32,), n_val=16, n_test=16, grid_size=101, em_ladder=(2, 4), ladder=(2, 4),
             net_checkpoints=(20, 40))


# ---------------------------------------------------------------------------------------- shared scoring

def _a_val(cfg, seed):
    """The clean validation chains behind E03's validation bundle (e03._bundle draws them first; E58's context)."""
    from markov_score.priors import LaplaceAR1
    prior, rng = LaplaceAR1(FROZEN.rho), rng_for("exp31-val", seed)
    return np.stack([prior.sample(rng, N_SITES) for _ in range(cfg["n_val"])])


def _scores(log_k, val, test, grid, w):
    """Validation error on all sites and on the bulk, test error on all sites, held-out evidence per validation view."""
    v_all, v_bulk, t_all, ev = [], [], [], 0.0
    n_views = 0
    for t in T_SCHEDULE:
        al, de = alpha_delta(t)
        X, m_ref = val[t]
        m, _, logz = grid_bp_batch(grid, w, log_k, X, al, de, return_evidence=True)
        v_all.append(e03._sq(m, m_ref, X, t, e03.REGIONS["all"]))
        v_bulk.append(e03._sq(m, m_ref, X, t, e03.REGIONS["bulk"]))
        ev += float(np.sum(logz)); n_views += len(X)
        Xt, mt = test[t]
        mt_hat, _ = grid_bp_batch(grid, w, log_k, Xt, al, de)
        t_all.append(e03._sq(mt_hat, mt, Xt, t, e03.REGIONS["all"]))
    return e03._risk(v_all), e03._risk(v_bulk), e03._risk(t_all), ev / n_views


def _flag(rows, oracle_key, crit):
    """Mark the oracle-selected and the criterion-selected checkpoint (ties: the earliest)."""
    ok = [r for r in rows if np.isfinite(r[oracle_key])]
    best_o = min(ok, key=lambda r: (r[oracle_key], r["checkpoint"]))["checkpoint"] if ok else None
    fc = [r for r in rows if np.isfinite(r["criterion"])]
    if fc:
        sign = -1.0 if crit in MAXIMISE else 1.0
        best_f = min(fc, key=lambda r: (sign * r["criterion"], r["checkpoint"]))["checkpoint"]
    else:
        best_f = None
    for r in rows:
        r["sel_oracle"] = int(r["checkpoint"] == best_o)
        r["sel_free"] = int(r["checkpoint"] == best_f)
    return rows


# ---------------------------------------------------------------------------------------- kernel arms

def _kernel_cell(args):
    cfg, seed, n, arm = args
    import torch
    torch.set_num_threads(int(cfg["torch_threads"]))
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    A, val, test, base = e03._cell_common(cfg, seed, n)
    t0 = time.perf_counter()
    try:
        if arm == "clean_energy":
            stats, held = clean_statistics(grid, A), clean_statistics(grid, _a_val(cfg, seed))
            k0 = e58.init_kernel(cfg, "laplace", f"energy_h{cfg['energy_hidden']}", seed, n, grid)
            saved, sec = e58.fit_clean_ladder(k0, grid, stats, set(cfg["ladder"]))
            walls = {it: it * sec for it in saved}
            crit = {it: e58.clean_nll(saved[it], grid, held) for it in saved}
        else:
            groups = [(X, *alpha_delta(t)) for X, t in e03._one_view_groups(A, seed)]
            if arm == "em_bp":
                k0 = MixtureInnovationKernel.init(cfg["components"], rho=0.3, var=0.8, rng=rng_for("e30-em-init", seed, n))
                ladder = cfg["em_ladder"]
            else:
                k0 = e56._init_kernel(cfg, arm, seed, n, hidden=cfg["energy_hidden"])
                ladder = cfg["ladder"]
            _, trace, saved = fit_em(k0, grid, w, groups, n_iters=max(ladder), tol=FROZEN.em_loglik_tol,
                                     checkpoints=set(ladder))
            cum = np.concatenate([[0.0], np.cumsum(trace.seconds)])
            walls = {it: float(cum[min(it, len(cum) - 1)]) for it in saved}
            crit = None
    except FloatingPointError as exc:                      # a failed fit is a row and counts against (E33 amendment 3)
        print(f"  [E65] seed={seed} n={n:>5} {arm:12s} FAILED: {exc}", flush=True)
        return [{**base, "arm": arm, "kind": "kernel", "failed": 1, "stop_reason": f"FloatingPointError: {exc}"}]
    rows = []
    for it in sorted(saved):
        try:
            v_all, v_bulk, t_all, ev = _scores(saved[it].log_transition_matrix(grid), val, test, grid, w)
        except FloatingPointError:
            v_all = v_bulk = t_all = ev = np.nan
        c = crit[it] if crit is not None else ev
        rows.append({**base, "arm": arm, "kind": "kernel", "checkpoint": int(it), "val_all": v_all, "val_bulk": v_bulk,
                     "test_all": t_all, "criterion_name": CRITERION[arm], "criterion": c, "heldout_evidence": ev,
                     "wall_cum": walls[it], "n_params": int(np.asarray(saved[it].theta).size), "failed": 0})
    rows = _flag(rows, "val_all", CRITERION[arm])            # E30, E56 and E58 selected on all sites
    o = next(r for r in rows if r["sel_oracle"]); f = next(r for r in rows if r["sel_free"])
    print(f"  [E65] seed={seed} n={n:>5} {arm:12s} oracle ck={o['checkpoint']} test={o['test_all']:.4f} | "
          f"free ck={f['checkpoint']} test={f['test_all']:.4f}  ({time.perf_counter() - t0:.0f} s)", flush=True)
    return rows


# ---------------------------------------------------------------------------------------- networks

def _val_dsm(den, a_val, seed):
    """Mean eps-prediction DSM loss on the clean validation chains, each at every level, one fixed noise draw."""
    import torch
    rng = rng_for("e65-valdsm", seed)
    tot, n = 0.0, 0
    with torch.no_grad():
        for t in T_SCHEDULE:
            al, de = alpha_delta(t)
            z = rng.standard_normal(a_val.shape)
            x = torch.as_tensor(al * a_val + np.sqrt(de) * z, dtype=torch.float32)
            out = den.model(x, torch.full((len(a_val),), float(t)))
            target = torch.as_tensor(z if den.parameterization == "eps" else a_val, dtype=torch.float32)
            tot += float(((out - target) ** 2).mean(dim=1).sum()); n += len(a_val)
    return tot / n


def _net_cell(args):
    cfg, seed, n, arch = args
    import torch
    from markov_score import nn_torch as nt
    torch.set_num_threads(int(cfg["torch_threads"]))
    hp, lr, mode = NETS[arch]
    A, val, test, base = e03._cell_common(cfg, seed, n)
    a_val = _a_val(cfg, seed)
    ck = tuple(int(c) for c in cfg["net_checkpoints"])
    model = nt.build(arch, N_SITES, **hp)
    tcfg = nt.TrainConfig(steps=max(ck), batch_size=cfg["batch_size"], lr=lr, grad_clip=cfg["grad_clip"],
                          parameterization=mode, loss="dsm", checkpoints=ck, t_levels=T_SCHEDULE,
                          seed=int(rng_for("e03-net-seed", arch, str(hp), lr, mode, seed, 0).integers(0, 2**31 - 1)),
                          hutchinson=(arch != "window"))
    t0 = time.time()
    saved = nt.train(model, A, tcfg)
    wall = time.time() - t0
    rows = []
    for step in sorted(saved):
        den = nt.Denoiser(nt.load_checkpoint(model, saved[step]), mode)
        rows.append({**base, "arm": arch, "kind": "net", "checkpoint": step,
                     "val_all": e03._risk(e03._eval(den, val, "all")), "val_bulk": e03._risk(e03._eval(den, val, "bulk")),
                     "test_all": e03._risk(e03._eval(den, test, "all")), "criterion_name": "val_dsm",
                     "criterion": _val_dsm(den, a_val, seed), "heldout_evidence": np.nan,
                     "wall_cum": wall * step / max(ck), "n_params": nt.n_params(model), "failed": 0})
    rows = _flag(rows, "val_bulk", "val_dsm")                 # E03 selected on the bulk
    o = next(r for r in rows if r["sel_oracle"]); f = next(r for r in rows if r["sel_free"])
    print(f"  [E65 net] seed={seed} n={n:>5} {arch:11s} oracle ck={o['checkpoint']} test={o['test_all']:.4f} | "
          f"free ck={f['checkpoint']} test={f['test_all']:.4f}", flush=True)
    return rows


# ---------------------------------------------------------------------------------------- parts

def _key(cell):
    _, seed, n, arm = cell
    return f"s{seed}_n{n}_{arm}"


def part_kernels(cfg, out, workers):
    cells = [(cfg, s, n, a) for s in cfg["seeds"] for n in cfg["sizes"] for a in cfg["kernel_arms"]]
    write_csv(Path(out) / "curves_kernels.csv", run_cells(cells, _kernel_cell, workers, out, "curves_kernels.csv", key=_key))


def part_nets(cfg, out, workers):
    cells = [(cfg, s, n, a) for s in cfg["seeds"] for n in cfg["sizes"] for a in cfg["nets"]]
    write_csv(Path(out) / "curves_nets.csv", run_cells(cells, _net_cell, workers, out, "curves_nets.csv", key=_key))


def main() -> None:
    parser = experiment_parser("e65_selection", "checkpoint selection without the exact score")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    w = args.workers
    parts = {"kernels": ("noisy EM and clean ML, every checkpoint scored", lambda c, o: part_kernels(c, o, w)),
             "nets": ("E03's six networks, every checkpoint scored", lambda c, o: part_nets(c, o, w))}
    if args.list_parts:
        print("\n".join(parts)); return
    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir)
    tag = "_".join(selected) if args.only else "all"
    resolved = {"n_sites": N_SITES, "t_schedule": list(T_SCHEDULE), "quick": args.quick, "parts": list(selected),
                "overrides": args.set, **cfg}
    write_json(out / f"params_{tag}.json", {**resolved, **provenance(resolved)})
    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(cfg, out)


if __name__ == "__main__":
    main()
