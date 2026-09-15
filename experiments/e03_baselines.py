#!/usr/bin/env python
"""E3 — strong learned baselines under one fair recipe, and the information-matched arm.

Hypotheses (docs/RESEARCH_PLAN.md H3, H10). The archival screen (exp_31) gave the
long-range architectures one noise level per gradient step and one run, the window
head twelve levels and eight restarts. Here every architecture trains under the
recipe of `markov_score.nn_torch.TrainConfig` (per-example level from the twelve,
batch 64, Adam, grad-clip 1, checkpoint ladder to `net_steps` of ONE run, both
parameterisations).

Parts
-----
screen    dev seeds (two); sizes (32, 2048); every architecture × its hyper-grid × lr grid ×
          parameterisation; ONE run of `screen_steps` (16k) per configuration, checkpoint
          selected on the validation bundle (bulk); selection statistic = pooled bulk risk
          over dev seeds and sizes (as exp_31). Amended 15 Sep from (4 dev seeds, 3
          restarts, 32k): the measured single-thread cost (1–16 min per 32k-step run after
          the gather-matmul convolutions of `nn_torch.FastConv1d`) put the original screen
          at ~136 core-hours, beyond the laptop budget; the screen ranks configurations,
          it produces no headline number, and every configuration still gets four
          (seed, size) fits. The confirmatory protocol is unchanged.
confirm   confirmatory seeds; sizes (32, 128, 512, 2048); the winner of each
          architecture family from the screen, checkpoint selected per cell on its
          validation bundle; scored on the test bundle in every region. Bundles are the
          exp_31 bundles (same tags), so EM-BP rows from E02 and the legacy window rows
          from exp_31 can be joined on the bundle hash.
matched   the information-matched arm: the best two architectures trained by implicit
          score matching from the SAME one-view noisy bundle EM-BP sees (no clean
          chain, one corruption per chain, split across the twelve levels), checkpoint
          selected on the ISM objective of a held-out noisy bundle (no oracle), then
          scored against the true-kernel reference for reporting only; plus a
          one-corruption DSM arm (clean targets, one fixed noise draw per chain).
length    the length-agnostic winners fitted at L = 32 evaluated at L in (64, 128, 256)
          against true-kernel BP at those lengths; EM-BP (from E02) evaluated the same
          way in tools/.

Rows record architecture, hyperparameters, parameter count, steps, selected
checkpoint, wall-clock, region, sq_err / sq_ref, risk, bundle hashes. Provenance in
params_<part>.json. `--workers N` runs cells in N processes (torch single-threaded).
"""

from __future__ import annotations

import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

N_SITES = FROZEN.n_sites
T_SCHEDULE = tuple(FROZEN.t_grid)
BULK = slice(4, N_SITES - 4)
REGIONS = {"all": slice(0, N_SITES), "bulk": BULK, "centre": slice(N_SITES // 2, N_SITES // 2 + 1)}

GRIDS = {
    "mlp": [dict(hidden=(128, 128)), dict(hidden=(256, 256))],
    "window": [dict(radius=r, width=w) for r in (2, 4, 8) for w in (64, 128)],
    "tcn": [dict(channels=c, dilations=(1, 2, 4, 8)) for c in (32, 64)],
    "bigru": [dict(hidden=h) for h in (32, 64)],
    "transformer": [dict(d=d, heads=4, layers=l) for d, l in ((32, 2), (64, 2), (64, 4))],
    "unet": [dict(channels=c) for c in (16, 32)],
}

SETTINGS = dict(
    sizes=(32, 128, 512, 2048), screen_sizes=(32, 2048),
    seeds=tuple(range(16)), dev_seeds=(10_000, 10_001),
    n_val=1024, n_test=2048,
    grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    net_steps=32_000, net_checkpoints=(500, 1000, 2000, 4000, 8000, 16000, 32000),
    lrs=(1e-3, 3e-3), parameterizations=("eps", "x0"), batch_size=64, grad_clip=1.0,
    screen_restarts=1, screen_steps=16_000, archs=tuple(GRIDS),
    winners=None,                       # confirm: dict arch -> hp, or "auto:<screen.csv>"
    matched_archs=("window", "transformer"), matched_steps=32_000,
    lengths=(64, 128, 256),
    torch_threads=1,
)
QUICK = dict(sizes=(32,), screen_sizes=(32,), seeds=(10_000,), dev_seeds=(10_000,), n_val=64, n_test=64, grid_size=201,
             net_steps=200, net_checkpoints=(100, 200), lrs=(1e-3,), parameterizations=("eps",), screen_restarts=1, screen_steps=200,
             archs=("window", "transformer"), matched_steps=200, lengths=(64,))


# ----------------------------------------------------------------------------- data


def _bundle(prior, log_k, grid, weights, tag, seed, n_chains, t_values=T_SCHEDULE, n_sites=N_SITES):
    rng = rng_for(tag, seed)
    A = np.stack([prior.sample(rng, n_sites) for _ in range(n_chains)])
    out = {}
    for t in t_values:
        al, de = alpha_delta(t)
        X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
        m_ref, _ = grid_bp_batch(grid, weights, log_k, X, al, de)
        out[t] = (X, m_ref)
    return A, out


def _bundle_hash(bundle):
    import hashlib
    h = hashlib.sha256()
    for t in sorted(bundle):
        X, m = bundle[t]
        h.update(np.ascontiguousarray(X, dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(m, dtype=np.float64).tobytes())
    return h.hexdigest()[:12]


def _sq(m_hat, m_ref, X, t, sl):
    al, de = alpha_delta(t)
    s_hat = -(X[:, sl] - al * m_hat[:, sl]) / de
    s_ref = -(X[:, sl] - al * m_ref[:, sl]) / de
    return float(((s_hat - s_ref) ** 2).sum()), float((s_ref**2).sum())


def _risk(pairs):
    num = sum(e for e, _ in pairs); den = sum(r for _, r in pairs)
    return float(np.sqrt(num / den)) if den > 0 else float("nan")


def _eval(den, bundle, region):
    pairs = []
    for t in T_SCHEDULE:
        X, m_ref = bundle[t]
        pairs.append(_sq(den.posterior_mean(X, t), m_ref, X, t, REGIONS[region]))
    return pairs


def _one_view_groups(A, seed):
    """EM-BP's interface: one corruption per chain, chains split across the twelve levels."""
    idx = np.array_split(rng_for("exp31-em-split", seed).permutation(len(A)), len(T_SCHEDULE))
    noise = rng_for("exp31-em-noise", seed)
    out = []
    for t, ix in zip(T_SCHEDULE, idx):
        al, de = alpha_delta(t)
        sub = A[ix]
        out.append((al * sub + np.sqrt(de) * noise.standard_normal(sub.shape), t))
    return out


# ----------------------------------------------------------------------------- arms


def _train_select(cfg, arch, hp, lr, mode, A, val, seed, restart, loss="dsm", groups=None, select_on="bp"):
    import torch
    from markov_score import nn_torch as nt
    torch.set_num_threads(cfg["torch_threads"])
    ck = tuple(int(c) for c in cfg["net_checkpoints"])
    model = nt.build(arch, N_SITES, **hp)
    tcfg = nt.TrainConfig(steps=max(ck), batch_size=cfg["batch_size"], lr=lr, grad_clip=cfg["grad_clip"],
                          parameterization=mode, loss=loss, checkpoints=ck, t_levels=T_SCHEDULE,
                          seed=int(rng_for("e03-net-seed", arch, str(hp), lr, mode, seed, restart).integers(0, 2**31 - 1)),
                          hutchinson=(arch != "window"))
    t0 = time.time()
    saved = nt.train(model, A, tcfg, X_noisy_groups=groups)
    wall = time.time() - t0
    best, best_val, curve = None, np.inf, {}
    for step in sorted(saved):
        den = nt.Denoiser(nt.load_checkpoint(model, saved[step]), mode)
        if select_on == "bp":
            v = _risk(_eval(den, val, "bulk"))
        else:  # ISM objective on the held-out noisy bundle (no oracle)
            v = _ism_objective(den, val, tcfg)
        curve[step] = v
        if v < best_val:
            best, best_val = step, v
    den = nt.Denoiser(nt.load_checkpoint(model, saved[best]), mode)
    return den, dict(checkpoint=best, at_cap=best == max(saved), val_selected=best_val, wall=wall,
                     n_params=nt.n_params(model), val_curve=";".join(f"{k}:{v:.5f}" for k, v in sorted(curve.items())))


def _ism_objective(den, groups, tcfg):
    """Mean implicit-score-matching objective over held-out noisy (X, t) groups, exact trace."""
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
        obj = (0.5 * (s**2).sum(dim=1) + div).detach()
        tot += float(obj.sum()); n += x.shape[0]
    return tot / n


# ----------------------------------------------------------------------------- parts


def _cell_common(cfg, seed, n):
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    prior = LaplaceAR1(FROZEN.rho)
    log_k = prior.log_transition_matrix(grid)
    rng_train = rng_for("exp31-train", seed, n)
    A = np.stack([prior.sample(rng_train, N_SITES) for _ in range(n)])
    _, val = _bundle(prior, log_k, grid, weights, "exp31-val", seed, cfg["n_val"])
    _, test = _bundle(prior, log_k, grid, weights, "exp31-test", seed, cfg["n_test"])
    base = dict(seed=seed, n_chains=n, val_bundle=_bundle_hash(val), test_bundle=_bundle_hash(test))
    return A, val, test, base


def _screen_cell(args):
    cfg, seed, n, arch, hp, lr, mode = args
    A, val, test, base = _cell_common(cfg, seed, n)
    # the screen trains to `screen_steps` (checkpoint ladder truncated there); confirm trains to `net_steps`
    cfg_s = dict(cfg, net_checkpoints=tuple(c for c in cfg["net_checkpoints"] if c <= cfg["screen_steps"]) or (cfg["screen_steps"],))
    rows = []
    for restart in range(cfg["screen_restarts"]):
        den, info = _train_select(cfg_s, arch, hp, lr, mode, A, val, seed, restart)
        for region in REGIONS:
            pairs = _eval(den, test, region)
            rows.append({**base, "arch": arch, "hp": str(hp), "lr": lr, "mode": mode, "restart": restart, "region": region,
                         "sq_err": sum(e for e, _ in pairs), "sq_ref": sum(r for _, r in pairs), "risk": _risk(pairs), **info})
    print(f"  [E03 screen] seed={seed} n={n} {arch} {hp} lr={lr} {mode}: risk(all)={rows[0]['risk']:.4f} ckpt={rows[0]['checkpoint']} {rows[0]['wall']:.0f}s", flush=True)
    return rows


def part_screen(cfg, out, workers):
    cells = [(cfg, s, n, arch, hp, lr, mode)
             for s in cfg["dev_seeds"] for n in cfg["screen_sizes"]
             for arch in cfg["archs"] for hp in GRIDS[arch] for lr in cfg["lrs"] for mode in cfg["parameterizations"]]
    print(f"  screen: {len(cells)} cells", flush=True)
    rows = _run(cells, _screen_cell, workers, out, "screening.csv")
    write_csv(out / "screening.csv", rows)


def winners_from_screening(path) -> dict:
    """Per architecture family, the (hp, lr, mode) with the lowest pooled bulk risk over dev seeds, sizes, restarts."""
    import pandas as pd
    df = pd.read_csv(path)
    df = df[df.region == "bulk"]
    g = df.groupby(["arch", "hp", "lr", "mode"])[["sq_err", "sq_ref"]].sum()
    g["risk"] = np.sqrt(g.sq_err / g.sq_ref)
    out = {}
    for arch, sub in g.reset_index().groupby("arch"):
        best = sub.sort_values("risk").iloc[0]
        out[arch] = dict(hp=eval(best.hp), lr=float(best.lr), mode=best["mode"], screen_risk=float(best.risk))
    return out


def _confirm_cell(args):
    cfg, seed, n, arch, w = args
    A, val, test, base = _cell_common(cfg, seed, n)
    den, info = _train_select(cfg, arch, w["hp"], w["lr"], w["mode"], A, val, seed, 0)
    rows = []
    for region in REGIONS:
        pairs = _eval(den, test, region)
        rows.append({**base, "arch": arch, "hp": str(w["hp"]), "lr": w["lr"], "mode": w["mode"], "region": region,
                     "sq_err": sum(e for e, _ in pairs), "sq_ref": sum(r for _, r in pairs), "risk": _risk(pairs), **info})
    print(f"  [E03 confirm] seed={seed} n={n} {arch}: risk(all)={rows[0]['risk']:.4f} ckpt={rows[0]['checkpoint']}", flush=True)
    return rows


def part_confirm(cfg, out, workers):
    w = cfg["winners"]
    if isinstance(w, str) and w.startswith("auto:"):
        w = winners_from_screening(w[5:])
    if not w:
        raise SystemExit("confirm needs --set winners=\"'auto:<screening.csv>'\"")
    write_json(out / "winners.json", w)
    cells = [(cfg, s, n, arch, ww) for s in cfg["seeds"] for n in cfg["sizes"] for arch, ww in w.items()]
    rows = _run(cells, _confirm_cell, workers, out, "confirm.csv")
    write_csv(out / "confirm.csv", rows)


def _matched_cell(args):
    cfg, seed, n, arch, w = args
    A, val, test, base = _cell_common(cfg, seed, n)
    groups = _one_view_groups(A, seed)
    # held-out noisy bundle for ISM selection: the validation chains, one view each
    rng_v = rng_for("exp31-val", seed)
    Av = np.stack([LaplaceAR1(FROZEN.rho).sample(rng_v, N_SITES) for _ in range(cfg["n_val"])])
    val_groups = _one_view_groups(Av, seed + 7_000_000)
    rows = []
    cfg_m = dict(cfg, net_checkpoints=tuple(c for c in cfg["net_checkpoints"] if c <= cfg["matched_steps"]) or (cfg["matched_steps"],))
    den, info = _train_select(cfg_m, arch, w["hp"], w["lr"], w["mode"], None, val_groups, seed, 0, loss="ism", groups=groups, select_on="ism")
    for region in REGIONS:
        pairs = _eval(den, test, region)
        rows.append({**base, "arm": "ism_one_view", "arch": arch, "hp": str(w["hp"]), "lr": w["lr"], "mode": w["mode"], "region": region,
                     "sq_err": sum(e for e, _ in pairs), "sq_ref": sum(r for _, r in pairs), "risk": _risk(pairs), **info})
    # one-corruption DSM: clean targets, but a single fixed noise draw per chain (no redraw)
    den2, info2 = _train_select_fixed_corruption(cfg_m, arch, w, A, groups, val, seed)
    for region in REGIONS:
        pairs = _eval(den2, test, region)
        rows.append({**base, "arm": "dsm_one_corruption", "arch": arch, "hp": str(w["hp"]), "lr": w["lr"], "mode": w["mode"], "region": region,
                     "sq_err": sum(e for e, _ in pairs), "sq_ref": sum(r for _, r in pairs), "risk": _risk(pairs), **info2})
    print(f"  [E03 matched] seed={seed} n={n} {arch}: ism risk(all)={rows[0]['risk']:.4f}  one-corruption dsm {rows[3]['risk']:.4f}", flush=True)
    return rows


def _train_select_fixed_corruption(cfg, arch, w, A, groups, val, seed):
    """DSM with clean targets but exactly one (x, t) view per chain, never redrawn."""
    import torch
    from markov_score import nn_torch as nt
    torch.set_num_threads(cfg["torch_threads"])
    ck = tuple(int(c) for c in cfg["net_checkpoints"])
    model = nt.build(arch, N_SITES, **w["hp"])
    # pair each noisy view with its clean chain
    idx = np.array_split(rng_for("exp31-em-split", seed).permutation(len(A)), len(T_SCHEDULE))
    Xs = np.concatenate([X for X, _ in groups]); Ts = np.concatenate([np.full(len(X), t) for X, t in groups])
    As = np.concatenate([A[ix] for ix in idx])
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=w["lr"])
    saved = {}
    t0 = time.time()
    for step in range(1, max(ck) + 1):
        b = rng.integers(0, len(Xs), size=min(cfg["batch_size"], len(Xs)))
        x = torch.as_tensor(Xs[b], dtype=torch.float32); a = torch.as_tensor(As[b], dtype=torch.float32)
        t = torch.as_tensor(Ts[b], dtype=torch.float32)
        al = torch.exp(-t)[:, None]; de = 1 - torch.exp(-2 * t)[:, None]
        z = (x - al * a) / torch.sqrt(de)
        out = model(x, t)
        target = z if w["mode"] == "eps" else a
        loss = ((out - target) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]); opt.step()
        if step in ck:
            saved[step] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    wall = time.time() - t0
    best, best_val = None, np.inf
    for step in sorted(saved):
        den = nt.Denoiser(nt.load_checkpoint(model, saved[step]), w["mode"])
        v = _risk(_eval(den, val, "bulk"))
        if v < best_val:
            best, best_val = step, v
    return nt.Denoiser(nt.load_checkpoint(model, saved[best]), w["mode"]), dict(checkpoint=best, at_cap=best == max(saved), val_selected=best_val, wall=wall, n_params=nt.n_params(model), val_curve="")


def part_matched(cfg, out, workers):
    w = cfg["winners"]
    if isinstance(w, str) and w.startswith("auto:"):
        w = winners_from_screening(w[5:])
    cells = [(cfg, s, n, arch, w[arch]) for s in cfg["seeds"] for n in cfg["sizes"] for arch in cfg["matched_archs"]]
    rows = _run(cells, _matched_cell, workers, out, "matched.csv")
    write_csv(out / "matched.csv", rows)


def _length_cell(args):
    cfg, seed, n, arch, w = args
    A, val, test, base = _cell_common(cfg, seed, n)
    den, info = _train_select(cfg, arch, w["hp"], w["lr"], w["mode"], A, val, seed, 0)
    if not den.model.length_agnostic:
        return []
    rows = []
    prior = LaplaceAR1(FROZEN.rho)
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    log_k = prior.log_transition_matrix(grid)
    for Ltest in (N_SITES, *cfg["lengths"]):
        _, tb = _bundle(prior, log_k, grid, weights, ("e03-len-test", Ltest), seed, max(64, cfg["n_test"] * N_SITES // Ltest), n_sites=Ltest)
        pairs = []
        for t in T_SCHEDULE:
            X, m_ref = tb[t]
            pairs.append(_sq(den.posterior_mean(X, t), m_ref, X, t, slice(0, Ltest)))
        rows.append({**base, "arch": arch, "hp": str(w["hp"]), "L_test": Ltest, "risk": _risk(pairs),
                     "sq_err": sum(e for e, _ in pairs), "sq_ref": sum(r for _, r in pairs), **info})
    print(f"  [E03 length] seed={seed} n={n} {arch}: " + " ".join(f"L={r['L_test']}:{r['risk']:.4f}" for r in rows), flush=True)
    return rows


def part_length(cfg, out, workers):
    w = cfg["winners"]
    if isinstance(w, str) and w.startswith("auto:"):
        w = winners_from_screening(w[5:])
    cells = [(cfg, s, n, arch, ww) for s in cfg["seeds"] for n in cfg["sizes"] for arch, ww in w.items()]
    rows = _run(cells, _length_cell, workers, out, "length.csv")
    write_csv(out / "length.csv", rows)


def _run(cells, fn, workers, out=None, csv_name=None):
    """Run cells serially or in a spawn pool; each cell's rows are written by the worker to
    out/cells/<csv stem>/<key>.json and the CSV is rebuilt after every completion
    (experiments/common.py: run_cells; resumable, no result travels through the pool pipe)."""
    return run_cells(cells, fn, workers, out, csv_name)


PARTS = {"screen": part_screen, "confirm": part_confirm, "matched": part_matched, "length": part_length}


def main():
    ap = experiment_parser("e03_baselines", "learned baselines under one recipe; information-matched arm; length generalisation")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.list_parts:
        print("\n".join(PARTS)); return
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    os.environ.setdefault("OMP_NUM_THREADS", str(cfg["torch_threads"]))
    out = ensure_dir(Path(args.output_dir))
    for name, fn in select_parts(PARTS, args.only).items():
        print(f"[e03] part {name}, workers={args.workers}", flush=True)
        fn(cfg, out, args.workers)
        write_json(out / f"params_{name}.json", {**provenance(), "settings": {k: (v if not isinstance(v, dict) else str(v)) for k, v in cfg.items()},
                                                   "grids": {k: [str(h) for h in v] for k, v in GRIDS.items()},
                                                   "quick": bool(args.quick), "workers": args.workers, "frozen": FROZEN.as_dict()})


if __name__ == "__main__":
    main()
