#!/usr/bin/env python
"""E28 — Q9's second half: does the ranking hold at every noise level?

Jerome, 25 September 2026: "When you do this comparison between neural networks and the
EM-BP, you compare the score at some time, or you average over time?" — "You average over
time." The answer was correct: the estimand in `experiments/protocol.py` is
schedule-pooled. But a pooled scalar can hide a crossover, and the obvious next question is
whether the ranking survives level by level.

THE LEDGER WAS WRONG ABOUT THIS. It claimed "E24 and E25 report per-level curves beside the
pooled scalar". They do not: E24's `level` column is a TIMING level and E25's
information_ladder.csv has no level column at all. A header scan of every frozen output
finds no per-noise-level risk for the arm comparison anywhere. The nearest thing is E14,
whose `t_star` turns out to be ONE value per (rho, L) rather than an independent axis, so
its apparent level effect is a rho effect. Nothing had measured this.

WHAT THIS DOES, AND WHY IT IS ALMOST FREE. E03's `_eval` already accumulates one
(sq_err, sq_ref) pair PER LEVEL and then pools them; the pooled risk it reports is
sqrt(sum num / sum den) over exactly those pairs. So the per-level decomposition is the
individual pairs, and no retraining is needed -- the E03 confirm checkpoints are on the
cluster (384 of them, 27 MB) and are loaded, not refitted.

THE LOAD-BEARING VALIDATION. Because the pooled risk is a deterministic function of the
per-level pairs, E28 must reproduce E03's `confirm.csv` risk for the same
(seed, n_chains, arch, region) to floating-point accuracy. Part `verify` does exactly that
and EXITS NONZERO otherwise. Without it, per-level numbers that look plausible could be a
different estimand, which is the failure mode this project has hit before.

Arms: the six E03 winners (loaded from winners.json + the frozen checkpoints), the E02a
mixture kernel inside exact BP (`em_bp`), the covariance twin (`lmmse`), and `isotropic`.
The reference is exact grid BP, so it has risk 0 by construction and is not an arm.

PREDECLARED (registry E28.json), written before the confirmatory run:
  P1  em_bp is first at EVERY level. If it is not, the headline claim is level-dependent and
      the paper must say at which levels it holds.
  P2  lmmse improves RELATIVE to em_bp as t grows, because the score becomes Gaussian at
      high noise -- the quantitative content of the "learned scores are a low-pass filter of
      a Gaussian" framing Jerome asked us to relax. Falsified if the ratio is flat in t.
  P3  at least one pair of arms CROSSES, so the pooled scalar does hide something. If no
      pair crosses, the pooled number is a sufficient summary and that is worth saying.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from protocol import PROTOCOL  # noqa: E402
from markov_score import theory  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.kernels import MixtureInnovationKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

N_SITES = FROZEN.n_sites
T_SCHEDULE = tuple(FROZEN.t_grid)
REGIONS = {"all": slice(None), "bulk": slice(4, N_SITES - 4), "centre": slice(N_SITES // 2 - 2, N_SITES // 2 + 2)}

SETTINGS = dict(
    seeds=PROTOCOL.seeds, sizes=(128, 2048), components=8,
    # MUST be PROTOCOL's, not a number typed here: E02a and E03 both scored on
    # n_test = 2048, and the first E28 run hardcoded 1024, which made every pooled risk
    # a different estimand and was caught by part `verify` (N-26).
    n_val=PROTOCOL.n_val, n_test=PROTOCOL.n_test,
    grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    em_source="outputs/E02a/capacity.csv",
    winners="outputs/E03/winners.json",
    ckpt_dir=None,                 # E03 confirm's ckpts/; networks skipped if absent
    e03_confirm="outputs/E03/confirm.csv",
    archs=("mlp", "window", "tcn", "bigru", "transformer", "unet"),
    regions=("all",),
)
QUICK = dict(seeds=(0,), sizes=(128,), n_test=64, grid_size=201, archs=("window",))


# E03's own bundle, _sq and pooling, reproduced EXACTLY so the estimand is the same object.
# Imported rather than copied would be better, but e03_baselines' module-level work is heavy
# and the three functions are four lines each; any drift is caught by part `verify`.
def _bundle(prior, log_k, grid, weights, tag, seed, n_chains):
    rng = rng_for(tag, seed)
    A = np.stack([prior.sample(rng, N_SITES) for _ in range(n_chains)])
    out = {}
    for t in T_SCHEDULE:
        al, de = alpha_delta(t)
        X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
        m_ref, _ = grid_bp_batch(grid, weights, log_k, X, al, de)
        out[t] = (X, m_ref)
    return A, out


def _sq(m_hat, m_ref, X, t, sl):
    al, de = alpha_delta(t)
    s_hat = -(X[:, sl] - al * m_hat[:, sl]) / de
    s_ref = -(X[:, sl] - al * m_ref[:, sl]) / de
    return float(((s_hat - s_ref) ** 2).sum()), float((s_ref**2).sum())


def _pool(pairs):
    num = sum(e for e, _ in pairs); den = sum(r for _, r in pairs)
    return float(np.sqrt(num / den)) if den > 0 else float("nan")


def _theta_from_e02(path, seed, n, C):
    df = pd.read_csv(path)
    df = df[(df.arm == "em_bp") & (df.seed == seed) & (df.n_chains == n)
            & (df.components == C) & (df.region == "all")]
    if df.empty:
        raise SystemExit(f"no E02 fit for seed={seed} n={n} C={C} in {path}")
    th = np.array([float(v) for v in df.theta.iloc[0].split(";")])
    return MixtureInnovationKernel(rho=float(th[0]), pi=th[1:1 + C],
                                   mu=th[1 + C:1 + 2 * C], s2=th[1 + 2 * C:1 + 3 * C])


def _mean_fns(cfg, seed, n, grid, weights, sigma0):
    """{arm: (X, t) -> posterior mean}. Networks are LOADED, never refitted."""
    root = Path(__file__).resolve().parents[1]
    fns = {}
    try:
        kb = _theta_from_e02(root / cfg["em_source"], seed, n, cfg["components"])
        lk = kb.log_transition_matrix(grid)
        fns["em_bp"] = lambda X, t, lk=lk: grid_bp_batch(
            grid, weights, lk, X, *alpha_delta(t))[0]
    except SystemExit as e:
        print("  ", e, flush=True)
    fns["lmmse"] = lambda X, t: theory.lmmse_posterior_mean(X, sigma0, t)
    fns["isotropic"] = lambda X, t: (lambda al, de: al / (al**2 + de) * X)(*alpha_delta(t))

    if cfg.get("ckpt_dir"):
        # N-26 (27 Sep), CORRECTED. The first E28 run's `verify` refused the output: the
        # pooled risk disagreed with E03's confirm.csv by up to 2.878e-02 relative. I first
        # diagnosed that as float32 reduction-order drift from reloading 15 Sep checkpoints
        # under a newer torch, on the evidence that the disagreement was symmetric scatter
        # (111 of 192 cells high, 81 low) and largest for bigru and transformer. THAT WAS
        # WRONG. The cause was that this file hardcoded n_test = 1024 while E02a and E03 both
        # scored on 2048, so every pooled risk really was a different estimand -- exactly what
        # the gate said, and nothing to do with the checkpoints. At the correct n_test the
        # em_bp arm pools back to E02a's risk to 4.65e-11.
        #
        # The network arms are therefore kept, and the gate is left to judge them on the right
        # bundle rather than being told in advance what it will find. If a residual remains
        # once n_test is right, THEN it is worth attributing to the reload -- and it will be
        # measured, not assumed.
        import torch
        from markov_score import nn_torch as nt
        win = json.loads((root / cfg["winners"]).read_text())
        for arch in cfg["archs"]:
            ck = Path(cfg["ckpt_dir"]) / f"{seed}_{n}_{arch}.pt"
            if arch not in win or not ck.exists():
                print(f"   no checkpoint for {arch} at seed={seed} n={n}", flush=True)
                continue
            w = win[arch]
            model = nt.build(arch, N_SITES, **w["hp"])
            model.load_state_dict(torch.load(ck, map_location="cpu"))
            den = nt.Denoiser(model, w["mode"])
            fns[f"net:{arch}"] = (lambda d: (lambda X, t: d.posterior_mean(X, t)))(den)
    return fns


def _cell(args):
    cfg, seed, n = args
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    prior = LaplaceAR1(FROZEN.rho)
    log_k = prior.log_transition_matrix(grid)
    sigma0 = theory.chain_covariance(FROZEN.rho, FROZEN.innovation_variance, N_SITES)
    # E03's test bundle, from its own stream, so E28 scores on the same data E03 did.
    _, test = _bundle(prior, log_k, grid, weights, "exp31-test", seed, cfg["n_test"])

    rows = []
    for arm, f in _mean_fns(cfg, seed, n, grid, weights, sigma0).items():
        m_by_t = {t: f(test[t][0], t) for t in T_SCHEDULE}
        for region in cfg["regions"]:
            sl = REGIONS[region]
            pairs = [_sq(m_by_t[t], test[t][1], test[t][0], t, sl) for t in T_SCHEDULE]
            pooled = _pool(pairs)
            for t, (e, r) in zip(T_SCHEDULE, pairs):
                rows.append(dict(seed=seed, n_chains=n, arm=arm, region=region, t=t,
                                 sq_err=e, sq_ref=r,
                                 risk_level=float(np.sqrt(e / r)) if r > 0 else np.nan,
                                 risk_pooled=pooled))
        print(f"  [E28] seed={seed} n={n} {arm:16s} pooled={pooled:.4f}", flush=True)
    return rows


def part_levels(cfg, out, workers):
    cells = [(cfg, s, n) for s in cfg["seeds"] for n in cfg["sizes"]]
    rows = run_cells(cells, _cell, workers, out, "per_level.csv")
    write_csv(out / "per_level.csv", rows)


def part_verify(cfg, out):
    """The pooled risk rebuilt from E28's per-level pairs must equal E03's confirm.csv risk.

    Non-negotiable: if it does not, the per-level numbers are a different estimand and
    nothing downstream of them means what it says. Exits nonzero."""
    root = Path(__file__).resolve().parents[1]
    dest = out / "per_level.csv"
    if not dest.exists():
        raise SystemExit("part `verify` needs per_level.csv; run part `levels` first")
    mine = pd.read_csv(dest)
    e03 = pd.read_csv(root / cfg["e03_confirm"])
    mine["arch"] = mine.arm.str.replace("^net:", "", regex=True)
    nets = mine[mine.arm.str.startswith("net:")]
    if nets.empty:
        # The default configuration carries no network arms (N-26), so there is nothing to
        # check against E03. Verify the arms that ARE present against E02a instead: em_bp's
        # pooled risk is a deterministic function of the same per-level pairs.
        em = mine[mine.arm == "em_bp"]
        if em.empty:
            raise SystemExit("FATAL: neither network nor em_bp rows present; nothing verified")
        j = em.groupby(["seed", "n_chains", "region"]).agg(
            num=("sq_err", "sum"), den=("sq_ref", "sum")).reset_index()
        j["risk_e28"] = np.sqrt(j.num / j.den)
        ref = pd.read_csv(root / cfg["em_source"])
        ref = ref[(ref.arm == "em_bp") & (ref.components == cfg["components"])
                  & (ref.region == "all")]
        ref = ref.groupby(["seed", "n_chains", "region"]).risk.median().reset_index()
        m = j.merge(ref, on=["seed", "n_chains", "region"], how="inner")
        if m.empty:
            raise SystemExit("FATAL: no overlap with E02a capacity.csv — nothing was verified")
        m["rel"] = (m.risk_e28 - m.risk).abs() / m.risk
        worst = float(m.rel.max())
        m.to_csv(out / "verify.csv", index=False)
        print(f"[verify] {len(m)} (seed, n, region) em_bp cells against E02a capacity.csv")
        print(f"[verify] worst relative disagreement in the POOLED risk: {worst:.3e}")
        if worst > 1e-6:
            print(m.nlargest(8, "rel").to_string(), flush=True)
            raise SystemExit(
                f"FATAL: em_bp's per-level pairs do not pool back to E02a's risk "
                f"(worst {worst:.3e}). The decomposition is of a DIFFERENT estimand.")
        print("[verify] OK — the per-level decomposition is E02a's own pooled estimand")
        return
    j = nets.groupby(["seed", "n_chains", "arch", "region"]).agg(
        num=("sq_err", "sum"), den=("sq_ref", "sum")).reset_index()
    j["risk_e28"] = np.sqrt(j.num / j.den)
    ref = e03.groupby(["seed", "n_chains", "arch", "region"]).risk.median().reset_index()
    m = j.merge(ref, on=["seed", "n_chains", "arch", "region"], how="inner")
    if m.empty:
        raise SystemExit("FATAL: no overlap with E03 confirm.csv — nothing was verified")
    m["rel"] = (m.risk_e28 - m.risk).abs() / m.risk
    worst = float(m.rel.max())
    m.to_csv(out / "verify.csv", index=False)
    print(f"[verify] {len(m)} (seed, n, arch, region) cells against E03 confirm.csv")
    print(f"[verify] worst relative disagreement in the POOLED risk: {worst:.3e}")
    if worst > 1e-6:
        print(m.nlargest(8, "rel").to_string(), flush=True)
        raise SystemExit(
            f"FATAL: E28's per-level pairs do not pool back to E03's risk (worst {worst:.3e}). "
            "The per-level decomposition is of a DIFFERENT estimand; do not use it.")
    print("[verify] OK — the per-level decomposition is of E03's own pooled estimand")


def main() -> None:
    parser = experiment_parser("e28_per_level_ranking",
                               "Q9: does the arm ranking hold at every noise level?")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)

    parts = {"levels": ("per-level risk for every arm", lambda c, o: part_levels(c, o, args.workers)),
             "verify": ("pool back to E03's reported risk", part_verify)}
    if args.list_parts:
        print("\n".join(parts)); return
    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir)
    tag = "_".join(selected) if args.only else "all"
    resolved = {"n_sites": N_SITES, "t_schedule": list(T_SCHEDULE), "quick": args.quick,
                "parts": list(selected), "overrides": args.set, **cfg}
    write_json(out / f"params_{tag}.json", {**resolved, **provenance(resolved)})
    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(cfg, out)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
