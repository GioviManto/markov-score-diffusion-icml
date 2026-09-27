#!/usr/bin/env python
"""E29 — the experiment that could refute our own headline: score nets AT the kernel's
parameter count.

Jerome, 25 September 2026: "Exploit the structure by learning only one small neural network
that will learn a transition kernel ... instead of a huge neural network, like an attention,
that needs to understand that." The user has ranked the paper's claims: structure and sample
efficiency lead, PARAMETER EFFICIENCY supports, and the training-cost claim is stated
narrowly. This experiment exists because the supporting claim, as it stands, compares 1,516
kernel parameters against a 135,825-parameter transformer, and the first question any referee
asks is what a 1,516-parameter TRANSFORMER does. Nobody has run that.

It is therefore designed to be refutable: if any architecture at ~1,516 parameters reaches
EM-BP's risk, the parameter-efficiency claim is dead and the paper says so.

DESIGN. A capacity LADDER per architecture rather than a single matched point, because a
single point invites the objection that the chosen configuration was unlucky. Measured
parameter counts at L = 32 (markov_score.nn_torch.n_params, not estimated):

    window       r=1 w=4       81      mlp    (8,8)      696     bigru  h=4      329
                 r=2 w=8      209             (16,16)  1,488            h=8    1,105
                 r=2 w=32   1,601             (32,32)  3,456            h=16   4,001
                 r=4 w=64   5,505             (128,128) 26,016          h=64  59,009
    transformer  d=8  h2 l1    843      tcn    c=4        341     unet   c=4      933
                 d=16 h2 l1  2,579             c=8      1,193            c=8    3,529
                 d=32 h4 l2 18,057             c=16     4,433            c=16  13,713
                 d=64 h4 l4 135,825            c=64    66,881            c=32  54,049

Every architecture has a rung that brackets 1,516, and the ladders span three orders of
magnitude, so the deliverable is a CURVE of risk against parameter count with the kernel's
1,516 marked on it -- not a single contrast.

Everything else is E03's protocol, imported rather than reimplemented: the same training
bundles (`exp31-train`), the same validation and test bundles, the same checkpoint ladder and
validation selection, the same pooled estimand. Risk is recorded per level as well as pooled,
because E28 found that the pooled scalar hides a crossover at t ~ 0.5.

PREDECLARED (registry E29.json), before the confirmatory run:
  P1  no architecture at <= 2,600 parameters reaches em_bp's pooled risk at any M. FALSIFIED
      by a single counterexample, and the counterexample kills the parameter-efficiency claim.
  P2  the nets' risk falls monotonically with parameter count at fixed M. If it does not, the
      capacity ladder is confounded with optimisation difficulty and the comparison at 1,516
      is not a capacity statement.
  P3  the gap at matched parameter count is LARGER at small M than at large M, because the
      claim being supported is about structure buying sample efficiency, not about capacity.
  P4  a net needs at least an order of magnitude more parameters than the kernel to match it.
      This is the quantitative form of the supporting claim and the number the paper would
      quote.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from protocol import PROTOCOL  # noqa: E402
from markov_score.utils import ensure_dir, write_csv, write_json  # noqa: E402

# E03 is imported, not copied: same bundles, same trainer, same selection, same estimand.
import e03_baselines as e03  # noqa: E402

N_SITES = FROZEN.n_sites
T_SCHEDULE = tuple(FROZEN.t_grid)

# Ladders chosen so every architecture has a rung bracketing the MDN kernel's 1,516 params.
LADDERS = {
    "window": [dict(radius=1, width=4), dict(radius=2, width=8), dict(radius=2, width=32),
               dict(radius=4, width=64), dict(radius=4, width=128)],
    "mlp": [dict(hidden=(8, 8)), dict(hidden=(16, 16)), dict(hidden=(32, 32)),
            dict(hidden=(128, 128)), dict(hidden=(256, 256))],
    "bigru": [dict(hidden=4), dict(hidden=8), dict(hidden=16), dict(hidden=32), dict(hidden=64)],
    "transformer": [dict(d=8, heads=2, layers=1), dict(d=16, heads=2, layers=1),
                    dict(d=16, heads=4, layers=2), dict(d=32, heads=4, layers=2),
                    dict(d=64, heads=4, layers=4)],
    "tcn": [dict(channels=c, dilations=(1, 2, 4, 8)) for c in (4, 8, 16, 64)],
    "unet": [dict(channels=c) for c in (4, 8, 16, 32)],
}

SETTINGS = dict(
    seeds=PROTOCOL.seeds, sizes=(32, 128, 512, 2048),
    n_val=PROTOCOL.n_val, n_test=PROTOCOL.n_test,
    grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    net_steps=PROTOCOL.net_steps, net_checkpoints=PROTOCOL.net_checkpoints,
    batch_size=PROTOCOL.batch_size, grad_clip=PROTOCOL.grad_clip, torch_threads=1,
    archs=tuple(LADDERS),
    # One lr and one parameterisation: E03 already screened those, and re-screening them here
    # would make the ladder a search rather than a capacity sweep. Taken from E03's winners.
    winners="outputs/E03/winners.json",
    kernel_params=1516,          # the MDN arm of E25, for the marked line on the curve
)
QUICK = dict(seeds=(10_000,), sizes=(32,), n_val=64, n_test=64, grid_size=201,
             net_steps=200, net_checkpoints=(100, 200), archs=("window",))


def _cell(args):
    cfg, seed, n, arch, hp = args
    root = Path(__file__).resolve().parents[1]
    import json
    win = json.loads((root / cfg["winners"]).read_text()) if Path(root / cfg["winners"]).exists() else {}
    lr = win.get(arch, {}).get("lr", 1e-3)
    mode = win.get(arch, {}).get("mode", "eps")

    A, val, test, base = e03._cell_common(cfg, seed, n)
    den, info = e03._train_select(cfg, arch, hp, lr, mode, A, val, seed, 0)

    rows = []
    pairs = [e03._sq(den.posterior_mean(test[t][0], t), test[t][1], test[t][0], t,
                     e03.REGIONS["all"]) for t in T_SCHEDULE]
    pooled = e03._risk(pairs)
    for t, (e, r) in zip(T_SCHEDULE, pairs):
        rows.append({**base, "arch": arch, "hp": str(hp), "lr": lr, "mode": mode,
                     "n_params": info["n_params"], "t": t, "sq_err": e, "sq_ref": r,
                     "risk_level": float(np.sqrt(e / r)) if r > 0 else np.nan,
                     "risk_pooled": pooled, "checkpoint": info["checkpoint"],
                     "at_cap": info["at_cap"], "wall": info["wall"]})
    print(f"  [E29] seed={seed} n={n:>5} {arch:12s} {str(hp):34s} "
          f"params={info['n_params']:>7,} pooled={pooled:.4f}", flush=True)
    return rows


def part_ladder(cfg, out, workers):
    cells = [(cfg, s, n, a, hp) for s in cfg["seeds"] for n in cfg["sizes"]
             for a in cfg["archs"] for hp in LADDERS[a]]
    rows = run_cells(cells, _cell, workers, out, "ladder.csv")
    write_csv(out / "ladder.csv", rows)


def main() -> None:
    parser = experiment_parser("e29_matched_parameters",
                               "Score nets AT the kernel's parameter count: the refutable arm.")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)

    parts = {"ladder": ("risk against parameter count, per architecture",
                        lambda c, o: part_ladder(c, o, args.workers))}
    if args.list_parts:
        print("\n".join(parts)); return
    selected = select_parts(parts, args.only)
    out = ensure_dir(args.output_dir)
    tag = "_".join(selected) if args.only else "all"
    resolved = {"n_sites": N_SITES, "t_schedule": list(T_SCHEDULE), "quick": args.quick,
                "parts": list(selected), "overrides": args.set,
                "ladders": {k: [str(h) for h in v] for k, v in LADDERS.items()}, **cfg}
    write_json(out / f"params_{tag}.json", {**resolved, **provenance(resolved)})
    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(cfg, out)
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
