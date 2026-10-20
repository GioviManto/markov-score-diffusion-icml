"""E54: the standard network on the phi^4 field -- a CNN trained by plain denoising score matching.

Why (Jerome, call of 19 Oct 2026, 01:16-01:20): "just do denoising score matching ... standard DDPM is fine";
compare one direction at a time, an estimator that knows the structure against one that does not; a CNN
because the data are local. E33's networks were trained on ONE fixed noisy copy per sequence (the matched
interface, `dsm_one_corruption`) and only at M = 128 and 2048. The standard recipe draws a fresh level and
fresh noise for every clean training sequence at every step, which gives the network more information than
EM-BP has (EM-BP sees one noisy copy of each sequence). It is the network-favourable baseline.

One arm, nothing else:
    window CNN (E33's per-cell winning hyper-parameters, outputs/hpc_E33n_s0/winners.json),
    loss = DSM on clean sequences, a fresh t from the twelve-level schedule and fresh noise each step
           (`nn_torch.train`, loss "dsm", eps parameterisation), batch 64, Adam 1e-3, clip 1, 64k steps,
    checkpoint chosen on the validation bundle's BP risk (the rule the kernel arms' `risk` rows use),
    scored on E33's test bundle (same generator keys, so the bundle hashes match E33's rows: paired).

Predictions: experiments/registry/E54.json. Reading: tools/core/read_e54.py.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e14_regime_baselines as e14  # noqa: E402
import e33_field_learning as e33  # noqa: E402
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WINNERS = ROOT / "outputs" / "hpc_E33n_s0" / "winners.json"

SETTINGS = dict(
    cells=e33.SETTINGS["cells"], sizes=(32, 128, 512, 2048), seeds=tuple(range(8)),
    n_val=e33.SETTINGS["n_val"], n_test=e33.SETTINGS["n_test"],
    grid_size=e33.SETTINGS["grid_size"], half_width=e33.SETTINGS["half_width"],
    arch="window", lr=1e-3, steps=64_000, checkpoints=(500, 1000, 2000, 4000, 8000, 16000, 32000, 64000),
    batch_size=64, grad_clip=1.0, mode="eps", torch_threads=1,
)
QUICK = dict(cells=("T0.3_h1_L32",), sizes=(32,), seeds=(10_000,), n_val=32, n_test=32, grid_size=201,
             steps=200, checkpoints=(100, 200))


def _cell(args):
    cfg, cell_id, seed, n = args
    import torch
    from markov_score import nn_torch as nt
    torch.set_num_threads(cfg["torch_threads"])
    chain, grid, w, log_k, log_mu, sigma0, A, Av, val, test, base = e33._common(cfg, cell_id, seed, n)
    L = base["n_sites"]
    hp = json.loads(WINNERS.read_text())[f"{cell_id}|{cfg['arch']}"]["hp"]
    model = nt.build(cfg["arch"], L, **hp)
    tcfg = nt.TrainConfig(steps=cfg["steps"], batch_size=cfg["batch_size"], lr=cfg["lr"], grad_clip=cfg["grad_clip"],
                          parameterization=cfg["mode"], loss="dsm", checkpoints=tuple(cfg["checkpoints"]),
                          t_levels=tuple(e33.T_SCHEDULE),
                          seed=int(rng_for("e54-net", cfg["arch"], cell_id, seed, n).integers(0, 2**31 - 1)))
    t0 = time.time()
    saved = nt.train(model, A, tcfg)
    best, best_v, curve = None, np.inf, {}
    for step in sorted(saved):
        den = nt.Denoiser(nt.load_checkpoint(model, saved[step]), cfg["mode"])
        v = e14._pooled(lambda X, t: den.posterior_mean(X, t), val)
        curve[step] = v
        if v < best_v:
            best, best_v = step, v
    den = nt.Denoiser(nt.load_checkpoint(model, saved[best]), cfg["mode"])
    risk = e14._pooled(lambda X, t: den.posterior_mean(X, t), test)
    row = {**base, "arm": "dsm_standard", "arch": cfg["arch"], "hp": json.dumps(hp), "steps": cfg["steps"],
           "checkpoint": best, "at_cap": int(best == max(saved)), "val_selected": best_v, "risk": risk,
           "n_params": nt.n_params(model), "wall": time.time() - t0,
           "val_curve": ";".join(f"{k}:{v:.5f}" for k, v in sorted(curve.items()))}
    print(f"  [E54] {cell_id} seed={seed} M={n}: risk={risk:.4f} (ckpt {best}, {row['wall']:.0f}s)", flush=True)
    return [row]


def part_nets(cfg, out, workers):
    size = {c: int(c.rsplit("_L", 1)[1]) for c in cfg["cells"]}
    cells = [(cfg, c, s, n) for c in sorted(cfg["cells"], key=lambda c: -size[c])
             for n in sorted(cfg["sizes"], reverse=True) for s in cfg["seeds"]]
    print(f"  nets: {len(cells)} fits", flush=True)
    write_csv(out / "nets.csv", run_cells(cells, _cell, workers, out, "nets.csv",
                                          key=lambda c: f"{c[1]}_s{c[2]}_M{c[3]}"))


PARTS = {"nets": part_nets}


def main():
    ap = experiment_parser("e54_field_dsm", "standard DSM CNN on the phi^4 field, paired with E33's kernel arms")
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
