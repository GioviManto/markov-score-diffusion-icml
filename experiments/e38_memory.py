#!/usr/bin/env python
"""E38 -- peak memory of one posterior mean and one training step (the consolidated note, section 9.8).

E24 measured time. The note's list of what a fair runtime comparison must report also has memory use; nothing in
the project measured it. E38 measures the peak RESIDENT memory of one call, with E24's own constructors and inputs
(LaplaceAR1 at rho = 0.85, E24's rng tags, the six networks of E24's NET_ARMS, exact grid BP with chunk = None as E24
times it).

Each measurement runs in a FRESH subprocess (`--worker <json>`): the inputs are built and a warm-up call on four
sequences is made, then the measured call runs while a background thread samples the CURRENT resident memory from
/proc/self/statm every 0.5 ms (Linux; numpy and torch release the GIL inside their kernels, so the sampler runs);
the reading is the sampled peak minus the resident memory just before the call. ru_maxrss alone would not do: it is
a high-water mark, and an import or warm-up that peaked higher hides a smaller call entirely (it read +0.0 MB for a
small transformer in the smoke run). On macOS, which has no statm, the worker falls back to the ru_maxrss increment
and says so in the `method` column; the registered numbers come from the Linux run.

PREDECLARED (registry E38.json, written before any measurement):
  P1  grid BP's peak inference memory is linear in the batch and in the length: log-log slope in [0.8, 1.2] over
      B = 128 -> 2048 (L = 32) and over L = 32 -> 128 (B = 512).
  P2  at the operating point (B = 512, L = 32, N_g = 401) grid BP needs more peak memory per posterior-mean call
      than every one of the six networks.
"""
from __future__ import annotations

import gc
import json
import resource
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, select_parts  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

SETTINGS = dict(
    lengths=(32, 64, 128), batches=(128, 512, 2048), grids=(201, 401, 801), base_grid=401, base_batch=512, base_length=32,
    em_sizes=(512, 2048), net_train_steps=20, level_index=6,
)
QUICK = dict(lengths=(16,), batches=(32,), grids=(101,), base_grid=101, base_batch=32, base_length=16, em_sizes=(64,),
             net_train_steps=2)


def _maxrss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10   # bytes on macOS, kilobytes on Linux


class _PeakSampler:
    """Samples the current RSS (MB) from /proc/self/statm in a thread; `peak` after stop()."""

    def __init__(self, period: float = 5e-4):
        import os
        import threading
        self._page = os.sysconf("SC_PAGE_SIZE") / 2**20
        self.period, self.peak, self._stop = period, 0.0, threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def now(self) -> float:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * self._page

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.now())
            self._stop.wait(self.period)

    def __enter__(self):
        self.peak = self.now(); self._t.start(); return self

    def __exit__(self, *exc):
        self._stop.set(); self._t.join(); self.peak = max(self.peak, self.now()); return False


def _measure(call) -> dict:
    """Peak extra memory of call(): statm sampling on Linux, the ru_maxrss increment elsewhere."""
    gc.collect()
    if Path("/proc/self/statm").exists():
        s = _PeakSampler()
        before = s.now()
        with s:
            call()
        return dict(method="statm-sampled", base_mb=before, peak_mb=s.peak, increment_mb=max(s.peak - before, 0.0))
    before = _maxrss_mb()
    call()
    after = _maxrss_mb()
    return dict(method="maxrss-increment", base_mb=before, peak_mb=after, increment_mb=max(after - before, 0.0))


def worker(spec: dict) -> dict:
    """Run ONE measurement in this process and return it; meant for a fresh interpreter."""
    import e24_inference_cost as e24
    from markov_score.bp_grid import grid_bp_batch, make_grid
    from markov_score.priors import LaplaceAR1

    kind, arm, L, B, ng = spec["kind"], spec["arm"], int(spec["L"]), int(spec["B"]), int(spec["N_g"])
    cfg = dict(e24.SETTINGS)
    t = float(e24.T_GRID[spec["level_index"]])
    grid, w = make_grid(cfg["half_width"], ng)
    prior = LaplaceAR1(cfg["rho"])
    log_K = prior.log_transition_matrix(grid)
    if kind == "infer":
        _, X, a, d = e24._bundle(cfg, "e24-cost", L, B, t)
        if arm == "bp_grid":
            def f(Xb):
                return grid_bp_batch(grid, w, log_K, Xb, a, d, chunk=None)[0]
        else:
            from markov_score import nn_torch as nt
            hp = dict(e24.NET_ARMS)[arm]
            den = nt.Denoiser(nt.build(arm, L, **hp), "eps")

            def f(Xb):
                return den.posterior_mean(Xb, t)
        f(X[:4])
        m = _measure(lambda: f(X))
    elif arm == "em_estep":
        from markov_score.em import e_step_multi
        from markov_score.kernels import MixtureInnovationKernel
        A, X, a, d = e24._bundle(cfg, "e24-train", L, B, t)
        k = MixtureInnovationKernel.init(8, rho=0.3, var=0.8, rng=rng_for("e38-em", B))
        lk = k.log_transition_matrix(grid)
        e_step_multi(grid, w, lk, [(X[:4], a, d)])
        m = _measure(lambda: e_step_multi(grid, w, lk, [(X, a, d)]))   # the protocol's chunk = 256
    else:
        from markov_score import nn_torch as nt
        A, _, _, _ = e24._bundle(cfg, "e24-train", L, B, t)
        hp = dict(e24.NET_ARMS)[arm]
        steps = int(spec["steps"])

        def train(n):
            tc = nt.TrainConfig(steps=n, batch_size=64, lr=1e-3, parameterization="eps", loss="dsm", checkpoints=(n,),
                                t_levels=tuple(float(x) for x in e24.T_GRID), device="cpu", seed=1)
            nt.train(nt.build(arm, L, **hp), A, tc)
        train(1)
        m = _measure(lambda: train(steps))
    return dict(spec, **m)


def _run(spec, py=sys.executable) -> dict:
    r = subprocess.run([py, __file__, "--worker", json.dumps(spec)], capture_output=True, text=True,
                       env={"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1", **{k: v for k, v in __import__("os").environ.items() if k in ("PATH", "HOME", "PYTHONPATH")}})
    if r.returncode != 0:
        raise RuntimeError(f"worker failed for {spec}:\n{r.stderr[-2000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def part_memory(cfg, out, workers):
    import e24_inference_cost as e24
    nets = [a for a, _ in e24.NET_ARMS]
    lv = int(cfg["level_index"])
    specs = []
    for L in cfg["lengths"]:
        for B in cfg["batches"]:
            for arm in ["bp_grid", *nets]:
                specs.append(dict(kind="infer", arm=arm, L=L, B=B, N_g=cfg["base_grid"], level_index=lv))
    for ng in cfg["grids"]:
        if ng != cfg["base_grid"]:
            specs.append(dict(kind="infer", arm="bp_grid", L=cfg["base_length"], B=cfg["base_batch"], N_g=ng, level_index=lv))
    for M in cfg["em_sizes"]:
        specs.append(dict(kind="train", arm="em_estep", L=cfg["base_length"], B=M, N_g=cfg["base_grid"], level_index=lv))
    for arm in nets:
        specs.append(dict(kind="train", arm=arm, L=cfg["base_length"], B=max(cfg["em_sizes"]), N_g=cfg["base_grid"],
                          level_index=lv, steps=cfg["net_train_steps"]))
    rows = []
    for i, s in enumerate(specs):
        r = _run(s)
        rows.append(r)
        print(f"  [E38] {i + 1}/{len(specs)} {r['kind']:5s} {r['arm']:11s} L={r['L']:3d} B={r['B']:4d} N_g={r['N_g']}: "
              f"+{r['increment_mb']:8.1f} MB (base {r['base_mb']:.0f}, {r['method']})", flush=True)
    write_csv(out / "memory.csv", rows)


PARTS = {"memory": part_memory}


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        print(json.dumps(worker(json.loads(sys.argv[2]))))
        return
    ap = experiment_parser("e38_memory", "peak memory of one posterior mean and one training step")
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
        write_json(out / f"params_{name}.json", {**provenance(), "settings": cfg, "quick": bool(args.quick), "workers": 1})


if __name__ == "__main__":
    main()
