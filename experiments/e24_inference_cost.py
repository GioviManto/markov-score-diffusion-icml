#!/usr/bin/env python
"""E24 -- inference cost and scaling: what a posterior mean actually costs.

WHY THIS EXPERIMENT EXISTS
--------------------------
Jerome, 25 September 2026 (`docs/JEROME_CALL_2026-09-25.md`, Q7):

    "If I give you a noisy sequence and I want the posterior mean, how long does it
     take for a neural network and how long does it take for BP?"
    "This is a big question to me."

The answer given in the call was a guess ("I think it should take less time"), and no
such number exists anywhere in this project. Every experiment here reports *error*; none
reports the cost of producing one posterior mean. The headline claim we want to make --
one small kernel inside exact BP instead of one large score network -- is a claim about
cost as much as about accuracy, so the cost has to be measured rather than asserted.

The call also produced a WRONG scaling answer, which Jerome then built his question on
("it depends in a squared way from how many sequences and in a linear way from L").
Reading `bp_grid.grid_bp_batch`, the recursion is an (N_g, N_g) @ (N_g, B) matmul per
site over L sites for B independent chains, i.e.

    cost  ~  L * N_g^2 * B

-- linear in the number of sequences, linear in L, QUADRATIC IN THE GRID. Part `scaling`
measures all three exponents rather than leaving that as a reading of the source. The
distinction is not bookkeeping: it says the bottleneck Jerome identified by instinct is
in the GRID and not in the data. (What follows from that is more subtle than it first
looked, and part `anatomy` below is what settled it: the N_g^2 matmul is only 24% of a
sweep at the production resolution, so an FFT cannot be the answer -- see E23.)

WHAT IS AND IS NOT MEASURED
---------------------------
Measured: wall-clock to produce posterior means for a batch of noisy sequences, at
inference, for each arm, plus parameter counts.

Not measured: accuracy. It already lives in E03/E14 under the registered estimand, and
recomputing it here would duplicate that machinery in an experiment whose one job is
timing. Cost and error are joined in the document, not in this CSV.

Networks are timed *untrained*. A forward pass costs the same whatever the weights are,
so this measures the architecture, which is the object the question is about. The arms
are the architectures and hyper-parameters of E03's screen grid (`e03_baselines.GRIDS`),
so the numbers refer to models this project actually trains.

DEVICE IS A FACTOR, NOT A FOOTNOTE
----------------------------------
Timing a GPU-native score network against a CPU-only BP recursion and reporting the
ratio would be an artefact of what each arm was allowed to run on, not a property of
either algorithm. Both families have a GPU path here -- `bp_grid.grid_bp_batch` takes an
array module (`markov_score.backend.get_xp`, float64 throughout), and the networks take a
torch device -- so `devices` is a swept factor and every row records the device it ran
on. A cross-device comparison in the write-up must quote both.

Two consequences for the measurement, both of which silently corrupt GPU timings if
ignored:
  (i)  CUDA work is asynchronous. A timed region that only launches kernels measures
       launch overhead. Every cell therefore ends in a synchronising step.
  (ii) The question is "give me the posterior mean", so the cost is host-to-host: the
       result is brought back to numpy inside the timed region, transfer included. That
       is also what makes (i) automatic for the network arms, whose `posterior_mean`
       already copies back to the host.

Timing hygiene: one warm-up pass per cell, discarded (it also absorbs CUDA context
creation and cuBLAS handle setup); `n_repeat` timed passes; median and inter-quartile
range reported, because neither a laptop under a browser nor a shared cluster node is a
quiet machine and a mean is not robust to one scheduling hiccup. Thread counts (BLAS and
torch), the device name and the CUDA versions are recorded in the params file -- a
timing without them is not reproducible. This experiment deliberately does NOT set the
thread counts: it measures the machine as configured, and the batch script owns that.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score.bp_gaussian import gaussian_chain_bp  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

L_DEFAULT = FROZEN.n_sites
T_GRID = tuple(FROZEN.t_grid)

# The architectures and hyper-parameters E03 screens, one representative each: the
# LARGEST of the screened grid, so the comparison is against the networks at the
# capacity this project actually considers, not a token small one.
NET_ARMS = (
    ("mlp", dict(hidden=(256, 256))),
    ("window", dict(radius=4, width=128)),
    ("tcn", dict(channels=64, dilations=(1, 2, 4, 8))),
    ("bigru", dict(hidden=64)),
    ("transformer", dict(d=64, heads=4, layers=4)),
    ("unet", dict(channels=32)),
)

SETTINGS = dict(
    rho=FROZEN.rho,
    n_sites=L_DEFAULT,
    batch=512,
    n_repeat=5,
    levels=T_GRID,
    grid_size=FROZEN.n_grid,
    half_width=FROZEN.half_width,
    chunk=None,                      # None: one pass, no chunking, so the timing is the algorithm
    devices=("cpu",),                # ("cpu", "gpu") on a GPU node; see `_resolve_devices`
    # --- part `scaling` sweeps, one factor at a time around the base point -------
    scale_batches=(32, 64, 128, 256, 512, 1024, 2048),
    scale_sites=(8, 16, 32, 64, 128, 256),
    scale_grids=(101, 201, 401, 801, 1601),
    scale_level=T_GRID[6],
    scale_repeat=3,
    # --- part `training` ---------------------------------------------------------
    train_sizes=(32, 128, 512, 2048),
    train_repeat=3,
    train_steps=200,                 # measured budget; scaled to `protocol_steps` for reporting
    protocol_steps=32_000,           # PROTOCOL.net_steps, the budget the confirmatory runs use
    mdn_components=4,
    mdn_hidden=32,
    # --- part `capacity` --------------------------------------------------------
    cap_sizes=(128, 512, 2048),
    cap_components=(1, 2, 4, 8, 16, 32),
    cap_hidden=(16, 32, 64, 128, 256),
    cap_transformer=((32, 2), (64, 4), (128, 4), (256, 6)),
    cap_steps=100,
)
QUICK = dict(
    batch=32, n_repeat=2, levels=(T_GRID[3], T_GRID[8]), grid_size=201,
    scale_batches=(16, 32, 64), scale_sites=(8, 16), scale_grids=(101, 201), scale_repeat=2,
    train_sizes=(32, 128), train_repeat=1, train_steps=20,
    cap_sizes=(32, 128), cap_components=(1, 4), cap_hidden=(16, 32),
    cap_transformer=((32, 2),), cap_steps=10,
)


# ----------------------------------------------------------------------- helpers


def _timed(fn, n_repeat: int) -> dict:
    """One discarded warm-up, then `n_repeat` timed calls; median and IQR in seconds.

    `fn` must be host-to-host: it returns only once the result is available on the CPU.
    That is the definition of the quantity being measured ("give me the posterior mean")
    and it is what makes the timing valid on an asynchronous device -- a region that
    merely launches CUDA kernels measures launch overhead, not work.
    """
    fn()                                                   # warm-up, discarded
    ts = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts = np.sort(np.asarray(ts))
    return {
        "wall_median": float(np.median(ts)),
        "wall_min": float(ts[0]),
        "wall_iqr": float(np.subtract(*np.percentile(ts, [75, 25]))),
        "n_repeat": int(n_repeat),
    }


def _resolve_devices(requested) -> list[str]:
    """Keep only devices that actually exist, and say what was dropped.

    A "gpu" row produced on a node without one would be a false provenance claim, and
    `backend.get_xp('gpu')` already refuses to fall back silently. Here the request is
    filtered up front so a mixed sweep on a laptop runs its CPU half instead of dying.
    """
    from markov_score import backend
    from markov_score.backend import gpu_available

    strict = os.environ.get("E24_REQUIRE_GPU", "").strip() not in ("", "0")
    out = []
    for d in requested:
        d = str(d).lower()
        if d in ("cpu", "numpy"):
            out.append("cpu")
        elif d in ("gpu", "cuda", "cupy"):
            if gpu_available():
                out.append("gpu")
            elif strict:
                # A job submitted to a GPU partition that quietly produces CPU-only numbers
                # is worse than one that fails: on 25 September exactly that happened and the
                # cost table -- the one the job existed for -- was silently CPU-only while the
                # scaling table in the SAME process had GPU rows.
                raise SystemExit(
                    "E24_REQUIRE_GPU is set and no usable GPU was found.\n  probe: "
                    + (backend.last_probe_error or "(no error recorded)")
                    + "\nRefusing to produce a CPU-only cost table from a GPU job."
                )
            else:
                print("  [E24] 'gpu' requested but no usable device; dropping that arm"
                      f" (probe: {backend.last_probe_error or 'no error recorded'})", flush=True)
        else:
            raise SystemExit(f"unknown device {d!r}; expected 'cpu' or 'gpu'")
    if not out:
        raise SystemExit("no usable device among " + repr(tuple(requested)))
    return out


def _torch_device(device: str) -> str:
    """Map this experiment's device label onto a torch device string."""
    if device == "cpu":
        return "cpu"
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "device 'gpu' requested for a network arm but torch reports no CUDA device. "
            "Refusing to time a CPU forward pass and label it GPU."
        )
    return "cuda"


def _bundle(cfg, seed_tag: str, n_sites: int, batch: int, t: float):
    """A batch of noisy sequences from the Laplace chain at level t."""
    prior = LaplaceAR1(cfg["rho"])
    rng = rng_for(seed_tag, cfg["rho"], n_sites, batch)
    A = np.stack([prior.sample(rng, n_sites) for _ in range(batch)])
    alpha, delta = alpha_delta(t)
    X = alpha * A + np.sqrt(delta) * rng.standard_normal(A.shape)
    return A, X, alpha, delta


def _thread_info() -> dict:
    info = {k: os.environ.get(k, "") for k in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
    info["cpu_count"] = str(os.cpu_count())
    try:
        import torch
        info["torch_threads"] = str(torch.get_num_threads())
        info["torch_version"] = torch.__version__
        info["torch_cuda"] = str(torch.version.cuda)
        info["torch_cuda_available"] = str(torch.cuda.is_available())
        info["torch_devices"] = ", ".join(
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        )
    except Exception as exc:                                   # torch is optional here
        info["torch_threads"] = ""
        info["torch_version"] = f"unavailable ({exc})"
    try:
        from markov_score.backend import device_name, get_xp, gpu_available
        info["cupy_gpu_available"] = str(gpu_available())
        from markov_score import backend as _b
        info["cupy_probe_error"] = _b.last_probe_error
        info["bp_device_name"] = device_name(get_xp("auto")) if gpu_available() else "cpu:numpy"
    except Exception as exc:
        info["cupy_gpu_available"] = f"unavailable ({exc})"
    return info


# ----------------------------------------------------------------------- part: cost


def _bp_grid_caller(cfg, grid, weights, log_K, X, alpha, delta, device):
    """A host-to-host closure that runs one batched grid-BP sweep on `device`.

    Returns (callable, device_label). The arrays are moved to the device ONCE, outside
    the timed region, because the question is the cost of the recursion for data that is
    already where it is going to be computed -- the same convention the network arms get,
    whose weights also live on the device before the call. The result is brought back to
    the host INSIDE the closure, so the timing covers the whole answer and synchronises.
    """
    from markov_score.backend import device_name, get_xp, to_device, to_host

    xp = get_xp(device)
    g, w, lk, Xd = (to_device(a, xp) for a in (grid, weights, log_K, X))

    def call():
        means, _ = grid_bp_batch(g, w, lk, Xd, alpha, delta, xp=xp, chunk=cfg["chunk"])
        return to_host(means)

    return call, device_name(xp)


def part_cost(cfg, out):
    """Seconds per posterior mean, per arm, per device, over the twelve-level schedule."""
    rows = []
    n_sites, batch = int(cfg["n_sites"]), int(cfg["batch"])
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    prior = LaplaceAR1(cfg["rho"])
    log_K = prior.log_transition_matrix(grid)
    q = prior.q
    devices = _resolve_devices(cfg["devices"])

    for device in devices:
        for t in cfg["levels"]:
            _, X, alpha, delta = _bundle(cfg, "e24-cost", n_sites, batch, float(t))
            base = dict(device=device, level=float(t), batch=batch, n_sites=n_sites,
                        grid_size=int(cfg["grid_size"]), half_width=float(cfg["half_width"]))
            here = []

            # --- exact BP on the grid --------------------------------------------
            call, dev_name = _bp_grid_caller(cfg, grid, weights, log_K, X, alpha, delta, device)
            r = _timed(call, int(cfg["n_repeat"]))
            here.append({**base, "arm": "bp_grid", "family": "bp", "hp": f"N_g={cfg['grid_size']}",
                         "device_name": dev_name, "n_params": int(cfg["grid_size"]) ** 2, **r,
                         "sec_per_seq": r["wall_median"] / batch})

            # --- analytic Gaussian BP (the order-2 rung; closed form, no grid) ----
            # CPU only, and deliberately so: it is a scalar recursion with no array work
            # to offload, and a GPU row for it would be a slower number with no meaning.
            if device == "cpu":
                def _gauss():
                    for i in range(batch):
                        gaussian_chain_bp(X[i], cfg["rho"], q, alpha, delta)
                r = _timed(_gauss, int(cfg["n_repeat"]))
                here.append({**base, "arm": "bp_gaussian", "family": "bp", "hp": "closed form",
                             "device_name": "cpu:numpy", "n_params": 2, **r,
                             "sec_per_seq": r["wall_median"] / batch})

            # --- the network arms -------------------------------------------------
            for arch, hp in NET_ARMS:
                try:
                    import torch

                    from markov_score import nn_torch as nt
                    tdev = _torch_device(device)
                    model = nt.build(arch, n_sites, **hp).to(tdev)
                    den = nt.Denoiser(model, "eps")
                    npar = nt.n_params(model)
                    dname = (f"gpu:{torch.cuda.get_device_name(0)}" if tdev == "cuda" else "cpu:torch")
                except Exception as exc:
                    here.append({**base, "arm": arch, "family": "net", "hp": str(hp),
                                 "device_name": "", "n_params": -1, "wall_median": float("nan"),
                                 "wall_min": float("nan"), "wall_iqr": float("nan"),
                                 "n_repeat": 0, "sec_per_seq": float("nan"), "error": str(exc)})
                    continue
                # `posterior_mean` returns a numpy array, so it is host-to-host already.
                r = _timed(lambda: den.posterior_mean(X, float(t)), int(cfg["n_repeat"]))
                here.append({**base, "arm": arch, "family": "net", "hp": str(hp),
                             "device_name": dname, "n_params": npar, **r,
                             "sec_per_seq": r["wall_median"] / batch})

            rows.extend(here)
            summary = "  ".join(f"{x['arm']}={x['sec_per_seq'] * 1e3:.3f}ms" for x in here)
            print(f"  [E24 cost] {device} t={t}: {summary}", flush=True)

    write_csv(out / "cost.csv", rows)
    return rows


# --------------------------------------------------------------------- part: scaling


def _loglog_slope(xs, ys) -> float:
    """Least-squares exponent of y ~ x^s. NaN if fewer than two finite points."""
    x, y = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def _slopes(xs, ys) -> tuple[float, float, int]:
    """Full-range exponent, and the exponent over the LARGEST half of the range.

    Both are reported because the full-range fit is biased towards zero whenever fixed
    per-call overhead (Python dispatch, allocation, a device transfer) is comparable to
    the work -- which it is at the small end of any of these sweeps. The asymptotic claim
    "grid BP is quadratic in N_g" is a claim about the large end, so it has to be read
    off the large end; quoting the full-range number alone would understate every
    exponent and would have made the O(N_g^2) bottleneck look milder than it is.
    """
    order = np.argsort(np.asarray(xs, float))
    xs = np.asarray(xs, float)[order]
    ys = np.asarray(ys, float)[order]
    top = max(2, int(np.ceil(len(xs) / 2)))
    return _loglog_slope(xs, ys), _loglog_slope(xs[-top:], ys[-top:]), top


def part_scaling(cfg, out):
    """The three exponents of grid BP, and the network arms' exponents in B and L.

    One factor is varied at a time around the base point; everything else is held at
    SETTINGS. The predeclared prediction for grid BP is slope 1 in B, 1 in L, 2 in N_g.
    """
    rows = []
    t = float(cfg["scale_level"])
    rep = int(cfg["scale_repeat"])
    base_B, base_L, base_G = int(cfg["batch"]), int(cfg["n_sites"]), int(cfg["grid_size"])
    prior = LaplaceAR1(cfg["rho"])
    devices = _resolve_devices(cfg["devices"])

    for device in devices:
        for factor, values in (("batch", cfg["scale_batches"]),
                               ("n_sites", cfg["scale_sites"]),
                               ("grid_size", cfg["scale_grids"])):
            for v in values:
                n_sites = int(v) if factor == "n_sites" else base_L
                batch = int(v) if factor == "batch" else base_B
                grid_size = int(v) if factor == "grid_size" else base_G
                grid, weights = make_grid(cfg["half_width"], grid_size)
                log_K = prior.log_transition_matrix(grid)
                _, X, alpha, delta = _bundle(cfg, "e24-scale", n_sites, batch, t)
                call, dev_name = _bp_grid_caller(cfg, grid, weights, log_K, X, alpha, delta, device)
                r = _timed(call, rep)
                rows.append({"arm": "bp_grid", "family": "bp", "device": device,
                             "device_name": dev_name, "factor": factor, "value": int(v),
                             "batch": batch, "n_sites": n_sites, "grid_size": grid_size,
                             "level": t, **r})
                print(f"  [E24 scaling] {device} bp_grid {factor}={v}: {r['wall_median']:.3f}s", flush=True)

        # The networks, in B and L only: N_g is not one of their parameters.
        try:
            import torch

            from markov_score import nn_torch as nt
            tdev = _torch_device(device)
        except Exception as exc:
            print(f"  [E24 scaling] {device}: network arms skipped ({exc})", flush=True)
            continue
        dname = f"gpu:{torch.cuda.get_device_name(0)}" if tdev == "cuda" else "cpu:torch"
        for arch, hp in NET_ARMS:
            for factor, values in (("batch", cfg["scale_batches"]), ("n_sites", cfg["scale_sites"])):
                for v in values:
                    n_sites = int(v) if factor == "n_sites" else base_L
                    batch = int(v) if factor == "batch" else base_B
                    _, X, _, _ = _bundle(cfg, "e24-scale", n_sites, batch, t)
                    common = {"arm": arch, "family": "net", "device": device,
                              "device_name": dname, "factor": factor, "value": int(v),
                              "batch": batch, "n_sites": n_sites, "grid_size": 0, "level": t}
                    try:
                        den = nt.Denoiser(nt.build(arch, n_sites, **hp).to(tdev), "eps")
                        r = _timed(lambda: den.posterior_mean(X, t), rep)
                    except Exception as exc:
                        rows.append({**common, "wall_median": float("nan"), "wall_min": float("nan"),
                                     "wall_iqr": float("nan"), "n_repeat": 0, "error": str(exc)})
                        continue
                    rows.append({**common, **r})
            print(f"  [E24 scaling] {device} {arch} done", flush=True)

    write_csv(out / "scaling.csv", rows)

    # The exponents, with the predeclared prediction beside each measurement.
    predicted = {("bp_grid", "batch"): 1.0, ("bp_grid", "n_sites"): 1.0, ("bp_grid", "grid_size"): 2.0}
    slopes = []
    for device in devices:
        for arm in sorted({r["arm"] for r in rows}):
            for factor in ("batch", "n_sites", "grid_size"):
                sel = [r for r in rows if r["arm"] == arm and r["factor"] == factor
                       and r["device"] == device
                       and np.isfinite(r.get("wall_median", float("nan")))]
                if len(sel) < 2:
                    continue
                s, s_top, n_top = _slopes([r["value"] for r in sel], [r["wall_median"] for r in sel])
                slopes.append({"device": device, "arm": arm, "factor": factor,
                               "slope_full": s, "slope_top": s_top, "n_top": n_top,
                               "predicted": predicted.get((arm, factor), ""),
                               "n_points": len(sel),
                               "value_min": min(r["value"] for r in sel),
                               "value_max": max(r["value"] for r in sel)})
                tag = f"  (predicted {predicted[(arm, factor)]:+.1f})" if (arm, factor) in predicted else ""
                print(f"  [E24 slope] {device:4s} {arm:12s} {factor:10s} "
                      f"full={s:+.3f} top={s_top:+.3f}{tag}", flush=True)
    write_csv(out / "slopes.csv", slopes)
    return rows


def part_anatomy(cfg, out):
    """Where grid BP's time actually goes: the M^2 matmul, or everything else.

    WHY THIS PART EXISTS. Part `scaling` measured a grid-size exponent of ~1.3-1.5, not
    the 2.0 predicted from the operation count. Either the recursion is not doing the
    work we think it is, or it is not FLOP-bound. This part separates the two by timing,
    at each N_g, the bare `2 (L-1)` dgemms of shape (N_g, N_g) @ (N_g, B) that the
    recursion performs, against the full sweep.

    It decides a design question and not only a diagnostic one. If the matmul is a small
    share of the time, then an FFT update -- which replaces M^2 by M log M and touches
    nothing else -- cannot deliver much, because the remaining time is the O(B L N_g)
    elementwise work and memory traffic over the (B, L, N_g) message arrays. Truncating
    the representation to K << N_g modes shrinks BOTH terms. So this measurement is what
    tells E23 whether to build FFT acceleration or mode truncation first.
    """
    from markov_score.backend import device_name, get_xp, to_device, to_host

    rows = []
    B, n = int(cfg["batch"]), int(cfg["n_sites"])
    t = float(cfg["scale_level"])
    rep = int(cfg["scale_repeat"])
    prior = LaplaceAR1(cfg["rho"])
    _, X, alpha, delta = _bundle(cfg, "e24-anat", n, B, t)

    for device in _resolve_devices(cfg["devices"]):
        xp = get_xp(device)
        for M in cfg["scale_grids"]:
            M = int(M)
            grid, weights = make_grid(cfg["half_width"], M)
            log_K = prior.log_transition_matrix(grid)
            call, dev_name = _bp_grid_caller(cfg, grid, weights, log_K, X, alpha, delta, device)
            full = _timed(call, rep)

            # The matmul work the recursion performs: two sweeps over (L-1) edges, each a
            # (M, M) @ (M, B) product. Timed on the same device, host-to-host at the end so
            # an asynchronous device cannot hide the work.
            Kd = to_device(np.exp(log_K), xp)
            Inc = to_device(np.asfortranarray(np.random.default_rng(0).standard_normal((M, B))), xp)

            def _gemms():
                acc = None
                for _ in range(2 * (n - 1)):
                    acc = Kd @ Inc
                return to_host(acc)

            gemm = _timed(_gemms, rep)
            flops = 2 * (n - 1) * 2 * M * M * B          # 2 sweeps, 2 flops per MAC
            msg_bytes = 3 * B * n * M * 8                # ell + L + R, float64
            rows.append({
                "device": device, "device_name": dev_name, "grid_size": M, "batch": B,
                "n_sites": n, "level": t,
                "wall_full": full["wall_median"], "wall_gemm": gemm["wall_median"],
                "gemm_fraction": gemm["wall_median"] / full["wall_median"],
                "gflops_gemm": flops / gemm["wall_median"] / 1e9,
                "gflops_effective": flops / full["wall_median"] / 1e9,
                "message_array_mb": msg_bytes / 1e6,
                "n_repeat": rep,
            })
            print(f"  [E24 anatomy] {device} N_g={M:5d}: full={full['wall_median']:.4f}s "
                  f"gemm={gemm['wall_median']:.4f}s ({100 * rows[-1]['gemm_fraction']:.1f}%) "
                  f"gemm={rows[-1]['gflops_gemm']:.0f} GF/s  msg={rows[-1]['message_array_mb']:.0f} MB",
                  flush=True)

    write_csv(out / "anatomy.csv", rows)
    return rows


def part_training(cfg, out):
    """Training cost, and the decomposition that carries the whole argument.

    WHY THIS PART EXISTS. Part `cost` answered Jerome's Q7 and the answer was bad for us:
    at inference exact BP is 2.2x SLOWER than a transformer per posterior mean. But the
    claim the project is actually making -- one small kernel instead of one large score
    network -- was never an inference claim. It is a *training* claim, and it rests on a
    structural asymmetry that `MDNKernel`'s docstring already states:

        a score network backpropagates through every training example at every gradient
        step; EM-BP evaluates and updates the kernel ONLY on the N_g grid points, once
        per EM iteration, no matter how large the dataset is. Data enters the M-step
        exclusively through the accumulated statistics Xi.

    So the two halves of an EM iteration should scale completely differently:
        E-step  (exact BP over the data)     -- LINEAR in M, independent of the kernel
        M-step  (the kernel's own update)    -- INDEPENDENT of M, set by N_g and capacity

    PREDECLARED PREDICTION, fixed before this part was run: the fitted exponent of
    m_step wall-clock in M is ~0, and of e_step wall-clock in M is ~1. If the M-step
    exponent is materially above zero the mechanism is not what the docstring claims and
    the "small network" argument loses its basis.

    Both kernels are timed: the mixture kernel (closed-form M-step) and MDNKernel (a
    small MLP, gradient M-step) -- the latter being the arm Jerome proposed and the one
    the prediction matters for.

    The network arms are timed per gradient step and reported both per step and scaled to
    the protocol's 32k-step budget, so the two families can be put on one axis.
    """
    rows = []
    n_sites = int(cfg["n_sites"])
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    prior = LaplaceAR1(cfg["rho"])
    levels = list(cfg["levels"])
    rep = int(cfg["train_repeat"])

    from markov_score.em import e_step_multi
    from markov_score.kernels import MDNKernel, MixtureInnovationKernel

    for M in cfg["train_sizes"]:
        M = int(M)
        rng = rng_for("e24-train", M)
        A = np.stack([prior.sample(rng, n_sites) for _ in range(M)])
        # PROTOCOL.one_view: one noised view per sequence, sequences split evenly over levels.
        groups, per = [], max(1, M // len(levels))
        for j, t in enumerate(levels):
            sl = A[j * per:(j + 1) * per]
            if len(sl) == 0:
                continue
            al, de = alpha_delta(float(t))
            groups.append((al * sl + np.sqrt(de) * rng.standard_normal(sl.shape), al, de))

        kernels = {
            "em_bp_mixture": MixtureInnovationKernel.init(
                8, rho=0.3, var=0.8, rng=rng_for("e24-train-init", M)),
            "em_bp_mdn": MDNKernel.init(
                int(cfg["mdn_components"]), int(cfg["mdn_hidden"]),
                rng_for("e24-train-mdn-init", M)),
        }
        for arm, k0 in kernels.items():
            log_K = k0.log_transition_matrix(grid)
            e = _timed(lambda: e_step_multi(grid, weights, log_K, groups), rep)
            stats = e_step_multi(grid, weights, log_K, groups)
            m = _timed(lambda: k0.m_step(stats, grid), rep)
            rows.append({
                "arm": arm, "family": "em_bp", "M": M, "n_sites": n_sites,
                "grid_size": int(cfg["grid_size"]),
                "e_step_sec": e["wall_median"], "m_step_sec": m["wall_median"],
                "iter_sec": e["wall_median"] + m["wall_median"],
                "m_step_fraction": m["wall_median"] / (e["wall_median"] + m["wall_median"]),
                "n_params": int(getattr(k0, "theta", np.zeros(0)).size),
                "n_repeat": rep,
                # net-only columns, kept present so every row carries the same schema
                # (write_csv takes its fieldnames from the first row)
                "train_steps": "", "train_sec": "", "train_sec_at_protocol": "",
            })
            print(f"  [E24 training] M={M:5d} {arm:16s} e_step={e['wall_median']:6.3f}s "
                  f"m_step={m['wall_median']:6.3f}s  m_step share={100 * rows[-1]['m_step_fraction']:.0f}%",
                  flush=True)

        # --- the network arms: seconds per gradient step -------------------------
        try:
            import torch  # noqa: F401

            from markov_score import nn_torch as nt
        except Exception as exc:
            print(f"  [E24 training] torch unavailable, network arms skipped ({exc})", flush=True)
            continue
        for arch, hp in NET_ARMS:
            cfg_t = nt.TrainConfig(steps=int(cfg["train_steps"]), batch_size=64, lr=1e-3,
                                   parameterization="eps", loss="dsm",
                                   checkpoints=(int(cfg["train_steps"]),),
                                   t_levels=tuple(float(t) for t in levels), device="cpu",
                                   seed=int(rng_for("e24-train-net", arch, M).integers(0, 2**31 - 1)))
            model = nt.build(arch, n_sites, **hp)
            t0 = time.perf_counter()
            try:
                nt.train(model, A, cfg_t)
            except Exception as exc:
                print(f"  [E24 training] {arch} M={M} failed: {exc}", flush=True)
                continue
            dt = time.perf_counter() - t0
            rows.append({
                "arm": arch, "family": "net", "M": M, "n_sites": n_sites, "grid_size": 0,
                "e_step_sec": float("nan"), "m_step_sec": float("nan"),
                "iter_sec": dt / int(cfg["train_steps"]),
                "m_step_fraction": float("nan"),
                "n_params": nt.n_params(model), "n_repeat": 1,
                "train_steps": int(cfg["train_steps"]), "train_sec": dt,
                "train_sec_at_protocol": dt / int(cfg["train_steps"]) * int(cfg["protocol_steps"]),
            })
            print(f"  [E24 training] M={M:5d} {arch:16s} {dt / int(cfg['train_steps']) * 1e3:6.2f} ms/step "
                  f"-> {rows[-1]['train_sec_at_protocol']:7.0f}s at {cfg['protocol_steps']} steps", flush=True)

    write_csv(out / "training.csv", rows)

    # The exponents that decide whether the mechanism is real.
    slopes = []
    for arm in sorted({r["arm"] for r in rows if r["family"] == "em_bp"}):
        sel = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["M"])
        for col, pred in (("e_step_sec", 1.0), ("m_step_sec", 0.0), ("iter_sec", "")):
            vals = [r[col] for r in sel]
            if len(sel) < 2 or not all(np.isfinite(v) and v > 0 for v in vals):
                continue
            s, s_top, n_top = _slopes([r["M"] for r in sel], vals)
            slopes.append({"arm": arm, "quantity": col, "slope_full": s, "slope_top": s_top,
                           "n_top": n_top, "predicted": pred, "n_points": len(sel)})
            print(f"  [E24 training slope] {arm:16s} {col:12s} full={s:+.3f} top={s_top:+.3f}"
                  + (f"  (predicted {pred:+.1f})" if pred != "" else ""), flush=True)
    write_csv(out / "training_slopes.csv", slopes)
    return rows


def part_capacity(cfg, out):
    """Does the M-step stay flat in M as the MODEL grows? The "small vs huge network" claim.

    WHY THIS PART EXISTS. Part `training` showed the M-step cost is flat in dataset size
    (slope +0.008 over a 64x range) while the E-step is linear. That is half the argument.
    The other half -- the half that turns it into "one small kernel network instead of one
    huge score network" -- is what happens when the MODEL grows:

        EM-BP           the kernel is evaluated on N_g grid points once per iteration, so
                        capacity should multiply the (data-independent) M-step term only
        score network   every parameter is touched by every example at every step, so
                        capacity multiplies against data x steps

    PREDECLARED, fixed before running: the M-step exponent in M stays ~0 at EVERY capacity,
    while the M-step cost itself grows with capacity. If the M-step exponent in M rises with
    capacity, the mechanism is leakier than claimed and the argument weakens -- which is the
    outcome worth knowing, since this is the claim the paper would lead with.
    """
    rows = []
    n_sites = int(cfg["n_sites"])
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    prior = LaplaceAR1(cfg["rho"])
    levels = list(cfg["levels"])
    rep = int(cfg["train_repeat"])

    from markov_score.em import e_step_multi
    from markov_score.kernels import MDNKernel, MixtureInnovationKernel

    def bundle(M):
        rng = rng_for("e24-cap", M)
        A = np.stack([prior.sample(rng, n_sites) for _ in range(M)])
        groups, per = [], max(1, M // len(levels))
        for j, t in enumerate(levels):
            sl = A[j * per:(j + 1) * per]
            if len(sl) == 0:
                continue
            al, de = alpha_delta(float(t))
            groups.append((al * sl + np.sqrt(de) * rng.standard_normal(sl.shape), al, de))
        return A, groups

    for M in cfg["cap_sizes"]:
        M = int(M)
        A, groups = bundle(M)
        for C in cfg["cap_components"]:
            k0 = MixtureInnovationKernel.init(int(C), rho=0.3, var=0.8,
                                              rng=rng_for("e24-cap-mix", M, C))
            log_K = k0.log_transition_matrix(grid)
            e = _timed(lambda: e_step_multi(grid, weights, log_K, groups), rep)
            stats = e_step_multi(grid, weights, log_K, groups)
            m = _timed(lambda: k0.m_step(stats, grid), rep)
            rows.append({"arm": "em_bp_mixture", "family": "em_bp", "capacity": int(C),
                         "capacity_axis": "components", "M": M, "n_params": int(k0.theta.size),
                         "e_step_sec": e["wall_median"], "m_step_sec": m["wall_median"],
                         "sec_per_unit": m["wall_median"], "n_repeat": rep})
        for H in cfg["cap_hidden"]:
            k0 = MDNKernel.init(int(cfg["mdn_components"]), int(H),
                                rng_for("e24-cap-mdn", M, H))
            log_K = k0.log_transition_matrix(grid)
            e = _timed(lambda: e_step_multi(grid, weights, log_K, groups), rep)
            stats = e_step_multi(grid, weights, log_K, groups)
            m = _timed(lambda: k0.m_step(stats, grid), rep)
            rows.append({"arm": "em_bp_mdn", "family": "em_bp", "capacity": int(H),
                         "capacity_axis": "hidden", "M": M, "n_params": int(k0.theta.size),
                         "e_step_sec": e["wall_median"], "m_step_sec": m["wall_median"],
                         "sec_per_unit": m["wall_median"], "n_repeat": rep})
            print(f"  [E24 capacity] M={M:5d} mdn hidden={H:4d} params={rows[-1]['n_params']:6d} "
                  f"m_step={m['wall_median']:.4f}s", flush=True)

        try:
            import torch  # noqa: F401

            from markov_score import nn_torch as nt
        except Exception:
            continue
        for d, layers in cfg["cap_transformer"]:
            hp = dict(d=int(d), heads=4, layers=int(layers))
            cfg_t = nt.TrainConfig(steps=int(cfg["cap_steps"]), batch_size=64, lr=1e-3,
                                   parameterization="eps", loss="dsm",
                                   checkpoints=(int(cfg["cap_steps"]),),
                                   t_levels=tuple(float(t) for t in levels), device="cpu",
                                   seed=int(rng_for("e24-cap-net", d, layers, M).integers(0, 2**31 - 1)))
            model = nt.build("transformer", n_sites, **hp)
            t0 = time.perf_counter()
            try:
                nt.train(model, A, cfg_t)
            except Exception as exc:
                print(f"  [E24 capacity] transformer {hp} M={M} failed: {exc}", flush=True)
                continue
            dt = time.perf_counter() - t0
            rows.append({"arm": "transformer", "family": "net", "capacity": int(d),
                         "capacity_axis": "d_model", "M": M, "n_params": nt.n_params(model),
                         "e_step_sec": float("nan"), "m_step_sec": float("nan"),
                         "sec_per_unit": dt / int(cfg["cap_steps"]), "n_repeat": 1})
            print(f"  [E24 capacity] M={M:5d} transformer d={d} l={layers} "
                  f"params={rows[-1]['n_params']:6d} {dt / int(cfg['cap_steps']) * 1e3:.2f} ms/step",
                  flush=True)

    write_csv(out / "capacity.csv", rows)

    # The predeclared test: M-step exponent in M, AT EACH CAPACITY.
    slopes = []
    for arm in sorted({r["arm"] for r in rows if r["family"] == "em_bp"}):
        for cap in sorted({r["capacity"] for r in rows if r["arm"] == arm}):
            sel = sorted((r for r in rows if r["arm"] == arm and r["capacity"] == cap),
                         key=lambda r: r["M"])
            vals = [r["m_step_sec"] for r in sel]
            if len(sel) < 2 or not all(np.isfinite(v) and v > 0 for v in vals):
                continue
            sl, sl_top, n_top = _slopes([r["M"] for r in sel], vals)
            slopes.append({"arm": arm, "capacity": cap, "quantity": "m_step_sec",
                           "slope_full": sl, "slope_top": sl_top, "predicted": 0.0,
                           "n_params": sel[-1]["n_params"], "n_points": len(sel)})
            print(f"  [E24 capacity slope] {arm:14s} cap={cap:4d} m_step slope in M = "
                  f"{sl:+.3f} (predicted 0)", flush=True)
    write_csv(out / "capacity_slopes.csv", slopes)
    return rows


PARTS = {"cost": part_cost, "scaling": part_scaling, "anatomy": part_anatomy,
         "training": part_training,
         "capacity": part_capacity}


def main():
    ap = experiment_parser("e24_inference_cost",
                           "wall-clock per posterior mean, and the scaling exponents in B, L and N_g")
    args = ap.parse_args()
    if args.list_parts:
        print("\n".join(PARTS))
        return
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    out = ensure_dir(Path(args.output_dir))
    for name, fn in select_parts(PARTS, args.only).items():
        fn(cfg, out)
        write_json(out / f"params_{name}.json",
                   {**provenance(cfg), "settings": cfg, "quick": bool(args.quick),
                    "frozen": FROZEN.as_dict(), "threads": _thread_info(),
                    "net_arms": [[a, str(h)] for a, h in NET_ARMS]})


if __name__ == "__main__":
    main()
