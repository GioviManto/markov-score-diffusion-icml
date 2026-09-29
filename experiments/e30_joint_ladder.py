#!/usr/bin/env python
"""E30 — one ladder, one set of bundles: the kernel arms scored where the score nets are.

The gap this closes. The paper's spine is "a small kernel inside exact BP instead of a large
score network". Three comparisons support it and only two of them were ever measured on
shared data:

    em_bp vs mdn                E25, on `exp34-*` bundles                      MEASURED
    nets vs em_bp(E02a frozen)  E28/E29, on `exp31-*` bundles                  MEASURED
    mdn vs nets                 -- nothing                                     NOT MEASURED

E25's registered hypothesis is precisely about that third row: the MDN kernel "sits between
the unstructured networks and the mixture-innovation EM-BP on the information ladder". It has
never been tested, and it CANNOT be tested by reading E25's numbers next to E29's, for a
reason stronger than the fit difference recorded in E25.json on 28 Sep: `exp_34` draws its
data under the tags `exp34-train/val/test` while E03 draws under `exp31-train/val/test`.
Different tags are different random streams, so the two experiments are scored on DIFFERENT
DATA, not merely on the same data by different fits. Their risks are not comparable at any
precision and putting them in one table would be an arithmetic error, not a rounding one.

E30 therefore refits BOTH kernel arms on E03's bundles, with E03's corruption interface and
E03's estimand, so that every arm in the project sits on one axis. Nothing here is copied
from E03: `_cell_common`, `_bundle`, `_sq`, `_risk` and `_one_view_groups` are imported, so
the bundles are the same objects the nets were scored on rather than a reconstruction that
happens to agree.

Part `verify` is the gate and it runs on the same data the ladder does: E30's own `sq_ref`
per (seed, n_chains, t) must equal E28's and E29's EXACTLY -- 0.0, not a tolerance -- since
sq_ref depends only on the test bundle and the schedule. If it does not, the bundles differ
and the joint ladder is void; the part exits non-zero and says so. This is the check whose
absence let E28's first run produce a well-formed table of a different estimand (N-26).

PREDECLARED (registry E30.json), written before the confirmatory run:

  P1  mdn beats every score-net architecture at its best rung, at every M. This is E25's
      registered "sits between" hypothesis, finally testable. FALSIFIED by any architecture
      whose best rung beats mdn at any M.
      NOTE, left in deliberately: reading E25's mdn (0.1461 at M=128, exp34 bundles) beside
      E29's best net (0.1080, exp31 bundles) HINTS that this is false. That arithmetic is
      invalid -- it is the very cross-bundle comparison this experiment exists to replace --
      so it does not license editing the prediction. It is recorded so that a confirmation
      cannot later be read as having been the obvious outcome.
  P2  em_bp beats mdn at every M, replicating E25's 64/64 on different data. If it does not
      replicate, E25's answer to Q6 was bundle-specific and Q6 reopens.
  P3  the ordering is em_bp < mdn < best net at every M. P1 and P2 are its two halves; P3
      fails if either does.
  P4  em_bp refitted on exp31 lands within 10% of E28's frozen E02a kernel evaluated on the
      same bundles. A larger gap says the E02a fit and a fresh fit are not interchangeable,
      which would matter for every E28 and E29 ratio already reported.
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
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.em import fit_em  # noqa: E402
from markov_score.kernels import MDNKernel, MixtureInnovationKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

# E03 is imported, not copied: the same bundle objects the nets were scored on.
import e03_baselines as e03  # noqa: E402

N_SITES = FROZEN.n_sites
T_SCHEDULE = tuple(FROZEN.t_grid)

SETTINGS = dict(
    seeds=PROTOCOL.seeds, sizes=(32, 128, 512, 2048),
    n_val=PROTOCOL.n_val, n_test=PROTOCOL.n_test,
    grid_size=FROZEN.n_grid, half_width=FROZEN.half_width,
    arms=("em_bp", "mdn"),
    em_components=8,
    mdn_components=4, mdn_hidden=32,
    # MATCHED, both to 1200. E25's first pass capped the MDN at 400 and the cap bound the
    # result in 14/16 cells at M = 2048; the matched rerun is what made that answer readable.
    em_checkpoints=(10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200),
    mdn_checkpoints=(10, 20, 40, 60, 80, 120, 160, 220, 300, 400, 600, 800, 1200),
    torch_threads=1,
)
QUICK = dict(seeds=(0,), sizes=(32,), n_val=64, n_test=64, grid_size=201,
             em_checkpoints=(5, 10, 20), mdn_checkpoints=(5, 10, 20), arms=("em_bp",))


def _fit_and_select(cfg, kernel0, groups, grid, weights, val, test, ckpts):
    """EM through exact BP, validation-selected, scored with E03's estimand."""
    _, trace, saved = fit_em(kernel0, grid, weights, groups,
                             n_iters=max(ckpts), tol=FROZEN.em_loglik_tol, checkpoints=set(ckpts))

    def _pairs(kernel, bundle):
        log_k = kernel.log_transition_matrix(grid)
        out = []
        for t in T_SCHEDULE:
            X, m_ref = bundle[t]
            al, de = alpha_delta(t)
            m, _ = grid_bp_batch(grid, weights, log_k, X, al, de)
            # e03._sq, not a local copy: same score-space definition, same region slice.
            out.append(e03._sq(m, m_ref, X, t, e03.REGIONS["all"]))
        return out

    best, best_risk = None, np.inf
    for it in sorted(saved):
        r = e03._risk(_pairs(saved[it], val))
        if r < best_risk:
            best, best_risk = it, r
    return saved[best], _pairs(saved[best], test), {
        "checkpoint": best, "at_cap": int(best == max(saved)),
        "n_params": int(len(saved[best].theta)),
        "val_risk": best_risk,
        "converged": bool(trace.converged), "stop_reason": trace.stop_reason,
    }


def _cell(args):
    cfg, seed, n, arm = args
    grid, weights = make_grid(cfg["half_width"], cfg["grid_size"])
    A, val, test, base = e03._cell_common(cfg, seed, n)
    # E03's own corruption interface, so the kernel arms train on the SAME corrupted data
    # the score nets trained on -- drawn under `exp31-em-split` and `exp31-em-noise`, not
    # redrawn here. E03 hands the nets (X, t) pairs and fit_em wants (X, alpha, delta);
    # alpha_delta(t) is that change of variable exactly, so the sample is untouched.
    groups = [(X, *alpha_delta(t)) for X, t in e03._one_view_groups(A, seed)]

    if arm == "em_bp":
        # rho=0.3, var=0.8 are E25's initialisation, kept identical so that a difference
        # between E25's em_bp and E30's is attributable to the bundles and not to the start.
        k0 = MixtureInnovationKernel.init(cfg["em_components"], rho=0.3, var=0.8,
                                          rng=rng_for("e30-em-init", seed, n))
        ck = cfg["em_checkpoints"]
    else:
        k0 = MDNKernel.init(cfg["mdn_components"], cfg["mdn_hidden"],
                            rng_for("e30-mdn-init", seed, n))
        ck = cfg["mdn_checkpoints"]

    _, pairs, info = _fit_and_select(cfg, k0, groups, grid, weights, val, test, ck)
    pooled = e03._risk(pairs)
    rows = []
    for t, (e, r) in zip(T_SCHEDULE, pairs):
        rows.append({**base, "arm": arm, "t": t, "sq_err": e, "sq_ref": r,
                     "risk_level": float(np.sqrt(e / r)) if r > 0 else np.nan,
                     "risk_pooled": pooled, **info})
    print(f"  [E30] seed={seed} n={n:>5} {arm:6s} pooled={pooled:.4f} "
          f"ck={info['checkpoint']} at_cap={info['at_cap']} {info['stop_reason']}", flush=True)
    return rows


def part_ladder(cfg, out, workers):
    cells = [(cfg, s, n, a) for s in cfg["seeds"] for n in cfg["sizes"] for a in cfg["arms"]]
    rows = run_cells(cells, _cell, workers, out, "joint.csv")
    write_csv(out / "joint.csv", rows)


# sq_ref depends only on the test bundle and the schedule, so on the same bundles it agrees to
# float64 round-off; the comparison goes through a CSV, so it does NOT agree bit-for-bit. The
# first version of this gate demanded exactly 0.0 and failed on correct data at 5.8e-11
# absolute -- 4e-17 RELATIVE, below machine epsilon. Measured contrast, which is what sets the
# threshold: same bundles 4e-17 relative; a deliberately different bundle (grid 201, n_test 64)
# 2.4e+05 absolute, about 0.17 relative. Fifteen orders of magnitude separate them, so 1e-12
# cannot be met by different data and cannot be missed by the same data.
SQ_REF_RTOL = 1e-12


def part_verify(cfg, out):
    """THE GATE. If E30 is not on E28/E29's bundles, the joint ladder compares different data
    and must not be reported. Verified by exit code both ways before first use: exit 1 on a
    grid-201/n_test-64 bundle, exit 0 on the real one."""
    root = Path(__file__).resolve().parents[1]
    mine = pd.read_csv(out / "joint.csv")
    checked, worst, missing = 0, 0.0, []
    for name, path, cols in (("E28", root / "outputs/hpc_E28/per_level.csv", ("seed", "n_chains", "t")),
                             ("E29", root / "outputs/hpc_E29/ladder.csv", ("seed", "n_chains", "t"))):
        if not path.is_file():
            missing.append(name)
            continue
        other = pd.read_csv(path)
        ref = other.groupby(list(cols))["sq_ref"].first()
        m = mine.groupby(list(cols))["sq_ref"].first()
        common = ref.index.intersection(m.index)
        if len(common) == 0:
            missing.append(f"{name} (no overlapping cells)")
            continue
        d = ((ref.loc[common] - m.loc[common]).abs() / ref.loc[common].abs()).max()
        worst = max(worst, float(d))
        checked += len(common)
        print(f"[verify] {name}: {len(common)} overlapping (seed, n, t) cells, "
              f"max RELATIVE |dsq_ref| = {d:.3e}")
    write_json(out / "verify.json", {"checked": checked, "worst_rel_sq_ref_diff": worst,
                                     "rtol": SQ_REF_RTOL, "missing": missing})
    # A DETECTED mismatch outranks a missing comparison: exiting 2 ("cannot verify") when a
    # real disagreement has already been measured would report the weaker fact and bury the
    # stronger one. Check what was measured first.
    if worst > SQ_REF_RTOL:
        print(f"[verify] FAIL — sq_ref differs by {worst:.3e} relative over {checked} cells "
              f"(rtol {SQ_REF_RTOL:.0e}); E30 is NOT on E28/E29's bundles and the joint ladder "
              f"is void", flush=True)
        raise SystemExit(1)
    if missing:
        print(f"[verify] CANNOT VERIFY against {missing}: fetch those outputs first", flush=True)
        raise SystemExit(2)
    print(f"[verify] OK — {checked} cells, sq_ref agrees to {worst:.3e} relative "
          f"(rtol {SQ_REF_RTOL:.0e}): one ladder, one estimand",
          flush=True)


def main() -> None:
    parser = experiment_parser("e30_joint_ladder",
                               "kernel arms refitted on E03's bundles so mdn, em_bp and the "
                               "score nets sit on one ladder")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)

    parts = {"ladder": ("fit em_bp and mdn on E03's bundles",
                        lambda c, o: part_ladder(c, o, args.workers)),
             "verify": ("sq_ref identical to E28/E29", part_verify)}
    if args.list_parts:
        print("\n".join(parts)); return
    selected = select_parts(parts, args.only)
    # `verify` compares against outputs/hpc_E28 and outputs/hpc_E29, which are LOCALLY FETCHED
    # artefacts and are export-ignored from the deploy archive. On the cluster they cannot
    # exist, so a gate that runs there can only ever say "cannot verify" -- which is what the
    # first confirmatory run did: four jobs completed all 128 cells and then exited 2 on a
    # comparison that was impossible by construction (N-30). It is therefore dropped from a
    # DEFAULT run and must be asked for explicitly, which is only sensible after a fetch. It
    # is NOT made to pass when the references are missing: that would turn it into a gate that
    # covers nothing, which is the failure this project keeps rediscovering.
    if "verify" in selected and not args.only:
        root = Path(__file__).resolve().parents[1]
        if not (root / "outputs/hpc_E28/per_level.csv").is_file():
            print("[verify] SKIPPED in a default run: E28/E29 outputs are local, fetched "
                  "artefacts and are absent here. Run `--only verify` after "
                  "`hpc/icml_fetch.sh`; it is a post-fetch gate, not a job step.", flush=True)
            selected = {k: v for k, v in selected.items() if k != "verify"}
    out = ensure_dir(args.output_dir)
    tag = "_".join(selected) if args.only else "all"
    resolved = {"n_sites": N_SITES, "t_schedule": list(T_SCHEDULE), "quick": args.quick,
                "parts": list(selected), "overrides": args.set, **cfg}
    write_json(out / f"params_{tag}.json", {**resolved, **provenance(resolved)})
    for name, (label, fn) in selected.items():
        print(f"[{name}] {label} ...", flush=True)
        fn(cfg, out)


if __name__ == "__main__":
    main()
