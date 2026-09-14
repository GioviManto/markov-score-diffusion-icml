# exp_32_capacity — first run (em_cap=400) is confounded; do not cite it

Job chain `635645→635646→637047` (sweep) + `635648` (contrast), commit
`99acb49`, clean provenance, 160/160 cells complete. The predeclared C=16 vs
C=8 contrast came back **resolved, not equivalent, in the direction "C=16 is
worse"** at both N=128 and N=512, on both held-out log-evidence per edge and
schedule-level score risk. See `capacity_contrasts.csv` in this directory for
the exact numbers.

## Why this run cannot be trusted as-is

Every one of the 320 fits in this sweep (16 seeds x 2 sizes x 5 components x
2 inits) hit the `em_cap=400` iteration ceiling without satisfying the formal
`tol=1e-9` convergence criterion — `em_converged` is `False` in every row.

Checked directly (not just inferred from the flag): at iteration 400 on a
representative fit, C=8's held-out evidence moved 0.025 nats over the last
100 iterations (per-edge gain 9.9e-9 — a hair past the tolerance, essentially
flat). C=16's moved **0.77 nats over the same 100 iterations** (per-edge gain
6.9e-8, thirty times larger). C=16 was still climbing substantially when the
cap cut it off; C=8 was not.

This is the capacity-vs-convergence-rate confound the whole review is about,
recurring in a different place. It is not the same failure as the withdrawn
shape-based capacity claim (that one was about the fitted innovation SHAPE
needing ~2000 iterations while evidence settles by ~40; this one is about
held-out EVIDENCE itself not yet having settled for the higher-capacity model
at a shared iteration cap) — but the shape of the mistake is identical: a
shared budget that is generous enough for the simpler model and not the
richer one, read as a property of the models rather than of the budget.

## What is NOT in question

- The sweep's own provenance and completeness: clean, 160/160, verified.
- That the extra capacity is being used: effective component count rises from
  6.85 (C=8) to 13.33 (C=16) at both sizes, and the narrowest fitted
  component stays well-resolved (s_min/h 3.6-4.4, comfortably above the ~2
  floor) at both capacities. This is not an unused-capacity or
  under-resolved-component artefact.
- The statistical design itself: 16 paired seeds, validation-selected
  initialisation, predeclared equivalence region, paired bootstrap. All of
  that is sound; what it was fed was under-converged at C=16.

## The fix

`em_cap` raised from 400 to 1200 in
`experiments/exp_32_capacity_equivalence.py`, empirically confirmed rather
than just matched to convention: a representative C=16 fit run to 2000
iterations shows gain/edge falling from 6.9e-8 at 400 to ~6e-9 by 800, then
PLATEAUING there through 2000 (6.08e-9 @1200, 5.92e-9 @1600, 6.09e-9 @2000).
It never crosses the strict tol=1e-9 threshold at any practical iteration
count, but 800-1200 already captures essentially all of the real movement.
`em_converged` will likely still read `False` for C=16 even at the new cap,
and that is expected: the field is diagnostic-only in this experiment, never
used to gate the comparison.

## Second run (em_cap=1200): parallelised, and one premature contrast to ignore

Commit `35578c9`. To finish faster, the sweep was split into three parallel
seed-range lanes writing to separate output directories rather than one
sequential chain:

    exp_32_capacity     seeds 0-6   (continuation of the original lane)
    exp_32_capacity_b   seeds 7-11
    exp_32_capacity_c   seeds 12-15

**A contrast job (`639099`) fired prematurely** on `exp_32_capacity` alone,
covering only seeds 0-6 (~63/160 cells): cancelling the old sequential
chain's not-yet-started shards satisfied its `afterany` dependency, since
`afterany` fires on any terminal state of its target including cancellation
— it has no way to know the intended meaning was "wait for the real work,"
not "wait for something to happen to job 639098." Its
`capacity_contrasts.csv` in this directory is **from that premature,
incomplete run and must not be cited**. Notably the N=512 confidence interval
there is already visibly wider than in earlier partial checks, which is
itself the demonstration of why the full seed count matters.

**Do not** report anything from that file until the contrast has been
regenerated after all three lanes finish and their `capacity_equivalence.csv`
files are concatenated into one. That merge + rerun has not happened yet as
of this note.

To make accidental use harder rather than merely discouraged, the file has
been renamed on the cluster:

    outputs/35578c9-20260827T153657Z/exp_32_capacity/
        DO_NOT_CITE_premature_contrasts_job639099.csv
        DO_NOT_CITE_premature_params_contrast_job639099.json

A merge script globbing `capacity_contrasts.csv` now finds nothing rather
than finding a plausible-looking 16-row file computed on 40% of the seeds.
Renamed rather than deleted: it is the evidence for what went wrong, and
the wider CI in that N=512 interval is worth keeping as a demonstration of
why the full seed count matters.

## Progress as of 30 Aug 2026

Deployed commit `35578c9`, verified on the cluster: `params_sweep.json`
records `em_cap = 1200`, so the corrected cap is what is running.

    lane                seeds   target cells   done
    exp_32_capacity     0-6     70             70   COMPLETE
    exp_32_capacity_b   7-11    50             13   running (job 641517)
    exp_32_capacity_c   12-15   40             13   running (job 641519)

Lane a is finished and correct. Nothing may be concluded from it alone ---
that is exactly the mistake job 639099 made.

`exp_31_confirm` is the more important of the two reruns, since it is the
structured-baseline measurement the external review identifies as the
outstanding one, and **it has not started**: all six of its shards sit
PENDING behind priority and node-drain reasons. The 114 rows in
`outputs/569a67c-.../exp_31_confirm/` are from the earlier partial run and
are not the corrected protocol.

**Do not** report "capacity is not equivalent past C=8" from either the
em_cap=400 run or the incomplete em_cap=1200 partial run. **Do not** report
"capacity saturates by C=8" either — that claim is still what was withdrawn
from the thesis, and neither run restores it. The honest state until the
complete, merged, correctly-gated run lands is: unresolved.
