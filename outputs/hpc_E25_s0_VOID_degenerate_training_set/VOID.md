# VOID — every E25 run of 25–26 September 2026

These four output directories (jobs 669323, 669324, 669325, 669326; ~30 CPU-hours) are
**void and must not be read as results**, including by a future generator or a merge step.

## Why

`experiments/exp_34_information_ladder.py` built its training set as

    A = np.stack([prior.sample(rng_for("exp34-train", seed, n_chains), N_SITES)
                  for _ in range(n_chains)])

`rng_for()` returns a fresh generator seeded from its arguments, and the loop variable `_`
is not one of them, so it was re-seeded identically on every iteration. **Every row of `A`
is the same chain.** Measured directly: 1 unique row out of 8. The "training set of M
chains" was one sequence replicated M times, for every M and every seed.

This is the same defect as amendment **N-22**, which voided E14's first runs on 20
September. It was fixed then in `e14_regime_baselines.py`, and in `e02_capacity.py` — where
the broken line was left behind as dead code (`... if False else None`) instead of being
deleted — but it was never swept for elsewhere. It was therefore still present in
`exp_32`, `exp_33` and `exp_34`, none of which had ever been executed, so nothing failed
until E25 ran.

## How it showed up in the numbers, before the cause was known

- both arms **flat in M** (em_bp 0.288 / 0.228 / 0.305 / 0.314 at M = 32 … 2048), which is
  what replicating one chain predicts: more copies carry no more information;
- `em_bp` test risk 0.23–0.31, i.e. **worse than this project's own LMMSE floor of 0.168**,
  where the headline reports 0.018;
- the validation-selected checkpoint was the **first or second rung** of a ladder reaching
  1200 — MDN selected iteration 5 in all 64 cells — because a model overfits one sequence
  immediately;
- `stop_reason = censored` in 63 of 64 `em_bp` cells.

Any one of those should stop a reader; together they are unambiguous.

## What was done

- `exp_34`, `exp_32` and `exp_33` fixed (generator hoisted out of the comprehension);
- the dead broken line in `e02_capacity.py` deleted, so it cannot be copied again;
- `tests/test_no_degenerate_bundles.py` added: an AST check over `experiments/`, `src/` and
  `tools/` that fails on any comprehension re-seeding `rng_for` per iteration. It found
  `exp_33`, which a hand-written grep had missed;
- E25 resubmitted from the fixed driver.

## Open question this raises, NOT yet answered

`exp_32_capacity_equivalence.py` carried the same defect and is cited by
`docs/CLAIM_EVIDENCE_MATRIX.md`, `docs/SCIENCE_AUDIT.md`, `docs/NEGATIVE_RESULTS.md` and by
`tools/make_tab_capacity.py`. It has no outputs in this repository, so the numbers those
documents cite came from the archival tree rather than from here. **Whether the "no mixture
capacity improves on a single Gaussian innovation" conclusion (audit C12, and the void
thesis capacity sweep of Ch. 10) rests on a degenerate training set is now an open
question** — a training set of one replicated chain would suppress exactly the benefit a
richer kernel is supposed to show. This needs checking against the archival run's own code
before that conclusion is repeated anywhere.
