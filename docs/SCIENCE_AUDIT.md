# Science audit

An audit is a re-derivation, not a re-reading. Every entry below names a number the paper
or the report states, the frozen file it came from, and the result of computing it a second
way — `tools/audit_claims.py`, plain pandas on the CSVs, no shared helpers with the
generators. The tool runs in `make check` and in CI; a disagreement fails the build.

## 18 September 2026

**Scope.** Every headline macro in `paper/sections/*-numbers.tex` that the body cites:
capacity (E02a), baselines under one recipe, the information-matched comparison and the
length study (E03), generation (E05), misspecification break-evens (E08), the regime sweep
(E13), the quadrature envelope (E11), learning through noise (E06), the phase ladder (E07),
sample complexity (E12). 55 quantities. **Result: all agree** with their independent
recomputation to the macro's own rounding, including the headline paired factor
7.3 [6.2, 8.7] (recomputed 7.344 [6.17, 8.73] with a fresh bootstrap stream) and the phase
floors 0.295 / 0.168 / 0.085, the last re-pooled from the per-level rows rather than read
from the pooled ones.

**Three findings, all in what was *said*, none in what was *computed*.**

1. **A shipped table disagreed with the paragraph beside it.** The PDF sent on the morning
   of 18 September carried a generation table (Table 4) with numbers from the superseded
   four-seed E05 run — `exact −0.072, em–bp −0.431` at M=128 — while the prose next to it
   quoted the frozen sixteen-seed macros (`0.070`, `0.164`). Cause: `tab-generation.tex`
   had not been regenerated after the sixteen-seed fetch, and the local pre-send check was
   run without `--regenerate --diff`. Fix: regeneration and diff are now the default in
   `tools/check_manuscript.py` and cannot be skipped by omission; CI already ran them.
   Corrected table: `paper/sections/tab-generation.tex`.

2. **The meeting report contradicted itself.** `reports/em-chapters-2026-09-18.tex`
   Summary item 5 quoted the four-seed generation reading (kurtosis 1.887 vs 1.896, gap
   0.022, "if anything the smaller gap") while its own §7 table carried the sixteen-seed
   result (deficit +0.09 [+0.00, +0.19], kurtosis 1.80 vs 1.94). The summary now states the
   sixteen-seed numbers and says the four-seed reading was overturned.

3. **A confidence interval was reported from a Monte Carlo draw that straddled the
   prediction it tested.** The covariance-coordinate sample-complexity slope (E12) was
   stated as 5.40 [2.13, 7.57] "excluding 8". That interval came from a two-point
   estimator and one bootstrap stream; the pre-declared OLS-over-levels estimator gave
   [2.24, 8.20] on one stream and [2.27, 7.73] on another at B = 4000 — the upper limit
   crossed 8 depending on the seed. With eight seeds the bootstrap distribution has 6 435
   distinct resamples and is enumerated exactly: **[2.28, 7.72]**, which excludes 8 and
   contains 4. The conclusion survives; the number and the estimator were wrong.
   `tools/make_sample_complexity.py` enumerates the exact distribution whenever feasible
   and reports the Monte Carlo alternative only above that limit. Corrected in
   `docs/NEGATIVE_RESULTS.md`, `docs/CLAIM_EVIDENCE_MATRIX.md`, and the report.

**Blocks of the audit tool itself that were wrong on first run, and were fixed before
any conclusion was drawn from them:** the E07 block read a boolean weight column as the
floor; the E03 matched block compared the ISM arm where the paper's headline is the
one-corruption arm. Both are corrected in the tool; neither touched the paper.

**Still open, not auditable from data yet:** whether the estimator "leaves the bound" at
t ≥ 1.4 in E06 (budget-confounded, E06c queued); whether the E07 Student-t ρ=0 cell moves
under a wider domain (E07b/E07c queued); the sample-complexity exponents (E12b queued).

## How to run

    make check          # regenerate + diff, typed literals, references, frozen sources,
                        # build + Overfull gate, and this audit
    python tools/audit_claims.py

A number that reaches a PDF sent to anyone has passed both.

## 18–19 September 2026 — the full audit (reports/audit-2026-09-18.pdf)

**Scope.** Both primary documents (the ICML manuscript and the 18 September meeting
report) audited as a hostile ICML reviewer, a statistician and a numerical analyst; every
theorem re-derived; every experiment re-read from its frozen CSV; seven new registered
computations (E15–E21) run on the laptop to settle what the re-reading raised.

**Findings in what was computed:** none. Every frozen number agrees with its
recomputation. The closure-constant precision (N-18) is a precision statement, not a
disagreement.

**Findings in what was said** (all corrected in the manuscript on branch
`claude/audit-2026-09-18`): Theorem 3's hypothesis versus its proof (exponential moment
now assumed; the tail bound corrected; a general-order theorem and a local-asymptotic-
minimax corollary added); the "information-matched 7.3×" label (N-15); the "attains the
CRB" claim (N-14); the Student-t departure (N-16); the belief-mass sentence (N-17); the
constant's precision (N-18, N-19); the E12b analysis plan (N-20); the "confirms e^{4t}"
sentences in §1 and the contributions; the dangling `\cref{thm:gap}`; the provenance
sentence on under-resolved fits (flagged, not refused); the E08 network named (radius 2);
the seed count per experiment; the novelty paragraph.

**New results now in the paper:** the closed-form misspecification bias and its order in ε
(E17; reproduces the M = 2048 errors to ≤ 0.011 with the lag-one projection); the
fourth-order Edgeworth correction as the content of the closure gap (E16); the moment
estimators as the rate-correct, inefficient competitor (E19); the exact radius-r locality
floor (E15); the wide-domain closure panel with the heavy-tailed regime (E18); the
oracle-free selection control (E20); the information curve at ρ = 0.5 (E21).

**Tools.** `tools/make_audit_numbers.py` emits every new number as a macro
(`paper/sections/audit-numbers.tex`); `tools/make_baseline_numbers.py` now emits the ISM
arm as the matched headline and the fixed-corruption arm separately, and the networks'
at-cap rates.
