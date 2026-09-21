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

**Still open, not auditable from data yet (as of 19 Sep):** whether the estimator "leaves
the bound" at t ≥ 1.4 in E06 (E06c); whether the E07 Student-t ρ=0 cell moves under a
wider domain (E07b/E07c); the sample-complexity exponents (E12b). The first two landed on
20 September (section below); E12b is running.

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

## 20 September 2026 — the cluster harvest, the E14 void, and a second read of the paper

**Landed and frozen:** E06c (the E06 ladder extended to 8000: the ρ̂ departure at t ≥ 1.4
was partly budget — RMSE 0.0325 → 0.0215 at t = 1.4, 0.343 → 0.244 at t = 2, at-cap
0.88 → 0.19 and 0.75 → 0.56; nothing below t = 1 moves by more than 0.0001); E07b/E07c
(the Student-t pooled floor moves by 0.005 from (801, 8) to (801, 12) and by 0.0002 from
(801, 12) to (1601, 16); every other family by ≤ 0.001; the paper quotes Student-t from
E07c); E11b (the envelope at L = 64 and 128: worst drift on the operating grid
5.4e-2 → 5.8e-2 → 6.6e-2, a factor 1.22; frozen-configuration drift 9.2e-6, 7.2e-6,
4.5e-6). Macros learnBudget*, phaseWide*/phaseStudentMove*/phaseOtherMoveWide, gridLen*.

**E14 is void before 20 Sep 15:10 (N-22).** The driver re-created the random generator
inside its sampling comprehension, so every training chain was one draw. Found by running
E03's and E14's training functions on the same data (both ≈ 1.2 on E14's data, both
0.166–0.169 on E03's). The first diagnosis (Hutchinson's one-probe divergence) was wrong
and is recorded as wrong; the estimator change it prompted stays because it is E03's.
Fixed with an assertion; the corrected run (jobs 664280/664281, revision 81cf7ae) is
queued. The lesson is in N-22: a fitted parameter that contradicts the generating law is a
data check, and a driver should assert the size of its training set.

**Findings in what was said, from the compendium's line-by-line read (fixed in the
manuscript on 20 Sep):** the column-mass defect's mechanism (it is a boundary-column
truncation effect, flat in N_g, not a resolution effect; App. B); the M-step statistics
S_ab written as powers where the code uses S_00 = Σ Ξ u_l², S_01 = Σ Ξ u_k u_l,
S_11 = Σ Ξ u_k² (§4); "verified against finite differences to 1e-9 (App. C)" — the
verification lives in the tests at 1e-6 relative and against enumerated evidence, and
the text now says so; the initial law μ missing from the recursion (App. B, now
m⃗₁ ≡ μ); the "convergence asymmetry reported for this model" invoked without evidence
(removed); "depends on the data only through their covariance" (through the sample
covariance, whose expectation is that matrix; §2); "accurate to 1e-5 for every family"
qualified to the frozen correlation (§2); the M-step cost sentence in the conclusion made
precise. **Open from the same read:** E06's clean-data arm is the channel at t = 1e-3,
where the likelihood is about one grid cell wide, below the code's own three-points-per-
s.d. resolution rule; a comparison with the exact clean MLE (fit_clean) is run below.

**Second read, continued (20 Sep, from the compendium's diffusion and foundations
chapters; fixed in the manuscript):** the (3,1) and (2,2) terms of Λ₄ in App. A were
written with cumulants of a_i where the expansion of log E exp(s·a − b|a|²/2) has
cumulants of a_i² (the difference is a function of Σ₀ alone, so "Λ₄ − Λ₄ᴳ is linear in
κ₄" survives; the display is corrected and the identity stated); the "zero γ-mean"
sentence in the second-moment paragraph (false as a statement about the mean, since
He₃² contains He₂; replaced by the explicit α² factor); the heavy-tails sentence said the
e^{-νt} term is carried by |a| ≳ e^{2t} (the bulk boundary of the proof) where the
events of probability e^{-νt} have |a| of order e^{t}; "t* is an exact scale: L²(γ) iff
e^{2t} > λ_max − 1" conflated t* with ½ log(λ_max − 1) (rephrased); the corollary's local
minimax statement now carries the lim c → ∞; interface B's checkpoint rule (oracle
validation, as A) is stated; the generation paragraph no longer calls the exact arm's
cross-entropy difference a "floor" and says the residual kurtosis is measured at t_min;
the generalised-Gaussian range ends at excess kurtosis −1 (β = 6), not −1.1; "differ only
through cumulants of order four and higher" now says the families are symmetric; the
moment estimators' "unbiased" restricted to the marginal statistics; the regime
paragraph's kurtosis formula attributed to the stationary marginal.

**Clean-data arm of E06 (the channel at t = 1e-3, one grid cell of likelihood width):**
compared with the exact clean maximum-likelihood fit (`fit_clean`, the same seeds, data,
initialisation and iteration count) on seeds 0–3: |Δρ̂| ≤ 0.0012 and |Δκ̂₄| ≤ 0.034
(values 0.8543/0.8548, 0.8516/0.8526, 0.8478/0.8477, 0.8512/0.8524 for ρ̂; 2.689/2.704,
2.746/2.780, 2.855/2.860, 3.122/3.135 for κ̂₄). The "optimiser's own floor" sentence
stands; the check is not frozen and is recorded here only.

**Open, code-level (not in the paper's text):** `gen_metrics.covariance_error` compares
the t_min sample's covariance with Σ₀ rather than Σ_{t_min}, a relative floor of about
0.035 that even the exact arm cannot go below (the generation table's covariance column
must be read against the exact arm's own value, which it is); the `gen_metrics.py`
docstring's claim that the lag fourth cumulant has "a known nonzero value" is wrong (it is
zero for the true law at every t_min; the column is a check for spurious dependence).

**21 Sep: the archival capacity sweep is void (N-23).** exp_32 trained every arm on one
chain (same defect as N-22, present at the frozen commit). The paper's App. F paragraph and
the Section 5 sentence that attributed the earlier conclusion to a stopping rule are
rewritten: the sweep is void for a prior reason and the stopping asymmetry is EM on a
single chain. No number in the paper comes from exp_32.

**21 Sep: the manuscript gate had been vacuous for the E-series fragments (tool finding,
cluster-ops session, fixed in d6596cc).** `check_manuscript.py` regenerated fragments in a
scratch tree whose `tools/` was a symlink, so every generator that locates the repository
through `Path(__file__).resolve()` (all E-series and audit generators) wrote into the real
`paper/sections/` and the scratch diff compared nothing; only the legacy cwd-relative
generators were ever diffed. `tools/` is now copied into the scratch tree; the gate fails on a
hand-perturbed digit and passes on the tree as committed, so no drift had accumulated in any
fragment shipped since 18 Sep. Every "make check green" statement before this date should be
read as "typed literals, references, freeze and build green; regenerate-and-diff green for
the legacy fragments only". Four more generators are registered with the gate
(grid_envelope, regime_numbers, sample_complexity, regime_baselines).
