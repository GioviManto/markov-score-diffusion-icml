# Negative results and withdrawn readings

Kept deliberately. Each entry says what was hypothesised, what was measured, and what
it now means. Entries from the archival record are included so that the history is in
one place.

## Inherited from the archival record

**N-A1. "Capacity does not help": withdrawn as an artefact (15 Sep 2026).** The
exp_32 sweep (C ∈ {1, 2, 4, 8, 16}, M ∈ {128, 512}, 16 seeds) reported the single
Gaussian innovation as best on held-out log-evidence and score risk. The audit
established that C = 1 alone stopped at the likelihood tolerance near the validation
optimum (31/32 cells, median 71 iterations) while every C ≥ 4 fit ran to a fixed 1200
iterations, past the optimum by ~15× — a regime that costs 1.1–1.9× in risk by exp_07's
own measurement. The reading is withdrawn; H2/E02 re-runs it with the checkpoint
discipline of exp_31. The prior manuscript's Appendix L sentence attributing the margin
to covariance is retired (`CURRENT_STATE_AUDIT.md` N1, N6).

**N-A2. Reverse-generation comparison (2026-08).** Withdrawn by the previous authors
because the fits used a fixed 40-iteration budget. The converged rerun
(`legacy-unfrozen/exp_16_cluster/components_converged_*`) shows the EM–BP arm
reproducing the target innovation kurtosis at every C; it remains uncertified (dirty
tree) and its network arms unreconverged. Re-run as E05.

**N-A3. The M = 8192 headline row.** Withdrawn by the previous authors: produced from
a tree with three uncommitted files on the code path. Not restored.

**N-A4. Ratio 9–14× and 7.0–17.6×.** Retired estimands (selected-vs-pinned budgets;
cellwise-mean ratio). The pooled 6.6–10.9× replaces them; the cellwise mean is kept in
the aggregation appendix as the larger, disclosed alternative.

**N-A5. "Long-range architectures lost the screen for reasons other than capacity."**
True as far as parameter counts go, but the audit found the screen's optimisation
protocol unequal (one noise level per step for the long-range arms vs twelve for the
window head; eight restarts vs one). Not a negative result about the architectures;
a negative result about the screen. Re-run as E03.

**N-A6. Rung-1 ρ slope −0.27.** Predicted −0.5; unexplained in the manuscript. Kept
open (A2 in the audit); H6/E06 may explain it as an effective-sample-size effect of
the one-view protocol.

## New (this project)

*(none yet — entries are added the day an experiment returns a result against its
hypothesis)*

## The networks do NOT stop at the second-order floor (15 September 2026)

**The claim I was about to make.** The draft abstract and introduction said the learned
denoisers "sit at or above the second-order floor of the pooled score error, the structured
estimator far below it". That reading came from the archival comparison, where the networks
were screened unequally and evaluated mostly at small sample sizes.

**What the full screen shows.** On the development seeds, bulk region (`outputs/E03/screening.csv`,
68 configurations x 4 seeds x 3 restarts x 32k steps):

| M | best network | which | LMMSE floor | EM-BP (C = 8) |
|---|---|---|---|---|
| 32 | 0.147 | window head | 0.169 | 0.100 |
| 2048 | 0.056 | bidirectional GRU | 0.169 | 0.023 |

At M = 2048 every architecture except the fully connected one is far **below** the floor
(0.056-0.071 against 0.169). Given enough clean sequences and fresh corruptions at every
gradient step, a neural denoiser plainly does learn fourth-order structure. The claim as
drafted is false and has been removed from the abstract and the introduction.

**What survives, and is stronger.** At equal sequence count the structured estimator is still
ahead -- 0.023 against 0.056 at M = 2048, 0.100 against 0.147 at M = 32 -- and it is ahead
while seeing strictly less: one noised view per sequence, never a clean chain, against the
networks' clean targets with a fresh corruption at every step. The honest statement is about
the information interface, not about a floor the networks cannot cross. E03's
information-matched arm measures exactly that gap.

**A second finding in the same table.** Which architecture wins depends on the sample size:
the radius-4 window head is best at M = 32 (0.147 against the bidirectional GRU's 0.171) and
worst of the structured five at M = 2048 (0.071 against 0.056). Locality is a good prior when
data is scarce and a binding constraint when it is not. The confirmatory table reports both
sizes for this reason.

**Status.** Development seeds; the confirmatory table (E03 confirm, running) replaces these
numbers. The direction is a factor of three and is not a seed effect.

## The asymptotic exponents are not observable in this laboratory (16 September 2026)

**What was predicted.** The sample-complexity corollary says the error of an estimate of the
correlation from data observed at level t grows like e^{2t} and that of the innovation's fourth
cumulant like e^{4t}, so their ratio grows like e^{2t}.

**Why E06 could not test it.** At the frozen rho = 0.85 the crossover sits at t_* = 1.17, and the
shape estimate saturates at t = 1: its RMSE reaches 3, the whole distance from a Gaussian
innovation to the truth, because the Cramer-Rao bound for that coordinate has already crossed the
coordinate's own range at t = 0.98. There is no window in which the estimate is both asymptotic and
informative. Fitting slopes anyway would have returned 3.64 and 0.09 against predictions of 2 and 4
-- the second being nothing but the saturation plateau -- so the analysis reports no slope and says
why.

**E06b, registered to fix this, does not fix it.** Moving to rho = 0.5 (t_* = 0.54) with M = 2048
and 16 seeds opens a window, but only a three-level one: t in [0.6, 0.9], since the kurtosis RMSE
is already 1.85 at t = 0.9. The fitted log-slopes are 3.26 for the correlation and 5.97 for the
kurtosis, against predictions of 2 and 4, with a ratio of 2.71 against a predicted 2.

**The honest reading.** The predicted ORDERING holds and is not in doubt -- shape is lost faster
than correlation, at every configuration measured. The predicted EXPONENTS are not confirmed by
these runs: both measured slopes are steeper than predicted by about the same amount, on a window
three levels wide whose upper end is already bending into saturation. Two mechanisms plausibly
explain the overshoot and neither is tested here: RMSE contains a bias term that the Cramer-Rao
bound does not describe, and the upper levels of the window are already leaving the regime where
the estimator is efficient.

**What would actually test it.** The window's width is set by the gap between the crossover
t_* = 0.5 log lambda_max(Sigma_0) and the level at which the bound crosses the coordinate's range,
which moves only logarithmically in M. Reaching a decade of window needs a much smaller rho
together with a much larger M, or an estimand that does not saturate (the dispersion of the
estimate across seeds rather than its RMSE, which drops the bias term and is a different claim).
Neither is in the current budget.

**What is claimed instead.** E01 establishes the information rates exactly, by computation rather
than by fitting. E06 establishes that the estimator attains the bound (1.02x). Those two together
give the corollary its content wherever the coordinate remains identifiable, and the point at which
identifiability is lost is itself measured (t = 0.98 at rho = 0.85). The paper states the rates as
a property of the channel, verified directly, and does not claim a fitted confirmation of the
exponents from estimator error.

## From the 18 September harvest (E08, E11, E12, E13)

**N-11. The tree-exact reference is not uniformly converged, and the binding parameter is the
domain, not the number of grid points.** Every number in the project is measured against grid
belief propagation at N_g = 401 points on [-8, 8]. E11 swept seven grids against a (3201, 12)
reference across five innovation families, four correlations and twelve levels. Two findings, one
reassuring and one not.

Reassuring: at the frozen configuration the operating grid is excellent. Laplace at rho = 0.85 has
a pooled drift of 9.2e-6, a worst-level drift of 3.7e-5 and 2.8e-4 of the belief mass in the outer
2% of the domain. Nothing measured at the frozen point is in question.

Not: the worst-case drift over the whole sweep is **flat in N_g at fixed half-width** -- 5.411e-2
at (201, 8), 5.419e-2 at (401, 8), 5.424e-2 at (801, 8), 5.425e-2 at (1601, 8) -- and falls only
when the domain widens: 2.6e-2 at (401, 10), 3.0e-3 at (801, 12). That is the signature of
truncation, not of quadrature. Refining the grid at A = 8, which is what "refinement" has meant in
this project and in the thesis before it, cannot fix it. The offending configuration is
Student-t(5) at rho = 0: the posterior mass in the outer 2% of the domain is 1.00, i.e. the tails
are entirely outside [-8, 8], and the drift is 5.4e-2.

Consequence: **E07 reports that cell.** The family/correlation ladder covers Student-t at rho = 0,
so a reported number sits outside the validated envelope. E07b/E07c re-run the ladder at (801, 12)
and (1601, 16) to establish what it should be. Until they land, the Student-t rho = 0 entries of
the phase ladder are flagged and are not cited.

A second, milder observation from the same sweep: the column-mass defect of the discretised kernel
grows with correlation (about 1e-16 at rho = 0 to about 0.1 at rho = 0.95, every family) because a
narrow transition kernel is poorly resolved by a fixed grid, yet the drift at rho = 0.95 stays at
1e-3 or below. The defect is close to uniform across the grid and BP renormalises it away. It is
reported as a diagnostic, not as an error bar.

**N-12. E12's sample-complexity reading is confounded by the EM iteration ladder at its highest
level, and the confound, not the sample size, is what binds.** E12 was designed to escape E06's
saturation problem by asking how much data buys back a fixed accuracy as the level rises, which
does not saturate. At t = 0.6 and t = 0.8 it worked: the kurtosis RMSE falls cleanly with M
(0.87 -> 0.14 and 1.79 -> 0.32 over M = 256 to 8192) and M* can be read off. At t = 1.0 it did not:
the RMSE sits between 1.84 and 2.53 at every M and is **non-monotone in M**, and the fitted excess
kurtosis scatters between 0 and 3.4 across seeds with no concentration. The diagnosis is in the
run's own columns -- at t = 1.0 the EM is censored by the 1200-iteration cap in 40 of 40 cells, and
the validation-selected checkpoint is the last rung in 100% of cells at M >= 4096. The optimiser is
still travelling when it is cut off, so more data does not help: the experiment measured the
budget, not the sample complexity.

What the run does support, stated as such: the covariance coordinate's log-slope is 5.40
[2.28, 7.72] over three levels, which excludes 8 and contains the predicted 4; and the shape
coordinate never reaches the threshold at t = 1.0 even at M = 8192, which gives a one-sided bound
of d log M*/dt > 5.54, excluding 4. The two together separate the coordinates from opposite sides.
The point estimates where a crossing exists are 7.54 and 8.53 for the shape coordinate against a
predicted 8, and 4.44 for the paired ratio against a predicted 4, but their seed-bootstrap
intervals ([-2.88, 11.70], [-1.32, 11.93], [-9.54, 12.10]) exclude nothing and are **not** evidence
for the exponents. E12b re-runs it with five levels, 16 seeds and an EM ladder extended to 8000.

**N-13. The same censoring audit flags a reported claim in E06.** The fraction of cells whose
validation-selected checkpoint is the last rung of the ladder is 0.12-0.38 at t <= 1.0 -- validation
stops early of its own accord, so the cap is not binding, and the Cramer-Rao result over the
informative levels stands -- but it is 0.88 at t = 1.4 and 0.75 at t = 2.0, with median selected
checkpoint equal to the cap. Those are exactly the two levels at which the paper reports the
estimator leaving the bound. The departure may be information or it may be budget; the run as it
stands cannot tell. E06c re-runs the nine levels with the ladder extended to 8000. The sentence is
flagged in the manuscript until it lands.

The same audit clears two other experiments. In E02a the at_cap rate reaches 0.69 at M = 2048, but
E02a_v1 -- the same design with the ladder cut from 1200 to 400 -- reproduces every overlapping cell
to 0.001, so the estimates have plateaued well before the cap and the capacity conclusion does not
depend on it. In E13 the rate is at most 0.25 and C = 1 never censors.

## From the 18–19 September audit (reports/audit-2026-09-18.pdf)

**N-14. "The estimator attains the Cramér–Rao bound (1.02×)" is withdrawn.** The
average was over the five levels with RMSE < 2, of ratios 1.16, 1.51, 0.71, 0.87, 0.87 —
three of them below one, which for a bound on unbiased estimators means bias (−0.29 and
−0.66 at t = 0.4 and 0.7) or the wrong reference family (one parameter against an
eight-component mixture). Per level the truth is: unimodal fits with dispersion within the
sampling error of the one-parameter reference for t ≤ 0.4; a split at t = 0.7 (4/16
collapsed to a Gaussian innovation, the rest shrunk, dispersion 0.42× the reference);
15/16 collapsed at t = 1, one level below the level at which the reference crosses the
coordinate's range (0.98). The paper now reports the per-level table
(`paper/sections/audit-numbers.tex`, `aud{Collapsed,SdKurt,SdKurtGood,BiasKurt}T*`) and
says "at the reference while identifiable; collapses where the reference says it must".

**N-15. "Information-matched, 7.3×" was the wrong arm.** The arm behind 7.3 is denoising
score matching on (a, x_t) pairs with one fixed corruption (interface B): the network sees the
clean chain. The genuinely matched arm (interface C: implicit score matching on x_t only,
checkpoint selected on the implicit objective of a held-out noisy bundle, no oracle) is in
the same frozen file at 0.165, a paired factor 7.9 [6.6, 9.3], 16/16 seeds — larger. The
generator `tools/make_baseline_numbers.py` used to pick the best of the two arms by risk
and label it matched; it now emits the ISM arm as `matchedBest*` by definition and the
fixed-corruption arm as `matchedFixed*`. The ISM window head diverges at M = 32 (risk 1.2,
the held-out objective rising through the ladder), a known instability of implicit score
matching on tiny samples; that cell is not a measurement of the architecture.

**N-16. "Student-t5 departs from e^{-8t} as the moment condition predicts" was
truncation.** E18 recomputes the closure gap at (1601, 16) and (3201, 32): with the same
chains, Student-t5's local slope on [4, 4.5] is −4.0 on the (801, 8) grid, −5.7 at (1601, 16)
and −7.9 at (3201, 32), and the ratio to (1/6)‖κ4‖² reaches 1.06 at t = 5 on the widest
domain; ν = 7 and 9 behave the same way. On (801, 8) the reference carries the tail of the
truncated prior beyond t ≈ 3. The expected e^{-νt} tail term of a regularly varying law is
carried by chains with |a| ≳ e^{2t}, of probability ~1e-16 per site at t = 5 — never
sampled. Fig. 2c is redrawn from E18; the sentence is replaced by the heavy-tails paragraph
of §3. Theorem 3 is restated under an exponential-moment assumption (the proof needs it).

**N-17. "Essentially all of its belief mass outside the domain" was one site.** E11's
diagnostic is a maximum over 8192 (chain, site) pairs; regenerating the chains from their
tags gives 1, 1, 1, 2 sites beyond |a| = 8 per seed (expected 1.2), the largest at 15.4. The
mechanism (truncation, flat in N_g) is right; the sentence is replaced by the exceedance
count and its Poisson expectation (0.17 per 8192 sites at A = 12, 0.04 at A = 16).

**N-18. The closure constant was quoted to a precision the sample did not have.** The
per-chain squared gap is heavy-tailed (maximum ≈ 300× the median over 8192 chains); a
512-chain mean has ≈15 % relative s.e., and two 512-chain replicates of E01's Laplace
closure at t = 3 give 0.83 and 1.14. E01's "0.99" was the mean of four such replicates,
precise to ≈±10 %. E18 part "constant" (8192 chains, chain bootstrap) gives 0.92
[0.84, 1.01] at t = 3 and 0.99 [0.91, 1.08] at t = 4 for Laplace; the paper quotes the
interval.

**N-19. The shape-information constant is not reproduced on the chain — and the constant
is right.** E01's outer-product estimate plateaus at 0.81 (β = 2) and 0.92 (β = 3) of
(1/24)‖∂κ4‖²; E21's untruncated scalar quadrature reproduces it to 0.25 % at t = 1 and to
1e-4 by t = 2 for β ∈ {1.5, 2, 3}. The shortfall on the chain is the estimator's floor (a
cancellation of O(1) terms to α^4 ≈ 1e-5 against 1e-5 grid accuracy). The paper now says
which constant is verified where.

**N-20. E12b as pre-registered would have refuted the corollary with its own prediction.**
At ρ = 0.5 the local slope of log(1/I_eff) on E12b's levels [0.6, 1.0] is 6.1–6.9 for the
shape coordinate and 3.8–3.9 for the correlation (E21), because e^{-2t} λ_max is still of
order one there; the asymptotic 8 and 4 are the t → ∞ limits. The analysis plan was amended
before any E12b cell ran (`experiments/registry/E12b.json`): M*(t) is compared with the
exact information curve.

**N-21. Two-thirds of the window head's plateau is the boundary.** On the interior sites
the head's error at M = 2048 is 0.075 against 0.118 on all sites, and it improves from
0.118 to 0.082 as the chain lengthens to 256 (zero padding). E15 computes the exact
radius-4 floor at ρ = 0.85: 0.030 pooled (peak 0.065 at t ≈ 0.7–1.0), against the head's 0.118
(0.075 interior) and EM–BP's 0.023. **"Locality is a ceiling" is withdrawn**: the radius costs
0.030 of information; the plateau is the one-layer architecture's. The manuscript's sentence
"a radius-4 window cannot represent a message that has travelled the length of the chain" is
replaced by the floor.

**N-22. E14's first screen measured one chain, not a regime.** The first E14 screen (job
663143, 108 cells, 19–20 Sep) gave the window head test risks of 0.883 and 1.648 at the
frozen point (ρ = 0.85, L = 32, M = 2048, radius 4, width 64) against 0.166 for the same
network in E03's matched arm, with every held-out ISM curve rising from the first checkpoint.
The first diagnosis blamed the divergence estimator (E14 trained the window head with
Hutchinson's one probe where E03 used the exact trace), and the driver was changed to E03's
per-arm estimators and learning rates; a screen re-run with the exact trace (job 664056)
failed the same way. Running E03's own training function and E14's on the same data settled
it: both give 1.1–1.3 on E14's data and 0.166–0.169 on E03's. The cause was the data. In
`_common` the training chains and the validation chains were drawn as
`prior.sample(rng_for("e14-train", seed, L, rho), L) for _ in range(n)`, the generator
re-created inside the comprehension, so all 2048 training chains were one draw and both arms
fitted 2048 copies of a single chain (the quick run showed it: ρ̂ = 0.67, κ̂₄ = 0.46 on
"Laplace data at ρ = 0.85", and should have been read as a data defect). Fixed on 20 Sep:
one generator per set and an assertion that the chains are distinct; with the fix E14's
pipeline reproduces E03's 0.169 at the frozen point (2000 steps). `grep "sample(rng_for("`
finds no other current experiment with the pattern (E13 creates its generator once). The
screens 663143 and 664056 and the confirms 663144 and 664065 are void and are kept under
`outputs/E14_screen_invalid/`; the estimator change stays because it is E03's, not because
it was the cause. Two lessons for the protocol: a fitted parameter that contradicts the
generating law is a data check, not an estimator result; and every driver should assert
that its training set has as many distinct chains as it claims.

**N-23. The archival capacity sweep (exp_32) was fitted to one chain; the stopping-rule
diagnosis of N-A1 is withdrawn as the cause.** Found by the cluster-ops session on 21 Sep
while turning N-22 into a test (`tests/test_no_regenerated_rng.py`): the archival
`exp_32_capacity_equivalence.py` (and exp_33, exp_34, both uncited) draw their training set
as `prior.sample(rng_for("exp32-train", seed, n_chains), N_SITES) for _ in range(n_chains)`,
the generator re-created inside the comprehension, at the very commit (35578c99, 31 Aug)
whose outputs are frozen as `outputs/frozen/exp_32_capacity_merged/`. Every arm, capacity
and seed of the thesis's capacity conclusion was therefore fitted to a single chain of 32
sites; validation and test bundles were drawn correctly. The frozen numbers carry the
signature: the C = 1 arm's pooled risk ranges from 0.13 to 1.15 across seeds where
Proposition 1 puts a correct C = 1 fit on the LMMSE floor (0.167), which E02a and all
fifteen E13 cells reach to 0.01. The stopping asymmetry we had reported as the cause of the
earlier conclusion (tolerance fired in 31/32 C = 1 cells, in none with C ≥ 4) is real but is
EM's behaviour on one chain and says nothing about capacity. The paper's capacity appendix
now says the sweep is void and why; the retirement of the thesis's capacity claim (N-A1)
stands with a prior and stronger cause. The paper's own capacity evidence (E02a, E02a_v1,
E10, E13) draws chains correctly and passes the new test. The thesis text that cites exp_32
is the author's to correct; the cluster-ops session has told them where. Reproduction (21 Sep
11:35, cluster-ops session, laptop): the icml copy of exp_32 run as it stands for seed 0,
M = 512, C = 1 at the frozen configuration reproduces the frozen row to six decimals
(val log-evidence per edge −1.313205, test −1.308242, schedule score risk 0.202893, 61 EM
iterations), with one distinct chain among the 512 drawn. The frozen exp_32 numbers are
therefore the output of a one-chain training set with certainty, not by inference; the
rerun stays in that session's scratchpad and is not frozen.

**N-23, addendum (27 Sep).** Two quantities the 21 Sep entry did not put a number on, both
measured against the archival draw (`rng_for` and `_stable_hash` are byte-identical between
`src/markov_score/utils.py` and the archival `src/utils.py`, verified by diff, so
reproducing the call signature reproduces the archival stream exactly):

* **Every cell, not a sample of them.** All **32/32** (seed, M) cells of the archival sweep
  — 16 seeds × M ∈ {128, 512} — have **exactly one** unique training chain, while the
  validation and test bundles have 512 and 1024 distinct chains respectively, because
  `_bundle` hoists its generator (archival `exp_32_capacity_equivalence.py:159-161`) and the
  training draw does not (`:214-217`). `git log -L214,217` shows the lines unchanged since
  the file's first commit (3ec893c), and the frozen outputs stamp `git_commit = 35578c99`
  with `source_is_clean = true`, so no run of exp_32 was ever free of it.
* **The training-size axis was a no-op.** The M-step is fitted on transitions: at M = 128 it
  receives 3968 of them, of which **31 are distinct** — one chain's worth — and the same 31
  at M = 512. The frozen numbers carry that signature. Paired within (seed, C), going from
  M = 128 to M = 512 buys **+0.011 ± 0.094** in test log-evidence per edge and
  **−0.046 ± 0.302** in schedule score risk over 80 pairs: both inside one standard error of
  zero. A fourfold increase in nominal training data changed nothing, because the latent
  sample size did not increase at all.

This also explains the *direction* of the retired claim, which is worth stating because the
direction was what made it credible. Capacity does not merely fail to help in the frozen
data, it monotonically **hurts**: mean schedule score risk 0.280 → 0.397 → 0.460 → 0.458 →
0.463 and Hellinger² 0.037 → 0.068 → 0.097 → 0.102 → 0.103 for C = 1, 2, 4, 8, 16. That is
what fitting a flexible mixture to 31 effective samples looks like. Read as "a single
Gaussian innovation suffices" it also contradicts this project's central claim, since a
Laplace innovation has excess kurtosis 3 and E02a's C = 8 fits reach 0.018–0.090 against the
LMMSE floor of 0.168 that any correct C = 1 fit cannot beat (Proposition 1). Claim **E5** is
already RETIRED in `docs/CLAIM_EVIDENCE_MATRIX.md` (2026-09-21) and no number in the paper
comes from exp_32; scope claim **C12** in `docs/CURRENT_STATE_AUDIT.md` was already flagged
as self-inconsistent by that audit's W2. Nothing further to withdraw — this addendum closes
the question with numbers rather than leaving it as "to be checked".

**N-24. The E12b analysis path had four defects that fire only when the data lands; found
and fixed before it did (23 Sep).** E12b moves from 8 seeds and 3 levels to 16 and 5. Nothing
about that is visible until `outputs/E12b/ladder.csv` exists, so a self-audit went looking for
what the arrival would break. (i) The bootstrap is enumerated exactly while that is feasible;
sixteen seeds put the enumeration at C(31,16) = 300,540,195 multisets, so E12b takes the Monte
Carlo branch by construction — and that branch stood at B = 4000, which is the very setting the
18 September audit caught moving the covariance slope's upper limit between 7.7 and 8.2, i.e.
across the prediction it tests. The generator's own comment promised "a large B and a reported
error"; neither was implemented. B is now 200,000 and the branch reports its Monte Carlo error
(`scMcErr`) from MC_STREAMS = 5 blocks. Validated against ground truth by forcing the Monte
Carlo branch on E12, where the exact answer is known: limits [-3.15, 12.33], [2.27, 7.72],
[-9.64, 11.84] against the enumerated [-3.15, 12.35], [2.28, 7.72], [-9.66, 11.88], with a
self-reported se of 0.037 — the stated error covers the observed deviation. (ii)
`audit_claims.audit_sample_complexity` enumerated unconditionally and read `outputs/E12`
hard-coded: at sixteen seeds it would not have failed the gate, it would have hung it. It now
takes the same limit, B and stream as the generator (imported, not copied — the bootstrap
protocol is a declared parameter, not a result) and reads E12b when present. (iii) The gate's
OPTIONAL registry mapped the bare generator to E12's ladder, so adding an E12b key would have
left two generators writing one fragment and the later one winning silently — N-22's sibling and
exactly the trap E13/E13x sprang. A `SUPERSEDES` rule now drops the superseded key, and the
fragment carries a `scSource` stamp that the auditor asserts. (iv) Found by running the tool at E12b's
shape (16 seeds, 5 levels) on synthetic input in a scratchpad — structural smoke test only, no
value from it entered `outputs/`, a fragment or the paper, and it was deleted after: a
coordinate that never crosses its accuracy threshold has no M* and no slope, and the tool
emitted `\newcommand{\scSlopeShape}{nan}`, which would typeset the string "nan" anywhere the
macro was used. E12's shape coordinate resolved, so this never fired there; E12b runs to
t = 1.0, where E12 was already censored in 40/40 cells, so an unresolved shape coordinate is a
likely outcome and not a contrived one. Valueless macros are now WITHHELD and named on stdout:
prose that uses one then fails at LaTeX time with an undefined control sequence, which is the
loud version of the same fact. Runtime at E12b's shape is 13 s, so the gate does not slow
materially. None of this could be tuned to a result: the result does not exist yet.

**N-25. A refusal that did not refuse: `audit_claims` discarded its own return codes.** The
E13x source-stamp guard committed in 3386f21 ended `return 1`, and `main()` called `fn()`
without looking at what came back — failure travels only through `FINDINGS`. The guard printed
its warning and the gate exited 0. It was verified when written by confirming the message
appeared, not by confirming the exit code changed, which is the whole of the mistake: a guard
tested by its output rather than by its effect. Both stamp guards now append to `FINDINGS`, and
`main()` records a non-zero return as a finding in its own right so that a future guard written
the obvious way is not silent. Both were re-tested by exit code: moving `outputs/E13x` aside
and planting an `outputs/E12b` against an E12-stamped fragment each exit 1.

**N-26. A gate caught a real error, and my first diagnosis of it was wrong.** E28's `verify`
part rebuilds the pooled risk from its own per-level pairs and compares it against E03's
`confirm.csv`; job 671542 failed it at 2.878e-02. I attributed the disagreement to float32
reduction-order drift from reloading 15 September checkpoints, and the evidence fitted: the
scatter was symmetric (111 of 192 cells high, 81 low) and ordered by architecture complexity
(window 0.24% median, bigru 2.88%). That story was false. The real cause was that
`e28_per_level_ranking.py` hardcoded `n_test = 1024` where E02a and E03 both score on 2,048,
so every pooled risk was a different estimand. The tell was in the numbers from the start and
I read past it: `sq_ref` was 704,579 against E03's 1,417,692, a factor of 2.012. At the
correct `n_test` the gate passes at 4.650e-11, and on the rerun (672153) at 1.712e-15. Two
lessons, the second more useful than the first. (i) Take the constants from `PROTOCOL` and
never retype them — the line now reads `n_val=PROTOCOL.n_val, n_test=PROTOCOL.n_test` with a
comment saying why. (ii) A plausible mechanism that explains the *pattern* of a disagreement
is not a diagnosis; check the magnitude first. A factor of 2.012 in a denominator is not
float32 drift, and no amount of structure in the residuals makes it so. The void output is
kept on disk at `outputs/hpc_E28_VOID_estimand/`, because a wrong diagnosis is only checkable
against the numbers that produced it.

**N-27. Six jobs, one output directory, one merged CSV: five architectures' results were
overwritten and nothing noticed.** The six E29 ladder jobs (671654–671660) were each given a
single architecture via `--set 'archs=("bigru",)'` but all wrote to the same `{OUT}/E29`.
Each job rebuilds `ladder.csv` from the cells its own config names, so the last job to finish
(`bigru`, 20:40) left a `ladder.csv` containing bigru and nothing else. The fetched file had
3,840 rows where six architectures should give 21,504, and it looked perfectly well-formed:
right header, right seeds, right sizes, complete on its own terms. What exposed it was reading
the coverage rather than the file — `sorted({x['arch'] for x in rows})` returned one element
where six jobs had run. **No data was lost**: the 1,792 per-cell JSONs under `E29/cells/ladder/`
survived, one per (seed, size, arch, hyper-parameter), because `run_cells` writes each cell
before merging, and the full ladder was rebuilt from them with every cell present
(320/320, 320/320, 256/256, 320/320, 256/256, 320/320 per architecture). The near miss is
that a partial merged CSV is not distinguishable from a complete one by inspection, and the
freeze machinery would have frozen it. Two fixes: `icml_submit.sh` now gives each E29
architecture its own output directory, the way E25 already split its seeds, and
`tools/registry.py --check` reports row counts that a coverage expectation can be read
against. The general rule, which E25's seed split had already got right and E29 ignored:
**concurrent jobs must not share a merged output file.**

**N-28. The open question left by N-23 is closed: the archival capacity conclusion is void,
and the clean data reverses it.** N-23 recorded that `exp_32_capacity_equivalence.py` carried
the same re-seeded-generator defect and that the conclusion drawn from it — audit C12, "no
mixture capacity improves on a single Gaussian innovation" — was OPEN until checked against
*the archival run's own code*. Checked, on 28 September 2026, and not by reading it: running
the archival construction
(`research/nongaussian-bp/experiments/exp_32_capacity_equivalence.py:214-216`) under its own
`rng_for` gives **1 unique row out of 32, 128, 512 and 2048**. Every arm at every capacity and
every seed was fitted to one chain replicated. The defect manufactures the conclusion: with a
single distinct sequence there is nothing for a richer kernel to fit, so "extra components do
not help" is what the experiment was guaranteed to say. The paper already withdraws the
reading (`paper/body/A4-additional.tex:42-58`); what was missing was the measurement and a
replacement number.

**The replacement, from E02a, whose training set is clean** (the broken line at the frozen
commit `ea0435a:experiments/e02_capacity.py:196` is dead — guarded by `if False` — and 197-198
hoist the generator correctly): pooled over 16 seeds at `region=all`, a mixture beats a single
Gaussian innovation by **1.96x at M=32, 3.23x at 128, 4.78x at 512 and 7.23x at 2048**. The
void claim is not merely unsupported; it is backwards.

| M | K=1 | K=2 | K=4 | K=8 | K=16 | val-selected K (16 seeds) | risk there |
|---|---|---|---|---|---|---|---|
| 32 | 0.1743 | **0.0888** | 0.0978 | 0.1033 | 0.1044 | K=2 in 13/16 | 0.0876 |
| 128 | 0.1681 | **0.0520** | 0.0547 | 0.0570 | 0.0580 | K=2 in 10/16 | 0.0507 |
| 512 | 0.1677 | **0.0351** | 0.0379 | 0.0366 | 0.0367 | K=2 in 7, K=16 in 5 | 0.0332 |
| 2048 | 0.1672 | 0.0285 | 0.0252 | 0.0239 | **0.0231** | K=16 in 7, K=8 in 4 | 0.0223 |

Two things follow that are worth more than the correction. **(i) A third independent instance
of the capacity U-shape.** Beyond K=2, more components *hurt* at M=32 (0.0888 to 0.1044, 17%
worse) and help at M=2048. The same shape appears in the MDN kernel's parameters (Q6: 61x more
parameters, 2-2.6x worse) and in the score nets (E29: the largest rung worse than the best
smaller one in 16/16 seeds at M=128). Three different capacity axes, one phenomenon, and the
one Jerome named unprompted in the call. **(ii) The validation-selected capacity grows with the
data** — modal K=2 at M=32 and 128, shifting to K=8 and K=16 at M=2048 — which is the sample
-efficiency claim appearing on an axis it was not designed to be measured on, and selected on
validation rather than on test.

**A consequence for numbers already reported.** E28 and E29 take em_bp from E02a at a FIXED
K=8, which is not what validation selects at small M. Against the validation-selected kernel
the best score net is 2.13x at M=128 (reported 1.89x) and 2.51x at M=2048 (reported 2.34x).
**The published gaps are therefore conservative by 11-13%, not optimistic.** They are left as
they are, because E28's and E29's internal consistency rests on the single K=8 reference their
runs actually used; the refinement is recorded here rather than substituted in.

**N-29. I put a wall-clock experiment on a node I was myself loading with ten fitting jobs.**
`sacct` for 28-29 September shows `msd-e27-cost-big` and all ten E30/E31 jobs on **cnode07
simultaneously**: a timing run asking for 8 cpus beside ten 16-cpu fitting jobs. The damage is
measurable because the same cell exists in both runs of e27-cost-big. At $B = 2048$ the 400-step
reverse SDE reads **430.8 ms/seq on the contended node against 200.1 ms/seq on the earlier run**,
while `ancestral_true` barely moves (0.1801 against 0.1758 ms/seq). So the headline ratio goes
from 1,138x to **2,392x — contention more than doubled our apparent advantage.** It does not
cancel in the ratio, and it does not cancel in the direction that would merely add noise: 400
sequential network evaluations are hurt far more by a busy node than a handful of BLAS calls, so
the error **flatters the claim**. E24's registry already states the principle — "a timing at an
unstated thread count is not a reproducible number" — and node occupancy is the same kind of
unstated condition. Neither run is citable: the earlier one shared cnode08 with the E29 ladders.
**AMENDED 29 September, and it is worse than first written.** The contamination is not confined to the cost-big runs. `sacct` puts `msd-e27-cost` (671531) on cnode02 from 12:12 to 16:31 and `msd-e27` (671530) on cnode02 from 13:28 to 17:29 — **a three-hour overlap covering most of a 4h19m timing run, both jobs mine**. That is the run behind the cited *725x cheaper at equal error* and *1,436x against the 400-step SDE*. E24 is in the same position for a weaker reason: its runs did not overlap each other, but none requested exclusivity, and E24's own registry entry had required "the node uncontended" as a sentence that nothing enforced. **No cost number from E24 or E27 is citable until an uncontended rerun lands** (`e24x-cpu`, `e24x-gpu`, `e27x-cost`, and `e27-cost-big` as job 676603). What is NOT affected is every error measurement — contention changes wall-clock, not KL, so Q8's "one sweep generates at the Monte Carlo floor" stands untouched. `tests/test_submit_cases_disjoint_outputs.py` passes `e27` and `e27-cost` correctly and always will: it is about files clobbering each other, not about jobs contending, and the two are different failures needing different guards.

Fix: `MSD_HPC_EXCLUSIVE=1` adds `#SBATCH --exclusive`, and every wall-clock case must use it.
The lesson is narrower than "be careful": **do not queue a timing experiment and a compute
experiment into the same partition in the same session**, because the scheduler will co-locate
them and nothing in the output says so.

**N-30. A gate placed where it could not see its references, which cost 9.6 CPU-hours to
discover.** E30's `part_verify` compares its own `sq_ref` against `outputs/hpc_E28/per_level.csv`
and `outputs/hpc_E29/ladder.csv`. Those are **locally fetched** artefacts, and `.gitattributes`
export-ignores `outputs/hpc_*` from the deploy archive, so on the cluster they cannot exist. All
four confirmatory jobs therefore ran the full ladder — 128 of 128 cells, every one written — and
then exited 2 on a comparison that was impossible by construction. I tested the gate only where
its references happened to be present. **No data was lost** and the gate passed on the first
attempt once run in the right place: 1152 cells, worst relative `sq_ref` disagreement
**0.000e+00**. Fix: `verify` is dropped from a default run with a message saying it is a
post-fetch step, and is still available as `--only verify`. What it is deliberately NOT allowed
to do is pass when its references are absent — that would convert it into a gate covering
nothing, which is the failure mode of `registry.py --check` (silent for E22-E29) and of
`audit_claims` (N-25), and it is not worth trading a loud 2 for a quiet 0. The general rule:
**a cross-experiment gate belongs at the place where both experiments' outputs exist, which is
after the fetch, not inside the job.**

## N-31 — the sign-change density is not a physical observable under refinement (1 October 2026)

E34's P1 predicted that the kink density per unit length stays within 20% across the
refinement cells (h = 1, 0.5, 0.25 at fixed physical length), since they describe one physical
system. Measured on 4096 exact chains per cell: 0.120, 0.175, 0.247 kinks per unit length —
growing by 1.41 per halving of h, i.e. like 1/√h. A sign change between neighbours is not a
kink: the field is rough and crosses the barrier inside a domain, and a finer lattice resolves
more of those crossings (the continuum limit of a Brownian-like path has infinitely many zero
crossings per unit length). The same is true of the mean domain length in physical units
(5.09, 3.75, 2.72). The commit message of b33b362 claimed the opposite, written before the
table was read; this entry corrects it. Replacement (registered as P1b before being computed):
runs shorter than the kink width √2 (√2/h sites) are merged into their neighbours, and the
coarse kink density is the statistic the refinement cells are predicted to share. The raw
statistic stays in the table as what it is.

## N-32 — a 1.28 GB table in the history kept the branch off GitHub for five days (1 October 2026)

`outputs/hpc_E22c/shapes.csv` (1.28 GB) and `outputs/hpc_E23/representation.csv` (122 MB) were
committed on 26 September (86c3f50) with the first HPC fetch, and every push since was refused
by GitHub's 100 MB file limit, so twenty-six commits of post-call work existed only on one
laptop and in the cluster deployments until today. The fix is a replay of the 24 commits from
86c3f50 without the two files (`docs/COMMIT_MAP_2026-10-01.md` maps every heavy hash to its
pushed twin); the files stay on disk, now git-ignored, and were never frozen. Lesson: a fetched
cluster table is not a registered output until it is frozen, and a table that cannot be pushed
must not be committed -- `hpc/icml_fetch.sh` should refuse files above the limit rather than
leave the refusal to the remote five days later.

## N-33 — a global polynomial tilt fitted by clean maximum likelihood is ill-posed outside the data (1 October 2026)

On the laptop preview of E34's clean arm (base cell, M = 2048), `em_tilted_poly` (Legendre degree 6)
fitted by clean-data MLE gave a risk of 0.0047 on seed 0 and **1.79** on seed 1 — worse than the
isotropic map — while `em_tilted_spline` gave 0.0041 and 0.0074 and the mixture 0.27 on both. Every
tilted fit had coefficients at the optimiser's bound (±200). The mechanism: with clean data the
statistic Ξ is a counting measure, so no mass arrives beyond |a| ≈ 3.5, and the likelihood
increases without bound as log g → −∞ there (a smaller Z(a) for every source raises K at the
observed arrivals). The MLE therefore drives the tilt down outside the data support until the
bounds stop it. For a local basis (B-splines) the far coefficients go to −200 and the bulk is
untouched, which is the correct limit (g = 0 where nothing arrives); for a global basis the same
push bends the polynomial inside the bulk, and whether it does depends on the seed. The fixed soft
wall −(a/5)^12 does not help: it is 0.07 at |a| = 4, where the push already acts. The quartic site
energy (`em_pair_quartic`) is immune: g = ψ_f e^{−f} decays by construction. Under noisy EM the
posterior Ξ is smoother, but the same pressure exists; the registered E33 rows of the polynomial
tilt are read with the at-bound flag (`tools/make_field_potential_figs.py`, `n_at_bound`), and
the role of "the parsimonious exactly specified kernel" passes to the site-energy parametrisation
(5 parameters, E33 amendment 1). Lesson: a tilt basis needs either local support or a penalty;
"exactly specified up to the basis" says nothing about the MLE being finite.

**N-33, addendum (1 October, evening; the first noisy preview cells).** The pressure is not a
clean-data artefact. Through noise at the Θ = 2 cell, M = 512, the Legendre tilt's coefficients
reached the ±200 bound on seed 1 (risk 0.041 against 0.011 on seed 0), and the spline's far
coefficients sit at ±199 on both seeds with risks 0.028 and 0.056, the second ABOVE the Gaussian
closure's 0.041 on that cell. At Θ = 2 the marginal is narrow, so more of the basis is
unconstrained by arrivals and the push outside the data bends the fit inside it. The five-parameter
site energy on the same cells: 0.011 and 0.020. The registered E33 arms run as registered; the
remedy to register as a further amendment is a penalised tilt (a P-spline second-difference penalty
on the B-spline coefficients, a ridge on the Legendre ones), which keeps the M-step concave.
