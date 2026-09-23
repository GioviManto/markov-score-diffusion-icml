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

**N-24. The E12b analysis path had three defects that fire only when the data lands; found
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
fragment carries a `scSource` stamp that the auditor asserts. None of this could be tuned to a
result: the result does not exist yet.

**N-25. A refusal that did not refuse: `audit_claims` discarded its own return codes.** The
E13x source-stamp guard committed in 3386f21 ended `return 1`, and `main()` called `fn()`
without looking at what came back — failure travels only through `FINDINGS`. The guard printed
its warning and the gate exited 0. It was verified when written by confirming the message
appeared, not by confirming the exit code changed, which is the whole of the mistake: a guard
tested by its output rather than by its effect. Both stamp guards now append to `FINDINGS`, and
`main()` records a non-zero return as a finding in its own right so that a future guard written
the obvious way is not silent. Both were re-tested by exit code: moving `outputs/E13x` aside
and planting an `outputs/E12b` against an E12-stamped fragment each exit 1.
