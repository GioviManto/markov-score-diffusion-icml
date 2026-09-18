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
[2.13, 7.57] over three levels, which excludes 8 and contains the predicted 4; and the shape
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
