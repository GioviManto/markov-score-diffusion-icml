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
