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
