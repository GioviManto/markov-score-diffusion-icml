# Metrics — what every reported number measures

Created 6 Oct 2026 for step 1 of the consolidated note's section 29 ("recover exact metric definitions from frozen
outputs"). Each definition below is read from the code that wrote the frozen CSVs, with that code's location; where a
document states it too, the document is cited. Symbols are those of `docs/NOTATION.md`.

## 1. The headline: schedule-pooled relative score error ("risk")

For a posterior-mean map $\hat m(\mathbf x,t)$, the score it implies by Tweedie is
$\hat S(\mathbf x,t)=-(\mathbf x-e^{-t}\hat m)/\Delta_t$, and

$$\mathrm{risk}=\Bigl(\sum_{t}\sum_{j}\sum_{i}\bigl(\hat S_t(\mathbf x_j)_i-S_t(\mathbf x_j)_i\bigr)^2\Big/\sum_t\sum_j\sum_i S_t(\mathbf x_j)_i^2\Bigr)^{1/2},$$

summed over the twelve levels $t$ of `FROZEN.t_grid` (log-spaced on $[0.05,3]$), the test chains $j$ and all sites
$i$. Code: `experiments/e14_regime_baselines.py::_pooled`, used unchanged by E14, E30, E33, E34 (clean part), E35 and
E36. Paper: appendix "Estimand" (`paper/body/A3-protocol.tex`), which also defines the interior-site (5–28) and
centre-site variants used for selection and diagnosis.

- **Reference $S$.** Grid BP with the true kernel and the true first-site law on $N_g=401$ points of $[-8,8]$
  (`grid_bp_batch`, `log_mu` = the chain's stationary law) — the *reference* or *oracle* score; for E35 the closed-form
  score of its Gaussian construction; for E36 grid BP with the strided kernel $K^s$.
- **Test bundle.** $2048$ fresh chains (`PROTOCOL.n_test`), each noised once **at every level** with independent noise
  (`e33._bundle`, rng tag `e33-test`); the validation bundle has $1024$ (`PROTOCOL.n_val`). The bundles' hashes are
  stored per row (`val_bundle`, `test_bundle`), so two arms are on the same bundle iff the hashes agree.
- **Weighting.** Since $\hat S-S=e^{-t}(\hat m-m)/\Delta_t$, the pooled sum weights a level by $e^{-2t}/\Delta_t^2$
  relative to its posterior-mean error: the low levels dominate. The posterior-mean MSE per level is the same quantity
  up to that factor (field report, correspondence table).
- **Aggregation.** Means over seeds of the per-seed risk; ratios (e.g. arm / floor) are formed within seed and then
  averaged; intervals are percentile bootstraps over seeds (protocol).

## 2. Floors and references

- **Linear floor** (`lmmse_true`): the LMMSE posterior mean $e^{-t}\Sigma_0\Sigma_t^{-1}\mathbf x$ with the chain's
  true covariance $\Sigma_0$ (for the field, from the transfer-operator spectrum), scored by the risk above. It is the
  best any linear map can do, and by Proposition 1 what Gaussian closure computes (`theory.lmmse_posterior_mean`).
- **Isotropic reference**: the posterior mean of an i.i.d. $\mathcal N(0,1)$ prior, $e^{-t}\mathbf x$ — the error of
  ignoring the chain.
- **risk / floor**: the ratio reported in the field report's scoreboard and figure 6; below one means the arm uses
  structure beyond second order.

## 3. Learning: selection, failure, diagnostics

- **Selection rules** (`selection` column). `risk`: the checkpoint with the lowest pooled risk on the validation bundle
  (uses the reference, so it is an oracle selection, reported as such). `evidence`: the checkpoint with the highest
  held-out log evidence per edge of noisy validation chains under the fitted kernel (`e_step_multi(...).log_evidence`
  / edges), which needs no reference.
- **at_cap** = 1 when the selected checkpoint is the last one saved: the arm may still have been improving, and its
  number is an upper bound for a longer ladder.
- **failed** = 1 (E33 amendment 3): a `FloatingPointError` inside EM or test scoring; risk is NaN and the row counts
  against every prediction.
- **parked** (`tools/field_diagnostics.py`): the fitted kernel's own first-site law — the eigensolve the estimator
  used, or $\mathcal N(0,1)$ when it raised — puts more than $0.5$ of its mass beyond $|a|>3.5$
  (`invariant_mass_out`); `eig_failed` flags the fallback; `gz_mass_out` the mass of $\pi=gZ$ (information only).
- **Physics readings** (`pair_energy_kernel.potential_readings`): $\hat\Theta=\hat\gamma s^2/h$; the barrier
  $\hat V(0)-\min\hat V$ and the well $\arg\min_{\phi>0}\hat V$ of $\hat V(\phi)=(2\hat\Theta/h)\hat f(\phi/s)$ after
  removing its mean offset from $V$ on $|\phi|\le1.6$; the RMS deviation of $\hat V$ from $V$ there. True values
  $\Theta$, $1/4$, $1$.

## 4. Physical statistics of a batch of chains (E34, E37)

From `experiments/e34_field_physics.py::physical_statistics`, on standardised chains $a=\phi/s$:

- **kink density** (per bond): fraction of neighbouring pairs with a sign change; per unit length, divided by $h$.
- **coarse kink density / domain**: every interior run of constant sign shorter than the kink width $\sqrt2$
  ($\lceil\sqrt2/h\rceil$ sites) is merged into its neighbours, iterated (`_coarse_runs`); kinks are then the boundaries
  between domains at least one kink width long. The raw sign-change density is not a physical observable under
  refinement (it grows like $h^{-1/2}$; E34 P1).
- **domain length**: interior runs only (runs touching a chain end are censored and counted separately).
- **$C(r)$**: the lag-$r$ mean of $a_ia_{i+r}$ over chains and sites, with a standard error from the scatter of
  per-chain means.
- **excess kurtosis**, **skew** of the pooled marginal; **bimodality coefficient** $(\mathrm{skew}^2+1)/(\mathrm{excess
  kurtosis}+3)$; **barrier mass** $P(|\phi|<1/2)$.
- **correlation length** $\xi=1/\log(1/\rho_1)$ in sites, $\rho_1=\lambda_1/\lambda_0$ from the spectrum.
- **E37 distribution** (`experiments/e37_domain_distribution.py`): coarse domain lengths in physical units; the
  quantiles $q_{10},q_{50},q_{90}$ and the two-sample Kolmogorov–Smirnov statistic and $p$-value against the exact
  sample of the same cell and seed (`scipy.stats.ks_2samp`).

## 5. Cost

- **Inference time** (E24): the median of repeated timed calls of one posterior mean over a batch $B$, divided by $B$
  (`wall_median / batch`, s per sequence); CPU with the BLAS thread count recorded per row; the citable rows are the
  exclusive-node reruns (`hpc_E24x`, N-29).
- **Training time** (E33 `wall`): the wall clock of one fit at the stated $M$, on a shared node — indicative only.
- **Peak memory** (E38): the peak of the current resident set sampled every 0.5 ms during one call, minus the resident
  set just before it (`statm` on Linux, `task_info` on macOS; the `method` column), in a fresh process per
  measurement.

## 6. Grid and sanity checks

- **E32/E11 grid error**: the worst relative score error of the operating grid $(401,8)$ against a finer reference
  $(1601,8)$ or $(1601,12)$ over levels.
- **E39**: per level (not pooled over the schedule) $\|\hat S-S_{\rm ref}\|_F/\|S_{\rm ref}\|_F$ over 256 chains and 32
  sites, against the closed form and against the RTS smoother.
- **edge mass**: the largest fraction of a message's (or belief's) quadrature mass in the outer 5 % of the grid.

## 7. Generation (the paper's E05, E26, E27)

Defined in the paper's appendix and `src/markov_score/gen_metrics.py`: mean log-likelihood gap of generated samples
under the true $P_{t_{\min}}$ (grid evidence), the driving-noise excess kurtosis recovered by the linear filter, the lag
fourth cumulant, and sliced distances; see `paper/sections/tab-generation.tex` for the columns.
