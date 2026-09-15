# Research plan

Written 15 September 2026 after the audit (`CURRENT_STATE_AUDIT.md`). Revised at each
milestone; the revision log is at the end.

## 1. The question the paper should answer

> What information about structured data survives diffusion noise, and how much does
> explicit structure reduce the data needed to recover the score and the underlying
> dynamics?

## 2. The candidate headline insight

**Diffusion noise is a low-pass filter on statistical order.** Under the OU channel
every Hermite mode of degree m is attenuated by e^{−mt} exactly (Theorem A), so the
information one noised sample carries about an order-m feature of the data decays as
e^{−2mt} (Theorem B). Correlation lives at degree 2 and decays as e^{−4t}; non-Gaussian
shape first appears at degree 4 and decays as e^{−8t}. Three consequences, each testable
exactly in the Markov-chain laboratory:

1. the score of any distribution and the score of its covariance-matched Gaussian twin
   differ by a term of order e^{−4t} (squared: e^{−8t}) — Proposition 1 identifies this
   twin with Gaussian message closure, so the "price of second-order closure" has a
   rate and a constant (∝ ‖κ_4‖²);
2. learning the innovation *shape* through the channel costs e^{4t} more samples than
   learning the *correlation*, which explains the shape/correlation asymmetry the
   previous manuscript reported as an empirical finding;
3. the crossover to these rates is set by e^{2t} ≈ λ_max(Σ_0) ≤ (1+ρ)/(1−ρ), so a
   strongly correlated chain keeps its correlation information longer (t_* = ½ log
   ((1+ρ)/(1−ρ)) ≈ 1.26 at ρ = 0.85) — the natural axis of a phase diagram.

The learning claim then becomes: **an estimator that carries the correct factorisation
recovers the order-4 structure that unstructured and even locality-aware networks do
not reach at these sample sizes** — the audit's oracle ladder shows the MLP at
M = 4096 sitting on the LMMSE floor (0.179 vs 0.168) while EM–BP sits at 0.018.

If H2 below fails (the C = 8 mixture does not beat a Gaussian innovation on the
headline task), the headline reverts to a theory paper with the learning result scoped
to covariance recovery. That outcome is recorded, not hidden.

## 3. Ranked hypotheses and the experiments that test them

Ranking is by expected gain in acceptance probability per unit of compute. Every
experiment is sized for the local machine (8 cores, 8 GB, no GPU; cluster unreachable
from this session). Cost figures use the measured 0.35–0.6 ms per chain per level for
grid BP at N_g = 401 and 0.12–0.2 s per E-step over 256 chains.

### P0 — must resolve

**H1 (theory).** Theorems A and B hold; the rates are 4 (degree 2) and 8 (degree 4)
with the stated constants and the λ_max crossover.
*E1a* closed-form Gaussian-chain Fisher information I(ρ), I(q), efficient and
diagonal, for ρ ∈ {0, 0.3, 0.6, 0.85, 0.95}, t ∈ [0.02, 6] — seconds.
*E1b* shape information on generalised-Gaussian chains (β ∈ {1, 1.25, 1.5, 2, 3}) by
BP + Fisher identity at N_g = 801, twelve levels to t = 3, D = 512 chains, **8
replicate seeds** (the frozen run had one), with the Gaussian control (β = 2) checked
against E1a to locate the numerical floor — ~30 min.
*E1c* closure gap (relative score error, posterior-mean MSE, 1 − cos) for the five
families at N_g = 801 to t = 4, paired draws across t — ~15 min.
*E1d* scalar fine-quadrature checks of every constant (done for three; extend to the
tensor constant of the chain by direct evaluation of κ_4 contractions).
Falsifier: slopes that do not approach −4/−8 beyond t_* on the Gaussian control, where
the closed form leaves no numerical excuse.

**H2 (capacity).** On the headline task the mixture kernel's pooled error is below the
LMMSE floor (0.168) at every M, and a C = 1 kernel is not.
*E2a* refit EM–BP at C ∈ {1, 2, 4, 8} on the exp_31 bundles, M ∈ {32, 128, 512, 2048},
16 seeds, validation-selected checkpoints, headline scoring; report with the oracle
ladder (isotropic, LMMSE fitted ρ, LMMSE true ρ, exact) on the same test bundles —
~4 h wall on 4 workers.
*E2b* the one-parameter family: generalised-Gaussian innovations β ∈ {1, 1.25, 1.5, 2,
3, 6} (excess kurtosis +3 … −1.1) at fixed ρ = 0.85 and variance; measure exact-BP vs
LMMSE gap (deterministic), EM–BP(C = 8) vs C = 1 at M = 512 (16 seeds), Hellinger
recovery, and the window head's error — ~3 h wall.
Falsifier: C = 8 not below the floor ⇒ the previous Appendix L stands and the narrative
reverts (see §2).

**H3 (baselines).** Under a fair, identical training recipe the long-range
architectures (dilated TCN, bidirectional recurrence, small transformer, 1-D U-Net)
close part of the gap to EM–BP; the remaining gap is what explicit probabilistic
factorisation buys beyond architecture.
*E3* PyTorch (CPU) reimplementation of all arms with one recipe: per-example noise
level from the twelve, batch 64, Adam with grad-clip 1, LR ∈ {1e-3, 3e-3}, both
parameterisations, checkpoint ladder to 64k steps of one run, three restarts averaged
in the screen. Screen on dev seeds 10000–10003 at M ∈ {32, 2048}; confirm the best
three plus the window head on seeds 0–15 at M ∈ {32, 128, 512, 2048}. Parity test: the
torch MLP reproduces the NumPy MLP's validation curve within seed noise. Report
parameters, steps, wall-clock, selection, risk ± s.e. — screen ~6 h, confirm ~6 h.
Answers: what weight sharing buys (MLP → window), what locality buys (window radius),
what global propagation buys (TCN/transformer/U-Net/biRNN → window), what the
factorisation buys (EM–BP → best network).

**H4 (numerical refinement).** The headline cells move by < 1% under N_g = 801 and
< 2% under 1601 with the actual fitted kernels and held-out bundles.
*E4* re-score the E2a fits at M ∈ {32, 2048} on the same bundles at N_g ∈ {401, 801,
1601}, A ∈ {8, 10} — ~1.5 h.

**H5 (generation).** Score accuracy transfers: samplers driven by EM–BP reproduce the
excess kurtosis, fourth-order lag cumulants and the true-law log-likelihood of the
data; LMMSE and the networks do not.
*E5* reverse SDE with common random numbers across arms (exact BP, EM–BP
validation-selected, LMMSE, best two networks), 400 steps after a *paired* step study
using `nested_brownian_path` (the unwired fix), 2000 chains, M ∈ {128, 2048}, 4 seeds.
Predeclared metrics: innovation excess kurtosis; marginal 1-D Wasserstein; sliced
Wasserstein in R^L; energy distance; covariance Frobenius error; lag-1…5
autocorrelation error; lag-0/1 fourth-order cumulant; histogram KL; and mean
log-likelihood under the known P_0 (available in closed form because the chain is
known). Sampler discretisation is reported separately from score error via the exact
arm — ~6 h.

### P1 — high value

**H6 (learning through noise).** Fitting from data observed at a single level t, the
RMSE of ρ̂ grows as e^{2t} and that of κ̂_4 as e^{4t} beyond t_*.
*E6* M = 512, eight levels, 16 seeds, EM to a validation-selected checkpoint,
C = 8 — ~1.6 h. This is the bridge between the theorem and the learning story and a
candidate main-text figure.

**H7 (phase diagram).** The structure advantage is a function of (t − t_*(ρ), κ_4, M).
*E7a* oracle ladder over ρ ∈ {0, 0.3, 0.6, 0.85, 0.95} × five families × twelve levels
(deterministic, minutes) → the LMMSE-floor surface. *E7b* EM–BP vs window head at
ρ ∈ {0.3, 0.6, 0.95}, M ∈ {128, 2048}, 8 seeds — ~4 h.

**H8 (misspecification).** The advantage degrades gracefully in a measurable ε and
reverses at a locatable ε*; against the window head, not only the MLP.
*E8* ε-families with exact reference scores: Gaussian rank-one global latent (β),
Gaussian long-range precision coupling (γ), AR(2) truth with AR(1) fit (exact score by
the pair-state chain or by linear algebra for the Gaussian case), site-dependent ρ,
heteroskedastic innovations; M ∈ {512, 2048}, 8 seeds; separate the statistical gain
(small M) from the asymptotic bias (large M) — ~5 h.

**H9 (length generalisation).** A kernel fitted at L = 32 transfers to L ∈ {64, 128,
256} with no loss for EM–BP and the window head; global architectures degrade.
*E9* evaluation-only for EM–BP/window; for the torch arms, the length-agnostic ones
(TCN, transformer without learned absolute positions, U-Net) evaluated at longer L —
~2 h.

**H10 (information-matched arm).** A network trained by implicit/sliced score matching
from the *same* one-view noisy bundle EM–BP sees lands far above both EM–BP and the
DSM network, isolating the value of clean data from the value of structure.
*E10* torch, exact-trace implicit score matching for the window head (cheap
Jacobian trace) and Hutchinson for the best global net; validation on the SM objective
on a held-out noisy bundle; plus a one-corruption-per-chain DSM arm — ~4 h.

### P2 — only if it helps

**H11** learning-curve reanalysis with per-replicate slopes and censoring diagnostics
(reanalysis of E2a/E3 outputs; cheap). **H12** additional families after conclusions
stabilise. **H13** adaptive quadrature — only if E4 shows movement.

## 4. Pre-registration

Before each confirmatory run: exploratory runs on dev seeds → metrics, grids, seeds and
comparisons frozen in `experiments/configs/<id>.yaml` with a registry entry in
`docs/EXPERIMENT_REGISTRY.md` (status `PRE-REGISTERED`, config SHA-256) → confirmatory
seeds 0–15 untouched until then. Dev seeds are 10000–10003 throughout. Selection is on
per-seed validation bundles disjoint from test (the exp_31 discipline, not exp_07's
shared bundle).

## 5. Compute budget

| block | wall-clock (4 workers) | when |
|---|---|---|
| E1 (theory) | < 1 h | first |
| E2a + E2b (capacity) | ~7 h | night 1 |
| E6 (learning through noise) | ~2 h | night 1 |
| E4 (refinement) | ~1.5 h | day 2 |
| E3 screen + confirm | ~12 h | night 2 |
| E5 (generation) | ~6 h | day 3 |
| E7, E8, E9, E10 | ~15 h | nights 3–4 |

Memory: grid BP at N_g = 1601 with 2048 chains needs ~3.5 GB; E4 runs in chunks.

## 6. Milestones

01 import-and-audit (this document; done) · 02 reproducibility (CI, registry, check
script) · 03 theory-information-decay (E1, Theorem B written, Lean targets) ·
04 nongaussian-attribution (E2, E6) · 05 strong-baselines (E3, E10) ·
06 misspecification-and-scaling (E4, E7, E8, E9) · 07 generation-validation (E5) ·
08 icml-rewrite · 09 final-audit.

## Revision log

* 2026-09-15 — first version, after the audit.
* 2026-09-15 06:30 — E01 done and verified (rates −4/−8, constant to 6%); E02a confirmatory running; E06, E02b queued; E03–E08 scripted and smoke-tested; theorem positioned against Katsevich–Bandeira / Balanov et al. / Aarts et al. after the literature sweep (docs/RELATED_WORK_SWEEP.md).
