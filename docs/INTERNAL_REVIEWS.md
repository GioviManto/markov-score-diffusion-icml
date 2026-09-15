# Internal adversarial reviews

Three reviewers impersonated at each milestone. Each review names the strongest
objection, what would neutralise it, and whether the current plan does. Reviews are
appended, never edited, so the record shows what was believed when.

---

## Milestone 01–03 (15 September 2026): audit done, theory verified, manuscript skeleton

### Reviewer 1 — theory

**What is actually new mathematically?** The Hermite attenuation (Theorem 1) is
Mehler's formula; the cumulant version is in Aarts et al. (2025). The "information about
order-k structure scales as SNR^k" statement is Katsevich–Bandeira (2023) and, in
Fisher-information form with a Hermite expansion, Balanov–Bendory–Edidin (2026). The
authors' draft now says so in the theorem's own paragraph and in related work, which is
the right call; a version that presented Theorem 2 as new would have been caught.

**What survives as new:** (i) the score-level statement — the gap between the true score
and its covariance twin is (1/6)‖κ₄‖²_F e^{−8t} with an explicit constant, and by
Proposition 1 that gap is exactly what Gaussian message closure / the Kalman smoother
leaves behind; (ii) the crossover t_* = ½ log λ_max(Σ_0) as the scale at which the rates
set in, which is what makes the theorem usable on correlated data; (iii) verification on a
model where the score is exact, to the constant (0.99–1.06 at t ≥ 3 for Laplace).

**Hidden assumptions.** The remainder bound (1 + O(e^{−2t}λ_max)) is asserted, not yet
proved in the appendix; the finite-eighth-moment condition is stated but the tail
argument for |e^{−t}x| > 1 is a sketch. The efficient-vs-raw information coincidence for
the shape coordinate relies on Hermite orthogonality at leading order and is verified
numerically (nuisance cost ≤ 1.15), not proved. **Must fix before submission:** write the
full proof of Theorem 2 with the remainder, or downgrade the O(·) to "numerically the
correction is below 6% for t ≥ 3".

**Does the theorem say what the abstract claims?** The abstract says "Fisher information
… decays as e^{−2mt}" for order-m features. Theorem 2 proves it for m = 2 and m = 4 with
the stated constants; general m is a corollary of the same grading (the leading Hermite
term of degree m has γ-second moment ∝ e^{−2mt}) and should be stated as such, with the
constant left implicit, or the abstract should say "correlation … e^{−4t}, shape …
e^{−8t}". Recommendation: state the general-m version as a remark with a one-line proof
and keep the constants for m = 2, 4.

**Verdict:** the theory is correct and honestly positioned; the appendix proof is the
open item. Score 3/5 on novelty of the mathematics, 5/5 on correctness once the appendix
is written.

### Reviewer 2 — ML experiments

**Are the baselines competitive?** The archival screen was unequal (one noise level per
step for the long-range arms, twelve for the winner; eight restarts vs one; the winner's
learning rate never applied). The plan replaces it with one recipe in PyTorch for six
architectures, three restarts in the screen, per-example noise levels, and confirmatory
seeds untouched. Until E03 runs, every "against a window head" number in the manuscript
is the archival one and must be labelled as such.

**Is the comparison fair?** Two protocols are planned: the standard-recipe comparison
(clean data, fresh corruption) and an information-matched arm (implicit score matching on
the same one-view noisy bundle). The second is the one that answers "is it just the
information interface". It has not been run. The one-corruption DSM arm is a useful
intermediate.

**Does score error translate to generation?** The archival generation study was
withdrawn for unequal convergence; the converged rerun (uncertified) shows the fitted
kernel reproducing the target kurtosis at every capacity and the MLP generating a
Gaussian innovation. E05 with paired noise, validation-selected checkpoints for every arm,
and predeclared distributional metrics (marginal and sliced Wasserstein, energy distance,
lag cumulants, log-likelihood under the known law) is the right design. The
log-likelihood-under-the-known-law metric is the one a reviewer cannot argue with; make it
the headline generation number.

**Capacity.** The oracle ladder settles the direction (a Gaussian-innovation kernel cannot
go below 0.168; Table 1 is at 0.018–0.09), and the first dev cells confirm it (C = 1 at
0.169 for every M; C = 2 at 0.082 at M = 32). The confirmatory E02 must still run on the
untouched seeds before it is reported.

**Strongest objection:** "a one-dimensional chain of length 32 with 24 free parameters".
The length-extrapolation experiment (E09) and the phase diagram over ρ (E07) are the
answers; neither has run.

**Verdict:** the plan addresses the objections; nothing empirical beyond E01 is yet
reportable. Score 2/5 today, provisionally 4/5 if E02, E03, E05 land as designed.

### Reviewer 3 — significance

**Why should ICML care about a one-dimensional Markov chain?** Because it is the only
setting the reviewer knows where a non-Gaussian, correlated, nonlinear score is exact and
the higher-order structure is tunable at fixed covariance. The general lesson —
diffusion noise erases structure order by order at rates fixed by Hermite degree, so
second-order learners are bounded away from the truth by an e^{−8t}-scaled term and
learners of shape pay e^{4t} per level — is about diffusion models, not about chains. The
paper must make that lesson the first sentence of the abstract and the last of the
conclusion, and must connect it to the observation (Wang–Vastola; Li–Dai–Qu) that learned
scores are "unreasonably" Gaussian at high noise: the e^{−8t} gap is why.

**What would make it land:** a single figure (Figure 2) that a reader remembers: the
scaled information curves reaching their asymptote after t_*, the two slopes −4 and −8,
and the closure constant reproduced. That figure exists. The second memorable figure
should be the ladder: isotropic → learned networks → LMMSE floor → EM–BP → exact, which
turns "6.6–10.9×" into "the networks learned second-order structure and stopped".

**Risk:** the paper is read as "known SNR^k law + classical EM". The defence is the
combination — exactness of the laboratory, the score-level constant, the crossover, and
the learning consequences (E06's e^{2t}/e^{4t} separation, which no prior work measures).
E06 is therefore not optional.

**Verdict:** significance 4/5 conditional on E06 and the ladder figure; 2/5 if the paper
ships with the theory alone.

### Actions taken from this review

* Theorem 2's paragraph and related work cite Katsevich–Bandeira, Balanov et al., Aarts
  et al., Wang–Vastola and Li–Dai–Qu; novelty claims narrowed to the score-level
  statement, the crossover, verification and learning consequences (done, `d5005c8`).
* Appendix proof of Theorem 2 with remainder: open (A1-proofs placeholder).
* General-m remark: open.
* E02 confirmatory, E03, E05, E06, E07, E09: scheduled (`RESEARCH_PLAN.md` §5).
