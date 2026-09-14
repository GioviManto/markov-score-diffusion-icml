# Results — current state

**Giovanni Mantovani · MSc thesis, Bocconi · last updated 4 September 2026**

The single current record of what is established, what is withdrawn, and what is
open. Every number here is read from the generated files in
`overleaf/thesis/sections/`, which are written by scripts from frozen outputs —
not transcribed by hand, so this file and the thesis cannot disagree.

Two earlier reports, `research/nongaussian-bp/RESULTS_FOR_JEROME_AND_MARC.md`
(12 Aug) and `EXECUTIVE_SUMMARY.md` (13 Aug), are **frozen point-in-time
snapshots** and are correct as such. Do not send them and do not edit them; they
carry their own change tables. This file supersedes both.

---

## 1. The result in one paragraph

For data with sequential structure, the diffusion score is not something that
has to be learned. A locally specified Markov law induces a *globally* dependent
joint score, and that score is nonetheless computable exactly by local inference
on the latent chain, because coordinatewise noising multiplies the prior by
unary factors: it reweights nodes without creating edges, so the posterior
factor graph is still a chain, a chain is a tree, and sum–product on a tree is
exact. The same recursion that computes the score also yields the pairwise
posterior statistics needed to *learn* the transition law when it is unknown.
Against score-matching networks trained on identical data, the structured
estimator is 7.0–17.6× more accurate with 24 free parameters — and that
advantage has a measurable end.

---

## 2. Established results

### 2.1 Exact scores, known dynamics

| Result | Status | Where |
|---|---|---|
| Joint score of the noised stationary Gaussian AR(1) chain, closed form, diagonalised in one eigenbasis for all `t` | exact | thesis ch. 5 |
| Band-fill law `(Q_t)_{i,i+d} = (-1)^{d-1}(2t)^{d-1}(Q_0^d)_{i,i+d} + O(t^d)` at small `t` | exact, small-`t` asymptotic | ch. 5 |
| Return to isotropy at rate `e^{-2t}` | exact | ch. 5 |
| BP on the chain ≡ Kalman filter + RTS smoother, update for update | exact | ch. 6 |
| BP score matches the matrix score to double-precision rounding | numerical | ch. 6 |
| Influence of evidence at distance `d` decays as `q(α,t)^d` | exact | ch. 6 |
| Radius-`r` windowed estimator error decays at the same rate, `r = 2…13` | **measured, not derived** | ch. 6 |
| Truncating the inference beats truncating the score matrix at small `t` (12.3% vs 21.0% at `t = 0.05`) | numerical | ch. 6 |

The last two are deliberately separate claims. Windowing changes the boundary
conditions of the recursion in a way the influence-coefficient argument does not
cover, so the rate agreement is a numerical law over the tested range and not a
corollary.

### 2.2 The propagator question, settled

The project began from the conjecture that a propagator `P_t` maps one block of
the joint score to the next. The answer:

- **The propagator exists and is exact — it is the RTS smoother.** Verified to
  machine precision (~1e-15) against the precision-matrix score.
- **The naive row-shift reading is only asymptotic.** The precision is Toeplitz
  in the bulk but not at the boundaries; deviation falls from ~9% (short chain,
  low noise) to ~0.02% (long chain, high noise).
- **No exact bond factorisation exists for `L ≥ 2`.** The two-frame cross term
  is `−αΔ_t`, zero only when `α = 0` or `t = 0`.

So the propagator is a smoother, not a shift — which is the more useful answer,
because a smoother generalises to non-Gaussian innovations (BP) and to unknown
kernels (EM).

### 2.3 Beyond Gaussianity

| Result | Value | Status |
|---|---|---|
| Moment-projected Gaussian messages return **exactly** the LMMSE estimator of the covariance-matched Gaussian model | — | theorem |
| …and the same estimator for every innovation law with matching first two moments | — | theorem |
| Median relative score error, Laplace innovations, `t = 0.05` | 0.24 | numerical |
| …below `10⁻²` by | `t ≈ 1` | numerical |
| …below `10⁻³` by | `t ≈ 2.1` | numerical |
| Grid-BP reference, interior error | 9.6 × 10⁻⁴ | validated |
| Grid-BP reference, edge error | 1.9 × 10⁻² | validated |

The theorem matters more than the numbers: it means the measured gap is the
price of discarding everything past second moments, and **not** the error of one
implementation.

### 2.4 Learning the dynamics

Fisher's identity turns one forward–backward BP pass into the exact gradient of
the marginal log-likelihood — no differentiation through the recursion. The
E-step is structurally exact for any innovation law; the implementation adds
controlled grid quadrature and a generalised M-step.

**The convergence result, which is a finding and not a diagnostic:**

| Quantity | Median updates to settle | Max |
|---|---|---|
| Autoregressive coefficient `ρ` | **80** | 130 |
| Innovation variance | 68 | — |
| Innovation **shape** | **229** | 638 |

Ratio 2.8×, at 16 seeds, `N = 512`, `C = 8`, cap 800. At the 40-iteration budget
used by earlier work, **105 of 112 shape coordinates had not settled (94%)**. A
stopping rule read off the correlation trace therefore certifies a kernel that
has not converged in the one coordinate every non-Gaussian claim depends on.
This is what caused the two withdrawals in §3.

### 2.5 What supplying the structure is worth

**Against an unstructured network** (validation-tuned MLP, identical data):

| | |
|---|---|
| Advantage range | **7.0–17.6×** lower relative score error |
| Across | 8 training-set sizes, 16 seeds |
| EM–BP free parameters | 24 |
| Largest training set | 4,096 sequences |
| Validation-selection audit | 13.07× under validation selection vs 13.06× under test-set oracle |

The selection audit matters: the advantage is not an artifact of how each arm
chose its stopping point. Both arms select on a disjoint validation bundle;
giving each its test-set optimum instead moves the mean advantage by 0.01×.

**Against a structure-aware baseline** (weight-shared window head, chosen by a
screen over 32 configurations on disjoint seeds):

| | |
|---|---|
| Advantage range | **2.3–6.1×** |
| Cells where EM–BP wins | 192 of 192 |
| EM–BP parameters vs window head | 22 vs 27 |
| Screen winner | radius 4, width 64, lr 0.003, `eps` parameterisation |

Under four alternative aggregations the floor stays at 1.98× with spread 0.48,
so the direction and order of magnitude do not depend on the averaging choice.

### 2.6 Where the advantage ends

Two controlled departures from the Markov assumption:

| Mechanism | Range tested | Advantage at max violation | Break-even |
|---|---|---|---|
| Rank-one global latent, strength `β` | 0 → 1 | 1.06× (from 16.4×) | not reached |
| Long-range precision coupling, `γ` | 0 → 0.4 | **0.45×** (network wins) | **γ ≈ 0.1** |

The value of the structural assumption is exactly its correctness, and this puts
a number on where it expires. Reporting this is what makes the comparison
honest rather than a strawman.

### 2.7 Mixture capacity — resolved

Rerun to a convergence criterion at 16 seeds after the earlier design was
withdrawn. At `N = 128` the verdict is **resolved**: adding components past the
best gives log-evidence −4.4 × 10⁻⁴ with CI excluding zero, and the direction is
*opposite* to what the earlier, smaller, unconverged design suggested. At
`N = 512` the question remains **unresolved** (11 of 32 coordinates unsettled at
the 1,200-iteration cap). Certified provenance: `exp_32`, passes
`require_clean` unmodified.

### 2.8 Marginal blindness — the rotating ring

For a planar trajectory near a circle, the rotation rate carries **exactly zero
Fisher information in every one-frame marginal** and positive information in the
trajectory. A per-frame model cannot estimate it at all, however much data it
sees. This is the argument for the joint score in a single exact statement, and
it is why the ring now opens the research narrative.

For the stationary Gaussian chain the corresponding statement is that the
one-frame marginals are `α`-free outright; the argument does not extend past
Gaussian innovations, which is why the ring is needed.

### 2.9 Speciation and collapse — computed, newly used

Previously computed and unused; now the backbone of thesis ch. 2.

- **Speciation cascade.** A Gaussian hierarchical prior has one covariance
  eigenvalue per level, hence a *ladder* of speciation times. Six predicted from
  the spectrum alone against six measured: **worst-case disagreement 3%**,
  coarse-to-fine ordering.
- **Collapse.** Empirical-score dynamics collapse onto a training point past
  `t_C`; measured against the closed-form excess-entropy criterion across three
  chain lengths. Postponing collapse costs exponentially many sequences.
- **Why BP does not memorise.** The empirical score's sufficient statistic is
  the training set (`N × n` numbers, growing with both). A BP score's is the
  fitted kernel — 24 numbers, independent of dimension — reached only through an
  average. There is no training point for a trajectory to collapse onto, so the
  memorisation axis does not exist for it at any `N`.

---

## 3. Withdrawn

| Claim | Why | Where it lives now |
|---|---|---|
| Reverse-generation / FID comparison | Fits used a fixed 40-iteration budget, far shorter than the shape needs; scores converged estimators against unconverged ones | compendium; thesis ch. 11 lists it as the open question |
| Earlier capacity design ("saturates near eight components") | Same cause, too few seeds | superseded by §2.7, which reverses the direction |
| Ratio "9–14×" | Both arms now select their budget on a disjoint validation bundle; previously EM's was selected and the network's pinned | superseded by §2.5 |
| "Twelve free parameters" | Correct at `C = 4`; frozen config has used `C = 8` since the paired sweep | now 24 |
| AMP/TAP bulk fixed point, existence boundary, `α_c = √2 − 1` | Correct, but answers none of the six research questions | compendium |

**Number drift to watch.** The headline ratio has moved twice: 9–14× → 7.0–12.5×
→ **7.0–17.6×**. Anything quoting the older figures is stale. Read
`overleaf/thesis/sections/efficiency-numbers.tex`, never a prose file.

---

## 4. Open

1. **Reverse generation with converged fits.** Select both arms on a convergence
   criterion rather than a fixed iteration count, then compare. This is the one
   question the thesis leaves open rather than answers.
2. **Capacity at `N = 512`.** 11 of 32 coordinates unsettled at the current cap.
3. **Beyond chains.** BP is exact here because a chain is a tree. Nothing here
   speaks to general graphs.
4. **Real data.** Nothing here speaks to images, video, or practical
   architectures.

---

## 5. Where things are

| | |
|---|---|
| Thesis source | `overleaf/thesis/` — self-contained, 144 pages, builds with `./check.sh` |
| Live numbers | `overleaf/thesis/sections/*.tex`, generated |
| Thesis figures | `overleaf/thesis/figures/`; six built by `tools/make_thesis_figures.py` |
| Claim ledger | `overleaf/compendium/chapters/ch11-claim-audit.tex` |
| Experiments | `research/nongaussian-bp/experiments/` |
| Frozen outputs | `research/nongaussian-bp/outputs/frozen/` |
| Tests | `.venv/bin/python -m pytest -q` — 445 passed, 27 skipped (~14 min) |

**Thesis structure as of 3 September:** 11 chapters plus a new Chapter 2 on the
statistical mechanics of diffusion. Order follows the research: context → model
→ rotating ring → Gaussian chain by matrices → by belief propagation → Laplace →
learning parameters → learning the kernel → comparison → conclusions.
