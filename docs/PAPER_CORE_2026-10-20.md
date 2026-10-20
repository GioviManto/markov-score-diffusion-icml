# The paper's core after the call of 19 October 2026: what we keep, where it comes from, what we drop

Written 20 Oct 2026 from the call transcript (`~/Code/Thesis/Allen-Cahn SPDE and Thesis presentation.vtt`, times in
brackets) and the action list made the same day (`~/Code/Thesis/jerome-call-2026-10-19-paper-plan.md`). Rule for this
file: a result enters the paper only if it is simple, understood line by line, and comes from a frozen output. The
paper itself is written by hand in a separate blank LaTeX document [01:21:42, 01:36:15]; nothing here is paper prose.

## 1. The paper in three messages (Jérôme's summary figure) [01:31:08–01:32:18]

1. **Diffusion breaks the Markov property of the data at intermediate times.** The precision matrix of the noisy
   Gaussian chain is tridiagonal at t = 0, dense near t* ≈ 0.08, close to diagonal again at large t.
2. **Belief propagation that knows the chain structure learns the exact score from few samples**, far better than a
   network: on the Laplace chain, EM–BP with 32 sequences is more accurate than a CNN trained on 2,048.
3. **It works on data from a physical model**: the stochastic Allen–Cahn (φ⁴) field. A 5-parameter kernel learned
   through the noise is 3–17% of the linear error; it gives back the temperature and the double well.

Selling point: **data efficiency** and **a model you can read** (5 parameters cannot memorise one sequence), not
speed [01:09:25, 01:23:54]. Outlook: inferring a model from a few noisy time series.

Figure 1 (draft): `reports/core/figures/fig1-summary.pdf`, panels (a) precision snapshots, (b) Laplace chain,
(c) φ⁴ field. Built by `tools/core/figures.py summary`.

## 2. Skeleton (Jérôme's order [01:22:28–01:24:16]) with the result that fills each step

| step | content | result and source | figure |
|---|---|---|---|
| 1 | Problem: networks generalise by learning the score; to understand what they learn we need the exact score | literature only | — |
| 2 | Setting: a stationary first-order chain, noised by OU. Markov → not Markov → Markov again | closed form (Gaussian chain, ρ = 0.85) | 1a |
| 3 | BP computes the exact score of any first-order chain, also with no closed form; Tweedie | method; `src/markov_score/bp_minimal.py` | (schematic) |
| 4 | The grid: why (messages are densities on ℝ), and the check | Gaussian chain vs closed form: 2e-13 from 81 points (E39); φ⁴: 2e-11 at 101 points vs 1601 (E47a) | 2 |
| 5a | EM–BP with the right shape of kernel | Laplace: 25-par. mixture kernel, E30 `em_bp` (= frozen E02a fits, 0.4%) | 1b, 3 |
| 5b | A neural network learning the kernel inside BP | Laplace: MDN 1,516 par., E30 `mdn`; field: E33 `em_mdn`, E51 readings | 3, 4 |
| 5c | A full network for the score, trained by standard DSM | Laplace: CNN (window head), E03 `confirm`; field: E54 | 1b, 1c, 3 |
| 6 | Allen–Cahn field as the real use case | E33 (site energy, MDN), E54 (CNN), E51 (temperature, potential) | 1c, 4 |

## 3. Results we keep (body)

Numbers are score error divided by the linear floor of the same data (1 = the best linear denoiser = Gaussian
closure), mean over 16 seeds unless stated. Risk = pooled relative score error over 12 levels t ∈ [0.05, 3]
(`docs/METRICS.md` §1).

**B1. Noise destroys locality, then restores it.** Gaussian chain ρ = 0.85, L = 24: the mass of |Q_t| beyond the
first off-diagonal peaks at t* ≈ 0.08. Closed form, no experiment. (Figure 1a.)

**B2. Grid BP is exact to round-off with few points on smooth kernels.** Gaussian chain against the closed-form score:
relative error 1.6e-3 at 41 points, 2e-13 from 81 on (E39). φ⁴ base cell against 1601 points: 4.6e-4 at 51,
1.8e-11 at 101, 1e-14 at 401 (E47a). Do not overclaim: on the Laplace kink the error falls algebraically, not
exponentially (4e-5 at 201 points, 9e-6 at 401, against 3201 points on a wider domain; E11), still far below any
learning error (≥ 0.02). The paper uses 401 points. (Figure 2.)

**B3. Gaussian messages ⇔ linear score** (proved; two equations, section 6 Q10 below). Gaussian closure computes
the best linear estimator, −Σ_t⁻¹x. This is the "linear floor" every number is divided by.

**B4. Laplace chain (ρ = 0.85, L = 32): three estimators against the number of sequences** (E30, E03, E28 floor 0.167):

| M | 32 | 128 | 512 | 2048 |
|---|---|---|---|---|
| EM–BP, mixture kernel (25 par.) | 0.59 | 0.33 | 0.22 | 0.14 |
| neural kernel in BP (MDN, 1,516 par.) | 1.26 | 0.85 | 0.53 | 0.39 |
| CNN, standard DSM (5,505 par.) | 1.11 | 0.85 | 0.72 | 0.71 |
| best of six networks (reference only) | 1.11 | 0.72 | 0.47 | 0.36 |

EM–BP at M = 32 beats the CNN at M = 2048; EM–BP at M = 128 (risk 0.055) beats the best of six architectures at
M = 2048 (0.060). The networks see clean sequences and fresh noise at every step; EM–BP sees one noisy copy of each
sequence. On this chain the neural kernel does **not** beat the CNN at small M (it does from M = 512). (Figures 1b, 3a.)

**B5. Where the gain is (per noise level, M = 128, E28).** EM–BP is below the CNN at every level; above t ≈ 1 the
best linear estimator is better than both (the score is almost linear there and the learned kernel's small error
dominates). (Figure 3b.) Not to be hidden: it is the honest shape of the result.

**B6. φ⁴ field, six cells (E33, E54).** At M = 2048 the 5-parameter site energy is 0.03–0.17 of the linear floor on
every cell; the neural kernel 0.23–0.84; the paper's mixture kernel ≈ 1.0 (a linear drift cannot hold a domain); the
CNN: **E54, pending**. At M = 32 the site energy is 0.24–0.28 on the cold cells and above the floor on the two hottest
(1.37, 1.43). Selection caveat: with the oracle-free likelihood rule the site energy is 0.04–0.23 (not 0.03–0.17).

**B7. The fitted kernel reads as physics, also when it is a neural network (E51).** For any kernel of the gradient
class, the mixed second derivative of log K is 1/γ (→ temperature) and −½ log K(a|a) is the site energy (→ potential).
Site energy: Θ̂/Θ = 1.000 at M = 2048 (base cell), barrier 0.248 (true 0.25). Neural kernel: Θ̂/Θ within 10% on all six
cells at M = 2048 (0.91–1.00); its potential is right only where 2Θ/h is small (base cell barrier 0.20; off by 1.5–3.4×
on fine or hot lattices, where V = (2Θ/h) f amplifies the kernel's error). (Figure 4.)

## 4. Results for the appendix (one paragraph or one table each)

| id | result | source |
|---|---|---|
| A1 | log Z is not in closed form for the quartic chain; log λ₀ per site converged to 2e-13 (2401 vs 4801 points); the same solver reproduces the Gaussian closed form to 2e-14; low-Θ limit = one harmonic well (gap 0.08 at Θ = 0.2) | E52, E32 |
| A2 | Correlation length grows like e^{E_kink/Θ} (kink gas), fitted slope 0.988 vs 2√2/3 = 0.943 | E50 kink part |
| A3 | Speciation: site-averaged commitment, t_c 1.61 → 0.46 from Θ = 0.2 to 2; exact commits earlier than its Gaussian twin | E50 |
| A4 | Per-site polarisation: domain bulks commit as blocks, kinks are located last by a short front; no avalanche across kinks | E53 |
| A5 | Spline (25 par.) vs site energy (5 par.): same class, more variance, messy when hot ("the awkward baby") | E33, fig-field-potential-frozen |
| A6 | Full field table: every cell × M × arm, with at-cap shares | E33, E54 |
| A7 | Neural kernel earns its parameters only off-class: on a Laplace-bond chain the MDN beats the site energy (0.056 vs 0.103); MDN numbers not budget-limited (4× budget moves them 0.4–4.5%) | E49, E41 |
| A8 | Oracle-free stopping (held-out likelihood) costs little | E20, E33 evidence rows |
| A9 | Robustness of B4 across ρ ∈ {0.5, 0.85, 0.95}, L ∈ {32, 128} (EM–BP wins 16/16 seeds everywhere) | E14, E13x |
| A10 | Hot-cell barrier error is a sample-size limit (M^-0.54 to M = 32768) | E42 |
| A11 | Cost, with thread count: 101 grid points lose nothing on φ⁴ and are 6× cheaper than 401 | E47, E24x |
| A12 | Grid BP code: the minimal 30-line version equals production at 1e-10 | `tests/test_bp_minimal.py` |

## 5. Results we drop from the paper (kept in the thesis and the registry)

- Fisher/Hermite information decay, sample complexity by order, Edgeworth (E01, E12, E12b, E16, E21): correct but
  not the message (structure, not Fisher information) and hard to explain in a seminar.
- Message-shape ansätze, Fourier representation, FFT transition, moment-projected BP (E22, E22c, E23, E40, E39
  projected arm): answers to questions, not results. One sentence: "a grid is the natural choice and it is accurate".
- The architecture zoo, implicit score matching, matched-parameter ladders, budget sweeps (E25, E29, E31, the ISM
  arms of E03/E14/E33): one network, trained the standard way [01:19:31–01:20:31].
- Legendre tilt (ill-posed, N-33), log-linear unrestricted kernel (E43), stride protocol (E36, E36x, E48), state
  augmentation and the global latent (E35, E35x, E46), long-range coupling (E08) beyond one sentence in "limits".
- Generation by one forward sweep (E26, E27), inpainting (E45), quench (E44), physical statistics and domain-length
  distribution (E34, E37), memory (E38), capacity of the mixture (E02a, E09, E10), phase diagram and regime sweeps
  (E07, E13), learning curves through noise (E06), moment estimators (E19), misspecification bias (E17), wide domain
  (E18), locality floor (E15).

## 6. Jérôme's questions of 19 October, answered

**Q1. Is log Z known exactly?** [00:07:16–00:12:40] log Z_L = L log λ₀ + O(1), λ₀ the top eigenvalue of the transfer
operator. For a quadratic potential λ₀ is in closed form; for the quartic double well it is not (in the continuum
limit the eigenproblem is the Schrödinger equation of the quartic double well, which has no closed-form spectrum).
We compute it to 2e-13 and the same solver reproduces the Gaussian closed form to 2e-14 (E52; log λ₀ = −0.186 per site on the base cell). Literature: Scalapino–Sears–Ferrell 1972, Krumhansl–Schrieffer 1975; Khare–Habib–Saxena 1997 state the φ⁴ transfer-integral equation has no exact solution. So the model is not a
trivial toy, as Jérôme expected. Context: ξ ∝ e^{E_kink/Θ} at low temperature (E50).

**Q2. Why 5 parameters, why stop at fourth order?** [00:15:32–00:16:35] The lattice energy is a quadratic bond plus a
quartic site potential, so a degree-4 polynomial site energy plus one bond variance contains the truth exactly.

**Q3. Does the neural kernel beat a full network?** [00:17:39] On the field, yes at large M: 0.23–0.84 of the floor
against the CNN's (E54, pending; the one-copy CNN of E33 is 0.48–3.3). On the Laplace chain only from M = 512; at
M = 32 the CNN is better (1.11 vs 1.26). The parsimonious kernel beats both everywhere.

**Q4. The CNN's window.** [00:18:17–00:20:28] Radius 2, 4, 8 sites at h = 1, ½, ¼ (the same physical reach, 2),
chosen on development seeds among the radii offered. The correlation length is ≈ 6 sites.

**Q5. How are Θ̂ and the barrier read off the spline? Does it fit W or f?** [00:22:58–00:25:18, 01:38:39] The spline
kernel is K(a'|a) = N(a'; a, γ) g(a')/Z(a): a Gaussian bond with one free variance γ times a free arrival tilt g
(24 B-splines). So it fits the whole pair energy W, but with the bond fixed to be Gaussian. Θ̂ = γ̂ s²/h comes from γ
alone; the site energy is f = −½ log(g/Z) = −½ log K(a|a) + const, and V̂ = (2Θ̂/h) f̂. Both readings are identities
of the gradient class, not parameters of the spline (E51 P1: they agree with the spline's own parameters to 1e-14).

**Q6. What can be exported to the MDN (an effective temperature)?** [00:23:37, 00:27:33] The temperature, yes: the
average mixed curvature of log K gives Θ̂ within 10% on all six cells (E51). The potential, only where the site energy
is a large part of the kernel (small 2Θ/h). The spread of the curvature measures how far the MDN is from the gradient
class (cv ≈ 0.4–1.0); it does not predict the MDN's error well (Spearman 0.24).

**Q7. Figure 2, first row as site energy against the MDN (and the full network).** [00:25:26–00:26:43] Done:
Figure 4 (potential and temperature, site energy vs MDN) and Figure 1c (error vs M, site energy vs MDN vs CNN).

**Q8. Per-site polarisation: avalanche or all at once?** [00:28:06–00:33:07, 01:14:37] Neither, exactly (E53). The
bulk of each long domain commits as a block; the posterior locates each kink last, by a front of a few sites moving
from the bulk to the wall (commitment time 0.44 at the kink, 1.46 ten sites inside, Θ = 0.2); short domains commit
last; nothing propagates across kinks. Appendix at most.

**Q9. Does EM see clean sequences?** [01:15:12–01:17:26] No: EM–BP sees only the noisy sequences (the E-step is BP on
x_t). The DSM networks see the clean sequences in their loss, so the comparison favours the networks. Remark to keep:
the Gaussian noise is artificial for sequence data (one could mask), but the aim is to understand diffusion.

**Q10. Gaussian messages ⇔ linear score: the two equations.** [00:59:04–01:00:32] With Gaussian messages
m_i = N(μ_i, v_i), K(a'|a) = N(ρa, q) and ℓ_i(a) ∝ N(x_i; αa, Δ), one forward step is

  v_{i+1} = ρ² (1/v_i + α²/Δ)⁻¹ + q,      μ_{i+1} = ρ (1/v_i + α²/Δ)⁻¹ (μ_i/v_i + α x_i/Δ).

The variances never involve x; the means are linear in x; the same holds backwards and for the belief. So
E[a|x] = Hx and the score −(x − αHx)/Δ is linear; on a Gaussian chain it is the exact score −Σ_t⁻¹x. Any BP whose
messages are forced to be Gaussian (moment matching, AMP/TAP in this setting) therefore returns a linear estimator,
at best the LMMSE one, which is the linear floor. Proved (thesis Thm 4.3, 5.2); checked numerically by E39.

**Q11. The grid: how many points, without overclaiming.** [01:02:22–01:04:46] See B2. A natural choice for a
continuous variable; very accurate with few points on smooth kernels; slower on a kink; a very irregular law could
need more.

**Q12. Selling point.** [01:09:25] Data efficiency (B4, B6) and readable parameters (B7); 5 numbers cannot memorise
one sequence of 32 values.

**Q13. Networks.** [01:19:31] Standard DSM, one CNN (local data), fresh noise and level at every step, checkpoint on
validation (E03 for the Laplace chain, E54 for the field). Implicit score matching: one remark, as the link to EM.

**Q14. Is the stationary kernel directional?** [00:30:37] No. The field's chain is reversible: π(a)K(a'|a) is
symmetric (`docs/FIELD_MODEL.md`, reversibility), so the chain read left to right or right to left has the same law.

**Q15. Allen–Cahn uses the same grid BP?** [01:13:56] Yes, the same code (401 points on [−8, 8] in units of the
standard deviation); 101 points give the same risk (E47b).

## 7. Code to review by hand [01:33:02]

1. `src/markov_score/bp_minimal.py` (30 lines): the whole algorithm; `tests/test_bp_minimal.py` holds it equal to the
   production `bp_grid.grid_bp_batch` at 1e-10 and to the closed form on the Gaussian chain; the test was shown to fail
   on a planted orientation bug.
2. `src/markov_score/kernel_readings.py` (two identities) and `tests/test_kernel_readings.py`.
3. The M-step of the site-energy kernel (`pair_energy_kernel.py`, eigensolve + perturbation-theory gradient): the one
   non-trivial piece of the estimator. To be read line by line before the paper.
4. `em.py` E-step (pairwise beliefs from the same BP).

## 8. Open, and what is not claimed

- E54 (CNN on the field, standard DSM): running, 8 seeds, 64k steps; the first fits end at the step cap, so the CNN's
  numbers are upper bounds.
- Stopping uses the exact score on validation for EM–BP and the DSM networks; the likelihood rule changes the site
  energy from 3–17% to 4–23% of the floor.
- BP at 401 points is slower per sequence than a network; at 101 points it is as fast as the transformer on one thread.
  Not a selling point.
- Venue and timeline are the author's decision (ICML ≈ 20 Jan 2027; JSTAT/MLST ≈ 3 months; arXiv by end of December
  helps Jérôme's CNRS application).
