# Audit computation: the oracle ladder on the variance-matched chains

Computed 14 September 2026 on the M1 laptop with the archival research package
(`Diffusion/research/nongaussian-bp`, HEAD `44d02f8`), script
`tools/audit/oracle_ladder.py` in this repository (a copy of the scratch script that
produced these numbers). 512 chains per family, L = 32, ρ = 0.85, N_g = 401, A = 8,
the frozen twelve-level schedule, seed 12345. Deterministic given the seed.

Three oracles are scored against true-kernel grid BP with the paper's estimand (squared
score error and reference score energy summed over the twelve levels before the square
root):

* **isotropic**: S(x) = −x, the score of N(0, I) — what a model that learned nothing
  about the data beyond its unit variance would output;
* **LMMSE**: S(x) = −Σ_t^{-1} x with the *true* Σ_0 — by Proposition 1 exactly what a
  Gaussian-innovation kernel with the true ρ computes;
* **true-kernel BP**: the reference (error 0).

| family | excess kurtosis of ε | pooled rel. error, isotropic | pooled rel. error, LMMSE |
|---|---|---|---|
| Laplace | +3.0 | 0.670 | **0.168** |
| Student-t(5) | +6.0 | 0.668 | 0.158 |
| uniform | −1.2 | 0.671 | 0.149 |
| bimodal (κ = 0.6) | −0.72 | 0.663 | 0.082 |
| Gaussian | 0 | 0.662 | 0.000 (3e-15) |

Per-level LMMSE relative error on the Laplace chain:

| t | 0.050 | 0.072 | 0.105 | 0.153 | 0.222 | 0.322 | 0.467 | 0.677 | 0.982 | 1.425 | 2.068 | 3.000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rel. error | 0.282 | 0.226 | 0.173 | 0.132 | 0.093 | 0.067 | 0.041 | 0.023 | 0.011 | 3.1e-3 | 3.7e-4 | 1.1e-5 |
| ‖S‖²/L | 3.79 | 3.20 | 2.61 | 2.16 | 1.84 | 1.52 | 1.26 | 1.14 | 1.06 | 0.99 | 1.00 | 1.00 |

(The per-level median in the paper's Figure 1 is 0.24 at t = 0.05; the energy-weighted
value here is 0.28. The pooled number is dominated by the low levels because the
reference energy is largest there.)

Fitted log-slopes of the LMMSE relative error over t ≥ 0.9: −3.46 (Laplace), −3.26
(Student), −3.15 (uniform), −3.34 (bimodal); consecutive-interval slopes on the last
two intervals: −3.3, −3.8 (Laplace). Prediction for t → ∞: −4 (see
`THEORY_NOTES.md`, §2). Fitted slope of the isotropic gap: −1.87 (prediction −2).

## Reading against the paper's tables

| estimator | pooled rel. error | where |
|---|---|---|
| isotropic (learned nothing) | 0.670 | this audit |
| MLP, M = 32 | 0.602 | Table 1 |
| MLP, M = 4096 | 0.179 | Table 1 |
| **LMMSE oracle, true ρ** | **0.168** | this audit |
| window head, M = 2048 | 0.127 | Table 3 |
| EM–BP, M = 32 | 0.090 | Table 1 |
| EM–BP, M = 2048 | 0.026 | Table 1 |
| EM–BP, M = 4096 | 0.018 | Table 1 |
| true-kernel BP | 0 | reference |

Consequences:

1. A Gaussian-innovation kernel (C = 1) with any ρ cannot go below 0.168 on this
   task, by Proposition 1. Table 1's EM–BP is 2–9× below that floor at every M. The
   C = 8 mixture is therefore recovering higher-order structure, and Appendix L's
   attribution of the margin to "the Markov factorisation and its covariance, and not
   to the mixture's capacity" cannot be right for the headline task.
2. The unstructured MLP at the largest size lands almost exactly on the LMMSE floor:
   at these sample sizes it has learned second-order structure and essentially
   nothing beyond it. The window head at M = 2048 is 25% below the floor.
3. The natural headline figure is therefore a *ladder*: isotropic → learned networks →
   LMMSE floor → EM–BP → exact, with the LMMSE floor drawn as a horizontal line that
   Proposition 1 makes exact.
