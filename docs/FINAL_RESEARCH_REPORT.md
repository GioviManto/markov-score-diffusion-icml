# Final research report (living document; last updated 15 September 2026, 06:50)

This file is rewritten at the end of the project. Until then it records, milestone by
milestone, what changed scientifically, so that an interruption never loses the
reasoning. Sections marked *pending* await a registered experiment.

## 1. What changed scientifically

**The paper's central object moved from an estimator to a law.** The starting
manuscript's contribution was a careful laboratory ("the score is exact here") plus a
classical estimator (EM around exact inference) and a sample-efficiency comparison. The
continuation puts a theorem at the centre: under the Ornstein–Uhlenbeck channel every
Hermite mode of degree m is attenuated by e^{−mt} exactly, so the Fisher information a
noised sequence carries about order-m structure decays as e^{−2mt}, and the gap between
the true score and the score of the covariance-matched Gaussian — which Proposition 1
identifies with Gaussian message closure — is (1/6)‖κ₄‖²_F e^{−8t} with a crossover
t_* = ½ log λ_max(Σ₀). The laboratory is what makes the law testable to the constant.

**Honest positioning.** The literature sweep (`RELATED_WORK_SWEEP.md`) found the
SNR^k information law in Katsevich–Bandeira (2023) and Balanov–Bendory–Edidin (2026)
and the cumulant attenuation in Aarts et al. (2025). The paper therefore claims the
score-level statement with its constant, the crossover, the exact verification, and the
learning consequences — not the law itself.

## 2. New results obtained

| result | status | evidence |
|---|---|---|
| Exact Gaussian-chain information curves reach slope −4.00 and the predicted constant at every ρ; approach governed by t_* | done | `outputs/E01/gaussian_information.csv`, `sections/theory-numbers.tex` |
| Shape information slope → −8 for the smooth members of the generalised-Gaussian family (−7.94 at β = 2, −8.01 at β = 3); Laplace member floors at t ≈ 2.5 | done | `outputs/E01/shape_information.csv` |
| Closure-gap constant ‖κ₄‖²_F/6 reproduced to 1–6% on the Laplace and bimodal chains; Student-t(5) departs (infinite sixth moment); uniform reaches double precision near t = 3.5 | done | `outputs/E01/closure_gap.csv` |
| The oracle ladder: LMMSE floor 0.168 (Laplace, pooled); the archival MLP at M = 4096 sits on it; the archival EM–BP far below | done | `AUDIT_ORACLE_LADDER.md`, `sections/oracle-numbers.tex` |
| Capacity on the headline task under identical checkpoint selection (C ∈ {1,2,4,8}, 16 seeds) | *running* (E02a; first cells: C = 1 = floor at every M; C = 2 already 2× below at M = 32) | `outputs/E02a/` |
| Learning through noise: RMSE(ρ̂) ∝ e^{2t}, RMSE(κ̂₄) ∝ e^{4t} | *queued* (E06) | — |
| Tunable non-Gaussian family: C = 8 vs C = 1 gap tracks the exact closure gap | *queued* (E02b) | — |
| Fair baselines (six architectures, one recipe), information-matched arm, length extrapolation | *pending* (E03) | — |
| Generation with paired noise and the exact-evidence log-likelihood metric | *pending* (E05) | — |
| Refinement on the actual fitted models | *pending* (E04) | — |
| Phase diagram of the LMMSE floor over ρ: the pooled floor falls from 0.29 (ρ = 0) to 0.085 (ρ = 0.95) on the Laplace chain; families order by κ₄ at every ρ | done | `outputs/E07/ladder.csv`, `paper/figures/fig_phase.pdf` |
| Misspecification against a locality-aware baseline | *pending* (E08) | — |
| Lean: Mehler identity n ≤ 4, κ₄ difference, Gaussian information constant, scalar score-gap coefficient | *in progress* (`lean/ThesisAudit/PaperTheory.lean`) | — |

## 3. Original claims: survived, weakened, strengthened, removed

* **Survived:** tree-exactness and the validated grid reference; Proposition 1;
  Proposition 2; the convergence asymmetry (now explained by Theorem 2 rather than only
  measured); the headline advantage over the MLP (pending the fair re-screen for the
  window head).
* **Strengthened:** "the channel removes more information about shape than about
  correlation" went from a one-seed pre-asymptotic measurement in a companion document
  to a theorem with constants, verified on the chain.
* **Weakened / reframed:** "against a screened window head the advantage is 2.3–6.1×" —
  the screen was unequal; the number is retained as archival until E03 replaces it.
* **Removed:** "the margin is earned by the Markov factorisation and its covariance, not
  by the mixture's capacity" (Appendix L of the starting manuscript) — contradicted by
  the LMMSE floor and by the first E02 cells; its origin (asymmetric stopping) is
  documented in `NEGATIVE_RESULTS.md`.

## 4. Headline theorem (current)

Theorem 2 of the manuscript (`paper/body/03-theory.tex`), with the crossover paragraph
and the sample-complexity corollary.

## 5. Headline experiment (current)

Figure 2 (`paper/figures/fig_theory.pdf`): exact information curves reaching the
asymptote after t_*, the two slopes −4 and −8, the closure constant reproduced. The
candidate second headline is the ladder figure (E02a + E03).

## 6. Remaining risks before submission

1. E03 must show the fair-recipe baselines; if a global architecture closes most of the
   gap, the "explicit factorisation buys X" claim must be re-scoped.
2. The appendix proof of Theorem 2 carries an informal remainder bound; the Lean targets
   cover the algebra only.
3. Compute: everything runs on one laptop; the confirmatory queue takes about a day.
4. The 8-page body is not yet cut to length; the experiments section is placeholders.
5. ICML 2027 kit not yet published; the 2026 kit is provisional.
