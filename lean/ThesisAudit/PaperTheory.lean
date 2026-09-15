import Mathlib
import ThesisAudit.Ch05b

/-!
# Algebraic lemmas behind the paper's theory (`docs/THEORY_NOTES.md`)

Four targets of `docs/THEORY_NOTES.md` §6, stated so that every hypothesis is a *number*
(a moment, a coefficient, a matrix entry) rather than a measure.  The measure-level
instantiation (that `N(0,1)` has moments `0, 1, 0, 3`; that the Fisher information of a
Gaussian family is the trace formula) is deliberately **not** part of these statements.

| theorem | `docs/THEORY_NOTES.md` | what it checks |
|---|---|---|
| `paper_mehler_he_n` | §1, Theorem A (scalar Mehler identity, `n ≤ 4`) | `E[He_n(αa + βz)] = αⁿ He_n(a)` whenever `E` is ℝ-linear with `E[1,z,z²,z³,z⁴] = 1,0,1,0,3` and `β² = 1 − α²` |
| `paper_he4_moment_difference`, `paper_he4_moment_difference_functional` | §1, "Consequences (exact)", degree-4 bullet | two zero-mean laws with equal second moments have `E He₄ − E' He₄ = κ₄ − κ₄'` |
| `paper_gaussian_fisher_rate` | §2, B3 (exact Gaussian-chain information) | `e^{4t} · ½ tr(Σ_t⁻¹ ∂Σ_t Σ_t⁻¹ ∂Σ_t) → ½ tr(∂Σ₀ᵀ ∂Σ₀) = ½‖∂Σ₀‖²_F` |
| `paper_scalar_score_gap_coefficient`, `paper_Lam4diff_hasDerivAt` | §2, B1 (scalar unit-variance constant `κ₄/6`) | `∂_s(Λ₄ − Λ₄^G)` at `s = αx`, `β = α²` is `(κ₄/6) α³ He₃(x)` |

The Hermite polynomials are Mathlib's `Polynomial.hermite` (probabilists' convention, over
`ℤ`) cast to `ℝ`; the explicit forms `He₀..He₄ = 1, X, X²−1, X³−3X, X⁴−6X²+3` are *proved*
from `Polynomial.hermite_succ` (`paperHe_zero` … `paperHe_four`), so nothing here depends
on a hand-typed polynomial.

No `sorry`, no `axiom`, no `native_decide`.
-/

namespace ThesisAudit

open Polynomial Matrix

noncomputable section

/-! ## Hermite polynomials over ℝ, anchored to Mathlib -/

/-- Probabilists' Hermite polynomial `He_n` as a real polynomial: Mathlib's
`Polynomial.hermite n : ℤ[X]` mapped along `ℤ → ℝ`. -/
def paperHe (n : ℕ) : Polynomial ℝ := (Polynomial.hermite n).map (Int.castRingHom ℝ)

theorem paperHe_zero : paperHe 0 = 1 := by simp [paperHe]

theorem paperHe_one : paperHe 1 = X := by simp [paperHe]

theorem paperHe_two : paperHe 2 = X ^ 2 - 1 := by
  simp [paperHe, hermite_succ]; ring

theorem paperHe_three : paperHe 3 = X ^ 3 - 3 * X := by
  simp [paperHe, hermite_succ]; ring

theorem paperHe_four : paperHe 4 = X ^ 4 - 6 * X ^ 2 + 3 := by
  simp [paperHe, hermite_succ]; ring

/-- Evaluation of the four explicit forms (used to display `He_n(a)` on the right-hand
side of the Mehler identity). -/
theorem paperHe_eval_four (a : ℝ) : (paperHe 4).eval a = a ^ 4 - 6 * a ^ 2 + 3 := by
  rw [paperHe_four]; simp

theorem paperHe_eval_three (a : ℝ) : (paperHe 3).eval a = a ^ 3 - 3 * a := by
  rw [paperHe_three]; simp

theorem paperHe_eval_two (a : ℝ) : (paperHe 2).eval a = a ^ 2 - 1 := by
  rw [paperHe_two]; simp

/-! ## Target 1 — Mehler identity for `n ≤ 4` (THEORY_NOTES §1, Theorem A) -/

/-- Linearity bookkeeping: an ℝ-linear functional on `ℝ[X]` applied to an explicit
degree-`≤ 4` polynomial is the coefficient vector contracted with the "moments"
`E 1, E X, …, E X⁴`. -/
theorem paper_lin_expand (E : Polynomial ℝ →ₗ[ℝ] ℝ) (c0 c1 c2 c3 c4 : ℝ) :
    E (C c0 + C c1 * X + C c2 * X ^ 2 + C c3 * X ^ 3 + C c4 * X ^ 4)
      = c0 * E 1 + c1 * E X + c2 * E (X ^ 2) + c3 * E (X ^ 3) + c4 * E (X ^ 4) := by
  have h0 : C c0 = c0 • (1 : Polynomial ℝ) := by rw [smul_eq_C_mul, mul_one]
  rw [h0, ← smul_eq_C_mul, ← smul_eq_C_mul, ← smul_eq_C_mul, ← smul_eq_C_mul]
  simp only [map_add, map_smul, smul_eq_mul]

/-- `He_n(αa + βz)` expanded as a polynomial in `z`, for `n = 0, …, 4`. -/
theorem paperHe_zero_comp (u β : ℝ) :
    (paperHe 0).comp (C u + C β * X)
      = C 1 + C 0 * X + C 0 * X ^ 2 + C 0 * X ^ 3 + C 0 * X ^ 4 := by
  rw [paperHe_zero, one_comp]; simp

theorem paperHe_one_comp (u β : ℝ) :
    (paperHe 1).comp (C u + C β * X)
      = C u + C β * X + C 0 * X ^ 2 + C 0 * X ^ 3 + C 0 * X ^ 4 := by
  rw [paperHe_one, X_comp]; simp

theorem paperHe_two_comp (u β : ℝ) :
    (paperHe 2).comp (C u + C β * X)
      = C (u ^ 2 - 1) + C (2 * u * β) * X + C (β ^ 2) * X ^ 2 + C 0 * X ^ 3 + C 0 * X ^ 4 := by
  rw [paperHe_two]
  simp only [add_comp, sub_comp, mul_comp, pow_comp, X_comp, C_comp, one_comp, ofNat_comp,
    Nat.cast_ofNat]
  simp only [map_add, map_sub, map_mul, map_pow, map_ofNat, map_zero, map_one]
  ring

theorem paperHe_three_comp (u β : ℝ) :
    (paperHe 3).comp (C u + C β * X)
      = C (u ^ 3 - 3 * u) + C (3 * u ^ 2 * β - 3 * β) * X + C (3 * u * β ^ 2) * X ^ 2
        + C (β ^ 3) * X ^ 3 + C 0 * X ^ 4 := by
  rw [paperHe_three]
  simp only [add_comp, sub_comp, mul_comp, pow_comp, X_comp, C_comp, one_comp, ofNat_comp,
    Nat.cast_ofNat]
  simp only [map_add, map_sub, map_mul, map_pow, map_ofNat, map_zero, map_one]
  ring

theorem paperHe_four_comp (u β : ℝ) :
    (paperHe 4).comp (C u + C β * X)
      = C (u ^ 4 - 6 * u ^ 2 + 3) + C (4 * u ^ 3 * β - 12 * u * β) * X
        + C (6 * u ^ 2 * β ^ 2 - 6 * β ^ 2) * X ^ 2 + C (4 * u * β ^ 3) * X ^ 3
        + C (β ^ 4) * X ^ 4 := by
  rw [paperHe_four]
  simp only [add_comp, sub_comp, mul_comp, pow_comp, X_comp, C_comp, one_comp, ofNat_comp,
    Nat.cast_ofNat]
  simp only [map_add, map_sub, map_mul, map_pow, map_ofNat, map_zero, map_one]
  ring

/-- **Target 1 (THEORY_NOTES §1, Theorem A; scalar Mehler identity, `n ≤ 4`).**

Let `E : ℝ[X] →ₗ[ℝ] ℝ` be any ℝ-linear functional whose values on `1, X, X², X³, X⁴` are the
first four moments `1, 0, 1, 0, 3` of `N(0,1)` (so `E p` plays the role of `E_z[p(z)]`), and let
`β² = 1 − α²` (so `β = √(1−α²)` up to sign).  Then for every `n ≤ 4`

  `E[ He_n(α a + β z) ] = αⁿ · He_n(a)`,

where `He_n(αa + βz)` is the polynomial `(He n).comp (C (α a) + C β * X)` in the variable
`z = X`.  This is exactly the identity `E_z[He_n(αa + √(1−α²) z)] = αⁿ He_n(a)` of
THEORY_NOTES §1, with the Gaussian replaced by its first four moments — the only property
of `N(0,1)` the degree-`≤ 4` identity uses.  Nothing is assumed about `E` beyond linearity and
those four numbers; in particular `E` need not be positive or normalised in any other sense. -/
theorem paper_mehler_he_n (E : Polynomial ℝ →ₗ[ℝ] ℝ)
    (hE0 : E 1 = 1) (hE1 : E X = 0) (hE2 : E (X ^ 2) = 1) (hE3 : E (X ^ 3) = 0)
    (hE4 : E (X ^ 4) = 3)
    (a α β : ℝ) (hβ : β ^ 2 = 1 - α ^ 2) (n : ℕ) (hn : n ≤ 4) :
    E ((paperHe n).comp (C (α * a) + C β * X)) = α ^ n * (paperHe n).eval a := by
  interval_cases n
  · rw [paperHe_zero_comp, paper_lin_expand, hE0, hE1, hE2, hE3, hE4, paperHe_zero]
    simp
  · rw [paperHe_one_comp, paper_lin_expand, hE0, hE1, hE2, hE3, hE4, paperHe_one]
    simp
  · rw [paperHe_two_comp, paper_lin_expand, hE0, hE1, hE2, hE3, hE4, paperHe_eval_two]
    linear_combination hβ
  · rw [paperHe_three_comp, paper_lin_expand, hE0, hE1, hE2, hE3, hE4, paperHe_eval_three]
    linear_combination (3 * (α * a)) * hβ
  · rw [paperHe_four_comp, paper_lin_expand, hE0, hE1, hE2, hE3, hE4, paperHe_eval_four]
    linear_combination (3 * (β ^ 2 + 1 - α ^ 2) + 6 * (α * a) ^ 2 - 6) * hβ

/-! ## Target 2 — degree-4 Hermite moments differ by `κ₄` (THEORY_NOTES §1, consequences) -/

/-- Fourth cumulant of a zero-mean law with second and fourth moments `m2, m4`:
`κ₄ = m₄ − 3 m₂²`.  (Mathlib has no `cumulant`; this is the local definition.) -/
def paperKappa4 (m2 m4 : ℝ) : ℝ := m4 - 3 * m2 ^ 2

/-- Under a zero-mean law with second and fourth moments `m2, m4`,
`E[He₄] = E[a⁴ − 6a² + 3] = m₄ − 6 m₂ + 3`.  Stated through an arbitrary ℝ-linear functional
`E` on `ℝ[X]` (no normalisation or moment values assumed). -/
theorem paper_he4_expectation (E : Polynomial ℝ →ₗ[ℝ] ℝ) :
    E (paperHe 4) = E (X ^ 4) - 6 * E (X ^ 2) + 3 * E 1 := by
  have h : paperHe 4 = C 3 + C 0 * X + C (-6) * X ^ 2 + C 0 * X ^ 3 + C 1 * X ^ 4 := by
    rw [paperHe_four]
    simp only [map_neg, map_ofNat, map_zero, map_one]
    ring
  rw [h, paper_lin_expand]
  ring

/-- **Target 2 (THEORY_NOTES §1, "Consequences (exact)", degree-4 bullet), moment form.**

For two zero-mean laws with moments `(m2, m4)` and `(m2', m4')` and **equal second moments**
`m2 = m2'`, the difference of the fourth Hermite moments `E He₄ − E' He₄`
(`= (m4 − 6 m2 + 3) − (m4' − 6 m2' + 3)`) equals the difference of fourth cumulants
`κ₄ − κ₄'`.  The equal-second-moment hypothesis is the variance-matching constraint of the
paper's chain family; without it the identity is false (the `−3 m₂²` terms would not cancel
against `−6 m₂`). -/
theorem paper_he4_moment_difference (m2 m4 m2' m4' : ℝ) (h : m2 = m2') :
    (m4 - 6 * m2 + 3) - (m4' - 6 * m2' + 3) = paperKappa4 m2 m4 - paperKappa4 m2' m4' := by
  subst h
  unfold paperKappa4
  ring

/-- **Target 2, linear-functional form** (connecting to the `E` of Target 1).

For two normalised ℝ-linear functionals `E, E'` on `ℝ[X]` (`E 1 = E' 1 = 1`) with equal second
moments `E X² = E' X²`, `E He₄ − E' He₄ = κ₄(E X², E X⁴) − κ₄(E' X², E' X⁴)`. -/
theorem paper_he4_moment_difference_functional (E E' : Polynomial ℝ →ₗ[ℝ] ℝ)
    (hE0 : E 1 = 1) (hE'0 : E' 1 = 1) (h2 : E (X ^ 2) = E' (X ^ 2)) :
    E (paperHe 4) - E' (paperHe 4)
      = paperKappa4 (E (X ^ 2)) (E (X ^ 4)) - paperKappa4 (E' (X ^ 2)) (E' (X ^ 4)) := by
  rw [paper_he4_expectation, paper_he4_expectation, hE0, hE'0, mul_one]
  exact paper_he4_moment_difference _ _ _ _ h2

/-! ## Target 3 — Gaussian Fisher-information rate (THEORY_NOTES §2, B3) -/

variable {n : ℕ}

/-- `Σ_t = e^{−2t} Σ₀ + (1 − e^{−2t}) I` — the noised covariance of the OU channel
(`α² = e^{−2t}`, `Δ = 1 − e^{−2t}`). -/
def paperSig (S0 : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Real.exp (-2 * t) • S0 + (1 - Real.exp (-2 * t)) • (1 : Matrix (Fin n) (Fin n) ℝ)

/-- Closed-form Gaussian Fisher information about a covariance parameter `θ`,
`I_t(θ) = ½ tr(Σ_t⁻¹ ∂_θΣ_t Σ_t⁻¹ ∂_θΣ_t)`, with `∂_θΣ_t = e^{−2t} ∂_θΣ₀ = e^{−2t} dS`.
`⁻¹` is Mathlib's `Matrix.inv` (the true inverse whenever `Σ_t` is invertible, see
`paperSig_isUnit_det`). -/
def paperFisher (S0 dS : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) : ℝ :=
  (1 / 2) * Matrix.trace ((paperSig S0 t)⁻¹ * (Real.exp (-2 * t) • dS)
    * (paperSig S0 t)⁻¹ * (Real.exp (-2 * t) • dS))

theorem paper_exp_neg_two_tendsto :
    Filter.Tendsto (fun t : ℝ => Real.exp (-2 * t)) Filter.atTop (nhds 0) := by
  have he : Filter.Tendsto (fun t : ℝ => Real.exp (-t)) Filter.atTop (nhds 0) :=
    Real.tendsto_exp_neg_atTop_nhds_zero
  have h2 : Filter.Tendsto (fun t : ℝ => Real.exp (-t) * Real.exp (-t)) Filter.atTop
      (nhds 0) := by simpa using he.mul he
  refine h2.congr fun t => ?_
  rw [← Real.exp_add]; congr 1; ring

/-- `Σ_t → I` entrywise as `t → ∞`. -/
theorem paperSig_tendsto (S0 : Matrix (Fin n) (Fin n) ℝ) :
    Filter.Tendsto (paperSig S0) Filter.atTop (nhds 1) := by
  have h1 := paper_exp_neg_two_tendsto.smul_const S0
  have h2 := ((tendsto_const_nhds (x := (1 : ℝ))).sub paper_exp_neg_two_tendsto).smul_const
    (1 : Matrix (Fin n) (Fin n) ℝ)
  have h := h1.add h2
  simp only [zero_smul, sub_zero, one_smul, zero_add] at h
  exact h

/-- `Σ_t⁻¹ → I` entrywise as `t → ∞` (matrix inversion is continuous at the identity, whose
determinant `1` is a unit). -/
theorem paperSig_inv_tendsto (S0 : Matrix (Fin n) (Fin n) ℝ) :
    Filter.Tendsto (fun t => (paperSig S0 t)⁻¹) Filter.atTop (nhds 1) := by
  have hc : ContinuousAt Inv.inv (1 : Matrix (Fin n) (Fin n) ℝ) := by
    refine continuousAt_matrix_inv 1 ?_
    rw [Matrix.det_one, Ring.inverse_eq_inv']
    exact continuousAt_inv₀ one_ne_zero
  have h := hc.tendsto.comp (paperSig_tendsto S0)
  rw [inv_one] at h
  exact h

/-- Exact factorisation: `e^{4t} I_t = ½ tr(Σ_t⁻¹ dS Σ_t⁻¹ dS)` — the `e^{−4t}` from
`∂_θΣ_t = e^{−2t} dS` (twice) cancels the prefactor exactly. -/
theorem paper_exp_mul_fisher (S0 dS : Matrix (Fin n) (Fin n) ℝ) (t : ℝ) :
    Real.exp (4 * t) * paperFisher S0 dS t
      = (1 / 2) * Matrix.trace ((paperSig S0 t)⁻¹ * dS * (paperSig S0 t)⁻¹ * dS) := by
  have hM : (paperSig S0 t)⁻¹ * (Real.exp (-2 * t) • dS) * (paperSig S0 t)⁻¹
        * (Real.exp (-2 * t) • dS)
      = (Real.exp (-2 * t) * Real.exp (-2 * t))
          • ((paperSig S0 t)⁻¹ * dS * (paperSig S0 t)⁻¹ * dS) := by
    simp only [Matrix.mul_smul, Matrix.smul_mul, smul_smul]
  have h : Real.exp (4 * t) * (Real.exp (-2 * t) * Real.exp (-2 * t)) = 1 := by
    rw [← Real.exp_add, ← Real.exp_add]
    have : 4 * t + (-2 * t + -2 * t) = 0 := by ring
    rw [this, Real.exp_zero]
  unfold paperFisher
  rw [hM, Matrix.trace_smul, smul_eq_mul]
  linear_combination
    (1 / 2 * Matrix.trace ((paperSig S0 t)⁻¹ * dS * (paperSig S0 t)⁻¹ * dS)) * h

/-- **Target 3 (THEORY_NOTES §2, B3; the constant `½‖∂_θΣ₀‖²_F`).**

With `Σ_t = e^{−2t} Σ₀ + (1 − e^{−2t}) I`, `∂_θΣ_t = e^{−2t} ∂_θΣ₀` and the closed-form
Gaussian Fisher information `I_t = ½ tr(Σ_t⁻¹ ∂_θΣ_t Σ_t⁻¹ ∂_θΣ_t)`,

  `e^{4t} · I_t  →  ½ tr(∂_θΣ₀ᵀ ∂_θΣ₀)  ( = ½ ‖∂_θΣ₀‖²_F )`   as `t → ∞`,

for **every** `Σ₀` (no positivity or symmetry of `Σ₀` is needed for the limit: `Σ_t → I`
entrywise and `Matrix.inv` is continuous at `I`; `Matrix.inv` is a total function, equal to
the true inverse wherever `Σ_t` is invertible, which `paperSig_isUnit_det` shows is the
case for all `t > 0` when `Σ₀ ⪰ 0`), and for symmetric `∂_θΣ₀ = dS` (used only to identify
`tr(dS dS)` with `tr(dSᵀ dS)`; this covers the correlation coordinate `ρ` and the scale
coordinate `q`). -/
theorem paper_gaussian_fisher_rate (S0 dS : Matrix (Fin n) (Fin n) ℝ) (hdS : dSᵀ = dS) :
    Filter.Tendsto (fun t => Real.exp (4 * t) * paperFisher S0 dS t) Filter.atTop
      (nhds ((1 / 2) * Matrix.trace (dSᵀ * dS))) := by
  have hg : Continuous fun M : Matrix (Fin n) (Fin n) ℝ =>
      (1 / 2) * Matrix.trace (M * dS * M * dS) :=
    continuous_const.mul
      ((((continuous_id.matrix_mul continuous_const).matrix_mul continuous_id).matrix_mul
        continuous_const).matrix_trace)
  have h := (hg.tendsto 1).comp (paperSig_inv_tendsto S0)
  have h2 : Filter.Tendsto (fun t => Real.exp (4 * t) * paperFisher S0 dS t) Filter.atTop
      (nhds ((fun M : Matrix (Fin n) (Fin n) ℝ => (1 / 2) * Matrix.trace (M * dS * M * dS)) 1)) := by
    refine h.congr fun t => ?_
    simp only [Function.comp_apply]
    exact (paper_exp_mul_fisher S0 dS t).symm
  have hlim : (fun M : Matrix (Fin n) (Fin n) ℝ => (1 / 2) * Matrix.trace (M * dS * M * dS)) 1
      = (1 / 2) * Matrix.trace (dSᵀ * dS) := by
    simp only [Matrix.one_mul, Matrix.mul_one, hdS]
  rwa [hlim] at h2

/-- Frobenius form of the limit, through the audit's `ch05b_frobSq`
(`‖A‖²_F = Σ A_{ij}² = tr(Aᵀ A)`, `ch05b_g_frob`). -/
theorem paper_gaussian_fisher_rate_frob (S0 dS : Matrix (Fin n) (Fin n) ℝ) (hdS : dSᵀ = dS) :
    Filter.Tendsto (fun t => Real.exp (4 * t) * paperFisher S0 dS t) Filter.atTop
      (nhds ((1 / 2) * ch05b_frobSq dS)) := by
  rw [ch05b_g_frob]
  exact paper_gaussian_fisher_rate S0 dS hdS

/-- Invertibility witness: if `Σ₀` is positive semidefinite then `Σ_t` is positive definite,
hence invertible, for every `t > 0` — so `(paperSig S0 t)⁻¹` in `paperFisher` is the genuine
inverse on the whole range `t > 0`. -/
theorem paperSig_isUnit_det (S0 : Matrix (Fin n) (Fin n) ℝ) (hS0 : S0.PosSemidef)
    (t : ℝ) (ht : 0 < t) : IsUnit (paperSig S0 t).det := by
  have hpos : 0 < 1 - Real.exp (-2 * t) := by
    have : Real.exp (-2 * t) < 1 := by
      rw [Real.exp_lt_one_iff]; linarith
    linarith
  have hA : (Real.exp (-2 * t) • S0).PosSemidef := hS0.smul (Real.exp_pos _).le
  have hB : ((1 - Real.exp (-2 * t)) • (1 : Matrix (Fin n) (Fin n) ℝ)).PosDef :=
    Matrix.PosDef.one.smul hpos
  have hSig : (paperSig S0 t).PosDef := Matrix.PosDef.posSemidef_add hA hB
  exact (Matrix.isUnit_iff_isUnit_det _).mp hSig.isUnit

/-! ## Target 4 — scalar score-gap leading coefficient `κ₄/6` (THEORY_NOTES §2, B1) -/

/-- `Λ₄ − Λ₄^G` in the scalar case.  The fourth-order joint cumulants of the linear form
`s·a` and the quadratic form `a²/2` are `cum(sa,sa,sa,sa) = s⁴κ₄`, `cum(sa,sa,a,a) = s²κ₄`,
`cum(a,a,a,a) = κ₄`, and the Gaussian twin has the same `Λ₂` and `κ₄ = 0`, so (THEORY_NOTES
§2, display for `Λ₄`)

  `Λ₄ − Λ₄^G = κ₄ · ( s⁴/24 − β s²/4 + β²/8 )`. -/
def paperLam4diff (s β k4 : ℝ) : ℝ :=
  k4 * ((1 / 24) * s ^ 4 - (1 / 4) * β * s ^ 2 + (1 / 8) * β ^ 2)

/-- `∂_s (Λ₄ − Λ₄^G)` at fixed `β`: the derivative of `paperLam4diff` in `s` is
`κ₄ (s³/6 − β s/2)`. -/
theorem paper_Lam4diff_hasDerivAt (β k4 s : ℝ) :
    HasDerivAt (fun s => paperLam4diff s β k4) (k4 * ((1 / 6) * s ^ 3 - (1 / 2) * β * s)) s := by
  have h4 : HasDerivAt (fun s : ℝ => s ^ 4) (4 * s ^ 3) s := by
    simpa using hasDerivAt_pow 4 s
  have h2 : HasDerivAt (fun s : ℝ => s ^ 2) (2 * s) s := by
    simpa using hasDerivAt_pow 2 s
  have h := (((h4.const_mul (1 / 24)).sub (h2.const_mul ((1 / 4) * β))).add_const
    ((1 / 8) * β ^ 2)).const_mul k4
  exact h.congr_deriv (by ring)

/-- **Target 4 (THEORY_NOTES §2, B1; scalar unit-variance leading coefficient `κ₄/6`).**

Substituting `s = αx` and `β = α²` (the values `s = αx/Δ`, `β = α²/Δ` at `Δ = 1`, i.e. the
leading order in `α = e^{−t}`) into `∂_s(Λ₄ − Λ₄^G) = κ₄ (s³/6 − β s/2)` gives

  `κ₄ ( (αx)³/6 − α²(αx)/2 ) = (κ₄/6) · α³ · (x³ − 3x) = (κ₄/6) · α³ · He₃(x)`.

Together with `S_t − S_t^G = (α/Δ) ∂_s(Λ − Λ^G)` (THEORY_NOTES §2, exact) this is the statement
that the coefficient of `He₃(x)` in the order-`α⁴ = e^{−4t}` term of `S_t − S_t^G` is `κ₄/6`,
which is what fixes `C₄ = κ₄²/6` in B1.

*What this checks:* the constants `1/24, 1/4, 1/8` of `Λ₄`, the derivative `1/6, 1/2`, and
the resulting `κ₄/6` in front of `He₃` — the algebraic slip the numerics of §4 discriminate
only weakly.  *What it does not check:* the remainder (`O(α⁵)` inside `∂_s(Λ − Λ^G)` and the
`1/Δ` corrections), which is analysis and is excluded by THEORY_NOTES §6. -/
theorem paper_scalar_score_gap_coefficient (k4 α x : ℝ) :
    k4 * ((1 / 6) * (α * x) ^ 3 - (1 / 2) * α ^ 2 * (α * x))
      = (k4 / 6) * α ^ 3 * (x ^ 3 - 3 * x) := by
  ring

/-- The same statement with `He₃` displayed through Mathlib's Hermite polynomial. -/
theorem paper_scalar_score_gap_coefficient_he3 (k4 α x : ℝ) :
    k4 * ((1 / 6) * (α * x) ^ 3 - (1 / 2) * α ^ 2 * (α * x))
      = (k4 / 6) * α ^ 3 * (paperHe 3).eval x := by
  rw [paperHe_eval_three]
  ring

/-- Chain of the two facts: the derivative of `Λ₄ − Λ₄^G` in `s`, evaluated at
`s = αx`, `β = α²`, is `(κ₄/6) α³ He₃(x)`. -/
theorem paper_scalar_score_gap_deriv (k4 α x : ℝ) :
    HasDerivAt (fun s => paperLam4diff s (α ^ 2) k4) ((k4 / 6) * α ^ 3 * (paperHe 3).eval x)
      (α * x) := by
  rw [← paper_scalar_score_gap_coefficient_he3]
  exact paper_Lam4diff_hasDerivAt (α ^ 2) k4 (α * x)

end

end ThesisAudit
