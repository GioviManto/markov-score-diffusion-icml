# Lean 4 audit — what the manuscript claims, and what is machine-checked

This directory holds a machine-checked audit of the display mathematics of the thesis
*Markov Score Diffusion*, carried out in Lean 4 (`leanprover/lean4:v4.33.0`) against `mathlib`.
Thirteen modules under `ThesisAudit/` (twelve audit modules plus `PaperTheory.lean`, §2), per-formula verdicts in `AUDIT_RESULTS/*.json`, findings in
[`AUDIT_REPORT.md`](AUDIT_REPORT.md) and [`PROSE_FINDINGS.md`](PROSE_FINDINGS.md).

The audit predates the current manuscript. Its purpose here is no longer the thesis but the paper:
**this file maps each statement of `paper/legacy-neurips/` onto the Lean theorem that covers it,
records the exact hypotheses that theorem carries, and names what is still informal.** Where a
theorem is weaker than the manuscript statement, the row says so. The verdict table and the build
instructions are retained below as §3.

---

## 1. Manuscript → Lean

Line numbers are of the `theorem`/`def` keyword in the named module. "Status" is the status of
*this row*, not of the surrounding cluster.

### Proposition 1 — Gaussian closure computes the LMMSE of the covariance-matched twin

`sections/prop-lmmse.tex` (statement), `sections/prop-lmmse-proof.tex` (proof),
appendix §"The Gaussian model and the second-order baseline".

| manuscript statement | Lean theorem (module:line) | hypotheses in the Lean statement | what remains informal | status |
|---|---|---|---|---|
| eq. (8) `â = Cov(a,x)Cov(x)⁻¹x = e^{-t}Σ₀(e^{-2t}Σ₀+Δ I)⁻¹x`, as a rewriting | `ch07_l_lmmse` (`ThesisAudit/Ch07.lean:776`) | `Cax = e^{-t}•S0`, `Cxx = e^{-2t}•S0 + Δ_t•1`; arbitrary `n` | Nothing mathematical: the proof is `rw [hCax, hCxx]`, i.e. substitution of its own hypotheses. The content is in the four rows below. | PROVED (vacuous on its own) |
| …that this gain is the *minimising* one | `ch07_l_lmmse_normal_eq` (`Ch07.lean:788`), `ch07_l_lmmse_excess` (`Ch07.lean:1336`), `ch07_l_lmmse_optimal` (`Ch07.lean:1354`) | normal-eq: `IsUnit Cxx.det`. excess: `Cxxᵀ = Cxx`, `Gstar * Cxx = Cax`; arbitrary `n` (state) and `m` (observation). optimal: `Cxx.PosSemidef`, `Gstar * Cxx = Cax` | Risk is the quadratic form `ch07_lmmse_risk` written in second moments (`tr Σₐ − tr(G Caxᵀ) − tr(Cax Gᵀ) + tr(G Cxx Gᵀ)`); the step from `E‖a−Gx‖²` to that expression is bilinearity of expectation, asserted by the definition rather than derived. | PROVED |
| `Cov(a,x)=e^{-t}Σ₀`, `Cov(x)=e^{-2t}Σ₀+Δ_t I` from the channel | `ch07_lmmse_covs_channel` (`Ch07.lean:1677`) | `0 ≤ t`; finite product sample space `X × Z` with weights `pa`, `pz` summing to 1; `z` centred (`∀ j, ∑ pz·z = 0`) with `ch07_cov pz z z = 1`; arbitrary `n` | The sample space is **finite and weighted** (sums, not integrals). No continuous-measure version exists in the file. Non-vacuity is witnessed by `ch07_ou_noise_witness`. | PROVED |
| eq. (7) forward moment map `(m,v) ↦ (ρm, ρ²v+σ²)` | `ch07_lmmse_fwd_map` (`Ch07.lean:831`) | `0 < q`, `0 < v`; all real `α, m, a` | Proved as a **Gaussian × Gaussian convolution integral** over ℝ. The proposition's clause *"returns the same estimator for every innovation law sharing the moments (0,σ²)"* is **not** formalised: no theorem in `ThesisAudit` quantifies over innovation laws, and no lemma states that the mean and variance of a convolution depend on the innovation only through its first two moments. | PROVED for Gaussian innovation; the moments-only clause is INFORMAL |
| eq. (7) backward moment map `(m,v) ↦ (m/ρ, (v+σ²)/ρ²)` | `ch07_lmmse_bwd_integral` (`Ch07.lean:841`), `ch07_lmmse_bwd_map` (`Ch07.lean:851`) | integral: `0 < q`, `0 < v`. map: `α ≠ 0`, `v + q ≠ 0` | Same Gaussian-innovation restriction. `α ≠ 0` is exactly the manuscript's side condition and is used exactly where the manuscript says it is (step (ii), re-reading the exponent as a density in the preceding state). | PROVED for Gaussian innovation |
| the induction: every message stays Gaussian along the chain | `ch06_bp_forward_closure_iterated` (`Ch06.lean:1237`), `ch06_bp_backward_closure_iterated` (`Ch06.lean:1281`), `ch06_bp_belief_gaussian` (`Ch06.lean:1321`) | `0 < σ²`, `0 < Δ_t`; every site index, every chain length; genuine Lebesgue integrals over ℝ | This is the **Gaussian chain's own** recursion, proved by induction on the site. It is *not* the projection of a non-Gaussian chain's messages. Backward precision is `≥ 0` (it is `= 0` at the right endpoint, where the message is the flat `μ_{←L−1}=1`) — the manuscript's remark that `r_n ≡ 1` is outside the domain of `P_G` is handled in Lean by carrying an unnormalised natural-form message, not by starting the induction at `ℓ_n`. | PROVED (Gaussian chain) |
| **the projection operator `P_G`, and the induction "every projected message equals the Gaussian chain's message"** | — | — | **Not formalised at all.** `P_G` is not defined anywhere in `ThesisAudit` (a grep for `Proj`/`momentMatch` across `Ch06.lean` and `Ch07.lean` returns only an unrelated `EuclideanSpace.proj` inside a proof). This is the load-bearing step of `prop-lmmse-proof.tex`, and it is prose. | INFORMAL |
| Remark on hypotheses: "the grid plays no part in it" | — | — | Correct as a statement about the Lean development too: none of the rows above discretises. The comparison target (unrestricted non-Gaussian messages represented numerically) is not formalised either. | INFORMAL |

**Reading of Proposition 1.** The two ends are machine-checked — the moment maps are real integrals,
and the optimality of the stated gain holds in arbitrary dimension for arbitrary second moments. The
*middle*, which is what the proposition actually asserts (projection ≡ covariance-matching, uniformly
over innovation laws), is not.

### Proposition 2 — Fisher identity, exact gradient of the discretised marginal likelihood

`sections/prop-fisher.tex`, `sections/prop-fisher-scope.tex`.

| manuscript statement | Lean theorem (module:line) | hypotheses in the Lean statement | what remains informal | status |
|---|---|---|---|---|
| `∇L = (1/p_θ)∇p_θ`, interchange of ∇ with the latent sum | `ch08a_em_fisher_step1` (`Ch08a.lean:598`) | finite latent `Z`; one real parameter; `HasDerivAt` of each `p_θ(z,x)` | The interchange is *free* because the sum is finite. This matches the proposition, which is explicitly about the finite-state model on the grid. | PROVED |
| `∇L = E_{a,c|x,θ}[∇ log p_θ(a,c,x)]` | `ch08a_em_fisher_general` (`Ch08a.lean:617`) | finite `Z`; one real parameter; positivity and differentiability of each `p_θ(z,x)` | one-dimensional parameter | PROVED |
| eq. (10), transition form: only transition factors depend on θ | `ch08a_em_fisher_transitions` (`Ch08a.lean:1269`) | `Fintype Z`, `Nonempty Z`, `u : Finset ι`; `∀ t z, 0 < P t z`; `∀ i ∈ u, ∀ z t, 0 < T i z t`; **`hlog`: `log (P t z) = K z + ∑_{i∈u} log (T i z t)` with `K` independent of `t`** (this is the complete-data log-likelihood split, supplied as a hypothesis, not smuggled into the shape of the integrand); `HasDerivAt (T i z) (T' i z) θ`. Conclusion mentions both `ch08a_L` and `ch08a_post`, so the LHS really is the marginal log-likelihood and the RHS a genuine posterior expectation | the parameter is a single real coordinate | PROVED |
| the same, at the manuscript's own chain | `ch08a_em_fisher_chain` (`Ch08a.lean:1308`) | the chapter's joint `ch08a_joint`, parameter `α`, index set `Finset.Icc 2 N`, transition factor `ch08a_ktilde`; positivity of `p`, `s`, `p1`, `φ`; **`hd`: `D i z = ∂_α K̃` is a hypothesis, not computed in closed form** | The manuscript's `θ` is the vector `(ρ, π, ν, s)`; only the single coordinate `α = ρ` is formalised. `D i z` is symbolic, exactly as `∇_θ log K̃` is symbolic in the display — so this matches the claim, but it is not an independent check of that derivative. | PROVED (one coordinate) |
| eq. (9) `Ξ_{kl} = ∑_μ ∑_i P(a_i=g_k, a_{i−1}=g_l ∣ x^μ)` and the pairing `⟨Ξ, ∇log K̃⟩` | `ch08a_estep_pair_expectation` (`Ch08a.lean:1158`), and the `prop:em-estep` cluster (`ch08a_estep_exact`, `_exact_pair`, `_pairwise`, `_cut_consistency`, `_exact_mixture`) | finite alphabet `A` with `DecidableEq`; arbitrary chain length, arbitrary site `i`, arbitrary `h : A → A → ℝ`; posterior expectation of any function of an adjacent pair is read off `fwd × ψ × φ × bwd` at that one transition | **There is no `Ξ` object in the Lean development.** The regrouping of `E[∑_i score_i]` into a sum over grid pairs `(k,l)` weighted by `Ξ_{kl}`, and the outer sum over sequences `μ`, are not stated. `ch08a_estep_pair_expectation` is exactly the ingredient that would make it a bookkeeping step, but the step is not taken. | PROVED (the ingredient); the `⟨Ξ, ·⟩` form is INFORMAL |
| `∇_θ L(θ)|_{θ^{(k)}} = ∇_θ Q(θ∣θ^{(k)})|_{θ^{(k)}}` — "one pass supplies the exact gradient" | `ch08a_em_fisher_Q` (`Ch08a.lean:641`) | finite `Z`, one real parameter | Both `L` and `Q` are exhibited with the same derivative value. The *cost* claim ("same asymptotic order as evaluating the likelihood", "accumulating Ξ dominates the constant at large N_g") is not formalised. | PROVED |
| Remark `rem:fisher-scope`: the continuum identity, and the Laplace kink | — | — | **Not formalised.** The continuum statement needs differentiability of `K_θ(a'∣a)` in θ, θ-independent support, and a dominating function; none of these appear. The Laplace argument (kink set Lebesgue-null, derivative a.e., locally uniform domination) is prose. The nearest formal object is `ch07_l_score_identity_prior` (`Ch07.lean:1032`), which does differentiate under the integral sign for an arbitrary finite measure `ν` on ℝ with `Integrable (fun a => a) ν` — but for the *score*, not for `∂_θ K`. | INFORMAL |

**Reading of Proposition 2.** This is the better-covered of the two. The proposition is deliberately a
claim about the finite-state model on the grid, and the Lean model is finite-state, so the usual
"only the discrete case" caveat is not a gap here — it is the claim. The two real gaps are the
vector parameter and the `Ξ` pairing.

### Displayed equations and appendix identities

| manuscript statement | Lean theorem (module:line) | hypotheses in the Lean statement | what remains informal | status |
|---|---|---|---|---|
| **eq. (6)** `E[a∣x] = e^{-t}Σ₀Σ_t⁻¹x`, `s(x,t) = −Σ_t⁻¹x` | `ch05b_g_score_matrix` (`Ch05b.lean:114`) | none — it is `rfl` on the module's own definition `ch05b_score := −(Σ_t⁻¹ *ᵥ x)` | Definitional on its own. | DEFINITION |
| …that this really is `∇ log P_t` | `ch05b_log_gaussPDF` (`Ch05b.lean:2594`), `ch05b_g_score_matrix_grad_log` (`Ch05b.lean:2627`) | `S0.IsSymm`; all `x`, all increments `v`. Conclusion: `P_t(x+v) = P_t(x)·exp(⟨s(x,t),v⟩ − ½ vᵀQ_t v)` — the linear part of the log-density increment is exactly `⟨−Q_t x, v⟩` and the whole remainder is the displayed quadratic | `P_t` is the explicit density `ch05b_gaussPDF` (`Ch05b.lean:2400`), **not** a `Measure.map` of the OU channel. That `ch05b_gaussPDF Σ_t` is the law of `x_t` is asserted by construction, not proved. | PROVED (as a density identity) |
| the Tweedie route to the same object, componentwise | `ch05b_g_tweedie` (`Ch05b.lean:434`) | `IsUnit S0.det`, `IsUnit (Σ_t).det`, `Δ_t ≠ 0`; arbitrary invertible `Σ₀`, every coordinate `k` | linear-algebra identity between `−Q_t x` and the posterior-mean expression | PROVED |
| `E[a∣x] = J⁻¹h` | `ch05b_g_posterior_mean` (`Ch05b.lean:288`) | `J` symmetric, `IsUnit J.det`, `J` positive definite; conclusion is *strict* minimality of `J⁻¹h` for the posterior exponent against every other `a` | `E[a∣x]` is characterised as the minimiser of the exponent, not as a conditional expectation of a measure. | PROVED |
| `(Σ_t)_{ij} = e^{-2t}ρ^{|i−j|} + Δ_t δ_{ij}` | `ch05a_g_Sigt_entrywise` (`Ch05a.lean:822`) | all `n` | — | PROVED |
| `Σ₀⁻¹` tridiagonal at every `t` (appendix §"The Gaussian model…") | `ch05b_Sig0_inv_eq_Q0` (`Ch05b.lean:1938`), `ch05b_noname_Q0_banded`, `ch05a_g_Q0_general` | `1 − ρ² ≠ 0`, `2 ≤ L` | The numerical claims attached to eq. (6) in the body (`9.2×10⁻¹⁵` relative, `1.6×10⁻¹⁴` against brute-force enumeration) are **not** formal. | PROVED |
| **eq. (11)** belief-weighted Yule–Walker, `ρ = S₀₁/S₀₀` | `ch08b_mstep_alpha` (`Ch08b.lean:569`, def), `ch08b_em_mstep_alpha` (`Ch08b.lean:572`), `ch08b_alpha_single_component` (`Ch08b.lean:589`) | general form: `∑_c s_c⁻²W_c ≠ 0`, `C` components. single-component: `ν₀ = 0`, `s²₀ ≠ 0`, `W₀ ≠ 0`, giving `α⁺ = P/W` | The Lean statistics `P, W, G, R, U, V` (`Ch08b.lean:206–222`) are sums over an abstract finite index `Ω` with belief weights `b` and responsibilities `r`; `ch08b_single_component_moments` (`Ch08b.lean:598`) identifies them at `C=1` with `∑ b·x·y` and `∑ b·x²`. **The identification of `Ω` with grid pairs `(k,l)` and of `b` with `Ξ` is not made** — so `S_{ab} = ∑_{kl} Ξ_{kl} g_k^a g_l^b` is matched in shape, not in referent. | PROVED (shape); the `Ξ`-on-the-grid reading is INFORMAL |
| eq. (11), `σ² = (S₁₁ − ρS₀₁)/W` | `ch08b_mstep_var` (`Ch08b.lean:497`, def), `ch08b_em_mstep_var` (`Ch08b.lean:500`) | `R ≠ 0` (the manuscript flags exactly this degeneracy) | The Lean form is the mixture one, `(G − 2αP + α²W − R(ν⁺)²)/R`. At `C = 1`, `ν⁺ = 0`, `α = P/W` it collapses to `(S₁₁ − ρS₀₁)/W` by arithmetic, but **that specialisation is not separately stated as a theorem.** | PROVED (mixture form); the paper's specialisation is INFORMAL |
| "exact EM, not a generalised variant" at this rung | `ch08b_em_mstep_alpha_max` (`Ch08b.lean:582`), `ch08b_em_alpha_block_ascent` (`Ch08b.lean:840`), `ch08b_em_mstep_var_max` (`Ch08b.lean:461`) | `B = ∑_c W_c/s_c² > 0` for the α-block; `0 < R`, `0 < S`, `0 < τ` for the variance block | Global *conditional* (block) maximality is proved; the excess is exactly `−(B/2)(α−α⁺)²`. The **joint** `(ρ,σ²)` maximisation at `C=1` in one step — which is what "exact EM" means for the manuscript's Rung 1 — is not stated as a single theorem. | PROVED (block-wise) |
| **eq. (13)** risk decomposition `E[‖a−u‖²∣x] = E[‖a−m‖²∣x] + ‖u−m‖²` | `ch07_l_excess_derived` (`Ch07.lean:932`) | finite `X` (observations), finite `Y` (latents), joint mass `p : X → Y → ℝ`; **`hm : ∀ x i, ∑_y p x y (a_y,i − m_x,i) = 0`** — exactly `E[a − m(x) ∣ x] = 0` componentwise, the tower property the manuscript proof invokes; `ah` any function of `x` alone. The cross term is *derived*, not assumed | Finite sample space: sums, not integrals. No `condExp` / regular conditional probability. Weaker forms with the cross term assumed are `ch07_l_excess` (`Ch07.lean:891`) and `ch07_l_excess_expect` (`Ch07.lean:905`). The prose reading ("irreducible floor", "a different loss selects a different functional") is not formalised. | PROVED (finite) |
| **Appendix B** Tweedie / score–posterior identity | `ch02_tweedie` (`Ch02.lean:674`) | `0 < N`, `Δ > 0`, `λ_i > 0`; prior is an **arbitrary finitely supported positive measure** on ℝ (`N` atoms at arbitrary locations, arbitrary positive weights, not necessarily normalised); one dimension. Both directions of the equivalence | Finite support. The appendix's own argument — differentiation under the integral with a dominating function, for a `P₀` with a finite first moment — is formalised *separately and only in the scalar case* as `ch07_l_score_identity_prior` (`Ch07.lean:1032`: arbitrary finite measure `ν` on ℝ with `Integrable (fun a => a) ν`, via `hasDerivAt_integral_of_dominated_loc_of_deriv_le`). | PROVED (finitely supported prior) |
| Appendix B, the `L`-frame joint version | `ch02_tweedie_joint_full` (`Ch02.lean:1138`) | arbitrary `L`, arbitrary `N` atoms, arbitrary distinguished coordinate `k`; prior an arbitrary finitely supported positive measure on ℝ^L (so the frames are coupled); the channel acts independently per frame; derivative taken along `Function.update x k` | Finite support again: the ℝ^L version for a general (non-atomic) `P₀` is not formalised. | PROVED (finitely supported prior) |
| **Appendix E** `Var(a_i)=1`, `Cov(a_i,a_j)=ρ^{|i−j|}` *whatever the shape of ε* | `ch03_ar1_toeplitz_of_iIndepFun` (`Ch03.lean:383`); via the structure `ch03_AR1`, `ch03_ar1_toeplitz_proc` (`Ch03.lean:288`) | `MemLp (a k) 2 μ`, `MemLp (η k) 2 μ`; **pathwise** recursion `a_{k+1} ω = α a_k ω + η_k ω`; measurability of the driving vector; `iIndepFun` of `(a₀, η₀, η₁, …)`; `Var[a₀] = 1`; `Var[η_k] = 1 − α²`. Conclusion `cov[a_j, a_k; μ] = α^{|j−k|}` for **all** `j,k` | **No Gaussianity anywhere** — this is exactly the appendix's "whatever the shape of `ε`", and it is proved, not assumed. Independence is derived, not hypothesised (`ch03_ar1_state_eq_lin` + `ch03_ar1_indep_of_iIndepFun`). Non-vacuity: `ch03_AR1_exists` (`Ch03.lean:509`) for every `α` with `α² ≤ 1`. | PROVED |
| Appendix E, the variance recursion `v_i = ρ²v_{i−1} + σ² = 1` | `ch05a_g_stationary_marginal` (`Ch05a.lean:185`) | none beyond the recursion | Encoded as a **scalar numeric recursion** `v₀=1, v_{k+1}=α²v_k+σ²`, proved `v_k = 1` by induction. It is not a statement about a process; the process-level version is the row above. | PROVED (recursion form) |
| Appendix E: strict vs covariance stationarity | `ch03_StrictlyStationary` (`Ch03.lean:28`), `ch03_WeaklyStationary` (`Ch03.lean:44`) | — | **Definitions only.** The two claims the appendix actually makes are *not* formalised: (i) "a Gaussian law is determined by two moments, so weak ⟹ strict"; (ii) "it is **not** strictly stationary for non-Gaussian `ε`", with the invariant law `∑_{k≥0} ρ^k ε_{−k}` and the burn-in `1/log(1/ρ) ≈ 6` sites. The only proved fact here is the sanity lemma `ch03_weak_stationary_even` (`γ(−d) = γ(d)`). | INFORMAL |
| eq. (7) as a *closure* statement (`P_G` order of operations, "a different order is a different algorithm") | — | — | Not formalised; see the Proposition 1 block. | INFORMAL |

---

## 2. Targets for the new theory — status

**Module `ThesisAudit/PaperTheory.lean` (15 September 2026): builds green under `lake build
ThesisAudit.PaperTheory` on `leanprover/lean4:v4.33.0` + Mathlib `v4.33.0`; no `sorry`, no
`axiom`, no `native_decide`; 31 declarations.** Every statement is algebraic: hypotheses are
numbers (moments, coefficients, matrix entries) and ℝ-linear functionals on `ℝ[X]`, never
measures. The measure-level instantiation (that `N(0,1)` has moments `0,1,0,3`; that the Fisher
information of a Gaussian family is the trace formula; that the score of the actual law has the
claimed expansion) and every `O(·)` remainder of `docs/THEORY_NOTES.md` §2 remain **informal**,
by design (§6 there). Line numbers are of the `theorem` keyword.

| target | Lean theorem (`PaperTheory.lean:line`) | hypotheses in the Lean statement | what it checks | what stays informal |
|---|---|---|---|---|
| 1. Mehler identity, `n ≤ 4` (Theorem A, scalar) | `paper_mehler_he_n` (:128); explicit forms `paperHe_zero`…`paperHe_four` (:39–49) proved from `Polynomial.hermite_succ` | `E : ℝ[X] →ₗ[ℝ] ℝ` with `E 1 = 1`, `E X = 0`, `E X² = 1`, `E X³ = 0`, `E X⁴ = 3`; `β² = 1 − α²`; `n ≤ 4` | `E[He_n(αa + βz)] = αⁿ He_n(a)` for `n = 0..4`, i.e. the attenuation `e^{−nt}` of degree-`n` modes with `α = e^{−t}` | that a Gaussian `z` satisfies the four moment hypotheses (Mathlib has no `E z⁴ = 3`); general `n` (needs the generating function, absent from Mathlib); the chain (tensor) form |
| 2. Degree-4 moments differ by `κ₄` | `paper_he4_moment_difference` (:171), `paper_he4_moment_difference_functional` (:181), via `paper_he4_expectation` (:154) | two laws as moment pairs `(m₂,m₄)`, `(m₂',m₄')` with `m₂ = m₂'` (functional form: `E 1 = E' 1 = 1`, `E X² = E' X²`); `κ₄ := m₄ − 3m₂²` defined locally (`paperKappa4`, :149) | `E He₄ − E' He₄ = κ₄ − κ₄'` under variance matching | the tensor version over sites (same identity index by index; not written) |
| 3. Gaussian Fisher-information rate (B3) | `paper_gaussian_fisher_rate` (:268), `paper_gaussian_fisher_rate_frob` (:289); `paper_exp_mul_fisher` (:238) exact factorisation; `paperSig_inv_tendsto` (:226); `paperSig_isUnit_det` (:298) | `Σ_t := e^{−2t}Σ₀ + (1−e^{−2t})I` (`paperSig`, :194), `I_t := ½ tr(Σ_t⁻¹ (e^{−2t}dS) Σ_t⁻¹ (e^{−2t}dS))` (`paperFisher`, :201) with Mathlib's total `Matrix.inv`; `dSᵀ = dS`; any `n`, any real `Σ₀` for the limit; `Σ₀ ⪰ 0`, `t > 0` for invertibility | `e^{4t} I_t → ½ tr(dSᵀ dS) = ½ ‖dS‖²_F` (`ch05b_frobSq`) as `t → ∞`; `e^{4t} I_t = ½ tr(Σ_t⁻¹ dS Σ_t⁻¹ dS)` exactly; `Σ_t` invertible on `t > 0` when `Σ₀ ⪰ 0` | that `I_t` *is* the Fisher information of the Gaussian family (the trace formula is taken as the definition); the rate of approach (`t_*`) is not bounded |
| 4. Scalar score-gap coefficient `κ₄/6` (B1) | `paper_scalar_score_gap_coefficient` (:348), `_he3` (:354), `paper_scalar_score_gap_deriv` (:362), via `paper_Lam4diff_hasDerivAt` (:323) | `Λ₄ − Λ₄^G` encoded as the explicit polynomial `paperLam4diff β k4 s` (:318) in the graded variables `s, β`; unit variance | `∂_s(Λ₄ − Λ₄^G)` at `s = αx`, `β = α²` equals `(κ₄/6) α³ He₃(x)` | that this polynomial is the fourth-order truncation of the damped CGF of the actual law (the graded expansion and its remainder, THEORY_NOTES §2) |
| 5. Proposition 1 projection step | not started | — | — | the projection operator `P_G` and the moments-only clause (see §1 and the plan below) |

**Reading.** Targets 1–4 check the four constants a reader is most likely to doubt (`e^{−nt}` per
degree, `κ₄` as the degree-4 difference, the `½` in the Fisher rate, the `1/6` in the score gap)
and nothing about convergence of the expansions. The paper's Theorem 2 is therefore *not*
machine-verified; its algebraic skeleton is.

### The original plan (retained)

These are the five targets of `docs/THEORY_NOTES.md` §6. The rule stated there holds: **do not
formalise the remainder bounds of B1/B2.** They are analysis, not algebra, and the value of Lean
here is catching an algebraic slip in a constant, not in an `O(·)` term. Every target below is
therefore stated so that its hypotheses are *numbers* (moments, coefficients) rather than measures,
with measure-level instantiation pushed into a separate, clearly-labelled witness lemma.

### Mathlib inventory, checked against the pinned `v4.33.0`

Verified in `/Users/gloriabagnato/Code/Thesis/Diffusion/formal/.lake/packages/mathlib`
(`lean/.lake` is a symlink to that tree).

**Hermite polynomials** — `Mathlib/RingTheory/Polynomial/Hermite/Basic.lean`, probabilists'
convention, over `ℤ`:

- `Polynomial.hermite : ℕ → Polynomial ℤ` (line 48)
- `Polynomial.hermite_succ : hermite (n+1) = X * hermite n - derivative (hermite n)` (54) — gives
  `He₀=1, He₁=X, He₂=X²−1, He₃=X³−3X, He₄=X⁴−6X²+3` by `simp`
- `hermite_zero` (63), `hermite_one` (66), `coeff_hermite_succ_zero` (75), `coeff_hermite_succ_succ`
  (78), `coeff_hermite_of_lt` (83), `coeff_hermite_self` (93), `degree_hermite` (101),
  `natDegree_hermite` (109), `leadingCoeff_hermite` (113), `hermite_monic` (116),
  `coeff_hermite_of_odd_add` (119), `coeff_hermite_explicit` (142), `coeff_hermite_of_even_add`
  (184), `coeff_hermite` (193)
- `Mathlib/RingTheory/Polynomial/Hermite/Gaussian.lean`:
  `Polynomial.deriv_gaussian_eq_hermite_mul_gaussian`, `hermite_eq_deriv_gaussian`,
  `hermite_eq_deriv_gaussian'` — the Rodrigues formula

**Gaps.** There is **no** generating-function identity (`exp(sx − s²/2) = ∑ He_n(x)sⁿ/n!`), **no**
Gaussian orthogonality relation, and **no** Mehler formula anywhere in Mathlib — a grep for `mehler`
over the whole library returns nothing, and only three files mention `hermite` at all (the two above
plus `NumberTheory/NumberField/Discriminant/Basic.lean`, which is Hermite's discriminant theorem and
unrelated). There is also no lemma integrating a Hermite polynomial against a Gaussian measure.

**Real Gaussian** — `Mathlib/Probability/Distributions/Gaussian/Real.lean`:

- `ProbabilityTheory.gaussianReal` (222), `gaussianPDFReal` (49)
- `integral_id_gaussianReal : ∫ x, x ∂gaussianReal μ v = μ` (540)
- `variance_id_gaussianReal` / `variance_fun_id_gaussianReal : Var[id; gaussianReal μ v] = v` (549, 550)
- `mgf_id_gaussianReal : mgf id (gaussianReal μ v) = fun t ↦ exp(μt + v t²/2)` (508), `mgf_gaussianReal`
  (495), `cgf_gaussianReal` (513), `charFun_gaussianReal` (486), `complexMGF_id_gaussianReal` (444)
- `integrableExpSet_id_gaussianReal = Set.univ` (523), `integrable_exp_mul_gaussianReal` (517)
- `memLp_id_gaussianReal` (all moments finite)
- map lemmas: `gaussianReal_map_const_mul`, `_add_const`, `_sub_const`, `_neg`, …

**Gap.** There is **no** `centralMoment` or explicit third/fourth-moment lemma for `gaussianReal`.
`E[z³]=0` and `E[z⁴]=3` are not in the library. They must be derived from `mgf_id_gaussianReal`
together with `ProbabilityTheory.iteratedDeriv_mgf_zero`
(`Mathlib/Probability/Moments/MGFAnalytic.lean:92`, `iteratedDeriv n (mgf X μ) 0 = μ[X^n]`), whose
hypothesis `0 ∈ interior (integrableExpSet X μ)` is discharged for free by
`integrableExpSet_id_gaussianReal`.

**Moments and cumulants** — `Mathlib/Probability/Moments/Basic.lean`:
`ProbabilityTheory.moment` (55), `centralMoment` (62), `centralMoment_two_eq_variance`,
`centralMoment_one`, `moment_one`, `mgf`, `cgf`. **There is no `cumulant` definition and no
kurtosis anywhere in `Mathlib/Probability/`** — `κ₄` must be defined locally.

**Trace and Frobenius norm**:

- `Matrix.trace` algebra, `Mathlib/LinearAlgebra/Matrix/Trace.lean`: `trace_add` (64), `trace_smul`
  (68), `trace_transpose` (73), `trace_sum` (107), `trace_sub` (132), `trace_one` (146),
  `trace_transpose_mul` (154), `trace_mul_comm` (158), `trace_mul_cycle` (161), `trace_diagonal` (48)
- `Mathlib/Analysis/Matrix/Normed.lean`: `Matrix.frobenius_norm_def` (550),
  `frobenius_nnnorm_def` (545), `frobenius_norm_transpose` (568), `frobenius_norm_conjTranspose`
  (578), `frobenius_norm_mul` (646)

**Gaps, both real.** (i) The Frobenius norm is a **local/scoped instance**
(`Matrix.frobeniusSeminormedAddCommGroup`, `@[instance_reducible, local instance]`, line 500) — it is
not the default norm on `Matrix` and must be opened deliberately. (ii) There is **no** Mathlib lemma
equating `‖A‖²_F` with `trace (Aᵀ * A)`; the nearest is
`Matrix.trace_conjTranspose_mul_self_eq_zero_iff` (`LinearAlgebra/Matrix/PosDef.lean:375`), which is
only the vanishing criterion. **The audit already supplies what is missing** and should be reused:
`ch05b_frobSq` (`Ch05b.lean:1072`) and `ch05b_g_frob : ch05b_frobSq A = Matrix.trace (Aᵀ * A)`
(`Ch05b.lean:1074`), plus `ch05b_g_frob_norm` (1082), `ch05b_g_frob_split` (1105) and
`ch05b_frobSq_conj` (2366).

**Fisher information.** No `fisherInformation` exists in Mathlib (grep returns nothing). `I_t` must
be defined locally.

**Asymptotics precedents inside the audit**: `ch02_exp_neg_two_tendsto` (`Ch02.lean:1510`),
`ch02_Delta_tendsto_one` (`Ch02.lean:1525`), `ch08b_score_amplification_asymptotic`
(`Ch08b.lean:697`), and the entrywise resolvent bound at `Ch05b.lean:2999`.

---

### Target 1 — `mehler_he_n`, `n ≤ 4`

*Statement.* For real `a`, `α`, and a random variable `z` whose first four moments are those of
`N(0,1)` (`0, 1, 0, 3`), `E[He_n(αa + √(1−α²) z)] = αⁿ He_n(a)` for `n ≤ 4`.

*Ingredients.* `Polynomial.hermite` + `hermite_succ` (to unfold `He_n` for the four concrete `n`),
`Polynomial.aeval` to evaluate in ℝ. For the non-vacuity witness: `integral_id_gaussianReal`,
`variance_id_gaussianReal`, `mgf_id_gaussianReal`, `iteratedDeriv_mgf_zero`,
`integrableExpSet_id_gaussianReal`.

*Plan.* State it as a pure moment identity, not a measure-theoretic one, exactly as §6 asks. Fix
`n ≤ 4`, take the four moment values as real hypotheses `m₁ = 0, m₂ = 1, m₃ = 0, m₄ = 3` and a
linear `E`. Expand `He_n(αa + βz)` with `β² = 1 − α²` by `norm_num [hermite_succ]` into a polynomial
in `a` and `z` of degree ≤ 4, push `E` through by linearity, substitute the four moments, and close
with `ring` after rewriting `β² = 1 − α²`. Four cases, no induction, no analysis. Keep the Gaussian
witness in a *separate* lemma: Mathlib gives the mean and variance directly, but `m₃ = 0` and
`m₄ = 3` require three or four derivative computations of `exp(t²/2)` at `0` through
`iteratedDeriv_mgf_zero`, and that is the single most tedious part of the target — isolating it keeps
the algebraic theorem clean and lets the witness be deferred without blocking Targets 2 and 4. Do
**not** attempt the general-`n` Mehler identity: it needs the generating function, which Mathlib does
not have, and §6 does not ask for it.

### Target 2 — `fourth_hermite_moment_difference`

*Statement.* For two zero-mean laws with equal second moments,
`E[He₄(a)] − E'[He₄(a)] = κ₄ − κ₄'`.

*Ingredients.* `Polynomial.hermite` + `hermite_succ`; `ProbabilityTheory.moment` and
`centralMoment` for the measure-level corollary. **No `cumulant` in Mathlib** — define
`kappa4 m₂ m₄ := m₄ − 3 * m₂^2` locally.

*Plan.* Trivial once stated at the level of moment numbers, which is the right level. `He₄(a) =
a⁴ − 6a² + 3`, so `E[He₄] = m₄ − 6m₂ + 3`; subtracting under `m₂ = m₂'` leaves `m₄ − m₄'`, which
equals `κ₄ − κ₄'` by the definition above. One `ring` after `hermite_succ` unfolds `He₄`. State it in
the general **equal-second-moment** form rather than at `m₂ = 1`, so that the variance-matching
constraint of the paper's chain family is visible in the hypotheses instead of hidden in a
normalisation — that constraint is precisely what makes the innovation families differ only through
`κ₄`, and it is the sort of hypothesis that goes missing in prose. The tensor version of §6(2) is the
same identity index by index and should be a corollary quantified over a finite index type, not a
second proof.

### Target 3 — `gaussian_fisher_information_rate`

*Statement.* With `Σ_t = α²Σ₀ + Δ I`, `α = e^{−t}`, `Δ = 1 − e^{−2t}`, and
`I_t(θ) = ½ tr(Σ_t⁻¹ ∂_θΣ_t Σ_t⁻¹ ∂_θΣ_t)`, prove `e^{4t} I_t(θ) → ½‖∂_θΣ₀‖²_F` as `t → ∞`.

*Ingredients.* `Matrix.trace_mul_comm`, `trace_mul_cycle`, `trace_add`, `trace_smul`, `trace_sum`;
`ch05b_frobSq` / `ch05b_g_frob` from the audit **in place of** Mathlib's Frobenius instance (which is
scoped, and which lacks the `‖A‖²_F = tr(AᵀA)` lemma entirely); `Filter.Tendsto` with the
`ch02_Delta_tendsto_one` / `ch02_exp_neg_two_tendsto` pattern. `I_t` has no Mathlib counterpart and
is defined locally as `fisherGauss Σ dΣ := (1/2) * trace (Σ⁻¹ * dΣ * Σ⁻¹ * dΣ)`.

*Plan.* Since `∂_θΣ_t = α² ∂_θΣ₀`, the `α⁴` factors cancel against `e^{4t}` exactly, leaving
`e^{4t} I_t = ½ tr(M_t ∂Σ₀ M_t ∂Σ₀)` with `M_t := Σ_t⁻¹` — so the whole content is `M_t → I`. That
is the one real obstacle: matrix inversion is awkward to handle continuously in Lean. Do **not** fight
`Matrix.nonsing_inv` continuity in general; do it by hand for this one family, writing
`Σ_t = Δ_t • (1 + (α²/Δ_t) • Σ₀)` and bounding `M_t − I` entrywise by exhibiting an explicit right
inverse via `Matrix.inv_eq_right_inv` — the same pattern already used in `Ch05b.lean:2999`, where
`ch05b_frobSq (Q_t − e^{2t}•Σ₀⁻¹) ≤ n·(4t/m²)²` is proved this way from the spectral data. Then
`trace` is a finite sum, so the limit passes through it componentwise, and
`tr(∂Σ₀ ∂Σ₀) = tr(∂Σ₀ᵀ ∂Σ₀) = ‖∂Σ₀‖²_F` by `ch05b_g_frob` **provided `∂_θΣ₀` is symmetric**. Carry
that symmetry as an explicit hypothesis rather than deriving it from the Toeplitz form, so the
theorem covers the scale coordinate `q` as well as `ρ`. This is the highest Lean cost and the lowest
mathematical risk of the five; it earns its place because the factor `½` is exactly the kind of
constant the audit exists to catch, and because B3 is the one place in §2 where the result is exact
rather than asymptotic — the target checks the constant the numerics in §4 and §7 are compared
against (`I·e^{4t} = 0.497, 0.501, 0.500`).

### Target 4 — `scalar_score_gap_leading_coefficient`

*Statement.* In the scalar unit-variance case, the coefficient of `He₃` in the order-`e^{−4t}` term
of the expansion of `∇ log(P_t/γ)` is `κ₄/6`.

*Ingredients.* `Polynomial.hermite`, `Polynomial.derivative`, `Polynomial.aeval`. Nothing in Mathlib
addresses cumulant generating functions or their gradings.

*Plan.* Keep this entirely inside formal-power-series / polynomial algebra, as §6 instructs — do
**not** state it as a claim about the score of the actual law, which would drag in precisely the
remainder bounds §6 excludes. Encode the fourth-order truncation of `Λ` as an explicit polynomial in
the two graded variables `s` and `β` with coefficients the joint cumulants, take `∂_s`, and prove
that at unit variance the `He₃` coefficient of `Λ₄ − Λ₄^G` is `κ₄/6`. The grading `j + 2k` is not one
of Mathlib's standard gradings, so carry it by hand as a finite list of coefficients rather than
building a `GradedAlgebra` instance — the instance would cost more than the theorem is worth. Payoff:
this catches a wrong `1/6`, `1/24` or sign, which the numerical check of §4 discriminates only
weakly (`gap·e^{8t}` measured at `1.39, 1.50, 1.55` against a predicted `κ₄²/6 = 1.5`).

### Target 5 — Proposition 1 algebra (existing; extend, do not rebuild)

*What already exists and should be reused,* per the table in §1: `ch07_l_lmmse_excess` /
`ch07_l_lmmse_optimal` (`Ch07.lean:1336, 1354`) for optimality of any solution of the normal
equations in arbitrary dimension; `ch07_lmmse_covs_channel` (`Ch07.lean:1677`) for the two
covariances derived from the channel; `ch07_lmmse_fwd_map` / `ch07_lmmse_bwd_map`
(`Ch07.lean:831, 851`) for the two moment maps as genuine integrals;
`ch06_bp_forward_closure_iterated` / `ch06_bp_backward_closure_iterated` / `ch06_bp_belief_gaussian`
(`Ch06.lean:1237, 1281, 1321`) for the Gaussian-chain closure by induction over sites.

*The gap is one named object.* The moment-projection operator `P_G` is not defined anywhere in
`ThesisAudit`. The cheapest honest target is therefore **not** the full proposition but the step it
turns on: define `P_G` on the two-parameter family of positive Gaussian multiples (the module already
carries messages in exactly that form, as `C • ch06_nat λ r`), and prove that the composite
*multiply by the Gaussian unary likelihood → integrate against the transition → project* equals the
map `(m,v) ↦ (ρm, ρ²v + σ²)` applied to the precision-weighted combination. With that step, the base
case and the chain assemble from the already-proved iterated closure, and Proposition 1 reduces to
theorems already in the file. **What would still be informal afterwards, and should be recorded as
such:** the clause *"and returns the same estimator for every innovation law sharing the moments
(0,σ²)"*. The forward map is proved as a Gaussian–Gaussian convolution and is never quantified over
innovation laws; closing that clause needs a separate lemma that the mean and variance of a
convolution depend on the innovation only through its first two moments, which is a genuine (if
short) measure-theoretic statement and not covered by anything above.

---

## 3. The audit itself (retained)

### Verdicts

Reproduce with `./audit_status.py`.

| module | items | proved | instance | defined | skipped | discrepancy | build |
|---|---:|---:|---:|---:|---:|---:|---|
| Ch05a Gaussian matrix I | 65 | 54 | 0 | 11 | 0 | 0 | ok |
| Ch05b Gaussian matrix II | 68 | 59 | 1 | 8 | 0 | 0 | ok |
| Ch06 Gaussian BP | 49 | 36 | 0 | 13 | 0 | 0 | ok |
| Ch08a EM parameters I | 40 | 20 | 0 | 19 | 0 | 1 | ok |
| Ch08b EM parameters II | 40 | 30 | 0 | 10 | 0 | 0 | ok |
| Ch07 Laplace prior | 32 | 19 | 0 | 13 | 0 | 0 | ok |
| Ch10 Comparison | 34 | 21 | 0 | 12 | 0 | 1 | ok |
| Ch04 The ring | 50 | 37 | 0 | 13 | 0 | 0 | ok |
| Ch03 The model | 41 | 26 | 0 | 15 | 0 | 0 | ok |
| Ch02 Statistical mechanics | 56 | 36 | 6 | 10 | 4 | 0 | ok |
| Ch01 Introduction (+ ch09 EM kernel) | 5 | 4 | 0 | 1 | 0 | 0 | ok |
| **total** | **480** | **342** | **7** | **125** | **4** | **2** | ok |

125 of the 480 items are definitions and assert nothing. Of the 355 that assert something, 342
(96.3%) are proved in the generality the thesis states them. No `sorry`, no `axiom`, no
`native_decide` (grep-verified, not inferred from the build).

**What the verdicts mean.** *proved* — Lean proves the claim in the stated generality, under the
thesis's own modelling hypotheses. *instance-checked* — only a special case, a weaker form or a
conditional version; the obstruction is recorded with the item. *defined* — the display introduces
notation. *skipped* — not statable with this Mathlib; the missing ingredient is named. *discrepancy*
— the statement is wrong, with an explicit counterexample.

**Where it stops.** All four *skipped* items and most *instance-checked* ones have one cause:
**this Mathlib has no Itô stochastic integral**, so Itô's lemma, the generator identity, the
Fokker–Planck derivation and Anderson's time-reversal theorem cannot be *stated* here. Everything
downstream of that boundary is proved on `ℝᵈ` at arbitrary `d`. Separately, the converse direction of
`eq:sm-fokkerplanck` is proved on `ℝ` only.

**Findings.** Two errors in display mathematics, both minor, neither touching a derivation, a
theorem or a reported result: a Chapter 8 inline stationary-variance formula missing a squared-mean
term, and a Chapter 10 linear interpolation that should read 3.2 rather than 3.0. Both are one-line
edits, given in full in [`AUDIT_REPORT.md`](AUDIT_REPORT.md). A companion pass over *inline*
mathematics and quoted numbers found 37 further items, 12 of them outright errors, listed in
[`PROSE_FINDINGS.md`](PROSE_FINDINGS.md). Read the two counts together: most of what was found lives
in prose, not in displayed equations.

### Contents

| path | what it is |
|---|---|
| `ThesisAudit/ChNN.lean` | one module per thesis chapter (ch05 and ch08 split in two) |
| `ThesisAudit.lean` | root module; importing it builds the whole audit |
| `AUDIT_RESULTS/*.json` | per-formula verdicts, with thesis labels, line numbers and reasoning |
| `AUDIT_REPORT.md` | findings report, discrepancies first |
| `PROSE_FINDINGS.md` | inline-math and quoted-number audit |
| `audit_status.py` | prints the verdict table above from the JSON files |
| `make_derivations_tex.py` | regenerates the per-formula derivations document |
| `safe_build.sh` | serialised `lake build` (lockfile; safe under concurrent agents) |

### Building

```bash
cd lean
lake exe cache get        # fetch the mathlib build cache (large, far faster than building it)
lake build ThesisAudit    # must exit 0
./audit_status.py         # reproduce the verdict table
```

Requires the toolchain pinned in `lean-toolchain` (Lean 4.33.0); `elan` fetches it automatically.
`lean/.lake` is a symlink to a shared Mathlib build tree. Under concurrent agents use
`./safe_build.sh ThesisAudit.ChNN` instead of `lake build` directly.

To regenerate the derivations document (needs XeLaTeX or `tectonic`, and a monospace font with
mathematical coverage):

```bash
./make_derivations_tex.py && tectonic -X compile derivations.tex
```

---

## 4. Scope — what a green build does and does not mean

**The audit formalises display mathematics.** Each item is a formula that appears in a display
environment (or, in a few flagged cases, inline mathematics attached to one). The unit of work is a
formula, not an argument.

**A green build does not certify prose.** Lean checks the statements in `ThesisAudit/`, not the
sentences around the equations in the manuscript. The correspondence between the two was established
by reading, item by item, and each item's `note` in `AUDIT_RESULTS/*.json` records what was actually
established — including, in several cases, that it is less than the surrounding text claims. The
strongest evidence for this distinction is the audit's own history: adversarial fidelity passes
downgraded 40 items that had been recorded as proved without being proved, and found two `sorry`s
that a green build had not surfaced. Both of the display-math errors the audit found were in prose.

**A green build does not certify numerics.** Every number quoted in the manuscript — the
`9.2×10⁻¹⁵` relative agreement of the discretised recursion with eq. (6), the `10⁻⁹`
finite-difference check of eq. (10), the `1.6×10⁻⁹` nats of gradient-ascent/EM agreement, every
fitted slope, every table entry — is outside the scope of this directory. Where the audit does touch
a number it does so by re-deriving it symbolically and comparing, which is how the Chapter 10
interpolation discrepancy was found; that is a check on arithmetic, not a certification of a
computation.

**Nothing in the experiments is formally verified.** No code in `experiments/`, `src/`, `tools/` or
`hpc/` is covered. The Lean development shares no code with the implementation, references no
implementation file, and makes no claim about it. Where a Lean theorem and a piece of the pipeline
compute the same object, that agreement is a property of the mathematics they both implement, not
something the build checks.

**The new theory of `docs/THEORY_NOTES.md` is not yet covered at all.** §2 of this file is a plan,
not a result. Until those targets are proved, every statement in `THEORY_NOTES.md` §§1–7 — including
the exact ones — should be read as it labels itself there: exact identity, asymptotic statement with
a named error order, or numerical check.
