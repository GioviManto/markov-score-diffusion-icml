# Theory notes: what diffusion noise erases, order by order

Status: working derivation, 14 September 2026. Everything here is either (a) an exact
identity, (b) an asymptotic statement with the error order named, or (c) a numerical
check. Each item is labelled. Nothing here is yet a "theorem in the paper" until the
proof in `paper/appendix` is written and the Lean targets in `lean/README.md` are
either proved or listed as informal.

## 0. Setting and conventions

Clean data `a ∈ R^L`, law `P_0`, finite moments of every order (all innovation
families used: Gaussian, Laplace, Student-t with ν ≥ 9 for eighth moments, uniform,
symmetric Gaussian mixture, generalised Gaussian β ≥ 1). Coordinatewise
Ornstein–Uhlenbeck channel, `x_t = α a + √Δ z`, `α = e^{-t}`, `Δ = 1 − e^{-2t}`,
`z ~ N(0, I_L)`. `γ = γ_L` is the standard Gaussian on `R^L`; `He_k` are the
probabilists' Hermite polynomials (`He_0 = 1, He_1 = x, He_2 = x² − 1, He_3 = x³ − 3x,
He_4 = x⁴ − 6x² + 3`), multi-index products `H_ν(x) = Π_i He_{ν_i}(x_i)`, degree `|ν|`.

The chain family used in the paper is variance-matched: `Var a_i = 1`,
`Cov(a_i, a_j) = ρ^{|i−j|}` for every innovation law. So `Σ_0` is the same for every
family and the families differ only through cumulants of order ≥ 4 (odd cumulants
vanish by symmetry).

Two operators:

* `S_t(x) = ∇ log P_t(x)` — the true score.
* `S_t^G(x) = −Σ_t^{-1} x`, `Σ_t = α² Σ_0 + Δ I` — the score of the covariance-matched
  Gaussian twin; by Proposition 1 of the paper this is exactly what single-Gaussian
  message closure (moment-projected BP) computes.

## 1. Exact attenuation of Hermite moments (Theorem A — exact, all t)

**Statement.** For every multi-index `ν` with `E_{P_0}|a|^{|ν|} < ∞`,

    E_{P_t}[H_ν(x_t)] = e^{−|ν| t} · E_{P_0}[H_ν(a)].

**Proof.** The classical Mehler identity: for scalar `a` and `z ~ N(0,1)`,
`E_z[He_n(α a + √(1−α²) z)] = αⁿ He_n(a)`. (Generating-function proof:
`exp(sx − s²/2) = Σ_n He_n(x) sⁿ/n!`; substitute `x = αa + √(1−α²)z` and take `E_z`,
which turns `exp(s√(1−α²) z − s²(1−α²)/2)` into 1; what is left is
`exp(sαa − s²α²/2) = Σ_n He_n(a)(αs)ⁿ/n!`.) Coordinates are independent given `a`, so
the product form gives the multi-index version; then integrate over `a ~ P_0`. ∎

**Consequences (exact).**
* Cumulants: `κ_m(x_t) = e^{−mt} κ_m(a)` for `m ≥ 3` (Gaussian noise has no cumulants
  above order 2); `Cov(x_t) = e^{−2t} Σ_0 + Δ I`.
* Degree-2 Hermite moment: `E_{P_t}[He_2-type] = e^{−2t}(Σ_0 − I)`. This is the
  "correlation coordinate".
* Degree-4 Hermite moment: `E_{P_0}[H_ν]`, `|ν| = 4`, equals the fourth cumulant
  `κ_4[ν]` plus symmetrised products of `(Σ_0 − I)` entries. The products are the same
  for every variance-matched family, so **the degree-4 Hermite moments of two
  variance-matched laws differ exactly by their fourth-cumulant tensors**, and that
  difference is attenuated by `e^{−4t}`.

Nothing above needs `P_0/γ ∈ L²(γ)`; only finite moments. This matters because for
the Laplace chain `P_0/γ ∉ L²(γ)` (exponential tails), and for the Gaussian chain at
`ρ = 0.85` it also fails (`χ²(N(0,Σ_0) ‖ N(0,I)) = ∞` because `λ_max(Σ_0) ≈ 12 > 2`).
So the "spectral decomposition of the OU semigroup on L²(γ)" route is NOT available
for our prior; Theorem A is the moment version that is.

## 2. High-noise expansion of the score and of parameter information (Theorem B — asymptotic)

Write `s = αx/Δ`, `β = α²/Δ` (both `O(e^{−t})`, `O(e^{−2t})`), and the
"damped cumulant generating function"

    Λ(s, β) := log E_{P_0}[ exp( s·a − β |a|²/2 ) ].

Exact identities (Bayes' rule, no approximation):

    log P_t(x) = log γ_Δ(x) + Λ(αx/Δ, α²/Δ),
    E[a | x_t = x] = ∇_s Λ(s, β)|_{s = αx/Δ, β = α²/Δ},
    S_t(x) = −x/Δ + (α/Δ) ∇_s Λ.

`Λ` is real-analytic in `(s, β)` for `β > 0` (the Gaussian damping makes the tilted
measure integrable for every `s`), and its Taylor coefficients at `(0,0)` are the joint
cumulants of the linear form `s·a` and the quadratic form `|a|²/2` under `P_0`.
Grade a monomial `s^j β^k` by `j + 2k` ("order in α"). By symmetry only even orders
appear:

    Λ = Λ_2 + Λ_4 + Λ_6 + …,
    Λ_2 = ½ sᵀ Σ_0 s − ½ β tr Σ_0,
    Λ_4 = (1/24) κ_4[s,s,s,s] − ¼ β Σ_i cum(s·a, s·a, a_i, a_i) + ⅛ β² Σ_{ij} cum(a_i,a_i,a_j,a_j).

For the Gaussian twin `Λ^G` has the same `Λ_2` and Gaussian values of the fourth-order
joint cumulants. Since every fourth-order joint cumulant of polynomial forms in `a`
differs from its Gaussian value by a linear contraction of `κ_4`, **`Λ_4 − Λ_4^G` is
linear in `κ_4`** and vanishes iff `κ_4 = 0`.

### B1. Score gap between the true score and its covariance twin

    S_t − S_t^G = (α/Δ) ∇_s (Λ − Λ^G) = (α/Δ) [ ∇_s (Λ_4 − Λ_4^G) + O(α⁵) ].

`∇_s Λ_4` is `O(α³)`, so the gap is `O(α⁴) = O(e^{−4t})` pointwise on `|αx| ≲ 1`, and

    E_{P_t} ‖S_t − S_t^G‖² = C_4(P_0) e^{−8t} (1 + O(e^{−2t} λ_max(Σ_0))),
    E_{P_t} ‖E[a|x_t] − E^G[a|x_t]‖² = C_4'(P_0) e^{−6t} (1 + …),

with `C_4` an explicit quadratic form in `κ_4`. In the scalar unit-variance case the
leading term is `(κ_4/6) He_3(x) e^{−4t}` so `C_4 = κ_4²/6` (checked numerically
below, §4). For the chain `C_4` is a contraction of the tensor
`κ_4[i,j,k,l] = κ_4(ε) Σ_{m ≤ min(i,j,k,l)} ρ^{i+j+k+l−4m}` (the initial site is
Gaussian and contributes nothing).

### B2. Fisher information about a parameter, by lowest Hermite degree

Let `P_0^θ` be a smooth family and `I_t(θ) = E_{P_t}[(∂_θ log P_t)²]` the (population)
Fisher information carried by one noised sample at level `t`. Then
`∂_θ log P_t = ∂_θ Λ`, and grading as above:

* if `∂_θ Σ_0 ≠ 0` (a correlation or scale coordinate; `ρ`, `q`):
  `I_t(θ) = e^{−4t} · ½ ‖∂_θ Σ_0‖_F² · (1 + O(e^{−2t} λ_max))`;
* if `∂_θ Σ_0 = 0` but `∂_θ κ_4 ≠ 0` (a shape coordinate at fixed variance, e.g. the
  generalised-Gaussian exponent `β`, or a mixture-shape parameter under the
  variance-matching constraint):
  `I_t(θ) = e^{−8t} · (1/24) ‖∂_θ κ_4‖²_{H} · (1 + O(e^{−2t} λ_max))`,
  where `‖·‖_H` is the Hermite (symmetrised) norm of the tensor; in the scalar case it
  is `(∂_θ κ_4)²/24`.
* Cross-information between a degree-2 and a degree-4 coordinate is `O(e^{−6t})` at
  most and, because `He_2` and `He_4` are `γ`-orthogonal, the *leading* terms do not
  couple. Hence the efficient information (after projecting out the other coordinate)
  has the same leading exponent as the raw diagonal. (Empirically, `exp_22` finds the
  nuisance cost of the shape coordinate ≤ 1.15 — consistent.)

**Sample-complexity corollary.** To estimate a shape coordinate to fixed precision
from data observed at level `t` needs `M ∝ e^{8t}` sequences; a correlation coordinate
needs `M ∝ e^{4t}`. The ratio of the two informations is `∝ e^{−4t}`: this is the
robust *contrast* statement that survives the pre-asymptotic corrections, because the
`(1 + O(e^{−2t}λ_max))` factors partially cancel.

### B3. Exact Gaussian-chain information (no asymptotics needed)

For the Gaussian chain `x_t ~ N(0, Σ_t(θ))` exactly, so

    I_t(θ_i, θ_j) = ½ tr( Σ_t^{-1} ∂_i Σ_t Σ_t^{-1} ∂_j Σ_t ),   ∂_i Σ_t = α² ∂_i Σ_0.

This is closed form for all `t` and all `ρ`. It gives the full curve (not only the
asymptote) of the correlation information, the exact crossover, and the exact
`ρ–q` collinearity that `exp_22` observed (`corr_{ρq} → 0.9`). It also gives the
degree-2 rate `e^{−4t}` exactly as `t → ∞` with constant `½‖∂_ρ Σ_0‖_F²`.

### B4. The crossover scale

The expansion parameter is not `α` but `α² λ_max(Σ_0)`, with
`λ_max(Σ_0) ≤ (1+ρ)/(1−ρ)` (Toeplitz bound; ≈ 12.3 at ρ = 0.85). Define

    t_* (ρ) := ½ log( (1+ρ)/(1−ρ) ).

For `t ≪ t_*` the long-wavelength covariance modes are still `O(1)` and the decay is
governed by the full spectrum of `Σ_t` (slower, mode-dependent); for `t ≫ t_*` the
asymptotic exponents `2m` per Hermite degree `m` apply. At `ρ = 0.85`, `t_* ≈ 1.26`,
which is exactly where the frozen experiments stop (`exp_22`: `t ≤ 1.6`; `exp_02`:
`t ≤ 3`). This explains, rather than contradicts, the "shallower than predicted"
slopes recorded in the thesis (`−6` instead of `−8`, `−3` instead of `−4` over
`t ≤ 1.6`): they are pre-asymptotic. The consecutive-interval slopes of `exp_02`
steepen monotonically (`−2.9, −4.5, −5.7` for the posterior-mean MSE toward `−6`;
`−2.8, −3.2, −3.8` for the relative score error toward `−4`; `−5.5, −6.5, −7.5` for
`1 − cos` toward `−8`), exactly as the crossover picture predicts.

For the phase diagram this gives the natural dimensionless axis: `e^{2t}/λ_max(Σ_0)`,
i.e. `t − t_*(ρ)`.

## 3. What is proved, what is asymptotic, what is checked

| item | status | assumptions |
|---|---|---|
| Theorem A (Hermite/cumulant attenuation) | exact, elementary | finite moments of the relevant order |
| `Λ` identities for posterior mean and score | exact | none beyond a density for `P_0` |
| Grading of `Λ` by order in `α`; `Λ_4 − Λ_4^G` linear in `κ_4` | exact algebra | finite fourth moments |
| B1, B2 leading-order rates | asymptotic as `t → ∞` with named error order | finite eighth moments (for the `L²(P_t)` remainder); dominated convergence for the tail `|αx| ≳ 1`, which has probability `≤ exp(−c e^{2t})` under `P_t` |
| B3 | exact | Gaussian chain |
| B4 crossover | heuristic bound; exact for the Gaussian chain via B3 | — |

A fully rigorous version of B1/B2 for finite `L` is a Taylor expansion of `Λ` with an
explicit remainder plus a tail bound, and is planned for the appendix. The paper will
state B1/B2 as a theorem with the `(1 + O(e^{−2t}λ_max))` correction and give the proof
in the appendix; the scalar case is short enough for the main text.

Related literature to cite honestly (this is not claimed as new mathematics but as a
new *use* with an exactly solvable non-Gaussian test bed): Mehler's formula / OU
spectral theory; the low-SNR expansions of mutual information and MMSE in Gaussian
channels (Prelov & Verdú 2004; Guo, Shamai & Verdú 2005, 2011), where the second-order
SNR term of the MMSE depends on fourth-order statistics. Our `SNR = α²/Δ`, so
`e^{−2t}` is the SNR. What is new: (i) the order-by-order separation of *parameter
information* in a structured prior into correlation (`e^{−4t}`) and shape (`e^{−8t}`)
with explicit constants and crossover; (ii) the identification of the covariance-twin
score gap as the `e^{−8t}` object (via Proposition 1); (iii) exact verification with
tree-exact inference and the learning consequence (§5).

## 4. Numerical verification, scalar case (done 14 Sep 2026, `scratchpad/scalar_exponents.py`)

Fine quadrature (8001 points on [−16,16], FFT convolution), unit-variance generalised
Gaussian priors. Local log-slopes over `t ∈ [1, 2]`:

| quantity | prediction | measured (Laplace) | measured (β=1.5) | measured (β=4) |
|---|---|---|---|---|
| `E(S_t + x)²` (score gap to `N(0,1)`) | −8, constant `κ_4²/6 = 1.5` | −7.86, −7.93; `gap·e^{8t}` = 1.39, 1.50, 1.55 | −7.97, −7.78 | −8.05 |
| `I_t(β)` shape at fixed variance | −8 | −7.7, −8.6 | −8.1 | — |
| `I_t(v)` variance | −4, constant ½ | −3.99, −4.00, −4.01; `I·e^{4t}` = 0.497, 0.501, 0.500 | −3.99, −4.00 | — |

Beyond `t ≈ 2.5` the eighth-power quantities fall below the double-precision floor of
the log-density finite differences (`~1e-9`) and the scalar check is no longer
informative; the chain experiments must use the closed-form B3 for the degree-2
coordinates and a fine grid (`N_g ≥ 801`) with replicate seeds for the shape
coordinate, and must stop at the level where the outer-product estimate is still above
its numerical floor (to be measured by the Gaussian-chain control, where the closed
form is available).

## 5. Learning consequence to test (the experiment that ties theory to the EM–BP claim)

Fit the mixture-innovation kernel by EM–BP from `M` sequences all observed at one
level `t`, for `t` on a grid, 16 seeds, to a validation-selected checkpoint. Measure
the RMSE of `ρ̂` and of the fitted excess kurtosis `κ̂_4`. Predictions:

    RMSE(ρ̂)   ∝ e^{2t}  (for t ≫ t_*),    RMSE(κ̂_4) ∝ e^{4t},

so the ratio grows like `e^{2t}`. The one-view-per-chain protocol of the headline
comparison spreads `M` over twelve levels; by B2 the shape information is carried
almost entirely by the lowest four levels, which quantifies why the shape coordinate is
slow through the channel (thesis Remark 1) beyond the mixture-label mechanism.

## 6. Lean targets (algebraic, protective of central derivations)

1. `mehler_he_n` for `n ≤ 4`: for any real `a`, `α`, and a random variable `z` with the
   first four moments of `N(0,1)` (`0, 1, 0, 3`), `E[He_n(αa + √(1−α²) z)] = αⁿ He_n(a)`.
   Pure polynomial identity over `ℝ` once `E` is linear; mathlib has `Polynomial.hermite`
   (probabilists' convention, `hermite_succ`) and `gaussianReal` moments.
2. `fourth_hermite_moment_difference`: for two zero-mean laws with equal second moments,
   `E[He_4(a)] − E'[He_4(a)] = κ_4 − κ_4'`. (Scalar; the tensor version is the same
   identity index by index.)
3. `gaussian_fisher_information_rate`: for `Σ_t = α²Σ_0 + ΔI` and the closed-form
   `I_t`, `e^{4t} I_t(θ) → ½ ‖∂_θ Σ_0‖_F²` — the finite-dimensional matrix-analysis part.
4. `scalar_score_gap_leading_term`: the coefficient `κ_4/6` of `He_3` in the scalar
   expansion of `∇ log(P_t/γ)` at order `e^{−4t}` (formal power-series algebra).
5. Existing: Proposition 1 (LMMSE closure) algebra — check what `ThesisAudit/Ch07.lean`
   already covers and reuse.

Do not formalise the remainder bounds of B1/B2; they are analysis, not algebra, and
the value of Lean here is catching an algebraic slip in a constant, not in an
`O(·)` term.
