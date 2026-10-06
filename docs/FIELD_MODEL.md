# The φ⁴ field as a second laboratory

Written 1 October 2026 after Jerome's email (relaying a conversation with Marc): test the
estimator on Markov data from a simple physical model, the stochastic Allen–Cahn / φ⁴
equation, whose stationary states are a Gibbs measure. This note is the derivation the code
rests on (`src/markov_score/field.py`, `src/markov_score/tilted_kernel.py`), what is proved,
what is checked numerically, and one thing in the email that does not survive contact with
the chain. Experiments: E32 (oracles, deterministic) and E33 (learning).

## 1. The chain

The SPDE ∂_τφ = ∂ₓ²φ − V′(φ) + √(2Θ) η in one space dimension, V(φ) = (φ² − 1)²/4, is the
gradient flow of H[φ] = ∫[½(∂ₓφ)² + V(φ)] dx at temperature Θ; its stationary law is
exp(−H/Θ). (Notation: `docs/NOTATION.md`; τ is the SPDE's time, t the diffusion time, and the
code's `T` is Θ.) On a lattice of spacing h,

    H_h(φ) = Σᵢ [ (φᵢ₊₁ − φᵢ)² / 2h + h V(φᵢ) ],

a nearest-neighbour energy, so a spatial snapshot is exactly a first-order Markov chain in
the site index (Jerome's point, and the whole reason the machinery applies).

**Transfer operator.** With the site terms split symmetrically,

    𝖳(φ, φ′) = e^{−hV(φ)/2Θ} e^{−(φ′−φ)²/2Θh} e^{−hV(φ′)/2Θ},

a symmetric, positive, Hilbert–Schmidt kernel. Its largest eigenvalue λ₀ is simple with a
positive eigenfunction ψ₀ (Perron–Frobenius). The infinite-volume Gibbs measure is the
stationary chain with marginal π = ψ₀² and kernel

    K(φ′ | φ) = 𝖳(φ, φ′) ψ₀(φ′) / (λ₀ ψ₀(φ)),

reversible with respect to π. The lag-k covariance is Σₙ (λₙ/λ₀)^k ⟨ψ₀|φ|ψₙ⟩², so
ρ₁ := λ₁/λ₀ plays the role ρ plays for the linear chains and the correlation length in sites
is ξ = 1/log(1/ρ₁). There is no phase transition at any Θ > 0 in one dimension: ξ is finite at
every temperature and grows roughly like e^{E_kink/Θ} as Θ → 0, a crossover. The chain exposed to the package is standardised, a = φ/s with
s² = Var_π(φ).

**The tilted-walk identity.** Rearranging the eigen-equation,

    K(φ′ | φ) = N(φ′; φ, Θh) g(φ′) / Z(φ),   g = ψ₀ e^{−hV/2Θ},   Z(φ) = ∫ N(φ′; φ, Θh) g(φ′) dφ′,

and Z(φ) = λ₀ ψ₀(φ) e^{+hV(φ)/2Θ}: the kernel is a Gaussian step tilted by a function of the
*arrival* point, and the column normaliser is the h-transform. This is Jerome's "Gaussian
increment tilted by ψ₀(φ′)e^{−hV/Θ}/ψ₀(φ)", written so that the normaliser is explicit.

**Where the two laboratories meet.** A Gaussian linear chain with 0 < ρ < 1 is the member of
this family with log g quadratic (step variance q/ρ, tilt variance 1 + ρ). No other linear
chain is (Laplace driving noise is not a Gaussian step times a function of φ′), and no
other member of this family has state-independent driving noise. The two families intersect
at the Gaussian chain and nowhere else. The mixture kernel is misspecified on the
field chain through its *drift*, which is nonlinear: the conditional mean step of a tilted
walk is exactly γ (log Z)′(a) (exponential tilting: the drift is the gradient of the
log-normaliser), which for the field is Θh (log ψ₀)′(φ) + (h²/2) V′(φ), so the small-h limit
is the diffusion dφ = Θ (log ψ₀)′ dx + √Θ dW in the site index. The drift is the log-derivative
of the *ground state*, not −V′, and the noise is √Θ (the Wiener factor of exp(−∫(∂φ)²/2Θ));
an earlier version of this note wrote √(2Θ), the time-SPDE's noise, which is wrong here. The
misspecification is through the drift, not through the shape of the driving noise, a different
mechanism from every E08 axis.

## 2. The estimator: the learned pair energy as a transition kernel

Jerome proposed to parametrise a pair energy W_θ(φ, φ′) and learn it by Boltzmann learning,
E_data[∇W] − E_model[∇W], with log Z from the transfer-matrix pass. Written as a transition
kernel the same object is

    K_θ(a′ | a) = N(a′; a, γ) exp(B(a′)θ) / Z_θ(a),

with a linear basis B for log g (Legendre polynomials, or cubic B-splines) and the column
normaliser computed on the grid. Then

* the family contains the truth exactly (up to the basis), by the identity above;
* the M-step objective Q(θ) = ⟨Ξ, log K_θ⟩ is concave in θ at fixed γ (linear minus a sum
  of column log-partitions), solved by L-BFGS-B with the analytic gradient
  Ξ̃ = Ξ − (column mass) × (model column distribution), which is exactly the
  "posterior expectation minus model expectation" of Boltzmann learning, with no chain-wide
  partition function because the model is a kernel;
* the E-step is the existing exact BP; Fisher's identity is verified against finite
  differences of the evidence (`tests/test_tilted_kernel.py`).

Two capacities are run: Legendre degree 6 (7 parameters) and 24 B-splines (25 parameters,
the mixture kernel's count), so the capacity comparison with the paper's estimator is at
matched parameter count.

**Reversibility and the site energy (1 October, afternoon).** For any γ > 0 and any positive
g with finite Z_g(a) = ∫ N(a′; a, γ) g(a′) da′, the tilted walk K_g(a′|a) = N(a′; a, γ) g(a′)/Z_g(a)
is reversible with invariant density

    π_g ∝ g Z_g,        since π_g(a) K_g(a′|a) ∝ g(a) g(a′) N(a′; a, γ) is symmetric in (a, a′).

Equivalently K_g is the ground-state transform of the symmetric operator
𝖳(a, a′) = u(a) N(a′; a, γ) u(a′) with u = √(g/Z_g), whose Perron eigenfunction is ψ = √(g Z_g)
with eigenvalue exactly 1 (𝖳 ψ = u ∫ N g = u Z_g = ψ). So every tilted walk is the Gibbs chain of
the symmetric pair energy

    W(a, a′) = (a′ − a)²/2γ + f(a) + f(a′),        f = −½ log(g / Z_g),

and conversely every pair energy of this gradient form is a tilted walk (the identity of §1).
For the field's own (γ, g), f = (h/2Θ) V(s a) + const to 1e-10 (`tests/test_pair_energy_kernel.py`).
Three consequences.

1. *The directed kernel and the symmetric pair energy are one model class.* Jérôme's question
   (which to prefer) dissolves: no tilted walk is irreversible, and the symmetric form has the
   same single free function. What differs is the algorithm: the directed form's normaliser is
   per column (no chain-wide log Z, concave M-step in the tilt), the symmetric form's is the
   Perron eigenvalue.
2. *A fitted kernel reads as physics.* With a = φ/s, γ = Θh/s² and f(a) = (h/2Θ) V(s a) + const, so
   Θ̂ = γ̂ s²/h and V̂(φ) = (2Θ̂/h) f̂(φ/s) + const (`pair_energy_kernel.potential_readings`: the
   temperature, the barrier height, the well position). The temperature is identified by the
   bond variance alone, which is the one parameter that enters no tilt.
3. *The physicist's parametrisation.* log g = log ψ_f − f is a transcendental function of the
   potential (hence the 7- and 25-parameter bases), while f itself is a quartic. `PairEnergyKernel`
   (`src/markov_score/pair_energy_kernel.py`) parametrises f = Σ_{m≤4} c_m (a/2)^m and log γ: five
   parameters, exactly specified for the field, kernel by one dense eigensolve of 𝖳_f per
   evaluation (15 ms on the 401-point grid), invariant law ψ_f² in closed form, M-step by L-BFGS-B
   with the analytic gradient of the Perron eigenvector (first-order perturbation theory in the
   complete discretised eigenbasis; <ψ_n|∂𝖳/∂c_m|ψ_0> = −(λ_0 + λ_n)<ψ_n|B_m ψ_0>), not concave.
   This is literally the pair energy W_θ of Jérôme's proposal with its Boltzmann learning
   written as a kernel. Arm `em_pair_quartic`, E33 amendment 1 (predictions P7, P8), its own
   cluster shards (`hpc_E33q_s*`).

Because the data are strictly stationary, every EM arm uses its *own* kernel's invariant
density as the initial law (`em.fit_em` now accepts a callable `log_mu`); the M-step
maximises the edge term only, and any decrease of the evidence is recorded, not hidden.

## 3. What does not extend: Gaussian closure of the messages

Proposition 1 identifies moment-projected sum-product with the LMMSE of the covariance twin,
and its proof uses the linear transition. On the field chain the projected recursion is
*not a well-posed estimator*, and for an unexpected reason. From a state far outside the
bulk the kernel jumps to a′ ~ (a/4cγ)^{1/3}, almost regardless of a, so the backward
message R(a) = E[ℓ(aᵢ₊₁) | aᵢ = a] is nearly flat across the whole operating grid (it decays
only as exp(−c′a^{2/3}) far beyond it). Its moment projection is a near-uniform Gaussian that
discards the backward information. Measured: 28% of a projected message's mass in the outer
5% of the (401, 8) grid at t = 0.47 (`tests/test_bp_projected.py`). On a grid wide enough
for the Gaussian and Laplace chains' backward messages the proposition's identity holds to
10⁻⁸, so this is a property of the field chain, not of the implementation.

Consequently "the Gaussian closure" of the field chain is its covariance twin, the LMMSE of
the true covariance, by Proposition 1's identification; the projected recursion is kept as a
diagnostic and its edge mass is recorded per cell (E32 P2). Jerome's expectation that closure
"fails badly" by smearing across domain walls is then measured by the linear floor's error,
which is what the floor has always measured.

## 4. Numerics and what is exact

ψ₀ is computed by a dense symmetric eigensolve of the trapezoid-discretised operator on a
fine physical grid (2401 points on [−12, 12]). For this smooth, super-exponentially decaying
kernel the trapezoid rule converges faster than any power of the spacing: halving the
spacing and widening the domain move λ₀, ρ₁ and s by less than 10⁻¹⁰ (`tests/test_field.py`).
The far tail where ψ₀ falls below 10⁻¹³ of its maximum (stationary mass below 10⁻²⁶) is
carried as log ψ₀ = −∞; a kernel column leaving such a state is the normalised 𝖳 column.

Two samplers: continuum rejection (propose the Gaussian step, accept with g/max g, log ψ₀ by
cubic spline; burn-in from the slowest mode's rate, as `stationary.py` does) and lattice
inverse-CDF on the fine grid's finite-state chain (transfer-matrix sampling). They agree in
variance, lag-one correlation and kurtosis within Monte Carlo error.

The reference (oracle) score is grid BP with the chain's kernel and stationary law on the operating
grid; E32's refinement part measures its quadrature error against (1601, 8). The step width
√(Θh)/s is at least 8 grid cells on every cell used, so the usual resolution condition is met.

## 5. Cells (E32 → E33)

At h = 1 the iso-correlation line ρ₁ = 0.85 passes through Θ = 0.30 (ρ₁ = 0.848, marginal
excess kurtosis −1.35). The temperature sweep keeps ρ₁ = 0.85 by solving h*(Θ) for
Θ ∈ {0.3, 0.5, 1, 2}; the refinement sweep starts from the Θ = 0.3 cell and halves h twice
with L ∝ 1/h, so the physical length (32 lattice units) is fixed and the correlation length
in sites grows 6 → 12 → 24. This is the reading of Jerome's prediction that the advantage of
structure grows under refinement because a fixed window's receptive field does not.

## 6. What the email got right, and the two corrections

Right: exactly Markov; the h-transform form of the kernel; the mixture kernel is
misspecified and a learned kernel is needed (now a measured claim, E33 P1/P2); exact
transfer-matrix sampling; per-step non-Gaussianity O(h) while the marginal is O(1)
non-Gaussian; correlation length in sites ∝ 1/h; no transition in 1D.

Corrections: (i) Gaussian closure of the messages is not well-posed on this chain (§3), so
the closure arm is the covariance twin; (ii) BP is indifferent to the correlation length but
not to the step width, so the fine-h end is bounded by the grid, which §4 checks rather than
assumes. One addition: the receptive-field prediction is partly already measured along
E14's ρ axis; the refinement cells give it a physical reading.
