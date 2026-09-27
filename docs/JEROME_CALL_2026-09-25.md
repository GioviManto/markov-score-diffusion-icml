# Jerome, 25 September 2026 — the question ledger

What this document is: every question Jérôme raised in the call of 25 September 2026,
and our answer to it. One section per question. A question is **OPEN** until an answer
is backed by a registered run under `docs/EXPERIMENT_REGISTRY.md`; until then the
section says what we believe and what would settle it.

Source: `call-jerome-25_09.vtt` (full transcript, 3033 utterances). Quotations below are
his words from that transcript, lightly de-stuttered. Where the call reached a wrong or
incomplete answer, §5 records the correction — those are ours to make, not his.

Status vocabulary: **OPEN** (no evidence yet) · **IN PROGRESS** (experiment registered
and running) · **ANSWERED** (registered run, evidence cited) · **SCOPED OUT** (with the
reason).

---

## 1. Ledger

| # | Question | Closed by | Status |
|---|---|---|---|
| Q1 | Is Gaussian-message BP provably the same as using the Gaussian (linear) score? Is the route through the Kalman filter? | Lemma + E23 | **IDENTIFICATION CONFIRMED** (6.7e-16 against our own RTS smoother); the *lemma text* drafted for it was **false as stated** — §2 Q1 |
| Q2 | With a Laplace *ansatz* for the messages and a Laplace innovation, is BP exact? | E22, E22c | **ANSWERED — no.** Confirmed at scale: 432/432 cells |
| Q3 | What do the messages $\mu_k$ actually look like? Is it Gaussian $\ast\ P_\varepsilon$? | E22, E22c | **ANSWERED — his conjecture is very nearly right** |
| Q4 | Is closure possible only for the Gaussian, or for any finitely-parameterised family? | Lemma + E22 | **ANSWERED empirically** |
| Q5 | Can BP be done in Fourier space — truncation on modes instead of a real-space grid? | E23 | **ANSWERED — no, and for a reason worth reporting.** §2 Q5 |
| Q6 | Can $K$ be a neural network, fitted by EM through BP? Does it overfit? | E25 | OPEN — two void attempts (N-23, N-24/N-25); third running (671442–5) |
| Q7 | Posterior mean for one noisy sequence: how long for a network, how long for BP? | E24, E24gpu2 | **ANSWERED on BOTH devices** — BP is ~15× *slower*; not a CPU artefact, §2 Q7 |
| Q8 | Can we generate without a score — flat marginal, run BP once? | E26 (weak), **E27** (strong) | **WEAK FORM ANSWERED — yes.** Strong form built and registered as E27, predeclared, not yet run |
| Q9 | Is the score compared at a fixed $t$ or averaged over $t$? | E24 | PARTIAL (answered in call; to be reported both ways) |
| Q10 | Do the information curves quantify information about $\rho$ and $\beta$, or about $\kappa_2$ and $\kappa_4$? | E01 relabel | RESOLVED in call — implementation pending |
| Q11 | How does EM–BP scale in $M$ (sequences) and $L$ (sites)? | E24 | **ANSWERED** — the call's answer was wrong, and so was ours; §5.1 |

---

## 2. Questions

### Q1 — Gaussian closure vs the linear score

> *"Is there a proof in a sense, or...?"* … *"But maybe the route is through the Kalman
> filter, actually."*

**Status:** The **identification is confirmed numerically**; the **lemma text drafted for
it on 26 Sep was false as stated** and must not be used. Details below, because the way it
was false is the interesting part.

**Confirmed (26 Sep).** An RTS smoother was written from scratch, independently of
`bp_gaussian`, for the linear-Gaussian state-space model matched to the chain's first two
moments. Against moment-projected sum-product: `max|BP − RTS| = 6.66e-16` on the posterior
means and `4.44e-16` on the variances; against a dense LMMSE solve `2.0e-15`; and against
$-\Sigma_t^{-1}x$ as the score, `8.9e-15`. So Jérôme's guess — *"maybe the route is through
the Kalman filter"* — is right, and the four objects are the same object.

**But the lemma as drafted was FALSE, and the error is instructive.** The draft dropped
Proposition 1's hypothesis *"and an initial message Gaussian or projected by $\Pgauss$"*
(`paper/sections/prop-lmmse.tex:16`), on the grounds that it looked like a technicality.
It is not. Counterexample, computed: $L=32$, $\rho=0.85$, $q=0.2775$, $v_0=1$, innovation
$\nu=\mathrm{Laplace}(0,1/\sqrt2)$, $2\times10^5$-point quadrature, $t=1.0$ — the posterior
means differ from the RTS/LMMSE answer by **4.148e-02 at site 0** and 5.404e-03 at site 5.
The reason is that $\Pgauss$ does **not commute with the likelihood tilt**, so a
non-Gaussian initial message is not equivalent to its projection, and the error propagates
down the chain rather than being absorbed at the first site.

Three further errors in the same draft, all now corrected in the working notes:

1. *"If $v_0 = q/(1-\rho^2)$ … then $\Sigma_0 = \rho^{|i-j|}$"* — **missing the factor
   $v_0$**. The correct statement is $\Sigma_0 = v_0\,\rho^{|i-j|}$.
2. *"the trivial per-site $\alpha/(\alpha^2+\Delta)\cdot x_i$ at $\rho=0$"* — holds only at
   unit per-site variance. In general it is $\alpha v_i/(\alpha^2 v_i + \Delta)\cdot x_i$.
3. *"both algorithms carry a state in $\mathbb{R}^2$ per site and advance it by the same
   fixed map"* — **false for the backward sweep.** BP's backward map is autonomous in
   $(\lambda, r)$; RTS's gain $J_i = \rho P_{i|i}/P_{i+1|i}$ depends on stored forward
   quantities. The two agree on the *answer*, not on the recursion.

**And the novelty claim was false too.** The identification is already in this repository:
`reports/compendium/parts/12-gaussian-closure-lmmse.tex:124-175` contains it, and the
archival thesis `compendium/chapters/ch06-gaussian-bp.tex:563-571` reports 80
configurations with maximum disagreement 1.3e-15. The genuine gap is narrow and purely
editorial: `paper/sections/prop-lmmse.tex` never says the word *Kalman*, while
`paper/body/06-related.tex:15-16` already cites `\cref{prop:lmmse}` for *"the Gaussian
part of its score is the Kalman smoother of the covariance twin"*. One sentence in the
proposition's remark closes it. **No new lemma is needed and none should be written.**

**What we have.** Proposition 1 of the current manuscript states that single-Gaussian
message closure computes exactly the LMMSE estimator of the covariance-matched Gaussian
twin, for every innovation law with the same first two moments. `exp_02` part (B)
measures the resulting error. So the content exists; what does not exist is the clean
statement Jérôme asked for, with the Kalman route made explicit.

**What we believe.** The forward–backward sweep with mean/variance-only updates *is* the
Kalman filter–smoother for the linear-Gaussian state-space model matched to the chain's
first two moments. Closure under the Gaussian family holds because the two message
operations — multiplication by a Gaussian likelihood and convolution with a Gaussian
transition — both preserve Gaussianity, and mean/variance are sufficient statistics.

**What closes it.** A lemma in `docs/THEORY_NOTES.md` with three equalities —
Gaussian-message BP = Kalman smoother = exact BP on the covariance-matched Gaussian
model = the linear score — plus the numerical check that falls out of E23: the
order-2 rung of the cumulant ladder must reproduce `bp_gaussian.gaussian_chain_bp` to
machine precision.

**His view on how much to sell it:** *"So this is not really interesting."* Agreed — it
is a validation check and a rung of a ladder, not a result.

---

### Q2, Q3, Q4 — the message shapes, answered together

E22 answers all three at once, because they are one question: *what family do the BP
messages live in?* Evidence: `outputs/E22/` (four innovation families × twelve levels ×
64 chains, grid (801, 12), messages fitted by direct KL minimisation over each
candidate's own parameters).

**Confirmed at scale (E22c, job 669837, `outputs/hpc_E22c/`).** The laptop result was one
configuration; E22c swept $\rho \in \{0.5, 0.7, 0.85, 0.95\}$, $L \in \{32, 128\}$,
$N_g \in \{801, 1601\}$, four innovation families, twelve levels — **432 cells**, and both
verdicts hold in **432/432**: the ansatz is at the measured quadrature floor at site 1, and
strictly above it and growing with depth in the interior. Growth factor from site 1 to the
deepest site spans 8.0e7 to 1.15e11 (median 1.4e10).

**The load-bearing control:** doubling the grid does not move the error.
$\mathrm{KL}(N_g{=}1601)/\mathrm{KL}(N_g{=}801) = 1.000$ over 1512 cells. The failure is
the *model*, not the quadrature — which is the one alternative explanation that would have
made the whole result vacuous. There is also no $L$-dependence: the error saturates with
depth rather than being an end effect.

**The construction being tested.** Jérôme's *"Gaussian plus the family of epsilon"* is,
made precise, **a Gaussian pushed through one transition of the true chain**:
$\mu = \mathcal{N}(m, v)$ dilated by $\rho$ and convolved with $p_\varepsilon$. Two free
parameters. Compared against a plain Gaussian (2 params) and a Gaussian+Laplace mixture
(4 params).

#### The answer to Q3: his conjecture is very nearly right, and dramatically better than a Gaussian

KL(message ‖ best fit), median over chains, at the deepest site (31):

| family | $t$ | Gaussian | **transition ansatz** | Gaussian+Laplace |
|---|---|---|---|---|
| gaussian (control) | any | 1e-17 | 1e-15 | 1e-16 |
| laplace | 0.05 | 3.5e-2 | **2.2e-5** | 1.2e-3 |
| laplace | 3.00 | 3.4e-3 | **7.7e-4** | 5.6e-4 |
| student | 0.05 | 2.9e-2 | **1.8e-6** | 5.0e-4 |
| student | 3.00 | 5.4e-3 | 8.8e-4 | **3.0e-4** |
| uniform | 0.05 | 3.4e-2 | **3.7e-4** | 3.4e-2 |

His two-parameter family beats a plain Gaussian by **1 600×** (Laplace) and **16 000×**
(Student-$t$) at small $t$. It also beats the *four*-parameter Gaussian+Laplace mixture by
**50×** there. That is worth saying plainly: **the right two-parameter family beats a
wrong four-parameter family by a factor of fifty.** Structure beats capacity — which is
the paper's own thesis, showing up in a place we did not plant it.

(On `uniform`, Gaussian+Laplace matches the plain Gaussian exactly: the optimiser drove
the Laplace weight to zero. The family simply has nothing to offer a compact-support
innovation.)

#### The answer to Q2: not exact — exact one step in, then it fails by ten orders

| | KL at site 1 | KL at site 31 | growth |
|---|---|---|---|
| laplace, $t = 0.98$ | 1.3e-14 | 6.3e-4 | 4.9e10 |
| student, $t = 0.98$ | 2.2e-14 | 4.3e-4 | 1.9e10 |
| uniform, $t = 0.98$ | 3.0e-14 | 3.5e-4 | 1.1e10 |

**P1 holds everywhere** (site-1 KL between 2.5e-15 and 3.1e-14, i.e. machine precision —
as derived, because $L_1$ is the Gaussian prior pushed through exactly one transition).
**P2 holds everywhere**: the divergence grows by ten orders of magnitude into the chain
interior, for every family and every level. The family is *not* a fixed point of the
recursion, exactly as the algebra said: $(G \ast p_\varepsilon)\cdot\mathcal{N}$ is not of
the form $G' \ast p_\varepsilon$.

So Jérôme's instinct — *"to me, it's not immediate that they match"* — was right, and the
honest version of his own follow-up (*"the good Gaussian BP iteration, you can only do for
the Gaussian"*) is confirmed.

#### The answer to Q4

Both halves, now with evidence. *Exact* closure: Gaussian-only, and site 1 versus site 31
is the demonstration. *Projected* closure: available for any family, with a price that is
now measured rather than asserted — 1e-5 to 1e-3 nats per message for the best
two-parameter family. Whether that price is acceptable is a question about the downstream
score error, not about the messages, and it is what E23 will answer.

#### P3 was falsified, and the reason is instructive

Predicted: the ansatz error would be smallest at large $t$, because the likelihood
flattens and the product step stops doing anything. **Measured: the opposite.** For
Laplace and Student the error *grows* monotonically with $t$ (2.2e-5 → 7.7e-4;
1.8e-6 → 8.8e-4). Only `uniform` is non-monotone.

The prediction had the large-$t$ limit backwards. With the likelihood flat, the recursion
becomes a *pure repeated convolution*, which drives the message to the chain's
**stationary law** — an infinite convolution of scaled innovations, which is as far as one
can get from "one Gaussian through one transition". Conversely at *small* $t$ the
likelihood is sharp, $L_i \cdot \ell_i$ is nearly a point mass, and the next message is
nearly $p_\varepsilon$ shifted — which is *exactly* in the family. The ansatz is best
where the data is most informative.

This also explains why the message kurtosis plateaus rather than vanishing (below): the
messages do not Gaussianise as $t\to\infty$, they converge to the stationary law, which
for these chains is not Gaussian.

#### What this hands E23: the mode budget

Standardised message cumulants at site 31 (median $|\kappa|$ over chains):

| family | $t$ | $|\kappa_3|$ | $|\kappa_4|$ | $|\kappa_5|$ | $|\kappa_6|$ |
|---|---|---|---|---|---|
| gaussian (control) | any | 3e-16 | 5e-16 | 2e-15 | 5e-15 |
| laplace | 0.05 | 9.5e-3 | 2.12 | 7.3e-4 | **17.8** |
| laplace | 3.00 | 1.3e-2 | 0.48 | 1.8e-2 | **1.04** |
| student | 0.05 | 3.1e-3 | 3.39 | 0.72 | **228** |
| student | 3.00 | 2.2e-2 | 0.77 | 0.20 | **12.8** |
| uniform | 0.05 | 1.2e-2 | 0.82 | 1.6e-3 | 3.89 |

Three consequences, all of them design constraints rather than observations:

1. **Odd cumulants are negligible** ($10^{-3}$–$10^{-2}$) — the innovations are symmetric
   and the messages inherit it. A truncation can carry even orders only, halving the
   budget.
2. **Order 4 is not enough.** $|\kappa_6|$ is between 1 and 228, never small. A
   Gaussian-plus-$\kappa_4$ truncation — the natural first rung, and the one
   `e16_edgeworth_oracle` already builds — will miss a large term. E23 must go to order 6
   at least, and heavy-tailed families will want more.
3. **$\kappa_4$ decays with $t$ but plateaus** (2.12 → 0.48 for Laplace), for the reason
   above: the message tends to the stationary law, not to a Gaussian. So the mode budget
   does *not* shrink to nothing at high noise, and a $t$-adaptive truncation saves less
   than one might hope.

### Q5 — Fourier BP

> *"I wonder if we even need a grid in real space. Because maybe we can do everything in
> Fourier. … Instead of having a grid, we have a truncation on the number of Fourier
> modes. … Let's say the first 20 Fourier modes explain 99% of this space, we discard the
> rest."*

and on why it matters:

> *"The bottleneck of this whole thing, honestly, is the grid. The grid updates of
> messages, it's a pain in the ***. I wish we didn't have to do this."*

**Status:** **ANSWERED — and the answer is no.** Not the main build; the build is
cancelled, and the reason is worth a paragraph of the paper.

E23 measured the premise before committing to a solver: *how many numbers does a BP
message actually need, per representation?* On the Laplace chain, 144 cells per
representation (4 $\rho$ × 12 noise levels × 3 sites), KL of the reconstructed message
against the true one, degrees of freedom swept:

| representation | dof at KL $10^{-2}$ | dof at KL $10^{-4}$ | cells reaching $10^{-4}$ | non-monotone |
|---|---|---|---|---|
| grid (uniform trapezoid) | 45 | 151 | **144/144** | **0/144** |
| Fourier, shared domain | 101 | 151 | 140/144 | 92/144 |
| Fourier, domain adapted per message | **33** | **47** | 53/144 | 10/144 |
| cumulant (Gram–Charlier) | 5 | — | 24/144 | **144/144** |

Three findings, in order of how much they matter:

1. **Jérôme's intuition is quantitatively right where it converges.** Where both reach
   KL $10^{-4}$, the adapted Fourier basis beats the grid in **53/53 cells**, median ratio
   **4.28×** (47 dof against 151). 47 dof is ≈ 23 modes — close to his *"the first 20
   Fourier modes explain 99%"*.
2. **It is nevertheless unusable as a solver basis**, because it reaches KL $10^{-4}$ in
   only **53/144 cells** and is non-monotone in the truncation in 10/144. A solver must be
   able to *choose* a truncation and know the error it is accepting. The grid is monotone
   in **144/144**, and that — not its dof count — is what makes it the safe basis.
3. **The cumulant route is dead.** Gram–Charlier is non-monotone in **144/144 cells**:
   adding cumulants makes the reconstruction worse. This is not a bug, it is the known
   fact that Edgeworth/Gram–Charlier expansions are *asymptotic, not convergent*; E22's
   $|\kappa_6| \in [1, 228]$ said the same thing from the other direction.

And even the 4.28× would not have bought 4.28×: E24's `anatomy` part showed only 23.8% of
a BP sweep at $N_g=401$ is the $N_g^2$ matmul (the rest is $O(BLN_g)$ elementwise work and
traffic over 158 MB of message arrays), so making the matmul *free* caps the speed-up at
$1/(1-0.238) = 1.3\times$. Two independent reasons, measured rather than argued.

**A by-product worth recording:** uniform trapezoid quadrature is not merely adequate, it
is the right choice. Gauss–Legendre and Clenshaw–Curtis are ~10× **worse** here, confirmed
over 144 cells at scale. For a smooth decaying integrand the Euler–Maclaurin error terms
cancel and the trapezoid rule is spectrally accurate; the Laplace kink is not
polynomial-like, which is exactly what the polynomial rules are built for.

**The original design argument, kept because the reasoning is still sound and only the
conclusion changed:**

**Why it should work.** In characteristic-function space the three BP operations are:

| operation | real space | Fourier space |
|---|---|---|
| transition $a \mapsto \rho a$ | dilation of the density | $\varphi(\omega) \mapsto \varphi(\rho\omega)$ — exact, free |
| add innovation $\varepsilon$ | convolution with $P_\varepsilon$ | multiply by $\varphi_\varepsilon(\omega)$ — pointwise |
| Gaussian likelihood | pointwise product | convolution with a narrow Gaussian |

Two of the three become trivial. The third is the hard one and is discussed under
"honest risk" below.

**The dividend beyond speed.** Truncating the *cumulant* expansion (the log of
$\varphi$) at order $m$ gives a ladder in which

- order 2 = Gaussian closure = Kalman = the linear score (so **Q1 falls out as the
  bottom rung**),
- order 4 = the first $\kappa_4$ correction (`e16_edgeworth_oracle.py` already builds the
  fourth-order Edgeworth oracle, so this rung is partly in place),
- untruncated = exact BP.

And it puts the existing decay theory to work instead of leaving it standalone: the OU
channel acts on cumulants as $\kappa_m \mapsto e^{-mt}\kappa_m$ for $m\ge 3$, so the
order-by-order decay result becomes *the statement of how many modes you need at each
noise level*.

**One precision, since it is load-bearing.** His

> *"the Gaussian closure is really the first two Fourier modes, since the Fourier
> transform of the Gaussian is a Gaussian"*

is loose as stated — a Gaussian is not two Fourier modes. The correct version is that
its **log**-characteristic function is quadratic, i.e. two cumulants. Same intuition;
this is the version that generalises to the ladder above.

**E24 has already redirected this build — read before starting E23.**

The plan of record was to build FFT acceleration first as the "guaranteed win", on the
reasoning that it replaces the dense $O(N_g^2)$ matmul with $O(N_g \log N_g)$. E24's
`anatomy` part says that reasoning was wrong at the size we actually run:

> at $N_g = 401$ the matmul is **23.8 %** of the sweep. The other 76 % is $O(B L N_g)$
> elementwise work and memory traffic over 158 MB of $(B, L, N_g)$ message arrays.

So an FFT that made the matmul **free** would buy at most $1/(1 - 0.238) \approx 1.3\times$
at the operating point. That is not worth a week, and it would have been a week spent
before anyone noticed.

What does attack the other 76 % is Jérôme's *actual* proposal — **truncation**, not
acceleration. Carrying $K \approx 20$ modes instead of 401 grid points shrinks the message
arrays by ~20× and the operator by far more; it reduces the term that dominates *and* the
term the FFT would have addressed. **E23 therefore builds mode/cumulant truncation first,
and FFT acceleration only if a large-$N_g$ regime turns out to matter** (the matmul share
does reach 55 % at $N_g = 1601$, so the FFT is the right tool for the refinement studies —
E04, E11 — and the wrong one for the headline).

This is the single most useful thing E24 produced, and it came from a two-day experiment
that was only supposed to answer a cost question.

**Honest risk that remains.** Truncation can fail at small $t$, where a sharp likelihood is
*broad* in $\omega$ — messages stop being mode-sparse exactly where resolution is needed.
E22's message cumulants $\kappa_3\ldots\kappa_6$ versus $t$ are the direct measurement of
how many modes are actually needed at each level, which is why E22 runs before E23 rather
than beside it.

A second, separate difficulty worth stating now: the dilation $\varphi(\rho\omega)$ is
exact as a *function*, but on a uniform $\omega$-grid it requires resampling, so some
interpolation error re-enters. The cumulant representation avoids this entirely
($\kappa_m \mapsto \rho^m\kappa_m$), which is an argument for making the cumulant ladder
the primary object and the FFT the accelerator.

**Correctness oracles.** Two exact ones, which is why this is safe to build: the
analytic Gaussian chain (`bp_gaussian.gaussian_chain_bp`), and grid BP at $N_g = 801$.

---

### Q6 — A neural network for the kernel

> *"An intermediate model between the neural network and BP would be to actually fit $K$
> with a neural network."* … *"Exploit the structure by learning only one small neural
> network that will learn a transition kernel. And that's the only thing that it's doing,
> instead of a huge neural network, like an attention, that needs to understand that —
> because when you learn the kernel, you've exploited the structure. The structure is the
> fact that everything depends on one kernel."*

and on the training signal:

> *"The nice thing is that you actually don't need noise anymore … the loss only depends
> on the training data. You no longer have to noise anything."*

**Status:** OPEN. **Two attempts void, for two unrelated reasons; a third is running
(jobs 671442–671445).** This is Jérôme's headline suggestion and it is the one question
still without an answer, so the failures are recorded in full rather than summarised.

**Attempt 1 (jobs 669323–669326, 25–26 Sep): VOID — degenerate training set (N-23).**
`exp_34` drew its training set as
`prior.sample(rng_for("exp34-train", seed, n_chains), N_SITES) for _ in range(n_chains)`,
with the generator re-created *inside* the comprehension. Every training set was therefore
one chain replicated $M$ times (measured: 1 unique row out of 8). ~30 CPU-hours. **The
numbers flagged themselves before the cause was known**: both arms flat in $M$, em_bp worse
than this project's own LMMSE floor of 0.168, the checkpoint pinned at the first rung, and
63/64 cells censored. Fixed in exp_32/33/34, and `tests/test_no_degenerate_bundles.py` now
fails on the pattern anywhere in `experiments/`, `src/` or `tools/` — it found a fourth
instance (exp_33) that a grep had missed, and on 27 Sep it caught a *fifth*, in E27's own
`mc_floor` arm, before anything ran.

**Attempt 2 (jobs 669832–669835, 26 Sep): the em_bp arm is good; the MDN arm never
trained (N-24, N-25).** Three defects, all in the shared EM machinery rather than in
`exp_34`:

1. `fit_em` declared convergence on **one** below-tolerance increment. The measured
   increment sequence was `0.130, 0.032, 0.012, 0.018, 0.007, 0.016, 0.0076, 0.0062, 0.000`
   — plainly still climbing — and it stopped at iteration 10 of 400. Fixed: convergence now
   requires `patience = 3` *consecutive* below-tolerance non-negative increments.
2. `MDNKernel.m_step` mutated `self.net.params` in place and returned `self`. So the four
   saved "checkpoints" were four references to **one** object (the mixture kernel gave 5/5
   distinct). Validation checkpoint selection was a no-op, which disabled the exact
   overfitting control the experiment exists to test. Fixed: `m_step` returns a new kernel.
3. The stall was permanent because the learning rate was never reduced on a rejected step.
   Fixed: total line-search rejection now returns a kernel with `lr * 0.5`.

After all three: 121 iterations instead of 10, 6/6 distinct checkpoints, monotone evidence.
The landed attempt-2 data shows what was at stake — em_bp has a proper learning curve
(test risk 0.103 / 0.068 / 0.034 / 0.025 at $M = 32/128/512/2048$), while the MDN arm sat
at 0.19–0.51, **flat and noisy in $M$, with `checkpoint = 5` and `converged = True` in
essentially every cell**. Reported naively that is *"the neural kernel is 10× worse"*. It is
not a model result at all; the arm barely trained.

**Predeclared reading of attempt 3, written 27 Sep before the data lands.** The two arms do
*not* get matched iteration budgets: `em_checkpoints` tops out at 1200 and
`mdn_checkpoints` at 400, and in attempt 2 em_bp was still `censored` (never converged) in
several cells at 1200 while the fixed MDN converges by ~121. So em_bp gets up to 3× the EM
iterations. That asymmetry favours **em_bp**, which is the conservative direction for the
claim "a small neural kernel helps": if the MDN still wins, it wins against a
better-resourced baseline; if em_bp wins, the result is confounded with budget and must be
reported as *"em_bp at least matches, and had not finished improving"* rather than as a
verdict on capacity. A matched-budget rerun is the follow-up either way. Also noted: the
`at_cap` column means "the best checkpoint is the last one saved", which after early
convergence is not the same as "the iteration cap bound the result" — read it together with
`converged` and `stop_reason`, which are both in the CSV.

**The original assessment, which still stands:**

**What already exists in this repository.** `MDNKernel` (`src/markov_score/kernels.py:653`)
is exactly this object: a small MLP whose heads parameterise a per-parent-state Gaussian
mixture, placed *inside* the exact BP recursion and fitted by the same Fisher-identity
generalised M-step the mixture kernel uses. It is not given the linear-autoregressive
form — its mean and scale are arbitrary functions of the parent state, not $\rho a$. Its
docstring already contains the argument Jérôme made independently:

> the network is evaluated and backpropagated *only at the $M$ grid points*, once per EM
> iteration, no matter how large the dataset is. Data enters exclusively through $\Xi$.
> A vanilla score network, by contrast, must backpropagate through every training example
> at every step.

`experiments/exp_34_information_ladder.py` fits and scores it on the frozen protocol as
the missing rung of an information ladder. **It has never been run** — there is no output
directory for it anywhere in the tree.

**So the first action here is to run what we already have**, not to write something new.

**The failure mode he predicted, unprompted:**

> *"The real issue is going to be message passing with this, because then it will overfit.
> So the problem that you already observed with a few centroids is going to get worse."*

That is the experimental question, and it is why E23 comes first: a kernel flexible
enough to be interesting produces messages the grid may not resolve. Controls: held-out
evidence stopping (E20's protocol), a smoothness penalty on $\log K_\theta$, and the
mixture-$C$ ladder run beside it so the reader sees where flexibility stops paying.

**Two design points not settled in the call.**
- *Normalisation.* `MDNKernel.log_transition_matrix` column-normalises under the same
  trapezoidal rule BP uses, making the fitted object a proper density *of the discretised
  model*. That is the right choice and it should be stated, because it ties the learned
  kernel to the grid — and therefore to E23.
- *No-noise training.* His claim that the loss needs no noising is right for learning $K$
  at $t=0$ from clean sequences. But the headline comparison is against denoisers trained
  on noised data, so the arms must be matched on *information given*, not on convenience.
  E25 reports both: $K_\theta$ fitted from clean sequences, and fitted through the channel
  from one noisy view, which is the interface the networks get.

---

### Q7 — Inference cost

> *"If I give you a noisy sequence and I want the posterior mean, how long does it take for
> a neural network and how long does it take for BP?"* … *"This is a big question to me."*

**Status:** **ANSWERED on both devices** (E24 CPU; E24gpu2, job 670404, on an H200 NVL).
**The answer given in the call was wrong, and in the unhelpful direction — and our own
first correction of it was also wrong.**

**The one-line answer.** At the frozen operating point exact grid BP costs **~15× more per
posterior mean than the transformer, on the CPU *and* on the H200**: 0.3196 ms against
0.0254 ms on CPU (12.6×), 0.0400 ms against 0.0028 ms on the GPU (14.3×). An earlier note
in this project recorded that *"BP is slower at inference was a CPU artefact"*. **That was
wrong.** What is a CPU artefact is the *absolute* BP number, not the ratio.

**What the GPU did change: the mechanism, and the large-batch behaviour.** On the H200,
`bp_grid`'s wall clock is **flat at 0.0204–0.0215 s from batch 32 to batch 2048** — a 64×
range for a 5.6% change. It is launch-latency-bound, and inherently so: the $L$-site
recursion is serial, so a sweep is $2(L-1) = 62$ sequential kernel launches however wide
the batch. The transformer is flat only to batch 512 (0.00134–0.00140 s) and then becomes
compute-bound (0.0020 at 1024, 0.0035 at 2048). So the per-sequence ratio narrows:

| batch | BP ms/seq (H200) | transformer ms/seq | ratio |
|---|---|---|---|
| 32 | 0.637 | 0.042 | 15.2× |
| 512 | 0.041 | 0.0027 | 14.8× |
| 1024 | 0.021 | 0.0020 | 10.3× |
| 2048 | 0.0105 | 0.0017 | **6.1×** |

BP's per-sequence cost keeps falling as $1/B$ while the transformer's has bottomed out, so
the curves cross: extrapolating the flat 0.0215 s against the transformer's ~0.0017 ms/seq
gives a **crossover near batch ≈ 12 600**. The honest framing is therefore not "BP is
slower" or "BP is faster" but: *BP pays a fixed serial latency and amortises it over the
batch; the network pays per example and saturates.* Which wins is a batch-size question
with a crossover we can quote.

**The original CPU table, kept for the record:**

E24, at the frozen operating point ($\rho = 0.85$, $L = 32$, $N_g = 401$, $A = 8$,
$B = 512$ sequences, 4 threads on the M1 laptop), milliseconds per posterior mean per
sequence, essentially flat across the twelve levels:

| arm | ms / sequence | parameters | vs exact BP |
|---|---|---|---|
| mlp (256×256) | 0.002 | 1.1e5 | 0.006× |
| window (r=4, w=128) | 0.018 | 3.5e4 | 0.05× |
| **bp_gaussian** (closed form) | **0.036** | 2 | 0.11× |
| tcn (c=64) | 0.057 | 1.1e5 | 0.17× |
| bigru (h=64) | 0.058 | 5.1e4 | 0.18× |
| unet (c=32) | 0.077 | 6.4e4 | 0.23× |
| transformer (d=64, h=4, l=4) | 0.151 | 2.0e5 | 0.46× |
| **bp_grid** (exact) | **0.330** | — | 1× |

Exact grid BP is **2.2× slower than the transformer** and **18× slower than the window
head**. The call's *"I think it should take less time"* does not survive measurement.

**Why, and this is the part that matters.** E24's `anatomy` part separates the
quadratic-in-$N_g$ matmul from everything else:

| $N_g$ | full | matmul | matmul share | matmul GFLOP/s | message arrays |
|---|---|---|---|---|---|
| 101 | 0.034 s | 0.003 s | 8.7 % | 219 | 40 MB |
| 201 | 0.069 s | 0.010 s | 14.4 % | 259 | 79 MB |
| **401** | **0.169 s** | **0.040 s** | **23.8 %** | **253** | **158 MB** |
| 801 | 0.434 s | 0.155 s | 35.7 % | 263 | 315 MB |
| 1601 | 1.336 s | 0.739 s | 55.3 % | 220 | 630 MB |

Three readings, in order of importance:

1. **BP is not implemented badly.** Its matmul sustains ~250 GFLOP/s in float64, which is
   essentially BLAS peak on this machine. There is no easy constant factor to recover.
2. **It is simply doing more work, in a more expensive number format.** A transformer
   carries a site's state as a $d = 64$ vector; BP carries it as a 401-point function.
   Per site the dominant operator is $d^2 = 4{,}096$ against $N_g^2 = 160{,}801$ — a 39×
   ratio, and BP is float64 where the network is float32. Comparable total work,
   comparable time. **The cost is the price of the representation, not of the algorithm.**
3. **At the operating point BP is not even FLOP-bound.** Only 24 % of the time is the
   $N_g^2$ matmul; the other 76 % is $O(B L N_g)$ elementwise work and traffic over 158 MB
   of message arrays. This is why the measured grid-size exponent is 1.3–1.5 and not 2
   (§5.1).

**This changes E23.** See Q5.

**What is still missing:** the GPU row (`hpc/icml_submit.sh e24-gpu`), and the same table
at equal *error* rather than at equal configuration — a network that is faster and worse
has not won anything. The error numbers exist in E03/E14; joining them is a write-up step,
not a new run.

---

### Q7b — Training cost, which is where the argument actually lives

Not a question Jérôme asked, but the one his Q7 turns into once Q7 is answered badly for
us. The claim the project is making — *one small kernel instead of one large score
network* — was never an inference claim. At inference BP loses. At training the structure
of the cost is completely different, and that difference is measurable.

**Predeclared prediction** (fixed in `e24_inference_cost.part_training` before running):
an EM iteration splits into an E-step (exact BP over the data) that is **linear in $M$**,
and an M-step (the kernel's own update) that is **independent of $M$**, because the kernel
is evaluated only on the $N_g$ grid points and the data reaches it only through the
accumulated statistics $\Xi$.

**Measured** (E24 `training`, `outputs/E24/training.csv`, seconds per EM iteration):

| $M$ | E-step | M-step (mixture) | M-step (MDN) | M-step share, MDN |
|---|---|---|---|---|
| 32 | 0.037 | 1.035 | 0.086 | 70 % |
| 128 | 0.068 | 1.046 | 0.099 | 60 % |
| 512 | 0.231 | 1.065 | 0.094 | 27 % |
| 2048 | 0.858 | 1.068 | 0.073 | 8 % |

| quantity | slope in $M$ (full / top half) | predicted |
|---|---|---|
| E-step | +0.77 / +0.95 (mixture), +0.77 / +0.87 (MDN) | +1 ✓ |
| **M-step** | **+0.008 / +0.002 (mixture), −0.037 / −0.180 (MDN)** | **0 ✓** |

**The M-step cost is flat to three decimal places across a 64× range of dataset sizes.**
The mechanism `MDNKernel`'s docstring asserts is real and now measured: *a score network
pays for its parameters on every example at every step; EM–BP pays for its parameters once
per iteration, on the grid, whatever the dataset.*

**But the headline number is not a blowout, and should not be sold as one.** At $M = 2048$:

| arm | training cost at its protocol budget |
|---|---|
| window head | 101 s (32k steps) |
| EM–BP, MDN kernel | 368 s (400 EM iterations) |
| EM–BP, mixture kernel | 771 s (400) / 2316 s (1200) |
| transformer | 1011 s (32k steps) |

EM–BP is *comparable* to the networks in absolute training cost, not orders below. What is
categorically different is how the cost responds to scaling **the model**: for EM–BP only
the (data-independent) M-step term grows, while for a score network model size multiplies
against data × steps. The MDN kernel at 0.07–0.10 s per M-step is ~11× cheaper to update
than the mixture kernel's closed-form step, and is 8 % of the iteration at $M = 2048$.

**The measurement this argument still needs** (deferred to E25, where it belongs): sweep
the *kernel capacity* and confirm the M-step stays flat in $M$ as the kernel grows, against
a score network whose per-step cost grows with its parameter count. That is the "small
network vs huge network" claim stated as an experiment rather than as an intuition.

---

### Q8 — Generation without a score

> *"If we have the exact score, or say the exact BP — so not even a score — I think we
> don't even need [it] to generate. We can just start from a flat marginal and run BP
> once."*

**Status:** **WEAK FORM ANSWERED (E26, landed 26 Sep). STRONG FORM built, registered and
predeclared as E27; not yet run.** The distinction was recorded in E26's registry entry
before its results landed, so it did not get lost.

**Weak form — does a BP score drive generation? Yes, and the contrast is sharp.**
E26 drove the same reverse SDE (400 steps, identical Brownian path per arm, reference
sample = the clean chain through the channel at $t_{\min}$) with four scores:

| arm | excess kurtosis of the AR residual | reference | $\mathrm{KL}$ of the residual histogram |
|---|---|---|---|
| exact (true-kernel BP) | 1.66–1.78 | 1.87 | 6.2e-4 |
| em_bp (fitted mixture) | 1.18 | 1.87 | 2.6e-3 |
| lmmse (covariance twin) | **−0.01** | 1.87 | 2.2e-2 |
| isotropic | −0.03 | 1.87 | 3.8e-1 |

The load-bearing row is **lmmse: excess kurtosis −0.01 where the truth is 1.87.** The
covariance-matched Gaussian score generates a *Gaussian*. It gets the covariance right
(cov_fro 0.069 against exact's 0.071) and destroys every higher-order structure in the
law. This is the cleanest statement in the project of why the "learned scores are
basically a low-pass filter of a Gaussian" framing is not the whole story — the part it
throws away is exactly the part that makes the samples the right shape.

**Strong form — is the reverse SDE needed at all? E27 (`experiments/e27_direct_sampler.py`,
registered 2026-09-27, predeclared, smoke-tested only).** On a chain, sum-product returns
the exact conditional laws, so forward-filtering–backward-sampling draws exactly from the
chain law in **one sweep**: no score, no integrator, no discretisation in $t$. Four arms,
all emitting samples at $t_{\min}$ against E26's own reference stream so the two
experiments are paired:

- `ancestral_true` — $a_1\sim\mu$, $a_{i+1}\sim K(\cdot|a_i)$ on the grid. The
  flat-marginal limit, where one forward pass *is* the generator.
- `ffbs_true` — the faithful reading: draw $x$ at $t_{\max}=3$, run **one** batched BP
  forward filter conditioned on it, sample backward.
- `ancestral_em` — the same pass with the fitted kernel. **This is the arm that matters:**
  it separates kernel error from integrator error, which nothing in this project has done.
  E26's em_bp figure (1.18 against 1.87) confounds the two.
- `mc_floor` — i.i.d. draws from the true law in continuous space, so every other arm's
  excess over it is the part that is not Monte Carlo. E26 had no such arm, which is why its
  `exact` deficit (1.66 against 1.87) could not be read as bias.

Correctness is pinned against the project's own exact solver, not against a second
implementation of the same idea (`tests/test_ffbs_sampler.py`, 8 tests): the FFBS forward
filter equals $\mathrm{normalise}(\text{grid\_bp forward}\times\ell_i)$ to **1.4e-15**, the
last site's filter equals the last site's BP belief to **8.9e-16**, and 20 000 FFBS
*draws* of one fixed observation reproduce `grid_bp`'s posterior means within four Monte
Carlo standard errors at every site, with variance matching $\mathrm{Var}_{BP} + h^2/12$.

**Cost, from the smoke run** ($N_g=201$, $B=64$, so indicative only): `ancestral_true`
0.071 ms/sequence, `ffbs_true` (batched) against `reverse_sde_100` at 13.7 ms/sequence.
The predeclared claim is a speed-up of order $n_{\text{steps}}$, and part `steps` asks the
harder converse — how many reverse-SDE steps buy the accuracy one exact sweep gives for
free, i.e. cost *at equal error* rather than cost at a fixed step count.

**What this does to the framing, if P1–P3 hold.** The diffusion apparatus in this project
is not needed to *sample*; it is needed to *learn the kernel from noisy observations*.
Once the kernel is known, sampling on a chain is one sweep. That is a statement about what
the paper is about, and it is Jérôme's, not ours.

---

### Q9 — Fixed $t$ or averaged?

> *"When you do this comparison between neural networks and the EM–BP, you compare the
> score at some time, or you average over time?"* — *"You average over time."*

**Status:** PARTIAL. The answer given was correct (the estimand is schedule-pooled; see
`experiments/protocol.py`, `estimand`). But a pooled number can hide a crossover, and
the natural question after his is *"does the ranking hold at every $t$?"*. E24 and E25
report per-level curves beside the pooled scalar.

---

### Q10 — What the information curves measure

> *"I don't think you explicitly can quantify information on beta, because to resolve beta,
> technically you need all the modes, all the cumulants. So I think here it is more … the
> information of the second cumulant and of the fourth cumulant."*

**Status:** RESOLVED in the call, in his favour. Implementation pending.

He is right. The curves are the information about $\kappa_2$ and $\kappa_4$, not about
$\rho$ and $\beta$. This is a change in `e01_information_decay.py` and its figure, not
only an axis label, and it should propagate to every place the paper says "shape
information".

---

### Q11 — Scaling in $M$ and $L$

See §5.1 — the answer given in the call was wrong and needs correcting in writing.

---

## 3. The experiments

| id | question | what it does | cost |
|---|---|---|---|
| **E24** | Q7, Q9, Q11 | wall-clock per posterior mean per arm; parameter counts; scaling in $M$, $L$, $N_g$; per-level curves | 2 d |
| **E22** | Q2, Q3, Q4 | dump grid messages, fit Gaussian / Gaussian $\ast P_\varepsilon$ / Gaussian+Laplace; message cumulants $\kappa_3..\kappa_6$ vs $t$ and site | 2 d |
| **E23** | Q5, Q1 | Fourier / cumulant BP: hybrid FFT and cumulant ladder, against the two exact oracles; truncation-order convergence; timing | 1–1.5 wk |
| **E25** | Q6 | run `exp_34`'s `MDNKernel` rung; then the clean-vs-noised interface contrast and the overfitting controls | 2 wk |
| **E26** | Q8 | BP denoiser in the reverse SDE; one-shot flat-marginal sampler | 3 d |
| — | Q1, Q4 | lemma in `docs/THEORY_NOTES.md` | 2 d |

Order: E24 and E22 first (cheap, and they answer him fastest), then E23, which is what
makes E25 safe, then E26.

---

## 4. What he asked for that is not a question

Recorded here so it is not lost, though the current phase is research, not writing.

- **Figures first.** *"Personally I like to do figures first. If you could do a selection
  of like 5, 6 figures that you think are really the ones that show what you want to
  show."* Plus a summary Figure 1: *"a mix of four different figures … really shows
  everything you want to show, to do a one paragraph summary of what you did."* Both his
  ICML papers are built this way (`icml-jerome/`).
- **Carve the signal out.** *"There's a bit of entropy as to, you try a lot of different
  things … it's important to isolate what results are actually important."*
- **Do not over-sell the information-decay result.** *"I wouldn't emphasize this too much
  per se … it's a very generic thing. You Gaussianize something, so the more non-Gaussian
  it is, the more information [is destroyed]."*
- **Drop the clean-input network arm** (BestNet-B). *"It's going to [wreck] it massively.
  … What it needs to learn is to just subtract the two. … Let's ignore it."* Agreed: the
  arm is degenerate.
- **Notation.** The kernel is invisible where it acts (eqs. 4.1–4.2); $\psi$ in the BP
  section becomes $K_\theta(a'|a)$ in §6 with no stated transition. Unify.
- **Nomenclature.** Move away from "innovation" / AR(1) econometrics naming.
- **The message is not "EM–BP is the best architecture."** It is: *"the structure that we
  are ignoring with other models can be very useful, and we should encode it"* —
  architecturally and in the modelling.
- **Bring Marc in.**

---

## 5. Corrections to the call

### 5.1 The scaling answer was wrong

Asked how EM–BP scales, the call answered: *"I think it depends in a squared way from how
many sequences and in a linear way from $L$."* Jérôme then built his cost question on that.

Reading `bp_grid.grid_bp_batch`, this is not what the code does. The recursion is a
`(N_g, N_g) @ (N_g, B)` matmul per site, over $L$ sites, for a batch of $B$ independent
chains. So the cost is

$$O(L \cdot N_g^2 \cdot B)$$

— **linear** in the number of sequences $B$, linear in $L$, and **quadratic in the grid
size $N_g$**. The chains are independent; nothing couples them.

The likely origin of the error is that the number of sequences and the number of grid
points are both called $M$ in different parts of the project (`n, m = log_ell.shape` uses
`m` for the grid). The quadratic is in the *grid*, not in the *data*.

**Measured (E24 `scaling`, laptop CPU, exponents fitted over the full range and over the
largest half):**

| factor | range | slope (full) | slope (top half) | predicted |
|---|---|---|---|---|
| sequences $B$ | 32 → 2048 | +1.04 | +1.19 | +1 ✓ |
| sites $L$ | 8 → 256 | +1.12 | +1.13 | +1 ✓ |
| grid $N_g$ | 101 → 1601 | +1.27 | +1.47 | +2 ✗ |

So the call's answer is corrected on both counts: the dependence on the number of
sequences is **linear, not quadratic**, and the dependence on $L$ is linear as predicted.

**And our own prediction was wrong too**, which is worth stating plainly rather than
quietly dropping: $N_g$ came out at 1.3–1.5, not 2. The `anatomy` part says why — at
these sizes the recursion is not FLOP-bound. Only 24 % of the time at $N_g = 401$ is the
$N_g^2$ matmul; the rest is $O(B L N_g)$ traffic, which dilutes the quadratic term. The
operation count is quadratic; the *wall-clock* is not, until $N_g \gtrsim 1600$ where the
matmul share passes 55 %.

The correction that survives: the bottleneck Jérôme identified by instinct is real, but it
is the **representation size**, not specifically the matmul — which is why truncating the
representation beats accelerating the matmul. See Q5.

### 5.2 "The best network" needs a name

Table rows labelled "BestNet-C" hide which architecture won, and the call had to ask.
Every reported arm should name its architecture, and the per-architecture table should be
one lookup away.

---

## Revision log

- **2026-09-25** — created from the call transcript. Eleven questions logged, all OPEN
  except Q10 (resolved in call) and Q11 (§5.1 correction pending measurement). Found
  `MDNKernel` already implemented and `exp_34` written but never run.
- **2026-09-25, later** — E24 run on the laptop (CPU). **Q7 answered and the call's guess
  refuted**: exact grid BP is 2.2× *slower* than the transformer per posterior mean, not
  faster. **Q11 answered**: linear in the number of sequences, linear in $L$. Our own
  predeclared $N_g$ exponent of 2.0 was *also* falsified (measured 1.3–1.5); the
  `anatomy` part, added in response, shows the recursion is memory-bound rather than
  FLOP-bound at the operating point — only 24 % of the sweep is the $N_g^2$ matmul.
  **This redirected E23**: mode/cumulant truncation first, FFT acceleration only for the
  large-$N_g$ refinement studies, because an FFT alone caps out at ~1.3×.
- **2026-09-26** — first HPC results. **E24c-capacity confirms the headline mechanism**: the
  M-step exponent in $M$ stays ~0 at *every* kernel capacity (MDN hidden 16→256, mixture
  $C$ 1→32), and 136× more kernel parameters costs 1.8× more per M-step against
  6.3×-per-step for 175× more transformer parameters. Honestly flagged: at $M = 8192$ the
  E-step (4.1 s) dwarfs the MDN M-step (0.13 s), so EM–BP's *total* training cost is
  comparable to a transformer's — the claim is about parameters, not totals.
  **All four E25 jobs void (N-23)**, and a GPU job wrote a CPU-only cost table (root cause:
  `libnvrtc.so.12` missing from the default library path, so the first `gpu_available()`
  probe failed and the second succeeded after torch loaded its own bundled copy; fixed by
  `hpc/icml_gpu_env.sh` and a fail-closed `E24_REQUIRE_GPU` guard, tested by exit code).
- **2026-09-27** — **Q2/Q3/Q4 confirmed at scale** (E22c, 432/432 cells; the
  grid-refinement control gives $\mathrm{KL}(1601)/\mathrm{KL}(801) = 1.000$ over 1512
  cells, so the failure is the model and not the quadrature). **Q5 answered and the Fourier
  solver cancelled**: an adapted Fourier basis does beat the grid 4.28× in 53/53 cells where
  both converge — close to Jérôme's "first 20 modes" — but reaches KL $10^{-4}$ in only
  53/144 cells and is non-monotone in 10; Gram–Charlier is non-monotone in 144/144. **Q7
  answered on the H200 and our own earlier correction corrected**: the ~15× gap is *not* a
  CPU artefact; BP is launch-latency-bound on GPU (flat over a 64× batch range) so the gap
  narrows to 6.1× at batch 2048 with a crossover near batch ≈ 12 600. **Q8's weak form
  answered** (E26: the LMMSE score generates excess kurtosis −0.01 where the truth is 1.87)
  and **the strong form built** as E27, predeclared, with correctness pinned against
  `grid_bp` to 1e-15 and 20 000 draws checked against BP's own marginals. **Q1's
  identification confirmed to 6.7e-16 — and the lemma drafted for it found FALSE as
  stated**, with a computed counterexample; no new lemma is needed, one sentence in
  Proposition 1's remark closes it. E25 resubmitted a third time (671442–671445).
  Deploy fixed: `git archive` was shipping 1.5 GB of *fetched cluster results* back to the
  cluster, which killed the scp twice and is why attempt 2's resubmission silently never
  happened.
