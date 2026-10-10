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
| Q2 | With a Laplace *ansatz* for the messages and a Laplace innovation, is BP exact? | E22, E22c | **ANSWERED — no.** Confirmed at scale: the interior failure in 432/432 cells, site 1 at the quadrature floor in 416/432 (corrected 9 Oct; it read 432/432 for both) |
| Q3 | What do the messages $\mu_k$ actually look like? Is it Gaussian $\ast\ P_\varepsilon$? | E22, E22c | **ANSWERED — his conjecture is very nearly right** |
| Q4 | Is closure possible only for the Gaussian, or for any finitely-parameterised family? | Lemma + E22 | **ANSWERED empirically** |
| Q5 | Can BP be done in Fourier space — truncation on modes instead of a real-space grid? | E23 + `docs/FOURIER_ANALYSIS.md` | **ANSWERED analytically.** Closed form for the message, decay $\exp(-(\log b\omega)^2/\log\frac1\rho)$ at $R^2=0.99997$; the tilt is banded and hardest at LOW noise; cumulants provably divergent (Cramér fails). Solver still not worth building, now for a mechanism |
| Q6 | Can $K$ be a neural network, fitted by EM through BP? Does it overfit? | E25, E25m | **ANSWERED — no, twice over.** A neural kernel loses to a mixture kernel AND to a comparable score net. The 1,516-parameter MDN kernel loses to the 25-parameter mixture kernel in **64/64 cells at matched budget**, median 2.04–2.57×, and E30 puts all arms on one estimand: em_bp $<$ best net $<$ mdn, §2 Q6 |
| Q7 | Posterior mean for one noisy sequence: how long for a network, how long for BP? | E24 (`hpc_E24x`, `hpc_E24x_gpu`), E47 part (c) | **ANSWERED on BOTH devices** — BP is 12.5× *slower* than the transformer on CPU and 15.1× on the H200 at 8 threads (exclusive nodes); not a CPU artefact. With one thread on a 101-point grid it is faster than the transformer (corrected 9 Oct; it read ~15×, closed by E24 and the shared-node run E24gpu2), §2 Q7 |
| Q8 | Can we generate without a score — flat marginal, run BP once? | E26 (weak), **E27** (strong) | **ANSWERED, both forms — yes.** One BP sweep generates at the Monte Carlo floor and is 731× cheaper than the reverse SDE at equal error (exclusive node, 8 threads, B = 2048; corrected 9 Oct, it read 725× from the shared-node run), §2 Q8 |
| Q9 | Is the score compared at a fixed $t$ or averaged over $t$? | **E28** | **ANSWERED.** Pooled, and the pooled scalar *does* hide a crossover: em_bp first at the 8 (of 12) lowest levels at $M=128$ and the 10 lowest at $M=2048$, the LMMSE **oracle** above. The crossover **moves with data**, $t\approx0.94\to1.90$ for a 16× increase, §2 Q9 |
| Q10 | Do the information curves quantify information about $\rho$ and $\beta$, or about $\kappa_2$ and $\kappa_4$? | `tools/cumulant_information.py` | **DONE — and a RECOMPUTE, not a relabel:** a relabel would have been wrong by 1.6–1526× (corrected 9 Oct; it read 2.3–1518×, the $t = 3$ column alone, 2.36–1518), §2 Q10 |
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

**Confirmed at scale (E22c, job 669837, `outputs/hpc_E22c/`, frozen 9 Oct).** The laptop
result was one configuration; E22c swept $\rho \in \{0.5, 0.85, 0.95\}$, $L \in \{32, 128\}$,
$N_g \in \{801, 1601\}$, three non-Gaussian noise families (Laplace, Student-$t$, uniform;
the Gaussian control is in `shapes_summary.csv` but not in the verdict table), twelve
levels — **432 cells**. The interior verdict holds in **432/432**: the ansatz is strictly
above the quadrature floor and growing with depth. The site-1 verdict holds in
**416/432**: the 16 exceptions are uniform noise at $\rho = 0.5$, $t \le 0.153$, where the
site-1 KL (2.4e-15 to 4.3e-15) is still machine precision in absolute terms but above that
cell's measured floor (3.1e-17 to 3.6e-16). Growth factor from site 1 to the deepest site
spans 6.7e7 to 3.3e11 (median 1.4e10). *(Corrected 9 Oct against the frozen
`closure_verdict.csv`: this paragraph read $\rho \in \{0.5, 0.7, 0.85, 0.95\}$, four
families, 432/432 for both verdicts, and a growth of 8.0e7 to 1.15e11.)*

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

*(Corrected 9 Oct: the table and the factors above are the laptop run's, 64 chains,
unfrozen. The frozen E22c at the same configuration — $\rho = 0.85$, $L = 32$,
$N_g = 801$, site 31, 512 chains — gives 2 272× (Laplace) and 16 464× (Student-$t$) over
the plain Gaussian, and 78× (Laplace) and 280× (Student-$t$) over the four-parameter
mixture; the transition ansatz's KL for Laplace at $t = 0.05$ is 1.6e-5. Same direction,
stronger.)*

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
two-parameter family *(corrected 9 Oct: that range is the laptop's single configuration;
across the 432 frozen E22c cells the transition ansatz's deepest-site KL runs from 4.3e-9
to 2.0e-3, median 2.3e-4)*. Whether that price is acceptable is a question about the downstream
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
2. **Order 4 is not enough.** $|\kappa_6|$ is between 1 and 228, never small *(corrected
   9 Oct: in the frozen E22c at site 31, $\rho = 0.85$, $L = 32$, $N_g = 801$ this holds for
   Laplace (1.04–17.8) and Student-$t$ (12.7–228; 18.7–342 at $N_g = 1601$) noise; uniform
   noise goes down to 0.24 on both grids, and
   across all frozen cells and sites the minimum is 0.0064)*. A
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

**Status:** **ANSWERED, analytically as well as empirically.** The *solver* is not worth
building; the *analysis* was, and it is in **`docs/FOURIER_ANALYSIS.md`** (three results, 11
tests, reproduced by `tools/fourier_analysis.py`). Per the user's direction of 27 Sep — explore
Fourier analytically first — E23's negative measurement became the input to a theory question
rather than the end of one.

**The three analytical results, in one place:**

1. **The message has a closed form in Fourier.** Two of BP's three operations are free: the
   transition is a dilation $\omega\mapsto\rho\omega$, the innovation a multiplication by
   $\varphi_\varepsilon$. Unrolling gives
   $\varphi_{k+1}(\omega) = \varphi_\mu(\rho^k\omega)\prod_{j<k}\varphi_\varepsilon(\rho^j\omega)$,
   **verified against our own `grid_bp` to 2.16e-8** — the grid's quadrature error, not model
   error. Its $k\to\infty$ limit is an *infinite product*, the stationary law's CF, which
   **explains E22's falsified P3**: with a flat likelihood the recursion drives messages to an
   object in no finite-parameter family, so the ansatz error grows with depth rather than
   shrinking.
2. **The spectrum decays like a Gaussian in $\log\omega$**:
   $\log|\varphi_\infty(\omega)| \sim -(\log b|\omega|)^2/\log(1/\rho)$, fitted slope
   matching $-1/\log(1/\rho)$ to within 1.7% at $\rho = 0.85$ and 0.8–6.3% over
   $\rho\in[0.5,0.99]$, $R^2 = 0.99997$. So **the truncation level is analytically
   computable** — which means E23's measured non-monotonicity cannot come from the message's
   own spectrum. (The asymptotic form over-predicts $\omega_{\max}$ by 1.4–1.8× at practical
   tolerances; evaluate the product instead.)
3. **The likelihood tilt is the one hard operation, and it is hardest at LOW noise.** It is a
   convolution in mode space with a Gaussian of width $\sigma_\omega = \alpha/\sqrt{\Delta}$,
   which *decreases* with $t$: bandwidth 17.7 modes at $t = 0.05$ down to **0.29 at $t = 3$**,
   banded at 10 of 12 levels. This is the opposite of the natural guess. It answers the
   tractability question — and it does *not* revive the solver, because E24 measured that the
   spectral step is only 23.8% of a sweep.

**And the cumulant failure now has a theorem behind it, after two of my own explanations were
refuted.** The Gram–Charlier expansion converges in $L^2$ only under Cramér's condition
$\int q^2/\phi < \infty$, and **it fails for every standardised BP message at every noise
level** (the integral grows by ×2.4e193 from half-width 8 to 20 at $t = 0.05$, and still by
×4.9e43 at $t = 3$). One basis has a convergence theorem this chain satisfies; the other has
one it violates. That replaces the weaker argument we had been using ($|\kappa_6|$ runs
1–228, a symptom rather than a cause) *(corrected 9 Oct: 1–228 is the unfrozen laptop
reading for Laplace and Student-$t$ at site 31; the frozen E22c at site 31, $\rho = 0.85$,
$L = 32$, $N_g = 801$ gives Laplace 1.04–17.8, Student-$t$ 12.7–228 and uniform 0.24–3.8,
see "What this hands E23", consequence 2, above)*.

The two refuted explanations are kept in the note because they are why the third can be
trusted: the "it's the metric and the positive cone" story died when the *unweighted* $L^2$
error was found to diverge *before* any negative mass appears, and "it's my arithmetic" died
against a Gaussian control that reproduces order 30 to 2.9e-13. **The first Cramér test was
also wrong twice over** — it used the raw rather than the standardised message, and it looked
for the divergence by *refining the grid*, which reads 6114.3, 6113.3, 6113.1, 6113.0 at
$N_g = $ 1601…12801 and detects nothing. **Grid refinement tests quadrature; only widening the
domain tests tail integrability.**

**The build is still cancelled, and the earlier reasoning below still stands — but note the
one substantive change: the tilt, which we had assumed was the blocker, is not. The blocker is
that BP is memory-bound, so there is nothing for a faster spectral step to buy.**

E23 measured the premise before committing to a solver: *how many numbers does a BP
message actually need, per representation?* On the Laplace chain, 144 cells per
representation (3 $\rho \in \{0.5, 0.85, 0.95\}$ × 12 noise levels × 4 sites
$\{4, 8, 16, 31\}$), KL of the reconstructed message against the true one, degrees of
freedom swept (median over the cells that reach each KL level):

| representation | dof at KL $10^{-2}$ | dof at KL $10^{-4}$ | cells reaching $10^{-4}$ | non-monotone |
|---|---|---|---|---|
| grid (uniform trapezoid) | 45 | 151 | **144/144** | **0/144** |
| Fourier, shared domain | 101 | 151 | 140/144 | 92/144 |
| Fourier, domain adapted per message | **33** | **47** | 53/144 | 10/144 |
| cumulant (Gram–Charlier) | 5 | 5 | 24/144 | **144/144** |

*(Corrected 9 Oct against the frozen `outputs/hpc_E23/dstar.csv`: the description read
"4 $\rho$ × 12 noise levels × 3 sites", and the cumulant row's dof at KL $10^{-4}$ read
"—"; it is order 5 in every one of the 24 cells that reach $10^{-4}$. The cumulant
reaches KL $10^{-2}$ in only 108/144 cells. All four noise families, 576 cells per
representation: grid 45 / 151, shared Fourier 65 / 101, adapted Fourier 23 / 33,
cumulant 5 / 5, non-monotone 0, 186, 188 and 358 of 576.)*

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
   $|\kappa_6| \in [1, 228]$ said the same thing from the other direction *(corrected 9 Oct:
   that range is the unfrozen laptop reading; frozen E22c at site 31, $\rho = 0.85$,
   $L = 32$, $N_g = 801$ gives Laplace 1.04–17.8, Student-$t$ 12.7–228, uniform 0.24–3.8)*.

And even the 4.28× would not have bought 4.28×: E24's `anatomy` part showed only 23.8% of
a BP sweep at $N_g=401$ is the $N_g^2$ matmul (the rest is $O(BLN_g)$ elementwise work and
traffic over 158 MB of message arrays), so making the matmul *free* caps the speed-up at
$1/(1-0.238) = 1.3\times$. Two independent reasons, measured rather than argued.
*(Corrected 9 Oct: 23.8% is the unfrozen laptop run, and so are the 23.8% in result 3
above and the 55% at $N_g = 1601$ below. The frozen exclusive-node anatomy
(`outputs/hpc_E24x/`, `outputs/hpc_E24x_gpu/`) puts the matmul at 12.1% (cnode07) and
15.2% (gnode16's CPU) of a sweep at $N_g = 401$, and 29.4% / 23.9% at $N_g = 1601$, so a
free matmul buys at most 1.14–1.18×. The conclusion is the same, more strongly.)*

**A by-product worth recording:** uniform trapezoid quadrature is not merely adequate, it
is the right choice. Gauss–Legendre and Clenshaw–Curtis are 4–6× **worse** here (median
relative score error at 51, 101 and 201 nodes over the 144 frozen quadrature cells, four
noise families × 3 $\rho$ × 12 levels; over all ten node counts 2.3–20.4×: 2.4–2.5× at 31,
4.5–6.7× at 51–601 except 151, where it is 2.3–2.4×, and 20× at 801)
*(corrected 9 Oct against `outputs/hpc_E23/quadrature_summary.csv`: it read "~10× worse,
confirmed over 144 cells at scale"; ~10× was the smoke run)*. For a smooth decaying integrand the Euler–Maclaurin error terms
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

**Status: ANSWERED 28 September 2026 — and the answer is no.** Jobs 671442–671445 (MDN iteration cap 400) and the matched-budget rerun 671661–671664 (cap 1200, equal to em_bp's) both completed. **The neural kernel loses to the 25-parameter mixture kernel in 64 of 64 cells**, by a median of 2.04× at $M=32$, 2.24× at 128, 2.28× at 512 and 2.57× at 2048. It does not win a single cell at any size or seed.

**The arm did train this time.** That is the first thing to check, because attempt 2's headline number was an artefact of an arm that never moved. Validation-selected checkpoints now span 20–1200 against attempt 2's `checkpoint = 5` with `converged = True` in essentially every cell, and only 7 of 64 cells report convergence, so the arm is still improving rather than stalled. The N-24/N-25 fixes did what they were meant to.

**The budget confound, which the reading rule below required to be settled rather than argued about.** Raising the MDN cap from 400 to 1200 moved the selected checkpoint in 0/16 cells at $M=32$, 2/16 at 128, 9/16 at 512 and 14/16 at 2048, improving test risk at $M=2048$ by 23.7%. So the cap *was* binding at the largest size, and is now relaxed. The clean, **budget-free** comparison is at $M=32$ and $M=128$, where the MDN is at its cap in 0/16 cells and em_bp in 0–1/16: there em_bp wins 32/32 by 2.04× and 2.24×. At $M=2048$ the MDN is still at cap in 15/16 cells and em_bp in 10/16 even at 1200, so the 2.57× at the largest size is *not* budget-free and is reported with that caveat.

**The control that makes the contrast readable:** em_bp's test risk is identical between the two runs to $0.000\times10^{0}$ over all 64 cells, as it must be — same cap, same bundles. The only thing that changed is the MDN's budget.

**He predicted this, unprompted, in the call.** *"The real issue is going to be message passing with this, because then it will overfit; the problem you already observed with a few centroids is going to get worse."* Inside exact BP, 61× more kernel parameters buys 2–2.6× worse test risk. E29 then finds the same U-shape in the score networks: at $M=128$ the largest rung is worse than the best smaller rung in 16/16 seeds for bigru, tcn and transformer. The overfitting is a property of fitting a flexible object through message passing, not of this particular kernel.

**SHARPENED 29 September by E30, and the sharpening is against his suggestion twice over.**
The gap flagged below has been closed: E30 refits *both* kernel arms on E03's bundles, so the
MDN, em_bp and every E29 architecture rung finally sit on one estimand. The gate passes at
**0.000e+00** over 1152 cells. Measured there, pooled over 16 seeds:

| arm | params | $M=32$ | $M=128$ | $M=512$ | $M=2048$ |
|---|---|---|---|---|---|
| em_bp, mixture kernel | 25 | **0.1022** | **0.0573** | **0.0365** | **0.0239** |
| best score net, any rung | 329–4,001 | 0.1760 | 0.1080 | 0.0709 | 0.0560 |
| mdn, neural kernel | 1,516 | 0.2162 | 0.1445 | 0.0900 | 0.0695 |
| mdn / em_bp | | 2.12 | 2.52 | 2.47 | 2.91 |
| best net / mdn | | 0.81 | 0.75 | 0.79 | 0.81 |

**The ordering is em_bp $<$ best net $<$ mdn.** The neural kernel is the *worst* of the three:
the best score net is **19–25% better than it at every dataset size**, and the MDN is beaten by
13 rungs at $M=32$, 12 at $M=128$ and 9 at $M=2048$ — by bigru at every size. E25's registered
hypothesis, that the MDN is an *intermediate* model sitting between the unstructured networks
and mixture EM–BP, is **wrong**: it is below both.

So the answer to Q6 is no in a second, sharper sense. It is not merely that a neural kernel
loses to a mixture kernel. **Putting a neural network in the kernel slot loses to putting a
comparable network in the score slot.** The win is not "learn $K$ with a network rather than
learn the score with a network" — it is exact BP *with a parsimonious kernel*. Twenty-five
parameters beat every architecture by 1.72–2.34×; fifteen hundred parameters in the same slot
lose to them. That is consistent with the three-axis capacity U-shape (N-28): validation picks
$K=2$ for the mixture at $M=32$ and $128$ — about five parameters — so 1,516 is far past the
optimum for this problem, wherever it is spent.

Two controls make this readable. **Q6 replicates on different data**: em_bp beats mdn in 64/64
cells with ratios 2.12/2.52/2.47/2.91 against E25's 2.04/2.24/2.28/2.57, so the answer is not
bundle-specific. And **a fresh em_bp fit lands within 0.42% of E28's frozen E02a kernel** on the
same bundles, which was predeclared as P4 at 10% — so the two are interchangeable and every E28
and E29 ratio already reported stands. Budget-free at $M=32$ and $M=128$ (`at_cap` 0/16 and
1/16 for em_bp, 0/16 and 2/16 for mdn).

**Why this needed a new experiment rather than arithmetic** (recorded 28 September, closed by E30 the next day). E25's numbers could not be placed beside E29's, and the reason is stronger than the 27% disagreement between E25's fresh em_bp fit and E28's frozen E02a one: `exp_34` draws its data under the tags `exp34-train/val/test` and E03 under `exp31-train/val/test`. **Different tags are different random streams, so the two experiments are scored on different data**, and quoting 0.1445 beside 0.1080 would have been an arithmetic error rather than a rounding one. Resisting that was the point: the numbers are three decimal places apart and look like one table. E30 makes them one table properly, by refitting rather than by aligning.

**The three attempts are kept in full below rather than summarised, because the first two failed for unrelated reasons and both failures were legible in the numbers before the cause was known.** This is Jérôme's headline suggestion and it is the one question
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

### E29 — the refutable arm: score nets *at* the kernel's parameter count

Not one of Jérôme's eleven questions, but the experiment his framing invites and the one a
referee asks for first. The parameter-efficiency claim compares a small kernel against a
135,825-parameter transformer; nobody had run a *small* transformer. E29 is a capacity
**ladder** per architecture — six architectures, 16 seeds, four dataset sizes, 1,792 cells,
parameter counts measured rather than estimated and spanning 81 → 135,825, every architecture
bracketing 1,516. Designed to be refutable: one architecture reaching em_bp at ~1,516
parameters kills the claim. Reference arm is E28's em_bp, valid because E29 and E28 share the
test bundle exactly ($\text{sq\_ref}$ agrees on 10,752 overlapping rows at $0.000\times10^{0}$).

| $M=128$, em_bp $=0.0570$ | best rung | risk | /em_bp | | $M=2048$, em_bp $=0.0239$ | best rung | risk | /em_bp |
|---|---|---|---|---|---|---|---|---|
| bigru | 1,105 | **0.1080** | **1.89** | | bigru | 4,001 | **0.0560** | **2.34** |
| tcn | 1,193 | 0.1347 | 2.36 | | unet | 3,529 | 0.0627 | 2.62 |
| transformer | 843 | 0.1424 | 2.50 | | transformer | 5,193 | 0.0628 | 2.63 |
| unet | 933 | 0.1428 | 2.50 | | tcn | 4,433 | 0.0683 | 2.85 |
| window | 5,505 | 0.1428 | 2.50 | | window | 5,505 | 0.1175 | 4.91 |
| mlp | 3,456 | 0.3839 | 6.73 | | mlp | 26,016 | 0.2077 | 8.68 |

**P1 confirmed, and more strongly than predicted.** The prediction was that nothing at
$\le 2{,}600$ parameters would reach em_bp. In fact **nothing reaches it at any capacity on
the ladder** — the top rung, a 135,825-parameter transformer, is 2.90× em_bp's risk at
$M=128$. The best net at $\le 2{,}600$ is bigru(1,105) at 1.89×.

**P2 falsified: risk is U-shaped in capacity, not monotone.** At $M=128$ the largest rung is
worse than the best smaller rung in **16/16 seeds** for bigru, tcn and transformer and 15/16
for unet, and the per-seed argmin sits at 1,105 (bigru, 12/16 seeds), 843 (transformer, 8/16),
933 (unet, 7/16), 1,193 (tcn, 8/16). mlp and window are the exceptions at 5–6/16 — they are
weak enough to be still improving. So the best score network on this problem is a
*thousand*-parameter one, and it still loses to 25 parameters inside exact BP. This is the
same overfitting Jérôme predicted for the neural kernel (Q6), now visible in the score nets.

**P3 falsified, in the opposite direction, and the honest restatement is a rate.** The
matched-parameter gap is *smaller* at small $M$ (bigru ~1,105: 1.89× at $M=128$ against 2.77×
at $M=2048$), not larger. The sample-efficiency content survives when stated as a rate rather
than a gap: over a 16× increase in data em_bp improves 2.39× (0.0570 → 0.0239) while the best
net improves 1.93× and a fixed bigru(1,105) only 1.63×.

**P4 is not testable as stated**, because its premise fails: no net matches the kernel at any
capacity, so "the smallest parameter count at which one does" does not exist within a ladder
reaching 5,433× the mixture kernel's 25 parameters.

**E31 (29 September): the confound attacked rather than restated, and the gap survives.** The
same ladder at $M=2048$ with the step budget raised 4× to 131,072, every rung rerun rather than
the few that came closest. All four predictions confirmed. **No architecture reaches em_bp at
any capacity even at 4× the budget**; best-net/em_bp narrows from 2.34× to **2.09×** and the
U-shape persists in 16/16 seeds for five of six architectures. The `at_cap` fraction falls from
62% to 34%, so this reduces the confound rather than eliminating it.

What settles the question is *where* the extra budget helped. It helped the under-trained small
rungs and did nothing at the top: bigru(329) $-28.6\%$, unet(933) $-23.8\%$, bigru(1,105)
$-22.5\%$, window(19,201) $-21.7\%$ — against bigru(59,009) $+0.0\%$, mlp(84,768) $+0.3\%$,
transformer(135,825) $+0.3\%$. **The large rungs had already converged at 32,000 steps, so the
top of the ladder is capacity-limited, not budget-limited**, which is exactly the point at
issue. One row goes the other way and is recorded rather than dropped: unet(54,049) got
$16.8\%$ *worse* with 4× the training, i.e. at fixed capacity a longer budget overfits.

**The confound that must be quoted with it.** At $M=2048$, **278 of 448 net cells (62 %)
select the last checkpoint** — median selected step 32,000 of a 32,000 cap — so the networks
had not finished training and the $M=2048$ gap could narrow with a longer budget. At $M=128$
only 23 % are at cap and the median selected step is 4,000 of 32,000, so that comparison is
effectively budget-free. **Quote $M=128$'s 1.89× as the clean number**; the $M=2048$ figures
are reported with the caveat attached, not without it.

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

*(Corrected 9 Oct. The numbers above and the batch table below are the shared-node runs
E24c and E24gpu2, which are not citable for timing (N-29). The frozen exclusive-node reruns,
8 threads, give 0.519 ms against 0.042 ms on CPU (**12.5×**, cnode07, `outputs/hpc_E24x/`)
and 0.0405 ms against 0.0027 ms on the H200 (**15.1×**, `outputs/hpc_E24x_gpu/`); the batch
table becomes 15.6× at batch 32, 15.0× at 512, 10.5× at 1024 and 6.2× at 2048, with BP flat
at 0.0204–0.0219 s.)*

*(Added 9 Oct, the one-thread caveat: E47 part (c), `outputs/hpc_E47t/`, exclusive node
cnode01, ONE thread, B = 512, L = 32.)* BP takes 0.107 ms per sequence at $N_g = 101$ and
0.667 ms at $N_g = 401$; the transformer takes 0.208 ms, the U-Net 0.152 ms and the
bidirectional GRU 0.063 ms. The 12.5× above was measured with 8 threads, which the
networks use and BP barely does: from one thread to eight, BP at $N_g = 401$ goes from
0.667 to 0.524 ms and the transformer from 0.208 to 0.036 ms (`scaling.csv` at
$t = 0.4665$, B = 512, in `outputs/hpc_E47t/` and `outputs/hpc_E24x/`; two different exclusive
nodes). Per core, BP on the 101-point grid is about 1.9× *faster* than the transformer and
1.7× slower than the GRU; on the 401-point grid it is 3.2× slower than the transformer.
How coarse the grid may be is a property of the kernel: 101 points reproduce the true
$\varphi^4$ kernel's BP to 2e-11 (E47 part (a)), while the linear chain's fitted
8-component mixture kernels move by 2.2% at 101 points ($M = 2048$) and need 151 points to
stay within 0.08% (E47 part (d)). Unregistered comparison, inference only.

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

**What is still missing:** the GPU row (`hpc/icml_submit.sh e24-gpu`) *(corrected 9 Oct:
it exists, `outputs/hpc_E24x_gpu/`, above)*, and the same table
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
predeclared as E27; not yet run.** *(Corrected 9 Oct: E27 ran on 27 Sep and was frozen
on 28 Sep, with exclusive-node cost reruns frozen on 5 Oct; see the ledger row and the
5 Oct log entry. Its P1–P5 were adjudicated on 9 Oct in the registry: P1, P3, P4, P5
confirmed; P2 falsified as a tie: the one-sweep sampler's kurtosis is closer to the
reference than the 400-step reverse SDE's in only 18 of 32 pairs, median error 0.071
against 0.085, paired t = −1.2.)* The distinction was recorded in E26's registry
entry before its results landed, so it did not get lost.

**Weak form — does a BP score drive generation? Yes, and the contrast is sharp.**
E26 drove the same reverse SDE (400 steps, identical Brownian path per arm, reference
sample = the clean chain through the channel at $t_{\min}$) with four scores:

| arm | excess kurtosis of the residual $a_{i+1} - \rho a_i$ | reference | $\mathrm{KL}$ of the residual histogram |
|---|---|---|---|
| exact (true-kernel BP) | 1.86 | 1.92 | 7.5e-4 |
| em_bp (fitted mixture) | 1.79 (1.73 at $M = 128$, 1.79 at $M = 2048$) | 1.92 | 1.9e-3 (3.5e-3, 1.4e-3) |
| lmmse (covariance twin) | **−0.0002** | 1.92 | 2.2e-2 |
| isotropic | −0.001 | 1.92 | 3.8e-1 |

*(Corrected 9 Oct: medians over the frozen `outputs/hpc_E26/generation.csv`, 16 seeds ×
$M \in \{128, 2048\}$, 32 rows per arm; the reference is the median over the 16 seeds. The
table first printed here was seed 0 only: exact 1.66 ($M = 128$) and 1.78 ($M = 2048$),
em_bp 1.18 ($M = 128$), lmmse −0.01, isotropic −0.03, reference 1.87, KL 6.2e-4 / 2.6e-3 /
2.2e-2 / 3.8e-1. The exact arm ranges from 1.66 to 2.23 over its 32 rows.)*

The load-bearing row is **lmmse: excess kurtosis −0.0002 where the reference is 1.92**
(seed 0 read −0.01 against 1.87). The covariance-matched Gaussian score generates a
*Gaussian*. It gets the covariance right (median cov_fro 0.071 against exact's 0.071;
seed 0: 0.069 against 0.071) and destroys every higher-order structure in the law. This is the cleanest statement in the project of why the "learned scores are
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
  E26's em_bp figure (1.18 against 1.87) confounds the two. *(Corrected 9 Oct: 1.18 and
  1.87 are seed 0, $M = 128$; the 16-seed median is 1.79 (1.73 at $M = 128$) against 1.92.)*
- `mc_floor` — i.i.d. draws from the true law in continuous space, so every other arm's
  excess over it is the part that is not Monte Carlo. E26 had no such arm, which is why its
  `exact` deficit (1.66 against 1.87) could not be read as bias. *(Corrected 9 Oct: seed 0,
  $M = 128$; the 16-seed median is 1.86 against 1.92.)*

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

**Status:** The answer given was correct (the estimand is schedule-pooled; see
`experiments/protocol.py`, `estimand`). The second half — *"does the ranking hold at every
$t$?"* — is **built and submitted as E28**, not yet run at scale.

**Correcting this entry.** It previously said *"E24 and E25 report per-level curves beside the
pooled scalar."* **That was wrong.** E24's `level` column is a *timing* level and E25's
`information_ladder.csv` has no level column at all. A header scan of every frozen output
finds no per-noise-level risk for the arm comparison anywhere in the project. The nearest
thing, E14's `t_star`, turns out to be `theory.crossover_time` evaluated per $(\rho, L)$ —
**one value per configuration** — so what looks like a level effect in E14 is a $\rho$
effect. Nothing had measured this.

**What E14 does say, stated properly, because nobody had.** Reading within a configuration,
EM–BP is first in all six regimes, but its margin over the LMMSE *oracle* shrinks
monotonically with $\rho$: **14.6× at $\rho = 0.5$, 12.4× at $\rho = 0.85$, 5.6× at
$\rho = 0.95$** ($L = 128$; 11.5× / 8.7× / 4.3× at $L = 32$). The other arms reorder too —
`lmmse_true` moves from fifth place to third as $\rho$ rises.

**E28** (`experiments/e28_per_level_ranking.py`, registered, predeclared) needs no training:
the 384 E03 confirm checkpoints are loaded from the cluster, the E02a kernel from its `theta`
column, and E03's own test-bundle stream and `_sq` are reproduced exactly. E03's `_eval`
already accumulates one $(\text{sq\_err}, \text{sq\_ref})$ pair *per level* and then pools
them, so the per-level decomposition is just those pairs reported instead of summed — and
that makes the validation exact rather than approximate: **part `verify` rebuilds the pooled
risk from E28's own pairs and requires it to match E03's `confirm.csv` to 1e-6, exiting
non-zero otherwise.** Without it, plausible-looking per-level numbers could be a
decomposition of a different estimand.

**CONFIRMATORY RUN LANDED 28 September 2026 (job 672153, 16 seeds, 12 levels, 6 architectures
plus two oracles).** The gate passes: 192 cells, worst relative disagreement with E03's own
pooled risk **1.712e-15**. Risk here is E03's estimand,
$\mathrm{risk}=\sqrt{\sum \text{sq\_err}/\sum \text{sq\_ref}}$ — a relative *RMSE*, so these
ratios square to the relative-MSE form.

| | $t=0.05$ | 0.105 | 0.222 | 0.467 | 0.677 | 0.982 | 1.425 | 3.0 |
|---|---|---|---|---|---|---|---|---|
| **$M=128$** em_bp | **0.0950** | **0.0582** | **0.0353** | **0.0214** | **0.0175** | 0.0147 | 0.0117 | 0.0030 |
| lmmse (oracle) | 0.2780 | 0.1770 | 0.0950 | 0.0408 | 0.0233 | **0.0107** | **0.0032** | **0.0000** |
| best network | 0.1895 | 0.1276 | 0.0904 | 0.0713 | 0.0565 | 0.0476 | 0.0424 | 0.0411 |
| lmmse/em_bp | 2.93 | 3.04 | 2.69 | 1.91 | 1.33 | 0.72 | 0.27 | 0.00 |
| **$M=2048$** em_bp | **0.0431** | **0.0221** | **0.0119** | **0.0062** | **0.0048** | **0.0038** | **0.0029** | 0.0007 |
| lmmse (oracle) | 0.2780 | 0.1770 | 0.0950 | 0.0408 | 0.0233 | 0.0107 | 0.0032 | **0.0000** |
| lmmse/em_bp | 6.45 | 8.00 | 8.00 | 6.56 | 4.82 | 2.81 | 1.10 | 0.02 |

**P1 — falsified, as predeclared it would be.** em_bp is first at the 8 lowest levels at
$M=128$ and the 10 lowest at $M=2048$; the LMMSE *oracle* takes the rest. The prediction was
left unedited after the smoke precisely so this would be a falsification and not a hindsight.

**P2 — confirmed above the peak, falsified below it, and the nuance is the interesting part.**
lmmse/em_bp is *not* monotone in $t$: it **rises** from 2.93 at $t=0.05$ to 3.04 at $t=0.105$
($M=128$), and from 6.45 to 8.32 at $t=0.153$ ($M=2048$), before falling away. So over the
bottom two or three levels the LMMSE oracle gets relatively *worse*, and only then improves.
The "learned scores are a low-pass filter of a Gaussian" framing is right at high $t$, wrong
at low $t$, and *most* wrong not at the very bottom of the schedule but just above it.

**P3 — confirmed, in 16/16 seeds at both sizes.** The pooled scalar hides a real crossover.

**The new result, which the smoke could not see: the crossover moves with the data.** Median
$t_\times = 0.94$ (range 0.62–1.38) at $M=128$, and $1.90$ (range 1.33–2.07) at $M=2048$ — a
16× increase in training data roughly **doubles** the range of $t$ over which a fitted kernel
beats the oracle that knows the true covariance. That is the sample-efficiency claim showing
up on an axis it was not designed to be measured on. It is also distinct from
`theory.crossover_time` $= 1.1667$, which is a property of the model and does not move with $M$.

**Not one network is first at any level, at either size.** best-net/em_bp rises monotonically
with $t$: $1.99 \to 13.75$ at $M=128$ and $2.04 \to 44.24$ at $M=2048$. The per-level view was
built to find where the headline fails; it does not rescue a single architecture anywhere.

**The smoke run already shows a crossover, which would falsify our own P1.** At grid 201,
$n_\text{test} = 64$, seed 0, $n = 128$, with the real E02a fit and no network arms:

| $t$ | em_bp | lmmse (oracle) | isotropic | lmmse/em_bp |
|---|---|---|---|---|
| 0.05 | **0.0786** | 0.2673 | 0.8758 | 3.40 |
| 0.2216 | **0.0389** | 0.0879 | 0.6852 | 2.26 |
| 0.4665 | **0.0284** | 0.0373 | 0.5023 | 1.31 |
| 0.6769 | 0.0279 | **0.0188** | 0.3765 | 0.67 |
| 1.4250 | 0.0197 | **0.0033** | 0.1112 | 0.17 |
| 3.0000 | 0.0050 | **0.0000** | 0.0052 | 0.0018 |

EM–BP is first at the seven lowest levels and the LMMSE oracle is first at the five highest,
crossing between $t = 0.4665$ and $t = 0.6769$. **The pooled scalars are em_bp 0.0540 against
lmmse 0.1632 — a 3× advantage that hides the crossover completely.** That is precisely the
thing Jérôme's question was aimed at, and it is the single strongest argument for reporting
the curve and not only the scalar.

**And the reading is not that EM–BP fails.** Above the crossover there is nothing
non-Gaussian left to recover, so knowing $\rho$ *exactly* beats estimating a whole kernel:
the oracle's model error vanishes with $t$ while the fitted kernel's estimation error does
not. This is the *relaxation* of the low-pass-filter framing pointing the other way — above
the crossover the Gaussian picture is **right**, and the paper should say where the crossover
is instead of claiming uniform dominance. Note this is **not** `theory.crossover_time`, which
is 1.1667 at the frozen configuration and defined by $e^{2t}\lambda_{\max}(\Sigma_0) = 1$; the
relationship between the two deserves a line once the confirmatory numbers exist.

The table above is one seed at half the grid resolution with 64 test chains and no network
arms. It is a smoke test, it is reported as one, and the predeclared P1 has deliberately been
left unedited in `experiments/registry/E28.json`: a prediction revised after seeing data is
not a prediction.

---

### Q10 — What the information curves measure

> *"I don't think you explicitly can quantify information on beta, because to resolve beta,
> technically you need all the modes, all the cumulants. So I think here it is more … the
> information of the second cumulant and of the fourth cumulant."*

**Status:** **DONE (27 Sep), and he was more right than the call made it sound.**
`tools/cumulant_information.py`, `outputs/E01/cumulant_information.csv`,
`tests/test_cumulant_information.py` (10 tests).

He is right, and the fix is **a recomputation, not a relabel**. E01's `shape` part computes
the Fisher information about $\theta = (\rho, q, \beta)$. Going to
$\varphi = (\rho, \kappa_2, \kappa_4)$ is a change of coordinates, so it needs the delta
method. For `GeneralizedGaussianKernel`, which holds $\mathrm{Var}(\varepsilon) = q$ at every
$\beta$:

$$\kappa_2 = q, \qquad \kappa_4 = \gamma_2(\beta)\,q^2, \qquad
  \gamma_2(\beta) = \frac{\Gamma(5/\beta)\Gamma(1/\beta)}{\Gamma(3/\beta)^2} - 3,$$

so $J = \partial\varphi/\partial\theta$ is lower triangular with rows $(1,0,0)$, $(0,1,0)$,
$(0,\ 2\gamma_2 q,\ \gamma_2' q^2)$, and
$\mathrm{eff}(\varphi_k) = 1/(j_k^\top I_\theta^{-1} j_k)$. Two coordinates come out
unchanged — $\rho$ and $\kappa_2 = q$, because their rows of $J$ are unit vectors, which is
the internal check that $J$ is assembled right — and $\kappa_4$ does not, because $j_3$ mixes
in the $q$ direction.

**A relabel would have been wrong by a factor between 1.6 and 1526** (the `relabel_ratio`
of `outputs/E01/cumulant_information.csv`, median over the 8 replicates of each $\beta$ and
$t$, over every $\beta$ and every $t$ in the file; 2.2–1524 at the $t = 0.4665$ of the table
below) *(corrected 9 Oct: it read "between 2.3 and 1518", which is the $t = 3$ column
alone, 2.36–1518)*. Across $\beta$ at a fixed noise level, medians over 8 replicates:

| $\beta$ | $\gamma_2$ | $\gamma_2'$ | $\mathrm{eff}(\beta)$ | $\mathrm{eff}(\kappa_4)$ | $\mathrm{eff}(\kappa_4)/\mathrm{eff}(\kappa_2)$ |
|---|---|---|---|---|---|
| 1.00 (Laplace) | 3 | −8.50 | 0.1382 | 0.288 | 0.033 |
| 1.25 | 1.527 | −4.05 | 0.0374 | 0.379 | 0.045 |
| 1.50 | 0.762 | −2.30 | 0.0140 | 0.443 | 0.050 |
| 2.00 (Gaussian) | 0 | −1.00 | 0.00284 | 0.479 | 0.057 |
| 3.00 | −0.582 | −0.333 | 0.000343 | 0.525 | 0.061 |

*(at $t = 0.4665$; $q = \kappa_2 = 0.2775$.)*

**The qualitative reading of the $\beta$ curve inverts.** $\mathrm{eff}(\beta)$ falls by
**403×** from $\beta = 1$ to $\beta = 3$ (48.5× at $t = 0.05$, 440× at $t = 1.425$), which
reads naturally as *"shape information collapses as the law approaches the Gaussian."* It
does not. $\mathrm{eff}(\kappa_4)$ **rises**, by 1.83×. The collapse is the coordinate going
degenerate: $|\gamma_2'|$ shrinks from 8.50 to 0.333 over that range, a factor 25.5, and it
enters $\mathrm{eff}(\kappa_4)$ **squared** — $25.5^2 = 650$, which accounts for the
observed 403–440×. Near $\beta = 2$ a change in $\beta$ barely moves any cumulant, so
$\beta$ is nearly unidentifiable *however much data you have*. That is exactly Jérôme's
objection, and it is now a measured factor rather than a conceptual worry.

**The decay-in-$t$ claim SURVIVES, and is stronger in the honest coordinates.** This was the
thing at risk, so it is the thing to report. Over $t$ from 0.05 to 4.0 on the Laplace chain
($\beta = 1$):

| quantity | decay over the full $t$ range |
|---|---|
| $\mathrm{eff}(\rho)$ | $9.1\times10^5$ |
| $\mathrm{eff}(\kappa_2)$ | $3.6\times10^6$ |
| $\mathrm{eff}(\kappa_4)$ | $5.9\times10^8$ |

$\kappa_4$'s information decays **163× faster than $\kappa_2$'s** — and the asymmetry
widens with $\beta$ (1.0e5× at $\beta=1.5$, 1.6e6× at $\beta=3$). The
parameterisation-free statement of the project's central claim is therefore the ratio
$\mathrm{eff}(\kappa_4)/\mathrm{eff}(\kappa_2)$, which on the Laplace chain runs
**0.071 → 0.033 → 0.0048** at $t = 0.05,\ 0.47,\ 1.43$: shape carries 7% of scale's
information at low noise and 0.5% by $t = 1.43$. *"The channel removes information about
shape faster than about correlation"* is not an artefact of using $\beta$; it is 163× in
cumulant coordinates, and it should be stated that way from now on.

**What is verified, not assumed.** $\gamma_2'$ in closed form against a central difference at
$\beta \in \{0.8, 1, 1.25, 1.5, 2, 3, 5\}$; the reparameterisation inverts to 1e-9; and
$I_\theta$ is reconstructed from the stored `diag_*`/`corr_*` and then **checked against the
stored `eff_*`** before anything is computed from it — worst relative error **8.6e-13** over
560 rows, and the tool exits non-zero if that check fails, because rebuilding a matrix from
summary statistics is exactly the step that silently produces plausible numbers when the
convention has changed.

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

*(Corrected 9 Oct: the table and the shares above are the laptop's exploratory, unfrozen
run. The frozen exclusive-node run, `outputs/hpc_E24x/slopes.csv` (cnode07, 8 threads),
gives slopes (full / top half) of +1.03 / +1.00 in $B$, of +1.03 / +1.04 in $L$
and of +1.16 / +1.22 in $N_g$; its anatomy puts the matmul at 12.1% of the sweep at
$N_g = 401$ and 29.4% at 1601, so the 55% is not reached by 1601 on that node. Same
verdicts: linear in $B$ and $L$, well below 2 in $N_g$. With one thread, E47 part (c)
gives +1.20 over $N_g = 41$–401.)*

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
- **2026-09-28** — **the queue drained; Q6 and Q9 both answered, and the ledger is complete
  at 11 of 11.** E25 attempt 3 (671442–5) and the matched-budget rerun (671661–4): the MDN
  kernel **loses to the 25-parameter mixture kernel in 64/64 cells**, median 2.04–2.57×, and
  the arm genuinely trained this time (checkpoints 20–1200, not pinned at 5). The budget
  confound the reading rule demanded be *settled* rather than argued: raising the MDN cap
  400→1200 moved the checkpoint in 0/16 cells at $M=32$ and 14/16 at $M=2048$, so the clean
  budget-free comparison is $M=32$ and $M=128$ — and em_bp wins those 32/32. **Jérôme
  predicted this failure mode unprompted in the call.** E28 (672153): gate passes at
  1.712e-15; P1 falsified, P2 confirmed only above a peak at $t\approx0.1$–$0.15$, P3
  confirmed in 16/16 seeds; **the crossover moves with data**, $t\approx0.94\to1.90$ over a
  16× increase. E29: **no architecture reaches em_bp at any capacity up to 135,825
  parameters** — but 62 % of its $M=2048$ cells are at the training cap, so the clean number
  is $M=128$'s 1.89×. Three of my predictions falsified (E28 P1 as expected, E29 P2 and P3
  not). Two defects found and fixed: **N-27**, six E29 jobs sharing one output directory so
  five architectures' merged rows were clobbered (no data lost — rebuilt from 1,792 surviving
  per-cell files), and a latent second instance in `e24-gpu`/`e24c-gpu` that the new guard
  `tests/test_submit_cases_disjoint_outputs.py` found on its first run. **N-26** written up
  properly at last: the E28 `n_test` error *and* my wrong first diagnosis of it.
- **2026-09-29** — **E30 and E31 land; Q6's answer sharpens against the suggestion twice over,
  and two of my own errors cost 9.6 CPU-hours and a timing claim.** E30 put the MDN, em_bp and
  every E29 rung on one estimand for the first time (gate: 1152 cells, worst relative `sq_ref`
  disagreement **0.000e+00**). **P1 and P3 falsified**: the ordering is em_bp $<$ best net $<$
  mdn — the neural kernel is the *worst* of the three, beaten by the best score net by 19–25%
  at every size and by bigru at every size. E25's "intermediate model" hypothesis is wrong. So
  the win is exact BP *with a parsimonious kernel*, not neural parameterisation of the kernel:
  a network in the kernel slot loses to a comparable network in the score slot. **P2 and P4
  confirmed**, and P4 matters for what is already published — a fresh em_bp fit lands within
  **0.42%** of E28's frozen E02a kernel, so every E28 and E29 ratio stands. E31 confirmed all
  four of its predictions: no architecture reaches em_bp at 4× the budget, the gap narrows only
  2.34×→2.09×, and the extra budget helped the small rungs while the large ones had already
  converged — the top of the ladder is capacity-limited, not budget-limited. **N-29**: I queued
  `e27-cost-big` onto the same node as ten fitting jobs, which inflated the reverse-SDE cost
  ratio from 1,138× to 2,392×; contention flatters us, so neither cost-big run is citable and
  `MSD_HPC_EXCLUSIVE=1` now exists. **N-30**: E30's gate compared against locally fetched
  outputs that are export-ignored from the deploy archive, so all four jobs completed 128/128
  cells and then exited 2 on a comparison impossible on the cluster; it is a post-fetch step
  and is no longer run inside the job — but it is still not allowed to pass when its references
  are missing.

- **2026-10-05** — N-29 resolved for the two cost answers: the exclusive-node reruns landed (sacct AllocCPUS = the whole node). **Q7** stands: BP / transformer = 12.5x on CPU (two different nodes) and 15.1x on the H200 (hpc_E24x, hpc_E24x_gpu). **Q8** stands: the ancestral sampler is 731x cheaper than the reverse SDE at equal error (200 steps, B = 2048, 8 BLAS threads; was 725x) and 1465x cheaper than the 400-step SDE (hpc_E27x); single-threaded the 400-step ratio is 2343x, so every cost citation carries its thread count. From the consolidated note: the domain-length distribution (section 24.5) is E37; the global-latent repair and the subsampling protocol at sixteen seeds are E35x and E36x (submitted).

- **2026-10-06 (evening)** — the consolidated note's section 29 and appendix F closed. **Notation** (§29 step 2,
  F.5, and this ledger's §4): one table, `docs/NOTATION.md`, implemented in `paper/notation.tex` and applied to the
  paper, the field report, `FIELD_MODEL.md` and the library docstrings; "innovation" became *driving noise*, "AR(1)"
  the *linear chain*; collisions settled (γ, ξ, h, K/T, M/N_g, α); the archival thesis gets a read-only concordance.
  **Metric definitions** (§29 step 1): `docs/METRICS.md`, from the code that wrote the frozen outputs. **The Gaussian
  sanity check** (§29 step 3, F.4, §28 fig. 2): a proposition with its proof in the paper's appendix B and E39 —
  grid BP agrees with the closed form and an independent RTS smoother to 1.9e-13 at (401, 8); one clause falsified as
  written (moment-projected BP on the grid, up to 0.26: the grid truncates its widening backward messages).
  **The FFT transition** (§29 step 5, §9.2): E40 — second-order accurate and 5.8x *slower* than dense BP at
  N_g = 401 on one core; tuned, break-even near N_g = 3200; the 25 Sep anatomy's cap was right. **FFBS** (§9.7):
  `markov_score.ffbs`. **Information plots** relabelled by statistical order (F.3). **Figures first** (§28 and the
  call's summary figure): `reports/figures-first-2026-10-06.pdf`, all seven figures plus the summary, from frozen
  outputs.

- **2026-10-09 (night)** — the ledger audited against the frozen data, and the coarse grid read.
  **Three experiments found unfrozen, and frozen:** E22c, E23 and E26 had run on 26 Sep (clean at bd7ad42 and
  343ec92) and been read, but never frozen; frozen 9 Oct (the small tables; `shapes.csv` and `representation.csv`
  by hash only). **Corrections made in place, each marked "(corrected 9 Oct)":** Q2 (site 1 at the quadrature
  floor in 416/432 cells, not 432/432; three $\rho$ and three non-Gaussian noise families, not four and four);
  Q3 and Q4 (the factors were the laptop's single configuration; frozen E22c gives 2 272× and 16 464× over the
  plain Gaussian and 78× and 280× over the four-parameter mixture); $|\kappa_6|$; Q5 (the table is
  3 $\rho$ × 12 levels × 4 sites; the cumulant reaches KL $10^{-4}$ in 24/144 cells, at order 5; Gauss–Legendre
  and Clenshaw–Curtis are 4–6× worse, not ~10×; the free-matmul cap is 1.14–1.18× on the exclusive nodes, not
  1.3×); Q7 (the exclusive-node 12.5× and 15.1×, and the one-thread caveat below); Q8 (the E26 table was seed 0;
  the 16-seed medians are exact 1.86, em_bp 1.79, reference 1.92; E27's P1–P5, never adjudicated, now are:
  P2 falsified as a tie, the rest confirmed); Q10 (1.6–1526×, not 2.3–1518×); §5.1 (the
  frozen slopes). No verdict in this ledger changed.
  **E47 parts (a), (c) and (d) read** (P6–P9 confirmed, P10 falsified). (a) The coarse grid: BP with the true
  $\varphi^4$ kernel is exact to 2e-11 at 101 points (worst 1.8e-11 over the six cells, against (1601, 8));
  51 points give 1.7e-5 to 7.2e-4. (d) The linear chain's frozen headline kernels stay within 0.08% of their
  401-point risk at 151 points; at 101 the 8-component mixtures move 2.2% ($M = 2048$), because their narrowest
  components are finer than the 101-point spacing (P10). (c) One thread, exclusive node (`outputs/hpc_E47t/`):
  BP 0.107 ms per sequence at $N_g = 101$ and 0.667 ms at 401, against the transformer's 0.208, the U-Net's 0.152
  and the GRU's 0.063 ms. E24x's 12.5× was measured at 8 threads, which the networks use and BP barely does.
  **Queued behind the cluster maintenance window** (no start before 10 Oct 12:00): E47 part (b), the learned
  estimator refitted on the coarse grid (691998, 691999); E48, the MDN on clean strided data with its ladder
  extended to 16 000 (692000–692003); E49, the MDN on the out-of-class Laplace-bond chain (692004, 692007).
  **E50 registered and predeclared** (P1–P7) before any cell ran, now running on the laptop: speciation on the
  $\varphi^4$ field, i.e. when the exact posterior commits to the sign of a site, $|2P(a_i > 0 \mid x_t) - 1|$
  against $t$, against the Gaussian twin with the same covariance; and the kink energy of the lattice chain down
  to $h = 1/32$ (continuum value $2\sqrt2/3$). Jérôme's email of 30 Sep, item 10.
  **Two suggestions of the consolidated (SPDE) note DECLINED, with a measured reason.** Section 9.2's truncated
  rank-$R$ representation (cost $O(LGR)$: the messages stay on the $G$-point grid, only the transition is
  compressed) and section 23's interpolation-free $D\,G\,D$ Fourier update (multiply by $D$, convolve with the
  Gaussian $G$ by FFT, multiply by $D$; no dilation, so no resampling) both accelerate only the transition
  (matrix–vector) step. On the exclusive nodes that step is 12–15% of a BP sweep at $N_g = 401$
  (`outputs/hpc_E24x/anatomy.csv` 12.1%, `outputs/hpc_E24x_gpu/anatomy.csv` CPU 15.2%), so even a free
  transition caps either at 1.14–1.18×, about 1.15×; E40's FFT transition was 5.8× *slower* than dense BP at
  $N_g = 401$ on one core. Coarsening the grid (E47) reduces every term instead: with one thread, BP is 6.2×
  faster at $N_g = 101$ and 3.9× faster at 151 than at 401.
