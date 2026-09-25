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
| Q1 | Is Gaussian-message BP provably the same as using the Gaussian (linear) score? Is the route through the Kalman filter? | Lemma + E23 | OPEN |
| Q2 | With a Laplace *ansatz* for the messages and a Laplace innovation, is BP exact? | E22 | OPEN |
| Q3 | What do the messages $\mu_k$ actually look like? Is it Gaussian $\ast\ P_\varepsilon$? | E22 | OPEN |
| Q4 | Is closure possible only for the Gaussian, or for any finitely-parameterised family? | Lemma + E22 | OPEN |
| Q5 | Can BP be done in Fourier space — truncation on modes instead of a real-space grid? | E23 | OPEN — **but E24 has already changed the design; see below** |
| Q6 | Can $K$ be a neural network, fitted by EM through BP? Does it overfit? | E25 | OPEN |
| Q7 | Posterior mean for one noisy sequence: how long for a network, how long for BP? | E24 | **ANSWERED (laptop CPU)** — BP is *slower*, §2 Q7 |
| Q8 | Can we generate without a score — flat marginal, run BP once? | E26 | OPEN |
| Q9 | Is the score compared at a fixed $t$ or averaged over $t$? | E24 | PARTIAL (answered in call; to be reported both ways) |
| Q10 | Do the information curves quantify information about $\rho$ and $\beta$, or about $\kappa_2$ and $\kappa_4$? | E01 relabel | RESOLVED in call — implementation pending |
| Q11 | How does EM–BP scale in $M$ (sequences) and $L$ (sites)? | E24 | **ANSWERED** — the call's answer was wrong, and so was ours; §5.1 |

---

## 2. Questions

### Q1 — Gaussian closure vs the linear score

> *"Is there a proof in a sense, or...?"* … *"But maybe the route is through the Kalman
> filter, actually."*

**Status:** OPEN (as a stated lemma; the fact is already used throughout).

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

### Q2 — A Laplace *ansatz* for the messages

> *"What if you take a Laplace ansatz for the messages in the BP? … And then is this
> exact when the innovation is exact? To me, it's not immediate that they match."*

**Status:** OPEN.

**What we believe, and why it is probably "no".** It is almost certainly *not* exact, and
his own instinct in the call was right. Two message operations act on $\mu_k$:
multiplication by the Gaussian likelihood, and convolution with $P_\varepsilon$. Even
with a Laplace innovation, multiplying a Laplace by a Gaussian does not give a Laplace,
and convolving a Laplace with a Laplace does not give a Laplace. So the family is not
closed under either operation, and any fixed finite parameterisation carries
misspecification error — which is the point the call landed on (*"then there will be
some misspecification"*).

The honest version of his intuition is the one he reached himself:

> *"Where you're right is that the good Gaussian BP iteration — to really update just the
> variance and the mean — this you can only do for the Gaussian. I think this is true."*

**What closes it.** E22 fits the best 4-parameter Gaussian+Laplace ansatz to the true
grid messages and reports the residual against the grid reference, for a Laplace chain,
across $t$ and site position. If the residual is at or below the grid's own quadrature
error the ansatz is viable; otherwise it is a closure approximation with a measurable
price, and we report the price.

---

### Q3 — What do the messages actually look like?

> *"If you look at the messages, $\mu_K$ … what do they look like? … I wonder if it's not
> Gaussian plus — you see what I mean? Like if you say $\mu$ of $a$ is equal to sort of
> Gaussian [convolved with] the family of $\varepsilon$, so $P_\varepsilon$ — is this exact?"*

**Status:** OPEN. This is the cheapest question on the list and it has never been looked at.

**What closes it.** E22 dumps the grid messages and plots them. Three candidate fits,
scored by KL and Hellinger against the grid reference: (i) Gaussian, (ii) Gaussian
$\ast\ P_\varepsilon$ — his conjecture, (iii) 4-parameter Gaussian+Laplace. Reported
alongside the message cumulants $\kappa_3\ldots\kappa_6$ as functions of $t$ and site,
which is the representation-independent version of the same question and feeds directly
into E23's truncation order.

Note his conjecture has a plausible mechanism: the forward message at site $k$ is a
convolution of the (Gaussian-tilted) previous message with $P_\varepsilon$, so a
"Gaussian $\ast\ P_\varepsilon$" form is what one step produces from a Gaussian start.
The question is whether it is a *fixed point* — whether the form survives iteration. We
expect it does not exactly, and E22 measures the drift.

---

### Q4 — Is closure Gaussian-only?

> *"So you think closure is only possible in the Gaussian case?"* … *"I think it's possible
> for any family that is characterised by a finite amount of parameters."*

**Status:** OPEN.

**What we believe.** Two different claims got mixed in the call and need separating:

1. *Exact* closure — the family is preserved by both message operations, so BP in the
   parameters is exact. This holds for the Gaussian and essentially only for it
   (stable-family exceptions aside, none of which have finite variance).
2. *Projected* closure — represent $\mu_k$ in a finite family and project after each
   update. This is available for **any** family, always, and is what he meant by
   *"you provide a family and then you fit it in this way; this should be doable always."*
   It is approximate, and its error is a measurable quantity, not a yes/no.

The distinction matters because it decides what Fourier BP buys: not exactness in a new
family, but a *controlled* truncation whose order is a dial. See Q5.

**What closes it.** The lemma of Q1 for part 1; E22's measured projection error for
part 2.

---

### Q5 — Fourier BP

> *"I wonder if we even need a grid in real space. Because maybe we can do everything in
> Fourier. … Instead of having a grid, we have a truncation on the number of Fourier
> modes. … Let's say the first 20 Fourier modes explain 99% of this space, we discard the
> rest."*

and on why it matters:

> *"The bottleneck of this whole thing, honestly, is the grid. The grid updates of
> messages, it's a pain in the ***. I wish we didn't have to do this."*

**Status:** OPEN. This is the main build.

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

**Status:** OPEN — **but far less open than the call suggested.**

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

**Status:** ANSWERED on CPU (E24, `outputs/E24/cost.csv`). GPU row pending.
**The answer given in the call was wrong, and in the unhelpful direction.**

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

### Q8 — Generation without a score

> *"If we have the exact score, or say the exact BP — so not even a score — I think we
> don't even need [it] to generate. We can just start from a flat marginal and run BP
> once."*

**Status:** OPEN.

**What we believe.** He is right in principle, and it is a sharper statement than it
first sounds: for a tree-structured posterior, sampling can be done by a single
forward–backward sweep followed by an ancestral pass, with no reverse SDE and no score at
all. The diffusion machinery is then only needed when the factorisation is *not*
available. That inverts the usual framing and is worth an experiment.

**What closes it.** E26: (a) BP as the denoiser inside the reverse SDE, against the
score-network denoiser on the same Brownian path (`e05_generation.py` already has this
scaffold); (b) the one-shot forward–backward + ancestral sampler, scored on the same
sample metrics. The comparison of interest is cost and sample quality, jointly.

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
