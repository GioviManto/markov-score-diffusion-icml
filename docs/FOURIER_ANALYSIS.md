# Q5 analytically: the BP message in Fourier

Reproduce: `python tools/fourier_analysis.py` → `outputs/fourier_analysis/fourier_analysis.csv`.
Pinned by `tests/test_fourier_analysis.py` (11 tests).

Jérôme, 25 September 2026: *"I wonder if we even need a grid in real space. Because maybe we
can do everything in Fourier. … Let's say the first 20 Fourier modes explain 99% of this space,
we discard the rest."* E23 then **measured** the premise and found something odd: an adapted
Fourier basis needs ~23 modes — close to his guess — but converges non-monotonically in 91 of
144 cells, while Gram–Charlier fails in **144/144**. That is a phenomenon to explain, not a
solver to write, which is why this note exists.

---

## T1. The message has a closed form in Fourier, and its spectrum decays like a Gaussian in log ω

One forward BP step is $L_{i+1}(a') = \int p_\varepsilon(a' - \rho a)\, g_i(a)\, da$ with
$g_i = L_i \ell_i$. Substituting $y = \rho a$ makes it a convolution with a *dilated* $g_i$, so

$$\varphi_{i+1}(\omega) = \varphi_\varepsilon(\omega)\,\psi_i(\rho\omega), \qquad \psi_i = \widehat{g_i}. \tag{A}$$

**Two of the three BP operations are free in Fourier**: the transition is the dilation
$\omega \mapsto \rho\omega$, and adding the innovation is the multiplication by
$\varphi_\varepsilon$. With a flat likelihood $\psi_i = \varphi_i$, and unrolling (A) from
$\varphi_1 = \varphi_\mu$:

$$\boxed{\;\varphi_{k+1}(\omega) = \varphi_\mu(\rho^k\omega)\prod_{j=0}^{k-1}\varphi_\varepsilon(\rho^j\omega)\;} \tag{B}$$

**Verified against this project's own `grid_bp`:** max absolute error **2.16e-8** at sites
1, 2, 4, 8, 16 and 31 — which is the grid's quadrature error at $h = 0.006$, not model error.

As $k\to\infty$, $\varphi_\mu(\rho^k\omega)\to1$, so the message tends to
$\prod_{j\ge0}\varphi_\varepsilon(\rho^j\omega)$: the stationary law's characteristic
function, **an infinite product**. This is the clean explanation of **E22's falsified P3** — we
predicted the ansatz error would be *smallest* at large $t$ and it *grows*, because with a
flat likelihood the recursion drives messages to exactly this object, which lies in no
finite-parameter family.

### The decay law

For Laplace($b$), $\varphi_\varepsilon(\omega) = (1 + b^2\omega^2)^{-1}$, so
$\log|\varphi_\infty| = -\sum_j \log(1 + b^2\rho^{2j}\omega^2)$. The terms with
$b\rho^j|\omega|\gg1$ contribute $-2\log(b\rho^j|\omega|)$ each, and there are
$J = \log(b|\omega|)/\log(1/\rho)$ of them. Summing the arithmetic progression:

$$\log|\varphi_\infty(\omega)| \;\sim\; -\,\frac{\bigl(\log b|\omega|\bigr)^2}{\log(1/\rho)}. \tag{C}$$

**A Gaussian decay in $\log\omega$** — super-polynomial, sub-exponential. Fitted over
$3 \le \omega \le 3\times10^4$:

| $\rho$ | fitted slope | $-1/\log(1/\rho)$ | ratio | $R^2$ |
|---|---|---|---|---|
| 0.50 | −1.5339 | −1.4427 | 1.063 | 0.99993 |
| 0.70 | −2.8988 | −2.8037 | 1.034 | 0.99995 |
| 0.85 | −6.2578 | −6.1531 | **1.017** | **0.99997** |
| 0.95 | −19.654 | −19.496 | 1.008 | 0.99999 |
| 0.99 | −100.55 | −99.499 | 1.011 | 0.99999 |

The ratio → 1 as $\rho\to1$, as it must: the derivation assumes many terms in the product.

**Do not use (C) numerically at practical tolerances.** Inverting it gives
$\omega_{\max}(\eta) \approx b^{-1}\exp\sqrt{\log(1/\eta)\log(1/\rho)}$, which over-predicts by
**1.4–1.8×** at $\eta = 10^{-2}\ldots10^{-6}$ (predicted 6.4/9.1/12.0 against measured
3.6/5.9/8.3), because the asymptotic constant — fitted intercept −6.57 nats — is the whole
budget when $\omega_{\max}$ is only 4–8. The leading term explains the *shape*; for a number,
evaluate the product in (B) directly, which costs nothing.

That is worth stating plainly: **for the flat-likelihood message the exact spectrum is
available in closed form, so a truncation level can be chosen analytically.** E23's measured
non-monotonicity in 10/144 cells therefore cannot come from the message's own spectrum — it
must come from the tilt or from the adapted-domain machinery. Which is T2.

---

## T2. The likelihood tilt is the one operation that does not simplify — and it is hardest at *low* noise

The Gaussian likelihood is a product in real space, hence a **convolution in mode space**:
$\ell_i(a) \propto \mathcal{N}(a;\, x_i/\alpha,\, \Delta/\alpha^2)$, so

$$\widehat{\ell_i}(\omega) \;\propto\; e^{i\omega x_i/\alpha}\, e^{-\omega^2/(2\sigma_\omega^2)},
\qquad \sigma_\omega = \frac{\alpha}{\sqrt{\Delta}} = \frac{e^{-t}}{\sqrt{1-e^{-2t}}}.$$

$\sigma_\omega$ is **decreasing in $t$**. On a domain of half-width $W$ the mode spacing is
$\pi/W$, so the bandwidth in modes is $\approx 3\sigma_\omega W/\pi$ ($W = 6$):

| $t$ | 0.05 | 0.0725 | 0.105 | 0.153 | 0.222 | 0.467 | 0.982 | 2.07 | 3.00 |
|---|---|---|---|---|---|---|---|---|---|
| $\sigma_\omega$ | 3.08 | 2.53 | 2.07 | 1.67 | 1.34 | 0.81 | 0.40 | 0.13 | 0.05 |
| bandwidth (modes) | **17.7** | **14.5** | 11.8 | 9.6 | 7.7 | 4.6 | 2.3 | 0.7 | 0.3 |

**The tilt is banded at 10 of the 12 levels and effectively dense at the two lowest**, and the
hard end is **low noise** — the opposite of the natural guess that high noise is difficult. At
$t = 3$ the kernel is 0.29 modes wide, i.e. a delta: the tilt is essentially free. Measured
kernel decay at $t = 0.05$: 1.000, 0.986, 0.944, 0.878, 0.697, 0.398, 0.125, 0.003 at offsets
0, 1, 2, 3, 5, 8, 12, 20 modes.

**What this does and does not change.** It answers the question — the tilt *is* tractable, with
bandwidth $\alpha/\sqrt{\Delta}$, narrowing with $t$, which is the opposite of what we would
have guessed. It does **not** revive the solver: E24's `anatomy` part measured that only 23.8%
of a BP sweep at $N_g = 401$ is the $N_g^2$ matmul (the rest is $O(BLN_g)$ elementwise work and
traffic over 158 MB of message arrays), so making the spectral step free caps the speed-up at
$1/(1-0.238) = 1.3\times$. Two independent measured reasons, and this note adds the
qualification rather than the reversal.

---

## T3. Why cumulants diverge where Fourier converges — after two of my own answers were wrong

The Hermite / Gram–Charlier expansion converges in $L^2$ iff **Cramér's condition** holds:
$\int q(z)^2/\phi(z)\,dz < \infty$ for $q$ the density being expanded, $\phi$ the standard
normal. Getting to this took refuting two explanations, and both refutations are the reason
the conclusion can be trusted.

**Refuted #1 — "it is the metric and the positive cone."** A truncated Gram–Charlier series is
not a density, so the story went, and KL punishes the negative lobes. But the **unweighted
$L^2$** error of the truncation diverges too, and it starts diverging *before* any negative
mass appears: at $t = 0.2216$ the order-6 truncation has **zero** negative mass and the $L^2$
error has already risen from 2.17e-2 to 7.58e-2. At $t = 0.05$ the $L^2$ error runs
0.125 → 0.045 → 0.194 → 0.496 → 1.99 → 8.94 → … → **7.9e6** at order 30.

**Refuted #2 — "it is my arithmetic."** $c_n = \mathbb{E}[He_n(Z)]/n!$ is a
catastrophic-cancellation setup: at $n=30$ the summands total 5.6e15 in absolute value for a
result of order 0.06. So a control with a known answer: for a **Gaussian** message every $c_n$
with $n\ge3$ must vanish and the reconstruction must be exact at every order. Measured
$\max_{n\ge3}|c_n| = $ 1.9e-18 and order-30 $L^2$ error **2.9e-13**, on three different grids.
The machinery is correct; $1/n!$ crushes the cancellation noise.

**And the first Cramér test was itself wrong, in two ways worth recording.**

1. **Wrong variable.** It evaluated $\int p^2/\phi$ on the **raw** message, whose variance is
   0.33–0.99. A narrow density against the standard normal integrates fine, so the condition
   looked satisfied (1.03–1.49). But the expansion is applied to the **standardised** message,
   and stretching a narrow density to unit variance is exactly what breaks integrability.
2. **Wrong refinement.** It looked for the divergence by **refining the grid**, which cannot
   see it. For a standardised Laplace on a fixed domain the integral reads 6114.3, 6113.3,
   6113.1, 6113.0 at $N_g$ = 1601, 3201, 6401, 12801 — flat. **Grid refinement tests
   quadrature; only widening the domain tests tail integrability.** Widened, the same integral
   reads 6.1e3, 9.4e15, 1.7e35, 2.9e61, 4.7e94 at half-width 8, 12, 16, 20, 24.

Corrected, with the controls in place ($\int q^2/\phi = 1.0000000000$ exactly for a
standardised Gaussian):

| $t$ | msg var | $A=8$ | $A=12$ | $A=16$ | $A=20$ | verdict |
|---|---|---|---|---|---|---|
| 0.05 | 0.331 | 2.3e+23 | 5.5e+66 | 1.7e+131 | 5.6e+216 | **DIVERGES** (×2.4e193) |
| 0.222 | 0.441 | 2.7e+13 | 4.9e+43 | 6.3e+89 | 5.1e+151 | **DIVERGES** (×1.9e138) |
| 0.982 | 0.751 | 60.5 | 1.9e+15 | 1.6e+38 | 2.8e+70 | **DIVERGES** (×4.6e68) |
| 3.00 | 0.992 | 1.09 | 4.9e+06 | 4.7e+21 | 5.3e+43 | **DIVERGES** (×4.9e43) |

**Cramér's condition fails for every standardised BP message, at every noise level.** The
Gram–Charlier expansion is provably outside its $L^2$ convergence class on this chain: adding
cumulants is not guaranteed to help, and does not. A Fourier basis is complete in $L^2$ for any
finite-variance density. **That is the structural reason for E23's 144/144 against 10/144** —
one basis has a convergence theorem the Laplace chain satisfies and the other has one it
violates — and it replaces the weaker argument we had been using (that $|\kappa_6|$ runs
1–228, which is a symptom rather than a cause).

It also explains the $t$-dependence: the divergence is worst at low $t$ (×2.4e193) and mildest
at high $t$ (×4.9e43), because a narrow message is stretched more by standardisation. It never
becomes finite, because the underlying law is Laplace-tailed at every level.

---

## What this means for the solver question

| | status | evidence |
|---|---|---|
| Transition and innovation in Fourier | **free**, exactly | T1 (A), closed form verified to 2.2e-8 |
| Message spectrum | **closed form**, decay $\exp(-(\log b\omega)^2/\log\tfrac1\rho)$ | T1 (B), (C); $R^2 = 0.99997$ |
| Truncation level | **analytically computable** for the flat-likelihood message | T1 |
| Likelihood tilt | **banded**, width $\alpha/\sqrt\Delta$; hardest at LOW noise | T2 |
| Cumulant / Hermite basis | **provably divergent** here | T3, Cramér fails at every level |
| A Fourier BP solver | **still not worth building** | E24: the spectral step is 23.8% of a sweep, so ≤1.3× |

The negative conclusion from E23 stands, and it now stands on a mechanism rather than on four
separate empirical observations. The one substantive change is that the tilt — which we had
assumed was the blocker — is *not* the blocker; the blocker is that BP is memory-bound, so
there is nothing for a faster spectral step to buy.

## Owed

- One sentence in `paper/sections/prop-lmmse.tex`'s remark naming the Kalman smoother (Q1).
  Deferred: the user's instruction on 27 Sep was results first, no paper work.
