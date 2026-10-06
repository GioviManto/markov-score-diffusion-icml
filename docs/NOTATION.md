# Notation — one table for the code, the paper, the reports and the thesis

Created 6 Oct 2026 for step 2 of the consolidated note's section 29 ("Harmonise notation throughout the code,
report, paper, and thesis") and its appendix F item 5 ("Harmonise transition, message, score, and noise
notation"). The call of 25 Sep asked for the same thing twice (`docs/JEROME_CALL_2026-09-25.md` §4: "the kernel is
invisible where it acts … unify"; "move away from 'innovation' / AR(1) econometrics naming").

This file is the reference. `paper/notation.tex` implements it as macros; the field report
(`reports/field-laboratory-2026-10-01.tex`), `docs/FIELD_MODEL.md` and the library docstrings follow it. Dated
reports already sent (meeting briefs, follow-ups, audits, the 1 Oct primer) are records and are not rewritten. The
thesis tree (`~/Code/Thesis/markov-score-diffusion-thesis`) is archival and read-only; section 6 below is its
concordance.

The starting point was the note's own glossary (its appendix E) and its forward–backward pseudocode (appendix B).
Its symbols are adopted wherever they do not collide with something already fixed. Where they do, the paper's
symbol wins, because the paper is the document the notation is for and its macros were already chosen to avoid
collisions; section 3 lists every case.

## 1. The canonical symbols

### Data, channel, score

| object | symbol | the note | code | remark |
|---|---|---|---|---|
| clean chain, site $i$ | $a_i$, vector $\mathbf a\in\mathbb R^L$ | $X_i$ | `A`, `a` | Biroli–Mézard convention: $a$ clean, $x$ noised |
| noised chain | $x_i$, $\mathbf x$ | $Y_i$ | `X`, `x` | |
| field value at site $i$ | $\phi_i$; standardised $a_i=\phi_i/s$ | $\phi_i$ | `phi`; `scale` $=s$ | $s$ = stationary standard deviation |
| diffusion time (noise level) | $t$ | — | `t`, `level` | |
| channel | $x=e^{-t}a+\sqrt{\Delta_t}\,z$, $\Delta_t=1-e^{-2t}$ | $Y=X+\sigma Z$ | `alpha` $=e^{-t}$, `delta` $=\Delta_t$ | decay factor $e^{-t}$; where a proof or the code abbreviates it, $\alpha=e^{-t}$ (never $\alpha_t$, never anything else) |
| additive-noise scale | $\sigma_t=\sqrt{e^{2t}-1}$ | $\sigma$ | — | $e^{t}x=a+\sigma_t z$: the note's channel after rescaling |
| noised law, clean law | $P_t$, $P_0$ | $p_\sigma$, $p_\theta$ | — | |
| (data) score | $S(\mathbf x,t)=\nabla_{\mathbf x}\log P_t(\mathbf x)$ | $s_\sigma(y)$ | `score` | Tweedie: $S_i=-(x_i-e^{-t}\langle a_i\rangle)/\Delta_t$ |
| posterior mean | $\langle a_i\rangle_{x,t}=\mathbb E[a_i\mid\mathbf x_t=\mathbf x]$ | $\mathbb E[X_i\mid y]$ | `means` | |
| parameter score | $u_\theta(\mathbf x)=\partial_\theta\log p_\theta(\mathbf x)$ | $u_\theta(y)$ | — | Fisher information $I(\theta)=\mathbb E[u_\theta u_\theta^\top]$ |
| Gaussian score | $S^{\mathrm G}(\mathbf x,t)=-\Sigma_t^{-1}\mathbf x$, $\Sigma_t=e^{-2t}\Sigma_0+\Delta_t I$ | $-(\Sigma_X+\sigma^2I)^{-1}y$ | `lmmse_*` arms | |

### The chain and its transition

| object | symbol | the note | code | remark |
|---|---|---|---|---|
| number of sites | $L$ | $L$ ($N$ in the field sections) | `n_sites`, `L` | |
| number of training sequences | $M$ | — | `n_chains`, `M` | never the grid size (the 25 Sep call's error, §5.1) |
| inference batch | $B$ | — | `batch`, `B` | |
| initial law | $\mu(a_1)$ | $\mu_\theta$ | `log_mu` | |
| stationary law | $\pi$ | $\pi$ | `log_stationary` | |
| **transition kernel** | $K(a'\mid a)$, parametric $K_\theta$ | $K_\theta(x'\mid x)$; $P(\phi'\mid\phi)$ in §12–15 | `log_transition_matrix` | the one symbol for a normalised transition, in every document |
| linear chain | $a_{i+1}=\rho a_i+\varepsilon_i$, so $K(a'\mid a)=p_\varepsilon(a'-\rho a)$ | same, "innovations" | `*AR1` priors | see the nomenclature in §2 |
| driving noise, its density, variance | $\varepsilon_i$, $p_\varepsilon$, $q$ | $\varepsilon_i$, $q(\cdot)$, $q$ | `eps`, `q`, `innovation_excess_kurtosis()` | $q=1-\rho^2$ in the paper's chains |
| mixture kernel | $K(a'\mid a)=\sum_{c=1}^{C}\pi_c\,\mathcal N(a'-\rho a;\mu_c,s_c^2)$ | $\sum_c\pi_c\mathcal N(x';\mu_c,s_c^2)$ | `MixtureInnovationKernel` | $C=1$ is the linear-Gaussian kernel |
| EM auxiliary function | $\mathcal Q(\theta\mid\theta')$ | $Q(\theta\mid\theta^{\rm old})$ | `q_value` | |
| expected transition statistic | $\Xi(a,a')=\sum_{m}\sum_i p_\theta(a_i=a,a_{i+1}=a'\mid\mathbf x^{(m)})$ | $\sum\xi_i$ | `ExpectedStatistics.xi` | $\mathcal Q=\langle\Xi,\log K_\theta\rangle$ |
| evidence | $\log p_\theta(\mathbf x)$, $\mathcal L(\theta)$ | $\log p_\theta(y)$ | `log_evidence` | |

### Belief propagation

| object | symbol | the note | code | remark |
|---|---|---|---|---|
| local likelihood | $\ell_i(a)=\ell_t(x_i\mid a)\propto\exp[-(x_i-e^{-t}a)^2/2\Delta_t]$ | $g_\sigma(y_i\mid x_i)$ | `log_ell` | $g$ is the tilt (field), so not $g$ |
| **forward message** | $\overrightarrow m_i(a)\propto\int K(a\mid a')\,\ell_{i-1}(a')\,\overrightarrow m_{i-1}(a')\,\mathrm da'$, $\overrightarrow m_1=\mu$ | $\alpha_i=\overrightarrow m_i\,\ell_i$ | `forward` | the cavity message: it excludes $\ell_i$ |
| **backward message** | $\overleftarrow m_i(a)\propto\int K(a'\mid a)\,\ell_{i+1}(a')\,\overleftarrow m_{i+1}(a')\,\mathrm da'$, $\overleftarrow m_L=1$ | $\beta_i=\overleftarrow m_i$ | `backward` | excludes $\ell_i$ |
| one-site posterior marginal | $p(a_i\mid\mathbf x)\propto\overrightarrow m_i\,\ell_i\,\overleftarrow m_i$ | $\gamma_i$ | `beliefs` | written out, not $\gamma_i$ (§3) |
| pair posterior marginal | $p(a_i{=}a,a_{i+1}{=}a'\mid\mathbf x)\propto\overrightarrow m_i(a)\ell_i(a)K(a'\mid a)\ell_{i+1}(a')\overleftarrow m_{i+1}(a')$ | $\xi_i$ | summed into `xi` | written out, not $\xi_i$ (§3) |
| grid size, points, weights | $N_g$; $u_1,\dots,u_{N_g}$ on $[-A,A]$; $w_k$ | $G$; $z_a$; $w_a$ | `n_grid`/`grid_size`, `grid`, `weights` | |
| grid spacing | $\Delta u=2A/(N_g-1)$ | — | `dx` | not $h$, which is the lattice spacing |
| Gaussian moment projection | $\Pi_{\mathrm G}$ | "Gaussian closure" | `grid_bp_projected_batch` | |

### The field

| object | symbol | the note | code | remark |
|---|---|---|---|---|
| temperature | $\Theta$ | $\Theta$ | `T` (and in cell names, `T0.3_…`) | $T$ in prose would sit next to $t$ |
| lattice spacing | $h$ | $h$ | `h` | |
| potential, continuum energy | $V(\phi)=(\phi^2-1)^2/4$, $H[\phi]$ | $V$, $H$ | `potential` | |
| lattice energy | $H_h(\phi)=\sum_i[(\phi_{i+1}-\phi_i)^2/2h+hV(\phi_i)]$ | $H_h$ | — | |
| pair energy | $W(\phi,\phi')$, parametric $W_\theta$ | $W_\theta$ | `PairEnergyKernel` | |
| transfer operator | $\mathsf T(\phi,\phi')=e^{-W(\phi,\phi')}$ | $K(\phi,\phi')$ | `log_transfer` | sans-serif $\mathsf T$, so that $K$ stays the transition |
| leading eigenpair | $\lambda_0$, $\psi_0$ | $\lambda_0$, $\psi_0$ | `FieldSpectrum` | $\pi\propto\psi_0^2$ for symmetric $\mathsf T$ |
| tilted walk | $K_g(a'\mid a)=\mathcal N(a';a,\gamma)\,g(a')/Z_g(a)$ | — | `TiltedWalkKernel` | |
| bond variance | $\gamma=\Theta h/s^2$ | — | `gamma`, `fit_gamma` | $\hat\Theta=\hat\gamma s^2/h$ |
| tilt, its normaliser | $g$, $Z_g$ | — | | $\pi_g\propto gZ_g$ |
| site energy | $f=-\tfrac12\log(g/Z_g)$ | — | `site_energy` | $W=(a'-a)^2/2\gamma+f(a)+f(a')$ |
| correlation length | $\xi$ (in sites, $\xi=1/\log(1/\rho_1)$) | $\xi$ | `corr_length_sites` | the only $\xi$ |
| second eigenvalue ratio | $\rho_1=\lambda_1/\lambda_0$ | — | `rho1` | plays $\rho$'s role |
| kink energy | $E_{\rm kink}$ | $E_{\rm kink}$ | — | |

## 2. Nomenclature

- **Driving noise, not innovation.** $\varepsilon_i$ in $a_{i+1}=\rho a_i+\varepsilon_i$ is the chain's driving noise;
  $p_\varepsilon$ its law. "Innovation" is the time-series word Jérôme flagged in the call ("this is from AR1 in the
  economic context, no?" — "it's the noise"). "Driving" is kept wherever the corruption noise of the channel is in
  the same sentence.
- **Linear chain, not AR(1).** The model is "the linear chain with Laplace driving noise", "the Gaussian linear
  chain" (the discrete Ornstein–Uhlenbeck chain). AR(1) survives only as a code identifier (`LaplaceAR1`, …).
- **Kernel names.** *Linear-Gaussian kernel* ($C=1$); *mixture kernel* (driving noise a $C$-component Gaussian
  mixture; the code's `MixtureInnovationKernel`, the arms' `em_mixture`); *MDN kernel*; *tilted walk*; *site
  energy* (`em_pair_quartic`); *spline tilt*; *Legendre tilt*.
- **Smoother, not filter.** The full Gaussian forward–backward sweep is the Kalman (Rauch–Tung–Striebel)
  *smoother*; only the forward sweep is the Kalman filter (the note's section 26.2).
- **Exact.** "Exact" is said of the tree recursion on functions, of the finite-state recursion the grid defines, or
  of a closed form. Grid BP is called the *reference*, with its quadrature and truncation stated, never "exact" on
  its own (section 26.2).
- **Crossover, not transition.** The one-dimensional field has a sharp low-temperature crossover, not a phase
  transition (section 20 and 26.2).

## 3. Collisions and how they are settled

| symbol | competing meanings | settled |
|---|---|---|
| $K$ | transition (note §2–4, paper) / transfer kernel $e^{-W}$ (note §13) / grid size (the paper's first draft) | $K$ = transition only; transfer operator $\mathsf T$; grid size $N_g$ |
| $P$ | transition (note §12–15) / laws $P_0$, $P_t$ | transition is $K$; $P$ is a law |
| $M$, $m$ | training sequences / grid size (`n, m = log_ell.shape`) | $M$ = sequences; grid size $N_g$ (`n_grid` in code) |
| $L$ | chain length / the forward message array `L` in `bp_grid` | $L$ = length; arrays renamed `fwd`, `bwd` |
| $\alpha$, $\beta$ | forward/backward messages (note) / decay $e^{-t}$ (code `alpha` in thirty modules; the paper's proofs) / the autoregression coefficient (thesis) / the generalised-Gaussian exponent $\beta$ (paper) | messages are the paper's arrows $\overrightarrow m_i$, $\overleftarrow m_i$, which collide with nothing and are exactly the code's `forward`, `backward`; $\alpha=e^{-t}$ where abbreviated; $\rho$ the autoregression coefficient; $\beta$ the generalised-Gaussian exponent |
| $\gamma$ | one-site marginal (note) / Gaussian reference measure (paper §3, appendix A) / bond variance (field, data column `fit_gamma`) / SNR $\gamma_t$ (thesis) | $\gamma$ = bond variance; the Gaussian reference measure is $\mathcal N_L$; the marginal is written $p(a_i\mid\mathbf x)$ |
| $\xi$ | pair marginal (note) / correlation length (note §20–21, field) | $\xi$ = correlation length; the pair marginal is written out |
| $h$ | lattice spacing (field) / quadrature spacing (`\gridh` in the paper's macros) | $h$ = lattice spacing; quadrature spacing $\Delta u$ |
| $\varphi$, $\phi$ | innovation density (`\innov` in the paper) / field value | $\phi$ = field; driving-noise density $p_\varepsilon$ |
| $g$ | likelihood $g_\sigma$ (note) / tilt (field) | $g$ = tilt; likelihood $\ell$ |
| $q$ | driving-noise variance (note §5.2, paper) / its density (note §15) | $q$ = variance; density $p_\varepsilon$ |
| $T$ | temperature (code, cell names) / transfer operator | prose: $\Theta$ temperature, $\mathsf T$ operator; `T` only in code |
| $\sigma$ | additive noise scale (note) / none in the paper | $\sigma_t=\sqrt{e^{2t}-1}$ when the note's form is needed |
| $s$, $b$ | field scale / mixture component scale $s_c$ / the natural parameters $(s,b)$ of the paper's §3 ($\Lambda(s,b)$) / E35's latent amplitude $b$ | $s$ = field scale; $s_c$ subscripted; §3's $(s,b)$ and E35's $b$ are local to their sections |

## 4. What the code keeps, and why

Identifiers that name frozen data are not renamed: CSV columns (`T`, `n_chains`, `fit_gamma`, `corr_length_sites`),
arm names (`em_mixture`, `lmmse_true`, …), class and method names that frozen experiment scripts import (`LaplaceAR1`,
`MixtureInnovationKernel`, `innovation_excess_kurtosis`, …) and the `alpha`/`delta` argument names of the BP functions (twelve signatures, thirty
modules). A rename would not change one number and would break the link between a frozen output and the commit that
produced it. What changed is what a reader of the code meets: the docstrings of the core modules use the symbols
above, the grid size is `n_grid` where it was `m`, and `bp_grid`'s message arrays are `fwd`/`bwd` where they were
`L`/`R`. The table in §1 is the bridge.

## 5. Where each document stands (6 Oct 2026)

| document | status |
|---|---|
| `paper/notation.tex` + `paper/body/*.tex` | harmonised (macros redefined; prose renamed) |
| `reports/field-laboratory-2026-10-01.tex` | harmonised |
| `docs/FIELD_MODEL.md` | harmonised |
| `src/markov_score/` core modules | docstrings and local names harmonised; identifiers kept (§4) |
| dated reports (briefs, follow-ups, audits, primer) | records; not rewritten |
| thesis | archival, read-only; concordance in §6 |

## 6. Thesis concordance

The thesis (`~/Code/Thesis/markov-score-diffusion-thesis`) is archival and is not edited. This is the translation,
from a read-only survey on 6 Oct 2026. `chN` is `chapters/chNN-*.tex`. Its macros live in `preamble.tex:125-168`;
its `notation.tex` is a comment block that `main.tex` never inputs, and three macros there render differently from
the preamble (`\corr`, `\ivar`, `\nseq`). The thesis has no glossary.

| object | canonical | thesis | where |
|---|---|---|---|
| clean / noised chain | $a_i$ / $x_i$ (1-based) | $a_k$ / $x_k$, 0-based in ch03, 1-based from ch06 | ch03:23-28, ch06:19-20 |
| decay, variance | $e^{-t}$, $\Delta_t$ | $e^{-t}$ ($\mu:=e^{-t}$ in ch05), $\Delta_t$; SNR $\gamma_t$; $c_t=e^{2t}-1$ ($=\sigma_t^2$) | ch05:33, ch03:355-358 |
| score | $S(\mathbf x,t)$ | $S(x,t)$; $s_t(x)$ in ch02; network $s_\phi$ | ch03:73, ch02:137 |
| autoregression coefficient | $\rho$ | $\alpha$ | ch02-07; `\corr` renders $\alpha$ |
| driving noise, variance | $\varepsilon_i$, $q$ | "innovation" $\eta_k$ (ch03, ch08) or $\varepsilon_k$ (ch05, ch06), $\sigma_\eta^2$ | ch03:37-41, ch05:24-26 |
| transition kernel | $K(a'\mid a)$, $K_\theta$ | $K(a'\mid a)$, $K_\theta$; the Laplace kernel is $M(a'\mid a)$; factor form $\psi_{\rm tr}^{(k)}$ | ch03:24, ch05:101, ch04:88 |
| messages | $\overrightarrow m_i$, $\overleftarrow m_i$ | $\overrightarrow m$, $\overleftarrow m$; also $\mu_{\to k}$, $\mu_{\leftarrow k}$ (ch04), $\nu_{i\to a}$ (generic), $f_k$, $\beta_k$ (grid, ch05) | ch04:37-40,155-167, ch05:132-137 |
| local likelihood | $\ell_i$ | $\psi_{\rm ob}^{(k)}$ (ch04), $\ell_k$ (ch05), $\phi_i$ (ch06) | ch04:91, ch05:129, ch06:94 |
| one-site / pair marginal | $p(a_i\mid\mathbf x)$ / $p(a_i,a_{i+1}\mid\mathbf x)$ | $p(a_k\mid x)$, ${\rm bel}_k$ / $b_i(a_{i-1},a_i)$ | ch04:172, ch06:90 |
| grid size, points, spacing | $N_g$, $u_k$, $\Delta u$ | $N_g$, $g_i=-A+ih$, $h$ | ch05:123-127 |
| training sequences | $M$ | $N$ (`\nseq` renders $N$) | ch02:184 |
| EM auxiliary, statistic | $\mathcal Q$, $\Xi$ | $Q$; per-component statistics $R_c,U_c,V_c,\dots$ (no $\Xi$) | ch06:50-54,104-114 |
| mixture | $C$, $\pi_c$, $\mu_c$, $s_c^2$ | $C$, $\pi_c$, $\nu_c$, $s_c^2$ | ch06:24-31 |
| moment projection | $\Pi_{\rm G}$ | $\Pi_G$ | ch05:217 |

"Innovation" occurs 61 times in the thesis chapters (ch05 19, ch06 16, ch08 14) and once in the abstract; "AR(1)" 10
times. The thesis's own collisions, for whoever revises it: $\beta$ is an inverse temperature (ch02:28), a backward
grid message (ch05:136), an off-diagonal of $J$ (ch08:267) and a latent strength (ch08:506); $\mu$ is $e^{-t}$
(ch05:33) and a message (ch04:155); $\nu$ is a message, a mixture mean and the Student-$t$ degrees of freedom; $J$
is a posterior precision, an EM Jacobian, a Fisher matrix and a test-set size; $\xi$ is the channel noise (ch03:67)
and a correlation length (ch08:311); $h$ is an information field, the grid spacing and a hidden state. Applying this
table to the thesis means writing to the archival tree, which only the author can decide.
