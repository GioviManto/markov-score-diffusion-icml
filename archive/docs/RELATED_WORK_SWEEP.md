# Related-work sweep (ICML submission) — 15 September 2026

**Scope.** Literature relevant to: exact diffusion scores for a non-Gaussian first-order Markov chain under coordinatewise Ornstein–Uhlenbeck (OU) noise (tree posterior ⇒ sum-product exact); Gaussian message closure = LMMSE of the covariance-matched twin; EM via Fisher's identity from noised sequences only; the central result that under the OU channel every Hermite mode of degree m is attenuated by e^{-mt}, so Fisher information about order-m structure decays as e^{-2mt} (pairwise correlation e^{-4t}, non-Gaussian shape e^{-8t}) and the covariance-twin score gap is ∝ e^{-8t}; sample-complexity consequences; comparisons with neural denoisers; misspecification; length generalisation.

**Verification protocol.** Every entry below was checked this session against its arXiv abstract page, publisher page, or proceedings page (WebSearch + WebFetch). Title/authors/ID/year are as displayed there. Venue marked with † = arXiv ID verified but venue taken from memory, not re-verified in this sweep. Nothing after 2024 was taken from memory. Entries are numbered per theme; cross-references use `T<theme>.<n>`.

**Headline for the authors (read this first).** The *substance* of the e^{-2mt} statement already exists in the low-SNR estimation literature, in SNR-power form rather than OU-time form: Balanov–Bendory–Edidin (2026, T3.14) prove Fisher information restricted to directions first identifiable at moment order k scales as SNR^k; Katsevich–Bandeira (2023, T3.9) prove the log-likelihood expands in σ^{-2} with the order-k term matching k-th moments; the MRA/orbit-recovery papers (T3.10–T3.13) turn this into σ^{2k} sample complexity. Aarts et al. (2025, T3.24) already write cumulant attenuation κ_n(t)=κ_n(0)e^{-nkt/2} under an OU-type drift. Under OU, SNR_t = e^{-2t}/(1−e^{-2t}) ≈ e^{-2t}, so "SNR^k" *is* "e^{-2kt}". What is not in the literature: the Hermite-mode form for the score of a *Markov chain*, the covariance-twin score gap ∝ e^{-8t}, the exact sum-product score + EM-through-noise pipeline, and the head-to-head with neural denoisers on length generalisation. See "Closest five".

---

## Theme 1 — Exact / tractable diffusion models and solvable settings

| # | Citation | What they prove / do | How it differs from ours | ID |
|---|---|---|---|---|
| T1.1 | Biroli, Mézard. *Generative diffusion in very large dimensions.* J. Stat. Mech. 2023, 093402. | Exact high-dimensional analysis of the backward process for a Gaussian model and the Curie–Weiss model; shows symmetry breaking in reverse diffusion and that recovering the relative weight of the two states needs #data ≫ dimension. | Solvable models are i.i.d./mean-field, not a Markov chain; no Fisher-information-vs-noise-level statement; no learning from noised data. | arXiv:2306.03518 |
| T1.2 | Ghio, Dandi, Krzakala, Zdeborová. *Sampling with flows, diffusion, and autoregressive neural networks from a spin-glass perspective.* PNAS 121(27) e2311810121, 2024. | Compares generative samplers with MCMC/Langevin on spin-glass and planted models; generative models fail at first-order transitions where MCMC succeeds, and vice versa. | Sampling hardness of a *known* measure; we study estimation/learning of an unknown Markov law through noise. | arXiv:2308.14085; doi:10.1073/pnas.2311810121 |
| T1.3 | Cui, Krzakala, Vanden-Eijnden, Zdeborová. *Analysis of learning a flow-based generative model from limited sample complexity.* ICLR 2024. | Closed-form learned velocity field for a two-layer autoencoder on a high-dim Gaussian mixture; generated-mean error Θ(1/n), Bayes-optimal. | Mixture target, finite-n asymptotics of a specific architecture; no time-dependence of information. | arXiv:2310.03575 |
| T1.4 | Cui, Pehlevan, Lu. *A solvable model of learning generative diffusion: theory and insights.* NeurIPS 2025. | Tight asymptotics for a two-layer autoencoder diffusion model trained by online SGD on a low-dim-manifold target; mode collapse and model collapse under retraining. | Manifold/Gaussian target; ours is a discrete-in-time Markov sequence with exact score, no SGD asymptotics. | arXiv:2501.03937 |
| T1.5 | Shah, Chen, Klivans. *Learning mixtures of Gaussians using the DDPM objective.* NeurIPS 2023. | GD on the DDPM objective provably recovers parameters of spherical Gaussian mixtures (two components from random init; K components with warm start). | Parameter recovery for GMMs from *clean* data; we recover a Markov transition law from *noised* data. | arXiv:2307.01178 |
| T1.6 | Gatmiry, Kelner, Lee. *Learning mixtures of Gaussians using diffusion models.* arXiv 2024. | Quasi-polynomial algorithm for identity-covariance GMMs via piecewise-polynomial score regression. | Same as T1.5. | arXiv:2404.18869 |
| T1.7 | Chen, Kontonis, Shah. *Learning general Gaussian mixtures with efficient score matching.* arXiv 2024. | Poly-time/sample learning of general GMMs without separation, via diffusion-based score matching. | Same as T1.5. | arXiv:2404.18893 |
| T1.8 | Pierret, Galerne. *Diffusion models for Gaussian distributions: exact solutions and Wasserstein errors.* arXiv 2024 (v5 2025). | Closed-form backward SDE/ODE solutions for Gaussian data; exact Wasserstein errors of numerical schemes. | Gaussian target only; our covariance-matched twin *is* this Gaussian case, used as the baseline whose gap to the true score we quantify (∝ e^{-8t}). | arXiv:2405.14250 |
| T1.9 | Raya, Ambrogioni. *Spontaneous symmetry breaking in generative diffusion models.* NeurIPS 2023; J. Stat. Mech. 2024. | Generation splits into a linear phase around a central fixed point and an attractor phase; the instability window creates diversity. | Phenomenology of sampling dynamics; no per-mode information decay. | arXiv:2305.19693 |
| T1.10 | Ambrogioni. *The statistical thermodynamics of generative diffusion models: phase transitions, symmetry breaking and critical instability.* arXiv 2023. | Equilibrium-stat-mech reading of diffusion; mean-field critical phenomena. | Same as T1.9. | arXiv:2310.17467 |
| T1.11 | Ventura, Achilli, Silvestri, Lucibello, Ambrogioni. *Manifolds, random matrices and spectral gaps: the geometric phases of generative diffusion.* ICLR 2025. | Spectral gaps of the score Jacobian reveal sub-manifold dimensionality; three geometric phases derived with stat-phys tools. | Spectrum of score Jacobian vs. our Hermite spectrum of the score; no sequence structure. | arXiv:2410.05898 |
| T1.12 | El Alaoui, Montanari, Sellke. *Sampling from the Sherrington–Kirkpatrick Gibbs measure via algorithmic stochastic localization.* FOCS 2022. | O(n²) sampler for SK at β<1/2 via stochastic localization + AMP for the tilted mean. | Known-measure sampling; AMP as message passing is a loop-graph cousin of our exact sum-product on a chain. | arXiv:2203.05093 |
| T1.13 | Montanari. *Sampling, diffusions, and stochastic localization.* arXiv 2023. | Shows equivalence between stochastic-localization samplers and diffusion models in the Gaussian setting. | Framework paper; no learning through noise. | arXiv:2305.10690 |
| T1.14 | Ghane, Bao, Akhtiamov, Hassibi. *Gaussian universality for diffusion models.* arXiv 2025 (v3 Sept 2025). | Test error of GLMs trained on diffusion-generated data matches that on a Gaussian mixture with matched means/covariances — depends only on first two moments. | Downstream-linear-task universality; complements our claim that the covariance twin captures everything up to e^{-8t} corrections at high noise. | arXiv:2501.07741 |
| T1.15 | Wang, Zavatone-Veth, Pehlevan. *A random matrix theory perspective on the consistency of diffusion models.* ICML 2026 (oral). | RMT explains why models trained on disjoint splits agree: shared Gaussian statistics already predict much of the generated images; finite-sample effects via data spectrum. | Image data, empirical denoiser; our Gaussian-dominance is exact per Hermite degree for a Markov chain. | arXiv:2602.02908 |
| T1.16 | Kamb, Ganguli. *An analytic theory of creativity in convolutional diffusion models.* ICML 2025 (spotlight). | Locality + equivariance give closed-form "local score" machines predicting conv-only diffusion outputs (median r²≈0.95). | Analytic *approximate* score for images; our score is exact and the analysis is in noise-time rather than architecture. | arXiv:2412.20292 |
| T1.17 | Wu, Cai. *Diffusion models are statistically optimal for learning low-dimensional multi-modal distributions.* ICML 2026. | Near-optimal Õ(ε^{-k∨2}) sample rate in W₁ for unions of k-dim subspaces. | Minimax rates in intrinsic dimension; no noise-level-resolved information. | arXiv:2605.30153 |

---

## Theme 2 — Diffusion on graphical models; belief propagation as denoising; U-Nets/transformers as BP

| # | Citation | What they prove / do | How it differs from ours | ID |
|---|---|---|---|---|
| T2.1 | Mei. *U-Nets as belief propagation: efficient classification, denoising, and diffusion in generative hierarchical models.* ICLR 2025. | For tree-structured generative hierarchical models, BP denoising is the optimal denoiser and U-Nets can implement it with efficient sample complexity. | Tree = hierarchy over image pixels, denoiser architecture is the object; our tree is a *time chain* with continuous non-Gaussian states, and we derive the exact score in closed form and then *learn the law* through noise. | arXiv:2404.18444 |
| T2.2 | Mei, Wu. *Deep networks as denoising algorithms: sample-efficient learning of diffusion models in high-dimensional graphical models.* IEEE Trans. Inf. Theory (Xplore doc. 10857317). | Score functions of MRFs (Ising, RBM, sparse coding) are well approximated by unrolled variational-inference denoisers ⇒ efficient sample complexity. | Approximate (mean-field/VI) denoisers on loopy graphs; ours is exact sum-product on a chain. | arXiv:2309.11420 |
| T2.3 | Garnier-Brun, Mézard, Moscato, Saglietti. *How transformers learn structured data: insights from hierarchical filtering.* ICML 2025, PMLR 267:18831–18847. | On tree-structured sequences, encoder-only transformers approximate the exact BP inference; longer-range correlations (deeper tree levels) are learned sequentially. | Discrete symbols on trees, MLM/root-classification tasks; we compare transformers to an exact continuous-state chain score and study *noise-time* rather than training-time ordering. | arXiv:2408.15138 |
| T2.4 | Sclocchi, Favero, Wyart. *A phase transition in diffusion models reveals the hierarchical nature of data.* PNAS 122(1) e2408799121, 2025. | In the Random Hierarchy Model the exact score is computed by BP; forward–backward experiments show a phase transition where class identity flips while low-level features persist. | Hierarchical (deep-tree) data with discrete leaves; we have a shallow chain with continuous non-Gaussian states and derive explicit per-degree decay. | arXiv:2402.16991 |
| T2.5 | Sclocchi, Favero, Levi, Wyart. *Probing the latent hierarchical structure of data via diffusion models.* ICLR 2025 (also J. Stat. Mech. 2025). | Forward–backward experiments as a probe of latent hierarchy; correlated token changes measured on text and images. | Same as T2.4. | arXiv:2410.13770 |
| T2.6 | Cagnetta, Petrini, Tomasini, Favero, Wyart. *How deep neural networks learn compositional data: the random hierarchy model.* Phys. Rev. X 14, 031001 (2024). | Sample complexity to learn RHM tasks equals the point where low-level feature correlations become detectable. | Supervised, not diffusion; sets up the RHM used by T2.4/T2.5. | arXiv:2307.02129 |
| T2.7 | Cagnetta, Wyart. *Towards a theory of how the structure of language is acquired by deep neural networks.* NeurIPS 2024. | PCFG data: token–token correlations resolve deeper latent variables as training-set size grows. | Correlation range grows with *data size*; ours: correlation information shrinks with *noise time* as e^{-4t}. | arXiv:2406.00048 |
| T2.8 | Favero, Sclocchi, Cagnetta, Frossard, Wyart. *How compositional generalization and creativity improve as diffusion models are trained.* ICML 2025. | PCFG diffusion: coherence across scales increases with training; polynomial sample complexity in context size; RG analogy. | Training-time ordering on hierarchical data; no exact continuous-state score. | arXiv:2502.12089 |
| T2.9 | Garnier-Brun, Biggio, Beltrame, Mézard, Saglietti. *Biased generalization in diffusion models.* arXiv 2026. | A training phase where test loss falls yet samples sit anomalously close to training data; coarse structure learned sample-independently first, fine detail later. | Training dynamics; complements our noise-time hierarchy. | arXiv:2603.03469 |

---

## Theme 3 — Low-SNR / high-noise expansions; MMSE–MI; Hermite/OU analyses; what survives noising

| # | Citation | What they prove / do | How it differs from ours | ID |
|---|---|---|---|---|
| T3.1 | Guo, Shamai, Verdú. *Mutual information and minimum mean-square error in Gaussian channels.* IEEE Trans. Inf. Theory 51(4):1261–1282, 2005. | I-MMSE: dI/dsnr = ½·MMSE for any input distribution. | Global identity; our result resolves the MMSE *per Hermite degree* for a Markov chain. | doi:10.1109/TIT.2005.844072 |
| T3.2 | Prelov, Verdú. *Second-order asymptotics of mutual information.* IEEE Trans. Inf. Theory 50(8):1567–1580, 2004. | Low-SNR expansion of MI to second order; the second-order coefficient depends on fourth-order input statistics. | Scalar/vector channel, no Markov structure; first appearance of "order-4 statistics enter at SNR²", which our e^{-8t} generalises to score gaps. | IEEE TIT 2004 |
| T3.3 | Guo, Wu, Shamai, Verdú. *Estimation in Gaussian noise: properties of the minimum mean-square error.* IEEE Trans. Inf. Theory 57(4):2371–2385, 2011. | MMSE is real-analytic in SNR; low-SNR Taylor coefficients determined by input moments. | Same as T3.1. | IEEE TIT 2011 |
| T3.4 | Wu, Guo, Verdú. *Derivative of mutual information at zero SNR: the Gaussian-noise case.* IEEE Trans. Inf. Theory 57(11):7307–7312, 2011. | Conditions under which I(snr)/snr → ½ as snr→0. | Leading order only. | IEEE TIT 2011 |
| T3.5 | Wu, Verdú. *Functional properties of minimum mean-square error and mutual information.* IEEE Trans. Inf. Theory 58(3), 2012. | MMSE is concave in the joint law, weakly continuous and W₂-Lipschitz in the input. | Regularity, not rates by order. | IEEE TIT 2012 |
| T3.6 | Dytso, Poor, Shamai. *A general derivative identity for the conditional mean estimator in Gaussian noise and some applications* (arXiv 2021); journal version *Conditional mean estimation in Gaussian noise: a meta derivative identity with applications*, IEEE Trans. Inf. Theory 69:1883 (2023). | k-th derivative of E[X\|Y=y] equals the (k+1)-th conditional cumulant. | Gives the cumulant ladder for the *denoiser*; we exploit the same ladder in Hermite form to bound the twin gap. | arXiv:2104.01883 |
| T3.7 | Nguyen. *Beyond the I-MMSE relation: derivatives of mutual information in Gaussian channels.* arXiv 2023. | All SNR-derivatives of MI expressed via conditional cumulants (a cumulant–moment-type relation). | Same as T3.6, MI side. | arXiv:2303.02500 |
| T3.8 | Kong, Brekelmans, Ver Steeg. *Information-theoretic diffusion.* ICLR 2023. | I-MMSE rewritten as an exact diffusion likelihood: log-density = integral of MMSE over SNR. | Uses the *total* MMSE curve; we decompose it by Hermite degree. | arXiv:2302.03792 |
| T3.9 | Katsevich, Bandeira. *Likelihood maximization and moment matching in low SNR Gaussian mixture models.* Comm. Pure Appl. Math. 76(4):788–842, 2023. | Asymptotic series of the GMM log-likelihood in the low-SNR parameter; "likelihood optimization in the low SNR regime reduces to a sequence of least squares optimization problems" matching successively higher moments. | Static GMM/cryo-EM; same SNR-by-moment-order phenomenon as our e^{-2mt}, stated in σ^{-2} powers rather than OU time; no score, no Markov chain. | arXiv:2006.15202 |
| T3.10 | Perry, Weed, Bandeira, Rigollet, Singer. *The sample complexity of multireference alignment.* SIAM J. Math. Data Sci. 1(3):497–517, 2019. | Sample complexity for generic signals rises from 1/SNR at high SNR to 1/SNR³ at low SNR (third-order invariants needed). | Group-orbit model, not a Markov chain; the σ^{2k} law is our "sample complexity through diffusion noise" in a different model. | arXiv:1707.00943 |
| T3.11 | Bandeira, Rigollet, Weed. *Optimal rates of estimation for multi-reference alignment.* arXiv 2017. | Matching upper/lower bounds for MRA viewed as a group-constrained GMM; rates via moment tensors and KL control. | Same as T3.10. | arXiv:1702.08546 |
| T3.12 | Bandeira, Blum-Smith, Kileel, Perry, Niles-Weed, Wein. *Estimation under group actions: recovering orbits from invariants.* Appl. Comput. Harmon. Anal. 66:236–319, 2023. | Sample complexity ↔ degree of invariants needed; for cryo-EM "the number of images required to recover the molecule structure scales as σ^6". | Same as T3.10. | arXiv:1712.10163 |
| T3.13 | Romanov, Bendory, Ordentlich. *Multi-reference alignment in high dimensions: sample complexity and phase transition.* SIAM J. Math. Data Sci. 2021. | Phase transition in α = L/(σ² log L); method of moments is rate-optimal at fixed L, SNR→0. | Same as T3.10. | arXiv:2007.11482 |
| T3.14 | Balanov, Bendory, Edidin. *The generalized method of moments is (almost) statistically efficient in low-SNR Gaussian latent-variable models.* arXiv 2026. | Thm 1.2 (layered local geometry): for h,g in layer k, ⟨h, I_obs g⟩ = (SNR^k/k!)⟨DT_k[h], DT_k[g]⟩ + O(SNR^{k+1}) — Fisher information about directions first identifiable at moment order k scales as SNR^k; Hermite-moment expansion of the likelihood (App. B). | **This is the e^{-2mt} statement in SNR form** (SNR_t≈e^{-2t} under OU). Differences: i.i.d. latent-variable/orbit models, no Markov chain, no score-function/Hermite-mode statement, no covariance-twin gap, no denoiser comparison. | arXiv:2605.30095 |
| T3.15 | Wang, Vastola. *The unreasonable effectiveness of Gaussian score approximation for diffusion models and its applications.* TMLR 2024. | Learned score is dominated by its linear (Gaussian) part at moderate-to-high noise; error bound O(tr Σ/σ²); "analytical teleportation" skips 15–30% of steps. | Empirical/qualitative dominance with a crude bound; no cumulant-order rate. Our twin gap ∝ e^{-8t} is the sharp version for a Markov chain. | arXiv:2412.09726 |
| T3.16 | Wang, Vastola. *Diffusion models generate images like painters: an analytical theory of outline first, details later.* arXiv 2023. | Coarse-to-fine generation explained by a Gaussian-dominated early phase. | Same as T3.15. | arXiv:2303.02490 |
| T3.17 | Li, Dai, Qu. *Understanding generalizability of diffusion models requires rethinking the hidden Gaussian structure.* NeurIPS 2024. | As models generalise, learned denoisers become increasingly linear ≈ optimal denoiser of the Gaussian with empirical mean/covariance. | Empirical, images; the "hidden Gaussian" is our covariance twin, whose exact gap we compute. | arXiv:2410.24060 |
| T3.18 | Li, Chen. *Critical windows: non-asymptotic theory for feature emergence in diffusion models.* ICML 2024. | For mixtures of strongly log-concave components, features emerge in provably bounded time windows. | Mixture data; ours resolves structure by polynomial degree. | arXiv:2403.01633 |
| T3.19 | Biroli, Bonnaire, De Bortoli, Mézard. *Dynamical regimes of diffusion models.* Nat. Commun. 15, 9957 (2024). | Speciation time from covariance spectrum; collapse time from excess entropy. | Speciation is a second-moment (e^{-4t}-class) phenomenon in our language; no higher-order rates. | arXiv:2402.18491 |
| T3.20 | Kadkhodaie, Guth, Simoncelli, Mallat. *Generalization in diffusion models arises from geometry-adaptive harmonic representations.* ICLR 2024. | Two DNNs on disjoint halves learn the same score; denoisers shrink in geometry-adaptive harmonic bases. | Harmonics of *images*; ours are Hermite harmonics of the *noise channel*. | arXiv:2310.02557 |
| T3.21 | Falck, Pandeva, Zahirnia, Lawrence, Turner, Meeds, Zazo, Karmalkar. *A Fourier space perspective on diffusion models.* arXiv 2025. | High-frequency Fourier components lose SNR first; proposes equal-rate Fourier corruption. | Fourier (spatial) modes vs. our Hermite (distributional) modes. | arXiv:2505.11278 |
| T3.22 | Dieleman. *Diffusion is spectral autoregression.* Blog, 2 Sept 2024. | Diffusion on images ≈ autoregression over spatial frequencies. | Blog; same distinction as T3.21. | sander.ai/2024/09/02 |
| T3.23 | Rissanen, Heinonen, Solin. *Generative modelling with inverse heat dissipation.* ICLR 2023†. | Coarse-to-fine generation by inverting the heat equation. | Spatial smoothing, not distributional Gaussian smoothing. | arXiv:2206.13397 |
| T3.24 | Aarts, Habibi, Wang, Zhou. *On learning higher-order cumulants in diffusion models.* Mach. Learn.: Sci. Technol. 6 (2025) 025004. | Exact moment/cumulant generating functionals; with constant drift k: κ_{n>2}(t) = κ_n(0)·e^{-nkt/2} ("higher-order cumulants decay exponentially"); conserved without drift; validated on a toy model and lattice φ⁴. | **Contains the mode-n attenuation e^{-nt} in cumulant form.** No Fisher-information/sample-complexity corollary, no Markov chain, no score-gap bound. | arXiv:2410.21212 |
| T3.25 | Khoo, Oster, Peng. *Optimization-free diffusion model — a perturbation theory approach.* arXiv 2025. | Expands the score in eigenfunctions of the backward Kolmogorov (OU) operator; score estimation as a linear system. | Uses the OU eigenbasis numerically, not for information decay. | arXiv:2505.23652 |
| T3.26 | Vastola. *Generalization through variance: how noise shapes inductive biases in diffusion models.* ICLR 2025. | Path-integral theory of what under/over-parameterised models learn from the noisy DSM target. | Training-noise bias; not channel information. | arXiv:2504.12532 |
| T3.27 | Handke, Stančević, Koulischer, Demeester, Ambrogioni. *The entropic signature of class speciation in diffusion models.* ICML 2026. | Class-conditional entropy of a latent given x_t pinpoints speciation windows on EDM2/SD1.5. | Semantic-variable information; no polynomial-degree decomposition. | arXiv:2602.09651 |
| T3.28 | Li, Zhang, Li, Chen, Zhu, Wang, Qu. *Understanding representation dynamics of diffusion models via low-dimensional modeling.* NeurIPS 2025. | Feature quality peaks at intermediate noise; explained via low-dim structure. | Representation quality, not estimation information. | arXiv:2502.05743 |
| T3.29 | Premkumar. *On the separability of information in diffusion models.* ICML 2026. | Pixel models spend capacity on fine perceptual detail; CFG amplifies MI early. | Image-level information budget. | arXiv:2509.23937 |
| T3.30 | Kumar, Nagaraj, Sarkar. *Dimension-free score matching and time bootstrapping for diffusion models.* arXiv 2025. | Nearly dimension-free sample complexity for score learning across noise levels. | Uniform-in-time complexity; we resolve which *statistics* are learnable at which noise level. | arXiv:2502.10354 |
| T3.31 | Koehler, Heckett, Risteski. *Statistical efficiency of score matching: the view from isoperimetry.* ICLR 2023†. | Score matching ≈ MLE efficiency iff small isoperimetric constant. | Efficiency of the estimator, not information in the channel. | arXiv:2210.00726 |
| T3.32 | Pabbaraju, Rohatgi, Sevekari, Lee, Moitra, Risteski. *Provable benefits of score matching.* NeurIPS 2023. | Score matching computationally easier than MLE for some exponential families with comparable statistical efficiency. | Same as T3.31. | arXiv:2306.01993 |
| T3.33 | Stancevic, Ambrogioni. *The information dynamics of generative diffusion.* arXiv 2025. | Conditional-entropy production rate = expected score divergence; bifurcations as symmetry breaking. | Rate of information *flow*, not decay by order. | arXiv:2508.19897 |

Also seen, lower relevance: Sarkar, *Score shocks: the Burgers equation structure of diffusion generative models*, arXiv:2604.07404; Albrychiewicz, Franco Valiente, Chen, *Dynamical regimes of multimodal diffusion models*, arXiv:2602.04780; Zhang, Hu, Liu, Okyay, Gao, *Concurrence of symmetry breaking and nonlocality phase transitions in diffusion models*, arXiv:2605.04830; Karczewski et al., *The spacetime of diffusion models*, ICLR 2026 oral, arXiv:2505.17517.

---

## Theme 4 — Memorisation vs generalisation

| # | Citation | What they prove / do | How it differs from ours | ID |
|---|---|---|---|---|
| T4.1 | Kadkhodaie, Guth, Simoncelli, Mallat. ICLR 2024 (see T3.20). | Strong generalisation once N is large; disjoint halves give same score. | Image DNNs; we have an exact score and exact memorisation-free EM. | arXiv:2310.02557 |
| T4.2 | Biroli, Bonnaire, De Bortoli, Mézard. Nat. Commun. 2024 (see T3.19). | Speciation then collapse onto training points at a time set by excess entropy. | Empirical-score memorisation; we learn a parametric law, so memorisation is replaced by misspecification. | arXiv:2402.18491 |
| T4.3 | Bonnaire, Urfin, Biroli, Mézard. *Why diffusion models don't memorize: the role of implicit dynamical regularization in training.* NeurIPS 2025. | Two timescales τ_gen ≪ τ_mem; τ_mem grows with dataset size. | Training-dynamics regularisation; orthogonal to noise-time information. | arXiv:2505.17638 |
| T4.4 | Achilli, Ventura, Silvestri, Pham, Raya, Krotov, Lucibello, Ambrogioni. *Losing dimensions: geometric memorization in generative diffusion.* arXiv 2024 (OpenReview ICLR 2025 submission). | Memorisation is gradual: latent dimensionality collapses feature by feature, salient first. | Geometric collapse of the empirical score. | arXiv:2410.08727 |
| T4.5 | Achilli, Ambrogioni, Lucibello, Mézard, Ventura. *Memorization and generalization in generative diffusion under the manifold hypothesis.* arXiv 2025. | Hidden-manifold model + REM: curse of dimensionality avoided for structured data; generalisation precedes memorisation. | Manifold data; no chain. | arXiv:2502.09578 |
| T4.6 | Favero, Sclocchi, Wyart. *Bigger isn't always memorizing: early stopping overparameterized diffusion models.* ICLR 2026. | Generalisation achieved before memorisation; τ_mem scales with dataset size; PCFG model of hierarchical acquisition. | Training-time; PCFG data. | arXiv:2505.16959 |
| T4.7 | George, Veiga, Macris. *Denoising score matching with random features: insights on diffusion models from precise learning curves.* AISTATS 2026. | Exact test/train error of random-feature DSM on Gaussian data; noise samples per data point m controls the memorisation/generalisation regime. | Gaussian target; this is the "George et al. 2025" transition paper. | arXiv:2502.00336 |
| T4.8 | Pham, Raya, Negri, Zaki, Ambrogioni, Krotov. *Memorization to generalization: emergence of diffusion models from associative memory.* arXiv 2025 (v3 2026). | Beyond Dense-AM capacity, spurious minima become the first generative states. | Energy-landscape view. | arXiv:2505.21777 |
| T4.9 | Buchanan, Pai, Ma, De Bortoli. *On the edge of memorization in diffusion models.* arXiv 2025. | Critical model size where the generalising solution's weighted loss exceeds the memorising one; phase transition in GD-trained models. | Capacity-driven; we vary noise-time and misspecification instead. | arXiv:2508.17689 |
| T4.10 | Ye, Zhu, Tao, Chen. *Provable separations between memorization and generalization in diffusion models.* arXiv 2025. | Ground-truth score does not minimise the empirical DSM loss; empirical score needs network size ∝ sample size. | Approximation-theoretic. | arXiv:2511.03202 |
| T4.11 | Merger, Goldt. *Local coverage governs memorization in diffusion models.* arXiv 2026. | Memorisation is local: sparse regions memorise, dense regions generalise (KDE link). | Density-coverage view. | arXiv:2606.14390 |
| T4.12 | Kaiser, Kollmann. *Diffusion models memorize in training — and generalize in inference.* arXiv 2026. | Overfitting gap largest at intermediate noise; sampling trajectories avoid it. | Empirical mechanism. | arXiv:2603.13419 |
| T4.13 | Hunt, Kamb, Ganguli. *An exact information theory of generalization phase transitions in Bayesian diffusion models.* arXiv 2026. | "Bayesian information-restricted diffusion" (BIRD): memorises iff I(restricted noisy obs; training set) > ln\|D\|; local BIRD tracks early-training U-Nets/DiTs (r²≈0.85–0.93). Uses a Gaussian MI bound; no Hermite/cumulant decomposition. | Information budget vs. dataset size; ours is information budget vs. *polynomial order*. | arXiv:2607.08041 |
| T4.14 | Zhang, Zhou, Lu, Guo, Wang, Shen, Qu. *The emergence of reproducibility and generalizability in diffusion models.* ICML 2024. | Different models map the same noise to near-identical samples in both regimes. | Empirical. | arXiv:2310.05264 |
| T4.15 | Song, Kim, Chen, Du, Kakade, Sitzmann. *Selective underfitting in diffusion models.* arXiv 2025. | Better models fit the score accurately in some regions and underfit in others. | Region-of-input view. | arXiv:2510.01378 |
| T4.16 | Wang, Finn, Liu. *The two clocks and the innovation window: when and how generative models learn rules.* arXiv 2026. | τ_rule vs τ_mem; window widens with N, narrows with rule complexity. | Rule-learning in training time. | arXiv:2605.10019 |
| T4.17 | Marion, Wu. *Understanding diffusion models requires rethinking (again) generalization.* arXiv 2026 (position). | Argues the field should study what is learned pre-memorisation. | Position paper. | arXiv:2605.06077 |
| T4.18 | Niedoba, Zwartsenberg, Wood. *Filtered posterior mean collections: a unified framework for analytical models of diffusion generalization.* arXiv 2026. | Unifies patch-based analytic denoisers (query precision, response weights, source distribution). | Image-denoiser surrogates. | arXiv:2605.24192 |

---

## Theme 5 — Score matching from noisy data only; ambient diffusion; SURE/EM training

| # | Citation | What they prove / do | How it differs from ours | ID |
|---|---|---|---|---|
| T5.1 | Hyvärinen. *Estimation of non-normalized statistical models by score matching.* JMLR 6:695–709, 2005. | Score matching objective; consistency for unnormalised models. | Clean data. | jmlr.org/papers/v6/hyvarinen05a |
| T5.2 | Vincent. *A connection between score matching and denoising autoencoders.* Neural Comput. 23(7):1661–1674, 2011. | DSM = score matching against a Parzen-smoothed density. | Clean data. | doi:10.1162/NECO_a_00142 |
| T5.3 | Song, Garg, Shi, Ermon. *Sliced score matching: a scalable approach to density and score estimation.* UAI 2019. | Random projections make implicit score matching scalable. | Clean data. | arXiv:1905.07088 |
| T5.4 | Kim, Ye. *Noise2Score: Tweedie's approach to self-supervised image denoising without clean images.* NeurIPS 2021. | Tweedie + score estimation from noisy images (Gaussian, Poisson, Gamma). | Denoising, not distribution/parameter learning. | arXiv:2106.07009 |
| T5.5 | Daras, Shah, Dagan, Gollakota, Dimakis, Klivans. *Ambient diffusion: learning clean distributions from corrupted data.* NeurIPS 2023. | Extra corruption + prediction of the corrupted image ⇒ learn E[x_clean\|more-corrupted]; memorisation-free. | Masking corruptions, images, non-parametric. | arXiv:2305.19256 |
| T5.6 | Daras, Dimakis, Daskalakis. *Consistent diffusion meets Tweedie: training exact ambient diffusion models with noisy data.* ICML 2024. | First provable sampler of the clean law from Gaussian-noisy data: double Tweedie + consistency loss below the observed noise level. | Nonparametric; cannot exploit a Markov prior; we learn a parametric transition law by EM with exact E-step. | arXiv:2404.10177 |
| T5.7 | Daras, Rodriguez-Munoz, Klivans, Torralba, Daskalakis. *Ambient diffusion omni: training good models with bad data.* NeurIPS 2025 (spotlight). | Low-quality/OOD images help at high noise; bias–variance trade-off over diffusion times. | Data-quality routing; no identifiability theory. | arXiv:2506.10038 |
| T5.8 | Rozet, Andry, Lanusse, Louppe. *Learning diffusion priors from observations by expectation maximization.* NeurIPS 2024. | EM with diffusion posterior sampling in the E-step yields proper diffusion priors from incomplete/noisy data. | Nonparametric prior, approximate E-step; ours uses Fisher's identity with an exact (sum-product) E-step. | arXiv:2405.13712 |
| T5.9 | Bai, Wang, Chen, Sun. *An expectation-maximization algorithm for training clean diffusion models from corrupted observations.* NeurIPS 2024. | EMDiffusion: alternate reconstruction and refinement; SOTA on inpainting/denoising/deblurring. | Same as T5.8. | arXiv:2407.01014 |
| T5.10 | Hosseintabar, Chen, Daras, Torralba, Daskalakis. *DiffEM: learning from corrupted data with diffusion models via expectation maximization.* arXiv 2025. | Conditional diffusion in the E-step; convergence guarantees. | Same as T5.8. | arXiv:2510.12691 |
| T5.11 | Kawar, Elata, Michaeli, Elad. *GSURE-based diffusion model training with corrupted data.* TMLR 2024. | GSURE loss trains diffusion models from linearly degraded + Gaussian-noisy data. | SURE surrogate, images. | arXiv:2305.13128 |
| T5.12 | Aali, Arvinte, Kumar, Tamir. *Solving inverse problems with score-based generative priors learned from noisy data.* Asilomar 2023. | SURE-Score: joint denoising + DSM from Gaussian-noisy samples. | Same as T5.11. | arXiv:2305.01166 |
| T5.13 | Park, Shoushtari, An, Kamilov. *Measurement score-based diffusion model.* arXiv 2025. | Learns partial measurement scores from noisy subsampled measurements only. | Imaging measurement model. | arXiv:2505.11853 |

---

## Theme 6 — HMM / state-space parameter learning under noise; identifiability; Fisher's identity

| # | Citation | What they prove / do | How it differs from ours | ID |
|---|---|---|---|---|
| T6.1 | Cappé, Moulines, Rydén. *Inference in Hidden Markov Models.* Springer, 2005. | Reference for forward–backward, EM, and "the score function and invoking the Fisher identity" for HMMs. | Our E-step is exactly this machinery under a *continuous-state* chain with OU emission; we add the noise-level-resolved information analysis. | ISBN 978-0-387-40264-2 |
| T6.2 | Douc, Moulines, Olsson, van Handel. *Consistency of the maximum likelihood estimator for general hidden Markov models.* Ann. Statist. 39(1), 2011. | Strong consistency of the MLE for general-state-space HMMs; information-theoretic identifiability argument. | Gives consistency of our EM target; no rates in noise level. | arXiv:0912.4480 |
| T6.3 | Allman, Matias, Rhodes. *Identifiability of parameters in latent structure models with many observed variables.* Ann. Statist. 37(6A):3099–3132, 2009. | Kruskal-based generic identifiability incl. HMMs. | Finite-state; ours is continuous non-Gaussian state with known Gaussian channel. | arXiv:0809.5032 |
| T6.4 | Hsu, Kakade, Zhang. *A spectral algorithm for learning hidden Markov models.* J. Comput. Syst. Sci. 78(5):1460–1480, 2012. | SVD-based method-of-moments HMM learning with sample-complexity guarantees. | Discrete observations; moment methods here are the natural competitor to EM at high noise (cf. T3.14). | arXiv:0811.4413 |
| T6.5 | Anandkumar, Hsu, Kakade. *A method of moments for mixture models and hidden Markov models.* COLT 2012. | Tensor moment methods for multi-view mixtures and HMMs. | Same as T6.4. | arXiv:1203.0683 |
| T6.6 | Gassiat, Rousseau. *Nonparametric finite translation hidden Markov models and extensions.* Bernoulli 22(1):193–212, 2016. | Identifiability of finite translation HMMs (latent + noise) with dependent latents. | Finite latent alphabet. | doi:10.3150/14-BEJ631 |
| T6.7 | Gassiat, Cleynen, Robin. *Inference in finite state space non parametric hidden Markov models and applications.* Stat. Comput. 26:61–71, 2016. | Nonparametric finite HMMs identifiable if transition matrix full rank and emissions linearly independent. | Finite state. | arXiv:1306.4657 |
| T6.8 | Gassiat, Le Corff, Lehéricy. *Identifiability and consistent estimation of nonparametric translation hidden Markov models with general state space.* JMLR 21(115):1–40, 2020. | Observation = latent (general space, Markov) + independent noise of *unknown* law: both laws identifiable under a light-tail assumption; consistent estimators. | Closest identifiability result to ours; but noise law unknown there vs. known OU here, and no rates in noise level. | arXiv:1902.01070 |
| T6.9 | Gassiat, Le Corff, Lehéricy. *Deconvolution with unknown noise distribution is possible for multivariate signals.* Ann. Statist. 50(1):303–323, 2022 (+ erratum). | Identifiability up to translation when the signal has two dependent components; rates and adaptivity. | Dependence (as in a Markov pair) is what makes deconvolution possible — same structural lever we use. | arXiv:2006.14226; doi:10.1214/21-AOS2106 |
| T6.10 | Poyiadjis, Doucet, Singh. *Particle approximations of the score and observed information matrix in state space models with application to parameter estimation.* Biometrika 98(1):65–80, 2011. | SMC estimators of score and OIM via Fisher/Louis identities. | Approximate; our chain admits exact sum-product. | doi:10.1093/biomet/asq062 |
| T6.11 | Cappé. *Online EM algorithm for hidden Markov models.* J. Comput. Graph. Statist. 2011. | Recursive EM combining stochastic approximation with recursive smoothing. | Online variant; ours is batch. | arXiv:0908.2359 |
| T6.12 | Shumway, Stoffer. *An approach to time series smoothing and forecasting using the EM algorithm.* J. Time Ser. Anal. 3:253–264, 1982. | EM + Kalman smoother for linear Gaussian state-space models. | Gaussian case = our covariance-twin EM. | doi:10.1111/j.1467-9892.1982.tb00349.x |
| T6.13 | Lefebvre, Nuel. *Observed Fisher information in hidden Markov models — application to a noisy Gaussian random walk.* arXiv 2026. | Linear-time score and observed Fisher information (Oakes' identity + forward–backward) for a Gaussian random walk observed in Gaussian noise. | Gaussian latent chain only; we treat non-Gaussian chains and quantify how Fisher information decays with noise level. | arXiv:2606.02118 |
| T6.14 | Gao, Li, Chen. *Parameter estimation of state space models using particle importance sampling.* AISTATS 2025. | Consistent score estimators with linear cost via particle importance sampling. | Approximate. | arXiv:2502.00904 |

---

## Theme 7 — Sequence / time-series diffusion; structured priors; length generalisation

| # | Citation | What they prove / do | How it differs from ours | ID |
|---|---|---|---|---|
| T7.1 | Rasul, Seward, Schuster, Vollgraf. *Autoregressive denoising diffusion models for multivariate probabilistic time series forecasting* (TimeGrad). ICML 2021, PMLR 139:8857–8868. | Per-step diffusion conditioned on an RNN state. | Engineering baseline; no exact score. | arXiv:2101.12072 |
| T7.2 | Tashiro, Song, Song, Ermon. *CSDI: conditional score-based diffusion models for probabilistic time series imputation.* NeurIPS 2021. | Conditional diffusion imputation; 40–65% gains. | Same. | arXiv:2107.03502 |
| T7.3 | Alcaraz, Strodthoff. *Diffusion-based time series imputation and forecasting with structured state space models* (SSSD). TMLR. | S4 backbone inside diffusion. | Same. | arXiv:2208.09399 |
| T7.4 | Kollovieh, Ansari, Bohlke-Schneider, Zschiegner, Wang, Wang. *Predict, refine, synthesize: self-guiding diffusion models for probabilistic time series forecasting* (TSDiff). NeurIPS 2023. | Unconditional TS diffusion with self-guidance. | Same. | arXiv:2307.11494 |
| T7.5 | Yuan, Qiao. *Diffusion-TS: interpretable diffusion for general time series generation.* ICLR 2024†. | Decomposed trend/season transformer denoiser. | Same. | arXiv:2403.01742 |
| T7.6 | Biloš, Rasul, Schneider, Nevmyvaka, Günnemann. *Modeling temporal data as continuous functions with stochastic process diffusion.* ICML 2023. | Diffusion in function space with GP-structured noise. | Structured *noise* prior; our structure is in the *data* (Markov), noise is coordinatewise OU. | arXiv:2211.02590 |
| T7.7 | Kollovieh, Lienen, Lüdke, Schwinn, Günnemann. *Flow matching with Gaussian process priors for probabilistic time series forecasting* (TSFlow). ICLR 2025. | GP source distribution + OT couplings. | Same as T7.6. | arXiv:2410.03024 |
| T7.8 | Chang, Tang, Gross, Da Costa de Azevedo. *How I warped your noise: a temporally-correlated noise prior for diffusion models.* ICLR 2024 (oral). | ∫-noise: temporally correlated Gaussian noise for video. | Same as T7.6. | openreview.net/forum?id=pzElnMrgSD |
| T7.9 | Chen, Martí Monsó, Du, Simchowitz, Tedrake, Sitzmann. *Diffusion forcing: next-token prediction meets full-sequence diffusion.* NeurIPS 2024†. | Independent per-token noise levels; rolls out "past the training horizon, where baselines diverge". | Length extrapolation by design; we measure length generalisation of fixed denoisers against an exact score. | arXiv:2407.01392 |
| T7.10 | Ruhe, Heek, Salimans, Hoogeboom. *Rolling diffusion models.* ICML 2024†. | Sliding-window denoising with increasing noise on later frames. | Same as T7.9. | arXiv:2402.09470 |
| T7.11 | Kazemnejad, Padhi, Natesan, Das, Reddy. *The impact of positional encoding on length generalization in transformers.* NeurIPS 2023. | NoPE beats APE/ALiBi/Rotary for length generalisation in decoder-only models. | Positional-encoding view; we contrast conv (translation-equivariant) vs. transformer denoisers on a stationary chain. | arXiv:2305.19466 |
| T7.12 | Press, Smith, Lewis. *Train short, test long: attention with linear biases enables input length extrapolation* (ALiBi). ICLR 2022†. | Linear attention biases extrapolate to longer inputs. | Same as T7.11. | arXiv:2108.12409 |
| T7.13 | Huang, Yang, Bhattamishra, Sarrof, Krebs, Zhou, Nakkiran, Hahn. *A formal framework for understanding length generalization in transformers.* ICLR 2025. | Characterises which functions remain identifiable from longer inputs with learnable APE. | Formal language tasks. | arXiv:2410.02140 |
| T7.14 | Bai, Kolter, Koltun. *An empirical evaluation of generic convolutional and recurrent networks for sequence modeling* (TCN). arXiv 2018. | Dilated causal convolutions rival RNNs. | Our TCN baseline. | arXiv:1803.01271 |
| T7.15 | Lin, Li, Li, Li, Gao. *Diffusion models for time series applications: a survey.* arXiv 2023. | Survey. | — | arXiv:2305.00624 |
| T7.16 | Yang et al. *A survey on diffusion models for time series and spatio-temporal data.* ACM Comput. Surv. (accepted). | Survey. | — | arXiv:2404.18886 |
| T7.17 | Koohfar. *Structured noise modeling for enhanced time-series forecasting.* arXiv 2025. | Learnable GP-blur perturbations in a forecast-blur-denoise loop. | Same as T7.6. | arXiv:2511.19657 |

---

## Theme 8 — Cumulants / higher-order statistics in generative models; kurtosis and diffusion

| # | Citation | What they prove / do | How it differs from ours | ID |
|---|---|---|---|---|
| T8.1 | Aarts, Habibi, Wang, Zhou. (see T3.24) MLST 2025. | κ_n(t)=κ_n(0)e^{-nkt/2} under drift; conserved without drift; score regenerates cumulants in the backward pass. | No Fisher/sample-complexity consequence; no chain; no twin gap. | arXiv:2410.21212 |
| T8.2 | Wang, Aarts, Zhou. *Diffusion models as stochastic quantization in lattice field theory.* JHEP 05 (2024) 060. | Diffusion ↔ stochastic quantisation; samples φ⁴ configurations. | Physics sampling. | arXiv:2309.17082 |
| T8.3 | Habibi, Aarts, Wang, Zhou. *Diffusion models learn distributions generated by complex Langevin dynamics.* Proc. Lattice 2024. | Learns complex-Langevin-generated distributions. | Physics sampling. | arXiv:2412.01919 |
| T8.4 | Nguyen. (see T3.7) arXiv 2023. | MI derivatives ↔ conditional cumulants. | — | arXiv:2303.02500 |
| T8.5 | Dytso, Poor, Shamai. (see T3.6) IEEE TIT 2023. | Denoiser derivatives ↔ conditional cumulants. | — | arXiv:2104.01883 |
| T8.6 | Prelov, Verdú. (see T3.2) IEEE TIT 2004. | Fourth-order statistics enter MI at second order in SNR. | — | IEEE TIT 2004 |
| T8.7 | Ghane, Bao, Akhtiamov, Hassibi. (see T1.14) arXiv 2025. | Linear-task universality in first two moments. | — | arXiv:2501.07741 |
| T8.8 | Pandey, Pathak, Xu, Mandt, Pritchard, Vahdat, Mardani. *Heavy-tailed diffusion models.* arXiv 2024. | Student-t perturbation kernels (t-EDM, t-Flow) for extreme events. | Heavy-tailed *noise*; we study Gaussian noise on heavy/asymmetric *data*. | arXiv:2410.14171 |
| T8.9 | Tang, Touzi, Zhang, Zhou. *Tweedie's formulae and diffusion generative models beyond Gaussian.* arXiv 2026. | Tweedie for GBM, BESQ, CIR channels. | Non-Gaussian channel. | arXiv:2605.19391 |
| T8.10 | Wang, Vastola. (see T3.15) TMLR 2024. | Gaussian score dominates at high noise. | — | arXiv:2412.09726 |

No paper titled or centred on "kurtosis and diffusion (models)" was found; the kurtosis-in-diffusion literature is entirely about heavy-tailed noise kernels (T8.8) or MRI (irrelevant).

---

## Closest five (ranked by likely reviewer overlap with the central theorem)

Our claim, restated: under coordinatewise OU noise the Hermite mode of degree m of the data density (relative to the Gaussian reference) is multiplied by e^{-mt}; hence Fisher information about any parameter that is first visible in order-m structure decays as e^{-2mt} (pairwise correlation of a Markov chain: m=2 ⇒ e^{-4t}; non-Gaussian marginal/transition shape: m=4 ⇒ e^{-8t}); the score of the covariance-matched Gaussian twin differs from the true score by O(e^{-8t}); consequences for the sample complexity of EM through diffusion noise.

### 1. Balanov, Bendory, Edidin (2026), arXiv:2605.30095 — *The generalized method of moments is (almost) statistically efficient in low-SNR Gaussian latent-variable models*
- **What they prove.** For a broad class of Gaussian latent-variable models (GMMs, orbit recovery/MRA/cryo-EM), Theorem 1.2 ("layered local geometry"): for parameter directions h,g in "layer k" (first identifiable at moment order k), ⟨h, I_obs(θ*) g⟩ = (SNR^k / k!)·⟨DT_k(θ*)[h], DT_k(θ*)[g]⟩_F + O(SNR^{k+1}) — i.e. **Fisher information about order-k directions scales as SNR^k**, and the GMoM information operator matches layer by layer. Appendix B is a Hermite-moment expansion of the likelihood.
- **What we prove.** The same exponent, but for a *Markov chain* observed through the *OU* channel in *diffusion time*: with SNR_t = e^{-2t}/(1−e^{-2t}) ≈ e^{-2t} their SNR^k is our e^{-2kt}. We additionally prove the Hermite-mode attenuation of the *score* (not only of the likelihood/Fisher information), the covariance-twin score gap ∝ e^{-8t}, exact sum-product computability of the score, and identifiability of the transition law from noised sequences via EM.
- **What experiment they do.** Analytical; numerical illustrations on GMM/orbit-recovery models (no neural networks, no sequences).
- **What is genuinely new here.** Honest statement: **the "information about order-k structure scales as SNR^k" law is already in this paper (and, in σ^{-2k} form, in T3.9–T3.13).** New in ours: the diffusion-time/OU phrasing with explicit e^{-mt} Hermite attenuation; the Markov-chain (dependent, sequential) setting rather than i.i.d. latent-variable models; the score-level (denoiser-level) consequence and the e^{-8t} twin gap; EM with an exact E-step rather than method of moments; and the empirical comparison with neural denoisers. We should cite this paper in the theorem statement and frame ours as the sequential/diffusion instance, not as the discovery of the SNR^k law.

### 2. Katsevich, Bandeira (2023), CPAM 76(4):788–842, arXiv:2006.15202 — *Likelihood maximization and moment matching in low SNR Gaussian mixture models* (with Perry et al. 2019, arXiv:1707.00943; Bandeira et al. 2023, arXiv:1712.10163 as corollaries)
- **What they prove.** Asymptotic expansion of the GMM log-likelihood in the small parameter σ^{-2}: "likelihood optimization in the low SNR regime reduces to a sequence of least squares optimization problems, in which successively higher moments are matched" — the order-k term of the expansion carries only k-th-moment information. Perry et al. turn this into sample complexity 1/SNR³ for generic MRA (third-order invariants needed); Bandeira et al. (orbit recovery) state that for cryo-EM "the number of images required … scales as σ^6".
- **What we prove.** The dynamic version: information about order-m structure decays as e^{-2mt}, with m=2 (correlation) and m=4 (shape) separated by a factor e^{-4t}; and consequences for how many noised sequences EM needs at noise level t.
- **What experiment they do.** Analytical, with cryo-EM/MRA as the motivating model; no learning experiments.
- **What is genuinely new here.** Same as (1): the σ^{2k} sample-complexity law is established; ours transports it to a Markov chain under OU and connects it to the score/denoiser and to EM. Cite as the origin of the sample-complexity scaling.

### 3. Aarts, Habibi, Wang, Zhou (2025), Mach. Learn.: Sci. Technol. 6, 025004, arXiv:2410.21212 — *On learning higher-order cumulants in diffusion models*
- **What they prove.** From exact cumulant-generating functionals: for a linear drift k(t), κ_n(t) = κ_n(0)·f^n(t,0), f = exp(−½∫k); for constant drift **κ_{n>2}(t) = κ_n(0)·e^{-nkt/2}** ("higher-order cumulants decay exponentially" and "the distribution at the end of the forward process is a normal distribution, up to exponentially suppressed terms"); cumulants are conserved for variance-exploding schemes; the score encodes them so the backward pass regenerates them.
- **What we prove.** The Hermite-mode statement e^{-mt} is the orthogonal-polynomial counterpart of their cumulant decay (with unit-rate OU, k=2, giving e^{-nt}); we go on to the Fisher-information square (e^{-2mt}), the twin score gap e^{-8t}, and the learning consequences.
- **What experiment they do.** Exactly solvable toy model with nonzero cumulants; scalar lattice φ⁴ theory; standard neural score networks.
- **What is genuinely new here.** **The mode-n attenuation e^{-nt} is in this paper in cumulant form.** Ours adds the information-theoretic square, the Markov-chain structure (temporal cumulants/correlations vs. one-site cumulants), the exact score, the twin-gap rate, and EM-through-noise. Cite explicitly when stating the e^{-mt} lemma.

### 4. Wang, Vastola (2024), TMLR, arXiv:2412.09726 — *The unreasonable effectiveness of Gaussian score approximation for diffusion models* (companion: Li, Dai, Qu, NeurIPS 2024, arXiv:2410.24060)
- **What they prove.** The learned score is dominated by its linear Gaussian part for moderate-to-high noise; approximation error O(tr Σ/σ²); "works for a larger range of noise scales than naive theory suggests". Li–Dai–Qu show empirically that as models generalise their denoisers become approximately the optimal denoiser of a Gaussian with the empirical mean/covariance.
- **What we prove.** The covariance-matched Gaussian twin's score differs from the exact score by O(e^{-8t}) for a non-Gaussian Markov chain — a sharp cumulant-order rate that explains why the Gaussian approximation works "for a larger range than naive theory suggests" (the naive bound decays like SNR, the true gap like SNR²·(fourth-order structure)).
- **What experiment they do.** CIFAR-10, FFHQ-64, AFHQ-64; fraction of unexplained variance vs. noise scale; sampler acceleration (FID 1.93 CIFAR-10).
- **What is genuinely new here.** The rate itself and its derivation via Hermite modes; the Markov-chain instance where the twin is exactly the LMMSE/Kalman smoother (our closure theorem). No overlap on the e^{-2mt} statement.

### 5. Mei (2025), ICLR 2025, arXiv:2404.18444 — *U-Nets as belief propagation* (companion: Garnier-Brun, Mézard, Moscato, Saglietti, ICML 2025, arXiv:2408.15138)
- **What they prove.** On tree-structured generative hierarchical models the exact denoiser is BP; U-Nets implement it with sample-efficient guarantees. Garnier-Brun et al. show transformers approximate exact BP on tree-structured sequences, learning longer-range correlations later in training.
- **What we prove.** For a first-order Markov chain under coordinatewise OU noise, the posterior over the clean sequence is a chain (tree) and the score is exactly sum-product; Gaussian message closure is the covariance-twin LMMSE; neural denoisers (MLP, window CNN, TCN, transformer, U-Net) are benchmarked against the exact score, including length generalisation and misspecification.
- **What experiment they do.** Mei: synthetic hierarchical models. Garnier-Brun et al.: synthetic tree-structured sequences, encoder-only transformers on root classification and MLM.
- **What is genuinely new here.** No overlap on the information-decay theorem. Overlap is on "exact score = BP on a tree, networks approximate it". New: continuous non-Gaussian states with closed-form Hermite analysis; the learning-from-noised-data (EM) problem; noise-time (not training-time) ordering of what is learnable; quantitative length-generalisation comparison against an exact oracle.

**Next in line (would be cited in the same paragraph):** Perry et al. 2019 (T3.10) and Bandeira et al. 2023 (T3.12) for the σ^{2k} sample-complexity law; Guo–Wu–Shamai–Verdú 2011 (T3.3) and Prelov–Verdú 2004 (T3.2) for moments entering MMSE/MI order by order in SNR; Gassiat–Le Corff–Lehéricy 2020 (T6.8) for identifiability of "Markov latent + independent noise"; Hunt–Kamb–Ganguli 2026 (T4.13) for an information-budget theory of diffusion generalisation; Sclocchi–Favero–Wyart 2025 (T2.4) for BP-exact scores on the RHM.

**Not found (after targeted searches).** No paper stating the covariance-twin score gap as e^{-8t} (or SNR²·κ₄) for any model; no paper doing EM with Fisher's identity for a non-Gaussian Markov chain observed through diffusion noise; no paper combining "which Hermite degrees survive noising" with neural-denoiser length generalisation. The Hermite eigen-decay of the OU semigroup itself (Mehler kernel, 1866; Bakry–Gentil–Ledoux 2014 textbook) is classical and should be cited as such rather than claimed.

---

## Search queries used (WebSearch unless noted; WebFetch on arXiv abs/html pages for every ID listed above)

1. diffusion models Hermite polynomials Ornstein-Uhlenbeck eigenfunctions score high noise Gaussian arXiv 2025
2. "diffusion models" learn low-order moments first high noise cumulants arXiv 2025
3. Biroli Mézard speciation diffusion models symmetry breaking dynamical regimes arXiv
4. Wang Vastola "diffusion models generate images like painters" / "unreasonable effectiveness of Gaussian score approximation"
5. Li Chen "critical windows" diffusion models features emerge
6. Guo Shamai Verdú "Estimation in Gaussian noise: properties of the minimum mean-square error" IEEE Trans IT 2011
7. Prelov Verdú "second-order asymptotics of mutual information" low SNR fourth-order 2004
8. Mei "U-Nets as belief propagation" diffusion models hierarchical arXiv 2024
9. Garnier-Brun Mézard Moscato Saglietti "how transformers learn structured data" belief propagation
10. Sclocchi Favero Wyart "phase transition in diffusion models reveals hierarchical nature of data" PNAS
11. Biroli Bonnaire de Bortoli Mézard "dynamical regimes of diffusion models" Nature Communications 2024
12. Kadkhodaie Guth Simoncelli Mallat "geometry-adaptive harmonic representations" ICLR 2024
13. Ghio Dandi Krzakala Zdeborová spin-glass perspective PNAS 2024
14. Cui Krzakala Vanden-Eijnden Zdeborová flow-based generative model limited sample complexity ICLR 2024
15. "Gaussian Universality for Diffusion Models" arXiv 2501.07741 authors
16. Li Dai Qu "hidden Gaussian structure" NeurIPS 2024
17. Mei Wu "deep networks as denoising algorithms" arXiv 2309.11420
18. Sclocchi Favero Levi Wyart "probing the latent hierarchical structure of data via diffusion models" ICLR 2025
19. belief propagation exact score diffusion model tree graphical model denoising sum-product 2025 arXiv
20. Bonnaire Urfin Biroli Mézard "why diffusion models don't memorize" 2025
21. Achilli Raya Ambrogioni "losing dimensions" geometric memorization
22. George 2025 diffusion models transition to memorization sample size arXiv
23. Daras "consistent diffusion meets Tweedie" ICML 2024
24. Daras Dimakis "Ambient Diffusion" NeurIPS 2023
25. Rozet "learning diffusion priors from observations by expectation maximization" NeurIPS 2024
26. Kim Ye "Noise2Score" NeurIPS 2021
27. Gassiat Le Corff Lehéricy "deconvolution with unknown noise distribution is possible for multivariate signals"
28. Gassiat Le Corff Lehéricy "nonparametric translation hidden Markov models with general state space" JMLR
29. Hermite expansion score function diffusion model degree k coefficient decays e^{-kt} Ornstein-Uhlenbeck Mehler
30. Fisher information Gaussian channel Hermite polynomial degree exponential decay low-degree statistics survive noise
31. Falck "Fourier space perspective on diffusion generative models" 2025
32. Kamb Ganguli "analytic theory of creativity in convolutional diffusion models" ICML 2025
33. Biroli Mézard "generative diffusion in very large dimensions" J. Stat. Mech. 2023
34. Cui Vanden-Eijnden Zdeborová "precise asymptotic analysis of learning diffusion models" 2501.03937
35. Kong Ver Steeg "Information-theoretic diffusion" ICLR 2023
36. Shah Chen Klivans "learning mixtures of Gaussians using the DDPM objective" NeurIPS 2023
37. Pierret Galerne "diffusion models for Gaussian distributions: exact solutions and Wasserstein errors"
38. Raya Ambrogioni "spontaneous symmetry breaking in generative diffusion models" NeurIPS 2023
39. Ventura Achilli Silvestri Raya Ambrogioni "manifolds, random matrices and spectral gaps" ICLR 2025
40. Katsevich Bandeira "likelihood maximization and moment matching in low SNR Gaussian mixture models"
41. Perry Weed Bandeira Rigollet Singer "sample complexity of multireference alignment" low SNR
42. Romanov Bendory Ordentlich multi-reference alignment high dimensions sample complexity low SNR
43. Bai "expectation-maximization algorithm for training clean diffusion models from corrupted observations"
44. Kawar Elata Michaeli Elad "GSURE-based diffusion model training with corrupted data"
45. Aali Arvinte Kumar Tamir SURE-Score 2023
46. Cappé Moulines Rydén "Inference in Hidden Markov Models" Fisher identity
47. Douc Moulines Olsson van Handel consistency MLE general HMM Ann. Statist. 2011
48. Poyiadjis Doucet Singh particle approximations of the score Biometrika 2011
49. Allman Matias Rhodes identifiability latent structure models Ann. Statist. 2009
50. Hsu Kakade Zhang spectral algorithm HMM JCSS 2012
51. Rasul et al. TimeGrad ICML 2021; Tashiro et al. CSDI NeurIPS 2021; Kollovieh et al. TSDiff NeurIPS 2023
52. Kazemnejad "impact of positional encoding on length generalization in transformers" NeurIPS 2023
53. diffusion language model / sequence diffusion length generalization extrapolation 2025 arXiv
54. kurtosis "diffusion models" non-Gaussian score fourth cumulant generative 2025 arXiv
55. Cagnetta Wyart "how the structure of language is acquired by deep neural networks" NeurIPS 2024
56. Montanari "sampling, diffusions, and stochastic localization"; El Alaoui Montanari Sellke SK stochastic localization
57. Dytso Poor Shamai "general derivative identity for the conditional mean estimator in Gaussian noise"
58. Mézard / Saglietti / Garnier-Brun 2025–2026 diffusion tree hierarchical belief propagation transformer denoiser
59. Anandkumar Hsu Kakade "method of moments for mixture models and hidden Markov models" COLT 2012
60. Gassiat Rousseau "nonparametric finite translation hidden Markov models and extensions" Bernoulli 2016
61. Guo Shamai Verdú "mutual information and minimum mean-square error in Gaussian channels" 2005
62. diffusion model time series structured prior Markov / Gaussian process / correlated noise 2025 arXiv
63. higher-order statistics / non-Gaussianity / bispectrum diffusion generated samples cosmology lattice cumulants
64. Wu Verdú "functional properties of MMSE and mutual information"; Wu Guo Verdú "derivative of mutual information at zero SNR"
65. Sander Dieleman "Diffusion is spectral autoregression" blog 2024
66. Edgeworth / Gram-Charlier expansion score function diffusion model cumulant correction
67. Shumway Stoffer 1982 EM algorithm state-space
68. Gassiat Cleynen Robin "inference in finite state space non parametric hidden Markov models" 2016
69. Chang "How I warped your noise" ICLR 2024
70. Hyvärinen score matching JMLR 2005; Vincent 2011; Song Garg Shi Ermon sliced score matching UAI 2019
71. Venue checks: "Losing dimensions" / "Bigger isn't always memorizing" / Pham et al. associative memory / Mei–Wu IEEE TIT / Ambient Diffusion Omni / TSFlow / Cui–Pehlevan–Lu
72. WebFetch (arXiv abs or html) for: 2607.08041, 2508.19897, 2505.23652, 2502.00336, 2410.21212 (v2 html), 2605.30095 (v1 html), 2605.06077, 2504.12532, 2510.01378, 2604.07404, 2602.04780, 2605.24192, 2605.04830, 2502.09578, 2508.17689, 2511.03202, 2606.14390, 2505.21777, 2505.16959, 2603.13419, 2506.10038, 2510.12691, 2505.11853, 2412.09726 (abs + html), 2006.15202, 1712.10163, 2502.12089, 2603.03469, 2303.02500, 2509.23937, 2505.17517, 2502.10354, 2602.02908, 2605.30153, 2410.03024, 2511.19657, 2410.02140, 2108.12409, 1803.01271, 2305.00624, 2404.18886, 2309.17082, 2412.01919, 2502.00904, 2104.01883, 2606.02118, 2602.09651, 2502.05743, 2510.24830, 2605.10019, 2605.19391, 2410.14171, 2310.03575, 2307.02129, 2404.18869, 2404.18893, 2310.17467, 1702.08546, 2210.00726, 2306.01993, 2310.05264, 2403.01742, 2208.09399, 2407.01392, 2402.09470, 2211.02590, 2206.13397, 0908.2359.
