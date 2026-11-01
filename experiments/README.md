# experiments — the runs behind every number

One script per registered experiment (`eNN_*.py`), each with `--quick` for a smoke pass, writing CSVs and a
`params_*.json` manifest (commit, dirty-tree flag, config hash, seed, host) into `outputs/<id>/`. The ledger is
`registry/<id>.json` (hypothesis, script, frozen-output hashes) and `docs/EXPERIMENT_REGISTRY.md`.

`frozen_config.py` fixes ρ, the grid, the noise schedule, the replicate count and the iteration budgets in one
place and every experiment imports it, so no run can quietly diverge from what the documents describe. It is
hash-pinned: **do not edit it, and do not edit a script that produced a frozen output** (even a comment makes the
provenance check report a dirty tree; a re-run is a new experiment id, never an overwrite).

Support files: `common.py` (output + manifest writing), `protocol.py`, `configs/` (named run settings),
`exp_22_shape_information.py` and `exp_34_information_ladder.py` (the two legacy drivers E22 and E25 still use).

## The experiments

*paper body / appendix* means the working paper (see `docs/PAPER_CORE_2026-10-20.md`, items B1–B7 and A1–A12) reads
the frozen output; *ledger* means registered, frozen and kept as evidence but not in the paper.

| id | script | role | what it tests |
|---|---|---|---|
| E01 | `e01_information_decay.py` | ledger | H1: Hermite-degree information decay; rates -4 (degree 2) and -8 (degree 4); crossover t_* = 0.5 log lambda_max(Sigma_0) |
| E02 | `e02_capacity.py` | paper body (B4) | H2: on the headline task the mixture kernel's pooled error lies below the LMMSE floor at every M and a C = 1 kernel does not; on the … |
| E02a_v1 | `e02_capacity.py` | paper body (B4) | Robustness of the capacity result to the EM iteration ladder: the protocol-1 ladder (to 400 iterations, capacities to 8) should reproduce the … |
| E03 | `e03_baselines.py` | paper body (B4) | H3/H10/H9: under one recipe, what weight sharing, locality, global propagation and explicit factorisation buy; the information-matched arm; length … |
| E04 | `e04_refinement.py` | ledger | H4: the headline cells move by < 1% at N_g = 801 and < 2% at 1601 (A = 8) and < 1% at A = 10, on the actual fitted kernels and the actual held-out … |
| E05 | `e05_generation.py` | ledger | H5: samplers driven by the fitted kernel reproduce the excess kurtosis, the lag fourth cumulant and the true-law log-likelihood of the data; LMMSE … |
| E06 | `e06_learning_through_noise.py` | ledger | H6: at fixed M the RMSE of rho_hat grows like e^{2t} and that of the fitted excess kurtosis like e^{4t} with the observation level t beyond t_* … |
| E06b | `e06_learning_through_noise.py` | ledger | Corollary (sample complexity by order) in its asymptotic window: at rho = 0.5 the crossover is t_* = 0.55, so the e^{2t} / e^{4t} regime of the two … |
| E06c | `e06_learning_through_noise.py` | ledger | H6c (a budget control, not a new phenomenon): the statement that the estimator leaves the Cramer-Rao bound at the two highest levels is a statement … |
| E07 | `e07_phase_diagram.py` | ledger | H7: the LMMSE floor surface over (rho, family, t) organises the learned estimators; the crossover t_*(rho) locates where the floor collapses |
| E07b | `e07_phase_diagram.py` | ledger | H7b: the family/correlation ladder is converged in the quadrature at the grid it was computed on. Prediction: moving from (N_g, A) = (401, 8) to … |
| E07c | `e07_phase_diagram.py` | ledger | H7b: the family/correlation ladder is converged in the quadrature at the grid it was computed on. Prediction: moving from (N_g, A) = (401, 8) to … |
| E08 | `e08_misspecification.py` | ledger | H8: the structured estimator's advantage degrades gracefully in a measurable misspecification parameter and reverses at a locatable eps*; the … |
| E09 | `e09_capacity_floor.py` | ledger | Capacity resolution (companion to E02a): the infinite-data KL-optimal C-component mixture as a reference for each capacity's error on the identical … |
| E10 | `e10_capacity_evidence.py` | ledger | The thesis's capacity conclusion ('no mixture capacity improves on a single Gaussian innovation', Chapter 10, held-out log-evidence per edge) … |
| E11 | `e11_grid_validation.py` | paper body (B2) | The quadrature reference (N_g = 401 on [-8,8]) is valid across every configuration the project uses, not only at the frozen point where it was … |
| E11b | `e11_grid_validation.py` | paper body (B2) | H11b: the quadrature envelope measured at L = 32 also covers the longer chains the project uses. Prediction: at fixed (family, rho, grid) the drift … |
| E12 | `e12_sample_complexity.py` | ledger | Corollary (sample complexity by order): at observation level t, a fixed accuracy in the covariance coordinate costs M proportional to e^{4t} … |
| E12b | `e12_sample_complexity.py` | ledger | H12 restated with the confound removed: M*(t) at fixed accuracy grows as e^{4t} for the covariance coordinate and e^{8t} for the shape coordinate, so … |
| E13 | `e13_regime_sweep.py` | paper appendix (A9) | The capacity result is a property of the estimator, not of the frozen corner of parameter space. Across correlations and chain lengths: the C = 1 … |
| E13x | `e13_regime_sweep.py` | paper appendix (A9) | As E13 (H13): the capacity contrast C = 8 against C = 1 holds at every (rho, L) of the sweep and the C = 1 arm coincides with the LMMSE of the true … |
| E14 | `e14_regime_baselines.py` | paper appendix (A9) | H14: the information-matched advantage of the structured estimator over screened networks is a property of the method, not of the frozen corner. … |
| E14b | `e14_regime_baselines.py --set family='bimodal' --set rhos=(0.85,) --set lengths=(32,)` | paper appendix (A9) | H14b: the information-matched advantage of the structured estimator is fourth-order structure of either sign, not heavy tails. With the symmetric … |
| E15 | `e15_locality_floor.py` | ledger | The window head's plateau (0.121 -> 0.118 from M = 512 to 2048) is an information ceiling of its radius: the best radius-4 estimator of any kind … |
| E16 | `e16_edgeworth_oracle.py` | ledger | Adding the degree-4 Hermite term of Theorem 3 back to the covariance twin (first Edgeworth correction, oracle Sigma_0 and kappa_4) closes the closure … |
| E17 | `e17_misspec_bias.py` | ledger | The M -> infinity error of EM-BP under each E08 misspecification is the closed-form score error of the KL projection of the noised Gaussian truth … |
| E18 | `e18_closure_wide.py` | ledger | On a domain wide enough to hold the prior's tail every family, Student-t_5 included, reaches the constant (1/6)//kappa_4//^2 of Theorem 3; E01's … |
| E19 | `e19_moment_estimators.py` | ledger | Method-of-moments estimators of rho and of the innovation excess kurtosis from the same one-level noised data as E06 are unbiased and show the e^{2t} … |
| E20 | `e20_evidence_selection.py` | paper appendix (A8) | EM-BP's reported errors do not depend on selecting the checkpoint with the true score: selection by held-out one-view log-evidence gives the same … |
| E21 | `e21_information_curve.py` | ledger | At rho = 0.5 the local slope of log(1/I_eff) on E12b's levels is below the asymptotic 8 (shape) and 4 (correlation) because the expansion parameter … |
| E22 | `e22_message_shapes.py` | ledger | Jerome's message ansatz (docs/JEROME_CALL_2026-09-25.md Q2/Q3/Q4): is a BP message of the form 'a Gaussian pushed through one transition of the true … |
| E23 | `e23_representation.py` | ledger | Jerome's Q5: 'I wonder if we even need a grid in real space... instead of having a grid, we have a truncation on the number of Fourier modes... let's … |
| E24 | `e24_inference_cost.py` | paper appendix (A11) | Inference cost of one posterior mean, measured rather than guessed (docs/JEROME_CALL_2026-09-25.md, Q7). Predeclared scaling prediction for grid BP … |
| E25 | `exp_34_information_ladder.py` | ledger | Jerome's proposal (docs/JEROME_CALL_2026-09-25.md Q6): a small neural network that learns the TRANSITION KERNEL inside exact BP, rather than a large … |
| E26 | `e05_generation.py` | ledger | Jerome's Q8: 'If we have the exact score, or say the exact BP -- so not even a score -- I think we don't even need it to generate. We can just start … |
| E27 | `e27_direct_sampler.py` | ledger | Jerome's Q8 in the STRONG form he actually proposed (transcript 01:19:08-01:19:20): 'The exact score, or say the exact BP, so not even a score, I … |
| E28 | `e28_per_level_ranking.py` | paper body (B5) | Q9's second half. Jerome asked whether the score comparison is at a fixed t or averaged over t; the answer given ('averaged') was correct, the … |
| E29 | `e29_matched_parameters.py` | ledger | The experiment that can refute this project's own supporting claim. The user has ranked the paper's three claims -- structure and sample efficiency … |
| E30 | `e30_joint_ladder.py` | paper body (B4) | The paper's spine is 'a small kernel inside exact BP instead of a large score network'. Three comparisons support it; only two were ever measured on … |
| E31 | `e29_matched_parameters.py` | ledger | E29's own recorded confound, attacked rather than argued. E29 concluded that no architecture reaches em_bp at any capacity up to 135,825 parameters … |
| E32 | `e32_field_oracles.py` | paper body (B6) | The 1D phi^4 field (stochastic Allen-Cahn) at temperature T on a lattice of spacing h is an exactly Markov, stationary, reversible chain whose kernel … |
| E33 | `e33_field_learning.py` | paper body (B6/B7) | On the phi^4 chain the paper's estimator (exact BP around a learned kernel) keeps its advantage over sequence denoisers, provided the kernel family … |
| E34 | `e34_field_physics.py` | ledger | Jerome's consolidated note (30 Sep 2026) names the metrics the field benchmark should carry beside the score error: C(r), correlation length, kink … |
| E35 | `e35_state_augmentation.py` | ledger | The consolidated note (30 Sep 2026, section 8 and figure 7 of its figures-first plan): a global latent makes the observed sequence non-Markov, a … |
| E35x | `e35_state_augmentation.py` | ledger | E35 (frozen 3 Oct, two seeds, a laptop design read qualitatively) at the protocol's sixteen seeds: a global sign latent leaves the paper's … |
| E36 | `e36_stride_protocol.py` | ledger | The consolidated note (section 21.1) recommends generating the field at fine resolution and subsampling systematically; the registered refinement … |
| E36x | `e36_stride_protocol.py` | ledger | E36's clean part (frozen 2 Oct, two seeds) at sixteen seeds, with the MDN's ladder extended so that its clause can be read: E36's P3 third clause … |
| E37 | `e37_domain_distribution.py` | ledger | The consolidated note (30 Sep 2026, section 24.5) asks to 'report at least' the domain-length DISTRIBUTION among the physical statistics; E34 … |
| E38 | `e38_memory.py` | ledger | The consolidated note (section 9.8) lists memory use among what a fair runtime comparison must report; E24 measured time only. E38 measures the peak … |
| E39 | `e39_gaussian_sanity.py` | paper body (B2) | The consolidated note asks for the Gaussian sanity check as a short proposition (section 29 step 3; appendix F item 4) and as figure 2 of its … |
| E40 | `e40_fft_bp.py` | ledger | The consolidated note's section 29 step 5 asks to 'implement FFT-based Gaussian convolution for the current Markov model' (section 9.2: for K(a'/a) = … |
| E41 | `e33_field_learning.py` | paper appendix (A7) | In E33 the MDN kernel's EM ladder (checkpoints 10..1200) ended at its last checkpoint in 140 of 384 risk-selected fits and in 73 of 96 (76%) at M = … |
| E42 | `e33_field_learning.py` | paper appendix (A10) | E33 reads the barrier height from the fitted 5-parameter site energy: within 20% in 14/16 seeds at T = 1 but only 6/16 at T = 2 (M = 2048), and … |
| E43 | `e43_free_kernel.py` | paper body (B6) | The field report's open question to Jerome and Marc: the Gaussian-bond-plus-site-energy kernel is the whole class of discretised gradient SPDEs and … |
| E44 | `e44_quench.py` | ledger | The paper's estimator assumes a stationary Markov chain in space. The physical model's stationary states are Gibbs, but snapshots taken during … |
| E45 | `e45_inpainting.py` | ledger | The note's section 9.7 asks for posterior sampling; the missing-data case was 'cheap, not asked'. A missing site is a site with no likelihood, and … |
| E46 | `e46_symmetry_diagnostic.py` | ledger | E35x (frozen, 16 seeds) found the augmented kernel LOWER than the first-order one at b = 0 (geometric mean 0.82 at M = 512, 0.73 at M = 2048 … |
| E47 | `e47_coarse_grid.py` | paper body (B2) | Jerome called the grid updates the bottleneck of the whole method and improving grid BP 'the most valuable next step'. Every experiment runs at N_g = … |
| E48 | `e36_stride_protocol.py` | ledger | E36x left one clause unresolved by the registered rule: at stride 4 the neural kernel (MDN) beats the site energy, but it selected its last … |
| E49 | `e43_free_kernel.py` | paper appendix (A7) | Jerome's Q6 asked whether the kernel can be a neural network fitted by EM and whether it overfits. On the AR(1) and phi^4 laboratories the answer was … |
| E50 | `e50_speciation.py` | paper appendix (A2/A3) | Jerome's email of 30 Sep 2026 (item 10) and the consolidated note (sections 18-20) ask to use the temperature sweep to study 'speciation': as the … |
| E51 | `e51_kernel_readings.py` | paper body (B7) | Jerome (call of 19 Oct 2026, 00:23-00:28, 01:38): understand exactly how the temperature and the potential are read off the spline, and find out what … |
| E52 | `e52_log_partition.py` | paper appendix (A1) | Jerome (call of 19 Oct 2026, 00:07-00:12): 'the log Z, can we compute it exactly or not?' For the chain, log Z_L = L log lambda_0 + O(1), lambda_0 … |
| E53 | `e53_site_polarisation.py` | paper appendix (A4) | Jerome (call of 19 Oct 2026, 00:29-00:33 and 01:14): follow the local field of each site, the exact posterior E[a_i / x_t] (here its sign … |
| E54 | `e54_field_dsm.py` | paper body (B6) | Jerome (call of 19 Oct 2026, 01:16-01:20) asked for the networks to be trained the standard way (denoising score matching, 'standard DDPM is fine') … |
| E55 | `e55_unet_energy.py` | paper body (neural kernel vs U-Net) | The user (26 Oct 2026), after Jerome's call of 19 Oct (three cases: EM-BP with the right kernel shape, a neural network learning the kernel inside … |
| E56 | `e56_kernel_families.py` | paper body (kernel families) | Jerome's step 1 after the call of 29 Oct 2026 (transcript 37:41, 28:18, 29:29): reproduce thesis Fig. 6.3 for other kernel families K_theta than the … |
| E57 | `e57_chain_families.py` | in flight: kernel families on chains | X2 of the 29 Oct plan (Jerome: 'try other parametric families of kernel on the Laplace case and other cases'). E56 asks whether EM-BP's advantage … |
| E58 | `e58_clean_kernel.py` | in flight: clean-data kernel | The user (31 Oct 2026): 'in the EM step we do not need noisy observations to get the kernel; once we have the kernel we use BP on the noisy … |
| E59 | `e59_forecasting.py` | in flight: forecasting | The user (31 Oct - 1 Nov 2026): a time-series experiment, on something more standard than the Laplace chain. A learned Markov kernel gives the WHOLE … |
