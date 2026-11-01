# Current-state audit of the manuscript and its evidence

Date: 14 September 2026. Auditor: automated principal-scientist pass over the paper
source (`markov-score-diffusion-paper`, commit `b2a458c`), the research repository
(`Diffusion/research/nongaussian-bp`, HEAD `44d02f8` on
`thesis-scientific-audit-fixes`), the public thesis repository
(`GioviManto/Markov_Score_Diffusion`, which carries a byte-identical snapshot of the
research package with cluster paths redacted) and the Lean audit (`Diffusion/formal`,
identical to the public `lean/`).

The audit is forensic: every claim was traced to the file that supports it, and
generators were re-executed. Where a number could not be found in any stored output
it is said so. Nothing was modified in the archival repositories (one accidental
re-stamp of a params file during regeneration was reverted; git reports both trees
clean).

Sections: 1 what the paper claims · 2 evidence and provenance · 3 strengths ·
4 weaknesses · 5 unresolved ambiguities · 6 exploratory vs confirmatory ·
7 reviewer objections · 8 dead/stale code · 9 irreproducible results ·
10 claims stronger than their evidence · 11 what the audit itself found that is new.

---

## 1. Principal scientific claims of the current paper

The paper (29 pp., NeurIPS 2023 style, single author) is organised as three "rungs".

| # | claim | where |
|---|---|---|
| C1 | For a first-order Markov chain under coordinatewise OU noise the posterior factor graph is a tree, so sum-product gives exact functional marginals and hence the exact score via Tweedie | §3, App. B, D |
| C2 | The grid discretisation (N_g = 401, A = 8) is controlled: truncation and quadrature separated and measured; agreement with the Gaussian closed form to 9.2e-15 and with brute-force enumeration to 1.6e-14 | §4, App. G |
| C3 | Proposition 1: single-Gaussian message closure computes exactly the LMMSE estimator of the covariance-matched Gaussian twin, for every innovation law with the same first two moments | §4, App. F |
| C4 | The closure gap (Figure 1) shrinks with t because P_t Gaussianises and Tweedie's prefactor attenuates sensitivity beyond second moments | §4 |
| C5 | Proposition 2: Fisher's identity gives the exact gradient of the discretised marginal likelihood from one forward–backward pass; EM and gradient ascent reach the same optimum for the Gaussian kernel | §5, App. H, K |
| C6 | Rung 1 (ρ, q) recovery: innovation-variance error falls at slope −0.52 (predicted −0.50); ρ at −0.27, unexplained | §5 |
| C7 | Rung 2 (C = 8 mixture, generalised EM/ECM): a rate asymmetry — ρ settles at median 80 updates, shape at 229 (max 638); two mechanisms (inner latent label; channel) | §6, App. I |
| C8 | Density recovery lags score recovery: score error falls 3.5× while transition Hellinger falls 2× over M = 32→2048 | §6, App. O |
| C9 | Headline (Table 1): EM–BP attains 6.6–10.9× lower schedule-pooled relative score error than a validation-selected MLP at equal clean trajectories, 8 sizes, 16 seeds | §7 |
| C10 | Against a screened weight-shared window head the advantage is 2.3–6.1×, increasing with M; EM–BP wins 192/192 cells; dilated CNN and bidirectional message passing lost the screen | §7, App. N |
| C11 | Rates: EM–BP error slope −0.336 vs network −0.256 in M, difference 0.081 ± 0.013, steeper in all 16 seeds; censoring makes this a lower bound | App. J |
| C12 | Scope: the capacity sweep does not show any mixture capacity improving on a single Gaussian innovation, so the margin is attributed to the Markov factorisation and its covariance, not to non-Gaussian shape | App. L |
| C13 | Misspecification: under a rank-one global latent the advantage falls from 16.4× to 1.06×; under long-range coupling γ the network wins past γ ≈ 0.1 (0.45× at γ = 0.4) | §8 (one sentence); numbers only in RESULTS.md / thesis |
| C14 | Information: the channel removes far more information about shape than about correlation (abstract) | abstract; evidence in thesis ch. 8/10, not in the paper |

---

## 2. Evidence and provenance, claim by claim

The chain `claim → generated macro/table → generator → frozen output → experiment →
sbatch → commit` was reconstructed for every generated artefact. Summary of the
re-execution: **13 of 13 generated section files and 4 of 4 figures regenerate today
and reproduce their numbers exactly** (three differ only in LaTeX presentation; one
gains a column). The macro discipline is genuine.

### 2.1 Generated artefacts

| artefact | generator (`tools/`) | inputs (`outputs/`) | experiment | commit of inputs | dirty tree? | gate |
|---|---|---|---|---|---|---|
| `efficiency-numbers`, `tab-efficiency` | `make_tab_efficiency.py` | `frozen/exp_07_certified_seed{0..15}`, `exp_07_n4096_seed*`, `exp_07_budget2048_seed*`, `exp_07_valsel_seed*`, `exp_07_ref_energy/ref_energy.csv` | `exp_07_em_vs_score_network.py` | `2ae68ac3` (certified), `286b305a` (4096), `54c5ce21` (budget), empty (valsel) | 4096 arm: 8 paths (all wavelet/FID, none on the code path) | `require_clean` **not called**; narrower closure check only |
| `rate-numbers` | `make_rate_analysis.py` | same | same | same | same | **no gate** |
| `tab-aggregation` | `make_aggregation_robustness.py` | same | same | same | same | imports gate, **never calls it** |
| `structured-numbers`, `tab-structured`, `structured-aggregation-numbers`, `tab-aggregation-structured` | `make_tab_structured.py`, `make_aggregation_structured.py` | `frozen/exp_31_confirm_merged/confirm.csv` (three shards) | `exp_31_structured_baseline.py --only confirm` | `569a67c6` | clean, archive digest present | `require_clean` strict ✓ |
| `screening-numbers`, `tab-screening` | `make_tab_screening.py` | `frozen/exp_31_screen/screening.csv` | `exp_31 --only screen` | `569a67c6` | clean | strict ✓ |
| `tab-hellinger` | `make_tab_hellinger.py` | `outputs/exp_07_certified_clean_seed*` (**not under frozen/**) | `exp_07` via `deploy_clean.sh` | `53b95a26` | clean, digest | strict ✓ |
| `convergence-numbers` | `make_convergence_numbers.py` | `frozen/exp_27_{seed16,shard0..3}`, `frozen/exp_06/clean_vs_noised_shape.csv` | `exp_27_shape_convergence.py` | **empty commit field** | — | legacy exemption (disclosed) |
| `grid-convergence-numbers` | `make_grid_convergence.py` | recomputed from source (5 min laptop; bit-identical) | — | `b1cd555a` | 7 paths | none; self-refuses if error > 1e-3 |
| `fig_closure_vs_t` | `make_figures.py fig_closure` | `frozen/exp_02`, `frozen/exp_03` | exp_02, exp_03 | empty | — | none |
| `fig_grid_domain` | `make_figures.py` | `frozen/exp_01`, `frozen/exp_18/boundary.csv` (**laptop rerun**, no commit/host/timestamp) | exp_01, exp_18 | empty / none | — | none |
| `fig_innovation_density` | `make_figures.py` | `outputs/exp_18/innovation_density.csv` (**unfrozen**, different noise schedule from the declared twelve levels) | exp_18 | none | — | none |
| `fig_screening` | `make_figures.py` | `frozen/exp_31_screen` | exp_31 | `569a67c6` | clean | none |

All commits exist in the archival repository. `outputs/frozen` is tracked (869 files).

### 2.2 The three-copy topology

No generator writes into the paper repository. They write into
`Diffusion/Markov_Score_Diffusion/{thesis,paper}/sections`; the paper repository is a
manual third-generation copy. The copy has drifted: `tab-efficiency.tex` in the
generator's target now carries a 95% CI column and a caption reading "four" instead of
"five" aggregations; `efficiency-numbers.tex` now emits two slope-SE macros the paper
lacks; and `tab-aggregation.tex` in the paper carries `\label{app:aggregation}` which
the generator no longer emits, so **re-running the generator into the paper breaks the
build** (undefined reference). `tools/check_paper.sh`, the manuscript gate, points at a
directory deleted on 17 August and has never run against this paper.

### 2.3 Typed literals

Of 37 literals typed in `main.tex`/`appendix.tex` rather than drawn from macros,
22 are supported by a stored output or by arithmetic. The rest:

| literal | location | status |
|---|---|---|
| underflow `0.27%` at production grid | App. G | **contradicted**: regenerated max is 0.29% (`fitted-narrow` family); the correct macro `\gczerofrac{0.29}` exists and is unused |
| cap-hit `6%`, `12%`, `61%` | App. J | **unsupported**: measured 7.3%, 16.1%, 25.5% (the 61% does not reproduce under any definition; 56% is the M = 2048 value) |
| `84×` worst/p90 boundary ratio | App. G | approximate (85.6 measured) |
| restricted-range slope difference `0.08` "in both cases" | App. J | one case generated (0.063), the other not |
| two-point slope difference `0.26 ± 0.14` | App. J | **not computed by any tool** |
| early stopping costs `5–36%` of the ratio | App. K | directionally supported, endpoints not reproducible |
| `9.2e-15`, `1.6e-14`, `1.8e-15` (validation vs closed form / brute force) | §4 | **not in any output**; only in docstrings; one docstring says 1.0e-14 for the same check |
| LMMSE departs by `0.65` at `E[a] = 1.3` | §4 | **not found anywhere** in code, tests or outputs |
| gradient ascent reaches EM to `1.6e-9` nats | App. H | **contradicted**: `frozen/exp_08` smallest gap is −1.7e-10 |
| kurtosis `28%` below at four inner sweeps | App. H | not computed by any tool |
| capacity control `24 configurations, 3.2k–297k` | App. K | no `capacity.csv` under frozen |

### 2.4 Lean

`Diffusion/formal` (= public `lean/`): 12 modules, 21,750 lines, Lean 4.33.0 + Mathlib,
builds green from cache in 10 s, no `sorry`/`axiom`/`native_decide` (grep-verified).
480 audited display items of the **thesis** (not the paper): 336 proved, 13
instance-checked, 125 definitions, 4 not formalised (Itô calculus absent from
Mathlib), 2 discrepancies (both minor, already corrected in the thesis). The paper's
Proposition 1 algebra is covered by `ThesisAudit/Ch07.lean` (Laplace chapter) and
Proposition 2's discrete identity by `Ch08a/b`; the mapping from *paper* propositions
to Lean theorems has not been written and is a deliverable of this project.

### 2.5 Tests and environment

`pytest` on the core suite (video/wavelet suites excluded): 401 passed, 27 skipped
(CuPy parity, no GPU), 4:48 on the M1 laptop, Python 3.12.5 / NumPy 2.5.2. Cluster
records say 3.13.11 / 2.5.1; regeneration reproduced under both.

---

## 3. Current strengths

1. **A genuinely exact non-Gaussian score reference.** Tree-exactness plus a
   controlled grid gives a validated target across a twelve-level schedule for a
   correlated, non-Gaussian model; the three levels of exactness are kept apart in
   the text (App. C). This is rare and valuable.
2. **Proposition 1** turns the Gaussian-closure gap into an interpretable quantity
   (the price of second-order closure). It is proved directly, and its hypotheses are
   examined.
3. **Fisher-identity learning from noised data only** with a one-view-per-chain
   marginal likelihood; the estimator never sees a clean trajectory.
4. **Provenance discipline.** Generated macros, frozen outputs, per-run params with
   commit and job ids, a provenance gate with a sharp "dirty on the code path"
   criterion, a withdrawn M = 8192 row on exactly that criterion, the retired
   cellwise estimand kept for the aggregation appendix.
5. **Symmetric validation selection** of both arms' budgets; multiple aggregations
   reported; paired seeds as the inferential unit.
6. **A structure-aware baseline screened on disjoint development seeds** and a
   confirmatory run on untouched seeds.
7. **Formal audit of the thesis mathematics** with an honest verdict table.
8. **Honest reporting of asymmetries** (information interface, capacity attribution,
   censoring).

---

## 4. Current weaknesses

### 4.1 Scientific

W1. **No general insight beyond "BP/EM works on a chain".** The paper's own framing
    ("What is not new here") concedes the estimator is classical; the contribution is
    the laboratory. An ICML reviewer will ask what the laboratory taught us.

W2. **The capacity attribution (C12) is inconsistent with the paper's own numbers.**
    A direct computation (this audit, `docs/AUDIT_ORACLE_LADDER.md`) shows that on
    the Laplace chain the covariance-matched LMMSE oracle *with the true ρ* has
    schedule-pooled relative score error **0.168**, whereas Table 1's EM–BP reaches
    0.090 (M = 32) down to 0.018 (M = 4096). A single-Gaussian innovation kernel
    cannot beat the LMMSE floor, by Proposition 1. So the C = 8 mixture *is*
    recovering non-Gaussian structure, and the sentence "the margin is earned by the
    Markov factorisation and its covariance, and not by the mixture's capacity" is
    wrong as a statement about the headline task. The capacity sweep that produced
    it was scored on a different bundle and region (paper's own admission); see §11
    for the reconstruction pending from `exp_32`. This is the single most damaging
    self-inflicted weakness in the manuscript: a hostile reviewer takes App. L at
    face value and concludes the density learning is decorative.

W3. **The information-decay claim (C14) is asserted in the abstract but its evidence
    is pre-asymptotic and not in the paper.** `exp_22` measures efficient information
    for (ρ, q, β) over t ∈ [0.05, 1.6] with one seed per cell and no error bars; the
    thesis's own boxed prediction (I_ρ ∝ e^{−4t}, I_shape ∝ e^{−8t}) is met by
    measured slopes of −3 and −6, described there as "shallower by a common offset".
    This audit finds (i) the deficit is a common multiplicative factor 0.76 on the
    rate, not an offset; (ii) the schedule stops at t = 1.6, below the crossover
    t_* = ½ log((1+ρ)/(1−ρ)) ≈ 1.26 at ρ = 0.85 where the asymptotic rates set in;
    (iii) the consecutive-interval slopes of the independent closure-gap experiment
    `exp_02` steepen monotonically toward the predicted values (−2.9, −4.5, −5.7 for
    the posterior-mean MSE → −6; −5.5, −6.5, −7.5 for 1 − cos → −8). The theory is
    right and the experiments were stopped too early to show it.

W4. **The neural screen was not fair to the long-range architectures.** In
    `exp_31`, the dilated CNN and bidirectional message-passing arms draw *one* noise
    level per gradient step while the window head draws a per-example level from all
    twelve (`src/seq_nets.py:348` vs `src/local_head.py`); the window head's
    checkpoint ladder is eight independent retrainings (selection over eight
    initialisations) while the others are snapshots of one run. The conclusion "the
    long-range arms did not lose for want of capacity" may be an optimisation
    artefact.

W5. **A recorded hyperparameter is wrong.** `exp_31` never passes `lr=hp["lr"]` to
    `train_local_head` (line 279–293), so every window row records `lr = 0.003`
    while training at the default 0.002, unclipped. The paper's screening table
    reports `lr 0.003`.

W6. **No information-matched comparison exists.** All neural arms regress on clean
    targets with fresh corruption each step; nothing learns from P_t samples alone.
    The paper says so, but says nothing else.

W7. **No generation result.** The reverse sampler, metrics and a converged rerun
    (`components_converged_*`, `em_iters = 2000`) exist and show the fitted-kernel
    arm reproducing the target innovation kurtosis at every C (1.83–1.94 vs 1.91),
    which dissolves the "capacity effect" the withdrawal cited; but no cell is
    provenanced (62/74 params lack a commit; 12 carry a dirty tree), the network arms
    were never reconverged, and the step-count "convergence study" uses independent
    Brownian paths per resolution (it is a robustness check, not a convergence
    study; the fix `nested_brownian_path` exists and is unwired).

W8. **Misspecification is one sentence in the paper.** The thesis has the numbers
    (rank-one latent, long-range coupling) against the MLP only; not against the
    window head; no measurable ε → advantage curve in the paper.

W9. **Single operating point.** ρ = 0.85, L = 32, Laplace, one schedule. Other
    families appear only in the closure figure.

W10. **Rates are descriptive and partly censored** (C11). The EM slope −0.336 is
     acknowledged as un-derived; the ρ slope −0.27 at Rung 1 is unexplained.

### 4.2 Provenance and presentation

P1. The paper is a manual copy that has drifted from its generators; regenerating
    breaks the build (§2.2).
P2. Seven typed literals unsupported or contradicted (§2.3).
P3. The headline table, rate macros and aggregation appendix bypass `require_clean`;
    `certify()` is called by no generator; the headline sources would fail the gate
    on the missing archive digest (legacy runs).
P4. `exp_07_ref_energy` (the headline denominator) has no params file.
P5. `frozen/exp_18` is a laptop rerun with no commit/host/timestamp, placed under
    `frozen/`; the density figure reads an unfrozen sibling with a different schedule.
P6. The tabulated screen (job 635628, 28 Aug) is not the execution whose CSV
    selected the confirmatory baseline (24 Aug run); same commit and archive digest,
    identification confirmed by the parameter count 4801.
P7. `exp_31_confirm_merged` was produced by three shards with three config hashes,
    not by the committed sbatch.
P8. Prose is thesis-like: long defensive qualification, repeated "what is not
    claimed" paragraphs, appendix material in the body (App. C is a discourse on the
    word "exact"), 29 pages for a 9-page body.

---

## 5. Unresolved ambiguities

A1. Whether capacity pays *on the headline metric*: the oracle-ladder computation
    says it must; `exp_32` said it does not; the two used different bundles/regions.
    Must be settled by scoring C ∈ {1, 2, 4, 8, 16} on the headline bundle and region.
A2. The Rung-1 ρ slope of −0.27 (vs −0.5): sampling floor, grid floor, or a genuine
    finite-M effect of the one-view protocol (information about ρ at high t is
    ∝ e^{−4t}, so most of the twelve levels contribute almost nothing and the
    effective M is much smaller than the nominal one).
A3. Whether the window head's advantage over the long-range arms survives a fair
    screen (per-example noise levels, equal restarts, applied learning rate).
A4. What "value per clean trajectory" means once an information-matched arm exists.
A5. Whether score accuracy transfers to generative accuracy; the converged rerun
    suggests yes for kurtosis, but with no metric beyond kurtosis, histogram KL and
    covariance error.
A6. The ρ–q collinearity confound in the information decay: about one unit of the
    efficient-ρ slope is collinearity growth (corr_{ρq}: 0.15 → 0.90), not channel
    loss.

---

## 6. Exploratory versus confirmatory

| experiment | status |
|---|---|
| exp_07 certified (Table 1), exp_31 confirm (Table 3), exp_31 screen (Table 2), exp_07 certified_clean (Hellinger), exp_27 (settling), exp_06 rate16/clean16 (Rung-1 rates) | confirmatory, validation-selected, 16 seeds |
| exp_01/02/03/18 (grid, closure) | deterministic diagnostics, confirmatory in nature |
| exp_22 (information), exp_06 price_of_noising | exploratory: one seed per cell, no error bars, pre-asymptotic schedule |
| exp_32/33 (capacity), exp_16 (generation), exp_12 (receptive field), exp_21/04 (misspecification), exp_34 (MDN ladder), exp_28/29 | exploratory or withdrawn; exp_12 part 1–3 select on the test set by design; exp_21 uses the MLP only |
| exp_07 part 2 (network capacity sweep) | pure test error; the headline MLP width (128,128) has no recorded selection rationale |

---

## 7. Likely reviewer objections (ranked)

R1. "Classical HMM + EM on a synthetic chain; what is the ML insight?" (W1)
R2. "Your own Appendix L says the non-Gaussian learning contributes nothing; then
    the paper is about covariance." (W2 — and the objection is *wrong*, but the paper
    invites it)
R3. "The comparison gives EM–BP the factorisation and the networks the clean data;
    it ranks nothing." (W6)
R4. "The long-range baselines were undertrained." (W4, W5)
R5. "Pointwise score error is not generation." (W7)
R6. "One ρ, one L, one family." (W9)
R7. "Misspecification is a sentence." (W8)
R8. "The information-decay claim in the abstract has no theorem and no figure." (W3)
R9. "Formally verified? The Lean audit is of a thesis, not of this paper." (§2.4)

---

## 8. Broken, stale or dead code

- `tools/check_paper.sh`: hard-coded path to a deleted directory; exits 1.
- `tools/make_figures.py fig_closure_vs_t`: KeyError (dispatch key is `fig_closure`).
- `experiments/exp_31_structured_baseline.py:279–293`: window arm ignores `lr`,
  `batch_size`, `grad_clip`.
- `experiments/exp_31_structured_baseline.py` `part_confirm`: no guard that
  `seed0 + seeds ≤ 10000` (dev-seed overlap possible under `--set`); progress print
  re-scores the test bundle twice.
- `experiments/exp_16_sampling_validation.py` `run_steps`: independent noise per
  resolution; `src/reverse.nested_brownian_path` unwired.
- `experiments/exp_05_reverse_dynamics.py`: compares samples at t_min against clean
  p_0 (wrong target; fixed in exp_16 but exp_05 never rerun); duplicates the
  integrator.
- `src/fid.py`: docstring/return mismatch; `exp_30` never run; no data dir.
- `outputs/CLUSTER_JOBS_14_15_AUG.md`: stale (says 1/12 converged cells ran; 12 did).
- `outputs/exp_16_summary/generation_rerun_summary.csv`: hand-built, no generator.
- 31 of 97 generated macros unused; several values typed by hand beside them.
- Adam re-implemented three times; no autodiff; adding any architecture requires a
  hand-derived backward pass.
- `frozen_config.em_shape_tol` and one other field declared but never read
  (disclosed in the paper).

---

## 9. Results that cannot be reproduced as recorded

- The M = 8192 row (withdrawn by the paper; 207 dirty paths, 3 on the code path).
- Every `exp_16` cell (no commit or dirty tree).
- `exp_22` and `exp_27`/`exp_06_clean16` carry empty commit fields (legacy runs
  before the stamp existed); reproducible by intent, not by record.
- `frozen/exp_18` (laptop, no stamp).
- The typed literals of §2.3 marked "not found".
- `exp_31_confirm_merged` from the committed sbatch (three shards were used).

---

## 10. Claims stronger than their evidence

- Abstract: "the channel … removes far more information about shape than about
  correlation" — true in direction, but evidence is one-seed, pre-asymptotic, in a
  companion document.
- App. L: "the margin over the network arms is earned by the Markov factorisation and
  its covariance, and not by the mixture's capacity" — contradicted by the LMMSE floor
  (0.168 pooled) versus Table 1 (0.018–0.090).
- App. N: "the long-range arms did not lose for want of capacity" — true, but they
  may have lost for want of a fair optimisation protocol.
- §4: validation figures 9.2e-15 / 1.6e-14 / 1.8e-15 — asserted with no stored run.
- App. H: "1.6e-9 nats" — contradicted by the stored output.
- Abstract/§7: "increasing across the sizes tested rather than falling" for the
  structured ratio — four points; the appendix itself says it cannot separate a floor
  from a slow decay.

---

## 11. What this audit established that is new

N1. **Oracle ladder on the Laplace chain** (512 chains, twelve levels, N_g = 401):
    schedule-pooled relative score error of the isotropic score −x is 0.670; of the
    LMMSE oracle (true ρ) 0.168; of true-kernel BP 0 by definition. Same for
    Student-t(5): 0.158; uniform: 0.149; bimodal: 0.082; Gaussian: 0 (1e-15). So the
    MLP at M = 4096 (0.179) sits at the LMMSE floor, the window head at M = 2048
    (0.127) slightly below it, and EM–BP (0.026 at M = 2048) far below it.
    **Learned unstructured networks at these sizes recover second-order structure and
    little else; EM–BP recovers the higher-order structure.** This reverses the
    interpretation of App. L.

N2. **High-noise rates** (scalar, fine quadrature, `scratchpad/scalar_exponents.py`):
    squared score gap to the Gaussian ∝ e^{−8t} with constant κ_4²/6 (measured
    1.39–1.55 vs 1.5 for Laplace); shape Fisher information ∝ e^{−8t}; variance
    information ∝ e^{−4t} with constant ½ (measured 0.497–0.501). See
    `docs/THEORY_NOTES.md`. In the chain the approach to these rates is governed by
    e^{−2t} λ_max(Σ_0), which explains every "shallower than predicted" slope in the
    frozen record.

N3. **The withdrawn generation comparison is recoverable**: the converged EM cells
    already reproduce the target kurtosis; what is missing is provenance,
    reconvergence of the network arms, a paired-noise step study and distributional
    metrics (none of Wasserstein/MMD exists in the code).

N4. **Local feasibility**: grid BP costs 0.33–0.6 ms per chain per level at
    N_g = 401 on the M1; one E-step over 256 chains 0.12–0.2 s; a 2000-chain,
    400-step reverse trajectory bundle ≈ 9 min per BP arm. The cluster is not
    reachable from this session (hostname does not resolve; VPN). Every new
    experiment in the plan is sized to run locally on 8 cores / 8 GB.

N5. ICML 2027 has not published a call or style kit (checked 14 Sep 2026: icml.cc
    2027 pages 404; the ICML2027 style URL 404; the only "ICML 2027 CFP" hit is the
    International Conference on Minority Languages). The ICML 2026 kit
    (`icml2026.zip`: 8-page main body at submission, unlimited appendix, mandatory
    impact statement, anonymised code) is adopted provisionally with the style files
    isolated under `paper/style/` for a one-line migration.

N6. **Mechanism of the capacity artefact (from the exp_32 record).** In
    `frozen/exp_32_capacity_merged` the validation rule selected only the
    *initialisation*; every capacity ran to a fixed 1200-iteration endpoint unless the
    10⁻⁹ likelihood tolerance fired first. It fired in 31/32 C = 1 cells (median 71
    iterations, essentially at the validation optimum that exp_07 measures at 40–80
    for these sizes) and in 0/96 cells with C ≥ 4, which therefore report checkpoints
    ~15× past their optimum — a regime exp_07 shows costs 1.1–1.9× in pooled risk.
    "C = 1 wins" is "C = 1 is the only arm that stopped in the right place". The
    project's own `NOTES_exp34` and `README_exp32_capacity.md` say the question is
    unresolved; the thesis chapters state it as resolved. Independent traces
    (`exp_16/cpoint_*`: EM–BP error 0.054 → 0.021 monotone in C ∈ {2…16} while the
    networks are flat; `exp_16/family_*`: EM–BP error tracks innovation heaviness
    while the networks are family-blind; the converged generation cells: the MLP
    generates innovations with excess kurtosis 0.03 against 1.97 for the exact arm)
    all point the same way. A single-Gaussian EM–BP would also lose to the screened
    window head (floor 0.13 < 0.168) at every M ≥ 128.

N7. **The misspecification table quotes the wrong quantity.** `exp_21` writes the
    relative *posterior-mean* error into a column named `score_rel_l2`; the ratios are
    unaffected (the α/Δ prefactor cancels), but the "EM–BP error, worst t" row and the
    thesis's "58%" at γ = 0.4 are posterior-mean errors, not score errors (the true
    worst-level score error is 34%, at t = 0.1, the opposite end of the schedule). The
    exp_21 network arm is an unscreened radius-6 local head with a fixed budget, not the
    screened window head, so the break-even γ ≈ 0.1 is an upper bound. exp_21 has one
    data draw per cell (no error bars) and five noise levels (a thesis caption says
    twelve). No exact reference exists for a long-range-coupled Laplace chain, so the γ
    sweep is Gaussian-only.
