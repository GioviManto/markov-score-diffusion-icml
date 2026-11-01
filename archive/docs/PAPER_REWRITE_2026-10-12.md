# Paper rewrite, 12 October 2026: what was asked, where it is

The paper was restructured to follow Jérôme's and Marc's suggestions (the call of 25 September, his email of 30 September, the
consolidated note of 30 September: its section 27 "A coherent paper structure" and section 28 "Figures-first plan") and to carry the
figures-first set. Length was deliberately not controlled: the cut comes afterwards (see "Cutting" below).

## The new structure (paper/main.tex)

| § | file | content | note's §27 step |
|---|---|---|---|
| 1 | `body/01-introduction.tex` | structural thesis, the two laboratories, contributions, **Figure 1 = summary figure** | 1 |
| 2 | `body/02-setup.tex` | a **general first-order kernel K**; the linear chain with non-Gaussian noise as the first instance, the φ⁴ field as the second | 1 |
| 3 | `body/03-inference.tex` | Tweedie, the tree posterior, BP, what "exact" means (quadrature stated), sampling; **Figure 2 = method schematic** | 2 |
| 4 | `body/04-sanity.tex` | Gaussian case: Prop. 1 (closure = LMMSE), **Prop. 2 (BP = Kalman smoother)**; **Figure 3 = Gaussian sanity check** | 4 |
| 5 | `body/04-learning.tex` | EM through noise, Fisher's identity, the kernel families (mixture, tilted walks, site energy, unrestricted, neural) | 3 |
| 6 | `body/05-experiments.tex` (+ `03-theory`, `05a…05d`) | the controlled non-Gaussian laboratory: what noise erases (**the Fisher/Hermite theory, demoted to a supporting subsection**), learning through noise, capacity, networks, **Figures 5–6 = non-Gaussian gain, sample efficiency**, generation | 5, 7 |
| 7 | `body/06-field.tex` | the φ⁴ benchmark: cells, temperature sweep (**Fig. = field sweep**), scoreboard, the fitted kernel as physics, physical statistics, refinement and runtime (**Fig. = accuracy vs cost**), speciation, missing data, quench | 6, 7 |
| 8 | `body/07-compute.tex` | cost, the coarse grid, one thread vs eight, why not Fourier modes | 8 |
| 9 | `body/08-limits.tex` | kernel class (unrestricted/neural kernels), global latent and its repair (**Figure = misspecification repair**), Gaussian departures, what is not claimed | 9 |
| – | `body/06-related.tex`, `body/07-conclusion.tex` | related work (+ physical Markov models), conclusion | |
| A | `body/A1…A5`, **`body/A6-field.tex`** | proofs, numerics, protocol, additional results, audit; **new: the φ⁴ field in detail and the cost tables** | |

Numbers come only from generated macros (`paper/sections/field-numbers.tex`, `tools/make_paper_numbers.py`, plus the older fragments) and
generated tables (`paper/sections/tab-field-*.tex`, `tab-cost-*.tex`); `make check` regenerates and diffs them, and `tools/audit_claims.py`
recomputes the headline ones independently.

## Jérôme's asks and where they went

| ask | where |
|---|---|
| Main message about **structure, not Fisher information**; "more non-Gaussian data is destroyed faster" is generic, do not over-emphasise | Intro: the structural thesis leads; the Fisher/Hermite theorems are a supporting subsection (§6, "What noise erases") |
| Contribution 1: exact optimal denoising via BP for **any Markovian distribution**, even without a closed-form score | §2 (general kernel K), §3, Intro contribution (i) |
| Contribution 2: the **Gaussian case as a sanity check** for any BP approximation | §4, Prop. 2, Fig. 3 |
| Contribution 3: **EM makes it trainable, and works very well** | §5, §6 (learning through noise, networks), §7 |
| **Frame the theory neutrally**, AR(1) as the first concrete Markov example (email 30 Sep, item 4) | §2: the kernel is general; the linear chain is "first instance" |
| A **more concrete example than AR(1)** in the paper (email 30 Sep, item 5): the φ⁴ field | §7, A6 |
| **Figures first**: 5–6 key figures and a **summary figure** at the start (action items 1–2) | Fig. 1 (summary), 2 (method), 3 (Gaussian sanity), gain-per-level, sample efficiency, field sweep, accuracy-vs-cost, misspecification repair; the **key six** are listed under "Cutting" |
| Add a **short statement/proof that Gaussian closure = Gaussian score**, and that the Gaussian forward–backward sweep is a **Kalman smoother** (action 5) | Prop. 1 and Prop. 2 in §4 (proofs in A1) |
| **Harmonise notation**; introduce K explicitly (action 6) | `paper/notation.tex`, `docs/NOTATION.md`; §2 introduces K before any use |
| **Report inference time** of EM-BP vs. networks (action 7) | §8 and Fig. 'accuracy vs cost'; thread count with every number |
| Drop the **AR(1)/"innovation"** terminology (action 8) | "linear chain", "driving noise" throughout |
| **Fourier modes** for BP messages (action 3; Jérôme doubted it) | §8 "Why not Fourier modes": measured premise (E23), FFT transition (E40), the 12–15 % ceiling (E24x) |
| **Laplace / Gaussian+Laplace ansatz** for messages: is it exact? | §8: no, exact at site 1 only (E22) |
| **Neural-network kernel inside BP**, "avoid an obvious result" | §5 (MDN kernel), §9: it earns its parameters only where the simple family is misspecified (E43, E48, E49) |
| **Sampling without diffusion** | §3 (sampling), §6 (generation): one forward sweep |
| Misspecification: a **global latent** leaves a bias | §9, Fig. 'misspecification repair' (E35x, E46) |
| **Speciation / commitment** across the temperature sweep (email item 10) | §7 "Speciation", Fig. speciation (E50) |
| **Discretisation**: h vs the correlation length; fixed-window networks fall behind (item 11) | §7 "Refinement" (E33 P5 and the radius screen) |
| The note's §26.2 **claims to avoid** | §9 "What we do not claim" |
| Fix the labels of the **information plot** (2nd and 4th cumulant, not ρ and β) (action 4) | Fig. 'what survives noise' panel (b), redrawn in cumulant coordinates |

## Cutting (to do, by the author)

The user asked for no length control; the manuscript is now far over the ICML body limit. A suggested cut, in order:

1. **Key figures** (Jérôme's "5–6" plus the summary): summary (Fig. 1), method (Fig. 2), Gaussian sanity (Fig. 3), per-level gain, sample efficiency,
   the field temperature sweep, accuracy vs cost. Everything else can go to the appendix.
2. **Section 6**: keep capacity and the matched comparison in two paragraphs; move the family, phase and regime sweeps and the learning-through-noise
   figure to the appendix.
3. **Section 7**: keep the scoreboard, the physics reading and refinement; move physical statistics, speciation and missing data to A6.
4. **Section 8**: keep the cost paragraph and the coarse grid; the Fourier paragraph can be three sentences.
5. **Section 9**: keep kernel class and the latent; Gaussian departures to the appendix.

Page budget: the ICML 2026 kit allows 8 pages at submission (the Overleaf/NeurIPS copy was 9); `overleaf/build.sh` is not used on this branch.

## Provenance

Every field result in the new text is read from a frozen, registered output (E32–E50; `docs/EXPERIMENT_REGISTRY.md`); the three figure/number
generators are `tools/make_paper_numbers.py`, `tools/make_fig_coarse_grid.py`, `tools/make_fig_speciation.py`, and the copies of the report's figures
are kept identical by `tools/sync_paper_figures.py` (`tests/test_paper_figures_synced.py`).

## Readings the paper takes that differ from a registry sentence (all deliberate, all in the macro definitions)

`tools/make_paper_numbers.py` carries the source file and the statistic of every field/cost macro (`DEFS`; `reports/numbers/paper_numbers.csv`).
Four of them are worth knowing because a reader comparing the paper with the registry will meet them:

| macro | paper | registry sentence | why |
|---|---|---|---|
| `lvlCrossA/B` | 0.98 / 2.07 | E28: median 0.94 / 1.90 | the paper reads the first of the twelve schedule levels at which the Gaussian score beats EM-BP (no interpolation, median over seeds); the registry's interpolation is not specified and could not be reproduced by five variants |
| `limMdnLinearLo/Hi` | 2.1 / 2.9 | E30: 2.12 / 2.91 | same statistic: pooled over seeds and levels, sqrt(sum sq_err / sum sq_ref) |
| `cmpFftSlowerLarge` | 2.2 | (E40 follow-up said "breaks even near 3 200") | the follow-up output is not frozen, so the paper states only the frozen E40 value: FFT-BP is still 2.2 times slower at N_g = 3201 |
| `cumDecayRatio` | 717 (not 164) | | the 164 is floor-limited; the text quotes the safer value at t = 3 |

`tools/audit_claims.py::audit_field` recomputes twenty of these headline macros from the frozen CSVs, without importing the generator.

## Known caveats

* Under Tectonic (XeTeX) the ICML kit prints "Title Suppressed Due to Excessive Size" as the running head (the old build did the same): the kit's height test on
  the running title fails with the substituted font. pdfLaTeX on Overleaf is unaffected. Not a content problem.
* `outputs/derived/E01/cumulant_information.csv` is not frozen; `tools/make_fig_theory.py` recomputes it in memory from the frozen
  `shape_information.csv` and aborts if the committed CSV differs.
* The manuscript is 39 pages (body about 22) against the 8-page ICML body limit; the cut is the author's (see "Cutting").
* The ICML header/line numbers in the PDF are the review-mode kit's.
