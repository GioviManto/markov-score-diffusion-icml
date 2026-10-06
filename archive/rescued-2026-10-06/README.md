# Rescued from the Trash, 6 October 2026

On 6 October 2026 (~23:21) three items went to the macOS Trash: `~/Code/Thesis/markov-score-diffusion-paper`,
`~/Code/Thesis/Overleaf_Upload`, and (earlier, undated) `overleaf.zip`. Before anything is emptied, every file in
them (856 after removing LaTeX build files and `__MACOSX`) was fingerprinted (SHA-1 of the contents) against the
copies that are safe on GitHub: the Overleaf export `GioviManto/markov-score-diffusion-overleaf`, the thesis repo
`GioviManto/Markov_Score_Diffusion`, and this repository. What matched nowhere is kept here, byte-identical to the
originals (each copy re-checked against the Trash).

## What was already safe, and where

- **`markov-score-diffusion-paper`** (the NeurIPS manuscript the work started from; a local repository with no remote,
  one commit `b2a458c` "Recovering Markov dynamics from noised sequences", nothing uncommitted): all 30 of its files are
  byte-identical to `paper/legacy-neurips/` in this repository. Nothing to rescue.
- **`Overleaf_Upload`** (the 13 September bundle of four Overleaf projects): everything is in the Overleaf project
  *Markov_Score_Diffusion* and its GitHub export, in the thesis repo, or in this repository, including the superseded
  114-page technical compendium in `archive/`, EXCEPT the files under `overleaf-upload-2026-09-13/` below.
- **`overleaf.zip`** (the 2 September bundle): 245 of its files are on disk in `~/Code/Thesis/Diffusion` and 44 more in
  that repository's git history, EXCEPT the files under `overleaf-bundle-2026-09-02/` below. Two generated files that
  matched nothing (`thesis/main.lof`, `main.lot`) were not kept.

## What is here

| file | why it is unique |
|---|---|
| `overleaf-upload-2026-09-13/README.txt` | the handover note to Marc and Jérôme for the four-document bundle (identical to the `HANDOVER.txt` that every project folder carried) |
| `overleaf-upload-2026-09-13/thesis/chapters/ch04-gaussian-bp.tex` | the bundle's version; the live Overleaf chapter has a different opening (8 lines) |
| `overleaf-upload-2026-09-13/thesis/chapters/ch08-comparison.tex` | the bundle's version carries the section "How much mixture capacity the estimator needs" (`sec:em-capacity`), absent from the live Overleaf chapter |
| `overleaf-bundle-2026-09-02/thesis/chapters/ch10-nongaussian-em-method.tex` | an earlier draft chapter found nowhere else |
| `overleaf-bundle-2026-09-02/thesis/chapters/ch11-nongaussian-em-results.tex` | an earlier draft chapter found nowhere else |
| `overleaf-bundle-2026-09-02/thesis/chapters/appJ-aggregation.tex` | an earlier draft appendix found nowhere else |
| `overleaf-bundle-2026-09-02/thesis/main.tex`, `preamble.tex` | the 2 September thesis driver and preamble (86 and 134 lines from the submitted ones) |
| `overleaf-bundle-2026-09-02/compendium/main.tex` | the 2 September compendium driver |

These are records: nothing in the build reads them.
