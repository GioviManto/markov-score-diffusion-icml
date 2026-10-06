HANDOVER — Markov score diffusion, four documents
Giovanni Mantovani, 13 September 2026 (supersedes the 12 September bundle)
==========================================================================

For Marc and Jerome. Four self-contained LaTeX projects, each of which
compiles on its own, plus the code and data they were built from (link at
the end).

WHAT CHANGED SINCE 12 SEPTEMBER
-------------------------------
The thesis was condensed for submission. The 230-page build you had is now
the COMPANION COMPENDIUM; the THESIS is a 96-page concise version of it,
written to the university's format and length guidance. Nothing scientific
moved: the same frozen outputs, the same generated tables and figures, the
same headline numbers (6.6-10.9x against the unstructured network under the
schedule-pooled estimand, 2.3-6.1x against the locality-aware baseline), the
same caveats. What the concise thesis dropped is derivation, tutorial
material and the rotating-ring precursor chapter; all of it is in the
compendium, which is the reference for anything the thesis states without
proof.

The earlier technical compendium (114 pp: background derivations,
experiment ledger, claim audit) is superseded and moved to archive/. Its
Chapter 11, the claim audit, lives only there.

THE FOUR DOCUMENTS
------------------
  thesis.zip      "Learning Scores for Markovian Dynamics in Generative
                  Diffusion", 96 pp. The submission: four mandatory blank/
                  dedication pages, contents, half-page AI disclosure, nine
                  chapters, bibliography, one appendix. Latin Modern 12 pt
                  (the same font as the compendium), 2.5 cm side margins,
                  about 29 lines per page. Carries title.tex and
                  abstract.tex for the university system, which adds the
                  title page and ingests the abstract separately. The file
                  COMPRESSION_REPORT.md inside records what was cut, what
                  was kept, and the page-by-page compliance audit.
  compendium.zip  Companion compendium, 228 pp: the extended research
                  record, i.e. the former 230-page thesis with a title page
                  in place of the mandatory front pages. Every derivation
                  written out, the toolbox derivations, the rotating-ring
                  chapter, both original appendices. Chapter and equation
                  numbers differ from the thesis's.
  paper.zip       "Recovering Markov dynamics from noised sequences",
                  29 pp. NeurIPS format; unchanged since 12 September. The
                  BODY ends on page 9, the submission limit; references
                  from page 10, appendix after. Please keep that boundary.
  workshop.zip    The four-page analytic note, 6 pp; unchanged; body within
                  page 4.
  archive/        The superseded technical compendium (folder and zip) and
                  a note. Where the compendium says "the archived technical
                  compendium", it means this.

WHAT IS STILL OPEN
------------------
1. Provenance. Appendix A of the thesis (and of the compendium, in more
   detail) classifies each experiment by how strongly its provenance is
   established, and they differ. Six runs record configuration and seed but
   no source revision, and that group includes experiments the thesis
   cites. Re-running today's code demonstrates today's code; it does not
   certify what produced a frozen output in July. A clean re-run under one
   certified archive is the honest fix and is a cluster campaign, not an
   edit.
2. The AI-use disclosure was rewritten to half a page and now cites the
   three systems used in the bibliography, as the guide asks. The
   acknowledgements page is unchanged and remains a draft.
3. Two chapters carry results at one training size only, flagged in the
   text where they appear.

HOW TO PUT THESE ON OVERLEAF
----------------------------
One project per zip: New Project > Upload Project, once per file, main
document main.tex, compiler pdfLaTeX (the default) for all four. Both the
thesis and the compendium use Latin Modern through the standard fontenc/
lmodern packages; nothing needs a Unicode engine.

All four were built locally with Tectonic; thesis.zip and compendium.zip
were verified by extracting them to an empty directory and compiling from
scratch.

If an Overleaf project for this work already exists, REPLACE it rather than
merging into it: the earlier online copies hold the 203-page or 230-page
thesis and, before the September audit, retired numbers (the 7.0-17.6x
headline and the radius-4 baseline). Merging would mix corrected numbers
with withdrawn ones.

THE ONE RULE FOR EDITING
------------------------
Everything under sections/ is GENERATED from frozen experiment outputs by
scripts, and every such file says so on its first line. A number edited
there is silently undone the next time its generator runs. Change the
generator, not the .tex. The prose cites macros rather than values, so
text, tables and figures cannot drift apart.

THE CODE AND THE DATA
---------------------
  https://github.com/GioviManto/Markov_Score_Diffusion

MIT licensed. The thesis sources are at the repository root, the
compendium under compendium/, the archived technical compendium under
archive/, and the research package under research/, including the frozen
output files the numbers are computed from, so the chain from raw
experiment to printed number can be checked end to end. The tag
extended-thesis-pre-compression preserves the 230-page build exactly as
it was. research/CODE_README.md is the guide; research/REDACTIONS.md
records the one substitution made before publication.
