# archive — not in the working paper, kept and restorable

Everything here was live until 1 Nov 2026. Nothing in the live tree imports from it, except that
`tests/test_frozen_config_is_read.py` also scans `legacy-code/` (the frozen configuration fields `ring_*`, `n_heldout`
and `efficiency_sizes` are now read only there). To restore a file: `git mv archive/legacy-code/<path> <path>`; the
tree mirrors the original paths.

| folder | what |
|---|---|
| `legacy-code/src/` | library modules outside the paper: wavelet, video, hierarchy, ring, FID, discrete, local head, sequence nets, spectral, sample metrics, non-Markov chain |
| `legacy-code/experiments/` | the thesis-era `exp_*` scripts (the registered `eNN` experiments stayed live) |
| `legacy-code/tools/` | the generators and gates of the superseded ICML paper (`make_*`, `check_manuscript`, `audit_claims`, `check_paper.sh`, …) and the 2026-09 audit scripts |
| `legacy-code/tests/` | their tests |
| `paper-icml-2026-10/` | the 6 Oct ICML paper (generated fragments, tables, ICML style kit, the NeurIPS manuscript it came from) |
| `docs/` | audits, internal reviews, the final research report, the imported legacy documents |
| `reports/` | briefing, explainer and primer PDFs and sources |
| `notebooks/`, `lean/` | the exploratory notebooks and the Lean 4 audit of the thesis formulas |
| `hpc-legacy/` | the thesis-era SLURM scripts |

Outside the repository, `~/Code/Thesis/archive/` holds the research tree, the thesis snapshot and the defence.
