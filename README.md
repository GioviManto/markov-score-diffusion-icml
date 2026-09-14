# markov-score-diffusion-icml

Private research repository continuing the paper *Recovering Markov dynamics from noised
sequences* toward the next ICML cycle (ICML 2027; call and style kit not yet published
as of 14 September 2026, so the ICML 2026 kit is used provisionally and isolated under
`paper/style/`).

The scientific programme, the audit of the starting point and the running record are
in `docs/`:

| document | what |
|---|---|
| `docs/CURRENT_STATE_AUDIT.md` | forensic reconstruction of the manuscript and its evidence at import |
| `docs/RESEARCH_PLAN.md` | ranked hypotheses and the experiments that test them |
| `docs/THEORY_NOTES.md` | the order-by-order information-decay derivation |
| `docs/CLAIM_EVIDENCE_MATRIX.md` | every claim → assumptions → proof or experiment → output → status |
| `docs/EXPERIMENT_REGISTRY.md` | every registered run: id, script, config, seeds, commit, outputs |
| `docs/NEGATIVE_RESULTS.md` | hypotheses that failed, kept |
| `docs/REPRODUCIBILITY.md` | clone → install → test → Lean → smoke → figures → paper |
| `docs/INTERNAL_REVIEWS.md` | adversarial reviews at each milestone |
| `docs/FINAL_RESEARCH_REPORT.md` | what changed, what survived, what remains |

## Layout

```
paper/            main.tex, sections/ (generated fragments), figures/, style/, legacy-neurips/
src/markov_score/ the library: grid and Gaussian BP, EM/ECM kernels, priors, channel, baselines, theory
experiments/      one script per experiment; frozen_config.py; configs/ and registry/ for new runs
tools/            generators that turn registered outputs into paper/sections and paper/figures; audit scripts
tests/            pytest suite (core: 401 tests); theory and oracle regression tests
lean/             Lean 4 audit of the mathematics (pinned toolchain), README maps propositions to theorems
outputs/          frozen/ (imported evidence), legacy-unfrozen/, registered new outputs; scratch/ ignored
hpc/              SLURM scripts of the archival runs (cluster coordinates from the environment)
notebooks/        the exploratory record, executed and committed with outputs
docs/legacy/      documents imported from the archival repositories
```

## Quick start

```bash
make install        # venv + editable install
make test-fast      # ~1 min: identities, BP/EM correctness, theory checks
make test           # ~5 min: the core suite
make lean           # Lean audit (fetches the mathlib cache once)
make smoke          # a --quick pass of the headline experiment
make tables figures # regenerate every generated fragment and figure from registered outputs
make paper          # paper/main.pdf
make check          # generated fragments vs committed manuscript
```

The archival sources are untouched: `Diffusion/research/nongaussian-bp` (working tree),
`GioviManto/Markov_Score_Diffusion` (public thesis + research snapshot),
`markov-score-diffusion-paper` (the manuscript this repository started from).
