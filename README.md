# Learning a Markov kernel through the noise

Research code for the working paper on **belief propagation (BP) around a learned Markov kernel**:
a diffusion process destroys the Markov structure of a sequence at intermediate noise levels, but the exact score is
still computable by BP if the chain's kernel is known — and the kernel can be learned by EM from the *noisy* sequences
alone. A kernel with a few parameters then beats a neural network trained on far more data, and on the stochastic
Allen–Cahn (φ⁴) field its parameters read as physics (temperature, double-well potential).

The paper itself is written by hand in a separate Overleaf-synced repository
(`GioviManto/markov-score-diffusion-icml-paper`). This repository holds the code, the experiments and the frozen
evidence behind every number in it.

## Read the code in this order

| step | file | what you learn |
|---|---|---|
| 1 | [bp_minimal.py](src/markov_score/bp_minimal.py) | the whole algorithm in 30 lines: sum-product on a chain, messages on a grid |
| 2 | [test_bp_minimal.py](tests/test_bp_minimal.py) | it equals the production BP to 1e-10 and the closed form on the Gaussian chain |
| 3 | [noising.py](src/markov_score/noising.py), [priors.py](src/markov_score/priors.py) | the Ornstein–Uhlenbeck channel and the clean chains (Gaussian, Laplace, mixture) |
| 4 | [bp_grid.py](src/markov_score/bp_grid.py) | production BP: messages as densities on a truncated grid |
| 5 | [em.py](src/markov_score/em.py), [kernels.py](src/markov_score/kernels.py) | EM through BP: E-step statistics (Fisher's identity) and the kernel M-steps |
| 6 | [pair_energy_kernel.py](src/markov_score/pair_energy_kernel.py), [field.py](src/markov_score/field.py) | the 5-parameter site-energy kernel and the φ⁴ chain |
| 7 | [kernel_readings.py](src/markov_score/kernel_readings.py) | reading temperature and potential off any fitted kernel (two identities) |
| 8 | [make_paper_figures.py](tools/paper_figs/make_paper_figures.py) | how each paper figure is produced from frozen outputs |

[src/markov_score/README.md](src/markov_score/README.md) says what "exact" means here and what is only a quadrature.
[examples/minimal](examples/minimal) is a ten-minute self-contained demonstration.

## Layout

```
src/markov_score/   the library (BP, EM, kernels, priors, field, baselines)
experiments/        one script per registered experiment, frozen_config.py, registry/<id>.json (the ledger)
outputs/            frozen/ (imported evidence) and <id>/ (registered runs, hash-pinned by the registry)
tests/              the executable audit: identities, oracles, provenance, one test per claim that was once wrong
tools/              freeze.py + registry.py (the evidence gates), paper_figs/ (the paper's figures), core/ (readers per experiment)
hpc/                SLURM submit/deploy/fetch for the Bocconi cluster (cluster coordinates come from the environment)
reports/            core/ (tables and figures of the paper's core), numbers/, paper-figures/ (generated PDFs)
docs/               the ledgers: registry, contract, metrics, notation, theory notes, negative results, paper core
examples/minimal/   the 10-minute demonstration
archive/            everything not in the working paper (kept, restorable with one `git mv`; see archive/README.md)
calls/              local only, gitignored: what Jérôme and Marc asked for
```

## Quick start

```bash
make install     # venv + editable install
make test-fast   # ~1 min: identities, BP/EM correctness, theory checks
make test        # ~10 min: the whole suite
make check       # the evidence ledger: every frozen output still matches its SHA-256
make figures     # regenerate the paper's figures from frozen outputs
```

## The rules that keep the evidence honest

1. **A number enters the paper only from a frozen output** (`docs/EXPERIMENTAL_CONTRACT.md`). `tools/freeze.py`
   pins each file by SHA-256; `make check` fails on any byte change.
2. **A re-run is a new experiment id, never an overwrite.** Do not edit a script that produced a frozen output; even a
   comment makes the provenance check report a dirty tree.
3. **Before moving or deleting anything under `outputs/`, run `make check`** and again afterwards.
4. Figures and tables follow [docs/FIGURE_TABLE_STANDARDS.md](docs/FIGURE_TABLE_STANDARDS.md).

## What is where in the science

[docs/PAPER_CORE_2026-10-20.md](docs/PAPER_CORE_2026-10-20.md) is the map from each paper result (B1–B7 body,
A1–A12 appendix) to its experiment and frozen output. [experiments/README.md](experiments/README.md) lists every
registered experiment with its role. [docs/THEORY_NOTES.md](docs/THEORY_NOTES.md) and
[docs/NEGATIVE_RESULTS.md](docs/NEGATIVE_RESULTS.md) hold the derivations and the hypotheses that failed.

Licence: see [LICENSE-NOTE.md](LICENSE-NOTE.md).
