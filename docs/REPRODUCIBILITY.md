# Reproducibility

From a clone to the paper's figures, on a laptop, with every step's expected outcome.

## Environment

* Python ≥ 3.11 (developed on 3.12–3.14). NumPy/SciPy/Matplotlib/pandas for the library; PyTorch (CPU) for the neural
  baselines (`pip install -e ".[nn]"`); CuPy or CUDA PyTorch optionally accelerate the batched grid recursion
  (`BP_DEVICE`, only for modules that import `markov_score.backend`).

```bash
make install         # python3 -m venv .venv; pip install -e ".[dev]"
```

## Tests

```bash
make test-fast       # ~1 min, the subset CI runs
make test            # ~10 min: 560 passed, 23 skipped (GPU parity and slow-marked tests) on an M1, 1 Nov 2026
```

## The evidence ledger

```bash
make check           # freeze.py --check (349 frozen files, 0 problems) and registry.py --check
make status          # what is frozen and what is not
```

Every number in the paper is read from a frozen output (`docs/EXPERIMENTAL_CONTRACT.md`). `make check` fails on any
byte change. Run it before and after moving or deleting anything under `outputs/`.

## The paper's figures

```bash
make figures         # tools/paper_figs/make_paper_figures.py --layout twocol --font times
```

Writes `reports/paper-figures/twocol/*.pdf` from `outputs/` (frozen and registered runs) and the vendored thesis
modules in `tools/paper_figs/thesis/`. `tools/overleaf_paper.sh figures` then sends them to the Overleaf-synced paper
repository. The generator prints every number it draws; the paper's `% src:` comments cite those printouts.

## Re-running an experiment

Each `experiments/eNN_*.py` runs with `--quick` as a smoke pass. A full run goes to the cluster through
`hpc/icml_submit.sh` (see `hpc/README.md`), is fetched with `hpc/icml_fetch.sh`, read with `tools/core/read_eNN.py`
and frozen with `tools/freeze.py <id>`. A re-run is a new experiment id, never an overwrite.

## What moved

The 6 Oct ICML paper, its table generators, the Lean audit and the notebooks are in `archive/`
(see `archive/README.md`); the instructions to rebuild them are in the git history before 1 Nov 2026.
