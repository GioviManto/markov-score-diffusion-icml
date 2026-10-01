# Reproducibility

From a clone to the paper, on a laptop, with every step's expected outcome.

## Environment

* Python ≥ 3.11 (developed on 3.12.5; the archival cluster runs recorded 3.13.11).
  NumPy/SciPy/Matplotlib/pandas only for the library; PyTorch (CPU) for the new neural
  baselines under `src/markov_score/nn_torch/` (`pip install -e ".[nn]"`); CuPy is
  optional and only accelerates the batched grid recursion.
* Lean 4 `v4.33.0` via `elan` (`lean/lean-toolchain`), Mathlib pinned in
  `lean/lake-manifest.json`.
* Tectonic for the paper (runs BibTeX itself).

```bash
make install         # python3 -m venv .venv; pip install -e ".[dev]"
```

## Tests

```bash
make test-fast       # ~1 min, the CI subset
make test            # ~4 min: 443 passed, 29 skipped (CuPy parity, no GPU) on an M1, 23 Sep 2026
```

The skips are GPU parity tests. The wavelet/video suites (exploratory branches not used
by the paper) are excluded from `make test`; run `pytest tests/test_wavelet*.py
tests/test_video*.py -m "not slow"` separately.

## Lean

```bash
make lean            # lake exe cache get (once, ~2 GB) && lake build ThesisAudit
cd lean && ./audit_status.py    # verdict table
```

Expected: `Build completed successfully`, no `sorry`/`axiom`/`native_decide`
(CI greps for them). Locally the build takes ~10 s with a warm cache. `lean/README.md`
maps each manuscript proposition to the theorem that covers it.

## Smoke experiment

```bash
make smoke           # exp_07 --quick into outputs/scratch/
```

## Regenerating the manuscript's numbers

Every generated fragment under `paper/sections/` and every figure under
`paper/figures/` is produced by a script in `tools/` from outputs under `outputs/`
(`frozen/`, `legacy-unfrozen/`, or a registered experiment directory). Nothing is typed.

```bash
make tables          # eight generators; each prints what it wrote and refuses on missing or unclean inputs
make figures         # tools/make_figures.py; fails loudly naming any missing input
make check           # tools/check_manuscript.py: regenerate, diff against the committed fragments, scan the .tex for typed literals
make paper           # paper/main.pdf
```

Regeneration was verified at import (14 Sep 2026): all eight table generators and all
four legacy figures reproduce the archival numbers exactly; the differences between the
regenerated fragments and the copies the previous manuscript shipped (a CI column, a
label) are recorded in `docs/CURRENT_STATE_AUDIT.md`, §2.2.

Known Tectonic quirk: the ICML style's running-head height test fails under Tectonic's
font metrics, so page headers read "Title Suppressed Due to Excessive Size" — for the
kit's own example paper as well. Under pdfLaTeX/TeX Live (what the submission system
uses) the running title appears. The style file is never edited to work around this.

## Running experiments

Legacy experiments (`experiments/exp_NN_*.py`) take `--quick`, `--only <part>`, `--set
key=value` and `--out <dir>`; they import `frozen_config.FROZEN` and write a
`params_<part>.json` beside their CSVs with commit, dirty-tree list, resolved config
hash, seeds, host, Python/NumPy versions and timestamp (`experiments/common.py`).
`NGBP_REQUIRE_CLEAN=1` makes a run refuse to start on a dirty tree.

New experiments (`experiments/e*_*.py`, registered in `docs/EXPERIMENT_REGISTRY.md`)
carry their settings in a `SETTINGS` dict at the top of the script, take `--quick`,
`--dev` (development seeds), `--only <part>`, `--set key=value`, `--workers N` and
`--output-dir`, and write the same provenance record; the registry entry
(`experiments/registry/<id>.json`) stores the frozen settings and the script's SHA-256.
Confirmatory runs use the seed block declared in the registry entry, which is frozen
before the run.

**Cell files.** Every multi-cell driver goes through `run_cells` in
`experiments/common.py`: the process that computes a cell writes that cell's rows to
`<output-dir>/cells/<csv stem>/<key>.json` itself, and the driver rebuilds the CSV from
the cell files after every completion. Cells whose file exists are skipped, so an
interrupted run is resumed by re-issuing the same command; a cell that raised leaves
`<key>.failed` with its traceback and stops the run. Nothing but a short key travels
through the pool's result pipe, and the driver detects completion from the files rather
than from the pipe. This replaced the earlier "return rows through `imap_unordered`"
driver on 15 September 2026, after the first confirmatory E02a launch (6 workers, 02:01)
computed fifteen cells whose results never reached the parent process: the workers kept
computing, the parent's result handler sat in a blocking read for five hours, and no CSV
was written (`outputs/scratch/queue.log`). A synthetic pool with 30 kB results did not
reproduce it; the cause was not identified and the design no longer depends on it.

Archival cluster scripts are in `hpc/`; they read `BP_CLUSTER_HOST`/`BP_CLUSTER_USER`
from the environment. Everything in this repository's plan was sized to run on eight
laptop cores (see `docs/RESEARCH_PLAN.md`, compute budget).

## Provenance rules

1. A number enters the manuscript only through a generator reading a registered output.
2. A registered output carries a params file with commit and an empty dirty list on the
   code path (`tools/provenance_gate.py`, `import_closure`).
3. Confirmatory seeds are declared in the registry before the run and never reused for
   selection.
4. `make check` fails if a regenerated fragment differs from the committed one or if a
   typed decimal in the `.tex` is not a generated macro.

Commit hashes cited in the registry, the params files and the deployment tags up to 1 October 2026 refer to the heavy local history; `docs/COMMIT_MAP_2026-10-01.md` maps each to the pushed commit.
