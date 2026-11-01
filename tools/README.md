# tools — evidence gates and the paper's figures

Run from the repository root (`.venv/bin/python tools/<script>`).

## Gates (the evidence ledger)

| script | what it does |
|---|---|
| [freeze.py](freeze.py) | pins every file a number may be read from by SHA-256 in `experiments/registry/<id>.json`; `--check` fails on any byte change, `--status` lists what is frozen |
| [registry.py](registry.py) | checks each registry entry: output directory, params files, producing commit, dirty-tree flag |
| [provenance_gate.py](provenance_gate.py) | refuses to typeset a table whose inputs cannot be attributed to a revision |

`make check` runs `freeze.py --check` and `registry.py --check`. Run it before and after any move or clean-up.

## The paper's figures

| path | what |
|---|---|
| [paper_figs/make_paper_figures.py](paper_figs/make_paper_figures.py) | every figure of the working paper (`--layout twocol --font times`); writes `reports/paper-figures/` |
| [paper_figs/thesis/](paper_figs/thesis) | the four thesis chapter modules the figures reuse (`fig_ch05/07/08/10`), vendored so nothing reads outside this repository; their data is `outputs/frozen/` |
| [figstyle.py](figstyle.py) | the save gate and palette: no prose in charts, symbols as math, paper fonts |
| [overleaf_paper.sh](overleaf_paper.sh) | `figures` copies the generated PDFs into the Overleaf-synced paper repository; `status` shows its state |

## Readers and calibrations (one per experiment that reaches the paper)

[core/](core): `read_e51.py` … `read_e59.py` print the frozen result of each experiment in the form the paper
quotes it; `calibrate_e57.py` and `calibrate_e59.py` fix the settings before a run; `check_e54_pairing.py` verifies the
paired design of the CNN comparison; `figures.py` draws the core figure set from frozen data; `audit_paper.py` recomputes every number typed in `reports/core/paper/main.tex` and exits 1 on any mismatch.

## Audits

[audit/](audit): `oracle_ladder.py` (isotropic score, LMMSE twin and true-kernel grid BP on the Laplace chain, per level) and `scalar_exponents.py` (scalar high-noise exponents by FFT quadrature).
