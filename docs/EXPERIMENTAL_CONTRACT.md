# Experimental contract: one frozen protocol, one varied factor, frozen outputs

This document states how a number gets into the paper. It is enforced by code
(`experiments/protocol.py`, `tools/freeze.py`, `tools/registry.py --check`,
`tools/check_manuscript.py`) and by CI; the prose here is the rationale.

## 1. Two frozen objects

* **The world** — `experiments/frozen_config.py: FROZEN`: the chain (ρ = 0.85, L = 32,
  Laplace innovation with q = 1 − ρ²), the OU channel, the quadrature grid (401 points on
  [−8, 8]), the twelve-level schedule on [0.05, 3], sixteen replicate seeds. Every
  experiment reads it; a run under a narrowed copy records the narrowing in its params.
* **The comparison** — `experiments/protocol.py: PROTOCOL`: everything the *arms* share so
  that they differ in one thing only: the bundle sizes and their random streams (train,
  validation 1024, test 2048, keyed by tag × seed × M), the sizes M ∈ {32, 128, 512, 2048},
  the confirmatory seeds 0–15 and the development seeds 10000–10003, the EM ladder and
  initialisation, the network recipe (batch 64, Adam, clip 1, both parameterisations, lr
  grid, 32k-step ladder) and the screen budget, the estimand (schedule-pooled relative score
  error), and the regions (all sites for the headline, the bulk for selection).

`protocol_hash()` digests both, and every params file written after 15 September records
`protocol_version` and the resolved settings, so which protocol produced a file is read
from the file, never remembered.

## 2. Ceteris paribus: the factor table

| experiment | the one factor varied | held at PROTOCOL |
|---|---|---|
| E02a | mixture capacity C ∈ {1, 2, 4, 8, 16} | bundles, seeds, M, ladder, selection rule, scoring |
| E02b | innovation shape β (generalised Gaussian) at fixed covariance | idem, M = 512, C ∈ {1, 8} |
| E03 | estimator family: architecture, training recipe, information interface | bundles, seeds, M, selection rule, scoring; EM–BP rows come from E02a on the same bundles |
| E04 | quadrature resolution (N_g, A) | the fitted models of E02a, the same test bundles |
| E05 | the score driving the sampler | one Brownian path and one initialisation per seed shared by every arm |
| E06 | observation level t of the training data | M = 512, C = 8, ladder, seeds |
| E07 | ρ and innovation family (oracles only) | schedule, chain length, seeds |
| E08 | misspecification mechanism and strength ε | M, seeds, both arms' budgets |
| E09 | capacity C at infinite data | the E02a test bundles |

Two arms that differ in more than the declared factor are not compared in the paper; if
a comparison needs two factors (E03's information-matched arm changes both the
architecture and the interface), the intermediate arm that changes only one of them is
run too (the same architecture under the standard recipe) so the difference decomposes.

## 3. Pairing

Every row carries the SHA-256 of the validation and test bundles it was scored on. Rows
from different experiments are joined on `(seed, M, test_bundle)`; a generator that joins
rows whose hashes differ raises. This is what makes "EM–BP versus the window head" a
paired comparison across E02a and E03 and not two averages.

## 4. Frozen outputs

A registered output becomes citable when it is **frozen**:

    python tools/freeze.py E02a

which pins every CSV and params file of `outputs/E02a/` by SHA-256 into
`experiments/registry/E02.json` (`frozen_outputs`, `frozen_at`, `frozen_commit`,
`frozen_run_commits`). Freezing refuses a directory whose params record a dirty code
path or no commit. From then on:

* `python tools/freeze.py --check` fails on any byte change of a frozen file; it runs in
  `tools/registry.py --check` and in `tools/check_manuscript.py` (CI).
* `tools/check_manuscript.py` fails if a generator's source directory exists but is not
  frozen: the manuscript's numbers come from immutable artefacts only.
* A re-run is a new artefact. It is frozen under a new registry key or a versioned
  directory (`outputs/E02a_v2/`), never over the old one, and the paper says which it cites.

## 5. Change policy (pre-registration)

* Confirmatory settings live in `PROTOCOL` and in the registry entry, frozen before the
  first confirmatory cell runs.
* An amendment before a run starts is allowed and is recorded with its timestamp and
  reason in the registry entry (`amendments`), and the script hash is re-recorded.
* A running confirmatory experiment is never changed. If its protocol turns out to be
  wrong (the 400-iteration ladder binding at M = 2048 for C ≥ 4, found 15 September), the
  fix is a new protocol version run as a new artefact; the old run is kept and reported
  as what it is (here: version 1, the laptop replication).
* Development seeds never enter a reported number; confirmatory seeds never enter a
  selection.

## 6. Versions

| version | where it ran | EM ladder | capacities | screen | status |
|---|---|---|---|---|---|
| 1 | laptop, 15 Sep 07:58 (E02a, E03 screen) | to 400 | 1, 2, 4, 8 | 2 dev seeds, 1 run, 16k steps | running; replication |
| 2 | Bocconi HPC (`hpc/icml_deploy.sh`, `hpc/icml_submit.sh all`) | to 1200 | 1, 2, 4, 8, 16 | 4 dev seeds, 3 restarts, 32k steps | registered 15 Sep 12:30, to run |
