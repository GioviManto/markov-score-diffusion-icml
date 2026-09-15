# Experiment registry

One row per registered run. A run is registered when its config is frozen
(`experiments/registry/<id>.json`, SHA-256 of the config recorded), and it becomes
citable when its output directory carries a params file with the commit, an empty
dirty list on the code path, and the seeds declared here. Status vocabulary:
`PRE-REGISTERED` (config frozen, not run) · `EXPLORATORY` (dev seeds; may inform
design, never a headline) · `CONFIRMATORY` (declared seeds, untouched until the run) ·
`DONE` · `WITHDRAWN` (with the reason).

Legacy runs the manuscript cites are listed in `CURRENT_STATE_AUDIT.md` §2.1 and are
not re-registered; their status there is authoritative.

| id | script | hypothesis | config | seeds | status | outputs | commit |
|---|---|---|---|---|---|---|---|
| E01 | `experiments/e01_information_decay.py` | H1 (theory rates, crossover) | `experiments/registry/E01.json` | deterministic (`gauss`); replicates `rng_for("e01-shape", beta, rep)`, rep 0–7; closure reps 0–3 | DONE (15 Sep; `tools/make_theory_numbers.py`, `tools/make_fig_theory.py`) | `outputs/E01/` | `e179391` |
| E02a | `experiments/e02_capacity.py` | H2 (mixture capacity on the headline task) | to freeze after a dev pass | dev 10000–10003; confirmatory 0–15 | CONFIRMATORY RUNNING (frozen 15 Sep 02:20; first launch 02:01 lost to the pool-transport hang, no result used; relaunched 07:19 on 6 workers with per-cell files) | `outputs/E02a/` | `e57a3eb` |
| E02b | `experiments/e02_capacity.py --only family` | H2 (tunable non-Gaussian family) | idem | 0–15 | QUEUED (after E06) | `outputs/E02b/` | — |
| E06 | `experiments/e06_learning_through_noise.py` | H6 (RMSE vs observation level) | `experiments/registry/E06.json` | 0–15 | QUEUED (after E02a) | `outputs/E06/` | — |
| E03 | `experiments/e03_baselines.py` | H3/H9/H10 (fair baselines, matched arm, length) | `experiments/registry/E03.json` | dev 10000–10003; confirm 0–15 | PRE-REGISTERED (smoke-tested) | `outputs/E03/` | — |
| E04 | `experiments/e04_refinement.py` | H4 (grid refinement on the fitted models) | `experiments/registry/E04.json` | 0–15 (E02a fits) | PRE-REGISTERED (smoke-tested) | `outputs/E04/` | — |
| E05 | `experiments/e05_generation.py` | H5 (generation) | `experiments/registry/E05.json` | 0–3 | PRE-REGISTERED (smoke-tested) | `outputs/E05/` | — |
| E07 | `experiments/e07_phase_diagram.py` | H7 (oracle-ladder phase diagram) | `experiments/registry/E07.json` | deterministic, 2 reps | DONE (15 Sep; `tools/make_fig_phase.py`) | `outputs/E07/` | `8003dfc` |
| E08 | `experiments/e08_misspecification.py` | H8 (misspecification) | `experiments/registry/E08.json` | 0–7 | PRE-REGISTERED (smoke-tested) | `outputs/E08/` | — |

Dev seeds are 10000–10003 everywhere; confirmatory seeds 0–15. A confirmatory row's
config file is committed before the run starts, and the commit that produced the
outputs is written back into this table by `tools/registry.py --sync`.
