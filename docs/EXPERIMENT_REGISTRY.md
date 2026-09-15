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
| E01 | `experiments/e01_information_decay.py` | H1 (theory rates, crossover) | `experiments/registry/E01.json` | deterministic (`gauss`); replicates `rng_for("e01-shape", beta, rep)`, rep 0–7; closure reps 0–3 | DONE (15 Sep; `tools/make_theory_numbers.py`, `tools/make_fig_theory.py`) | `outputs/E01/` | `8091a2d` (outputs), script at `e179391` |
| E02a | `experiments/e02_capacity.py` | H2 (mixture capacity on the headline task) | to freeze after a dev pass | dev 10000–10003; confirmatory 0–15 | DEV PASS RUNNING (15 Sep, `outputs/E02a_dev/`) | `outputs/E02a/` | — |
| E02b | `experiments/e02_capacity.py --only family` | H2 (tunable non-Gaussian family) | idem | idem | PRE-REGISTERED | `outputs/E02b/` | — |
| E06 | `experiments/e06_learning_through_noise.py` | H6 (RMSE vs observation level) | to freeze | 0–15 | PRE-REGISTERED | `outputs/E06/` | — |

Dev seeds are 10000–10003 everywhere; confirmatory seeds 0–15. A confirmatory row's
config file is committed before the run starts, and the commit that produced the
outputs is written back into this table by `tools/registry.py --sync`.
