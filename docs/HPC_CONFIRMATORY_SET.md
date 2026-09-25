# The confirmatory set — 25 September 2026

Defined after the Jerome call. The laptop runs of E22 and E24 answered four of his
eleven questions in an afternoon; this set re-runs them at a scale where the answers
cannot be artefacts of one configuration, and adds the two experiments that were written
but never executed.

## What "sure" can and cannot mean here

Worth stating once, because it sets the design. We cannot be *100 % sure* of anything
statistical — more data narrows a confidence interval, it does not close it. What this
set can do, and is built to do, is make each conclusion **robust to every factor that
could plausibly explain it away**, and make each one **falsifiable in advance**:

1. **Sweep the alternative explanations, not just the sample size.** E22's laptop run held
   $\rho$, chain length and grid resolution fixed. Sixteen times the chains at the *same*
   configuration would buy a tighter median and no new confidence. Sweeping $\rho$,
   $L$ and $N_g$ is what rules out "this is a property of $\rho = 0.85$" and "this is
   quadrature".
2. **Carry an exact control in every cell.** The Gaussian-innovation family is the one
   where all three candidate message families contain the truth exactly, so whatever
   divergence it shows is pure numerics. Every claim is measured against *its own* floor,
   not an assumed one.
3. **Predeclare, then look.** Every experiment's registry entry names the prediction and
   what would falsify it before the run. Two predictions have already failed this way
   (E24's $N_g$ exponent, E22's P3), and both failures were more informative than the
   confirmations.

A number that survives all three is as close to "sure" as this gets.

---

## The set

| job | id | question it closes | scale | partition |
|---|---|---|---|---|
| 669321 | **E22c** | Q2, Q3, Q4 — the message-family question, swept | 576 cells; 4 families × 12 levels × 3 $\rho$ × 2 $L$ × 2 grids × 512 chains | `compute`, 40 cpu, 2 d |
| 669375 | **E23** | Q5 — how many numbers a message needs | 4 representations × 12 dof × 4 families × 12 levels × 3 $\rho$; plus 3 quadrature rules × 10 node counts | `compute`, 40 cpu |
| 669327 | **E24c-cpu** | Q7, Q9, Q11 — cost and scaling, quiet node | 15 repeats, declared threads, train sizes to 8192 | `defq`, serial |
| 669334 | **E24c-gpu** | Q7 across devices | both devices in one process on an H200 | `medium_gpuh200` |
| 669376 | **E24c-capacity** | the other half of "small kernel vs huge network" | capacity × dataset size, M-step exponent refitted per capacity | `medium_cpu` |
| 669323/4/5/6 | **E25** | Q6 — the neural kernel inside exact BP | 16 seeds × 4 sizes × 2 arms | `compute`, 4 jobs |
| 669377 | **E26** | Q8 — does score accuracy transfer to generation | 4 arms on shared Brownian paths + step study | `compute` |

(Plus the pre-existing 667192, E12b, still pending since 23 September.)

### E22c — message shapes, everything swept

The laptop run showed the "Gaussian through one transition" ansatz is exact at site 1
(10⁻¹⁵) and fails by 10¹⁰ by site 31. Three things could still explain that away, and each
becomes a swept factor:

| held fixed on the laptop | swept here | what it rules out |
|---|---|---|
| $\rho = 0.85$ | 0.5, 0.85, 0.95 | "it is a strong-correlation effect" |
| $L = 32$ | 32, 128 | "the growth never saturates / is an end effect" |
| grid (801, 12) | (801, 12), (1601, 16) | **"it is quadrature"** |

The grid sweep is the load-bearing one. The laptop run argued the quadrature floor was
below the effect by citing E11's envelope. Running two resolutions makes it a property of
*this* experiment's own data: if the site-31 divergence is the same at both, it is not
quadrature, and no appeal to another experiment is needed.

512 chains (8× the laptop) gives the p90 and max columns, not only the median — a
distribution over chains, so a claim can be made about the worst case and not just the
typical one.

### E24c — cost, on a machine that is not also running a browser

The laptop run is marked EXPLORATORY in its registry entry for good reason: 4 of 8 threads
on a contended machine. It answered Jerome's Q7 directionally (BP 2.2× slower than the
transformer) and that is unlikely to reverse, but the number itself should come from a
quiet node with a declared thread count. `MSD_HPC_THREADS` now exists for exactly this, and
the params file records it — a timing at an unstated thread count is not reproducible.

`n_repeat` 5 → 15, `scale_repeat` 3 → 9, `train_sizes` extended to 8192 so the training
decomposition (E-step linear in $M$, M-step flat) is tested over a 256× range rather than
64×.

### E24c-gpu — the cross-device row

Both devices in **one process on one node**, so the ratio is not two jobs compared after
the fact. This matters more than it sounds: the networks are GPU-native and BP's GPU path
goes through `cupy`, so timing a GPU transformer against a CPU BP would report an artefact
of what each arm was allowed to run on.

*Resolved before submission*: `venv-cuda` had torch 2.6.0+cu124 but **no cupy**, so BP had
no GPU path at all. `cupy-cuda12x==13.6.0` was installed into it. The `< 14` pin is
load-bearing and is recorded in `hpc/bocconi_gpu.sbatch`: 14.x fails to JIT-compile its cub
kernels against this cluster's toolkit in a way that is *partial and misleading* — cuBLAS
matmuls still succeed, so a GPU looks usable right up until the first `.max(axis=...)` in
the recursion.

A second safeguard worth knowing about: `_resolve_devices` drops a `gpu` arm it cannot
satisfy rather than failing, so a GPU job on a node without a device would still produce
output. That output is self-describing — every row carries `device_name`, and the params
file records `cupy_gpu_available` — so a CPU-only fallback is visible rather than silent.
Check those two fields before quoting any GPU number.

### E25 — the neural kernel, finally run

`experiments/exp_34_information_ladder.py` fits `MDNKernel` — a small MLP whose heads
parameterise the transition kernel, placed *inside* the exact BP recursion — as the
missing rung of an information ladder. It has existed, complete, since before the call,
and has never been executed; there is no output directory for it anywhere in the tree.
It is precisely what Jerome proposed:

> *"Exploit the structure by learning only one small neural network that will learn a
> transition kernel… instead of a huge neural network, like an attention, that needs to
> understand that."*

16 seeds × sizes (32, 128, 512, 2048) × arms (em_bp, mdn). The driver is serial with its
own CSV resume, so the seeds are split across four jobs into separate output directories
and merged at fetch time.

**What E25 must also answer, from E24's training result.** E24 showed the M-step cost is
flat in dataset size (slope +0.008) while the E-step is linear — a score network pays for
its parameters on every example at every step, EM–BP pays once per iteration on the grid.
The claim that turns that into "small network beats huge network" is the *capacity* axis:
does the M-step stay flat in $M$ as the kernel grows, while a score network's per-step cost
grows with its parameter count? That sweep is not in `exp_34` and is the first follow-up.

---

## E23 — built and submitted, as a measurement rather than a solver

The plan of record was to build a Fourier BP solver. Three results since then said that
would be a week spent before finding out whether the premise holds:

- **E24 `anatomy`**: at $N_g = 401$ only 23.8 % of a sweep is the $N_g^2$ matmul. An FFT
  that made it *free* buys ≤ 1.3×. Acceleration is not where the time is.
- **E22**: $|\kappa_6|$ of the messages runs 1–228 and is never small, so a low-order
  truncation cannot be assumed cheap.
- **Quadrature**: Gauss–Legendre is ~10× *worse* than uniform trapezoid here. The Laplace
  kink and the narrow likelihood are not polynomial-like, while trapezoid is spectrally
  accurate for a rapidly decaying integrand. A better rule does not shrink the grid.

So E23 asks the question that is prior to any solver: **for the messages exact BP actually
produces, how many degrees of freedom does each representation need?** If twenty Fourier
modes really do explain 99 %, a solver is worth a week. If it takes two hundred, nothing
built on that representation beats the 401-point grid, and we should say so and stop. Two
days instead of two weeks, and a negative answer is as useful as a positive one.

**Fairness was a real problem here and the first version got it wrong.** Truncating on the
solver's shared $[-12, 12]$ domain — where the lowest harmonics are ~48× wider than a
typical message — makes "the first 20 modes" describe mostly empty space. That is a
strawman of the proposal. `fourier_adapted` truncates on a per-message window instead, and
the gap between the two variants is the price of the **shared domain**, reported separately
from the price of the representation. On the smoke run the adapted variant at 47 dof beat
the grid at 45 dof by **26×**, so that is where any win would come from.

Predictions R2–R4 all say the Fourier/cumulant route does *not* pay, and they are written
down in advance precisely because the project has an incentive to find otherwise — and
because the two previous experiments each falsified a prediction we preferred.

## E26 — generation, weak form

`e05_generation.py` was pre-registered on 15 September and never run. It answers the
*weak* form of Q8 — that BP can drive generation, paired against the same Brownian paths.
It does **not** answer the strong form Jerome actually proposed: one forward–backward sweep
from a flat marginal, with no reverse SDE and no score at all. That sampler does not exist
yet; the distinction is recorded in `experiments/registry/E26.json` so it is not lost when
the results land.

## Provenance

Every job runs from a `git archive` of a committed revision (`hpc/icml_deploy.sh`), with
the commit and the archive SHA-256 written into `REVISION` at the package root and into
every `params_*.json`. `run_cells` makes E22c resumable: a resubmitted job skips the cells
already on disk, so a timeout costs one cell rather than the run.

On return: `hpc/icml_fetch.sh`, then `tools/registry.py --check`, then `tools/freeze.py`
before any generator reads the outputs.
