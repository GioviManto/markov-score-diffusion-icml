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

| id | question it closes | scale | partition |
|---|---|---|---|
| **E22c** | Q2, Q3, Q4 — the message-family question, swept | 576 cells; 4 families × 12 levels × 3 $\rho$ × 2 $L$ × 2 grids × 512 chains | `compute`, 40 cpu, 2 d |
| **E24c-cpu** | Q7, Q9, Q11 — cost and scaling, quiet node | 15 repeats, declared threads, train sizes to 8192 | `defq`, serial |
| **E24c-gpu** | Q7 across devices | both devices in one process on an H200 | `medium_gpuh200` |
| **E25** (×4) | Q6 — the neural kernel inside exact BP | 16 seeds × 4 sizes × 2 arms | `compute`, 4 jobs |

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

*Blocked on one thing*: `venv-cuda` on the cluster has torch 2.6.0+cu124 but **no cupy**.
`hpc/bocconi_gpu.sbatch` records the version pin that must be respected —
`cupy-cuda12x < 14`, because 14.x fails to JIT-compile its cub kernels against this
cluster's toolkit in a way that is *partial and misleading* (cuBLAS matmuls still work, so
a GPU looks usable until the first `.max(axis=...)` in the recursion).

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

## Still to build: E23

Not in this wave, because it does not exist yet. E24 and E22 have between them already
fixed its design, which is the main reason they went first:

- **E24 killed the FFT-first plan.** At $N_g = 401$ only 23.8 % of a sweep is the $N_g^2$
  matmul; the rest is $O(B L N_g)$ traffic over 158 MB of message arrays. An FFT that made
  the matmul free buys ≤ 1.3×. Truncation attacks the part where the time actually is.
- **E22 fixed the mode budget.** Odd cumulants are negligible (10⁻³–10⁻²), so carry even
  orders only. **Order 4 is not enough**: $|\kappa_6|$ runs 1–228 and is never small, so the
  Gaussian-plus-$\kappa_4$ rung that `e16_edgeworth_oracle` already builds will miss a large
  term. Order 6 minimum; more for heavy tails. And $\kappa_4$ *plateaus* rather than
  vanishing at large $t$ — the messages converge to the chain's stationary law, not to a
  Gaussian — so a $t$-adaptive truncation saves less than one would hope.

E23 gets built against these constraints while this set runs.

---

## Provenance

Every job runs from a `git archive` of a committed revision (`hpc/icml_deploy.sh`), with
the commit and the archive SHA-256 written into `REVISION` at the package root and into
every `params_*.json`. `run_cells` makes E22c resumable: a resubmitted job skips the cells
already on disk, so a timeout costs one cell rather than the run.

On return: `hpc/icml_fetch.sh`, then `tools/registry.py --check`, then `tools/freeze.py`
before any generator reads the outputs.
