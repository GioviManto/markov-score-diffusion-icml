# Handoff — exp_24 wavelet fit, scale_mixture crash fixed

**Date:** 2026-08-16
**Status:** Done, verified locally. **Not** rerun on the cluster (deliberate).
**Scope:** Compendium-only. Wavelets are out of scope for the paper and the workshop, so
this is not on the critical path to the 16 Sep submission.

---

## What was broken

Bocconi job **627175** died after ~8 minutes on 2026-08-14 at
`experiments/exp_24_wavelet_fit.py:125`:

    row[f"rho_d{d}"] = float(getattr(k, "rho", np.nan))
    TypeError: only 0-dimensional arrays can be converted to Python scalars

`ScaleMixtureKernel` carries a vector `rho`; `float()` on it raises on the first fit. The
job was invoked via `hpc/bocconi_wavelet.sbatch` mode `imagefit_scale`, i.e.
`--set families="('gaussian','mixture','scale_mixture')"`.

## Correction to the recorded cause

The original note in `outputs/CLUSTER_JOBS_14_15_AUG.md` said the vector was **one entry
per wavelet detail level**. That is wrong, and it matters for column naming.

`src/scale_kernel.py` declares `rho: np.ndarray  # (C,)` and `init` sets
`np.full(n_components, rho)` — the vector indexes the **mixture component**, not the
level. The detail level is already the `d` in the column name: `kernels[orientation][d]`
is one kernel per (orientation, level), and the components live *inside* one kernel.
Columns are therefore `rho_d{d}_c{c}`, not `rho_d{d}_l{level}`.

`CLUSTER_JOBS_14_15_AUG.md` has been corrected in place.

## What changed

| File | Change |
|---|---|
| `experiments/exp_24_wavelet_fit.py` | Per-component rho columns; new `_aligned()` helper; `write_csv(..., _aligned(rows))` |
| `outputs/CLUSTER_JOBS_14_15_AUG.md` | Corrected the 627175 entry (wrong cause + fix record) |

**Two bugs, not one.**

1. *The reported crash.* A vector-rho kernel now emits one column per component,
   `rho_d{d}_c{c}`. Kernels with a genuine scalar (`GaussianAR1Kernel`,
   `MixtureInnovationKernel`) keep the plain `rho_d{d}` path unchanged.

   A mean over components was considered and rejected. At `d0` the fitted components run
   from **+0.17 to −0.10**; the mean (~0.026) is a number the model never uses and it
   hides a sign disagreement. (An earlier uncommitted working-tree patch did exactly
   this mean-plus-spread collapse — it has been replaced.)

2. *A second failure a few seconds behind it.* `write_csv` takes its header from the
   first row and `csv.DictWriter` raises on later rows carrying unlisted keys. Since
   `families` starts with `gaussian`, the `scale_mixture` row appended after it would
   have raised in turn once the families disagreed about rho columns. Rows are now
   aligned to the union of keys, missing entries left empty.

## Verification

`--quick`, both parts, all three families — the exact combination that crashed:

    [fit] gaussian:      14s, 6 iters, monotone violation 0
    [fit] mixture:       16s, 6 iters, monotone violation 0
    [fit] scale_mixture: 21s, 6 iters, monotone violation 0
    [denoise] t=0.4: mse_gaussian=0.3692 mse_mixture=0.3712 mse_scale_mixture=0.3757
    [denoise] t=0.8: mse_gaussian=0.5514 mse_mixture=0.5527 mse_scale_mixture=0.5543

- `scale_mixture` gets `rho_d{0..3}_c{0..3}`; scalar families get `rho_d{0..3}` with the
  per-component cells empty.
- With the **default** two families the header is byte-identical to the committed
  `outputs/exp_24_wavelet_fit/fit.csv`, so that baseline stays reproducible.
- `tests/test_scale_kernel.py`, `test_wavelet_model.py`, `test_wavelet.py` — 31 passed.

**Caveat on the numbers.** The real CIFAR archive is not on the laptop
(`data/cifar-10-python.tar.gz` is absent and `src/image_data.py` has no download path —
the smoke gate only runs on the cluster). Rather than pull ~170 MB, the run used a
fabricated archive of the same format filled with a pink-noise random field. **The run
confirms the code path and the CSV schema; the fitted values above are not science.**
Generator kept at `scratchpad/make_fake_cifar.py` (session-scoped, regenerate if needed).

## State / open items

- Changes are **uncommitted** in the working tree. The repo has many other modified files
  from prior sessions — do **not** blanket-commit; stage only the two files above.
- The cluster rerun (`bocconi_wavelet.sbatch` mode `imagefit_scale`) was **not** submitted,
  per instruction. Submitting it needs an explicit go-ahead. Note the walltime concern
  already recorded in that sbatch: `imagefit`'s gaussian family alone had not finished
  after 2h on medium_cpu's 6h cap at full size, and exp_24 fits every family sequentially
  before writing anything — no partial credit on a walltime kill. Run
  `imagefit_scale_calibrate` first and size the request from it.
- Real-CIFAR numbers for `scale_mixture` on images do not exist yet. Nothing in the paper
  or workshop depends on them.
