# The three cluster jobs of 14–15 Aug 2026 — what they settle

Pulled down 16 Aug 2026. Queue empty at time of writing; nothing else running.

| Job | Experiment | State | Verdict |
|---|---|---|---|
| `627164_2` | exp16, converged-capacity rerun | COMPLETED 5h12m | **Did not answer its question** — one cell only |
| `627165_0..9` | exp21, non-Markov robustness | COMPLETED 10/10 | **Settled**, and useful |
| `627175` | exp24, wavelet scale-mixture | **FAILED** in 8 min | Real bug, out of scope, not rerun |

---

## 627164 — the capacity rerun did not complete

The withdrawn capacity claim (that enlarging the innovation mixture buys generative
quality) needed a rerun at convergence rather than at 40 EM iterations. **Only array
task 2 ever ran.** `sacct -j 627164` lists `627164_2` and nothing else — tasks 0 and 1
were never submitted, so the sweep has exactly one cell:

    outputs/exp_16_cluster/components_converged_C2_seed2/generation.csv
    C = 2, seed = 2, n_chains = 2048

What that single cell says, at convergence:

| arm | generated innovation excess kurtosis | gap to reference |
|---|---|---|
| reference (grid BP, true kernel) | 1.894 ± 0.096 | — |
| EM–BP, C = 2 | 1.837 ± 0.061 | 1.20 σ |

So at C = 2, run to convergence, EM–BP is statistically consistent with the reference.
That is *suggestive* that the earlier "capacity matters" reading was indeed a
convergence artefact — a two-component mixture would not have sufficed under the old
account — but it is **one seed at one capacity**. It cannot replace the withdrawn claim.

**Status: the capacity attribution remains withdrawn and unreplaced.** It is compendium
material either way; the paper does not depend on it.

---

## 627165 — non-Markov robustness, complete and clean

All ten tasks completed. 225 rows across two contamination mechanisms and two innovation
families. `ratio_to_em` is the arm's relative score error divided by EM–BP's, so **> 1
means EM–BP is better**.

### The clean Markov case (β = 0, γ = 0) — evidence for the headline

| family | vs local CNN | vs global MLP |
|---|---|---|
| Gaussian | 18.3–24.1× | 28.7–37.0× |
| Laplace | 9.9× | 12.1× |

This is the "Markovianity makes learning easier" margin, measured. It is **not** at
frozen-config settings — it predates `frozen_config.py` — so it is not a paper number.
But it de-risks E9: the effect is large and present in both families.

### Global latent, a rank-one contamination — degrades gracefully

| β | 0.00 | 0.10 | 0.25 | 0.50 | 1.00 |
|---|---|---|---|---|---|
| Gaussian, vs CNN | 18.34 | 17.22 | 6.99 | 3.36 | 2.22 |
| Laplace, vs CNN | 9.90 | 10.61 | 6.50 | 3.05 | 2.12 |

EM–BP still wins at β = 1, where half the marginal variance is a shared constant.

### Long-range precision coupling — fails fast

| γ | 0.00 | 0.05 | 0.10 | 0.20 | 0.40 |
|---|---|---|---|---|---|
| Gaussian, vs CNN | 24.07 | 1.19 | **0.98** | 0.86 | 0.77 |
| Gaussian, vs MLP | 37.02 | 1.79 | 1.23 | 0.96 | 0.83 |

The margin **inverts at γ ≈ 0.10** against the CNN and reaches 0.77 at γ = 0.4.

### The scope statement this licenses

The structural prior survives rank-one contamination essentially intact, regardless of
innovation shape, and **should not be relied on under genuine long-range structure.** A
global latent makes the score residual exactly rank one, which a chain absorbs;
long-range coupling is not low-rank and cannot be represented. This is the honest
limitation, and it is now backed by a complete run rather than a partial one.

---

## 627175 — wavelet, a real bug

`experiments/exp_24_wavelet_fit.py:125`

    row[f"rho_d{d}"] = float(getattr(k, "rho", np.nan))
    TypeError: only 0-dimensional arrays can be converted to Python scalars

The scale-mixture kernel carries a **vector** `rho`, not a scalar, and the line assumes a
scalar and dies on the first fit. The vector is one entry **per mixture component**, shape
`(C,)` — *not* one per wavelet detail level, as first recorded here. The detail level is
already the `d` in the column name: `kernels[orientation][d]` is one kernel per
(orientation, level), and the components live inside a single kernel.

Fixed 2026-08-16 by giving a vector-rho kernel one column per component
(`rho_d{d}_c{c}`) and keeping the plain `rho_d{d}` for the families that genuinely have a
scalar. A mean over components was considered and rejected: at `d0` the fitted components
run from +0.17 to −0.10, so the mean is a number the model never uses and it hides a sign
disagreement.

The same line hid a second failure a few seconds further on. `write_csv` takes its header
from the first row and `csv.DictWriter` raises on later rows with unlisted keys, so once
the families disagree about their rho columns the `scale_mixture` row appended after
`gaussian` would have raised in turn; rows are now aligned to the union of keys before
writing. With the default two families the header is byte-identical to the committed
`fit.csv`, so that baseline stays reproducible.

Verified locally with `--quick` over all three families, `fit` and `denoise` both clean.
The real CIFAR archive is not on the laptop, so that run used a fabricated
archive of the same format — it confirms the code path, not any number. Wavelets are out
of scope for the paper and the workshop, so this is **not** rerun on the cluster; it stays
in the compendium queue.
