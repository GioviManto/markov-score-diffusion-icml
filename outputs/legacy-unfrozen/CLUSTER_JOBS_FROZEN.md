# The frozen batch — job ledger

Every number that may enter a document comes from a job listed here. All of them
run `hpc/bocconi_frozen.sbatch`, which imports `experiments/frozen_config.py`;
`tools/check_paper.sh` fails if any experiment sets rho, the grid or an
iteration count locally.

Sorted newest first. "Usable" means the result may be cited somewhere; where it
may be cited is the last column, and `—` means nowhere yet.

| Job | Task range | What | State | Usable | Cited in |
|---|---|---|---|---|---|
| `629985` | 110–122 (+123–125) | E9 with **both** budgets validation-selected — the network's training length as well as EM's iteration count | RUNNING | pending | — |
| `629962` | 90–105 | Rung 4a, matched convergence budget (8 seeds × 2 size groups, cap 3000, per-sequence tolerance) | RUNNING | pending | — |
| `629193` | 80–87 | Rung 4a v2 — one species, 600 steps, 16 seeds | COMPLETED 8/8 | partly | compendium §Rung 4a |
| `629091` | 60–75 | E9 sample efficiency, validation-selected EM stopping | COMPLETED 16/16 | not yet | — (protocol still asymmetric) |
| `628943` | 40–55 | E9 sample efficiency at 400 EM iterations | COMPLETED 16/16 | not yet | — |
| `628942` | 30–33 | Rung 4a v1 — **defective**, two species scored at one psi | COMPLETED 4/4 | **no** | compendium, as a correction |
| `628601` | — | shape convergence | COMPLETED 2h29m | yes | compendium |
| `628600` | — | clean-vs-channel rates | COMPLETED 4h01m | yes | compendium |
| `628571` | — | E9 sample efficiency, 16 seeds | COMPLETED | yes | paper §efficiency (with caveat) |
| `628434_5` | 5 | Rung 4a, first attempt | **TIMEOUT** 6h00m | no | — |

## The three that are worth knowing about

**`628434_5` — killed by the wall having written nothing.** 960 gradient fits at
~0.2 s per step is roughly ten hours, and the experiment writes once at the end.
Fixed by sharding on seed, which is exact rather than approximate because
`part_potential` reseeds per seed.

**`628942` — completed and produced nothing usable.** `part_potential` generated
two species at ±omega while `fit_potential` scores every trajectory at the single
scalar psi it is handed, so half the data was evaluated under the wrong rotation.
The estimator escaped by collapsing the ring — at r\* → 0 the ring is
rotation-invariant and so pays nothing for a wrong psi — which pinned r\* to the
lower clip in 51.2% of cells and made the *marginal* arm appear to beat the joint
arm. Pinned both ways by
`tests/test_ring.py::test_mixed_species_data_collapses_the_ring_under_a_single_psi`.

**`629193` — fixed the estimator, but not the comparison.** Clip pinning fell to
4.8% and the joint arm recovered its expected ordering, so the joint arm's
n-scaling is reportable. The joint-versus-marginal *margin* is not: at t ≤ 1 the
marginal arm hit the 600-step cap in 95% of cells against the joint arm's 57%, so
the two arms were not stopped by the same standard. `629962` reruns it under a
per-sequence tolerance, which is what the stopping rule should have been all
along: the old absolute tolerance was eight times stricter at n = 4096 than at
n = 512, i.e. it varied along the very axis the rung reports its scaling on.

**A correction, and a caveat on what `629962` can deliver.** The probe behind
that rerun was first read off its single first cell, which showed the joint arm
frozen from step 600 while the marginal arm improved to 2400 — a tidy story of
one arm converged and the other cut short. Across all twelve cells it does not
hold. The joint arm moves in five of six cells, and at n = 512, t = 0.4665 its
lambda error gets *worse* with more steps, 0.029 at 600 against 0.060 at 2400.
The parameter error is non-monotone in four of twelve cells (0.177 → 0.066 →
0.207 at 600/1200/2400), while the log-likelihood rises monotonically in all
twelve, as backtracking guarantees.

So more optimisation is not more accuracy: the finite-sample maximiser is not at
the truth, and the budget is a regularisation knob rather than a convergence
detail — the same thing exp_29 found for EM. `629962` therefore gives two arms
*stopped by the same rule*, not two arms *converged*. The fix that would actually
settle it is to select each arm's stopping point on held-out data, as the
efficiency experiment now does for both of its arms; that is not implemented for
this rung, and any margin `629962` reports carries the caveat.

**`629985` — the efficiency comparison, finally symmetric.** `629091` gave EM-BP
a validation-selected iteration count while the network trained to a fixed
20,000 steps. That is the same fairness error the validation-selection was
introduced to fix, moved rather than removed: training length is a bias/variance
knob for *both* arms — exp_29 shows EM's held-out error has an interior minimum
at every sample size, and SGD's does too — so selecting one budget and pinning
the other favours the selected arm. It favoured us, which is why it could not
ship. The network's training length is now chosen on the same bundle, jointly
with its parameterisation, over {500, 1000, 2000, 3500, 6000, 10000, 14000,
20000}. Each arm has exactly one selected quantity; the network still selects
over a 2-D grid against EM-BP's 1-D. Both sides carry `*_at_cap` columns, so the
effect of selection is measured per arm rather than inferred from the ratio.

Submitted as 13 shards (seeds 0–12) because the QOS submit limit is 30 and
`629962` already holds 16; seeds 13–15 go in automatically when slots free.

## Not yet run

- Nothing blocking. The efficiency table stays behind its `\needsdata` marker
  until `629985` lands and is checked.
