# exp_34 information ladder — MDN-in-BP rung, complete and verified

Jobs `637050`→`638094`, commit `a477e51`, clean (0 dirty, archive digest
present). 8 seeds × 4 sizes × 2 arms = 64/64 cells.

## The result

| n_chains | EM-BP mean test risk | MDN-in-BP mean test risk |
|---|---|---|
| 32   | 0.264 | 0.490 |
| 128  | 0.234 | 0.529 |
| 512  | 0.319 | 0.547 |
| 2048 | 0.394 | 0.575 |

Mixture-innovation EM-BP (linear-AR form + low-dimensional innovation
family) beats MDN-in-BP (Markov factorisation + homogeneity only, no
linear-AR constraint) by roughly 2× at every size tested, consistently. This
completes the review's §10.4 information ladder: locality/topology alone
(exp_31's window/bimp arms) closes most of the gap to an unstructured
network, and going further — from "one shared kernel, arbitrary shape" to
"one shared kernel, correct linear-AR shape" — buys another clear, separate
improvement. The two structural assumptions are doing different, additive
work.

## Why this one is not the exp_32 confound

Both arms here ran with the SAME discipline exp_31 uses: a checkpoint ladder
and selection on a validation bundle disjoint from test, not a single fixed
final checkpoint. `em_bp`'s selected checkpoints are 10-40 in 30/32 cells
(two outliers at 80, 300) — nowhere near the 1200-iteration cap. `mdn`
selects its smallest offered checkpoint (5) in 27/27 cells that used the
full-size ladder (the remaining cells at very small early debugging runs
used a coarser 2-5 range, same pattern). Both arms' held-out performance
peaks early and degrades with more iterations; both are protected from that
by validation selection, which is exactly what it is for.

The `converged=0%` flag on `em_bp` looks alarming out of context — it says
the *full* fit_em run to 1200 iterations never hit the strict `tol=1e-9`
break — but it does not describe the *reported* checkpoint, which is
selected at iteration 10-40 on a completely different criterion (validation
risk), long before that boundary is even relevant. This is a different
situation from exp_32, which compared un-selected final checkpoints across
capacities sharing one budget; exp_34 never does that.

## A secondary finding worth a follow-up, not blocking

MDN selects its *smallest* available checkpoint essentially universally.
Its net has 1516 parameters against EM-BP's 25, at training sizes as small
as 32 chains — a much higher-capacity, harder-to-constrain model overfitting
fast on modest data, which is coherent with the ladder's own thesis (the
correct parametric family is doing real work, and MDN's flexibility without
it is a liability, not an advantage). Whether MDN's true optimum sits below
checkpoint 5 (i.e. whether `mdn_checkpoints` should start even earlier) is
open and would sharpen the number but is very unlikely to change the
direction: MDN would need to close a 2× gap to a well-specified estimator
with 60× fewer parameters.

## Status

Verified, complete, ready to cite. Unlike exp_32, this does not need a
rerun.
