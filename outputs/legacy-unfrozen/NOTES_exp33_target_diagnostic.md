# exp_33 target diagnostic — result and how to read it

Job `635649`, commit `f47fe6d`, clean (0 dirty paths, archive digest present).
Six seeds, n=2048, `BiMessagePassing` (hidden=64), headline schedule.

## The numbers

| condition | mean test risk (6 seeds) | checkpoint selected (all 6 seeds) |
|---|---|---|
| `fixed_dsm`   | **2.209** | 500, 500, 500, 500, 500, 500 (never later) |
| `fresh_dsm`   | 0.098 | 16000, 8000, 32000, 16000, 16000, 16000 |
| `fixed_exact` | 0.195 | 8000, 8000, 32000, 32000, 8000, 8000 |
| `fresh_exact` | 0.073 | 32000, 32000, 32000, 16000, 32000, 32000 |

## What this actually shows

`fixed_dsm` — a finite training set (2048 clean chains, resampled every
step) with the ordinary single-sample DSM noise target — selects the
*smallest* checkpoint on the ladder in every one of six seeds. That means
validation risk got monotonically **worse** past 500 steps: the network is
overfitting to the DSM target's per-example noise realization, not to
anything about the chain. Both alternatives that remove one piece of that
combination (fresh data every step, or an exact posterior-mean target instead
of a noisy one) keep improving out to 8000-32000 steps and land an order of
magnitude lower.

This is not a training bug — the val/test risk agree closely per row (e.g.
seed 0: val 2.1179, test 2.1197), so checkpoint selection is behaving
correctly given what it sees. It is a real, seed-independent (6/6)
finite-data-plus-noisy-target overfitting effect, and it is large.

## Why this matters for exp_31, and why it does not undermine it

`fixed_dsm` **is** the ordinary training regime every network arm in this
project uses — exp_07, exp_31's own confirmatory arms, all of it: a finite
training set, DSM noise target. If this pathology generalizes across
architectures and sizes, a network arm trained for many steps without
validation-based checkpoint selection could be reporting a badly overfit
late checkpoint's error rather than its best one.

exp_31 already selects checkpoints on a validation bundle disjoint from
test, for exactly this reason (see its module docstring: "with checkpoints
the caller can select on validation and demonstrate saturation"). This
result is evidence that decision was load-bearing, not a formality: at
n=2048 with fixed data and a DSM target, the FINAL checkpoint would have
been a materially worse number than the SELECTED one. exp_31's design is
robust to this failure mode by construction. What this diagnostic adds is a
mechanism for *why* the widening gap in earlier (uncheckpointed) protocols
looked the way it did, beyond the budget-shrinks-with-n argument already in
the manuscript.

## What this does not yet establish

Six seeds, one architecture (BiMessagePassing), one size (n=2048). Whether
the same overfitting-to-noisy-target pattern holds for DilatedConv1d, for
the window head, or at smaller n (where it may or may not appear, since a
smaller fixed set is revisited even more often per step) is not measured
here. Not run: whether the effect appears at all four exp_31 sizes, or
whether it is specific to this architecture's capacity relative to 2048
examples.

## Where this belongs in the manuscript

Not written in yet. exp_31's confirmatory results should land first, so
this mechanism note can be checked against what actually happened in the
arms that matter for the headline structured-baseline claim, rather than
added as a free-standing tangent. If exp_31's window/conv/bimp arms show the
same "selects an early checkpoint, final would have been worse" pattern at
n=2048, this note is the explanation and belongs in `sec:em-architecture`
(thesis) / the structured-baseline appendix (paper). If they don't, the
effect may be specific to this diagnostic's exact setup and should be
reported more narrowly.
