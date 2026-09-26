"""A rejected generalized M-step must not be read as convergence (amendment N-24).

`fit_em`'s stopping test fired on ONE below-tolerance increment. That is correct for an
exact M-step, where a zero increment is a fixed point, and wrong for a GENERALIZED one:
`MDNKernel.m_step` is a backtracked Adam ascent, and when its line search rejects every
scale it returns the kernel unchanged, so the next increment is exactly 0.0.

E25's MDN arm died on this. Increments ran 0.130, 0.032, 0.012, 0.018, 0.007, 0.016,
0.0076, 0.0062, 0.000 and the run stopped at iteration 10 of a planned 400 labelled
"converged" — every preceding increment seven to eight orders above tol = 1e-9. On the
cluster it stopped at the first checkpoint in all 48 cells, which made a broken optimiser
look like a model that is 10x worse than the mixture kernel.
"""

from __future__ import annotations

import numpy as np

from markov_score.bp_grid import make_grid
from markov_score.em import fit_em
from markov_score.noising import alpha_delta
from markov_score.priors import GaussianAR1


class _StallingKernel:
    """Improves, stalls for exactly one step, then improves again.

    Stands in for a generalized M-step whose line search rejects a single proposal. The
    kernel itself is a Gaussian AR(1) whose rho walks towards the truth; `stall_at` makes
    one m_step return `self`, which reproduces the exactly-zero increment.
    """

    def __init__(self, rho, step=0, stall_at=3):
        self.rho, self.step, self.stall_at = rho, step, stall_at

    @property
    def theta(self):
        return np.array([self.rho])

    def log_transition_matrix(self, grid):
        return GaussianAR1(self.rho).log_transition_matrix(grid)

    def m_step(self, stats, grid):
        if self.step == self.stall_at:
            # line search rejected every scale: parameters unchanged
            return _StallingKernel(self.rho, self.step + 1, self.stall_at)
        return _StallingKernel(self.rho + 0.5 * (0.85 - self.rho), self.step + 1, self.stall_at)


def _groups(rho=0.85, n=32, m=48, seed=0):
    rng = np.random.default_rng(seed)
    prior = GaussianAR1(rho)
    A = np.stack([prior.sample(rng, n) for _ in range(m)])
    out = []
    for t in (0.2216, 0.4665, 0.9821):
        al, de = alpha_delta(t)
        out.append((al * A + np.sqrt(de) * rng.standard_normal(A.shape), al, de))
    return out


def test_single_rejected_step_does_not_end_the_run():
    grid, w = make_grid(8.0, 201)
    groups = _groups()
    _, trace = fit_em(_StallingKernel(0.3, stall_at=3), grid, w, groups, n_iters=12, tol=1e-9)
    # The stall is at step 3, so a patience-1 rule would stop around iteration 5.
    assert len(trace.log_evidence) > 6, (
        f"run ended after {len(trace.log_evidence)} iterations; a single rejected M-step "
        f"was read as convergence (stop_reason={trace.stop_reason!r})"
    )


def test_patience_one_reproduces_the_defect():
    """The guard is only meaningful if the old behaviour is still reachable and still wrong."""
    grid, w = make_grid(8.0, 201)
    groups = _groups()
    _, trace = fit_em(_StallingKernel(0.3, stall_at=3), grid, w, groups,
                         n_iters=12, tol=1e-9, patience=1)
    assert trace.stop_reason == "converged"
    assert len(trace.log_evidence) <= 6, (
        "patience=1 should stop at the rejected step; if it does not, this test no longer "
        "pins the behaviour the fix is about"
    )


def test_a_true_fixed_point_still_converges():
    """Patience must not turn genuine convergence into a censored run."""
    grid, w = make_grid(8.0, 201)
    groups = _groups()

    class _Fixed(_StallingKernel):
        def m_step(self, stats, grid):
            return self                                  # a real fixed point, every step

    _, trace = fit_em(_Fixed(0.85), grid, w, groups, n_iters=12, tol=1e-9)
    assert trace.converged and trace.stop_reason == "converged"
    assert len(trace.log_evidence) <= 6, "a fixed point should cost only `patience` iterations"
