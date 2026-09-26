"""Every kernel's m_step must return a NEW kernel and leave the receiver untouched.

`em.fit_em` builds its checkpoint ladder with `saved[it] = current`, so a kernel whose
m_step mutates itself and returns `self` makes every checkpoint an alias of one object,
all carrying the FINAL parameters.

That is what `MDNKernel` did (amendment N-25). Measured before the fix: 1 distinct object
and 1 distinct theta across 4 checkpoints, against 5 of 5 for `MixtureInnovationKernel`.
The damage was not a small numerical error — validation-based checkpoint selection became a
no-op for that kernel, which disabled the one control E25 exists to exercise, the
overfitting the advisor predicted unprompted. An arm cannot demonstrate early stopping if
all of its checkpoints hold the same parameters.

This test states the contract for every kernel, so a new rung cannot quietly break it.
"""

from __future__ import annotations

import numpy as np
import pytest

from markov_score.bp_grid import make_grid
from markov_score.em import e_step_multi, fit_em
from markov_score.kernels import GaussianAR1Kernel, MDNKernel, MixtureInnovationKernel
from markov_score.noising import alpha_delta
from markov_score.priors import LaplaceAR1
from markov_score.utils import rng_for


def _setup(m=48, n=24, n_grid=201):
    grid, w = make_grid(8.0, n_grid)
    prior = LaplaceAR1(0.85)
    rng = rng_for("mstep-contract", m)
    A = np.stack([prior.sample(rng, n) for _ in range(m)])
    groups = []
    for t in (0.2216, 0.4665, 0.9821):
        al, de = alpha_delta(t)
        groups.append((al * A + np.sqrt(de) * rng.standard_normal(A.shape), al, de))
    return grid, w, groups


def _kernels():
    return [
        ("mixture", MixtureInnovationKernel.init(4, rho=0.3, var=0.8, rng=rng_for("k-mix", 0))),
        ("gaussian", GaussianAR1Kernel(rho=0.3, q=0.8)),
        ("mdn", MDNKernel.init(3, 16, rng_for("k-mdn", 0))),
    ]


@pytest.mark.parametrize("name,kernel", _kernels(), ids=[n for n, _ in _kernels()])
def test_m_step_does_not_mutate_the_receiver(name, kernel):
    grid, w, groups = _setup()
    before = np.asarray(kernel.theta, dtype=float).copy()
    stats = e_step_multi(grid, w, kernel.log_transition_matrix(grid), groups)
    out = kernel.m_step(stats, grid)
    after = np.asarray(kernel.theta, dtype=float)
    assert np.array_equal(before, after), (
        f"{name}.m_step mutated the kernel it was called on; fit_em's checkpoint ladder "
        f"stores references, so this aliases every checkpoint to the final parameters"
    )
    assert out is not kernel, f"{name}.m_step returned the same object instead of a new kernel"


@pytest.mark.parametrize("name,kernel", _kernels(), ids=[n for n, _ in _kernels()])
def test_checkpoints_are_distinct_parameters(name, kernel):
    grid, w, groups = _setup()
    cks = {2, 4, 6}
    _, _, saved = fit_em(kernel, grid, w, groups, n_iters=8, tol=1e-12, checkpoints=cks)
    got = sorted(saved)
    if len(got) < 2:
        pytest.skip(f"{name} produced only {len(got)} checkpoint(s)")
    thetas = {np.asarray(saved[c].theta, dtype=float).tobytes() for c in got}
    ids = {id(saved[c]) for c in got}
    assert len(ids) == len(got), f"{name}: {len(ids)} distinct objects for {len(got)} checkpoints"
    assert len(thetas) == len(got), (
        f"{name}: {len(thetas)} distinct parameter vectors for {len(got)} checkpoints — "
        f"validation-based checkpoint selection would be a no-op"
    )
