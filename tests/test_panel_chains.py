"""E61's added chains (markov_score.panel_chains): the registered parameters (experiments/registry/E61.json), the
closed-form stationary law of the lattice diffusions, column densities on E03's grid, and sampled moments."""

from __future__ import annotations

import numpy as np
import pytest

from markov_score.bp_grid import make_grid
from markov_score.panel_chains import LATTICE, langevin_chain, pearson_chain, skew_chain


def test_the_registered_parameters_are_reproduced():
    sk = skew_chain()
    assert sk.eps == pytest.approx(0.744734, abs=1e-6) and sk.r == pytest.approx(0.858016, abs=1e-6)
    assert langevin_chain().delta == pytest.approx(0.307898, abs=1e-6)
    assert pearson_chain().delta == pytest.approx(0.162369, abs=1e-6)
    # Pearson diffusions have a linear drift: the continuum lag-one correlation is exp(-Delta), so Delta -> -log 0.85
    assert pearson_chain().delta == pytest.approx(-np.log(0.85), rel=2e-3)


@pytest.mark.parametrize("make", [langevin_chain, pearson_chain])
def test_the_lattice_generator_keeps_the_closed_form_stationary_law(make):
    ch = make()
    pi = ch._s ** 2 / np.sum(ch._s ** 2)                     # the closed-form law the generator was built from
    np.testing.assert_allclose(ch.p @ pi, pi, atol=1e-12)
    np.testing.assert_allclose(ch.p.sum(0), 1.0, atol=1e-12)
    assert ch.lag1(ch.delta) == pytest.approx(0.85, abs=1e-9)
    assert np.sum(pi * LATTICE ** 2) - np.sum(pi * LATTICE) ** 2 == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("make", [skew_chain, langevin_chain, pearson_chain])
def test_columns_are_densities_on_e03s_grid(make):
    grid, w = make_grid(8.0, 401)
    lk = make().log_transition_matrix(grid)
    mass = np.exp(lk).T @ w
    # columns whose mass the grid's edge does not cut (from |a| = 5 the Pearson diffusion's step has sd ~1.4: 0.6% of
    # its mass lands beyond 8, a real truncation that every chain on E03's grid shares; stationary mass there is 6e-5)
    inner = np.abs(grid) < 3
    np.testing.assert_allclose(mass[inner], 1.0, atol=2e-3)


def test_a_grid_off_the_lattice_is_refused():
    with pytest.raises(ValueError):
        langevin_chain().log_transition_matrix(np.linspace(-8, 8, 400))


@pytest.mark.parametrize("make", [skew_chain, langevin_chain, pearson_chain])
def test_sampled_moments(make):
    ch = make()
    rng = np.random.default_rng(3)
    A = np.stack([ch.sample(rng, 32) for _ in range(1500)])
    x, y = A[:, 12:-1].ravel(), A[:, 13:].ravel()             # sites past the N(0, 1) start
    assert x.var() == pytest.approx(1.0, abs=0.06)
    assert np.corrcoef(x, y)[0, 1] == pytest.approx(0.85, abs=0.02)
    if ch.name == "skew":
        assert ((x - x.mean()) ** 3).mean() / x.std() ** 3 == pytest.approx(1.0, abs=0.15)
