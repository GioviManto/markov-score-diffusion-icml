"""Chunked batch BP equals the unchunked pass (chains are independent)."""

import numpy as np

from markov_score.bp_grid import grid_bp_batch, make_grid
from markov_score.noising import alpha_delta
from markov_score.priors import LaplaceAR1


def test_chunking_is_exact():
    grid, weights = make_grid(6.0, 121)
    prior = LaplaceAR1(0.85)
    log_k = prior.log_transition_matrix(grid)
    rng = np.random.default_rng(3)
    A = np.stack([prior.sample(rng, 16) for _ in range(300)])
    al, de = alpha_delta(0.4)
    X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
    m0, v0, e0 = grid_bp_batch(grid, weights, log_k, X, al, de, return_evidence=True, chunk=None)
    m1, v1, e1 = grid_bp_batch(grid, weights, log_k, X, al, de, return_evidence=True, chunk=64)
    m2, v2 = grid_bp_batch(grid, weights, log_k, X, al, de)  # default chunk 256 < 300 rows
    assert np.allclose(m0, m1, rtol=1e-12, atol=1e-12) and np.allclose(v0, v1, rtol=1e-12, atol=1e-12)
    assert np.allclose(e0, e1, rtol=1e-12, atol=1e-10)
    assert np.allclose(m0, m2, rtol=1e-12, atol=1e-12)
    assert m1.shape == (300, 16) and e1.shape == (300,)
