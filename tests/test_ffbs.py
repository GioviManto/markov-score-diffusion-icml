"""FFBS draws exact samples of the grid posterior: path frequencies on a tiny grid, marginals on a real one."""
import itertools

import numpy as np

from markov_score.bp_grid import grid_bp_batch, make_grid
from markov_score.ffbs import grid_ffbs
from markov_score.noising import alpha_delta
from markov_score.priors import LaplaceAR1


def test_path_frequencies_match_the_enumerated_posterior():
    grid, w = make_grid(2.0, 4)
    prior = LaplaceAR1(0.6)
    log_K = prior.log_transition_matrix(grid)
    al, de = alpha_delta(0.4)
    x = np.array([[0.7, -0.2, 1.1]])
    mu = np.exp(-0.5 * grid**2)
    post = {}
    for path in itertools.product(range(4), repeat=3):
        p = mu[path[0]] * w[path[0]]
        for i in range(1, 3):
            p *= np.exp(log_K[path[i], path[i - 1]]) * w[path[i]]
        for i in range(3):
            p *= np.exp(-0.5 * (x[0, i] - al * grid[path[i]]) ** 2 / de)
        post[path] = p
    tot = sum(post.values())
    n = 200_000
    S = grid_ffbs(grid, w, log_K, x, al, de, np.random.default_rng(0), n_samples=n)[:, 0, :]
    idx = np.rint((S + 2.0) / (grid[1] - grid[0])).astype(int)
    keys, counts = np.unique(idx, axis=0, return_counts=True)
    freq = {tuple(k): c / n for k, c in zip(keys, counts)}
    for path, p in post.items():
        q = p / tot
        assert abs(freq.get(path, 0.0) - q) < 5 * np.sqrt(q * (1 - q) / n) + 1e-4


def test_site_marginals_match_grid_bp():
    grid, w = make_grid(8.0, 201)
    prior = LaplaceAR1(0.85)
    rng = np.random.default_rng(1)
    A = np.stack([prior.sample(rng, 10) for _ in range(4)])
    al, de = alpha_delta(0.3)
    X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
    log_K = prior.log_transition_matrix(grid)
    means, variances = grid_bp_batch(grid, w, log_K, X, al, de)
    S = grid_ffbs(grid, w, log_K, X, al, de, np.random.default_rng(2), n_samples=20_000)
    se = np.sqrt(variances / S.shape[0])
    assert np.all(np.abs(S.mean(axis=0) - means) < 5 * se)
    assert np.allclose(S.var(axis=0), variances, rtol=0.05)
