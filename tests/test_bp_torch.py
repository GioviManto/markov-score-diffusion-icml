"""The torch grid BP equals grid_bp_batch to round-off (the true Laplace kernel; per-sequence noise levels), and its
gradient with respect to the kernel's parameters is correct (finite differences)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.bp_torch import posterior_mean  # noqa: E402
from markov_score.energy_kernel import NeuralEnergyKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402


def _data(n=6, L=9, seed=0):
    rng = np.random.default_rng(seed)
    prior = LaplaceAR1(0.85)
    A = np.stack([prior.sample(rng, L) for _ in range(n)])
    return A, rng


def test_equals_grid_bp_batch_at_one_level():
    grid, w = make_grid(8.0, 101)
    lk = LaplaceAR1(0.85).log_transition_matrix(grid)
    A, rng = _data()
    al, de = alpha_delta(0.1)
    X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
    ref, _ = grid_bp_batch(grid, w, lk, X, al, de)
    got = posterior_mean(torch.as_tensor(grid), torch.as_tensor(w), torch.as_tensor(lk), torch.as_tensor(X), al, de)
    np.testing.assert_allclose(got.numpy(), ref, atol=1e-10, rtol=0)


def test_per_sequence_levels_equal_one_call_per_level():
    grid, w = make_grid(8.0, 101)
    lk = LaplaceAR1(0.85).log_transition_matrix(grid)
    A, rng = _data(n=4)
    ts = np.array([0.05, 0.3, 1.0, 3.0])
    ad = np.array([alpha_delta(t) for t in ts])
    X = ad[:, :1] * A + np.sqrt(ad[:, 1:]) * rng.standard_normal(A.shape)
    got = posterior_mean(torch.as_tensor(grid), torch.as_tensor(w), torch.as_tensor(lk), torch.as_tensor(X),
                         torch.as_tensor(ad[:, 0]), torch.as_tensor(ad[:, 1])).numpy()
    for b, t in enumerate(ts):
        al, de = alpha_delta(t)
        ref, _ = grid_bp_batch(grid, w, lk, X[b:b + 1], al, de)
        np.testing.assert_allclose(got[b], ref[0], atol=1e-10, rtol=0)


def test_gradient_through_the_energy_kernel_matches_finite_differences():
    grid_np, w_np = make_grid(8.0, 41)
    grid, w = torch.as_tensor(grid_np), torch.as_tensor(w_np)
    k = NeuralEnergyKernel.init(hidden=4, seed=1, half_width=8.0)
    net = k._net()
    with torch.no_grad():                                   # move off the uniform start so the gradient is generic
        for p in net.parameters():
            p.add_(0.3 * torch.randn_like(p))
    A, rng = _data(n=3, L=5)
    al, de = alpha_delta(0.2)
    X = torch.as_tensor(al * A + np.sqrt(de) * rng.standard_normal(A.shape))

    def loss():
        return posterior_mean(grid, w, k._log_k(net, grid_np), X, al, de).pow(2).sum()

    p = next(net.parameters())
    l0 = loss()
    (g,) = torch.autograd.grad(l0, p)
    eps = 1e-6
    with torch.no_grad():
        p.view(-1)[0] += eps
    l1 = loss().detach()
    with torch.no_grad():
        p.view(-1)[0] -= eps
    assert abs(float((l1 - l0.detach()) / eps) - float(g.view(-1)[0])) < 1e-4 * max(1.0, abs(float(g.view(-1)[0])))
