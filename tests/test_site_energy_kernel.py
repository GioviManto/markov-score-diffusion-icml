"""SiteEnergyKernel (E68) and the window CNN with position features."""
from types import SimpleNamespace

import numpy as np
import torch

from markov_score import nn_torch as nt
from markov_score.bp_grid import make_grid
from markov_score.nonstationary import NonstationaryLaplaceChain
from markov_score.site_bp import site_bp_batch
from markov_score.site_energy_kernel import SiteEnergyKernel


def test_starts_uniform_and_every_column_is_a_density():
    grid, w = make_grid(8.0, 61)
    k = SiteEnergyKernel.init(hidden=8, seed=0, half_width=8.0, n_links=5)
    lk = k.log_transition_matrices(grid)
    assert lk.shape == (5, 61, 61)
    assert np.allclose(np.einsum("kij,i->kj", np.exp(lk), w), 1.0)
    assert np.allclose(lk, lk[0, 0, 0])                                  # uniform before any data
    assert k.theta.size == 3 * 8 + 8 + 8 * 8 + 8 + 8 + 1


def test_m_step_raises_q_and_learns_the_links_apart():
    grid, w = make_grid(8.0, 61)
    chain = NonstationaryLaplaceChain("switch", 1.0, n_sites=8)
    rng = np.random.default_rng(0)
    A = chain.sample(rng, 400)
    al, de = np.exp(-0.1), 1 - np.exp(-0.2)
    X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
    r = site_bp_batch(grid, w, chain.log_kernels(grid), X, al, de, chain.log_mu(grid), pairs=True)
    stats = SimpleNamespace(xi=r.xi, n_edges=int(r.xi.sum().round()))
    k = SiteEnergyKernel.init(hidden=8, seed=1, half_width=8.0, n_links=7, n_inner=20)

    def q(kernel):
        return float((r.xi * kernel.log_transition_matrices(grid)).sum() / stats.n_edges)

    qs = [q(k)]
    for _ in range(15):
        k = k.m_step(stats, grid)
        qs.append(q(k))
    assert all(b >= a - 1e-12 for a, b in zip(qs, qs[1:]))              # generalised M-step: Q never decreases
    assert qs[-1] > qs[0] + 0.5
    lk = np.exp(k.log_transition_matrices(grid))
    j = np.argmin(np.abs(grid - 1.0))                                    # from a = 1: the mean of a' per link
    means = (lk[:, :, j] * (w * grid)[None, :]).sum(axis=1)
    assert means[0] > means[-1] + 0.1                                    # alpha 0.95 on the first links, 0.75 on the last


def test_window_pos_shapes_and_position_dependence():
    torch.manual_seed(0)
    model = nt.build("window_pos", 16, radius=2, width=8)
    x = torch.zeros(3, 16)
    out = model(x, torch.full((3,), 0.3))
    assert out.shape == (3, 16)
    assert out.detach().std(dim=1).max() > 0                             # zero input: only the position varies
    base = nt.build("window", 16, radius=2, width=8)
    assert torch.allclose(base(x, torch.full((3,), 0.3)).std(dim=1), torch.zeros(3), atol=1e-7)
