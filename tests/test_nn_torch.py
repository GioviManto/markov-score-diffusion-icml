"""The torch baselines: shapes, length-agnosticism, a DSM step that decreases the loss, and the
implicit-score-matching objective's exact-trace and Hutchinson forms agreeing in expectation."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from markov_score import nn_torch as nt  # noqa: E402

L = 16
T = (0.05, 0.3, 1.0, 3.0)
HPS = {"mlp": dict(hidden=(32, 32)), "window": dict(radius=2, width=16), "tcn": dict(channels=8, dilations=(1, 2)),
       "bigru": dict(hidden=8), "transformer": dict(d=16, heads=2, layers=1), "unet": dict(channels=8)}


@pytest.mark.parametrize("arch", list(HPS))
def test_forward_shapes_and_length_agnosticism(arch):
    torch.manual_seed(0)
    m = nt.build(arch, L, **HPS[arch])
    y = m(torch.randn(4, L), torch.full((4,), 0.5))
    assert y.shape == (4, L)
    if m.length_agnostic:
        y2 = m(torch.randn(4, 2 * L), torch.full((4,), 0.5))
        assert y2.shape == (4, 2 * L)
    assert nt.n_params(m) > 0


def test_dsm_training_reduces_loss_on_a_gaussian_chain():
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    rho = 0.8
    A = np.zeros((256, L)); A[:, 0] = rng.standard_normal(256)
    for i in range(1, L):
        A[:, i] = rho * A[:, i - 1] + np.sqrt(1 - rho**2) * rng.standard_normal(256)
    m = nt.build("window", L, radius=2, width=16)
    cfg = nt.TrainConfig(steps=300, batch_size=64, lr=3e-3, checkpoints=(10, 300), t_levels=T, seed=0)
    saved = nt.train(m, A, cfg)
    def loss_at(state):
        nt.load_checkpoint(m, state); m.eval()
        r = np.random.default_rng(1)
        a, x, z, t, al, de = nt._draw_batch(A, cfg, r)
        with torch.no_grad():
            return float(nt.dsm_loss(m, a, x, z, t, al, de, cfg))
    assert loss_at(saved[300]) < loss_at(saved[10])


def test_ism_exact_and_hutchinson_agree_in_expectation():
    torch.manual_seed(0)
    m = nt.build("window", L, radius=1, width=8)
    x = torch.randn(64, L); t = torch.full((64,), 0.5)
    al = torch.exp(-t)[:, None]; de = 1 - torch.exp(-2 * t)[:, None]
    cfg_e = nt.TrainConfig(t_levels=T, loss="ism", hutchinson=False)
    cfg_h = nt.TrainConfig(t_levels=T, loss="ism", hutchinson=True)
    rng = np.random.default_rng(0)
    exact = float(nt.ism_loss(m, x, t, al, de, cfg_e, rng))
    hut = np.mean([float(nt.ism_loss(m, x, t, al, de, cfg_h, np.random.default_rng(k))) for k in range(200)])
    assert abs(hut - exact) < 0.05 * max(1.0, abs(exact))
