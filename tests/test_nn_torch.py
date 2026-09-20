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


def test_fastconv1d_matches_native_conv1d():
    import torch
    import torch.nn.functional as F
    from markov_score.nn_torch import FastConv1d
    torch.manual_seed(0)
    for (cin, cout, k, pad, dil) in [(1, 64, 5, 2, 1), (64, 64, 3, 1, 1), (64, 64, 3, 4, 4), (64, 1, 1, 0, 1), (32, 16, 3, 1, 1)]:
        m = FastConv1d(cin, cout, k, padding=pad, dilation=dil)
        x = torch.randn(7, cin, 32)
        ref = F.conv1d(x, m.weight, m.bias, padding=pad, dilation=dil)
        assert torch.allclose(m(x), ref, atol=1e-5, rtol=1e-5), (cin, cout, k, pad, dil)
        # gradients agree too
        x.requires_grad_(True)
        (m(x) ** 2).sum().backward()
        g1 = x.grad.clone(); x.grad = None
        (F.conv1d(x, m.weight, m.bias, padding=pad, dilation=dil) ** 2).sum().backward()
        assert torch.allclose(g1, x.grad, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("radius,L", [(1, 16), (2, 16), (4, 32), (8, 37)])
def test_coloured_trace_equals_exact_trace_for_the_window_head(radius, L):
    """2r+1 coloured probes give the exact divergence of a radius-r window map, at every site
    including the zero-padded ends, and the exact loop is what E03 trained the window with."""
    torch.manual_seed(1)
    m = nt.build("window", L, radius=radius, width=8)
    x = torch.randn(5, L, requires_grad=True); t = torch.full((5,), 0.7)
    s = m(x, t)
    ex = nt.divergence(s, x, "exact", create_graph=False)
    co = nt.divergence(s, x, "coloured", radius=nt.receptive_radius(m), create_graph=False)
    assert torch.allclose(ex, co, atol=1e-5, rtol=1e-5)
    hut = np.mean([float(nt.divergence(s, x, "hutchinson", rng=np.random.default_rng(k), create_graph=False).mean()) for k in range(400)])
    assert abs(hut - float(ex.mean())) < 0.1 * max(1.0, abs(float(ex.mean())))


def test_coloured_trace_refuses_a_dense_map():
    m = nt.build("transformer", L, d=16, heads=2, layers=1)
    x = torch.randn(3, L, requires_grad=True); s = m(x, torch.full((3,), 0.5))
    with pytest.raises(ValueError):
        nt.divergence(s, x, "coloured", radius=nt.receptive_radius(m))


def test_ism_objective_helper_matches_the_exact_loop_and_train_config_trace():
    torch.manual_seed(2)
    m = nt.build("window", L, radius=2, width=8)
    den = nt.Denoiser(m, "eps")
    X = np.random.default_rng(0).standard_normal((12, L)); groups = [(X[:6], 0.3), (X[6:], 1.0)]
    a = nt.ism_objective(den, groups, "exact"); b = nt.ism_objective(den, groups, "coloured")
    assert abs(a - b) < 1e-4 * max(1.0, abs(a))
    cfg = nt.TrainConfig(t_levels=T, loss="ism", trace="coloured")
    assert nt.trace_mode(cfg) == "coloured"
    assert nt.trace_mode(nt.TrainConfig(loss="ism", hutchinson=True)) == "hutchinson"
    assert nt.trace_mode(nt.TrainConfig(loss="ism")) == "exact"
    assert nt.pick_device("cpu") == "cpu"


def test_ism_training_on_a_declared_device_runs_and_returns_cpu_states():
    torch.set_num_threads(1)
    m = nt.build("window", L, radius=1, width=8)
    X = np.random.default_rng(3).standard_normal((40, L))
    cfg = nt.TrainConfig(steps=20, batch_size=8, lr=1e-3, loss="ism", trace="coloured", checkpoints=(10, 20), t_levels=T, seed=0, device="cpu")
    saved = nt.train(m, None, cfg, X_noisy_groups=[(X[:20], 0.3), (X[20:], 1.0)])
    assert set(saved) == {10, 20} and all(v.device.type == "cpu" for v in saved[20].values())
    den = nt.Denoiser(nt.load_checkpoint(m, saved[20]), "eps")
    assert den.posterior_mean(X[:4], 0.3).shape == (4, L)
