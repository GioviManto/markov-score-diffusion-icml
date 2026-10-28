"""The DDPM U-Net of E55 and its training loop: shapes at every chain length E55 uses, the zero-initialised output,
determinism, EMA checkpoints that are snapshots (not aliases), and a learning check on a Gaussian chain whose exact
score is known in closed form."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from markov_score import theory  # noqa: E402
from markov_score import unet_ddpm as ud  # noqa: E402

T = (0.05, 0.3, 1.0, 3.0)


@pytest.mark.parametrize("L", [32, 64, 128, 512])
def test_shapes_at_every_length_e55_uses(L):
    torch.manual_seed(0)
    m = ud.UNetDDPM(base=8)
    y = m(torch.randn(3, L), torch.full((3,), 0.5))
    assert y.shape == (3, L)


def test_length_must_divide_by_the_downsampling():
    m = ud.UNetDDPM(base=8)
    with pytest.raises(ValueError):
        m(torch.randn(2, 30), torch.full((2,), 0.5))


def test_output_starts_at_zero_so_the_first_posterior_mean_is_x_over_alpha():
    m = ud.UNetDDPM(base=8)
    X = np.random.default_rng(0).standard_normal((5, 32))
    for t in T:
        np.testing.assert_allclose(ud.posterior_mean(m, X, t), X / np.exp(-t), rtol=1e-6, atol=1e-6)


def test_time_embedding_separates_the_frozen_levels():
    e = ud.timestep_embedding(torch.tensor((0.05, 0.0725, 0.1053, 3.0)), 32)
    d = torch.cdist(e, e)
    assert float(d[~torch.eye(4, dtype=bool)].min()) > 0.1


def _gaussian_chain(rng, n, L, rho=0.8):
    A = np.zeros((n, L))
    A[:, 0] = rng.standard_normal(n)
    for i in range(1, L):
        A[:, i] = rho * A[:, i - 1] + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    return A


def test_same_seed_same_checkpoints_and_ema_checkpoints_are_snapshots():
    A = _gaussian_chain(np.random.default_rng(1), 64, 32)
    cfg = ud.DSMConfig(steps=40, batch_size=16, lr=1e-3, warmup=5, checkpoints=(20, 40), seed=3)
    s1 = ud.train_dsm(ud.build(7, base=8), A, T, cfg)
    s2 = ud.train_dsm(ud.build(7, base=8), A, T, cfg)
    assert sorted(s1) == [20, 40]
    for k in s1[40]:
        assert torch.equal(s1[40][k], s2[40][k])
    assert any(not torch.equal(s1[20][k], s1[40][k]) for k in s1[20])


def test_the_ema_lags_the_raw_weights():
    A = _gaussian_chain(np.random.default_rng(2), 64, 32)
    m = ud.UNetDDPM(base=8)
    cfg = ud.DSMConfig(steps=60, batch_size=16, lr=1e-3, warmup=5, ema_decay=0.99, checkpoints=(60,), seed=0)
    saved = ud.train_dsm(m, A, T, cfg)
    raw = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    assert any(not torch.allclose(saved[60][k], raw[k]) for k in raw)


def test_dsm_learns_the_score_of_a_gaussian_chain():
    """The exact posterior mean of a Gaussian chain is linear (theory.lmmse_posterior_mean). After a short training the
    U-Net's score error must be well below that of its own starting point (posterior mean x / alpha)."""
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    L, rho = 32, 0.8
    A = _gaussian_chain(rng, 512, L, rho)
    sigma = rho ** np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
    cfg = ud.DSMConfig(steps=1500, batch_size=64, lr=1e-3, warmup=50, ema_decay=0.99, checkpoints=(1500,), seed=0)
    m = ud.UNetDDPM(base=16)
    saved = ud.train_dsm(m, A, T, cfg)
    m.load_state_dict(saved[1500])
    Xc = _gaussian_chain(rng, 256, L, rho)
    num = den0 = den = 0.0
    for t in T:
        al, de = np.exp(-t), 1 - np.exp(-2 * t)
        X = al * Xc + np.sqrt(de) * rng.standard_normal(Xc.shape)
        s_ref = -(X - al * theory.lmmse_posterior_mean(X, sigma, t)) / de
        s_hat = -(X - al * ud.posterior_mean(m, X, t)) / de
        s_0 = -(X - al * (X / al)) / de
        num += ((s_hat - s_ref) ** 2).sum(); den0 += ((s_0 - s_ref) ** 2).sum(); den += (s_ref ** 2).sum()
    assert np.sqrt(num / den) < 0.25 * np.sqrt(den0 / den)


def test_continuous_levels_draw_log_uniform_in_the_range():
    import math
    seen = []
    m = ud.build(0, base=8)
    orig = m.forward
    def spy(x, t):
        seen.append(t.detach().cpu().numpy().copy()); return orig(x, t)
    m.forward = spy
    A = _gaussian_chain(np.random.default_rng(3), 64, 32)
    cfg = ud.DSMConfig(steps=200, batch_size=64, lr=1e-3, warmup=5, checkpoints=(200,), seed=0, t_range=(0.005, 3.0))
    ud.train_dsm(m, A, T, cfg)
    t = np.concatenate(seen)
    assert t.min() >= 0.005 * (1 - 1e-6) and t.max() <= 3.0 * (1 + 1e-6)
    assert len(np.unique(np.round(t, 6))) > 1000                     # continuous, not the twelve levels
    u = (np.log(t) - math.log(0.005)) / (math.log(3.0) - math.log(0.005))
    assert abs(u.mean() - 0.5) < 0.02                                  # uniform in log t
