"""E59's data: the wind file is the registered one, the registered transform gives a standardised training series, the
splits and window counts are the registered ones, and the network examples line up with the forecast targets."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
pytest.importorskip("torch")
import e59_forecasting as e59  # noqa: E402


@pytest.mark.skipif(not e59.WIND.is_file(), reason="data/e59 not present")
def test_the_wind_series_is_the_registered_one():
    z, split = e59.wind_series()
    assert len(z) == 87672 and np.isfinite(z).all()
    assert [int((split == k).sum()) for k in (0, 1, 2)] == [61368, 8760, 17544]
    np.testing.assert_allclose([z[split == 0].mean(), z[split == 0].std()], [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(np.corrcoef(z[split == 0][:-1], z[split == 0][1:])[0, 1], 0.9731, atol=1e-4)


@pytest.mark.skipif(not e59.WIND.is_file(), reason="data/e59 not present")
def test_the_wind_windows_are_the_registered_ones():
    cfg = dict(e59.SETTINGS)
    A, V, T = e59.data(cfg, "wind", 0, 1917)
    assert A.shape == (1917, 32) and V.shape == (273, 32) and T.shape == (2190, 32)
    z, split = e59.wind_series()
    full = e59._windows(z[split == 0], 32)
    assert len(np.unique(A, axis=0)) == 1917                     # all training windows, each once
    np.testing.assert_array_equal(np.unique(A, axis=0), np.unique(full, axis=0))
    A2, _, _ = e59.data(cfg, "wind", 1, 128)
    assert A2.shape == (128, 32)


def test_setar_data_shapes_and_targets():
    cfg = dict(e59.SETTINGS, n_val_setar=16, n_test_setar=32)
    A, V, T = e59.data(cfg, "setar", 0, 64)
    assert A.shape == (64, 32) and V.shape == (16, 32) and T.shape == (32, 32)
    X, Y = e59.net_examples(A, A)
    assert X.shape == (64 * 9, 16) and Y.shape == (64 * 9, 4)
    np.testing.assert_array_equal(X[:64, -1], A[:, 15])           # the first origin is site 16
    np.testing.assert_array_equal(Y[:64, 3], A[:, 15 + 8])        # and its h = 8 target


def test_noise_is_the_registered_channel():
    cfg = dict(e59.SETTINGS)
    A = np.zeros((4000, 32))
    X, al, de = e59.noisy(cfg, A, "setar", "test", 0, 1)
    np.testing.assert_allclose([al, de], [np.exp(-0.2), 1 - np.exp(-0.4)])
    assert abs(X.std() - np.sqrt(de)) < 0.01
