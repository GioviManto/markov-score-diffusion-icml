"""The quantile networks learn calibrated quantiles of a Gaussian AR(1) chain: the trained 5-95% band covers about 90%
of fresh targets, and the median is close to the conditional mean rho^h a."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
from markov_score import forecast as fc  # noqa: E402
from markov_score import quantile_nets as qn  # noqa: E402


def _examples(rng, n, rho=0.85, window=8, horizons=(1, 2)):
    L = window + max(horizons)
    a = np.zeros((n, L)); a[:, 0] = rng.standard_normal(n)
    for i in range(1, L):
        a[:, i] = rho * a[:, i - 1] + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    return a[:, :window], np.stack([a[:, window - 1 + h] for h in horizons], axis=1)


@pytest.mark.parametrize("arch", ["qmlp", "qgru"])
def test_the_networks_learn_calibrated_quantiles(arch):
    rng = np.random.default_rng(0)
    X, Y = _examples(rng, 4000)
    Xv, Yv = _examples(rng, 1000)
    Xt, Yt = _examples(rng, 2000)
    model, info = qn.train(arch, X, Y, Xv, Yv, fc.LEVELS, steps=1500, checkpoints={500, 1500}, seed=0)
    q = qn.predict(model, Xt)
    cov = fc.coverage(q[:, 0, :], Yt[:, 0]).mean()
    assert 0.85 < cov < 0.95, cov
    median = q[:, 0, 9]
    assert np.sqrt(np.mean((median - 0.85 * Xt[:, -1]) ** 2)) < 0.12
    assert info["n_params"] > 0 and info["checkpoint"] in (500, 1500)
