"""E58's driver: scoring the TRUE kernel through the clean-fit path gives the reference itself (zero error), the clean
maximum likelihood recovers the Laplace chain's parameters, the clean fit never lowers the likelihood, the held-out
likelihood is a likelihood, and the networks are E03's screened winners."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
pytest.importorskip("torch")
import e58_clean_kernel as e58  # noqa: E402

from markov_score.em import clean_statistics  # noqa: E402
from markov_score.energy_kernel import NeuralEnergyKernel  # noqa: E402
from markov_score.kernels import LaplaceAR1Kernel  # noqa: E402

QUICK = dict(e58.SETTINGS, **e58.QUICK)


def test_the_true_laplace_kernel_scores_zero():
    ctx = e58.context(QUICK, "laplace", 0, 32)
    true = LaplaceAR1Kernel(rho=0.85, b=float(np.sqrt((1 - 0.85**2) / 2)))
    risk = e58.e03._risk(e58.score_pairs(e58.kernel_mean_fn(ctx, true), ctx["test"]))
    assert risk < 1e-10


def test_the_field_context_carries_its_stationary_law_and_covariance():
    ctx = e58.context(QUICK, "T0.3_h1_L32", 0, 32)
    assert ctx["log_mu"] is not None and ctx["mu_fn"] is not None
    np.testing.assert_allclose(np.exp(ctx["log_mu"]) @ ctx["w"], 1.0, atol=1e-8)
    np.testing.assert_allclose(ctx["cov"], ctx["cov"].T)
    risk = e58.e03._risk(e58.score_pairs(e58.lmmse_mean_fn(ctx["cov"]), ctx["test"]))
    assert 0.0 < risk < 1.0                                   # the linear floor: worse than BP, better than nothing


def test_clean_maximum_likelihood_recovers_the_laplace_chain():
    rng = np.random.default_rng(0)
    prior = e58.LaplaceAR1(0.85)
    A = np.stack([prior.sample(rng, 32) for _ in range(2048)])
    grid, w = e58.make_grid(8.0, 401)
    saved, _ = e58.fit_clean_ladder(LaplaceAR1Kernel(rho=0.3, b=0.6), grid, clean_statistics(grid, A), {1, 2})
    k = saved[2]
    assert abs(k.rho - 0.85) < 0.02 and abs(k.b - prior.b) / prior.b < 0.05


def test_the_clean_energy_fit_never_lowers_the_likelihood():
    rng = np.random.default_rng(1)
    prior = e58.LaplaceAR1(0.85)
    A = np.stack([prior.sample(rng, 32) for _ in range(256)])
    grid, w = e58.make_grid(6.0, 61)
    stats = clean_statistics(grid, A)
    k0 = NeuralEnergyKernel.init(hidden=8, seed=0, half_width=6.0)
    saved, sec = e58.fit_clean_ladder(k0, grid, stats, {0, 2, 5, 10})
    nll = [e58.clean_nll(saved[i], grid, stats) for i in sorted(saved)]
    assert all(b <= a + 1e-9 for a, b in zip(nll, nll[1:])) and nll[-1] < nll[0] - 0.1 and sec > 0


def test_the_held_out_nll_of_the_uniform_start_is_log_of_the_width():
    grid, w = e58.make_grid(6.0, 61)
    A = np.random.default_rng(2).standard_normal((64, 32))
    k0 = NeuralEnergyKernel.init(hidden=8, seed=0, half_width=6.0)
    np.testing.assert_allclose(e58.clean_nll(k0, grid, clean_statistics(grid, A)), np.log(12.0), rtol=1e-10)


def test_the_networks_are_e03s_winners():
    path = ROOT / "outputs" / "E03" / "winners.json"
    if not path.is_file():
        pytest.skip("outputs/E03/winners.json is a fetched artefact")
    w = json.loads(path.read_text())
    for arch, conf in e58.SETTINGS["nets"].items():
        assert (conf["hp"], conf["lr"], conf["mode"]) == (w[arch]["hp"], w[arch]["lr"], w[arch]["mode"]), arch
