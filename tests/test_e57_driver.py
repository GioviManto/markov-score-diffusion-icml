"""E57's driver: the network configurations are E03's screened winners, the exact chain covariance and the linear floor
are right on the Gaussian chain (where the linear estimator IS the exact one), and the one-view interface splits the
training chains over the twelve levels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
pytest.importorskip("torch")
import e57_chain_families as e57  # noqa: E402

from markov_score.bp_grid import make_grid  # noqa: E402
from markov_score.priors import GaussianAR1  # noqa: E402


def test_the_networks_are_e03s_winners():
    path = ROOT / "outputs" / "E03" / "winners.json"
    if not path.is_file():
        pytest.skip("outputs/E03/winners.json is a fetched artefact")
    w = json.loads(path.read_text())
    for arch, conf in e57.SETTINGS["nets"].items():
        assert conf["hp"] == w[arch]["hp"] and conf["lr"] == w[arch]["lr"] and conf["mode"] == w[arch]["mode"], arch


def test_chain_covariance_is_exact_on_the_gaussian_chain():
    grid, w = make_grid(8.0, 401)
    mean, cov = e57.chain_covariance(GaussianAR1(rho=0.85).log_transition_matrix(grid), grid, w, n_sites=12)
    i = np.arange(12)
    np.testing.assert_allclose(cov, 0.85 ** np.abs(i[:, None] - i[None, :]), atol=1e-9)
    np.testing.assert_allclose(mean, 0.0, atol=1e-12)


def test_the_linear_floor_is_the_exact_score_on_the_gaussian_chain():
    cfg = dict(e57.SETTINGS, n_val=8, n_test=64)
    grid, w = make_grid(8.0, 401)
    prior = GaussianAR1(rho=0.85)
    log_k = prior.log_transition_matrix(grid)
    _, test = e57.e03._bundle(prior, log_k, grid, w, "e57-test-gauss", 0, cfg["n_test"])
    mean, cov = e57.chain_covariance(log_k, grid, w)
    assert e57.e03._risk(e57._lmmse_pairs(cov, mean, test)) < 1e-6


def test_one_view_groups_cover_every_chain_once():
    A = np.random.default_rng(0).standard_normal((100, e57.N_SITES))
    groups = e57._one_view_groups("tanh", A, 0)
    assert len(groups) == 12 and sum(len(X) for X, _, _ in groups) == 100
    for (X, al, de), t in zip(groups, e57.T_SCHEDULE):
        np.testing.assert_allclose((al, de), (np.exp(-t), 1 - np.exp(-2 * t)))


def test_every_chain_is_built():
    assert [e57.chain_prior(c).name for c in e57.CHAINS] == ["gauss_mix_kappa0.6", "tanh_drift", "hetero_c0.25"]
