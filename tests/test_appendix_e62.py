"""E62's appendix: triple-state BP (tools/core/appendix_e62_arp.py) IS pair-state BP when the kernel ignores a_{t-3}, and
the AR(3) built from its partial autocorrelations has lag-one correlation 0.85 and unit variance."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "core"))
import appendix_e62_arp as ap  # noqa: E402
from markov_score.bp_grid import make_grid  # noqa: E402
from markov_score.pair_bp import pair_bp_batch, stationary_pair  # noqa: E402


def test_a_kernel_that_ignores_the_third_lag_is_pair_state_bp():
    grid, w = make_grid(4.0, 13)
    n = len(grid)
    rng = np.random.default_rng(0)
    k2 = rng.gamma(2.0, size=(n, n, n))
    k2 /= np.einsum("kji,k->ji", k2, w)[None]                                   # [k, j, i]
    mu2 = stationary_pair(np.log(k2), w)                                        # [j = a_2, i = a_1]
    k3 = np.repeat(k2[:, :, :, None], n, axis=3)                                # [l, k, j, i], no dependence on i
    mu3 = np.einsum("kji,ji->ijk", k2, mu2)                                     # law of (a_1, a_2, a_3)
    X = rng.standard_normal((3, ap.L))
    m3 = ap.triple_bp_means(grid, w, k3, X, 0.8, 0.36, mu3)
    m2, _ = pair_bp_batch(grid, w, np.log(k2), X, 0.8, 0.36, np.log(mu2))
    np.testing.assert_allclose(m3, m2, atol=1e-10)


def test_the_ar3_has_the_registered_lag_one_correlation_and_variance():
    phi, var = ap.ar3()
    p1, p2, p3 = phi
    # Yule-Walker for an AR(3), solved for (rho_1, rho_2, rho_3):
    #   rho_1 = p1 + p2 rho_1 + p3 rho_2,  rho_2 = p1 rho_1 + p2 + p3 rho_1,  rho_3 = p1 rho_2 + p2 rho_1 + p3
    a = np.array([[1 - p2, -p3, 0.0], [-(p1 + p3), 1.0, 0.0], [-p2, -p1, 1.0]])
    r1, r2, r3 = np.linalg.solve(a, [p1, p2, p3])
    assert abs(r1 - 0.85) < 1e-12
    assert abs((1 - p1 * r1 - p2 * r2 - p3 * r3) - var) < 1e-12               # unit stationary variance
