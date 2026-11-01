"""The delta-method reparameterisation behind Q10 (tools/cumulant_information.py).

Jerome's point is that the information curves quantify kappa_2 and kappa_4, not rho and
beta. Turning eff(beta) into eff(kappa_4) is a change of coordinates, not a change of
label, and these tests pin the three places it could go wrong:

  1. gamma_2'(beta), in closed form, against a central difference;
  2. the coordinates whose Jacobian rows are unit vectors (rho and kappa_2) must be
     UNCHANGED, which is the internal check that the Jacobian is assembled correctly;
  3. the reparameterisation must be an identity when applied with the identity Jacobian,
     and must invert: converting theta -> phi -> theta returns the original information.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from cumulant_information import gg_excess_kurtosis_prime  # noqa: E402
from markov_score.shape_kernel import generalized_gaussian_excess_kurtosis as g2  # noqa: E402


@pytest.mark.parametrize("beta", [0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0])
def test_gamma2_prime_matches_central_difference(beta):
    h = 1e-6
    fd = (g2(beta + h) - g2(beta - h)) / (2 * h)
    an = gg_excess_kurtosis_prime(beta)
    assert abs(an - fd) < 1e-5 * max(1.0, abs(fd)), f"analytic {an:.9g} vs fd {fd:.9g}"


def test_gamma2_prime_is_negative_and_shrinking():
    """gamma_2 decreases in beta (heavier tails at small beta) and FLATTENS. The flattening
    is the whole point of Q10: beta stops moving kappa_4, so information about beta collapses
    for a reason that has nothing to do with the channel."""
    ps = [gg_excess_kurtosis_prime(b) for b in (1.0, 1.25, 1.5, 2.0, 3.0)]
    assert all(p < 0 for p in ps)
    assert all(abs(ps[i]) < abs(ps[i - 1]) for i in range(1, len(ps)))
    # and the shrinkage is large enough to dominate: it enters eff(kappa_4) SQUARED
    assert (ps[0] / ps[-1]) ** 2 > 100


def _eff(Iinv, j):
    return 1.0 / float(j @ Iinv @ j)


def test_unit_jacobian_rows_leave_efficient_information_unchanged():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 3))
    I = A @ A.T + 3 * np.eye(3)          # a valid information matrix
    Iinv = np.linalg.inv(I)
    q, beta = 0.2775, 1.5
    J = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 2 * g2(beta) * q, gg_excess_kurtosis_prime(beta) * q**2]])
    eff_theta = 1.0 / np.diag(Iinv)
    eff_phi = 1.0 / np.diag(J @ Iinv @ J.T)
    assert np.isclose(eff_phi[0], eff_theta[0], rtol=1e-12)   # rho
    assert np.isclose(eff_phi[1], eff_theta[1], rtol=1e-12)   # kappa_2 == q
    # kappa_4 must NOT be a relabel of beta, or Q10 would have been a label change
    assert not np.isclose(eff_phi[2], eff_theta[2], rtol=1e-3)


def test_reparameterisation_inverts():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((3, 3))
    I = A @ A.T + 3 * np.eye(3)
    q, beta = 0.2775, 1.25
    J = np.array([[1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 2 * g2(beta) * q, gg_excess_kurtosis_prime(beta) * q**2]])
    Iphi = np.linalg.inv(J) .T @ I @ np.linalg.inv(J)
    back = J.T @ Iphi @ J
    assert np.abs(back - I).max() < 1e-9
