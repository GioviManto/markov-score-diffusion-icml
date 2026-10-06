"""Kalman filter and Rauch--Tung--Striebel smoother for the linear-Gaussian chain under the OU channel.

The state-space model is the paper's Gaussian linear chain seen through the channel:

    a_1 ~ N(0, v0),   a_{i+1} = rho a_i + eps_i,  eps_i ~ N(0, q),   x_i = alpha a_i + sqrt(delta) z_i,

with alpha = e^{-t} and delta = Delta_t = 1 - e^{-2t} (docs/NOTATION.md). The recursions are the textbook
covariance-form ones (Rauch, Tung and Striebel 1965; Sarkka, Bayesian Filtering and Smoothing, 2013, Thms 4.2
and 8.2), written here without reference to the BP code: `bp_gaussian.gaussian_chain_bp` is the two-filter
(information-form) smoother and `bp_grid` the quadrature of the functional recursion, so the Gaussian sanity
check (E39; the paper's proposition on the Gaussian chain) compares three different algorithms for one
conditional law rather than one algorithm with itself.

Batched over chains: X has shape (B, L). The filter and smoother variances do not depend on the data, so they
are returned once, shape (L,).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KalmanResult:
    """filtered_means (B, L): E[a_i | x_1..x_i]; filtered_vars (L,).
    smoothed_means (B, L): E[a_i | x_1..x_L]; smoothed_vars (L,)."""

    filtered_means: np.ndarray
    filtered_vars: np.ndarray
    smoothed_means: np.ndarray
    smoothed_vars: np.ndarray


def kalman_rts(X: np.ndarray, rho: float, q: float, alpha: float, delta: float, v0: float = 1.0) -> KalmanResult:
    """Kalman filter (predict, update) followed by the RTS backward pass."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n_chains, n_sites = X.shape
    mf = np.empty((n_chains, n_sites))
    vf = np.empty(n_sites)
    for i in range(n_sites):
        if i == 0:
            m_pred, v_pred = np.zeros(n_chains), float(v0)
        else:
            m_pred, v_pred = rho * mf[:, i - 1], rho**2 * vf[i - 1] + q
        gain = alpha * v_pred / (alpha**2 * v_pred + delta)
        mf[:, i] = m_pred + gain * (X[:, i] - alpha * m_pred)
        vf[i] = (1.0 - gain * alpha) * v_pred
    ms, vs = mf.copy(), vf.copy()
    for i in range(n_sites - 2, -1, -1):
        v_pred = rho**2 * vf[i] + q
        g = rho * vf[i] / v_pred
        ms[:, i] = mf[:, i] + g * (ms[:, i + 1] - rho * mf[:, i])
        vs[i] = vf[i] + g**2 * (vs[i + 1] - v_pred)
    return KalmanResult(filtered_means=mf, filtered_vars=vf, smoothed_means=ms, smoothed_vars=vs)


def closed_form_gaussian(X: np.ndarray, sigma0: np.ndarray, alpha: float, delta: float):
    """The Gaussian chain's exact posterior mean and score: <a> = alpha Sigma0 Sigma_t^{-1} x and
    S(x, t) = -Sigma_t^{-1} x with Sigma_t = alpha^2 Sigma0 + delta I. Returns (means, scores), each (B, L)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    sigma_t = alpha**2 * sigma0 + delta * np.eye(sigma0.shape[0])
    w = np.linalg.solve(sigma_t, X.T).T          # Sigma_t^{-1} x, per chain
    return alpha * w @ sigma0.T, -w
