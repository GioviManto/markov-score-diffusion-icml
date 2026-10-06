"""Order-by-order information decay under the Ornstein–Uhlenbeck channel.

Everything here is either an exact identity or an explicit leading-order constant of
the high-noise expansion written out in ``docs/THEORY_NOTES.md``. Nothing fits data.

Conventions (``markov_score.noising``): ``x_t = alpha a + sqrt(Delta) z`` with
``alpha = e^{-t}``, ``Delta = 1 - e^{-2t}``, ``z ~ N(0, I)``. ``He_n`` are the
probabilists' Hermite polynomials.

Contents
--------
Exact objects
    ``he``                      probabilists' Hermite polynomial values
    ``hermite_moment_attenuation``  E[He_n(x_t)] = e^{-nt} E[He_n(a)] (Theorem A), scalar
    ``chain_covariance``        Sigma_0(rho, q, L) for a_1 ~ N(0,1), a_i = rho a_{i-1} + eps
    ``chain_kappa4``            the fourth-cumulant tensor of the chain, (L, L, L, L)
    ``gaussian_chain_fisher``   closed-form Fisher information of x_t about (rho, q)
                                for the Gaussian chain (Theorem B3): exact at every t
    ``lmmse_posterior_mean``, ``lmmse_score``, ``isotropic_score``  the oracle ladder

Leading-order constants (Theorem B)
    ``score_gap_constant``      C_4 = ||kappa_4||_F^2 / 6:
                                E||S_t - S_t^G||^2 = C_4 e^{-8t} (1 + O(e^{-2t} lambda_max))
    ``shape_information_constant``  ||d_theta kappa_4||_F^2 / 24
    ``correlation_information_constant``  (1/2) ||d_theta Sigma_0||_F^2
    ``crossover_time``          t_* = (1/2) log lambda_max(Sigma_0)

All functions are pure NumPy and cheap; the (L, L, L, L) tensor at L = 32 is 8 MB.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- Hermite


def he(n: int, x: np.ndarray) -> np.ndarray:
    """Probabilists' Hermite polynomial He_n(x) by the three-term recursion."""
    x = np.asarray(x, dtype=float)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    h0, h1 = np.ones_like(x), x.copy()
    for k in range(1, n):
        h0, h1 = h1, x * h1 - k * h0
    return h1


def hermite_moment_attenuation(
    a: np.ndarray, t: float, n: int, rng: np.random.Generator, n_noise: int = 1
) -> tuple[float, float]:
    """Monte Carlo check of Theorem A on a sample ``a`` of a scalar law.

    Returns ``(E[He_n(x_t)], e^{-nt} E[He_n(a)])``; the two agree up to the Monte Carlo
    error of the first. Used by the tests, not by the experiments.
    """
    alpha, delta = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
    a = np.asarray(a, dtype=float)
    z = rng.standard_normal((n_noise,) + a.shape)
    x = alpha * a[None] + np.sqrt(delta) * z
    return float(he(n, x).mean()), float(alpha**n * he(n, a).mean())


# --------------------------------------------------------------------------- chain


def chain_covariance(rho: float, q: float, L: int) -> np.ndarray:
    """Cov(a) for a_1 ~ N(0, 1), a_i = rho a_{i-1} + eps_i, Var(eps) = q.

    At q = 1 - rho^2 this is the Toeplitz matrix rho^{|i-j|}; for general q the
    variances drift from 1 toward q/(1 - rho^2) along the chain, which is what makes
    (rho, q) a two-parameter family with Sigma_0 differentiable in both.
    """
    i = np.arange(L)
    var = rho ** (2 * i) + (q * (1.0 - rho ** (2 * i)) / (1.0 - rho**2) if abs(rho) < 1 else q * i)
    lo = np.minimum.outer(i, i)
    return rho ** np.abs(np.subtract.outer(i, i)) * var[lo]


def chain_kappa4(rho: float, kappa4_eps: float, L: int) -> np.ndarray:
    """Fourth-cumulant tensor kappa_4[i, j, k, l] of the chain.

    Cumulants are additive over the independent driving-noise terms; a_1 is Gaussian and
    contributes nothing; driving noise eps_m (m >= 2, 1-indexed) enters a_i with
    coefficient rho^{i-m} for i >= m. Hence
        kappa_4[i,j,k,l] = kappa_4(eps) * sum_{m=2}^{min(i,j,k,l)} rho^{i+j+k+l-4m}.
    With 0-indexed sites s = i - 1 the sum runs over m' = 1 .. min(s).
    """
    s = np.arange(L)
    # prefix sums P[n] = sum_{m=1}^{n} rho^{-4m}, then kappa = rho^{i+j+k+l} * P[min]
    # (numerically better to accumulate the geometric sum directly)
    m2 = np.minimum.outer(s, s)
    mins = np.minimum(m2[:, :, None, None], m2[None, None, :, :])
    tot = s[:, None, None, None] + s[None, :, None, None] + s[None, None, :, None] + s[None, None, None, :]
    # geometric sum sum_{m=1}^{n} rho^{tot-4m} = rho^{tot-4} * (1 - rho^{-4n}) / (1 - rho^{-4}) for rho != 1
    if abs(rho) < 1e-12:
        out = np.zeros((L, L, L, L))
        eq = (s[:, None, None, None] == s[None, :, None, None]) & (s[:, None, None, None] == s[None, None, :, None]) & (s[:, None, None, None] == s[None, None, None, :])
        out[eq & (s[:, None, None, None] >= 1)] = kappa4_eps
        return out
    r4 = rho**-4.0
    n = mins
    with np.errstate(over="ignore", invalid="ignore"):
        gsum = np.where(n >= 1, rho ** (tot - 4.0) * (1.0 - r4**n) / (1.0 - r4), 0.0)
    return kappa4_eps * gsum


def crossover_time(rho: float, q: float | None = None, L: int = 32) -> float:
    """t_* = (1/2) log lambda_max(Sigma_0): where e^{2t} lambda_max = 1."""
    q = 1.0 - rho**2 if q is None else q
    lam = np.linalg.eigvalsh(chain_covariance(rho, q, L)).max()
    return 0.5 * float(np.log(lam))


# --------------------------------------------------------------------------- Gaussian chain: exact Fisher information


def _sigma_t(sigma0: np.ndarray, t: float) -> np.ndarray:
    alpha, delta = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
    return alpha**2 * sigma0 + delta * np.eye(sigma0.shape[0])


def gaussian_chain_fisher(rho: float, q: float, L: int, t: float, h: float = 1e-5) -> dict:
    """Exact Fisher information of one noised chain x_t about theta = (rho, q).

    x_t ~ N(0, Sigma_t) with Sigma_t = alpha^2 Sigma_0(theta) + Delta I, so
        I_ij = (1/2) tr(Sigma_t^{-1} d_i Sigma_t Sigma_t^{-1} d_j Sigma_t),
    with d_i Sigma_t = alpha^2 d_i Sigma_0 evaluated by central differences (the map is
    smooth; h = 1e-5 gives ~1e-9 relative accuracy). Returns the 2x2 matrix, its
    diagonal, the efficient information 1/(I^{-1})_pp and the rho-q correlation.
    """
    alpha = float(np.exp(-t))
    st = _sigma_t(chain_covariance(rho, q, L), t)
    d_rho = (chain_covariance(rho + h, q, L) - chain_covariance(rho - h, q, L)) / (2 * h)
    d_q = (chain_covariance(rho, q + h, L) - chain_covariance(rho, q - h, L)) / (2 * h)
    ds = [alpha**2 * d_rho, alpha**2 * d_q]
    inv = np.linalg.inv(st)
    A = [inv @ d for d in ds]
    I = np.array([[0.5 * np.trace(A[i] @ A[j]) for j in range(2)] for i in range(2)])
    Iinv = np.linalg.inv(I)
    return {
        "I": I,
        "diag": np.diag(I).copy(),
        "efficient": 1.0 / np.diag(Iinv),
        "corr_rho_q": float(I[0, 1] / np.sqrt(I[0, 0] * I[1, 1])),
        "asymptote_rho": 0.5 * float(np.sum(d_rho**2)),  # times e^{-4t}
        "asymptote_q": 0.5 * float(np.sum(d_q**2)),
    }


# --------------------------------------------------------------------------- oracle ladder


def lmmse_posterior_mean(X: np.ndarray, sigma0: np.ndarray, t: float) -> np.ndarray:
    """E^G[a | x] = alpha Sigma_0 (alpha^2 Sigma_0 + Delta I)^{-1} x, rows of X."""
    alpha = float(np.exp(-t))
    st = _sigma_t(sigma0, t)
    return (alpha * sigma0 @ np.linalg.solve(st, X.T)).T


def lmmse_score(X: np.ndarray, sigma0: np.ndarray, t: float) -> np.ndarray:
    """S^G(x) = -Sigma_t^{-1} x: the score of the covariance-matched Gaussian twin."""
    return -np.linalg.solve(_sigma_t(sigma0, t), X.T).T


def isotropic_score(X: np.ndarray) -> np.ndarray:
    """S(x) = -x: the score of N(0, I), the marginal-variance-only oracle."""
    return -X


def score_from_mean(X: np.ndarray, m: np.ndarray, t: float) -> np.ndarray:
    alpha, delta = float(np.exp(-t)), float(1.0 - np.exp(-2.0 * t))
    return -(X - alpha * m) / delta


# --------------------------------------------------------------------------- leading-order constants


def score_gap_constant(kappa4: np.ndarray) -> float:
    """C_4 = ||kappa_4||_F^2 / 6 for E||S_t - S_t^G||^2 ~ C_4 e^{-8t}.

    Derivation: log(P_t/gamma) - log(P_t^G/gamma) = e^{-4t} (1/24) sum kappa_4[ijkl]
    :x_i x_j x_k x_l: + O(e^{-6t}); its gradient's m-th component is
    e^{-4t} (1/6) sum_{jkl} kappa_4[mjkl] :x_j x_k x_l:, whose gamma-second moment is
    (1/36) * 3! * sum_{jkl} kappa_4[mjkl]^2 by Wick's theorem. Sum over m.
    Scalar case: kappa_4^2 / 6 (checked against fine quadrature, THEORY_NOTES §4).
    """
    return float(np.sum(np.asarray(kappa4) ** 2)) / 6.0


def shape_information_constant(d_kappa4: np.ndarray) -> float:
    """||d_theta kappa_4||_F^2 / 24 for I_t(theta) ~ const * e^{-8t} (theta a shape coordinate)."""
    return float(np.sum(np.asarray(d_kappa4) ** 2)) / 24.0


def correlation_information_constant(d_sigma0: np.ndarray) -> float:
    """(1/2) ||d_theta Sigma_0||_F^2 for I_t(theta) ~ const * e^{-4t} (theta a covariance coordinate)."""
    return 0.5 * float(np.sum(np.asarray(d_sigma0) ** 2))


def excess_kurtosis_to_kappa4(excess_kurtosis: float, q: float) -> float:
    """kappa_4(eps) = (excess kurtosis) * q^2."""
    return float(excess_kurtosis) * q**2
