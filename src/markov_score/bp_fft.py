"""Grid BP with the transition applied by FFT convolution, for the linear chain K(a'|a) = p_eps(a' - rho a).

The consolidated note's section 9.2 and its section 29 step 5: "implement FFT-based Gaussian convolution for the current
Markov model". For K(a'|a) = p_eps(a' - rho a) the forward transition integral is a convolution after a change of
variables, v = rho a:

    m->_{i+1}(a') = int p_eps(a' - rho a) F(a) da = int p_eps(a' - v) F(v / rho) / rho dv,    F = m->_i ell_i,

and the backward one is a correlation evaluated at rho a:

    m<-_i(a) = c(rho a),    c(v) = int p_eps(a' - v) H(a') da',    H = ell_{i+1} m<-_{i+1}.

On the uniform grid u_1..u_{N_g} both are discrete convolutions with the kernel vector p_eps(m du), m = -(N_g - 1) ..
N_g - 1, done by FFT in O(N_g log N_g) per chain and site instead of the dense O(N_g^2) matmul. The price is the
rescaling: F(v / rho) and c(rho a) fall between grid points and are interpolated (cubic Catmull--Rom on the uniform
grid, zero outside [-A, A]), which is the "interpolation detail" the note flags. This is the error the dense recursion
does not have -- `bp_grid` evaluates K exactly at grid pairs -- and E40 measures it against dense grid BP, with the
speed.

Same normalisation, evidence-free; returns posterior means and variances like `grid_bp_batch`.
Symbols: docs/NOTATION.md (cavity messages m->, m<-; alpha = e^{-t}, delta = Delta_t).
"""

from __future__ import annotations

import numpy as np


def residual_log_density(prior, d: np.ndarray) -> np.ndarray:
    """log p_eps(d) for a linear-chain prior, read off its own transition matrix: the column of
    log K(a_out | a_in) at a_in = 0 is log p_eps(a_out)."""
    g = np.concatenate([np.asarray(d, dtype=float), [0.0]])
    return prior.log_transition_matrix(g)[:-1, -1]


def _cubic_gather(n: int, du: float, half_width: float, x: np.ndarray):
    """Indices (len(x), 4) and weights (len(x), 4) so that f(x) ~ sum_k w[:, k] f[idx[:, k]] for f on the uniform grid
    -A + j du, j = 0..n-1; Catmull--Rom (Keys a = -1/2); nodes outside the grid get weight 0 (f = 0 there)."""
    s = (np.asarray(x, dtype=float) + half_width) / du
    i0 = np.floor(s).astype(int)
    tau = s - i0
    t2, t3 = tau**2, tau**3
    w = np.stack([-0.5 * t3 + t2 - 0.5 * tau,
                  1.5 * t3 - 2.5 * t2 + 1.0,
                  -1.5 * t3 + 2.0 * t2 + 0.5 * tau,
                  0.5 * t3 - 0.5 * t2], axis=1)
    idx = i0[:, None] + np.arange(-1, 3)[None, :]
    inside = (idx >= 0) & (idx < n)
    w = np.where(inside, w, 0.0)
    return np.clip(idx, 0, n - 1), w


class FFTTransition:
    """Precomputed pieces of the FFT transition for one (grid, p_eps, rho)."""

    def __init__(self, grid: np.ndarray, weights: np.ndarray, log_p_eps_lags: np.ndarray, rho: float):
        n = len(grid)
        self.n, self.rho, self.weights = n, float(rho), np.asarray(weights, dtype=float)
        self.du = float(grid[1] - grid[0])
        self.half_width = float(-grid[0])
        k = np.exp(np.asarray(log_p_eps_lags, dtype=float))         # p_eps(m du), m = -(n-1)..(n-1), length 2n-1
        if k.shape != (2 * n - 1,):
            raise ValueError("log_p_eps_lags must hold the 2 N_g - 1 lags -(N_g-1)..(N_g-1)")
        self.nfft = int(2 ** int(np.ceil(np.log2(3 * n - 2))))
        self.k_hat = np.fft.rfft(k, self.nfft)                       # convolution kernel
        self.k_rev_hat = np.fft.rfft(k[::-1], self.nfft)             # correlation kernel
        # forward: F(u_m / rho) / rho on the grid; backward: c(rho u_j)
        self.f_idx, self.f_w = _cubic_gather(n, self.du, self.half_width, np.asarray(grid) / self.rho)
        self.b_idx, self.b_w = _cubic_gather(n, self.du, self.half_width, self.rho * np.asarray(grid))

    def _conv(self, g, kernel_hat):
        """Rows of g (B, n) linearly convolved with the length-(2n-1) kernel; output on the grid, (B, n)."""
        full = np.fft.irfft(np.fft.rfft(g, self.nfft, axis=1) * kernel_hat[None, :], self.nfft, axis=1)
        return full[:, self.n - 1: 2 * self.n - 1]

    def forward(self, f):
        """f = m->_i ell_i as a density on the grid (B, n) -> unnormalised m->_{i+1} on the grid."""
        g = (f[:, self.f_idx] * self.f_w[None]).sum(axis=2) / self.rho
        return self._conv(g * self.weights[None, :], self.k_hat)

    def backward(self, h):
        """h = ell_{i+1} m<-_{i+1} on the grid (B, n) -> unnormalised m<-_i on the grid."""
        c = self._conv(h * self.weights[None, :], self.k_rev_hat)
        return (c[:, self.b_idx] * self.b_w[None]).sum(axis=2)


def fft_bp_batch(grid, weights, transition: FFTTransition, X, alpha, delta, log_mu=None):
    """Posterior means and variances, shape (B, L) each, with the transition applied by `transition`."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n_chains, n_sites = X.shape
    w = np.asarray(weights, dtype=float)
    if log_mu is None:
        log_mu = -0.5 * grid**2 - 0.5 * np.log(2.0 * np.pi)
    z = X[:, :, None] - alpha * grid[None, None, :]
    log_ell = -0.5 * z**2 / delta
    ell = np.exp(log_ell - log_ell.max(axis=2, keepdims=True))      # (B, L, N_g)
    fwd = np.empty((n_sites, n_chains, len(grid)))
    bwd = np.empty_like(fwd)
    fwd[0] = np.exp(log_mu)[None, :]
    for i in range(n_sites - 1):
        out = transition.forward(fwd[i] * ell[:, i, :])
        out = np.maximum(out, 0.0)                                   # cubic interpolation can undershoot
        mass = out @ w
        if not np.all(np.isfinite(mass)) or np.any(mass <= 0):
            raise FloatingPointError(f"FFT forward message {i + 1} lost all mass.")
        fwd[i + 1] = out / mass[:, None]
    bwd[-1] = 1.0
    for i in range(n_sites - 1, 0, -1):
        out = np.maximum(transition.backward(ell[:, i, :] * bwd[i]), 0.0)
        mass = out @ w
        if not np.all(np.isfinite(mass)) or np.any(mass <= 0):
            raise FloatingPointError(f"FFT backward message {i - 1} lost all mass.")
        bwd[i - 1] = out / mass[:, None]
    raw = fwd.transpose(1, 0, 2) * ell * bwd.transpose(1, 0, 2)
    mass = raw @ w
    b = raw / mass[:, :, None]
    means = (b * (grid * w)[None, None, :]).sum(axis=2)
    variances = (b * (grid**2 * w)[None, None, :]).sum(axis=2) - means**2
    return means, variances
