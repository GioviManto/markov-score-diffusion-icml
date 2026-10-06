"""The one-dimensional phi^4 field as a Markov chain: the second laboratory.

Why this module exists
----------------------
Every chain prior in `priors.py` has the form a' = rho a + eps with a driving-noise law that is
free but state-independent. Jerome Garnier-Brun and Marc Mezard asked (1 October 2026) for a
test of the estimator on Markov data that come from a physical model rather than from that
construction. The stochastic Allen-Cahn / phi^4 equation in one space dimension,

    d_t phi = d_x^2 phi - V'(phi) + sqrt(2T) eta,        V(phi) = (phi^2 - 1)^2 / 4,

is the gradient flow of the energy E[phi] = int [ (d_x phi)^2 / 2 + V(phi) ] dx at temperature
T, so its stationary law is the Gibbs measure exp(-E/T). Discretised with spacing h,

    E_h(phi) = sum_i [ (phi_{i+1} - phi_i)^2 / (2h) + h V(phi_i) ],

which couples nearest neighbours only. A spatial snapshot is therefore EXACTLY a Markov chain
in the site index, and -- this is the point -- its transition kernel is not of the form
"rho a + driving noise": it is a Gaussian random-walk step tilted by a function of the arrival
point. The mixture kernel the paper fits is misspecified here (through the DRIFT,
which is nonlinear, not through the driving-noise shape), and the estimator that is exactly
specified is `tilted_kernel.TiltedWalkKernel`.

The transfer operator and the chain
-----------------------------------
Split the site terms symmetrically and write

    T(phi, phi') = exp(-h V(phi)/2T) exp(-(phi' - phi)^2 / 2Th) exp(-h V(phi')/2T).

T is symmetric, positive and (for a confining V) Hilbert-Schmidt on L^2(R), so it has a simple
largest eigenvalue lambda_0 > 0 with a positive eigenfunction psi_0 (Perron-Frobenius). The
infinite-volume Gibbs measure is the stationary Markov chain with

    pi(phi) = psi_0(phi)^2,       K(phi' | phi) = T(phi, phi') psi_0(phi') / (lambda_0 psi_0(phi)),

reversible with respect to pi (pi(phi) K(phi'|phi) is symmetric). The lag-k covariance is
sum_n (lambda_n / lambda_0)^k <psi_0| phi |psi_n>^2, so the correlation length in sites is
1 / log(lambda_0 / lambda_1), and lambda_1 / lambda_0 plays for this family the role rho plays
for the AR(1) family.

The tilted-walk form
--------------------
Rearranging the eigen-equation gives the identity this module and `tilted_kernel` rest on:

    K(phi' | phi) = N(phi'; phi, Th) g(phi') / Z(phi),       g = psi_0 exp(-hV/2T),
    Z(phi) = int N(phi'; phi, Th) g(phi') dphi' = lambda_0 psi_0(phi) exp(+hV(phi)/2T).

So the kernel is a Gaussian step of variance Th times a positive function of the ARRIVAL point,
divided by the column normaliser -- and the column normaliser IS the h-transform. A Gaussian
AR(1) chain with 0 < rho < 1 is the member with log g quadratic (step variance q/rho, tilt
variance 1 + rho), so the two laboratories intersect at the Gaussian chain and nowhere else.

Standardisation
---------------
Everything downstream (the channel x = e^{-t} a + sqrt(1 - e^{-2t}) z, the noise schedule, the
grid) assumes unit marginal variance. The chain exposed to the rest of the package is the
standardised field a = phi / s with s^2 = Var_pi(phi); the mean is zero by the phi -> -phi
symmetry, which the eigensolve respects to round-off and the code enforces.

Numerics, and what is exact
---------------------------
psi_0 is computed by a dense symmetric eigensolve of the trapezoid-discretised operator on a
FINE physical grid (default 2401 points on [-12, 12]). Trapezoid quadrature of a smooth,
super-exponentially decaying integrand converges faster than any power of the spacing, so the
eigenpair is accurate to near machine precision; `tests/test_field.py` halves the spacing and
checks. The far tail where psi_0 is below the eigensolver's resolution is carried as
log psi_0 = -inf: those states hold stationary mass below 1e-26 and are never reached. A kernel
COLUMN leaving such a state is replaced by the normalised T column so the matrix stays finite,
exactly as the uniform family's -inf entries are handled by every caller.

Two independent samplers are provided and must agree (the discipline of `stationary.py`):

  sample / sample_batch   exact in the continuum: propose phi' ~ N(phi, Th) and accept with
                          probability g(phi') / max g, with log psi_0 interpolated by a cubic
                          spline on the fine grid. Burn-in from the kernel's own mixing time
                          makes every site stationary, as `stationary.sample_stationary` does.
  sample_lattice          exact for the finite-state chain the fine grid defines: inverse-CDF
                          draws from the columns of the discretised kernel -- literally
                          transfer-matrix sampling, `discrete.py`'s philosophy.

The exact score of the field chain is grid BP (`bp_grid.grid_bp_batch`) with this module's
`log_transition_matrix` and `log_stationary` as the initial law, to quadrature accuracy on the
BP grid; `experiments/e32_field_oracles.py` measures that accuracy by refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh
from scipy.special import logsumexp

_LOG_2PI = float(np.log(2.0 * np.pi))
_RESOLVED_REL = 1e-13
"""Eigenvector components below this fraction of the maximum are eigensolver noise, not
psi_0: `eigh` is accurate to ~1e-16 of the operator norm, and the squared threshold
(stationary mass 1e-26) is what makes the discarded states unreachable."""


def potential(phi):
    """The symmetric double well V(phi) = (phi^2 - 1)^2 / 4: minima at +-1, barrier 1/4."""
    phi = np.asarray(phi, dtype=float)
    return 0.25 * (phi**2 - 1.0) ** 2


def _trapezoid_weights(grid: np.ndarray) -> np.ndarray:
    dx = float(grid[1] - grid[0])
    w = np.full(len(grid), dx)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def log_transfer(T: float, h: float, phi_in, phi_out) -> np.ndarray:
    """log T(phi_in, phi_out) with the site terms split symmetrically; broadcasts."""
    phi_in = np.asarray(phi_in, dtype=float)
    phi_out = np.asarray(phi_out, dtype=float)
    return (-(h / (2.0 * T)) * (potential(phi_in) + potential(phi_out))
            - (phi_out - phi_in) ** 2 / (2.0 * T * h))


@dataclass(frozen=True)
class FieldSpectrum:
    """Eigen-data of the discretised transfer operator at one (T, h), in PHYSICAL units.

    u, w        fine grid and its trapezoid weights
    lam         every eigenvalue, descending (lam[0] = lambda_0)
    psi         (n_fine, n_keep) leading eigenfunctions as columns, int psi_n^2 du = 1
    log_psi0    log psi_0 on the fine grid, -inf where unresolved
    resolved    the contiguous index range [lo, hi) where psi_0 is resolved
    overlap     c_n = <psi_0| phi |psi_n> for EVERY n (sum_n c_n^2 = Var_pi(phi) exactly,
                by completeness of the discretised eigenbasis)
    scale       s = sqrt(Var_pi(phi)), the standardisation
    log_g_max   log max_u g(u), g = psi_0 exp(-hV/2T): the rejection sampler's envelope
    """

    T: float
    h: float
    u: np.ndarray
    w: np.ndarray
    lam: np.ndarray
    psi: np.ndarray
    log_psi0: np.ndarray
    resolved: tuple
    overlap: np.ndarray
    scale: float
    excess_kurtosis: float
    log_g_max: float

    @property
    def lambda0(self) -> float:
        return float(self.lam[0])

    @property
    def rho1(self) -> float:
        """lambda_1 / lambda_0: the lag-1 decay of the slowest mode, the AR(1) rho's analogue."""
        return float(self.lam[1] / self.lam[0])

    @property
    def correlation_length_sites(self) -> float:
        return float(-1.0 / np.log(self.rho1))

    @property
    def stationary(self) -> np.ndarray:
        """pi = psi_0^2 on the fine grid (physical units); zero where unresolved."""
        out = np.zeros_like(self.u)
        lo, hi = self.resolved
        out[lo:hi] = np.exp(2.0 * self.log_psi0[lo:hi])
        return out

    def spline(self) -> CubicSpline:
        lo, hi = self.resolved
        return CubicSpline(self.u[lo:hi], self.log_psi0[lo:hi], extrapolate=False)


@lru_cache(maxsize=64)
def solve_spectrum(T: float, h: float, n_fine: int = 2401, half_width: float = 12.0,
                   n_keep: int = 256) -> FieldSpectrum:
    """Dense symmetric eigensolve of the trapezoid-discretised transfer operator.

    With W = diag(w) the discretised operator is T W; the similarity transform
    A = W^{1/2} T W^{1/2} is symmetric with the same spectrum, and psi_n = W^{-1/2} v_n is
    normalised in the quadrature L^2 norm exactly when v_n is a unit vector. Cached: a sweep
    over (T, h) solves each cell once per process.
    """
    if T <= 0.0 or h <= 0.0:
        raise ValueError(f"T and h must be positive, got T={T!r} h={h!r}.")
    u = np.linspace(-half_width, half_width, n_fine)
    w = _trapezoid_weights(u)
    sw = np.sqrt(w)
    A = np.exp(log_transfer(T, h, u[None, :], u[:, None])) * sw[:, None] * sw[None, :]
    lam, v = eigh(A)
    order = np.argsort(lam)[::-1]
    lam, v = lam[order], v[:, order]
    psi = v / sw[:, None]
    # Perron-Frobenius: the top eigenvector has one sign; fix it positive at its largest entry.
    if psi[np.argmax(np.abs(psi[:, 0])), 0] < 0:
        psi[:, 0] *= -1.0
    psi0 = psi[:, 0]
    mask = psi0 > _RESOLVED_REL * psi0.max()
    idx = np.flatnonzero(mask)
    lo, hi = int(idx[0]), int(idx[-1]) + 1
    if not mask[lo:hi].all():
        raise FloatingPointError("psi_0 is not resolved on a contiguous range; the operator is "
                                 "numerically disconnected (T too small for this grid?).")
    log_psi0 = np.full(n_fine, -np.inf)
    log_psi0[lo:hi] = np.log(psi0[lo:hi])
    pi = np.zeros(n_fine)
    pi[lo:hi] = psi0[lo:hi] ** 2
    mass = float(pi @ w)
    mean = float((u * pi) @ w) / mass
    var = float(((u - mean) ** 2 * pi) @ w) / mass
    m4 = float(((u - mean) ** 4 * pi) @ w) / mass
    if abs(mean) > 1e-8:
        raise FloatingPointError(f"stationary mean {mean:.3e} is not zero; symmetry is broken.")
    # <psi_0| phi |psi_n> for every n: completeness makes sum c_n^2 = Var exactly.
    overlap = (psi0 * u * w) @ psi
    log_g = log_psi0 - (h / (2.0 * T)) * potential(u)
    return FieldSpectrum(T=float(T), h=float(h), u=u, w=w, lam=lam, psi=psi[:, :n_keep],
                         log_psi0=log_psi0, resolved=(lo, hi), overlap=overlap,
                         scale=float(np.sqrt(var)), excess_kurtosis=float(m4 / var**2 - 3.0),
                         log_g_max=float(log_g[lo:hi].max()))


@dataclass(frozen=True)
class Phi4Chain:
    """The standardised phi^4 chain at temperature T and lattice spacing h.

    Implements the `priors.ChainPrior` surface the package expects of a chain prior --
    `name`, `sample`, `log_transition_matrix(grid)` -- plus `log_stationary(grid)` (the initial
    law BP must be given: the data are stationary, NOT started from N(0, 1)) and
    `covariance(L)` for the linear floor. `rho` is exposed as lambda_1 / lambda_0 so the
    burn-in and crossover helpers that read a correlation coefficient keep working.
    """

    T: float
    h: float
    n_fine: int = 2401
    half_width_fine: float = 12.0

    @property
    def name(self) -> str:
        return f"phi4_T{self.T:g}_h{self.h:g}"

    @property
    def spectrum(self) -> FieldSpectrum:
        return solve_spectrum(float(self.T), float(self.h), self.n_fine, self.half_width_fine)

    @property
    def rho(self) -> float:
        return self.spectrum.rho1

    @property
    def scale(self) -> float:
        return self.spectrum.scale

    @property
    def innovation_excess_kurtosis(self) -> float:
        """Excess kurtosis of the MARGINAL, the family's non-Gaussianity knob: the per-step
        driving noise is not state-independent here, so the AR(1) notion does not apply."""
        return self.spectrum.excess_kurtosis

    # ------------------------------------------------------------------ densities

    def _log_psi0_at(self, phi: np.ndarray) -> np.ndarray:
        sp = self.spectrum
        out = sp.spline()(phi)
        return np.where(np.isfinite(out), out, -np.inf)

    def log_tilt(self, grid: np.ndarray) -> np.ndarray:
        """log g(s a) on a standardised grid: the arrival-point tilt of the exact kernel."""
        phi = self.scale * np.asarray(grid, dtype=float)
        return self._log_psi0_at(phi) - (self.h / (2.0 * self.T)) * potential(phi)

    def log_transition_matrix(self, grid: np.ndarray, grid_out: np.ndarray | None = None) -> np.ndarray:
        """log K[out, in] of the STANDARDISED chain on the grid, K_a(a'|a) = s K_phi(s a'|s a)."""
        sp = self.spectrum
        s = sp.scale
        a_in = np.asarray(grid, dtype=float)
        a_out = a_in if grid_out is None else np.asarray(grid_out, dtype=float)
        lt = log_transfer(self.T, self.h, s * a_in[None, :], s * a_out[:, None])
        lp_in = self._log_psi0_at(s * a_in)
        lp_out = self._log_psi0_at(s * a_out)
        # -inf rows (an unreachable arrival state) are legitimate: K = 0 there, as the uniform
        # family's -inf entries are. The subtraction below produces NaN only in the columns
        # that are overwritten next, so the warning is silenced rather than the value hidden.
        with np.errstate(invalid="ignore"):
            log_k = np.log(s) + lt + lp_out[:, None] - np.log(sp.lambda0) - lp_in[None, :]
        bad = ~np.isfinite(lp_in)
        if bad.any():
            # A column leaving an unreachable state: the normalised T column, so the matrix
            # is finite everywhere and still a proper distribution in every column.
            lw = np.log(_trapezoid_weights(a_out))
            col = np.log(s) + lt[:, bad]
            log_k[:, bad] = col - logsumexp(col + lw[:, None], axis=0)[None, :]
        return log_k

    def log_stationary(self, grid: np.ndarray) -> np.ndarray:
        """log pi_a(a) = 2 log psi_0(s a) + log s on a standardised grid; -inf where unresolved."""
        s = self.scale
        return 2.0 * self._log_psi0_at(s * np.asarray(grid, dtype=float)) + np.log(s)

    def covariance(self, L: int) -> np.ndarray:
        """Cov(a_i, a_j) of the standardised stationary chain, i, j < L, by the spectral sum."""
        sp = self.spectrum
        r = np.clip(sp.lam[1:] / sp.lambda0, 0.0, None)
        c2 = sp.overlap[1:] ** 2 / sp.scale**2
        k = np.arange(L)
        lag = np.abs(k[:, None] - k[None, :])
        cov = np.zeros((L, L))
        for d in range(L):
            cov[lag == d] = float((r**d * c2).sum())
        return cov

    # ------------------------------------------------------------------ samplers

    def _step(self, rng: np.random.Generator, phi: np.ndarray) -> np.ndarray:
        """One exact transition of every chain in `phi` (physical units), by rejection."""
        sp = self.spectrum
        sd = float(np.sqrt(self.T * self.h))
        spline = sp.spline()
        hv = self.h / (2.0 * self.T)
        out = np.empty_like(phi)
        todo = np.ones(len(phi), dtype=bool)
        while todo.any():
            idx = np.flatnonzero(todo)
            prop = phi[idx] + sd * rng.standard_normal(len(idx))
            lg = spline(prop) - hv * potential(prop)
            lg = np.where(np.isfinite(lg), lg, -np.inf)
            # log U < log g - log g_max; a -inf tilt is rejected without a warning
            with np.errstate(divide="ignore"):
                accept = np.log(rng.random(len(idx))) < lg - sp.log_g_max
            out[idx[accept]] = prop[accept]
            todo[idx[accept]] = False
        return out

    def burn_in(self, target: float = 1e-12) -> int:
        """Steps after which the start is forgotten to `target`: the slowest mode decays as
        rho_1^k, exactly as rho^k does for the AR(1) family (`stationary.stationary_burn_in`)."""
        return int(np.ceil(np.log(target) / np.log(self.rho)))

    def sample_batch(self, rng: np.random.Generator, n_chains: int, n_sites: int,
                     burn_in: int | None = None) -> np.ndarray:
        """(n_chains, n_sites) strictly stationary standardised chains, exact in the continuum."""
        if burn_in is None:
            burn_in = self.burn_in()
        s = self.scale
        phi = s * rng.standard_normal(n_chains)
        for _ in range(burn_in):
            phi = self._step(rng, phi)
        out = np.empty((n_chains, n_sites))
        out[:, 0] = phi
        for i in range(1, n_sites):
            phi = self._step(rng, phi)
            out[:, i] = phi
        return out / s

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """One stationary chain of `n` sites (the `ChainPrior` surface)."""
        return self.sample_batch(rng, 1, n)[0]

    def sample_lattice(self, rng: np.random.Generator, n_chains: int, n_sites: int) -> np.ndarray:
        """The SECOND route: exact draws from the finite-state chain the fine grid defines
        (inverse-CDF on the columns of the discretised kernel), standardised, lattice-valued.
        Agreement with `sample_batch` in moments and autocorrelation is the check that the
        continuum sampler and the eigensolve describe one chain."""
        sp = self.spectrum
        lo, hi = sp.resolved
        u, w = sp.u[lo:hi], sp.w[lo:hi]
        psi0 = np.exp(sp.log_psi0[lo:hi])
        lt = log_transfer(self.T, self.h, u[None, :], u[:, None])
        K = np.exp(lt) * psi0[:, None] / (sp.lambda0 * psi0[None, :])
        P = K * w[:, None]
        P = P / P.sum(axis=0, keepdims=True)
        cdf = np.cumsum(P, axis=0)
        pi = psi0**2 * w
        cpi = np.cumsum(pi / pi.sum())
        idx = np.searchsorted(cpi, rng.random(n_chains))
        idx = np.minimum(idx, len(u) - 1)
        out = np.empty((n_chains, n_sites))
        out[:, 0] = u[idx]
        for i in range(1, n_sites):
            r = rng.random(n_chains)
            nxt = np.empty_like(idx)
            for b in range(n_chains):
                nxt[b] = min(int(np.searchsorted(cdf[:, idx[b]], r[b])), len(u) - 1)
            idx = nxt
            out[:, i] = u[idx]
        return out / sp.scale

    # ------------------------------------------------------------------ cumulants

    def fourth_cumulant_tensor(self, L: int, n_modes: int | None = None) -> np.ndarray:
        """kappa_4 of the standardised chain on L sites, by transfer-matrix four-point functions.

        E[phi_i phi_j phi_k phi_l] for i <= j <= k <= l is
        <psi_0| phi R^{j-i} phi R^{k-j} phi R^{l-k} phi |psi_0> with R = T / lambda_0, which in
        the eigenbasis is a triple sum over modes with the matrix Phi_nm = <psi_n| phi |psi_m>.
        The sum is truncated to the kept modes, whose ratios r_n decay fast; the truncation is
        reported by `tests/test_field.py` against the exact second moments. Zero mean, so
        kappa_4 = E[....] - (Sigma Sigma + two permutations). This is Theorem 2's constant for
        the second laboratory: (1/6) ||kappa_4||_F^2 is the predicted closure-gap amplitude.
        """
        sp = self.spectrum
        n = sp.psi.shape[1] if n_modes is None else min(n_modes, sp.psi.shape[1])
        psi = sp.psi[:, :n]
        r = np.clip(sp.lam[:n] / sp.lambda0, 0.0, None)
        Phi = (psi * (sp.u * sp.w)[:, None]).T @ psi / sp.scale  # <psi_n| a |psi_m>
        sig = self.covariance(L)
        # F[d1, d2, d3] = <psi_0| a R^{d1} a R^{d2} a R^{d3} a |psi_0> for every gap triple.
        pw = r[None, :] ** np.arange(L)[:, None]                 # (L, n): r_n^d
        left = Phi[0][None, :] * pw                              # (L, n): <psi_0| a R^{d1}
        mid = (left @ Phi)[:, None, :] * pw[None, :, :]          # (L, L, n): ... a R^{d2}
        right = (mid @ Phi)[:, :, None, :] * pw[None, None, :, :]  # (L, L, L, n): ... a R^{d3}
        F = right @ Phi[:, 0]                                    # (L, L, L)
        idx = np.sort(np.stack(np.meshgrid(*[np.arange(L)] * 4, indexing="ij"), axis=-1), axis=-1)
        i, j, k, l = (idx[..., q] for q in range(4))
        kap = F[j - i, k - j, l - k]
        a, b, c, d = np.meshgrid(*[np.arange(L)] * 4, indexing="ij")
        kap = kap - (sig[a, b] * sig[c, d] + sig[a, c] * sig[b, d] + sig[a, d] * sig[b, c])
        return kap


def h_for_correlation(T: float, rho_target: float, h_lo: float = 0.02, h_hi: float = 8.0,
                      n_fine: int = 1201, tol: float = 1e-4) -> float:
    """The spacing h at which lambda_1 / lambda_0 = rho_target at temperature T.

    rho_1 decreases monotonically in h (a larger spacing is a weaker coupling), so a bisection
    in log h finds the iso-correlation spacing. Solved on a coarser fine grid for speed; the
    returned h is then used with the default resolution, and the achieved rho_1 is recorded by
    the experiment rather than assumed.
    """
    f = lambda hh: solve_spectrum(float(T), float(hh), n_fine).rho1 - rho_target
    lo, hi = np.log(h_lo), np.log(h_hi)
    flo, fhi = f(np.exp(lo)), f(np.exp(hi))
    if flo < 0 or fhi > 0:
        raise ValueError(f"rho_1 = {rho_target} is not bracketed on h in [{h_lo}, {h_hi}] at T={T}: "
                         f"rho_1({h_lo}) = {flo + rho_target:.4f}, rho_1({h_hi}) = {fhi + rho_target:.4f}")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if f(np.exp(mid)) > 0:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


@dataclass(frozen=True)
class StridedChain:
    """The fine chain observed every `stride`-th site: the subsampling protocol of the consolidated
    note (section 21.1, "generate at fine resolution and subsample systematically").

    Exactly Markov with the `stride`-step kernel K^s of the fine chain (Chapman-Kolmogorov on the
    grid, K_s = K (W K)^{s-1} with W the quadrature weights), the same stationary law and marginal
    scale, effective spacing s h and lambda_1/lambda_0 to the power s. One physical object at three
    resolutions, as opposed to the registered refinement cells, which are three lattice models.
    For s >= 2 the kernel is NOT a tilted walk: marginalising the intermediate sites destroys the
    Gaussian-bond form, so the gradient-class kernels (tilted, pair energy) are misspecified on it
    while a state-dependent mixture is not (E36)."""

    fine: Phi4Chain
    stride: int

    def __post_init__(self):
        if int(self.stride) < 1:
            raise ValueError("stride must be a positive integer")

    @property
    def name(self) -> str:
        return f"{self.fine.name}_stride{self.stride}"

    @property
    def T(self) -> float:
        return self.fine.T

    @property
    def h(self) -> float:
        """The effective spacing of the observed sites."""
        return self.fine.h * self.stride

    @property
    def spectrum(self) -> FieldSpectrum:
        return self.fine.spectrum

    @property
    def rho(self) -> float:
        return self.fine.rho ** self.stride

    @property
    def scale(self) -> float:
        return self.fine.scale

    @property
    def innovation_excess_kurtosis(self) -> float:
        return self.fine.innovation_excess_kurtosis

    def log_transition_matrix(self, grid: np.ndarray, grid_out: np.ndarray | None = None) -> np.ndarray:
        if grid_out is not None:
            raise NotImplementedError("the strided kernel lives on one grid")
        lk = self.fine.log_transition_matrix(grid)
        if self.stride == 1:
            return lk
        w = _trapezoid_weights(np.asarray(grid, dtype=float))
        K = np.exp(lk)
        step = w[:, None] * K                      # (W K)[m, j] = w_m K[m, j]
        M = K
        for _ in range(self.stride - 1):
            M = M @ step                           # K_{s+1}[k, j] = sum_m K_s[k, m] w_m K[m, j]
        with np.errstate(divide="ignore"):
            return np.log(M)

    def log_stationary(self, grid: np.ndarray) -> np.ndarray:
        return self.fine.log_stationary(grid)

    def covariance(self, L: int) -> np.ndarray:
        return self.fine.covariance((L - 1) * self.stride + 1)[:: self.stride, :: self.stride]

    def sample_batch(self, rng: np.random.Generator, n_chains: int, n_sites: int) -> np.ndarray:
        return self.fine.sample_batch(rng, n_chains, (n_sites - 1) * self.stride + 1)[:, :: self.stride]

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self.sample_batch(rng, 1, n)[0]
