"""E61's calibration, from the TRUE kernels only: no training data, no noisy view, no fit to data.

    python tools/core/calibrate_e61.py [--families]

E61 runs the same arms on a panel of chains whose kernels are known in closed form, every chain matched to E03's Laplace
chain in variance (1) and lag-one correlation (0.85), so that a linear method sees the same covariance on every chain and
every difference comes from shape:

    gauss      a' = 0.85 a + sqrt(1 - 0.85^2) eps                     the control: linear is optimal
    laplace    E03's chain (LaplaceAR1)                                the anchor
    student    a' = 0.85 a + s T_nu (StudentTAR1), nu from this run    heavy tails, inside the shift class
    uniform    a' = 0.85 a + Uniform(-h, h) (UniformAR1)               bounded innovations (hard edges in the kernel)
    skew       a = (T(z) - m) / s, z a Gaussian AR(1) of correlation r, T(z) = sinh(asinh(z) + eps)
               (a sinh-arcsinh map: full support, Gaussian tails of different scales on the two sides); eps for a
               marginal skewness of 1, r for a lag-one correlation of 0.85. Outside every shift class: the conditional
               law changes its shape with a. Kernel and stationary law in closed form.

Two textbook diffusions, sampled every Delta (asked on 2 Nov 2026: "real" first-order Markov processes). Both are
reversible with a closed-form stationary law pi and diffusion coefficient D; the generator is the square-root
approximation on the 1201-point grid (rate D h^-2 sqrt(pi_j / pi_i) between neighbours: detailed balance with pi exactly),
and the transition over Delta is its matrix exponential, so the kernel is exact for the process that is sampled (a jump
to a cell of width 0.02, then a uniform position inside it). Full support, so they keep the package's start a_1 ~ N(0, 1)
(CIR's support is bounded below and cannot):

    langevin   dX = -V'(X) dt + sqrt(2) dW, V = h_b (x^2 / s^2 - 1)^2 with barrier h_b = 2: a bimodal, metastable
               double well; s for variance 1, Delta for a lag-one correlation of 0.85
    pearson    the Student (Pearson) diffusion dX = -theta X dt + sqrt(2 theta (nu + X^2) / (nu - 1)) dW, a Student-t
               stationary law, standardised; linear drift, state-dependent volatility; Delta for 0.85, nu the smallest
               of 5, 6, 8 with mass beyond |a| > 8 below 1e-4

Part 1 (always): each chain's variance, lag-one correlation, skewness, excess kurtosis and mass beyond |a| > 8 (E03's
grid edge) from the leading eigenvector of the grid operator on |a| <= 12 (1201 points); nu is the smallest of 3, 4, 5
with mass beyond 8 below 1e-4.
Part 2 (--families): on E03's grid (401 points on [-8, 8]) the best member of the shift mixture (C = 8) and of the free
2-D mixture (C = 4, E60's family) fitted to the exact pair law; KL per edge from the true kernel and the BP score error
(E03's estimand, raw: on the Gaussian control the linear floor IS the truth, so no ratio to it) on a calibration-only test
bundle (tag e61-calibration, 1024 chains, not a registered seed), next to the linear floor's.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.sparse.linalg import eigs
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "tools" / "core"))

G = np.linspace(-12.0, 12.0, 1201)
W = np.full(G.size, G[1] - G[0])
W[[0, -1]] *= 0.5
RHO, SKEW = 0.85, 1.0
_Z = np.linspace(-12.0, 12.0, 24001)                         # quadrature for the latent N(0, 1)
_PZ = norm.pdf(_Z) * (_Z[1] - _Z[0])


def stats(log_k: np.ndarray, g=G, w=W):
    """Mean, variance, lag-one correlation, skewness, excess kurtosis, mass beyond |a| > 8 of the stationary law."""
    k = np.exp(log_k - log_k.max(0, keepdims=True))
    k /= (k * w[:, None]).sum(0, keepdims=True)
    p = k * w[:, None]
    _, vec = eigs(p, k=1, which="LM")
    pi = np.abs(np.real(vec[:, 0]))
    pi /= pi.sum()
    m = pi @ g
    v = pi @ (g - m) ** 2
    cond_mean = (p * g[:, None]).sum(0)
    c1 = (pi * (g - m) * (cond_mean - m)).sum() / v
    return m, v, c1, pi @ (g - m) ** 3 / v ** 1.5, pi @ (g - m) ** 4 / v ** 2 - 3.0, float(pi[np.abs(g) > 8].sum())


# ------------------------------------------------------------------ the skewed chain


def _t(z, eps):
    return np.sinh(np.arcsinh(z) + eps)


def _t_inv(x, eps):
    return np.sinh(np.arcsinh(x) - eps)


def skew_moments(eps):
    """Mean, sd and skewness of T(z), z ~ N(0, 1)."""
    x = _t(_Z, eps)
    m = _PZ @ x
    s = np.sqrt(_PZ @ (x - m) ** 2)
    return m, s, (_PZ @ (x - m) ** 3) / s ** 3


def skew_log_k(eps, r, g):
    """log K(a' | a) of the standardised chain a = (T(z) - m) / s, z' = r z + sqrt(1 - r^2) e; rows a', columns a."""
    m, s, _ = skew_moments(eps)
    x = m + s * g
    z = _t_inv(x, eps)
    dz = s * np.cosh(np.arcsinh(x) - eps) / np.sqrt(1 + x ** 2)       # dz / da
    q = 1 - r ** 2
    return -0.5 * (z[:, None] - r * z[None, :]) ** 2 / q - 0.5 * np.log(2 * np.pi * q) + np.log(dz)[:, None]


def skew_lag1(eps, r):
    """The lag-one correlation of a for a latent correlation r (Gauss-Hermite on the bivariate latent law)."""
    m, s, _ = skew_moments(eps)
    zz, ww = np.polynomial.hermite_e.hermegauss(120)
    ww = ww / ww.sum()
    a1 = (_t(zz, eps) - m) / s
    a2 = (_t(r * zz[:, None] + np.sqrt(1 - r ** 2) * zz[None, :], eps) - m) / s
    return float(ww @ (a1[:, None] * a2 @ ww))


def calibrate_skew():
    eps = brentq(lambda e: skew_moments(e)[2] - SKEW, 0.01, 2.0, xtol=1e-12)
    r = brentq(lambda r: skew_lag1(eps, r) - RHO, 0.5, 0.99, xtol=1e-12)
    return eps, r


# ------------------------------------------------------------------ the two diffusions


class DiffusionChain:
    """A reversible 1-D diffusion with stationary law pi and diffusion coefficient d on the 1201-point grid G, sampled every
    delta: generator by the square-root approximation, transition by its matrix exponential (column-stochastic P[i, j] =
    Prob(cell j -> cell i)); a sample jumps between cells and sits uniformly inside its cell; a_1 ~ N(0, 1)."""

    def __init__(self, name, log_pi, d_mid, delta):
        self.name_, self.delta = name, delta
        h = G[1] - G[0]
        log_pi = np.maximum(log_pi, log_pi.max() - 60.0)     # mass below e^-60 of the peak: rates stay finite
        up = d_mid / h ** 2 * np.exp(0.5 * (log_pi[1:] - log_pi[:-1]))       # rate i -> i + 1
        down = d_mid / h ** 2 * np.exp(0.5 * (log_pi[:-1] - log_pi[1:]))     # rate i + 1 -> i
        q = np.diag(up, -1) + np.diag(down, 1)                                # q[to, from]
        q -= np.diag(q.sum(0))
        # symmetrise with pi^(1/2): one eigendecomposition gives P(delta) for every delta
        s = np.exp(0.5 * (log_pi - log_pi.max()))
        lam, u = np.linalg.eigh((q * s[None, :]) / s[:, None])
        self._s, self._lam, self._u = s, lam, u
        self.p = self.transition(delta)

    def transition(self, delta):
        p = (self._u * np.exp(self._lam * delta)[None, :]) @ self._u.T
        p = p * self._s[:, None] / self._s[None, :]
        p = np.clip(p, 0.0, None)
        return p / p.sum(0, keepdims=True)

    @property
    def name(self):
        return self.name_

    def log_k_fine(self):
        return np.log(np.maximum(self.p, 1e-300)) - np.log(G[1] - G[0])

    def log_transition_matrix(self, grid):
        idx = np.rint((np.asarray(grid) - G[0]) / (G[1] - G[0])).astype(int)
        if not np.allclose(G[idx], grid, atol=1e-9):
            raise ValueError("the grid must be a subset of the diffusion's 0.02 lattice on [-12, 12]")
        return self.log_k_fine()[np.ix_(idx, idx)]

    def sample(self, rng, n):
        h = G[1] - G[0]
        i = int(np.clip(np.rint((rng.standard_normal() - G[0]) / h), 0, G.size - 1))
        out = np.empty(n)
        cdf = np.cumsum(self.p, axis=0)
        for k in range(n):
            if k:
                i = min(int(np.searchsorted(cdf[:, i], rng.random())), G.size - 1)
            out[k] = G[i] + h * (rng.random() - 0.5)
        return out


def _lag1(chain, delta):
    p = chain.transition(delta)
    pi = np.exp(2 * np.log(chain._s))
    pi /= pi.sum()
    m = pi @ G
    v = pi @ (G - m) ** 2
    return float((pi * (G - m) * ((p * G[:, None]).sum(0) - m)).sum() / v)


def _matched(name, log_pi_fn, d_fn, delta0=0.1):
    """Scale for variance 1 (log_pi_fn(scale) at G), then delta for a lag-one correlation of 0.85."""
    def var(sc):
        lp = log_pi_fn(sc)
        p = np.exp(lp - lp.max())
        p /= p.sum()
        return p @ G ** 2 - (p @ G) ** 2
    sc = brentq(lambda sc: var(sc) - 1.0, 0.2, 5.0, xtol=1e-12)
    g_mid = 0.5 * (G[1:] + G[:-1])
    chain = DiffusionChain(name, log_pi_fn(sc), d_fn(sc, g_mid), delta0)
    delta = brentq(lambda dl: _lag1(chain, dl) - RHO, 1e-3, 10.0, xtol=1e-12)
    return DiffusionChain(name, log_pi_fn(sc), d_fn(sc, g_mid), delta), sc


def langevin_chain(barrier=2.0):
    """Double well V = barrier (x^2 / s^2 - 1)^2, unit temperature, D = 1."""
    return _matched("langevin", lambda s: -barrier * (G ** 2 / s ** 2 - 1) ** 2, lambda s, g: np.ones_like(g))


def pearson_chain(nu):
    """The Student diffusion in units a = X / c: pi(a) ~ (1 + c^2 a^2 / nu)^(-(nu + 1) / 2), D(a) = (nu + c^2 a^2) / ((nu - 1) c^2)
    with theta = 1 (time is in units of 1 / theta); c is the scale that makes Var a = 1 on the grid."""
    return _matched(f"pearson_nu{nu:g}", lambda c: -0.5 * (nu + 1) * np.log1p(c ** 2 * G ** 2 / nu),
                    lambda c, g: (nu + c ** 2 * g ** 2) / ((nu - 1) * c ** 2))


# ------------------------------------------------------------------ part 1


def chain_table():
    from markov_score.priors import GaussianAR1, LaplaceAR1, StudentTAR1, UniformAR1
    rows = {"gauss": GaussianAR1(RHO).log_transition_matrix(G), "laplace": LaplaceAR1(RHO).log_transition_matrix(G),
            "uniform": UniformAR1(RHO).log_transition_matrix(G)}
    for nu in (3.0, 4.0, 5.0):
        rows[f"student_nu{nu:g}"] = StudentTAR1(RHO, nu).log_transition_matrix(G)
    eps, r = calibrate_skew()
    rows["skew"] = skew_log_k(eps, r, G)
    print(f"skew chain: eps = {eps:.6f}, latent r = {r:.6f} (marginal skewness {skew_moments(eps)[2]:.4f}, "
          f"lag-one correlation {skew_lag1(eps, r):.6f} by quadrature)")
    print(f"{'chain':14s} {'mean':>8s} {'var':>8s} {'corr1':>8s} {'skew':>7s} {'exkurt':>7s} {'P|a|>8':>8s}")
    tails = {}
    for name, lk in rows.items():
        m, v, c1, sk, ku, tail = stats(lk)
        tails[name] = tail
        print(f"{name:14s} {m:8.4f} {v:8.4f} {c1:8.4f} {sk:7.3f} {ku:7.3f} {tail:8.1e}", flush=True)
    nu = next((n for n in (3, 4, 5) if tails[f"student_nu{n}"] < 1e-4), None)
    print(f"student: nu = {nu} (the smallest of 3, 4, 5 with mass beyond |a| > 8 below 1e-4)")
    diff = {"langevin": langevin_chain()[0]}
    for pnu in (5.0, 6.0, 8.0):
        diff[f"pearson_nu{pnu:g}"] = pearson_chain(pnu)[0]
    for name, ch in diff.items():
        m, v, c1, sk, ku, tail = stats(ch.log_k_fine())
        print(f"{name:14s} {m:8.4f} {v:8.4f} {c1:8.4f} {sk:7.3f} {ku:7.3f} {tail:8.1e}  Delta = {ch.delta:.6f}", flush=True)
        tails[name] = tail
    pnu = next((n for n in (5, 6, 8) if tails[f"pearson_nu{n}"] < 1e-4), None)
    print(f"pearson: nu = {pnu} (the smallest of 5, 6, 8 with mass beyond |a| > 8 below 1e-4)")
    return eps, r, nu, diff["langevin"], diff.get(f"pearson_nu{pnu}") if pnu else None


# ------------------------------------------------------------------ part 2


def families(eps, r, nu, langevin, pearson):
    import calibrate_e60 as c60
    import e03_baselines as e03
    import e57_chain_families as e57
    from markov_score.bp_grid import make_grid
    from markov_score.priors import GaussianAR1, LaplaceAR1, StudentTAR1, UniformAR1
    from markov_score.stationary import invariant_log_density_eig

    grid, w = make_grid(8.0, 401)
    chains = {"gauss": GaussianAR1(RHO), "laplace": LaplaceAR1(RHO), "student": StudentTAR1(RHO, float(nu)),
              "uniform": UniformAR1(RHO), "skew": SkewChainForCalibration(eps, r), "langevin": langevin, "pearson": pearson}
    print("best member per family, fitted to the exact pair law; error = BP score error (E03's estimand, raw) on the "
          "calibration bundle e61-calibration (1024 chains); floor = the LMMSE of the chain's exact covariance")
    print(f"{'chain':10s} {'family':8s} {'KL/edge':>10s} {'error':>7s} {'/floor':>7s}  notes")
    for name, prior in chains.items():
        lk_true = c60.column_normalised(prior.log_transition_matrix(grid), w)
        mu = np.exp(invariant_log_density_eig(prior, grid, w).log_density) * w
        P = np.exp(lk_true + np.log(w)[:, None]) * (mu / mu.sum())[None, :]
        P /= P.sum()
        _, test = e03._bundle(prior, lk_true, grid, w, "e61-calibration", 0, 1024)
        mean, cov = e57.chain_covariance(lk_true, grid, w)
        floor = e03._risk(e57._lmmse_pairs(cov, mean, test))
        exact = e03._risk(e57.e56._pairs(c60.Fixed("true", lk_true), test, grid, w))

        def report(fam, lk, notes):
            lk = c60.column_normalised(lk, w)
            on = P > 0
            kl = float((P[on] * (lk_true[on] - lk[on])).sum())
            err = e03._risk(e57.e56._pairs(c60.Fixed(fam, lk), test, grid, w))
            ratio = f"{err / floor:7.3f}" if floor > 1e-3 else f"{'-':>7s}"
            print(f"{name:10s} {fam:8s} {kl:10.2e} {err:7.4f} {ratio}  {notes}", flush=True)

        print(f"{name:10s} {'floor':8s} {'':10s} {floor:7.4f} {'':7s}  the true kernel through the same BP: {exact:.2e}")
        lk, d = c60.fit_shift(P, grid, name)
        report("shift8", lk, f"rho {d['rho']:.3f}")
        lk, d = c60.fit_free(P, grid, w, 4, 0)
        report("free4", lk, f"effective components {d['eff_components']:.2f}, slopes {np.round(np.sort(d['slopes']), 2).tolist()}")


class SkewChainForCalibration:
    """The skewed chain as a prior: closed-form kernel, a_1 ~ N(0, 1) as every prior in the package starts."""

    def __init__(self, eps, r):
        self.eps, self.r = eps, r
        self.m, self.s, _ = skew_moments(eps)

    @property
    def name(self):
        return "skew"

    def log_transition_matrix(self, grid):
        return skew_log_k(self.eps, self.r, grid)

    def sample(self, rng, n):
        a = np.empty(n)
        a[0] = rng.standard_normal()
        z = _t_inv(self.m + self.s * a[0], self.eps)
        for i in range(1, n):
            z = self.r * z + np.sqrt(1 - self.r ** 2) * rng.standard_normal()
            a[i] = (_t(z, self.eps) - self.m) / self.s
        return a


def main() -> int:
    eps, r, nu, langevin, pearson = chain_table()
    if "--families" in sys.argv:
        families(eps, r, nu, langevin, pearson)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
