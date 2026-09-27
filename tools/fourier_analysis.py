#!/usr/bin/env python
"""Q5 analytically: the BP message in Fourier, its mode decay, the tilt, and why cumulants fail.

The user's direction (27 Sep): the Fourier idea is worth exploring ANALYTICALLY before any
solver is built. E23 had already measured that an adapted Fourier basis needs ~23 modes but
converges non-monotonically in 91 of 144 cells while Gram-Charlier fails in 144/144, so there
was a phenomenon to explain rather than a solver to write.

Three results, each verified numerically against the project's own exact BP.

T1. CLOSED FORM AND MODE DECAY.
    One forward BP step is L_{i+1}(a') = int p_eps(a' - rho a) g_i(a) da with g_i = L_i ell_i.
    Substituting y = rho a makes it a convolution with a dilated g_i, so in Fourier

        phi_{i+1}(omega) = phi_eps(omega) psi_i(rho omega),      psi_i = CF of g_i.      (A)

    With a flat likelihood psi_i = phi_i, and unrolling from phi_1 = phi_mu,

        phi_{k+1}(omega) = phi_mu(rho^k omega) prod_{j=0}^{k-1} phi_eps(rho^j omega).     (B)

    As k -> inf this is prod_{j>=0} phi_eps(rho^j omega), the stationary law's CF: an INFINITE
    product, in no finite-parameter family. That is why E22's P3 was falsified -- with a
    flat likelihood the recursion drives messages to exactly this object.

    For Laplace(b), phi_eps = 1/(1 + b^2 omega^2), and the terms with b rho^j |omega| >> 1
    dominate; counting the J = log(b|omega|)/log(1/rho) of them gives

        log|phi_inf(omega)| ~ -(log(b|omega|))^2 / log(1/rho).                           (C)

    A GAUSSIAN DECAY IN log omega: super-polynomial, sub-exponential.

T2. THE TILT IS THE ONE OPERATION THAT DOES NOT SIMPLIFY, AND IT IS HARDEST AT LOW NOISE.
    Two of three BP operations are free in Fourier: the transition is the dilation in (A) and
    adding the innovation is the multiplication by phi_eps. The Gaussian likelihood is a
    product in real space, hence a CONVOLUTION in mode space, with

        phi_ell(omega) ∝ e^{i omega x_i/alpha} e^{-omega^2/(2 sigma_w^2)},
        sigma_w = alpha / sqrt(Delta) = e^{-t} / sqrt(1 - e^{-2t}).

    sigma_w DECREASES with t, so the tilt is widest at LOW noise and becomes a delta at high
    noise -- the opposite of the intuition that high noise is the hard case.

T3. WHY CUMULANTS DIVERGE WHERE FOURIER CONVERGES -- after two wrong answers.
    The Hermite/Gram-Charlier expansion converges in L^2 iff Cramer's condition holds:
    int q(z)^2/varphi(z) dz < infinity for q the density being expanded. TWO EXPLANATIONS WERE
    REFUTED BEFORE THIS ONE:

      * "it is the metric and the positive cone" -- refuted: the UNWEIGHTED L^2 error of the
        truncation diverges too, and it starts diverging BEFORE any negative mass appears
        (at t = 0.2216 the order-6 truncation has zero negative mass and the L^2 error has
        already risen from 2.2e-2 to 7.6e-2).
      * "it is my arithmetic" -- refuted by a control with a known answer: for a Gaussian
        message every c_n with n >= 3 comes out at 1e-17 to 1e-34 and the reconstruction's
        L^2 error stays at 1e-13 through order 30, on three different grids.

    And the FIRST Cramer test was itself wrong, in a way worth recording: it evaluated
    int p^2/varphi on the RAW message, whose variance is 0.33-0.99, so the integral converged
    and the condition looked satisfied. The expansion is applied to the STANDARDISED message,
    and standardising a narrow message to unit variance is exactly what breaks integrability.
    Worse, the first test looked for the divergence by REFINING THE GRID, which cannot show it:
    for a standardised Laplace the integral reads 6113 at N_g = 1601, 3201, 6401 and 12801 on a
    fixed domain. Grid refinement tests QUADRATURE; only WIDENING THE DOMAIN tests TAIL
    INTEGRABILITY. Widened, the same integral reads 6.1e3, 9.4e15, 1.7e35, 2.9e61, 4.7e94 at
    half-width 8, 12, 16, 20, 24.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
from numpy.polynomial.hermite_e import hermeval  # noqa: E402

from frozen_config import FROZEN  # noqa: E402
from markov_score.bp_grid import grid_bp, make_grid  # noqa: E402
from markov_score.noising import alpha_delta, log_likelihood_matrix  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv  # noqa: E402

RHO, L = FROZEN.rho, FROZEN.n_sites
T = tuple(FROZEN.t_grid)


def laplace_cf(omega, b):
    return 1.0 / (1.0 + (b * omega) ** 2)


def message_cf_closed_form(omega, k, rho, b):
    """(B): the CF of the k-th forward BP message under a FLAT likelihood."""
    out = np.exp(-0.5 * (rho**k * omega) ** 2)          # phi_mu(rho^k omega), mu = N(0,1)
    for j in range(k):
        out = out * laplace_cf(rho**j * omega, b)
    return out


def t1_closed_form(rows):
    prior = LaplaceAR1(RHO); b = prior.b
    grid, w = make_grid(12.0, 4001)
    log_k = prior.log_transition_matrix(grid)
    r = grid_bp(grid, w, log_k, np.zeros((L, len(grid))), np.zeros(L), 1.0, 1.0,
                return_messages=True)
    omega = np.linspace(0, 40, 801)
    print("T1(B)  CF of the grid BP message against the closed form (flat likelihood)")
    print(f"{'site k':>8} {'max abs err':>14}")
    worst = 0.0
    for k in (1, 2, 4, 8, 16, L - 1):
        meas = (r.forward[k][None, :] * np.exp(1j * omega[:, None] * grid[None, :])
                * w[None, :]).sum(axis=1).real
        err = float(np.abs(meas - message_cf_closed_form(omega, k, RHO, b)).max())
        worst = max(worst, err)
        print(f"{k:>8} {err:>14.3e}")
        rows.append(dict(target="T1_closed_form", key=f"site={k}", value=err))
    print(f"   worst {worst:.3e} -- the grid's own quadrature error at h = {grid[1]-grid[0]:.4f}\n")
    return worst


def t1_decay(rows):
    print("T1(C)  log|phi_inf| against -(log b|omega|)^2 / log(1/rho)")
    print(f"{'rho':>6} {'fitted slope':>14} {'-1/log(1/rho)':>15} {'ratio':>8} {'R^2':>11}")
    om = np.exp(np.linspace(np.log(3.0), np.log(3e4), 40))
    for rho in (0.5, 0.7, 0.85, 0.95, 0.99):
        b = float(np.sqrt((1 - rho**2) / 2))
        lp = np.array([-np.sum(np.log1p((b * rho ** np.arange(0, 20000) * o) ** 2)) for o in om])
        X = (np.log(b * om)) ** 2
        A = np.vstack([X, np.ones_like(X)]).T
        c, *_ = np.linalg.lstsq(A, lp, rcond=None)
        r2 = 1 - (lp - A @ c).var() / lp.var()
        tgt = -1.0 / np.log(1.0 / rho)
        print(f"{rho:>6} {c[0]:>14.5f} {tgt:>15.5f} {c[0]/tgt:>8.4f} {r2:>11.8f}")
        rows.append(dict(target="T1_decay", key=f"rho={rho}", value=c[0] / tgt))
    print()


def t2_tilt(rows):
    print("T2  tilt bandwidth in mode space, sigma_w = alpha/sqrt(Delta); W = 6 -> spacing pi/W")
    print(f"{'t':>8} {'sigma_w':>9} {'3 sigma_w in modes':>20} {'verdict':>9}")
    W = 6.0
    for t in T:
        al, de = alpha_delta(t)
        sw = al / np.sqrt(de)
        band = 3 * sw * W / np.pi
        print(f"{t:>8.4f} {sw:>9.4f} {band:>20.2f} {'banded' if band < 12 else 'DENSE':>9}")
        rows.append(dict(target="T2_tilt", key=f"t={t}", value=band))
    print("   widest at LOW noise, a delta at high noise -- opposite to the intuition.\n")


def _cramer_standardised(p, grid, w):
    m = float((grid * p * w).sum())
    v = float(((grid - m) ** 2 * p * w).sum())
    s = np.sqrt(v)
    z = (grid - m) / s
    ph = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    return float(((s * p) ** 2 / ph * (w / s)).sum()), v


def t3_cramer(rows):
    prior = LaplaceAR1(RHO)
    print("T3  Cramer's condition int q^2/varphi for the STANDARDISED message")
    print("    CONTROLS first (a test with no control is how the first two answers were wrong)")
    grid, w = make_grid(FROZEN.half_width, 3201)
    p = np.exp(-0.5 * grid**2 / 0.4) / np.sqrt(2 * np.pi * 0.4)
    v0, _ = _cramer_standardised(p, grid, w)
    print(f"      standardised Gaussian -> {v0:.10f}   (must be exactly 1)")
    rows.append(dict(target="T3_control_gaussian", key="std_normal", value=v0))
    print("      standardised Laplace, FIXED domain, refining the grid (cannot detect it):")
    for ng in (1601, 3201, 6401, 12801):
        g2, w2 = make_grid(FROZEN.half_width, ng)
        pl = np.exp(-np.abs(g2) * np.sqrt(2.0)) * np.sqrt(2.0) / 2
        val, _ = _cramer_standardised(pl, g2, w2)
        print(f"        N_g={ng:>6} -> {val:>12.4f}")
    print("      standardised Laplace, WIDENING the domain (does detect it):")
    for half in (8.0, 12.0, 16.0, 20.0, 24.0):
        g2, w2 = make_grid(half, 20001)
        pl = np.exp(-np.abs(g2) * np.sqrt(2.0)) * np.sqrt(2.0) / 2
        val, _ = _cramer_standardised(pl, g2, w2)
        print(f"        half-width={half:>5} -> {val:>12.4g}")
        rows.append(dict(target="T3_control_laplace", key=f"A={half}", value=val))

    print("\n    THE ACTUAL BP MESSAGES (site 16), standardised, over widening domains:")
    rng = rng_for("fourier-t3")
    A = prior.sample(rng, L)
    print(f"{'t':>8} {'msg var':>9} " + "".join(f"{'A=' + str(h):>13}" for h in (8.0, 12.0, 16.0, 20.0))
          + "   verdict")
    for t in (T[0], T[4], T[8], T[-1]):
        al, de = alpha_delta(t)
        x = al * A + np.sqrt(de) * rng.standard_normal(L)
        vals, var = [], np.nan
        for half in (8.0, 12.0, 16.0, 20.0):
            g2, w2 = make_grid(half, 12001)
            lk2 = prior.log_transition_matrix(g2)
            r2 = grid_bp(g2, w2, lk2, log_likelihood_matrix(g2, x, al, de), x, al, de,
                         return_messages=True)
            pm = r2.forward[16]; pm = pm / (pm * w2).sum()
            val, var = _cramer_standardised(pm, g2, w2)
            vals.append(val)
        grew = vals[-1] / vals[0]
        print(f"{t:>8.4f} {var:>9.4f} " + "".join(f"{v:>13.4g}" for v in vals)
              + f"   {'DIVERGES' if grew > 1.5 else 'finite'} (x{grew:.3g})")
        rows.append(dict(target="T3_message", key=f"t={t}", value=grew))
    print("\n    Cramer FAILS at every level: Gram-Charlier is provably outside its L^2")
    print("    convergence class here, which is E23's 144/144 against Fourier's 10/144.\n")


def t3_controls_that_refuted_the_other_answers(rows):
    """The Gaussian control that ruled out my own arithmetic, kept because it is what makes
    the T3 conclusion trustworthy rather than merely tidy."""
    print("T3 control  Gaussian message: every c_n with n >= 3 must be 0 and the")
    print("            reconstruction exact at EVERY order (this is what ruled out arithmetic)")
    print(f"{'N_g':>7} {'half':>6} {'max|c_n| n>=3':>15} {'L2 err at order 30':>20}")
    for half, ng in ((FROZEN.half_width, 1601), (FROZEN.half_width, 6401), (20.0, 20001)):
        grid, w = make_grid(half, ng)
        p = np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi)
        m = float((grid * p * w).sum()); v = float(((grid - m) ** 2 * p * w).sum())
        s = np.sqrt(v); z = (grid - m) / s
        ph = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        coef = np.zeros(31)
        for n in range(31):
            e = np.zeros(n + 1); e[n] = 1.0
            coef[n] = float((hermeval(z, e) * p * w).sum()) / math.factorial(n)
        rec = ph * hermeval(z, coef) / s
        l2 = float(np.sqrt(((p - rec) ** 2 * w).sum()))
        mx = float(np.abs(coef[3:]).max())
        print(f"{ng:>7} {half:>6} {mx:>15.3e} {l2:>20.3e}")
        rows.append(dict(target="T3_control_arith", key=f"Ng={ng},A={half}", value=l2))
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/fourier_analysis")
    args = ap.parse_args()
    rows: list[dict] = []
    print(f"rho = {RHO}, L = {L}, Laplace scale b = {LaplaceAR1(RHO).b:.6f}\n")
    t1_closed_form(rows)
    t1_decay(rows)
    t2_tilt(rows)
    t3_controls_that_refuted_the_other_answers(rows)
    t3_cramer(rows)
    out = ensure_dir(args.out)
    write_csv(out / "fourier_analysis.csv", rows)
    print(f"wrote {(out / 'fourier_analysis.csv').relative_to(ROOT)}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
