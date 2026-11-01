#!/usr/bin/env python
"""Q10: the information curves measure kappa_2 and kappa_4, not rho and beta.

Jerome, 25 September 2026: "I don't think you explicitly can quantify information on beta,
because to resolve beta, technically you need all the modes, all the cumulants. So I think
here it is more ... the information of the second cumulant and of the fourth cumulant."

He is right, and the fix is NOT a relabel of the axis. E01's `shape` part computes the
Fisher information of the generalised-Gaussian chain about theta = (rho, q, beta). Going to
phi = (rho, kappa_2, kappa_4) is a reparameterisation, so it needs the delta method:

    I_phi = J^{-T} I_theta J^{-1},        J = d phi / d theta,
    I_phi^{-1} = J I_theta^{-1} J^T,
    eff(phi_k) = 1 / (I_phi^{-1})_kk = 1 / (j_k^T I_theta^{-1} j_k),   j_k = row k of J.

For `GeneralizedGaussianKernel`, which holds Var(innovation) = q at every beta:

    kappa_2 = q,        kappa_4 = gamma_2(beta) * q^2,
    gamma_2(beta) = Gamma(5/b)Gamma(1/b)/Gamma(3/b)^2 - 3,

so J is lower triangular with rows (1,0,0), (0,1,0), (0, 2 gamma_2 q, gamma_2' q^2). Two
consequences, both checkable and both reported:

  * eff(kappa_2) == eff(q) and eff(rho) is unchanged: for those two coordinates the relabel
    IS exact, because their rows of J are unit vectors.
  * eff(kappa_4) != eff(beta), because j_3 mixes the q direction in. The ratio is what this
    tool prints, and it is the size of the error a relabel would have made.

WHAT IS RECONSTRUCTED AND WHY IT IS SAFE. shape_information.csv stores diag_*, eff_* and
corr_*, not the matrix. `markov_score.information.summarise` defines
corr = I / sqrt(outer(diag(I), diag(I))), so I_xx = diag_x and I_xy = corr_xy
sqrt(diag_x diag_y) recover I_theta exactly. The reconstruction is then VERIFIED against
the stored eff_* before anything is computed from it, and the tool exits non-zero if it
does not agree -- reconstructing a matrix from summary statistics is the kind of step that
silently produces plausible numbers when the convention is wrong.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import digamma, gammaln

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from markov_score.shape_kernel import generalized_gaussian_excess_kurtosis  # noqa: E402

NAMES = ("rho", "q", "beta")


def gg_excess_kurtosis_prime(beta: float) -> float:
    """d/dbeta of Gamma(5/b)Gamma(1/b)/Gamma(3/b)^2 - 3, in closed form.

    ln u = lnG(5/b) + lnG(1/b) - 2 lnG(3/b), so
    (ln u)' = -(1/b^2) [5 psi(5/b) + psi(1/b) - 6 psi(3/b)] and u' = u (ln u)'.
    """
    u = np.exp(gammaln(5.0 / beta) + gammaln(1.0 / beta) - 2.0 * gammaln(3.0 / beta))
    dlogu = -(1.0 / beta**2) * (
        5.0 * digamma(5.0 / beta) + digamma(1.0 / beta) - 6.0 * digamma(3.0 / beta)
    )
    return float(u * dlogu)


def _information_from_row(r) -> np.ndarray:
    d = np.array([r.diag_rho, r.diag_q, r.diag_beta], float)
    I = np.diag(d)
    for (i, j), c in (((0, 1), r.corr_rho_q), ((0, 2), r.corr_rho_beta), ((1, 2), r.corr_q_beta)):
        I[i, j] = I[j, i] = c * np.sqrt(d[i] * d[j])
    return I


def convert(df: pd.DataFrame, q: float) -> pd.DataFrame:
    out, worst_recon = [], 0.0
    for r in df.itertuples():
        I = _information_from_row(r)
        Iinv = np.linalg.inv(I)
        eff_stored = np.array([r.eff_rho, r.eff_q, r.eff_beta], float)
        eff_recon = 1.0 / np.diag(Iinv)
        worst_recon = max(worst_recon,
                          float(np.max(np.abs(eff_recon - eff_stored) / np.abs(eff_stored))))

        g2 = generalized_gaussian_excess_kurtosis(r.beta)
        g2p = gg_excess_kurtosis_prime(r.beta)
        J = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 2.0 * g2 * q, g2p * q**2]])
        Iphi_inv = J @ Iinv @ J.T
        eff_phi = 1.0 / np.diag(Iphi_inv)
        out.append(dict(
            beta=r.beta, rep=r.rep, t=r.t, n_chains=r.n_chains,
            kappa2=q, kappa4=g2 * q**2, gamma2=g2, gamma2_prime=g2p,
            eff_rho=eff_phi[0], eff_kappa2=eff_phi[1], eff_kappa4=eff_phi[2],
            eff_beta_stored=r.eff_beta,
            relabel_ratio=eff_phi[2] / r.eff_beta if r.eff_beta else np.nan,
        ))
    res = pd.DataFrame(out)

    if worst_recon > 1e-9:
        raise SystemExit(
            f"FATAL: I_theta reconstructed from diag_*/corr_* does not reproduce the stored "
            f"eff_* (worst relative error {worst_recon:.3e}). The `corr` convention in "
            f"markov_score.information.summarise must have changed; do not trust the output."
        )
    # The two coordinates whose Jacobian rows are unit vectors must come out UNCHANGED.
    for col, stored in (("eff_rho", df.eff_rho.to_numpy()), ("eff_kappa2", df.eff_q.to_numpy())):
        rel = np.max(np.abs(res[col].to_numpy() - stored) / np.abs(stored))
        if rel > 1e-9:
            raise SystemExit(f"FATAL: {col} should be an exact relabel, moved by {rel:.3e}")
    print(f"reconstruction check: worst relative error in eff_* = {worst_recon:.2e}")
    print("eff_rho and eff_kappa2 unchanged to 1e-9, as the Jacobian requires")
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, default=ROOT / "outputs/E01/shape_information.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/E01/cumulant_information.csv")
    ap.add_argument("--q", type=float, default=None,
                    help="innovation variance; default reads FROZEN.innovation_variance")
    args = ap.parse_args()
    if args.q is None:
        sys.path.insert(0, str(ROOT / "experiments"))
        from frozen_config import FROZEN
        args.q = float(FROZEN.innovation_variance)

    # Analytic derivative against a central difference: the one step here that is easy to
    # get wrong by a factor or a sign, and cheap to check.
    for b in (1.0, 1.25, 1.5, 2.0, 3.0):
        h = 1e-6
        fd = (generalized_gaussian_excess_kurtosis(b + h)
              - generalized_gaussian_excess_kurtosis(b - h)) / (2 * h)
        an = gg_excess_kurtosis_prime(b)
        if abs(an - fd) > 1e-5 * max(1.0, abs(fd)):
            raise SystemExit(f"FATAL: gamma_2'({b}) analytic {an:.9g} vs finite difference {fd:.9g}")
    print("gamma_2' analytic == central difference at beta in {1, 1.25, 1.5, 2, 3}")

    df = pd.read_csv(args.source)
    res = convert(df, args.q)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    print(f"\nq = kappa_2 = {args.q}\n")

    g = res.groupby("beta").agg(
        kappa4=("kappa4", "first"), gamma2=("gamma2", "first"),
        eff_beta=("eff_beta_stored", "median"), eff_kappa4=("eff_kappa4", "median"),
        ratio=("relabel_ratio", "median"))
    print("median over reps and noise levels:")
    print(g.to_string(float_format=lambda v: f"{v:.6g}"))
    print(f"\nwrote {args.out.relative_to(ROOT)}  ({len(res)} rows)")
    print("\nThe `ratio` column is the factor a RELABEL would have been wrong by: eff(kappa_4)")
    print("against the eff(beta) the curves currently report.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
