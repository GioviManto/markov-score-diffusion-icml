"""E52: log Z of the lattice field. Is it known in closed form, and how well can it be computed?

For a chain of L sites, log Z_L = L log lambda_0 + O(1), with lambda_0 the largest eigenvalue of the transfer
operator; the correlation length is xi = 1 / log(lambda_0 / lambda_1) sites. For a quadratic potential both are
in closed form; for the quartic double well they are not, so we compute them and check the computation.

The solver is deliberately plain: trapezoid rule, the operator symmetrised by the square roots of the weights,
a dense symmetric eigensolve of the top two eigenvalues. Predictions: experiments/registry/E52.json. Reading: tools/core/read_e52.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import experiment_parser, provenance  # noqa: E402
from e32_field_oracles import CELLS, resolve_cell  # noqa: E402
from markov_score.utils import ensure_dir, write_csv, write_json  # noqa: E402

HALF_WIDTH = 12.0
SIZES = (601, 1201, 2401, 4801)
AXIS_T = (0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0)


def quartic(phi):
    return 0.25 * (phi**2 - 1.0) ** 2


def harmonic(k):
    return lambda phi: 0.5 * k * phi**2


def harmonic_well(phi):
    return (phi - 1.0) ** 2           # V near phi = 1, V''(1) = 2


def top_two(V, T, h, n, half_width=HALF_WIDTH):
    """(lambda_0, lambda_1) of the trapezoid-discretised transfer operator."""
    u = np.linspace(-half_width, half_width, n)
    w = np.full(n, u[1] - u[0]); w[0] *= 0.5; w[-1] *= 0.5
    log_T = -(h / (2 * T)) * (V(u)[:, None] + V(u)[None, :]) - (u[None, :] - u[:, None]) ** 2 / (2 * T * h)
    A = np.exp(log_T) * np.sqrt(w)[:, None] * np.sqrt(w)[None, :]
    lam = eigh(A, eigvals_only=True, subset_by_index=[n - 2, n - 1])
    return float(lam[-1]), float(lam[-2])


def gaussian_closed_form(k, T, h):
    a = h * k / (2 * T) + 1 / (T * h)
    b = 1 / (T * h)
    c = np.sqrt(a**2 - b**2)
    return float(np.sqrt(2 * np.pi / (a + c))), float(b / (a + c))


def main():
    ap = experiment_parser("e52_log_partition", "log Z and xi of the lattice field from the transfer operator")
    args = ap.parse_args()
    out = ensure_dir(Path(args.output_dir))
    sizes = (601, 1201) if args.quick else SIZES
    rows = []
    for k in (1.0, 2.0):                                    # (a) Gaussian check
        for T in AXIS_T:
            l0, l1 = top_two(harmonic(k), T, 1.0, 2401)
            c0, r = gaussian_closed_form(k, T, 1.0)
            rows.append(dict(part="gaussian", potential=f"harmonic_k{k:g}", T=T, h=1.0, n=2401, log_lambda0=np.log(l0),
                             ratio=l1 / l0, closed_log_lambda0=np.log(c0), closed_ratio=r))
    points = [(T, 1.0, f"T{T:g}_h1") for T in AXIS_T]       # (b) the quartic
    for cell in CELLS:
        _, info = resolve_cell(cell)
        points.append((info["T"], info["h"], cell["id"]))
    for T, h, label in points:
        for n in sizes:
            l0, l1 = top_two(quartic, T, h, n)
            rows.append(dict(part="quartic", potential="double_well", label=label, T=T, h=h, n=n,
                             log_lambda0=np.log(l0), ratio=l1 / l0, xi_sites=1.0 / np.log(l0 / l1)))
    for T in AXIS_T:                                         # (c) the harmonic-well reference
        l0, _ = top_two(harmonic_well, T, 1.0, 2401)
        rows.append(dict(part="harmonic_well", potential="well_k2", T=T, h=1.0, n=2401, log_lambda0=np.log(l0)))
    write_csv(out / "spectrum.csv", rows)
    write_json(out / "params_spectrum.json", {**provenance(), "settings": {"half_width": HALF_WIDTH, "sizes": sizes,
                                                                          "axis_T": AXIS_T}, "quick": bool(args.quick)})
    print(f"  [E52] {len(rows)} rows")


if __name__ == "__main__":
    main()
