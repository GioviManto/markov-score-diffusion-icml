#!/usr/bin/env python
"""E23 -- how many numbers does a BP message actually need?

WHY THIS EXPERIMENT, AND WHY IT IS A MEASUREMENT AND NOT A NEW SOLVER
---------------------------------------------------------------------
Jerome, 25 September 2026 (`docs/JEROME_CALL_2026-09-25.md`, Q5):

    "The bottleneck of this whole thing, honestly, is the grid. The grid updates of
     messages, it's a pain in the ***. I wish we didn't have to do this."
    "I wonder if we even need a grid in real space. Because maybe we can do everything in
     Fourier. ... Instead of having a grid, we have a truncation on the number of Fourier
     modes. ... Let's say the first 20 Fourier modes explain 99% of this space."

The plan of record was to build a Fourier BP solver. Three measurements since then say
that would have been a week spent before finding out whether the premise holds:

  (1) E24 `anatomy`: at N_g = 401 only 23.8% of a BP sweep is the N_g^2 matmul; the rest
      is O(B L N_g) traffic over 158 MB of message arrays. An FFT that made the matmul
      FREE buys <= 1.3x. Acceleration is not where the time is -- TRUNCATION is.
  (2) E22: the messages carry large high-order structure. |kappa_6| runs 1 to 228 and is
      never small, so a low-order truncation cannot be assumed to be cheap.
  (3) A better quadrature rule does not help either. Gauss-Legendre, which should beat
      trapezoid on a smooth integrand, is an order of magnitude WORSE here (5e-2 vs
      8.8e-3 at 101 nodes): the Laplace kink and the narrow likelihood are not
      polynomial-like, while the uniform trapezoid rule is spectrally accurate for a
      rapidly decaying integrand (Euler-Maclaurin -- every boundary term vanishes).
      Part `quadrature` establishes this properly rather than on one configuration.

So the question that decides whether ANY of this is worth building is prior to the
solver: **for the messages exact BP actually produces, how many degrees of freedom does
each candidate representation need to reach a given accuracy?** If 20 Fourier modes
really do explain 99%, a Fourier solver is worth a week. If it takes 200, nothing built
on that representation can beat the 401-point grid, and we should say so and stop.

This experiment answers that, cheaply, on the true messages, before committing to a
solver. A negative answer is as useful as a positive one and costs two days instead of
two weeks.

WHAT IS MEASURED
----------------
Part `representation`: true messages from grid BP at a fine reference resolution, each
compressed to D degrees of freedom by each candidate, then scored by KL(true ||
reconstruction):

    fourier_D    keep the D lowest Fourier modes of the density, inverse-transform,
                 clip and renormalise -- literally "a truncation on the number of
                 Fourier modes"
    cumulant_D   Gram-Charlier: N(m, v) times a Hermite series through order D, with
                 coefficients from the message's own cumulants -- the representation in
                 which the OU channel acts diagonally (kappa_m -> e^{-mt} kappa_m), and
                 in which order 2 IS the Gaussian closure
    grid_D       D uniform points with trapezoid, the status quo, as the yardstick every
                 other representation has to beat

The headline number is D*(eps): the degrees of freedom each representation needs to
reach the accuracy the production grid already delivers.

A KNOWN MEASUREMENT FLOOR, stated so it is not read as a property of the method. Both
Fourier variants are scored by mapping the reconstruction back onto the reference grid,
and `fourier_adapted` additionally resamples onto and off its own window. Those two
interpolations put a floor of order 1e-5 on the measured KL at the quick settings, which
is why the Fourier curves flatten at high dof rather than continuing down. The production
reference grid is 3201 points, which pushes the floor well below the region where the
comparison is decided -- the LOW-dof end, where the representations differ by orders of
magnitude. Read the small-dof columns; treat a flat tail near 1e-5 as the instrument, not
the method.

PREDECLARED PREDICTIONS, fixed before the run:
  R1  grid_D reaches 1e-6 at D of a few hundred (it is the status quo; this is a check
      that the yardstick behaves).
  R2  fourier_D needs MORE than 20 modes -- Jerome's "first 20 explain 99%" is optimistic
      for these messages. 99% of the spectral mass is not 1e-6 in KL, and the messages
      have exponential (Laplace) or power-law (Student) tails whose transforms decay
      slowly. Measured in TWO variants, because reporting only the first would be a
      strawman: `fourier` truncates on the solver's shared [-12, 12] domain, which is what
      a drop-in replacement faces and where the lowest harmonics are 48x wider than a
      typical message; `fourier_adapted` truncates on a per-message window, which is what
      an actual Fourier solver would do. The gap between them is the price of the SHARED
      DOMAIN, reported separately from the price of the representation.
  R3  cumulant_D does NOT converge monotonically: Gram-Charlier is asymptotic, not
      convergent, so error falls then RISES with order. If so, the cumulant ladder is
      usable only at low order and the "one dial from Gaussian closure to exact" framing
      has a ceiling that must be stated.
  R4  no representation beats grid_D by the >4x that would be needed to matter, given
      that E24 caps the achievable speedup anyway.

R2, R3 and R4 are all predictions that the Fourier/cumulant route DOES NOT pay. They are
written down in advance precisely because the project has an incentive to find otherwise,
and because the previous two experiments both falsified a prediction we preferred.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score.bp_grid import grid_bp, grid_bp_batch, make_grid  # noqa: E402
from markov_score.noising import alpha_delta, log_likelihood_matrix  # noqa: E402
from markov_score.priors import GaussianAR1, LaplaceAR1, StudentTAR1, UniformAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

FAMILIES = {
    "laplace": LaplaceAR1, "gaussian": GaussianAR1,
    "student": StudentTAR1, "uniform": UniformAR1,
}

SETTINGS = dict(
    rhos=(0.5, 0.85, 0.95),
    n_sites=FROZEN.n_sites,
    families=("laplace", "gaussian", "student", "uniform"),
    levels=FROZEN.t_grid,
    n_chains=64,
    sites=(4, 8, 16, 31),
    ref_grid=(3201, 16.0),               # the reference every representation is scored against
    dofs=(5, 9, 15, 21, 31, 45, 65, 101, 151, 201, 301, 401),
    # --- part `quadrature` ------------------------------------------------------
    quad_rules=("uniform_trapz", "gauss_legendre", "clenshaw_curtis"),
    quad_nodes=(31, 51, 71, 101, 151, 201, 301, 401, 601, 801),
    quad_half_width=12.0,
    quad_chains=256,
    workers=1,
)
QUICK = dict(rhos=(0.85,), families=("laplace",), levels=(FROZEN.t_grid[3], FROZEN.t_grid[8]),
             n_chains=8, sites=(4, 16), ref_grid=(801, 12.0), dofs=(9, 21, 45, 101, 201),
             quad_nodes=(51, 101, 201), quad_chains=32)

_LOG2PI = float(np.log(2.0 * np.pi))


def _norm(p, w):
    m = float(np.sum(p * w))
    return p / m if m > 0 else p


def _kl(p, q, w):
    q = np.maximum(q, 1e-300)
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m]) * w[m]))


# ------------------------------------------------------------------ representations


def _fourier_truncate(p, grid, w, n_modes):
    """Keep the `n_modes` lowest Fourier modes of the density, then clip and renormalise.

    This is exactly the operation Jerome proposed: the message lives on a finite interval,
    is expanded in a Fourier series on it, and all but the lowest modes are discarded. The
    real FFT is used, so `n_modes` counts complex modes (DC plus n_modes-1 harmonics), i.e.
    2*n_modes - 1 real degrees of freedom; that is what the `dof` column reports so the
    comparison with a grid of D points is like-for-like.

    Clipping is necessary and is part of the method, not a fudge: a truncated Fourier
    series of a positive function is not positive, and a message has to be a density
    before BP can multiply it by anything.
    """
    c = np.fft.rfft(p)
    c[n_modes:] = 0.0
    q = np.fft.irfft(c, n=len(p))
    return _norm(np.maximum(q, 0.0), w)


def _fourier_adapted(p, grid, w, n_modes, n_std=8.0):
    """Fourier truncation on a domain matched to the MESSAGE, not to the solver's grid.

    WHY BOTH VARIANTS EXIST, AND WHY REPORTING ONLY THE FIRST WOULD BE A STRAWMAN.
    The solver's grid spans [-12, 12] because it has to hold every message of every chain
    at every noise level with one shared kernel matrix. A typical message is ~0.5 wide. So
    on the shared domain the lowest Fourier harmonics have wavelengths 24, 12, 8, 6 -- all
    far too broad to resolve the message at all, and "the first 20 modes" describes mostly
    empty space. That is a real constraint on a DROP-IN Fourier replacement, and `fourier`
    measures it.

    It is not a fair test of the IDEA. An actual Fourier solver would carry each message on
    its own scale. This variant does that: crop to mean +- n_std standard deviations,
    resample, truncate there, map back. Its dof count includes the two numbers (mean,
    scale) needed to say where the window is.

    The gap between the two is therefore the price of the shared domain, reported
    separately from the price of the representation -- which is the distinction that
    decides whether an adaptive-domain solver is worth building even if a drop-in is not.
    """
    mu = float(np.sum(grid * p * w))
    var = float(np.sum((grid - mu) ** 2 * p * w))
    s = np.sqrt(max(var, 1e-300))
    lo, hi = mu - n_std * s, mu + n_std * s
    m = len(grid)
    loc = np.linspace(lo, hi, m)
    vals = np.interp(loc, grid, p)
    c = np.fft.rfft(vals)
    c[n_modes:] = 0.0
    back = np.fft.irfft(c, n=m)
    q = np.interp(grid, loc, back, left=0.0, right=0.0)
    return _norm(np.maximum(q, 0.0), w)


def _hermite_coeffs(p, grid, w, order):
    """Gram-Charlier A coefficients through `order`, from the message's own moments."""
    mu = float(np.sum(grid * p * w))
    var = float(np.sum((grid - mu) ** 2 * p * w))
    s = np.sqrt(max(var, 1e-300))
    z = (grid - mu) / s
    # He_k via the probabilists' recurrence; c_k = E[He_k(Z)] / k!
    He = [np.ones_like(z), z]
    for k in range(1, order + 1):
        He.append(z * He[k] - k * He[k - 1])
    from math import factorial
    coeffs = [float(np.sum(He[k] * p * w)) / factorial(k) for k in range(order + 1)]
    return mu, s, He, coeffs


def _cumulant_truncate(p, grid, w, order):
    """Gram-Charlier reconstruction through `order`: N(m, v) times a Hermite series.

    Order 2 is the Gaussian closure exactly (all higher coefficients dropped), so this is
    the ladder whose bottom rung is the Kalman/LMMSE answer of Q1 and whose top is the
    exact message -- if it converges, which R3 predicts it does not.
    """
    mu, s, He, coeffs = _hermite_coeffs(p, grid, w, max(order, 2))
    z = (grid - mu) / s
    base = np.exp(-0.5 * z**2) / (s * np.sqrt(2.0 * np.pi))
    series = np.ones_like(z)
    for k in range(3, order + 1):
        series = series + coeffs[k] * He[k]
    return _norm(np.maximum(base * series, 0.0), w)


def _grid_subsample(p, grid, w, n_points):
    """D uniform points with trapezoid, interpolated back -- the status quo yardstick."""
    lo, hi = float(grid[0]), float(grid[-1])
    sub = np.linspace(lo, hi, n_points)
    vals = np.interp(sub, grid, p)
    back = np.interp(grid, sub, vals)
    return _norm(np.maximum(back, 0.0), w)


# --------------------------------------------------------------------- quadrature rules


def quad_rule(name: str, n: int, half_width: float):
    """(nodes, weights) for int_{-A}^{A} f(u) du under the named rule."""
    A = float(half_width)
    if name == "uniform_trapz":
        return make_grid(A, n)
    if name == "gauss_legendre":
        x, w = np.polynomial.legendre.leggauss(n)
        return A * x, A * w
    if name == "clenshaw_curtis":
        # Curtis-Clenshaw on [-1, 1] via the standard cosine weights, scaled to [-A, A].
        k = np.arange(n)
        x = np.cos(np.pi * k / (n - 1))
        w = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in range(1, (n - 1) // 2 + 1):
                b = 1.0 if j == (n - 1) // 2 and (n - 1) % 2 == 0 else 2.0
                s += b / (4.0 * j**2 - 1.0) * np.cos(2.0 * j * np.pi * i / (n - 1))
            w[i] = 2.0 / (n - 1) * (1.0 - s)
        w[0] *= 0.5
        w[-1] *= 0.5
        order = np.argsort(x)
        return A * x[order], A * w[order]
    raise SystemExit(f"unknown quadrature rule {name!r}")


# ------------------------------------------------------------------------- cells


def _cell_repr(item):
    cfg, family, t, rho = item
    n_sites = int(cfg["n_sites"])
    n_ref, a_ref = int(cfg["ref_grid"][0]), float(cfg["ref_grid"][1])
    grid, w = make_grid(a_ref, n_ref)
    prior = FAMILIES[family](float(rho))
    log_K = prior.log_transition_matrix(grid)
    alpha, delta = alpha_delta(float(t))
    rng = rng_for("e23-repr", family, float(t), float(rho))

    rows = []
    for c in range(int(cfg["n_chains"])):
        a = prior.sample(rng, n_sites)
        x = alpha * a + np.sqrt(delta) * rng.standard_normal(n_sites)
        res = grid_bp(grid, w, log_K, log_likelihood_matrix(grid, x, alpha, delta),
                      x, alpha, delta, return_messages=True)
        for site in cfg["sites"]:
            site = int(site)
            if site >= n_sites:
                continue
            p = _norm(np.maximum(res.forward[site], 0.0), w)
            base = dict(family=family, level=float(t), rho=float(rho), site=site, chain=c,
                        ref_grid=n_ref, ref_half_width=a_ref)
            for D in cfg["dofs"]:
                D = int(D)
                # fourier: D real dof == (D+1)//2 complex modes
                n_modes = max(1, (D + 1) // 2)
                rows.append({**base, "repr": "fourier", "dof": 2 * n_modes - 1,
                             "param": n_modes,
                             "kl": _kl(p, _fourier_truncate(p, grid, w, n_modes), w)})
                # the same truncation on a per-message window; +2 dof for (mean, scale)
                rows.append({**base, "repr": "fourier_adapted", "dof": 2 * n_modes + 1,
                             "param": n_modes,
                             "kl": _kl(p, _fourier_adapted(p, grid, w, n_modes), w)})
                # cumulant: order D means mean + variance + (D-2) Hermite coefficients
                order = max(2, min(D, 40))
                rows.append({**base, "repr": "cumulant", "dof": order,
                             "param": order,
                             "kl": _kl(p, _cumulant_truncate(p, grid, w, order), w)})
                rows.append({**base, "repr": "grid", "dof": D, "param": D,
                             "kl": _kl(p, _grid_subsample(p, grid, w, D), w)})
    return rows


def part_representation(cfg, out):
    cells = [(cfg, f, t, r) for f in cfg["families"] for t in cfg["levels"] for r in cfg["rhos"]]
    rows = run_cells(cells, _cell_repr, int(cfg["workers"]), out, "representation.csv",
                     key=lambda c: f"{c[1]}_t{c[2]}_r{c[3]}")
    _summarise_repr(rows, out)
    return rows


def _summarise_repr(rows, out):
    import collections

    g = collections.defaultdict(list)
    for r in rows:
        g[(r["family"], r["rho"], r["level"], r["site"], r["repr"], r["dof"])].append(float(r["kl"]))
    summary = []
    for (fam, rho, t, site, rep, dof), kls in sorted(g.items()):
        k = np.array(kls)
        summary.append({"family": fam, "rho": rho, "level": t, "site": site, "repr": rep,
                        "dof": dof, "n_chains": len(k),
                        "kl_median": float(np.median(k)), "kl_p90": float(np.percentile(k, 90))})
    write_csv(out / "representation_summary.csv", summary)

    # D*(eps): the smallest dof reaching each target, per representation. This is the
    # headline -- "how many numbers does a message need" -- and the direct test of R2.
    targets = (1e-2, 1e-4, 1e-6)
    dstar = []
    keys = sorted({(s["family"], s["rho"], s["level"], s["site"]) for s in summary})
    for (fam, rho, t, site) in keys:
        for rep in sorted({s["repr"] for s in summary}):
            sel = sorted((s for s in summary
                          if (s["family"], s["rho"], s["level"], s["site"], s["repr"])
                          == (fam, rho, t, site, rep)), key=lambda s: s["dof"])
            if not sel:
                continue
            rec = {"family": fam, "rho": rho, "level": t, "site": site, "repr": rep,
                   "kl_at_max_dof": sel[-1]["kl_median"], "max_dof": sel[-1]["dof"],
                   # R3: Gram-Charlier is asymptotic, so the error should stop falling.
                   "kl_min": min(s["kl_median"] for s in sel),
                   "dof_at_kl_min": min(sel, key=lambda s: s["kl_median"])["dof"],
                   "non_monotone": bool(sel[-1]["kl_median"] > 1.5 * min(s["kl_median"] for s in sel))}
            for eps in targets:
                hit = [s["dof"] for s in sel if s["kl_median"] <= eps]
                rec[f"dstar_{eps:g}"] = min(hit) if hit else ""
            dstar.append(rec)
    write_csv(out / "dstar.csv", dstar)

    for rep in sorted({d["repr"] for d in dstar}):
        sel = [d for d in dstar if d["repr"] == rep]
        got = [d["dstar_1e-06"] for d in sel if d["dstar_1e-06"] != ""]
        nm = sum(1 for d in sel if d["non_monotone"])
        print(f"  [E23 D*] {rep:9s} reached 1e-6 in {len(got)}/{len(sel)} cells"
              + (f", median D* = {int(np.median(got))}" if got else "")
              + f"; non-monotone in {nm}/{len(sel)}", flush=True)


def _cell_quad(item):
    cfg, family, t, rho, rule = item
    n_sites, B = int(cfg["n_sites"]), int(cfg["quad_chains"])
    A = float(cfg["quad_half_width"])
    prior = FAMILIES[family](float(rho))
    alpha, delta = alpha_delta(float(t))
    rng = rng_for("e23-quad", family, float(t), float(rho))
    Ach = np.stack([prior.sample(rng, n_sites) for _ in range(B)])
    X = alpha * Ach + np.sqrt(delta) * rng.standard_normal(Ach.shape)

    gref, wref = make_grid(float(cfg["ref_grid"][1]), int(cfg["ref_grid"][0]))
    ref, _ = grid_bp_batch(gref, wref, prior.log_transition_matrix(gref), X, alpha, delta, chunk=None)

    rows = []
    for n in cfg["quad_nodes"]:
        n = int(n)
        try:
            g, w = quad_rule(rule, n, A)
            m, _ = grid_bp_batch(g, w, prior.log_transition_matrix(g), X, alpha, delta, chunk=None)
            e = float(np.sqrt(np.mean((m - ref) ** 2) / np.mean(ref**2)))
        except Exception as exc:
            rows.append({"family": family, "level": float(t), "rho": float(rho), "rule": rule,
                         "nodes": n, "rel_err": float("nan"), "error": str(exc)})
            continue
        rows.append({"family": family, "level": float(t), "rho": float(rho), "rule": rule,
                     "nodes": n, "rel_err": e, "error": ""})
    return rows


def part_quadrature(cfg, out):
    cells = [(cfg, f, t, r, q) for f in cfg["families"] for t in cfg["levels"]
             for r in cfg["rhos"] for q in cfg["quad_rules"]]
    rows = run_cells(cells, _cell_quad, int(cfg["workers"]), out, "quadrature.csv",
                     key=lambda c: f"{c[1]}_t{c[2]}_r{c[3]}_{c[4]}")
    import collections

    g = collections.defaultdict(list)
    for r in rows:
        v = float(r["rel_err"]) if r["rel_err"] not in ("", None) else float("nan")
        if np.isfinite(v):
            g[(r["rule"], int(r["nodes"]))].append(v)
    summary = [{"rule": rule, "nodes": n, "n_cells": len(v),
                "rel_err_median": float(np.median(v)), "rel_err_p90": float(np.percentile(v, 90))}
               for (rule, n), v in sorted(g.items())]
    write_csv(out / "quadrature_summary.csv", summary)
    for rule in sorted({s["rule"] for s in summary}):
        line = "  ".join(f"{s['nodes']}:{s['rel_err_median']:.1e}"
                         for s in summary if s["rule"] == rule)
        print(f"  [E23 quad] {rule:16s} {line}", flush=True)
    return rows


PARTS = {"representation": part_representation, "quadrature": part_quadrature}


def main():
    ap = experiment_parser("e23_representation",
                           "how many degrees of freedom a BP message needs, per representation")
    ap.add_argument("--workers", type=int, default=None, help="parallel cells")
    args = ap.parse_args()
    if args.list_parts:
        print("\n".join(PARTS))
        return
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    if args.workers is not None:
        cfg["workers"] = int(args.workers)
    out = ensure_dir(Path(args.output_dir))
    for name, fn in select_parts(PARTS, args.only).items():
        fn(cfg, out)
        write_json(out / f"params_{name}.json",
                   {**provenance(cfg), "settings": cfg, "quick": bool(args.quick),
                    "frozen": FROZEN.as_dict()})


if __name__ == "__main__":
    main()
