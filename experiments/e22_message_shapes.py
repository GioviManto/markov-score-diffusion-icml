#!/usr/bin/env python
"""E22 -- what the BP messages actually look like, and whether a parametric ansatz can carry them.

WHY THIS EXPERIMENT EXISTS
--------------------------
Jerome asked the same question three ways in the call of 25 September
(`docs/JEROME_CALL_2026-09-25.md`, Q2/Q3/Q4), and it has never been looked at:

    "If you look at the messages, mu_K ... what do they look like?"
    "What if you take a Laplace ansatz for the messages in the BP? ... And then is this
     exact when the innovation is exact? To me, it's not immediate that they match."
    "I wonder if it's not Gaussian plus -- like if you say mu of a is equal to sort of
     Gaussian [convolved with] the family of epsilon, P_epsilon -- is this exact?"

It is the cheapest open question on the list and it decides two things at once: whether
the real-space grid can be replaced by a handful of parameters (Q2/Q4), and how many
modes a truncated representation needs at each noise level (E23, Q5).

WHAT THE ANSWER SHOULD BE, DERIVED BEFORE MEASURING
---------------------------------------------------
The forward recursion is

    L_{i+1}  =  dilate_rho( L_i * ell_i )  conv  p_eps                            (*)

with ell_i the Gaussian likelihood at site i. Take Jerome's ansatz L_i = G conv p_eps
with G Gaussian, and push it through (*):

  - L_i * ell_i = (G conv p_eps) * N  -- a product of a convolution with a Gaussian,
    which is NOT itself of the form (Gaussian conv p_eps);
  - so L_{i+1} is not in the family, and the ansatz is NOT closed.

Two limits where it IS exact, and they are the predeclared structure of the result:

  (1) SITE 1. L_1 is the prior, which this project takes Gaussian, so L_2 = dilate(N * ell)
      conv p_eps = (Gaussian) conv p_eps exactly. The ansatz is exact one step in, by
      construction, and any measured divergence there is quadrature error -- which makes
      site 1 the experiment's own numerical control.
  (2) LARGE t. As t grows the likelihood flattens (alpha -> 0), ell_i -> const, and the
      product step stops doing anything. The ansatz becomes exact again.

PREDECLARED PREDICTIONS, fixed before the run:
  P1  The gauss_through_transition divergence at site 1 is at quadrature level. MEASURED
      floor for this construction: 3.8e-17 nats, so P1 is tested against the Gaussian-
      innovation control rather than a hardcoded threshold.
  P2  It GROWS with site index into the chain interior, then saturates: the family is not
      a fixed point of the recursion.
  P3  It is non-monotone in t -- small at large t (flat likelihood), largest at
      intermediate t where the likelihood is informative but the message is not yet
      Gaussian.
  P4  A plain Gaussian ansatz is uniformly worse than Gaussian-conv-p_eps at small t, and
      the two converge as t grows (Gaussianisation).
  P5  Message excess kurtosis decays with t; the number of cumulants needed above the
      quadrature floor is the mode budget E23 has to carry.

A failure of P2 -- divergence flat in site index at the control level -- would mean the
family IS closed and the grid can be replaced by four numbers. That would be a much
better outcome than the one predicted here, which is why the prediction is written down
in advance.

WHAT IS MEASURED
----------------
Part `shapes`: for each innovation family and noise level, the forward messages of many
chains, against three candidate families fitted by direct numerical minimisation of
KL(true || candidate) over the candidate's own parameters -- so every family is given its
best shot rather than a moment match:
    gaussian            2 params  (m, v)
    gauss_through_transition  2 params  (m, v of the pre-transition Gaussian; the true K)
    gauss_plus_laplace  4 params  (weight, m, Gaussian v, Laplace scale)
Reported as KL in nats and Hellinger, per (family, level, site), median and p90 over chains.

Part `cumulants`: standardised cumulants kappa_3..kappa_6 of the same messages against t
and site -- the representation-independent form of the same question, and the direct
input to E23's truncation order.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score.bp_grid import grid_bp, make_grid  # noqa: E402
from markov_score.noising import alpha_delta, log_likelihood_matrix  # noqa: E402
from markov_score.priors import GaussianAR1, LaplaceAR1, StudentTAR1, UniformAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

FAMILIES = {
    "laplace": lambda rho: LaplaceAR1(rho),
    "gaussian": lambda rho: GaussianAR1(rho),        # numerical control: every ansatz exact
    "student": lambda rho: StudentTAR1(rho),
    "uniform": lambda rho: UniformAR1(rho),
}

SETTINGS = dict(
    rho=FROZEN.rho,
    n_sites=FROZEN.n_sites,
    families=("laplace", "gaussian", "student", "uniform"),
    levels=FROZEN.t_grid,
    n_chains=64,
    # A finer grid than the production 401: this experiment's whole point is to separate a
    # modelling error from a quadrature error, so the quadrature floor has to sit well below
    # the effect. E11's envelope puts (801, 12) two orders below (401, 8).
    grid_size=801,
    half_width=12.0,
    sites=(1, 2, 4, 8, 16, 31),          # 1 is the analytic control (see P1)
    workers=1,
)
QUICK = dict(families=("laplace",), levels=(FROZEN.t_grid[3], FROZEN.t_grid[8]),
             n_chains=8, grid_size=401, half_width=8.0, sites=(1, 4, 16))

_LOG2PI = float(np.log(2.0 * np.pi))


# ------------------------------------------------------------------ candidate families


def _norm(p: np.ndarray, w: np.ndarray) -> np.ndarray:
    m = float(np.sum(p * w))
    return p / m if m > 0 else p


def _gaussian(grid, w, m, log_v):
    v = np.exp(log_v)
    return _norm(np.exp(-0.5 * (grid - m) ** 2 / v - 0.5 * (np.log(v) + _LOG2PI)), w)


def _gauss_through_transition(grid, w, m, log_v, K):
    """Jerome's ansatz: a Gaussian pushed through ONE transition of the true chain.

    Returns normalise( K @ (N(m, e^log_v) * w) ), i.e. p_eps convolved with the DILATED
    Gaussian, computed with the recursion's own kernel matrix and its own quadrature.
    The free parameters describe the pre-transition Gaussian, which is the natural
    parameterisation: the message at site i+1 is this object whenever L_i * ell_i is
    Gaussian.

    WHY IT IS BUILT FROM `K` AND NOT FROM A CONVOLUTION MATRIX. The first version built
    C[i, j] = p_eps(u_i - u_j) by lookup and applied it to an already-dilated Gaussian.
    Mathematically the same continuous convolution; discretely NOT the same, because BP
    evaluates p_eps at the off-grid points u_i - rho u_j with the dilation folded into the
    kernel, while a grid lookup can only evaluate on the grid. The gap is an O(dx^2)
    quadrature difference, and it measured 5.9e-8 nats at site 1 -- where the ansatz is
    ANALYTICALLY EXACT. Reported as-is it would have looked like a small failure of the
    ansatz; it was a failure of the comparison. With `K` the same site-1 fit returns
    3.8e-17, so the experiment now has an exact numerical control and every residual above
    it is model error.
    """
    return _norm(K @ (_gaussian(grid, w, m, log_v) * w), w)


def _gauss_plus_laplace(grid, w, logit, m, log_v, log_b):
    a = 1.0 / (1.0 + np.exp(-logit))
    b = np.exp(log_b)
    lap = _norm(np.exp(-np.abs(grid - m) / b) / (2.0 * b), w)
    return _norm(a * _gaussian(grid, w, m, log_v) + (1.0 - a) * lap, w)


def _kl(p, q, w):
    """KL(p || q) in nats, on the quadrature grid; q floored so the log is finite."""
    q = np.maximum(q, 1e-300)
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m]) * w[m]))


def _hellinger(p, q, w):
    return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(np.maximum(p, 0) * np.maximum(q, 0)) * w))))


def _fit(make, x0, p, grid, w):
    """Best KL(p || family) over the family's own parameters (Nelder-Mead)."""
    from scipy.optimize import minimize

    def obj(theta):
        try:
            q = make(grid, w, *theta)
        except FloatingPointError:
            return 1e6
        if not np.all(np.isfinite(q)):
            return 1e6
        return _kl(p, q, w)

    res = minimize(obj, np.asarray(x0, float), method="Nelder-Mead",
                   options=dict(maxiter=2000, xatol=1e-6, fatol=1e-12))
    return float(res.fun), make(grid, w, *res.x), res.x


# --------------------------------------------------------------------------- moments


def _moments(p, grid, w, k_max=6):
    """Central moments 1..k_max of a grid density."""
    mu = float(np.sum(grid * p * w))
    cen = grid - mu
    return mu, [float(np.sum(cen**k * p * w)) for k in range(2, k_max + 1)]


def _std_cumulants(p, grid, w):
    """Standardised cumulants kappa_3..kappa_6 (skew, excess kurtosis, and two more)."""
    mu, m = _moments(p, grid, w, 6)
    m2, m3, m4, m5, m6 = m
    s = np.sqrt(max(m2, 1e-300))
    k3, k4 = m3, m4 - 3 * m2**2
    k5 = m5 - 10 * m3 * m2
    k6 = m6 - 15 * m4 * m2 - 10 * m3**2 + 30 * m2**3
    return dict(mean=mu, var=m2,
                k3=k3 / s**3, k4=k4 / s**4, k5=k5 / s**5, k6=k6 / s**6)


# ------------------------------------------------------------------------ the cell


def _cell(item):
    cfg, family, t = item
    rho, n_sites = cfg["rho"], int(cfg["n_sites"])
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    prior = FAMILIES[family](rho)
    log_K = prior.log_transition_matrix(grid)
    alpha, delta = alpha_delta(float(t))

    # The innovation density on the grid, centred -- the fixed p_eps of Jerome's ansatz.
    # Read off the kernel's own row at parent state 0, so it is exactly the law BP uses
    # rather than a second construction of "the same" density.
    K = np.exp(log_K)

    rng = rng_for("e22-shapes", family, float(t))
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
            mu, m = _moments(p, grid, w, 2)
            v = max(m[0], 1e-12)
            cand = {
                "gaussian": _fit(_gaussian, [mu, np.log(v)], p, grid, w),
                "gauss_through_transition": _fit(
                    lambda g, ww, mm, lv: _gauss_through_transition(g, ww, mm, lv, K),
                    # start from the moments of the message pulled back through the
                    # transition: mean/rho, and a variance floored away from zero
                    [mu / max(cfg["rho"], 1e-3), np.log(max(v, 1e-3))],
                    p, grid, w),
                "gauss_plus_laplace": _fit(
                    _gauss_plus_laplace, [0.0, mu, np.log(v), 0.5 * np.log(v)], p, grid, w),
            }
            base = dict(family=family, level=float(t), site=site, chain=c,
                        grid_size=int(cfg["grid_size"]), half_width=float(cfg["half_width"]),
                        rho=float(rho), **_std_cumulants(p, grid, w))
            for name, (kl, q, _theta) in cand.items():
                rows.append({**base, "ansatz": name, "kl_nats": kl,
                             "hellinger": _hellinger(p, q, w)})
    return rows


def part_shapes(cfg, out):
    cells = [(cfg, f, t) for f in cfg["families"] for t in cfg["levels"]]
    rows = run_cells(cells, _cell, int(cfg["workers"]), out, "shapes.csv",
                     key=lambda c: f"{c[1]}_t{c[2]}")
    _summarise(rows, out)
    return rows


def _summarise(rows, out):
    """Median and p90 over chains, per (family, level, site, ansatz) -- and the P1/P2 test."""
    import collections

    g = collections.defaultdict(list)
    for r in rows:
        g[(r["family"], r["level"], r["site"], r["ansatz"])].append(r)
    summary = []
    for (fam, t, site, ans), rs in sorted(g.items()):
        kl = np.array([r["kl_nats"] for r in rs])
        he = np.array([r["hellinger"] for r in rs])
        summary.append({
            "family": fam, "level": t, "site": site, "ansatz": ans, "n_chains": len(rs),
            "kl_median": float(np.median(kl)), "kl_p90": float(np.percentile(kl, 90)),
            "kl_max": float(kl.max()),
            "hellinger_median": float(np.median(he)),
            "k4_median": float(np.median([r["k4"] for r in rs])),
            "k3_median": float(np.median([r["k3"] for r in rs])),
        })
    write_csv(out / "shapes_summary.csv", summary)

    # P1 / P2, stated as a table the reader can check at a glance: does the
    # gauss_through_transition divergence stay at the site-1 (analytic) level as depth grows?
    #
    # THE FLOOR IS MEASURED, NOT ASSUMED. A hardcoded "< 1e-8 nats is quadrature" would be
    # a different claim on every grid, and the first smoke run duly failed it on the coarse
    # quick grid while the effect it was meant to detect was three orders larger. The
    # Gaussian-innovation family is the control: there p_eps IS Gaussian, so every one of
    # the three candidate families contains the true message exactly, and whatever KL is
    # measured there is pure quadrature. That number, at the same (level, site, ansatz), is
    # the floor everything else is judged against.
    floor = {(s["level"], s["site"], s["ansatz"]): s["kl_median"]
             for s in summary if s["family"] == "gaussian"}
    if not floor:
        print("  [E22] no gaussian control in this run: P1 falls back to the site-1 value "
              "as its own floor, which cannot separate quadrature from model error", flush=True)

    verdict = []
    for fam in sorted({s["family"] for s in summary}):
        if fam == "gaussian":
            continue
        for t in sorted({s["level"] for s in summary if s["family"] == fam}):
            sel = {s["site"]: s for s in summary
                   if s["family"] == fam and s["level"] == t and s["ansatz"] == "gauss_through_transition"}
            if 1 not in sel or len(sel) < 2:
                continue
            deep = max(sel)
            f1 = floor.get((t, 1, "gauss_through_transition"))
            fd = floor.get((t, deep, "gauss_through_transition"))
            verdict.append({
                "family": fam, "level": t,
                "kl_site1": sel[1]["kl_median"], "kl_deepest": sel[deep]["kl_median"],
                "deepest_site": deep,
                "quadrature_floor_site1": f1 if f1 is not None else "",
                "quadrature_floor_deepest": fd if fd is not None else "",
                "growth_factor": (sel[deep]["kl_median"] / sel[1]["kl_median"]
                                  if sel[1]["kl_median"] > 0 else float("inf")),
                # P1: site 1 is analytically exact, so it should sit ON the measured floor.
                "P1_site1_at_floor": (bool(sel[1]["kl_median"] < 10 * max(f1, 1e-300))
                                      if f1 is not None else ""),
                # P2: the interior should be far ABOVE the floor -- that is the model error
                # the ansatz carries, separated from quadrature by construction.
                "P2_interior_above_floor": (bool(sel[deep]["kl_median"] > 10 * max(fd, 1e-300))
                                            if fd is not None else ""),
                "P2_grows_with_depth": bool(
                    sel[deep]["kl_median"] > 10 * max(sel[1]["kl_median"], 1e-15)),
            })
            v = verdict[-1]
            p1 = ("ok" if v["P1_site1_at_floor"] is True else
                  "FAIL" if v["P1_site1_at_floor"] is False else "no control")
            print(f"  [E22 verdict] {fam:9s} t={t:<7.4f} KL(site 1)={sel[1]['kl_median']:.3e} "
                  f"KL(site {deep})={sel[deep]['kl_median']:.3e} "
                  f"x{v['growth_factor']:.0f}  P1={p1} "
                  f"P2={'grows' if v['P2_grows_with_depth'] else 'FLAT -- ansatz may be closed!'}",
                  flush=True)
    write_csv(out / "closure_verdict.csv", verdict)


def part_cumulants(cfg, out):
    """kappa_3..kappa_6 of the messages against t and site, from the `shapes` rows."""
    import collections

    src = Path(out) / "shapes.csv"
    if not src.exists():
        raise SystemExit("run --only shapes first (part `cumulants` reads shapes.csv)")
    import csv

    rows = list(csv.DictReader(open(src)))
    g = collections.defaultdict(list)
    for r in rows:
        if r["ansatz"] != "gaussian":                     # one row per (chain, site), not three
            continue
        g[(r["family"], float(r["level"]), int(r["site"]))].append(r)
    out_rows = []
    for (fam, t, site), rs in sorted(g.items()):
        rec = {"family": fam, "level": t, "site": site, "n_chains": len(rs)}
        for k in ("k3", "k4", "k5", "k6", "var"):
            v = np.array([float(r[k]) for r in rs])
            rec[f"{k}_median"] = float(np.median(v))
            rec[f"{k}_absmedian"] = float(np.median(np.abs(v)))
        out_rows.append(rec)
        if site == max(int(r["site"]) for r in rows):
            print(f"  [E22 cumulants] {fam:9s} t={t:<7.4f} site={site:3d} "
                  f"|k3|={rec['k3_absmedian']:.3e} |k4|={rec['k4_absmedian']:.3e} "
                  f"|k5|={rec['k5_absmedian']:.3e} |k6|={rec['k6_absmedian']:.3e}", flush=True)
    write_csv(Path(out) / "cumulants.csv", out_rows)
    return out_rows


PARTS = {"shapes": part_shapes, "cumulants": part_cumulants}


def main():
    ap = experiment_parser("e22_message_shapes",
                           "what the BP messages look like, and whether a parametric ansatz carries them")
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
