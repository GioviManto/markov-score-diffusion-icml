"""E45: inpainting with posterior uncertainty. What do E33's fitted kernels do when part of a noisy sequence is missing?

The consolidated note's section 9.7 asks for posterior sampling (FFBS, markov_score.ffbs) and the missing-data case was "cheap, not
asked". It is also the clearest use of an exact-BP denoiser that a network cannot match: a missing site is simply a site with no
likelihood, and the forward-filter backward-sample pass returns whole posterior sequences, so the answer comes with calibrated
uncertainty on where a kink sits inside a gap.

No refit. The kernels are E33's FROZEN fits at M = 2048 (risk-selected checkpoints, seeds 0-15), rebuilt from the `theta` column by
e34_field_physics.kernel_from_row. Test sequences are fresh exact draws from the true phi^4 chain, noised at t = 0.5, with a pattern
of sites hidden (a per-site variance of 1e8 makes the likelihood flat: changing a hidden observation changes the posterior by
exactly 0). Arms: `oracle` (the true kernel), `em_pair_quartic`, `em_tilted_spline`, `em_mixture` (the fits), and `linear` (the Gaussian
conditional with the TRUE covariance: the Kalman smoother with missing data).

Per (cell, seed, arm, pattern): the squared error of the posterior mean at the hidden sites relative to the energy there, and the
coverage of the 90% central posterior interval (FFBS: 200 samples with jitter; linear: mean +- 1.645 sd) at the hidden sites.

Predictions (experiments/registry/E45.json):
  P1  the oracle's 90% intervals cover the truth with frequency in [0.87, 0.93] on every (cell, pattern), pooled over seeds
  P2  the site energy: coverage within 0.03 of 0.90 on every (cell, pattern), and its hidden-site error within 10% of the oracle's in >= 14/16 seeds
  P3  the linear conditional is miscalibrated where a gap spans a kink: |coverage - 0.90| > 0.03 on (T0.3_h1_L32, gap16)
  P4  the posterior MEAN is not where the nonlinear model wins: on (T0.3_h1_L32, gap16) the linear error / oracle error (seed means) is in [0.9, 1.1]
  P5  the UNCERTAINTY is: on (T0.3_h1_L32, gap16) the linear 90% interval is at least 10% wider than the oracle's (seed means)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, run_cells, select_parts  # noqa: E402
from e32_field_oracles import CELLS, resolve_cell  # noqa: E402
from e34_field_physics import kernel_from_row  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.ffbs import grid_ffbs  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.stationary import invariant_log_density_eig  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MISSING_VARIANCE = 1e8
Z90 = 1.6448536269514722

SETTINGS = dict(cells=("T0.3_h1_L32", "T1_iso_L32"), seeds=tuple(range(16)), level=0.5, n_test=256, n_samples=200,
                patterns=("random50", "gap8", "gap16"), fit_size=2048,
                e33_k="outputs/hpc_E33k_s*/kernels.csv", e33_q="outputs/hpc_E33q_s*/kernels.csv",
                grid_size=FROZEN.n_grid, half_width=FROZEN.half_width, components=8)
QUICK = dict(cells=("T0.3_h1_L32",), seeds=(0,), n_test=16, n_samples=20, patterns=("gap8",), grid_size=201)


def hidden_mask(pattern, L, rng):
    m = np.zeros(L, bool)
    if pattern == "random50":
        m[rng.choice(L, size=L // 2, replace=False)] = True
    elif pattern == "gap8":
        m[L // 2 - 4: L // 2 + 4] = True
    elif pattern == "gap16":
        m[L // 2 - 8: L // 2 + 8] = True
    else:
        raise ValueError(pattern)
    return m


def linear_conditional(sigma, X, alpha, delta, hidden):
    """Mean and variance of a_hidden given the visible noisy sites, Gaussian with covariance sigma (zero mean)."""
    o, u = np.where(~hidden)[0], np.where(hidden)[0]
    Soo, Suo, Suu = sigma[np.ix_(o, o)], sigma[np.ix_(u, o)], sigma[np.ix_(u, u)]
    Kinv = np.linalg.inv(alpha**2 * Soo + delta * np.eye(len(o)))
    mean = np.zeros((X.shape[0], len(u)))
    mean = alpha * (X[:, o] @ Kinv) @ Suo.T
    var = np.diag(Suu - alpha**2 * Suo @ Kinv @ Suo.T)
    return mean, var


def _load_rows(spec, arm, n):
    paths = sorted(ROOT.glob(spec))
    d = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True) if paths else pd.DataFrame()
    return d[(d.arm == arm) & (d.n_chains == n) & (d.selection == "risk")] if not d.empty else d


def _cell(args):
    cfg, cell_id, seed = args
    cell = next(c for c in CELLS if c["id"] == cell_id)
    chain, info = resolve_cell(cell)
    L = cell["L"]
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    al, de = alpha_delta(cfg["level"])
    rng = rng_for("e45-test", seed, cell_id)
    A = chain.sample_batch(rng, cfg["n_test"], L)
    X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
    sigma = chain.covariance(L)
    kernels = {"oracle": (chain.log_transition_matrix(grid), chain.log_stationary(grid))}
    for arm, spec in (("em_pair_quartic", cfg["e33_q"]), ("em_tilted_spline", cfg["e33_k"]), ("em_mixture", cfg["e33_k"])):
        d = _load_rows(spec, arm, cfg["fit_size"])
        r = d[(d.cell == cell_id) & (d.seed == seed)] if not d.empty else d
        failed = r.iloc[0].get("failed") if len(r) == 1 else np.nan                  # old cells carry no `failed` column: NaN means not failed
        if len(r) != 1 or (not pd.isna(failed) and int(failed) == 1) or not isinstance(r.iloc[0].theta, str):
            kernels[arm] = None
            continue
        k = kernel_from_row(r.iloc[0], cfg, grid)
        kernels[arm] = (k.log_transition_matrix(grid), invariant_log_density_eig(k, grid, w).log_density)
    base = dict(cell=cell_id, seed=seed, level=cfg["level"], n_test=cfg["n_test"], T=info["T"], h=info["h"])
    rows = []
    for pattern in cfg["patterns"]:
        hidden = hidden_mask(pattern, L, rng_for("e45-mask", pattern, seed, cell_id))
        delta_site = np.where(hidden, MISSING_VARIANCE, de)[:, None]
        energy = float((A[:, hidden] ** 2).sum())
        for arm, kern in kernels.items():
            if kern is None:
                rows.append(dict(base, pattern=pattern, arm=arm, failed=1, rel_err=np.nan, coverage90=np.nan, width90=np.nan))
                continue
            lk, lmu = kern
            mean = grid_bp_batch(grid, w, lk, X, al, delta_site, log_mu=lmu)[0]
            S = grid_ffbs(grid, w, lk, X, al, delta_site, rng_for("e45-ffbs", arm, pattern, seed, cell_id), n_samples=cfg["n_samples"],
                          log_mu=lmu, jitter=True)
            lo, hi = np.quantile(S[:, :, hidden], [0.05, 0.95], axis=0)
            truth = A[:, hidden]
            rows.append(dict(base, pattern=pattern, arm=arm, failed=0, rel_err=float(((mean[:, hidden] - truth) ** 2).sum() / energy),
                             coverage90=float(((truth >= lo) & (truth <= hi)).mean()), width90=float((hi - lo).mean())))
        mean_u, var_u = linear_conditional(sigma, X, al, de, hidden)
        sd = np.sqrt(var_u)[None, :]
        truth = A[:, hidden]
        rows.append(dict(base, pattern=pattern, arm="linear", failed=0, rel_err=float(((mean_u - truth) ** 2).sum() / energy),
                         coverage90=float(((truth >= mean_u - Z90 * sd) & (truth <= mean_u + Z90 * sd)).mean()), width90=float(2 * Z90 * sd.mean())))
    print(f"  [E45] {cell_id} seed={seed}: " + " ".join(f"{r['pattern']}/{r['arm']}={r['rel_err']:.3f}" for r in rows if r["pattern"] == cfg["patterns"][-1]), flush=True)
    return rows


def part_inpainting(cfg, out, workers):
    cells = [(cfg, c, s) for c in cfg["cells"] for s in cfg["seeds"]]
    print(f"  inpainting: {len(cells)} cells", flush=True)
    write_csv(out / "inpainting.csv", run_cells(cells, _cell, workers, out, "inpainting.csv", key=lambda c: f"{c[1]}_s{c[2]}"))


PARTS = {"inpainting": part_inpainting}


def main():
    ap = experiment_parser("e45_inpainting", "missing sites, posterior sampling and calibrated uncertainty with E33's frozen fits")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.list_parts:
        print("\n".join(PARTS)); return
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    out = ensure_dir(Path(args.output_dir))
    for name, fn in select_parts(PARTS, args.only).items():
        fn(cfg, out, args.workers)
        write_json(out / f"params_{name}.json", {**provenance(), "settings": cfg, "quick": bool(args.quick), "workers": args.workers})


if __name__ == "__main__":
    main()
