#!/usr/bin/env python
"""E19 — the high-noise competitor: method-of-moments estimators on E06's data.

Theorem 2 gives closed-form unbiased moment estimators from one-level noised data:
    rho_hat        = mean_{mu,i} x_i x_{i+1} / alpha^2                       (degree 2; unit variances at every t)
    kappa4(a_i)    = kappa4(eps) (1 - rho^{4(i-1)}) / (1 - rho^4),  k4(x_i) = alpha^4 kappa4(a_i)   (degree 4)
    kappa4_eps_hat = sum_i k4_hat(x_i) / (alpha^4 sum_i c_i(rho_hat)),   excess kurtosis = kappa4_eps_hat / q_hat^2
No optimisation, no grid, no collapse: unbiased up to O(1/M), so the error tracks e^{2t} and e^{4t}
without saturating -- at a constant the Fisher information says is far from efficient. Same chains,
same noise draws and same levels as E06 (rng tags "e06-train", "e06-noise"), so the comparison with
the EM-BP rows of outputs/E06/levels.csv is paired by seed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import apply_overrides, experiment_parser, provenance, select_parts  # noqa: E402
from frozen_config import FROZEN  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.utils import ensure_dir, rng_for, write_csv, write_json  # noqa: E402

N = FROZEN.n_sites
SETTINGS = dict(rho=FROZEN.rho, n_chains=512, seeds=tuple(range(16)), levels=(0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.4, 2.0))
QUICK = dict(seeds=(0, 1), levels=(0.1, 1.0))


def part_moments(cfg, out):
    rho = cfg["rho"]; q = 1 - rho**2; prior = LaplaceAR1(rho); M = cfg["n_chains"]
    i = np.arange(1, N + 1)
    rows = []
    for t in cfg["levels"]:
        al, de = alpha_delta(t)
        for seed in cfg["seeds"]:
            rng = rng_for("e06-train", seed, t)
            A = np.stack([prior.sample(rng, N) for _ in range(M)])
            X = al * A + np.sqrt(de) * rng_for("e06-noise", seed, t).standard_normal(A.shape)
            rho_hat = float(np.mean(X[:, :-1] * X[:, 1:]) / al**2)
            m2 = (X**2).mean(0); m4 = (X**4).mean(0); k4x = m4 - 3 * m2**2
            c = (1 - rho_hat ** (4 * (i - 1))) / (1 - rho_hat**4)
            k4e = float(k4x.sum() / (al**4 * c.sum())); q_hat = 1 - rho_hat**2
            c0 = (1 - rho ** (4 * (i - 1))) / (1 - rho**4)
            rows.append(dict(t_obs=t, seed=seed, n_chains=M, fit_rho=rho_hat, fit_kurt=k4e / q_hat**2,
                             fit_kurt_oracle_rho=float(k4x.sum() / (al**4 * c0.sum())) / q**2, true_rho=rho, true_kurt=3.0))
        r = [x for x in rows if x["t_obs"] == t]
        print(f"  [E19] t={t:<4g} RMSE rho {np.sqrt(np.mean([(x['fit_rho']-rho)**2 for x in r])):.4f}  RMSE kurt {np.sqrt(np.mean([(x['fit_kurt']-3)**2 for x in r])):.3f}", flush=True)
    write_csv(out / "moments.csv", rows)


PARTS = {"moments": part_moments}


def main():
    ap = experiment_parser("e19_moment_estimators", "method-of-moments estimators of rho and shape on E06's noised data")
    args = ap.parse_args()
    if args.list_parts:
        print("\n".join(PARTS)); return
    cfg = dict(SETTINGS)
    if args.quick:
        cfg.update(QUICK)
    cfg = apply_overrides(cfg, args.set)
    out = ensure_dir(Path(args.output_dir))
    for name, fn in select_parts(PARTS, args.only).items():
        fn(cfg, out)
        write_json(out / f"params_{name}.json", {**provenance(), "settings": cfg, "quick": bool(args.quick), "frozen": FROZEN.as_dict()})


if __name__ == "__main__":
    main()
