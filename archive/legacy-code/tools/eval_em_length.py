#!/usr/bin/env python
"""Length generalisation of EM-BP: kernels fitted at L = 32 evaluated at longer chains.

The fitted transition kernel is length-independent by construction, so evaluation is all
that is needed: for each E02a fit (seed, M, C = 8) run grid BP with the fitted kernel on
test bundles of length L in (32, 64, 128, 256) drawn from the true chain (same streams as
experiments/e03_baselines.py part `length`, so the network rows join on (seed, M, L)),
and report the all-site pooled risk against true-kernel BP at that length. Writes
outputs/E03/length_em.csv (or --out).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from frozen_config import FROZEN  # noqa: E402
from markov_score.bp_grid import grid_bp_batch, make_grid  # noqa: E402
from markov_score.kernels import MixtureInnovationKernel  # noqa: E402
from markov_score.noising import alpha_delta  # noqa: E402
from markov_score.priors import LaplaceAR1  # noqa: E402
from markov_score.utils import rng_for, write_csv  # noqa: E402

T = tuple(FROZEN.t_grid)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--em", default="outputs/E02a/capacity.csv")
    ap.add_argument("--out", default="outputs/E03/length_em.csv")
    ap.add_argument("--lengths", default="32,64,128,256")
    ap.add_argument("--sizes", default="32,128,512,2048")
    ap.add_argument("--components", type=int, default=8)
    ap.add_argument("--n_test", type=int, default=2048)
    args = ap.parse_args()
    df = pd.read_csv(ROOT / args.em)
    df = df[(df.arm == "em_bp") & (df.components == args.components) & (df.region == "all")]
    sizes = [int(x) for x in args.sizes.split(",")]
    lengths = [int(x) for x in args.lengths.split(",")]
    grid, w = make_grid(FROZEN.half_width, FROZEN.n_grid)
    prior = LaplaceAR1(FROZEN.rho)
    log_k_true = prior.log_transition_matrix(grid)
    rows = []
    for seed in sorted(df.seed.unique()):
        for n in sizes:
            r = df[(df.seed == seed) & (df.n_chains == n)]
            if len(r) == 0:
                continue
            th = np.array([float(v) for v in r.theta.iloc[0].split(";")]); C = args.components
            kb = MixtureInnovationKernel(rho=float(th[0]), pi=th[1:1 + C], mu=th[1 + C:1 + 2 * C], s2=th[1 + 2 * C:1 + 3 * C])
            lk = kb.log_transition_matrix(grid)
            for L in lengths:
                rng = rng_for(("e03-len-test", L), seed)
                n_chains = max(64, args.n_test * FROZEN.n_sites // L)
                A = np.stack([prior.sample(rng, L) for _ in range(n_chains)])
                num = den = 0.0
                for t in T:
                    al, de = alpha_delta(t)
                    X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
                    m_ref, _ = grid_bp_batch(grid, w, log_k_true, X, al, de)
                    m, _ = grid_bp_batch(grid, w, lk, X, al, de)
                    s_ref = -(X - al * m_ref) / de; s = -(X - al * m) / de
                    num += ((s - s_ref) ** 2).sum(); den += (s_ref**2).sum()
                rows.append(dict(seed=seed, n_chains=n, arch="em_bp", L_test=L, risk=float(np.sqrt(num / den)), n_params=int(len(th))))
                print(f"  seed={seed} M={n} L={L}: {rows[-1]['risk']:.4f}", flush=True)
    out = ROOT / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, rows); print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
