"""Read E64 against its registered predictions (experiments/registry/E64.json).

    python tools/core/read_e64.py          # gates, table, verdicts P1-P4

One row per (seed, N, arm) in outputs/hpc_E64/generation.csv: the metrics of 2048 generated sequences against the
reference draw R of that cell. "The best network" at an N is the continuously retrained network (unet_cont, bigru_cont,
window_cont) with the lowest seed-mean SW2 there. P4's "log-likelihood gap" is read in absolute value, |loglik_gap|
(implementation note of 3 Nov, before any run): the gap is reference minus generated and is negative for
over-concentrated samples, so its size, not its sign, is the error.

A FAILED or missing arm counts AGAINST every prediction it enters. The verdict functions take the generation table, so
tests/test_read_e64.py drives them with made-up tables in both directions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(8))
SIZES = (128, 2048)
NETS = ("unet_cont", "bigru_cont", "window_cont")


def load(path="outputs/hpc_E64/generation.csv"):
    d = pd.read_csv(ROOT / path)
    return d[d.failed.fillna(0) == 0].assign(abs_kurt=lambda x: (x.kurt_gen - x.kurt_ref).abs(),
                                            abs_gap=lambda x: x.loglik_gap.abs())


def gates(d):
    k = d[d.frozen_risk.notna()]
    worst = float((k.rebuilt_risk / k.frozen_risk - 1).abs().max()) if len(k) else np.inf
    assert worst <= 1e-6, f"G1: a rebuilt kernel does not score its frozen row (max rel {worst:.1e})"
    assert (d.groupby(["seed", "n_chains"]).ref_hash.nunique() == 1).all(), "G2: arms of a cell saw different references"


def _v(d, arm, n, col):
    return d[(d.arm == arm) & (d.n_chains == n)].set_index("seed")[col].reindex(SEEDS).to_numpy(dtype=float)


def below(d, a, b, n, col):
    xa, xb = _v(d, a, n, col), _v(d, b, n, col)
    return int((np.isfinite(xa) & np.isfinite(xb) & (xa < xb)).sum())


def best_net(d, n):
    m = {a: np.nanmean(_v(d, a, n, "sw2")) if np.isfinite(_v(d, a, n, "sw2")).all() else np.inf for a in NETS}
    return min(m, key=m.get)


def verdicts(d):
    p1 = {n: int((np.isfinite(_v(d, "oracle_sampler", n, "sw2")) & (_v(d, "oracle_sampler", n, "sw2") <= 1.5 * _v(d, "exact", n, "sw2"))).sum())
          for n in SIZES}
    p2a = min(below(d, "clean_energy", net, 128, "sw2") for net in NETS)
    p2b = below(d, "clean_energy", best_net(d, 2048), 2048, "sw2")
    p3 = {n: below(d, "clean_energy", best_net(d, n), n, "abs_kurt") for n in SIZES}
    p4 = {n: min(below(d, "clean_energy", net, n, "abs_gap") for net in NETS) for n in SIZES}
    return {
        "P1": (all(v >= 7 for v in p1.values()), "oracle sampler SW2 <= 1.5 x Monte Carlo floor: " + ", ".join(f"N={n} {v}/8" for n, v in p1.items())),
        "P2": (p2a >= 7 and p2b >= 6, (f"clean_energy SW2 below every network at N=128 (min over nets {p2a}/8, need 7) and below "
                                       f"{best_net(d, 2048)} at N=2048 ({p2b}/8, need 6)")),
        "P3": (all(v >= 6 for v in p3.values()), "|kurtosis error| below the best network: " + ", ".join(f"N={n} {v}/8" for n, v in p3.items())),
        "P4": (all(v >= 6 for v in p4.values()), "|log-likelihood gap| below every network: " + ", ".join(f"N={n} min {v}/8" for n, v in p4.items())),
    }


def main(argv) -> int:
    d = load(*argv[:1])
    gates(d)
    cols = ["sw2", "energy", "w1_marginal", "cov_fro", "abs_kurt", "acf_err_max", "abs_gap"]
    print(d.groupby(["n_chains", "arm"])[cols].mean().to_string(float_format=lambda v: f"{v:.4f}"))
    for k, (ok, msg) in verdicts(d).items():
        print(f"{k}  {'CONFIRMED' if ok else 'FALSIFIED'}: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
