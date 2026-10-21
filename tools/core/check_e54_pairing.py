"""Are E54's test sets (built on the laptop) the same data as E33's (built on the cluster)?

The bundle hashes differ across platforms because the bytes of the noisy data and of the reference means pass through
eigensolves whose last bits depend on the BLAS library. The question that matters is whether the draws are the same.
For every (cell, seed) this rebuilds the test bundle exactly as E54 did (e33._common) and recomputes two deterministic
risks, the linear floor and the isotropic reference, which E33 recorded on the cluster (hpc_E33o/oracles.csv). Same
draws give the same risks to round-off. Writes reports/core/e54_pairing.csv; exit 1 if any relative deviation > 1e-12.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
import e14_regime_baselines as e14  # noqa: E402
import e33_field_learning as e33  # noqa: E402
from markov_score import theory  # noqa: E402


def main() -> int:
    o = pd.read_csv(ROOT / "outputs/hpc_E33o/oracles.csv").set_index(["cell", "seed", "arm"]).risk
    rows = []
    for cell in e33.SETTINGS["cells"]:
        for seed in range(8):
            *_, sigma0, A, Av, val, test, base = e33._common(dict(e33.SETTINGS), cell, seed, 32)
            lm = e14._pooled(lambda X, t: theory.lmmse_posterior_mean(X, sigma0, t), test)
            iso = e14._pooled(lambda X, t: (X - (1 - np.exp(-2 * t)) * X) / np.exp(-t), test)
            rows.append(dict(cell=cell, seed=seed, laptop_hash=base["test_bundle"],
                             lmmse_rel_dev=abs(lm / o[(cell, seed, "lmmse_true")] - 1),
                             iso_rel_dev=abs(iso / o[(cell, seed, "isotropic")] - 1)))
            print(f"  {cell} seed={seed}: lmmse {rows[-1]['lmmse_rel_dev']:.1e} iso {rows[-1]['iso_rel_dev']:.1e}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "reports/core/e54_pairing.csv", index=False)
    worst = float(df[["lmmse_rel_dev", "iso_rel_dev"]].max().max())
    print(f"{len(df)} (cell, seed) pairs, worst relative deviation {worst:.1e}")
    return 0 if worst < 1e-12 else 1


if __name__ == "__main__":
    raise SystemExit(main())
