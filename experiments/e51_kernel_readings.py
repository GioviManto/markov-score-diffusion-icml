"""E51: the temperature and the potential read off every learned field kernel, the MDN included.

Deterministic, no fitting. Every risk-selected E33 fit of the site energy, the spline and the MDN is rebuilt from
its `theta` column and read through `kernel_readings` (curvature of log K -> gamma -> Theta; diagonal of K -> site
energy -> V). For the two gradient-class arms the readings from their own parameters are written beside, as the
identity check. Predictions: experiments/registry/E51.json. Reading: tools/core/read_e51.py.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e33_field_learning as e33  # noqa: E402
from common import experiment_parser, provenance  # noqa: E402
from e34_field_physics import kernel_from_row  # noqa: E402
from markov_score.bp_grid import make_grid  # noqa: E402
from markov_score.kernel_readings import readings  # noqa: E402
from markov_score.pair_energy_kernel import potential_readings  # noqa: E402
from markov_score.utils import ensure_dir, write_csv, write_json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {"em_pair_quartic": "outputs/hpc_E33q_s*/kernels.csv", "em_tilted_spline": "outputs/hpc_E33k_s*/kernels.csv",
           "em_mdn": "outputs/hpc_E33k_s*/kernels.csv"}


def _rows():
    frames = []
    for arm, pattern in SOURCES.items():
        df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(ROOT / pattern)))])
        frames.append(df[(df.arm == arm) & (df.selection == "risk")])
    return pd.concat(frames, ignore_index=True)


def main():
    ap = experiment_parser("e51_kernel_readings", "temperature and potential read off every learned field kernel")
    args = ap.parse_args()
    out = ensure_dir(Path(args.output_dir))
    cfg = dict(e33.SETTINGS)
    grid, w = make_grid(cfg["half_width"], cfg["grid_size"])
    df = _rows()
    if args.quick:
        df = df.groupby("arm").head(2)
    rows = []
    for _, r in df.iterrows():
        base = dict(cell=r["cell"], T=float(r["T"]), h=float(r["h"]), scale=float(r["scale"]), arm=r["arm"],
                    n_chains=int(r["n_chains"]), seed=int(r["seed"]), risk=float(r["risk"]), at_cap=int(r["at_cap"]))
        try:
            k = kernel_from_row(r, cfg, grid)
            with np.errstate(invalid="ignore", divide="ignore"):
                rd = readings(k.log_transition_matrix(grid), grid, w, base["scale"], base["h"])
            row = {**base, "failed": 0, "theta_ratio": rd["theta_hat"] / base["T"],
                   **{k_: rd[k_] for k_ in ("barrier_hat", "well_hat", "potential_rms", "curvature_cv")}}
            if r["arm"] != "em_mdn":
                own = potential_readings(k.site_energy(grid), grid, k.gamma, base["scale"], base["h"])
                row.update(own_theta_ratio=own["theta_hat"] / base["T"], own_barrier=own["barrier_hat"])
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            row = {**base, "failed": 1, "error": repr(exc)}
        rows.append(row)
    write_csv(out / "readings.csv", rows)
    write_json(out / "params_readings.json", {**provenance(), "settings": {"sources": SOURCES, "grid_size": cfg["grid_size"],
                                                                          "half_width": cfg["half_width"]},
                                              "quick": bool(args.quick), "n_fits": len(rows)})
    print(f"  [E51] {len(rows)} fits read, {sum(r['failed'] for r in rows)} failed")


if __name__ == "__main__":
    main()
