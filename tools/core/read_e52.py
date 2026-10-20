"""Read E52 against its registered predictions (experiments/registry/E52.json)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
E32_XI = {0.2: 19.47, 0.3: 6.08, 0.5: 3.66, 0.7: 2.75, 1.0: 1.98, 1.5: 1.53, 2.0: 1.21}


def main() -> int:
    df = pd.read_csv(ROOT / "outputs/E52/spectrum.csv")
    pd.set_option("display.width", 200)
    g = df[df.part == "gaussian"]
    e_l = np.abs(g.log_lambda0 - g.closed_log_lambda0).max()
    e_r = np.abs(g.ratio - g.closed_ratio).max()
    v = {"P1": (e_l < 1e-10 and e_r < 1e-10, f"max |log lambda0 - closed| = {e_l:.1e}, max |ratio - closed| = {e_r:.1e}")}
    q = df[df.part == "quartic"].pivot_table(index=["label", "T", "h"], columns="n", values=["log_lambda0", "xi_sites"])
    d_l = (q[("log_lambda0", 2401)] - q[("log_lambda0", 4801)]).abs()
    d_x = (q[("xi_sites", 2401)] / q[("xi_sites", 4801)] - 1).abs()
    v["P2"] = (bool((d_l < 1e-10).all() and (d_x < 1e-8).all()), f"max dlog lambda0 {d_l.max():.1e}, max rel dxi {d_x.max():.1e}")
    axis = q.reset_index()
    axis = axis[axis.label.str.endswith("_h1") & axis.label.str.startswith("T")]
    match = [round(float(r[("xi_sites", 2401)]), 2) == E32_XI[round(float(r["T"].iloc[0] if hasattr(r["T"], "iloc") else r["T"]), 2)]
             for _, r in axis.iterrows()]
    v["P3"] = (all(match), f"{sum(match)}/{len(match)} correlation lengths equal E32's to two decimals")
    hw = df[df.part == "harmonic_well"].set_index("T").log_lambda0
    qa = df[(df.part == "quartic") & (df.n == 2401) & df.label.str.endswith("_h1") & df.label.str.startswith("T")].set_index("T").log_lambda0
    delta = (qa - hw).sort_index()
    mono = all(abs(delta.iloc[i]) < abs(delta.iloc[i + 1]) for i in range(len(delta) - 1))
    v["P4"] = (mono and abs(delta.loc[0.2]) < abs(delta.loc[2.0]) / 5,
               "Delta by Theta: " + ", ".join(f"{t:g}: {x:+.4f}" for t, x in delta.items()))
    print("quartic, n = 2401 (log lambda0 per site, xi in sites):")
    print(df[(df.part == "quartic") & (df.n == 2401)][["label", "T", "h", "log_lambda0", "xi_sites"]].to_string(index=False))
    print()
    for k, (ok, msg) in v.items():
        print(f"{k}: {'CONFIRMED' if ok else 'FALSIFIED'}  {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
