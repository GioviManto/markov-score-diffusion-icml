"""Read E52 against its registered predictions (experiments/registry/E52.json)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main(argv=()) -> int:
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
    # P3 against E32's FROZEN fixed-h correlation lengths (read from the CSV; the first reading used a typed list
    # that was shifted by one temperature, see E52.json outcome)
    e32 = pd.read_csv(ROOT / "outputs/E32/fixed_h.csv").query("h == 1.0").groupby("T").corr_length_sites.first()
    ours = df[(df.part == "quartic") & (df.n == 2401) & df.label.str.endswith("_h1") & df.label.str.startswith("T")].set_index("T").xi_sites
    both = pd.concat([e32.rename("e32"), ours.rename("e52")], axis=1).dropna()
    dev = (both.e52 / both.e32 - 1).abs()
    v["P3"] = (len(both) == 7 and bool((dev < 1e-6).all()), f"{int((dev < 1e-6).sum())}/{len(both)} equal E32's frozen values; max rel dev {dev.max():.1e}")
    hw = df[df.part == "harmonic_well"].set_index("T").log_lambda0
    qa = df[(df.part == "quartic") & (df.n == 2401) & df.label.str.endswith("_h1") & df.label.str.startswith("T")].set_index("T").log_lambda0
    delta = (qa - hw).sort_index()
    mono = all(abs(delta.iloc[i]) < abs(delta.iloc[i + 1]) for i in range(len(delta) - 1))
    v["P4"] = (mono and abs(delta.loc[0.2]) < abs(delta.loc[2.0]) / 5,
               "Delta by Theta: " + ", ".join(f"{t:g}: {x:+.4f}" for t, x in delta.items()))
    print("quartic, n = 2401 (log lambda0 per site, xi in sites):")
    print(df[(df.part == "quartic") & (df.n == 2401)][["label", "T", "h", "log_lambda0", "xi_sites"]].to_string(index=False))
    if "--tex" in argv:
        ax = df[(df.part == "quartic") & (df.n == 2401) & df.label.str.endswith("_h1") & df.label.str.startswith("T")].sort_values("T")
        fmt = lambda x: f"${x:.3f}$" if x < 0 else f"{x:.3f}"  # noqa: E731
        tex = "\n".join([r"\begin{table}[h]", r"\centering\small",
                          r"\caption{Free energy per site and correlation length of the lattice field, $h=1$.}", r"\label{tab:logz}",
                          r"\begin{tabular}{l" + "c" * len(ax) + "}", r"\toprule",
                          r"$\Theta$ & " + " & ".join(f"{t:g}" for t in ax["T"]) + r" \\", r"\midrule",
                          r"$\log\lambda_0$ & " + " & ".join(fmt(x) for x in ax.log_lambda0) + r" \\",
                          r"$\xi$ (sites) & " + " & ".join(f"{x:.2f}" for x in ax.xi_sites) + r" \\",
                          r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
        (ROOT / "reports/core/tab-logz.tex").write_text(tex)
        print("wrote reports/core/tab-logz.tex")
    print()
    for k, (ok, msg) in v.items():
        print(f"{k}: {'CONFIRMED' if ok else 'FALSIFIED'}  {msg}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
