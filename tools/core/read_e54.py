"""Read E54 against its registered predictions (experiments/registry/E54.json), paired with E33 by (cell, seed, M).

    python tools/core/read_e54.py            # verdicts + the table of seed means (risk / linear floor)
    python tools/core/read_e54.py --tex      # also writes reports/core/tab-field.tex
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS = range(8)
SIZES = (32, 128, 512, 2048)
CELLS = ("T0.3_h1_L32", "T0.3_h0.5_L64", "T0.3_h0.25_L128", "T0.5_iso_L32", "T1_iso_L32", "T2_iso_L32")
LABEL = {"T0.3_h1_L32": r"$\Theta=0.3$, $h=1$", "T0.3_h0.5_L64": r"$\Theta=0.3$, $h=\tfrac12$",
         "T0.3_h0.25_L128": r"$\Theta=0.3$, $h=\tfrac14$", "T0.5_iso_L32": r"$\Theta=0.5$", "T1_iso_L32": r"$\Theta=1$",
         "T2_iso_L32": r"$\Theta=2$"}


def _csvs(pattern):
    return pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(ROOT / pattern)))], ignore_index=True)


def load():
    cnn = pd.read_csv(ROOT / "outputs/E54/nets.csv").assign(arm="cnn")
    k = pd.concat([_csvs("outputs/hpc_E33k_s*/kernels.csv"), _csvs("outputs/hpc_E33q_s*/kernels.csv")], ignore_index=True)
    k = k[(k.selection == "risk") & k.arm.isin(["em_pair_quartic", "em_mdn"])]
    n = _csvs("outputs/hpc_E33n_s*/nets.csv")
    n = n[(n.arm == "dsm_one_corruption") & (n.arch == "window")].assign(arm="cnn_one_copy")
    cols = ["cell", "arm", "n_chains", "seed", "risk", "test_bundle", "at_cap"]
    df = pd.concat([cnn[cols], k[cols], n[cols]], ignore_index=True)
    df = df[df.seed.isin(SEEDS)]
    # pairing: every arm of a (cell, seed, M) must be scored on the same test bundle
    hashes = df.groupby(["cell", "seed", "n_chains"]).test_bundle.nunique()
    assert (hashes == 1).all(), f"unpaired bundles: {hashes[hashes > 1]}"
    floor = pd.read_csv(ROOT / "outputs/hpc_E33o/oracles.csv").query("arm == 'lmmse_true'").groupby("cell").risk.mean()
    df["ratio"] = df.risk / df.cell.map(floor)
    return df


def wins(df, a, b, cell, m):
    """Seeds in which arm a has lower risk than arm b (a missing or NaN row of a counts as a loss)."""
    x = df[(df.cell == cell) & (df.n_chains == m)].pivot_table(index="seed", columns="arm", values="risk")
    if a not in x or b not in x:
        return 0, 0
    ok = (x[a] < x[b]).fillna(False)
    return int(ok.sum()), len(SEEDS)


def main(argv) -> int:
    df = load()
    mean = df.groupby(["arm", "cell", "n_chains"]).ratio.mean()
    cap = df[df.arm == "cnn"].groupby(["cell", "n_chains"]).at_cap.mean()
    n_cnn = df[df.arm == "cnn"].groupby(["cell", "n_chains"]).seed.nunique()
    pd.set_option("display.width", 200)
    print("seed-mean risk / linear floor (seeds 0-7):")
    print(mean.unstack("cell")[list(CELLS)].round(3).to_string())
    print("\nCNN at-cap share:\n" + cap.unstack().round(2).to_string())
    print("\nCNN fits present:\n" + n_cnn.unstack().to_string())
    v = {}
    p1 = [(c, m, *wins(df, "cnn", "cnn_one_copy", c, m)) for c in CELLS for m in (128, 2048)]
    v["P1"] = (all(w >= 7 for *_, w, _ in p1), "; ".join(f"{c} M={m}: {w}/{n}" for c, m, w, n in p1))
    p2 = [(c, m, *wins(df, "em_pair_quartic", "cnn", c, m)) for c in CELLS for m in SIZES]
    v["P2"] = (all(w >= 7 for *_, w, _ in p2), f"{sum(w >= 7 for *_, w, _ in p2)}/24 pairs with >= 7/8 wins; min {min(w for *_, w, _ in p2)}/8")
    p3 = [c for c in CELLS if mean.get(("em_pair_quartic", c, 32), np.inf) < mean.get(("cnn", c, 2048), -np.inf)]
    v["P3"] = (len(p3) >= 4, f"{len(p3)}/6 cells: site energy at M=32 below the CNN at M=2048 ({', '.join(p3)})")
    p4 = [c for c in CELLS if mean.get(("em_mdn", c, 2048), np.inf) < mean.get(("cnn", c, 2048), -np.inf)]
    v["P4"] = (len(p4) >= 5, f"{len(p4)}/6 cells: MDN below the CNN at M=2048")
    p5 = [c for c in CELLS if mean.get(("cnn", c, 32), np.inf) < mean.get(("em_mdn", c, 32), -np.inf)]
    v["P5"] = (len(p5) >= 3, f"{len(p5)}/6 cells: CNN below the MDN at M=32 ({', '.join(p5)})")
    p6 = [c for c in CELLS if all(mean.get(("cnn", c, a), np.nan) > mean.get(("cnn", c, b), np.nan) for a, b in zip(SIZES, SIZES[1:]))]
    v["P6"] = (len(p6) == 6, f"{len(p6)}/6 cells strictly decreasing in M")
    print()
    for k_, (ok, msg) in v.items():
        print(f"{k_}: {'CONFIRMED' if ok else 'FALSIFIED'}  {msg}")
    if "--tex" in argv:
        rows = []
        for c in CELLS:
            cells = [f"{mean[(a, c, m)]:.2f}" for m in (32, 2048) for a in ("em_pair_quartic", "em_mdn", "cnn")]
            rows.append(LABEL[c] + " & " + " & ".join(cells) + r" \\")
        tex = "\n".join([
            r"\begin{table}[h]", r"\centering\small",
            r"\caption{Score error divided by the linear floor on the six field cells, mean of the 8 seeds shared by all "
            r"three estimators. Site: the five-parameter site-energy kernel; neural: the neural kernel inside BP; CNN: "
            r"the network trained by denoising score matching.}", r"\label{tab:field}",
            r"\begin{tabular}{lcccccc}", r"\toprule",
            r" & \multicolumn{3}{c}{$M=32$} & \multicolumn{3}{c}{$M=2048$} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
            r"cell & site & neural & CNN & site & neural & CNN \\", r"\midrule",
            *rows, r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
        (ROOT / "reports/core/tab-field.tex").write_text(tex)
        print("wrote reports/core/tab-field.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
