"""Read E55 against its registered predictions (experiments/registry/E55.json).

    python tools/core/read_e55.py            # pairing gates, tables, verdicts P1-P10, descriptive D1

Arms: unet, window (E55 nets), em_energy (E55 kernels), site_energy (E33's frozen em_pair_quartic on its six cells at
M <= 2048, E55's pair part elsewhere), mdn (E33's em_mdn) and cnn_e54 (E54's window CNN) as references. Kernel rows are
the risk-selected ones. Seeds 0-7. A missing or failed fit counts AGAINST every prediction it touches: a seed mean needs
all eight seeds finite, a paired win needs the predicted winner's row. Ratios are risk / the cell's linear floor (the
seed mean of E55's own lmmse_true rows).

The verdict functions take one long table (arm, cell, n_chains, seed, risk, at_cap) and the floor, so
tests/test_read_e55.py drives them with made-up tables in both directions.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(8))
E33_CELLS = ("T0.3_h1_L32", "T0.3_h0.5_L64", "T0.3_h0.25_L128", "T0.5_iso_L32", "T1_iso_L32", "T2_iso_L32")
BASE, L128, L512 = "T0.3_h1_L32", "T0.3_h1_L128", "T0.3_h1_L512"
SIZES_A = (32, 128, 512, 2048, 8192)
# matched numbers of transitions M (L - 1) on the base field: ~16k, ~64k, ~255k
TRIPLETS = (((BASE, 512), (L128, 128), (L512, 32)), ((BASE, 2048), (L128, 512), (L512, 128)),
            ((BASE, 8192), (L128, 2048), (L512, 512)))
COLS = ["arm", "cell", "n_chains", "seed", "risk", "at_cap"]


# ------------------------------------------------------------------ loading and gates


def _csvs(pattern):
    paths = sorted(glob.glob(str(ROOT / pattern)))
    if not paths:
        raise SystemExit(f"no files match {pattern}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def load():
    nets = _csvs("outputs/hpc_E55_nets_*/nets.csv")
    energy = _csvs("outputs/hpc_E55_energy_*/kernels.csv")
    pair = _csvs("outputs/hpc_E55_pair_*/pair.csv")
    orc = _csvs("outputs/hpc_E55_oracles/oracles.csv")
    # gate 1: every E55 part saw the same draws for a (cell, seed) -- CPU-built oracles and GPU-built fits alike
    parts = pd.concat([d[["cell", "seed", "test_draws"]] for d in (nets, energy, pair, orc)], ignore_index=True)
    n_draws = parts.groupby(["cell", "seed"]).test_draws.nunique()
    assert (n_draws == 1).all(), f"E55 parts disagree on the test draws: {n_draws[n_draws > 1]}"
    # gate 2: E55's CPU oracles reproduce E33's, so E55's test sets ARE E33's (tools/core/check_e54_pairing.py's test)
    o33 = pd.read_csv(ROOT / "outputs/hpc_E33o/oracles.csv").set_index(["cell", "seed", "arm"]).risk
    o55 = orc[orc.cell.isin(E33_CELLS) & orc.seed.isin(SEEDS)].set_index(["cell", "seed", "arm"]).risk
    dev = max(abs(o55[k] / o33[k] - 1) for k in o55.index)
    assert len(o55) == 2 * len(E33_CELLS) * len(SEEDS) and dev < 1e-12, f"E55 and E33 test sets differ ({dev:.1e})"
    # gate 3: the parity probe of the timing part (reference means on the GPU against the CPU, on the cluster)
    par = _csvs("outputs/hpc_E55_timing/timing.csv").query("what == 'parity'")
    assert len(par) and (par.draws_equal == 1).all() and float(par.risk.max()) < 1e-10, "GPU/CPU parity probe failed"
    # gate 4: E54's laptop CNN rows are E33's draws (reports/core/e54_pairing.csv, written before E54 was read)
    p54 = pd.read_csv(ROOT / "reports/core/e54_pairing.csv")
    assert float(p54[["lmmse_rel_dev", "iso_rel_dev"]].max().max()) < 1e-12, "E54 pairing gate"

    k = energy[energy.selection == "risk"].assign(arm="em_energy")
    pr = pair[pair.selection == "risk"].assign(arm="site_energy")
    q33 = _csvs("outputs/hpc_E33q_s*/kernels.csv")
    q33 = q33[(q33.selection == "risk") & (q33.arm == "em_pair_quartic")].assign(arm="site_energy")
    m33 = _csvs("outputs/hpc_E33k_s*/kernels.csv")
    m33 = m33[(m33.selection == "risk") & (m33.arm == "em_mdn")].assign(arm="mdn")
    c54 = pd.read_csv(ROOT / "outputs/E54/nets.csv").assign(arm="cnn_e54")
    overlap = set(map(tuple, pr[["cell", "n_chains"]].values)) & set(map(tuple, q33[["cell", "n_chains"]].values))
    assert not overlap, f"E55 refitted a frozen E33 site-energy cell: {sorted(overlap)[:3]}"
    df = pd.concat([d[COLS] for d in (nets, k, pr, q33, m33, c54)], ignore_index=True)
    df = df[df.seed.isin(SEEDS)]
    dup = df.duplicated(["arm", "cell", "n_chains", "seed"])
    assert not dup.any(), f"duplicate rows: {df[dup].head()}"
    floor = orc[orc.arm == "lmmse_true"].groupby("cell").risk.mean()
    return df, floor


# ------------------------------------------------------------------ the verdict arithmetic


def seed_mean(df, arm, cell, m, floor=None):
    """Seed mean of risk (or risk / floor); NaN unless all eight seeds have a finite row."""
    x = df[(df.arm == arm) & (df.cell == cell) & (df.n_chains == m)].set_index("seed").risk.reindex(SEEDS)
    if x.isna().any() or not np.isfinite(x).all():
        return np.nan
    v = float(x.mean())
    return v / float(floor[cell]) if floor is not None else v


def wins(df, a, b, cell, m):
    """Seeds in which arm a has a lower risk than arm b; a missing/NaN row of a is a loss, of b a win for a."""
    x = df[(df.cell == cell) & (df.n_chains == m) & df.arm.isin([a, b])].pivot_table(
        index="seed", columns="arm", values="risk", aggfunc="first").reindex(index=list(SEEDS))
    ra = x[a] if a in x else pd.Series(np.nan, index=x.index)
    rb = x[b] if b in x else pd.Series(np.nan, index=x.index)
    return int(((ra < rb) | (ra.notna() & rb.isna())).sum())


def _lt(a, b):
    return bool(np.isfinite(a) and np.isfinite(b) and a < b)


def at_cap_share(df, arm, cells, sizes):
    """Share of fits at the last checkpoint; a missing fit counts as at the cap (it cannot show it stopped early)."""
    n = hits = 0
    for c in cells:
        for m in sizes:
            x = df[(df.arm == arm) & (df.cell == c) & (df.n_chains == m)].set_index("seed").at_cap.reindex(SEEDS)
            n += len(SEEDS); hits += int(x.fillna(1).astype(float).sum())
    return hits / n


def verdicts(df, floor):
    v = {}
    r = lambda arm, c, m: seed_mean(df, arm, c, m, floor)  # noqa: E731
    p1 = [c for c in E33_CELLS if _lt(r("em_energy", c, 2048), r("unet", c, 2048)) and wins(df, "em_energy", "unet", c, 2048) >= 6]
    v["P1"] = (len(p1) >= 4, f"{len(p1)}/6 cells: energy kernel below the U-Net at M=2048 (seed mean, >= 6/8 paired): {p1}")
    p2 = [c for c in E33_CELLS if _lt(r("em_energy", c, 2048), r("unet", c, 8192))]
    v["P2"] = (len(p2) >= 3, f"{len(p2)}/6 cells: energy kernel at M=2048 below the U-Net at M=8192: {p2}")
    p3 = [(c, m, wins(df, "site_energy", "unet", c, m)) for c in E33_CELLS for m in SIZES_A]
    v["P3"] = (all(w >= 7 for *_, w in p3), f"{sum(w >= 7 for *_, w in p3)}/30 (cell, M) with >= 7/8 site-energy wins; "
               f"min {min(w for *_, w in p3)}/8")
    p4 = [c for c in E33_CELLS if _lt(r("site_energy", c, 32), r("unet", c, 2048))]
    v["P4"] = (len(p4) >= 4, f"{len(p4)}/6 cells: site energy at M=32 below the U-Net at M=2048: {p4}")
    p5 = [c for c in E33_CELLS if _lt(r("unet", c, 32), r("em_energy", c, 32))]
    v["P5"] = (len(p5) >= 3, f"{len(p5)}/6 cells: U-Net below the energy kernel at M=32: {p5}")
    cu = at_cap_share(df, "unet", E33_CELLS, (512, 2048, 8192))
    cw = at_cap_share(df, "window", E33_CELLS, (512, 2048, 8192))
    v["P6"] = (cu < 0.25 and cw < 0.25, f"last-checkpoint share at M >= 512: U-Net {cu:.2f}, window {cw:.2f} (each < 0.25)")
    p7 = [c for c in E33_CELLS if _lt(r("window", c, 2048), r("cnn_e54", c, 2048))]
    v["P7"] = (len(p7) >= 5, f"{len(p7)}/6 cells: window CNN (E55 recipe) below E54's at M=2048: {p7}")
    p8 = [c for c in E33_CELLS if _lt(r("unet", c, 2048), r("window", c, 2048))]
    v["P8"] = (len(p8) >= 4, f"{len(p8)}/6 cells: U-Net below the window CNN at M=2048: {p8}")
    f9 = seed_mean(df, "site_energy", BASE, 2048) / seed_mean(df, "site_energy", BASE, 32768)
    v["P9a"] = (bool(np.isfinite(f9) and f9 >= 2.5), f"site energy, base cell: risk(M=2048)/risk(M=32768) = {f9:.2f} (>= 2.5)")
    w9 = wins(df, "em_energy", "unet", BASE, 32768)
    v["P9b"] = (w9 >= 6, f"base cell M=32768: energy kernel below the U-Net in {w9}/8 seeds (>= 6)")
    for key, arm in (("P10a", "site_energy"), ("P10b", "em_energy")):
        spreads = [_spread(df, arm, t) for t in TRIPLETS]
        ok = sum(bool(np.isfinite(s) and s < 1.5) for s in spreads)
        v[key] = (ok >= 2, f"{arm}: max/min seed-mean risk within the matched-transition triplets "
                  + ", ".join(f"{s:.2f}" for s in spreads) + f"; {ok}/3 below 1.5 (>= 2)")
    return v


def _spread(df, arm, triplet):
    x = [seed_mean(df, arm, c, m) for c, m in triplet]
    return float(max(x) / min(x)) if all(np.isfinite(x)) else np.nan


def descriptive(df):
    return {"D1": "U-Net: max/min seed-mean risk within the matched-transition triplets "
            + ", ".join(f"{_spread(df, 'unet', t):.2f}" for t in TRIPLETS)}


def main(argv) -> int:
    df, floor = load()
    df = df.assign(ratio=df.risk / df.cell.map(floor))
    pd.set_option("display.width", 220)
    mean = df.groupby(["arm", "cell", "n_chains"]).ratio.mean()
    print("seed-mean risk / linear floor (seeds 0-7):")
    print(mean.unstack("cell").round(3).to_string())
    for arm in ("unet", "window", "em_energy"):
        print(f"\n{arm}: last-checkpoint share\n" + df[df.arm == arm].groupby(["cell", "n_chains"]).at_cap.mean().unstack().round(2).to_string())
        print(f"{arm}: fits present\n" + df[df.arm == arm].groupby(["cell", "n_chains"]).seed.nunique().unstack().to_string())
    print()
    for k, (ok, detail) in verdicts(df, floor).items():
        print(f"{k}: {'CONFIRMED' if ok else 'FALSIFIED'}  {detail}")
    for k, detail in descriptive(df).items():
        print(f"{k}: {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
