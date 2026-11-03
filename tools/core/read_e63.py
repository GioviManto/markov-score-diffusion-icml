"""Read E63 against its registered predictions (experiments/registry/E63.json, amendment of 3 Nov before code).

    python tools/core/read_e63.py                  # gates, D1 table, verdicts P1-P4, writes the wall-clock curves

Inputs: E63's timing (outputs/hpc_E63/timing.csv: seconds per update for every (arm, N), use time per test sequence)
and E65's per-checkpoint curves (outputs/hpc_E65_*/curves.csv: for every (seed, N, arm, checkpoint) the TEST error on
all 32 sites, `test_all`, and the flag `sel_oracle` on the checkpoint each arm's original rule selects). A curve is put
on a wall-clock axis as wall(c) = c x seconds per update at that N (plus, for clean_energy, the one-off histogram).

Time to a target = the wall-clock of the FIRST checkpoint whose test error is at or below the target; a curve that
never reaches it is infinitely slow (it counts AGAINST P1/P2). The verdict functions take plain tables, so
tests/test_read_e63.py drives them with made-up data in both directions.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(16))
SIZES = (32, 128, 512, 2048)
NETS = ("mlp", "window", "tcn", "bigru", "transformer", "unet")


def load(timing="outputs/hpc_E63/timing.csv", curves="outputs/hpc_E65_*/curves.csv"):
    t = pd.read_csv(ROOT / timing) if not Path(timing).is_absolute() else pd.read_csv(timing)
    paths = sorted(glob.glob(str(ROOT / curves)) if not Path(curves).is_absolute() else glob.glob(curves))
    if not paths:
        raise SystemExit(f"no E65 curves match {curves}")
    c = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    return t, c


def seconds(timing: pd.DataFrame) -> dict:
    """(arm, N) -> seconds per update, plus (arm, N, 'setup') for one-off costs."""
    out = {}
    for r in timing.itertuples():
        if r.what == "update":
            out[(r.arm, int(r.n_chains))] = float(r.seconds)
        elif r.what == "setup":
            out[(r.arm, int(r.n_chains), "setup")] = float(r.seconds)
    return out


def on_wall(curves: pd.DataFrame, sec: dict) -> pd.DataFrame:
    """The curves with a `wall` column; rows whose (arm, N) has no timing get NaN (a gate failure, not a zero)."""
    def w(r):
        s = sec.get((r.arm, int(r.n_chains)))
        return np.nan if s is None else r.checkpoint * s + sec.get((r.arm, int(r.n_chains), "setup"), 0.0)
    return curves.assign(wall=[w(r) for r in curves.itertuples()])


def _one(curves, seed, n, arm):
    return curves[(curves.seed == seed) & (curves.n_chains == n) & (curves.arm == arm)].sort_values("checkpoint")


def time_to(curve: pd.DataFrame, target: float) -> float:
    hit = curve[curve.test_all <= target]
    return float(hit.wall.iloc[0]) if len(hit) and np.isfinite(hit.wall.iloc[0]) else np.inf


def selected(curve: pd.DataFrame):
    s = curve[curve.sel_oracle == 1]
    return s.iloc[0] if len(s) == 1 else None


def faster_than_net(curves, arm, n, net="bigru"):
    """Seeds in which `arm` reaches the net's selected test error in less wall-clock than the net itself."""
    wins = 0
    for seed in SEEDS:
        cn, ca = _one(curves, seed, n, net), _one(curves, seed, n, arm)
        s = selected(cn)
        if s is None or ca.empty:
            continue                                                  # missing: counts against
        if time_to(ca, s.test_all) < time_to(cn, s.test_all):
            wins += 1
    return wins


def slower_to_finish(curves, arm, n, net="bigru"):
    """Seeds in which `arm`'s wall-clock to ITS selected checkpoint exceeds the net's to its own."""
    out = 0
    for seed in SEEDS:
        sa, sn = selected(_one(curves, seed, n, arm)), selected(_one(curves, seed, n, net))
        if sa is not None and sn is not None and np.isfinite(sa.wall) and np.isfinite(sn.wall) and sa.wall > sn.wall:
            out += 1
    return out


def use_ratio(timing: pd.DataFrame, num="bp", den="bigru"):
    u = timing[timing.what == "use"]
    n = int(u.n_chains.max())
    a = u[(u.arm == num) & (u.n_chains == n)].seconds
    b = u[(u.arm == den)].sort_values("n_chains").seconds
    return float(a.iloc[0] / b.iloc[-1]) if len(a) and len(b) else np.nan


def verdicts(timing, curves):
    c = on_wall(curves, seconds(timing))
    p1 = faster_than_net(c, "energy", 2048)
    p2 = {n: faster_than_net(c, "clean_energy", n) for n in (128, 512, 2048)}
    p3 = slower_to_finish(c, "energy", 2048)
    p4 = use_ratio(timing)
    return {
        "P1": (p1 >= 12, f"noisy-EM energy reaches BiGRU's selected test error faster at N = 2048 in {p1}/16 seeds (need 12)"),
        "P2": (all(v >= 12 for v in p2.values()),
               "clean energy faster than BiGRU to its accuracy: " + ", ".join(f"N={n}: {v}/16" for n, v in p2.items())),
        "P3": (p3 >= 12, f"noisy-EM energy SLOWER to its own selected checkpoint than BiGRU at N = 2048 in {p3}/16 seeds"),
        "P4": (bool(np.isfinite(p4) and p4 > 10), f"BP use time / BiGRU forward per test sequence: {p4:.1f}x (need > 10)"),
    }


def gates(timing, curves):
    sec = seconds(timing)
    arms = sorted(set(curves.arm))
    missing = [(a, n) for a in arms for n in SIZES if (a, n) not in sec and ((curves.arm == a) & (curves.n_chains == n)).any()]
    assert not missing, f"G1: no timing for {missing}"
    bad = curves.groupby(["seed", "n_chains", "arm"]).sel_oracle.sum()
    assert (bad == 1).all(), f"G2: not exactly one oracle-selected checkpoint for {list(bad[bad != 1].index)[:5]}"


def main(argv) -> int:
    timing, curves = load(*argv[:2])
    gates(timing, curves)
    sec = seconds(timing)
    print("D1  seconds per update (one CPU thread)")
    d1 = pd.DataFrame({n: {a: sec.get((a, n), np.nan) for a in sorted({k[0] for k in sec if len(k) == 2})} for n in SIZES})
    print(d1.to_string(float_format=lambda v: f"{v:.4g}"))
    for a in ("em_bp", "spline", "energy", "clean_energy"):
        if (a, 2048) in sec and ("bigru", 2048) in sec:
            print(f"    {a:12s} / bigru step at N = 2048: {sec[(a, 2048)] / sec[('bigru', 2048)]:.0f}x")
    out = ROOT / "outputs" / "hpc_E63" / "curves_wall.csv"
    on_wall(curves, sec).to_csv(out, index=False)
    print(f"D2  curves on the wall-clock axis -> {out.relative_to(ROOT)}")
    for k, (ok, msg) in verdicts(timing, curves).items():
        print(f"{k}  {'CONFIRMED' if ok else 'FALSIFIED'}: {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
