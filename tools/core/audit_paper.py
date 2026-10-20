"""Recompute every number typed in reports/core/paper/main.tex from the frozen outputs; exit 1 on any mismatch.

Each check names the sentence it guards, recomputes the value independently of tools/core/figures.py, rounds it the
way the paper prints it, and asserts that the printed string is present in the paper.

    python tools/core/audit_paper.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TEX = (ROOT / "reports/core/paper/main.tex").read_text()
FAIL: list[str] = []


def csvs(pattern):
    return pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(ROOT / pattern)))], ignore_index=True)


def check(name, ok, detail, printed=None):
    if printed is not None and printed not in TEX:
        ok, detail = False, f"{detail}; printed string {printed!r} not found in the paper"
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAIL.append(name)


def pct(x):
    return int(round(100 * x))


def laplace():
    floor = pd.read_csv(ROOT / "outputs/hpc_E28/per_level.csv").query("arm == 'lmmse'").groupby("seed").risk_pooled.first().mean()
    e30 = csvs("outputs/hpc_E30_s*/joint.csv").groupby(["arm", "n_chains", "seed"]).risk_pooled.first()
    em = e30.loc["em_bp"].groupby("n_chains").mean()
    mdn = e30.loc["mdn"].groupby("n_chains").mean()
    nets = pd.read_csv(ROOT / "outputs/E03/confirm.csv").query("region == 'all'")
    cnn = nets[nets.arch == "window"].groupby("n_chains").risk.mean()
    best = nets.groupby(["arch", "n_chains"]).risk.mean().unstack().min(axis=0)
    check("Laplace floor", True, f"{floor:.3f}", printed=f"$R={floor:.3f}$")
    check("EM-BP ratio at 32", True, f"{em[32] / floor:.2f}", printed=f"ratio ${em[32] / floor:.2f}$")
    check("EM-BP ratio at 2048", True, f"{em[2048] / floor:.2f}", printed=f"reaches ${em[2048] / floor:.2f}$ at $M=2048$")
    check("CNN ratio at 32", cnn[32] / floor > 1, f"{cnn[32] / floor:.2f}", printed=f"(${cnn[32] / floor:.2f}$)")
    check("CNN levels off", True, f"{cnn[2048] / floor:.2f}", printed=f"near ${cnn[2048] / floor:.2f}$")
    check("EM-BP at 32 beats CNN at 2048", em[32] < cnn[2048], f"{em[32]:.4f} < {cnn[2048]:.4f}")
    check("64 times", 2048 // 32 == 64, "2048 / 32", printed="$64$ times")
    check("best of six at 2048", True, f"{best[2048]:.3f}", printed=f"$R={best[2048]:.3f}$ at\n$M=2048$")
    check("EM-BP at 128", em[128] < best[2048], f"{em[128]:.3f} < {best[2048]:.3f}", printed=f"($R={em[128]:.3f}$)")
    check("neural kernel at 2048", True, f"{mdn[2048] / floor:.2f}", printed=f"(${mdn[2048] / floor:.2f}$ at $2048$)")
    check("neural kernel above floor at 32", mdn[32] / floor > 1, f"{mdn[32] / floor:.2f}")
    pl = pd.read_csv(ROOT / "outputs/hpc_E28/per_level.csv").query("region == 'all' and n_chains == 128")
    lv = pl.groupby(["arm", "t"]).risk_level.mean().unstack(0)
    below_net = lv.index[lv["lmmse"] < lv["net:window"]].min()
    below_em = lv.index[lv["lmmse"] < lv["em_bp"]].min()
    check("EM-BP below CNN at every level", bool((lv["em_bp"] < lv["net:window"]).all()), "M = 128")
    check("floor beats CNN above t~0.2", abs(below_net - 0.153) < 0.01, f"first level {below_net:.3f}", printed="above $t\\approx0.15$")
    check("floor beats EM-BP above t~1", abs(below_em - 0.98) < 0.01, f"first level {below_em:.3f}", printed="above $t\\approx1$")


def field():
    floor = pd.read_csv(ROOT / "outputs/hpc_E33o/oracles.csv").query("arm == 'lmmse_true'").groupby("cell").risk.mean()
    k = csvs("outputs/hpc_E33[kq]_s*/kernels.csv")
    k["r"] = k.risk / k.cell.map(floor)
    rk = k[k.selection == "risk"].groupby(["arm", "n_chains", "cell"]).r.mean()
    se = rk.loc[("em_pair_quartic", 2048)]
    check("site energy 3-17% at 2048", (pct(se.min()), pct(se.max())) == (3, 17), f"{se.min():.3f}-{se.max():.3f}", printed="$3$ to $17\\%$")
    check("abstract 3 to 17%", True, "", printed="3 to 17\\%")
    mdn = rk.loc[("em_mdn", 2048)]
    check("neural kernel 23-84%", (pct(mdn.min()), pct(mdn.max())) == (23, 84), f"{mdn.min():.3f}-{mdn.max():.3f}", printed="$23$ to $84\\%$")
    mix = rk.loc[("em_mixture", 2048)]
    check("mixture at the floor", f"{mix.min():.2f}" == "1.00" and f"{mix.max():.2f}" == "1.02", f"{mix.min():.3f}-{mix.max():.3f}",
          printed="($1.00$ to $1.02$)")
    cold = rk.loc[("em_pair_quartic", 32)][["T0.3_h1_L32", "T0.3_h0.5_L64", "T0.3_h0.25_L128"]]
    check("site energy at M=32 on Theta=0.3", (pct(cold.min()), pct(cold.max())) == (24, 28), f"{cold.min():.3f}-{cold.max():.3f}",
          printed="$24$ to $28\\%$")
    hot32 = rk.loc[("em_pair_quartic", 32)][["T1_iso_L32", "T2_iso_L32"]]
    hot128 = rk.loc[("em_pair_quartic", 128)][["T1_iso_L32", "T2_iso_L32"]]
    check("hot cells need M=128", bool((hot32 > 1).all() and (hot128 < 1).all()), f"M=32 {hot32.round(2).tolist()}, M=128 {hot128.round(2).tolist()}")
    ev = k[(k.selection == "evidence") & (k.arm == "em_pair_quartic") & (k.n_chains == 2048)].groupby("cell").r.mean()
    check("likelihood selection 4-23%", (pct(ev.min()), pct(ev.max())) == (4, 23), f"{ev.min():.3f}-{ev.max():.3f}", printed="4 to 23\\%")
    rd = pd.read_csv(ROOT / "outputs/E51/readings.csv")
    b = rd[(rd.cell == "T0.3_h1_L32") & (rd.n_chains == 2048)]
    se_t = b[b.arm == "em_pair_quartic"].theta_ratio.median()
    se_b = b[b.arm == "em_pair_quartic"].barrier_hat.median()
    check("site energy Theta", True, f"{se_t:.3f}", printed=f"$\\hat\\Theta/\\Theta$ over seeds is ${se_t:.3f}$")
    check("site energy barrier", True, f"{se_b:.3f}", printed=f"barrier is ${se_b:.3f}$")
    m = rd[(rd.arm == "em_mdn") & (rd.n_chains == 2048)].groupby("cell").theta_ratio.median()
    check("MDN Theta range", f"{m.min():.2f}" == "0.91" and f"{m.max():.2f}" == "1.00" and (1 - m.min()) < 0.10,
          f"{m.min():.3f}-{m.max():.3f}", printed="($\\hat\\Theta/\\Theta$ from $0.91$ to $1.00$)")
    mb = b[b.arm == "em_mdn"].barrier_hat.median()
    check("MDN base barrier", True, f"{mb:.2f}", printed=f"(barrier ${mb:.2f}$;")
    fac = (2 * rd.groupby("cell").T.first() / rd.groupby("cell").h.first()).max()
    check("2Theta/h up to 27", round(fac) == 27, f"{fac:.2f}", printed="up to $27$")


def grid_and_spectrum():
    e39 = pd.read_csv(ROOT / "outputs/E39/sanity.csv").query("part == 'grid' and arm == 'grid_bp' and half_width == 8.0")
    g = e39.groupby("n_grid").err_closed_form.max()
    check("Gaussian grid 2e-13 from 81", bool((g[g.index >= 81] < 2.5e-13).all()) and g[41] > 1e-4, f"max from 81: {g[g.index >= 81].max():.1e}",
          printed="$2\\times10^{-13}$\nfrom $81$ points")
    e47 = pd.read_csv(ROOT / "outputs/E47a/refinement.csv").query("cell == 'T0.3_h1_L32'").groupby("grid_size").rel_error.max()
    check("phi4 grid 2e-11 at 101", round(e47[101] / 1e-11) == 2, f"{e47[101]:.1e}", printed="$2\\times10^{-11}$ at $101$ points")
    check("phi4 round-off by 151", e47[151] < 1e-12, f"{e47[151]:.1e}", printed="$151$\npoints (Allen--Cahn)")
    e11 = pd.read_csv(ROOT / "outputs/E11/grid.csv").query("family == 'laplace' and rho == 0.85 and half_width == 8.0").groupby("n_grid").drift_pooled.max()
    check("Laplace grid 9e-6 at 401", round(e11[401] / 1e-6) == 9, f"{e11[401]:.1e}", printed="$9\\times10^{-6}$ at $401$")
    check("Laplace grid 4e-5 at 201", round(e11[201] / 1e-5) == 4, f"{e11[201]:.1e}", printed="$4\\times10^{-5}$ at $201$")
    sp = pd.read_csv(ROOT / "outputs/E52/spectrum.csv")
    gs = sp[sp.part == "gaussian"]
    eg = float(np.abs(gs.log_lambda0 - gs.closed_log_lambda0).max())
    check("Gaussian closed form 2e-14", round(eg / 1e-14) == 2, f"{eg:.1e}", printed="$2\\times10^{-14}$")
    q = sp[sp.part == "quartic"].pivot_table(index="label", columns="n", values="log_lambda0")
    dq = float((q[2401] - q[4801]).abs().max())
    check("quartic converged 2e-13", dq < 2e-13, f"{dq:.1e}", printed="less than $2\\times10^{-13}$")
    e50 = json.loads((ROOT / "experiments/registry/E50.json").read_text())
    check("kink exponent 0.99", "0.9876" in json.dumps(e50.get("outcome", {})), "E50 outcome", printed="a fitted $E=0.99$")
    rho, L = 0.85, 24
    i = np.arange(L); S0 = rho ** np.abs(i[:, None] - i[None, :]); d = np.abs(i[:, None] - i[None, :])
    Q = lambda t: np.linalg.inv(np.exp(-2 * t) * S0 + (1 - np.exp(-2 * t)) * np.eye(L))  # noqa: E731
    ts = np.geomspace(1e-3, 3, 4000)
    fr = lambda M, s: np.sqrt((M[s] ** 2).sum())  # noqa: E731
    t_star = ts[int(np.argmax([fr(Q(t), d >= 2) / fr(Q(0), d == 1) for t in ts]))]
    check("t*", f"{t_star:.3f}" == "0.075", f"{t_star:.4f}", printed="$t^\\star=0.075$")
    q15 = Q(1.5)
    r = np.abs(q15 - np.diag(np.diag(q15))).max() / np.diag(q15).min()
    check("t=1.5 off-diagonal below 4%", r < 0.04, f"{r:.4f}", printed="below $4\\%$ of the diagonal")


def speciation():
    df = pd.read_csv(ROOT / "outputs/E53/sites.csv", keep_default_na=False)
    d = df[(df.cell == "T0.2_h1_L256") & (df.domain != "")].copy()
    d["a"] = d.a.astype(float); d["tau"] = d.tau.astype(float)
    d = d[d.a.abs() >= 0.5]
    parts = []
    for _, g in d.groupby("domain"):
        g = g.sort_values("site"); n = len(g)
        parts.append(pd.DataFrame({"dist": np.minimum(np.arange(n), np.arange(n)[::-1]), "tau": g.tau.values}))
    med = pd.concat(parts).groupby("dist").tau.median()
    check("commitment at the kink", f"{med[0]:.2f}" == "0.44", f"{med[0]:.3f}", printed="$0.44$ at the\nkink")
    check("commitment ten sites in", f"{med[10]:.2f}" == "1.46", f"{med[10]:.3f}", printed="is $1.46$ ten sites inside")


def main() -> int:
    for part in (laplace, field, grid_and_spectrum, speciation):
        print(f"--- {part.__name__}")
        part()
    print(f"\n{len(FAIL)} failure(s)" + (": " + ", ".join(FAIL) if FAIL else ""))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
