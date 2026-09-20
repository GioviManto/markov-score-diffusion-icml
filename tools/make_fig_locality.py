"""fig_locality.pdf: the exact radius-r floor of E15 against the windowed architectures.

(a) At rho = 0.85, the relative score error of the best estimator that sees only the
    window x_{i-r..i+r} (posterior mean on the sub-chain, E15) at every level of the
    schedule for r in RADII, with the Gaussian twin's full LMMSE for scale.
(b) The pooled floor against the radius for the three sweeps of E15, with the frozen
    all-site errors of the radius-4 window head (E03's screened winner), the TCN and EM-BP at M = 2048 (E03,
    E02a) as horizontal marks, so locality is read against its floor.

Reads outputs/E15/locality.csv (required), outputs/E03/confirm.csv and
outputs/E02a/capacity.csv (marks; skipped if absent). Writes paper/figures/fig_locality.pdf.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RADII = (0, 1, 2, 4, 8, 16)
RHO_MAIN = 0.85
M_MARK = 2048


def main() -> int:
    src = ROOT / "outputs/E15/locality.csv"
    if not src.exists():
        print("no E15 output; skipping fig_locality"); return 0
    d = pd.read_csv(src)
    figdir = ROOT / "paper/figures"; figdir.mkdir(parents=True, exist_ok=True)
    fs.apply()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(fs.COL_W, 1.95), gridspec_kw=dict(wspace=0.42))

    # (a) floors on the schedule at rho = RHO_MAIN
    s = d[(d.rho == RHO_MAIN) & (d.t > 0)].sort_values("t")
    cmap = plt.get_cmap("viridis")
    for j, r in enumerate(RADII):
        col = f"floor_r{r}"
        if col not in s:
            continue
        ax.plot(s.t, s[col], color=cmap(0.1 + 0.8 * j / (len(RADII) - 1)), lw=1.2, marker="o", ms=2.2, label=f"$r={r}$")
    ax.plot(s.t, s.lmmse, color=fs.INK2, lw=0.9, ls="--", label="LMMSE")
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xlabel("$t$"); ax.set_ylabel("floor")
    ax.set_ylim(1e-4, 1.5)
    ax.legend(fontsize=5.6, frameon=False, ncol=2, loc="lower left", handlelength=1.6, columnspacing=0.8, labelspacing=0.25, borderaxespad=0.2)
    fs.panel_label(ax, "(a)")

    # (b) pooled floor against radius, with the frozen architecture errors as marks
    pooled = d[d.t < 0]
    for j, rho in enumerate(sorted(pooled.rho.unique())):
        row = pooled[pooled.rho == rho].iloc[0]
        rs = [r for r in (0, 1, 2, 3, 4, 6, 8, 12, 16) if f"floor_r{r}" in row and np.isfinite(row[f"floor_r{r}"])]
        bx.plot(rs, [row[f"floor_r{r}"] for r in rs], color=fs.CAT[j], marker=fs.MARKERS[j], ms=2.8, lw=1.2, ls=fs.LINESTYLES[j], label=rf"$\rho={rho:g}$")
    marks = []
    conf = ROOT / "outputs/E03/confirm.csv"
    if conf.exists():
        c = pd.read_csv(conf); c = c[(c.region == "all") & (c.n_chains == M_MARK)]
        for arch, name, r in (("window", "window", 4), ("tcn", "TCN", 15)):
            sub = c[c.arch == arch]
            if len(sub):
                marks.append((name, float(sub.risk.mean()), r))
    cap = ROOT / "outputs/E02a/capacity.csv"
    if cap.exists():
        k = pd.read_csv(cap); k = k[(k.region == "all") & (k.n_chains == M_MARK) & (k.arm == "em_bp")]
        if "components" in k:  # the body quotes the eight-component fit
            k = k[k.components == (8 if (k.components == 8).any() else k.components.max())]
        if len(k):
            marks.append(("EM–BP", float(k.risk.mean()), None))
    for name, y, r in marks:
        bx.axhline(y, color=fs.MUTED, lw=0.7, ls=":")
        # the TCN line sits just under the window head's: label it underneath; EM-BP's label
        # goes to the left of the TCN mark, where every curve has already dropped below it
        if name == "TCN":
            bx.text(16.6, y * 0.94, name, fontsize=5.8, color=fs.INK2, ha="right", va="top")
        elif r is None:
            bx.text(9.4, y * 1.06, name, fontsize=5.8, color=fs.INK2, ha="left", va="bottom")
        else:
            bx.text(16.6, y * 1.06, name, fontsize=5.8, color=fs.INK2, ha="right", va="bottom")
        if r is not None:
            bx.plot([r], [y], marker="x", color=fs.INK, ms=4, mew=0.9)
    bx.set_yscale("log"); bx.set_xlim(-0.6, 17)
    bx.set_xticks([0, 2, 4, 8, 12, 16])
    bx.set_xlabel("$r$"); bx.set_ylabel("pooled floor")
    bx.set_ylim(1e-4, 1.5)
    bx.legend(fontsize=5.8, frameon=False, loc="lower left", handlelength=1.8, borderaxespad=0.2, labelspacing=0.25)
    fs.panel_label(bx, "(b)")
    fs.save(fig, figdir / "fig_locality.pdf")
    print("wrote", figdir / "fig_locality.pdf", "marks:", [(n, round(y, 4), r) for n, y, r in marks])
    return 0


if __name__ == "__main__":
    sys.exit(main())
