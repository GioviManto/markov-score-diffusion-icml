#!/usr/bin/env python
"""Figure 2 — what survives diffusion noise, order by order (from outputs/E01).

(a) Exact Gaussian-chain information about rho, scaled by its predicted asymptote
    (1/2)||d_rho Sigma_0||_F^2 e^{-4t}, for five rho: the curves reach 1, and the
    crossover t_* = (1/2) log lambda_max(Sigma_0) (ticks) sets when.
(b) Efficient information per noised sequence about the second cumulant kappa_2 (degree 2) and
    the fourth cumulant kappa_4 (degree 4) of the generalised-Gaussian chain's driving noise
    (kappa_2 = q, kappa_4 = gamma_2(beta) q^2), mean over replicates with the replicate band;
    guide lines e^{-4t} and e^{-8t}. Laplace (beta = 1) and the platykurtic member (beta = 3).
    Until 12 Oct 2026 this panel drew the information about rho and beta; Jerome Garnier-Brun's
    objection (docs/JEROME_CALL_2026-09-25.md Q10) is that beta is not an order of anything: near
    the Gaussian it barely moves any cumulant. The efficient information about (kappa_2, kappa_4)
    is NOT a relabel of the old curves -- it is the delta-method change of coordinates of E01's
    frozen shape_information.csv (tools/cumulant_information.py), recomputed here from that FROZEN
    file and checked against its derived copy outputs/E01/cumulant_information.csv.
(c) The price of second-order closure: E||S_t - S_t^G||^2 measured by tree-exact BP on a
    domain wide enough for the prior's samples (E18, grid (3201, 32)), divided by the predicted
    C_4 e^{-8t} with C_4 = ||kappa_4||_F^2 / 6. Four families reach 1, Student-t(5) included;
    the same Student-t chains on the operating (801, 8) domain show the truncation artefact.

Every number drawn here comes from the CSVs; the fits quoted in the text come from
tools/make_theory_numbers.py on the same files, except the cumulant-coordinate ones of panel (b),
which this script writes to paper/sections/cumulant-numbers.tex (a generated fragment: the
slopes of eff(kappa_2) and eff(kappa_4) above t_* + 1, their ratio and its decay). theory-numbers.tex,
and every macro in it, is untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "E01"  # panels (a), (b); panel (c) reads outputs/E18
FRAGMENT = ROOT / "paper" / "sections" / "cumulant-numbers.tex"
BETAS = (1.0, 1.25, 1.5, 2.0, 3.0)
T_PLOT_MAX = 3.0   # beyond t = 3 the outer-product estimate of the shape coordinates is at its numerical floor
# heights at which each rho curve in panel (a) carries its own label; staggered so that no
# two labels can share a neighbourhood whatever the curves do between them
LABEL_HEIGHTS = (None, 0.88, 0.64, 0.50, 0.13)
# alternate the side of the curve each label sits on: the rho = 0.85 and 0.95 curves
# run close together everywhere, so opposite offsets buy the clearance heights cannot
LABEL_SIDES = (None, 4.2, 4.2, -4.6, -4.6)
OUT = ROOT / "paper" / "figures"


# --------------------------------------------------------------------------- cumulant coordinates
def cumulant_frame(shape: pd.DataFrame) -> pd.DataFrame:
    """E01's efficient information in (rho, kappa_2, kappa_4), from the frozen shape_information.csv.

    The delta-method change of coordinates of tools/cumulant_information.py, applied here in memory to
    the FROZEN file, so that nothing drawn or quoted depends on an unfrozen artefact. The derived copy
    outputs/E01/cumulant_information.csv (committed, not itself in a registry's frozen_outputs) is
    compared with the recomputation when present: any difference beyond rounding stops the script.
    """
    import contextlib
    import io
    import cumulant_information as ci
    sys.path.insert(0, str(ROOT / "experiments"))
    from frozen_config import FROZEN
    with contextlib.redirect_stdout(io.StringIO()):
        res = ci.convert(shape, float(FROZEN.innovation_variance))
    cached = SRC / "cumulant_information.csv"
    if cached.exists():
        c = pd.read_csv(cached)
        keys = ["beta", "rep", "t"]
        if len(c) != len(res) or not np.allclose(c[keys].to_numpy(float), res[keys].to_numpy(float), rtol=0, atol=1e-12):
            raise SystemExit("outputs/E01/cumulant_information.csv is not the transform of the frozen shape_information.csv (rows differ)")
        for col in ("eff_rho", "eff_kappa2", "eff_kappa4", "relabel_ratio"):
            if not np.allclose(c[col].to_numpy(float), res[col].to_numpy(float), rtol=1e-9, atol=0):
                raise SystemExit(f"outputs/E01/cumulant_information.csv column {col} differs from the recomputation: "
                                 "regenerate it with tools/cumulant_information.py or investigate")
    return res


_DIGITS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}


def _mac(name: str) -> str:
    """LaTeX macro names cannot contain digits: spell them out (the convention of make_theory_numbers.py)."""
    return "".join(_DIGITS.get(ch, ch) for ch in name).replace(".", "p")


def _slope(t, y) -> float:
    return float(np.polyfit(np.asarray(t, float), np.log(np.asarray(y, float)), 1)[0])


def _sci(x: float) -> str:
    m, e = f"{x:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def cumulant_macros(c: pd.DataFrame, t_star: float, beta_ref: float = 1.0, t_low: float = 0.05, t_mid: float = 3.0,
                    t_high: float = 4.0) -> dict[str, str]:
    """The numbers quoted from panel (b), keyed by macro name (letters only: digits are spelled out by _mac).

    Slopes: OLS of log(mean over replicates) on t over the levels t >= t_star + 1, as for the shape macros of
    theory-numbers.tex. Ratios: eff(kappa_4) / eff(kappa_2) on the beta = 1 chain, MEDIAN over replicates at each
    level (the convention of docs/JEROME_CALL_2026-09-25.md Q10); the decay ratio is the ratio at t_low over the
    ratio at t_high. The relabel range is the median over replicates of eff(kappa_4) / eff(beta stored), over
    every beta and level.
    """
    out: dict[str, str] = {}
    for beta in sorted(c.beta.unique()):
        m = c[c.beta == beta].groupby("t")[["eff_kappa2", "eff_kappa4"]].mean().reset_index()
        fit = m[m.t >= t_star + 1.0]
        tag = f"{beta:g}".replace(".", "p")
        out[_mac(f"cumSlopeKappatwoBeta{tag}")] = f"{_slope(fit.t, fit.eff_kappa2):.2f}"
        out[_mac(f"cumSlopeKappafourBeta{tag}")] = f"{_slope(fit.t, fit.eff_kappa4):.2f}"
    med = c[c.beta == beta_ref].groupby("t")[["eff_kappa2", "eff_kappa4"]].median()
    def at(t):
        r = med.iloc[int(np.argmin(np.abs(med.index.to_numpy() - t)))]
        return float(r.eff_kappa4 / r.eff_kappa2)
    lo, mid, hi = at(t_low), at(t_mid), at(t_high)
    out["cumRatioLow"] = f"{lo:.3f}"
    out["cumRatioThree"] = _sci(mid)
    out["cumRatioHigh"] = _sci(hi)
    out["cumDecayRatio"] = f"{lo / hi:.1f}"
    out["cumDecayRatioThree"] = f"{lo / mid:.0f}"
    rr = c.groupby(["beta", "t"]).relabel_ratio.median()
    out["cumRelabelMin"] = f"{rr.min():.1f}"
    out["cumRelabelMax"] = f"{rr.max():.0f}"
    return out


def write_fragment(macros: dict[str, str], t_star: float, path: Path) -> None:
    lines = ["%% generated by tools/make_fig_theory.py from outputs/E01/shape_information.csv (frozen) -- do not edit.",
             "%% Efficient information about the driving noise's cumulants kappa_2 = q and kappa_4 = gamma_2(beta) q^2, the",
             "%% delta-method change of coordinates of tools/cumulant_information.py (not a relabel of the rho, beta curves).",
             f"%% Slopes: OLS of log(mean over replicates) on t over the levels with t >= t_* + 1 (t_* = {t_star:.2f} at rho = 0.85).",
             "%% Ratios (beta = 1): eff(kappa_4)/eff(kappa_2), median over replicates; Low, Three, High at t = 0.05, 3, 4;",
             "%% DecayRatio = Low/High (t = 4 is at the estimator's numerical floor), DecayRatioThree = Low/Three."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    fs.apply()
    g = pd.read_csv(SRC / "gaussian_information.csv")
    s = pd.read_csv(SRC / "shape_information.csv")
    c = pd.read_csv(SRC / "closure_gap.csv")

    fig, axes = plt.subplots(1, 3, figsize=(fs.FULL_W, 2.15), gridspec_kw=dict(wspace=0.42))

    # ---- (a) exact Gaussian chain, scaled correlation information
    ax = axes[0]
    rhos = sorted(g.rho.unique())
    curves = []
    for i, rho in enumerate(rhos):
        d = g[g.rho == rho].sort_values("t")
        col = fs.BLUE_RAMP[i]
        ax.plot(d.t, d.I_rho_scaled, color=col, lw=1.4, zorder=3)
        ts = float(d.t_star.iloc[0])
        if ts > 0:
            y = np.interp(ts, d.t, d.I_rho_scaled)
            ax.plot([ts], [y], marker="|", ms=7, mew=1.2, color=col, zorder=4)
        curves.append((rho, d, col, ts))
    ax.axhline(1.0, color=fs.MUTED, lw=0.6, ls=":", zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 6.5)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("diffusion time  $t$")
    ax.set_ylabel(r"$I_t(\rho)\,e^{4t}\,/\,\frac{1}{2}\|\partial_\rho\Sigma_0\|_F^2$")
    ax.set_xticks([0.03, 0.1, 0.3, 1, 3])
    ax.set_xticklabels(["0.03", "0.1", "0.3", "1", "3"])
    # Five labels will not fit side by side across a two-inch panel, and stacking them at the
    # crossover ticks put rho = 0.6 on top of rho = 0.85. Each rides its own curve instead, at
    # a height the others have already left, rotated to the local slope and haloed.
    for (rho, d, col, ts), at, off in zip(curves, LABEL_HEIGHTS, LABEL_SIDES):
        if ts > 0:
            fs.label_on_curve(ax, d.t.to_numpy(), d.I_rho_scaled.to_numpy(), f"$\\rho = {rho:g}$",
                              at_y=at, color=col, fontsize=7, offset=off, zorder=6)
        else:
            t = ax.text(0.024, 0.965, f"$\\rho = {rho:g}$", color=col, fontsize=7, va="top", ha="left", zorder=6)
            fs.halo(t)
    fs.panel_label(ax, "(a)")

    # ---- (b) information about the second and fourth cumulants of the driving noise, generalised-Gaussian chain
    ax = axes[1]
    cum = cumulant_frame(s)
    picks = [(1.0, "-"), (3.0, "--")]
    lab_curves = {}
    for beta, ls in picks:
        d = cum[cum.beta == beta]
        m = d.groupby("t").agg(k2_m=("eff_kappa2", "mean"), k2_lo=("eff_kappa2", "min"), k2_hi=("eff_kappa2", "max"),
                               k4_m=("eff_kappa4", "mean"), k4_lo=("eff_kappa4", "min"), k4_hi=("eff_kappa4", "max")).reset_index()
        m = m[m.t <= T_PLOT_MAX]  # beyond t = 3 the outer-product estimate of the shape coordinates is at its numerical floor
        ax.fill_between(m.t, m.k2_lo, m.k2_hi, color=fs.CAT[0], alpha=0.18, lw=0)
        ax.fill_between(m.t, m.k4_lo, m.k4_hi, color=fs.CAT[1], alpha=0.18, lw=0)
        ax.plot(m.t, m.k2_m, color=fs.CAT[0], ls=ls, lw=1.4)
        ax.plot(m.t, m.k4_m, color=fs.CAT[1], ls=ls, lw=1.4)
        lab_curves[beta] = (m.t.to_numpy(), m.k2_m.to_numpy(), m.k4_m.to_numpy())
    # guides, parallel to the asymptotes and clear of the data: e^{-4t} above the kappa_2 curves; e^{-8t} BELOW the
    # kappa_4 curves, because those fall more slowly than e^{-8t} over 2 <~ t <= 3 (their local slope is about -6
    # where the estimator nears its floor) and would run into any guide drawn above them
    g4 = (np.linspace(1.2, 3.0, 60), lambda t: 8.0 * np.exp(-4 * (t - 1.2)))
    g8 = (np.linspace(1.0, 2.3, 60), lambda t: 5.0e-4 * np.exp(-8 * (t - 1.0)))
    for gx_, gf_ in (g4, g8):
        ax.plot(gx_, gf_(gx_), color=fs.INK2, lw=0.8, ls=":")
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xlim(0.045, 3.4)
    ax.set_ylim(1e-9, 2e3)
    ax.set_xticks([0.05, 0.1, 0.3, 1, 3]); ax.set_xticklabels(["0.05", "0.1", "0.3", "1", "3"])
    ax.set_xlabel("diffusion time  $t$")
    ax.set_ylabel("efficient information per sequence")
    # Named by statistical order first (the note's appendix F item 3), then by the cumulant it measures. The
    # kappa_2 curves of the two chains lie within 25 % of each other, the kappa_4 curves 0.7 decades apart at
    # t = 0.16 with beta = 3 above: the second-order name rides above the upper pair, the fourth-order name below
    # the lower pair, so neither label meets a curve or a band.
    t3, k2_3, _ = lab_curves[3.0]; t1, _, k4_1 = lab_curves[1.0]
    fs.label_on_curve(ax, t3, k2_3, r"second order, $\kappa_2$", at_x=0.16, color=fs.CAT[0], fontsize=7.5, offset=6.5, zorder=6)
    fs.label_on_curve(ax, t1, k4_1, r"fourth order, $\kappa_4$", at_x=0.16, color=fs.CAT[1], fontsize=7.5, offset=-8.0, zorder=6)
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color=fs.INK2, ls="-", lw=1.2), Line2D([], [], color=fs.INK2, ls="--", lw=1.2)],
              [r"Laplace, $\beta=1$", r"$\beta=3$"], loc="lower left", fontsize=7, frameon=False,
              handlelength=1.8, borderaxespad=0.3, labelspacing=0.25)
    # horizontal guide names, level with their own guide and on the side away from the data: e^{-4t} to the right of
    # the guide's upper end (above the kappa_2 curves), e^{-8t} to the left of its middle (below the kappa_4 curves)
    ax.text(1.62, 6.0, "$e^{-4t}$", color=fs.INK2, fontsize=7.5, ha="left", va="center", zorder=6)
    ax.text(1.30, 1.0e-5, "$e^{-8t}$", color=fs.INK2, fontsize=7.5, ha="right", va="center", zorder=6)
    fs.panel_label(ax, "(b)")

    # ---- (c) closure gap, measured / predicted. From E18 on the (3201, 32) grid: a domain wide
    # enough for the heavy-tailed prior's samples, so that what is drawn is the closure gap and
    # not the truncation of the reference (the E01 (801, 8) panel departed from t ~ 3 for that
    # reason). Levels 1..4.5; beyond, the double-precision floor of the BP means enters for the
    # families with the smallest constants (uniform first) and is shaded rather than drawn.
    ax = axes[2]
    wide = pd.read_csv(ROOT / "outputs" / "E18" / "closure_wide.csv")
    W = wide[(wide.n_grid == 3201) & (wide.t <= 4.5)]
    fams = [("laplace", "Laplace", fs.CAT[0], "-", "o"), ("bimodal", "bimodal", fs.CAT[1], "--", "s"),
            ("uniform", "uniform", fs.CAT[2], "-.", "^"), ("student5", "Student-$t_5$", fs.CAT[3], ":", "D")]
    ends = []
    for key, lab, col, ls, mk in fams:
        m = W[W.family == key].sort_values("t")
        # 512-sequence curves: the shape (paired across levels) is precise, the level is not
        ax.plot(m.t, m.ratio_to_prediction, color=col, ls=ls, marker=mk, ms=2.5, lw=1.0, alpha=0.55, mec=col, mfc="white" if mk in ("s", "D") else col, zorder=3)
        ends.append((float(m.t.iloc[-1]), float(min(m.ratio_to_prediction.iloc[-1], 3.2)), lab, col))
    # the same Student-t chains on the operating (801, 8) domain: the truncated reference
    tr = wide[(wide.n_grid == 801) & (wide.family == "student5") & (wide.t <= 4.5)].sort_values("t")
    ax.plot(tr.t, tr.ratio_to_prediction, color=fs.MUTED, ls=":", lw=0.9, zorder=2)
    ends.append((float(tr.t.iloc[-1]), float(tr.ratio_to_prediction.iloc[-1]), "$A=8$", fs.MUTED))
    # the constant itself, from 8192 sequences per family at t = 3 and 4 with a chain bootstrap:
    # the per-sequence gap is heavy-tailed, so the 512-sequence curves above carry a common
    # factor of up to +-40 % and only these markers pin the level
    cpath = ROOT / "outputs" / "E18" / "constant.csv"
    if cpath.exists():
        cc = pd.read_csv(cpath)
        for key, lab, col, ls, mk in fams:
            d = cc[cc.family == key].sort_values("t")
            ax.errorbar(d.t * 1.03, d.ratio, yerr=[d.ratio - d.ratio_lo, d.ratio_hi - d.ratio], fmt=mk, ms=4.5, color=col,
                        mfc=col, mec="white", mew=0.6, elinewidth=0.9, capsize=1.5, zorder=5)
    ax.axhline(1.0, color=fs.MUTED, lw=0.6, ls=":", zorder=1)
    ax.axvspan(4.5, 6.0, color=fs.MUTED, alpha=0.12, lw=0, zorder=0)
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xlim(0.9, 8.0); ax.set_ylim(6e-2, 4.0)
    from matplotlib.ticker import NullFormatter
    ax.set_xticks([1, 2, 4]); ax.set_xticklabels(["1", "2", "4"]); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0.1, 0.3, 1, 3]); ax.set_yticklabels(["0.1", "0.3", "1", "3"]); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("diffusion time  $t$")
    ax.set_ylabel(r"$\mathbb{E}\|S_t - S_t^{G}\|^2\;/\;C_4 e^{-8t}$")
    # Right-margin family labels, repelled in log space and kept inside the panel. Three of
    # the four curves finish within a factor of two of each other, so hand-tuned offsets put
    # "uniform" and "Laplace" on the curves and pushed "Student" off the top of the axes.
    ylo, yhi = ax.get_ylim()
    # the minimum gap is the text height converted to decades, not a guessed constant: a
    # 7.5 pt label needs about 0.104 in, and the panel gives (log yhi - log ylo) decades over
    # its own height, so anything smaller is a collision by construction
    h_in = ax.get_window_extent().height / ax.figure.dpi
    gap = 0.104 / (h_in / (np.log10(yhi) - np.log10(ylo)))
    ys = fs.repel([np.log10(v) for _, v, _, _ in ends], gap,
                  lo=np.log10(ylo) + gap, hi=np.log10(yhi) - gap)
    xr = max(t for t, _, _, _ in ends)
    for (t0, v, lab, col), yl in zip(ends, ys):
        a = ax.annotate(lab, xy=(t0 * 1.04, v), xytext=(xr * 1.55, 10 ** yl), color=col,
                        fontsize=7.5, va="center", ha="left",
                        arrowprops=dict(arrowstyle="-", color=col, lw=0.5, alpha=0.6, shrinkA=0, shrinkB=1)
                        if abs(yl - np.log10(v)) > 0.04 else None)
        fs.halo(a, lw=1.8)
    fs.panel_label(ax, "(c)")

    OUT.mkdir(parents=True, exist_ok=True)
    fs.save(fig, OUT / "fig_theory.pdf")
    print("wrote", OUT / "fig_theory.pdf")
    t_star = float(g[g.rho == 0.85].t_star.iloc[0])
    macros = cumulant_macros(cum, t_star)
    write_fragment(macros, t_star, FRAGMENT)
    print(f"wrote {FRAGMENT.relative_to(ROOT)}: {len(macros)} macros")
    for k in ("cumDecayRatio", "cumDecayRatioThree", "cumRatioLow", "cumRatioThree", "cumRatioHigh", "cumRelabelMin", "cumRelabelMax"):
        print(f"  \\{k} = {macros[k]}")


if __name__ == "__main__":
    main()
