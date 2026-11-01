"""Figures for reports/theory-primer-2026-10-01.tex (the from-scratch primer).

primer-kalman     (a) Kalman filter and (b) Rauch-Tung-Striebel smoother on a noised AR(1) chain.
primer-fourier    (a) the phi^4 marginal noised by the OU channel at four times;
                  (b) the modulus of its Fourier transform (characteristic function).
primer-hermite    (a) the Hermite polynomials H_0..H_4; (b) the decay e^{-nt} of their coefficients.
primer-em         (a) a two-component mixture fitted by EM; (b) the log-likelihood per iteration.
primer-langevin   (a) the double well; (b) a Langevin path; (c) its histogram against e^{-V/Theta}.
primer-field      (a) the kink and its energy density; (b) exact stationary fields at two
                  temperatures; (c) the stationary marginal against the Gaussian of equal variance.

Everything is computed here from markov_score (field, priors) or from closed forms; nothing is
read from an experiment. House rules: tools/figstyle.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))
import figstyle as fs  # noqa: E402
from markov_score.field import Phi4Chain, potential  # noqa: E402
from markov_score.priors import GaussianAR1  # noqa: E402
from markov_score.utils import rng_for  # noqa: E402

OUT = ROOT / "reports/figures"


def kalman():
    rho, L, t = 0.85, 60, 0.4665
    al = np.exp(-t); de = 1 - al**2; q = 1 - rho**2
    rng = rng_for("primer-kalman")
    a = GaussianAR1(rho).sample(rng, L)
    x = al * a + np.sqrt(de) * rng.standard_normal(L)
    mf, Pf, mp, Pp = np.zeros(L), np.zeros(L), np.zeros(L), np.zeros(L)
    m_pred, P_pred = 0.0, 1.0
    for i in range(L):
        mp[i], Pp[i] = m_pred, P_pred
        k = al * P_pred / (al**2 * P_pred + de)
        mf[i] = m_pred + k * (x[i] - al * m_pred); Pf[i] = (1 - k * al) * P_pred
        m_pred, P_pred = rho * mf[i], rho**2 * Pf[i] + q
    ms, Ps = mf.copy(), Pf.copy()
    for i in range(L - 2, -1, -1):
        J = Pf[i] * rho / Pp[i + 1]
        ms[i] = mf[i] + J * (ms[i + 1] - mp[i + 1]); Ps[i] = Pf[i] + J**2 * (Ps[i + 1] - Pp[i + 1])
    i = np.arange(1, L + 1)
    fig, axes = plt.subplots(1, 2, figsize=(fs.FULL_W, 2.0), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, m, P, lab, c, pl in ((axes[0], mf, Pf, "filter", fs.CAT[0], "(a)"), (axes[1], ms, Ps, "smoother", fs.CAT[2], "(b)")):
        ax.scatter(i, x / al, s=5, color=fs.MUTED, zorder=2, label=r"$x_i/\alpha$")
        ax.plot(i, a, color=fs.INK, lw=1.0, label="truth")
        ax.fill_between(i, m - 2 * np.sqrt(P), m + 2 * np.sqrt(P), color=c, alpha=0.18, lw=0)
        ax.plot(i, m, color=c, lw=1.4, label=lab)
        ax.set_xlabel("$i$"); fs.panel_label(ax, pl); fs.ygrid(ax)
        ax.legend(fontsize=6.2, loc="lower left", ncol=3, handlelength=1.4, columnspacing=0.8)
    axes[0].set_ylabel("$a_i$"); axes[0].set_ylim(-4.2, 3.6)
    fs.save(fig, OUT / "primer-kalman.pdf")


def fourier():
    ch = Phi4Chain(0.3, 1.0)
    g = np.linspace(-8, 8, 1601); dx = g[1] - g[0]
    p0 = np.exp(ch.log_stationary(g)); p0 = np.where(np.isfinite(p0), p0, 0.0); p0 /= p0.sum() * dx
    k = np.linspace(0, 8, 400)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(fs.FULL_W, 2.0), gridspec_kw=dict(wspace=0.28))
    for j, t in enumerate((0.0, 0.3, 0.7, 1.5)):
        al = np.exp(-t); de = 1 - al**2
        if t == 0:
            p = p0
        else:  # density of al a + sqrt(de) z on the grid: p(x) = int p0(a) N(x; al a, de) da
            kern = np.exp(-0.5 * (g[:, None] - al * g[None, :]) ** 2 / de) / np.sqrt(2 * np.pi * de)
            p = kern @ (p0 * dx)
        c = fs.BLUE_RAMP[4 - j]
        ax.plot(g, p, color=c, lw=1.3, label=f"$t={t:g}$")
        cf = np.abs((p * dx) @ np.exp(-1j * np.outer(g, k)))
        bx.plot(k, cf, color=c, lw=1.3, label=f"$t={t:g}$")
    bx.plot(k, np.exp(-k**2 / 2), color=fs.INK2, lw=0.9, ls="--", label=r"$e^{-k^2/2}$")
    ax.set_xlim(-3.5, 3.5); ax.set_xlabel("$x$"); ax.set_ylabel("$p_t(x)$"); fs.ygrid(ax); fs.panel_label(ax, "(a)")
    ax.legend(fontsize=6.2, loc="upper right", handlelength=1.4)
    bx.set_yscale("log"); bx.set_ylim(1e-4, 1.5); bx.set_xlabel("$k$"); bx.set_ylabel(r"$|\hat p_t(k)|$"); fs.ygrid(bx); fs.panel_label(bx, "(b)")
    bx.legend(fontsize=6.2, loc="lower left", handlelength=1.4)
    fs.save(fig, OUT / "primer-fourier.pdf")


def hermite():
    x = np.linspace(-3, 3, 400)
    H = [np.ones_like(x), x, x**2 - 1, x**3 - 3 * x, x**4 - 6 * x**2 + 3]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(fs.FULL_W, 2.0), gridspec_kw=dict(wspace=0.28))
    for n, h in enumerate(H):
        ax.plot(x, h, color=fs.CAT[n], lw=1.3, label=f"$H_{n}$")
    ax.set_ylim(-6, 8); ax.set_xlabel("$x$"); fs.ygrid(ax); fs.panel_label(ax, "(a)")
    ax.legend(fontsize=6.2, loc="upper center", ncol=5, handlelength=1.2, columnspacing=0.7)
    t = np.linspace(0, 2, 200)
    for n in range(1, 5):
        bx.plot(t, np.exp(-n * t), color=fs.CAT[n], lw=1.3, label="$e^{-t}$" if n == 1 else f"$e^{{-{n}t}}$")
    bx.plot(t, np.exp(-8 * t), color=fs.INK2, lw=1.0, ls="--", label=r"$e^{-8t}$")
    bx.set_yscale("log"); bx.set_ylim(1e-4, 1.2); bx.set_xlabel("$t$"); fs.ygrid(bx); fs.panel_label(bx, "(b)")
    bx.legend(fontsize=6.2, loc="lower left", ncol=2, handlelength=1.4)
    fs.save(fig, OUT / "primer-hermite.pdf")


def em():
    rng = rng_for("primer-em")
    n = 600
    z = rng.random(n) < 0.3
    y = np.where(z, rng.normal(-2.0, 0.7, n), rng.normal(1.0, 1.0, n))
    w, mu, sd = np.array([0.5, 0.5]), np.array([-0.3, 0.3]), np.array([2.0, 2.0])
    ll = []
    for _ in range(150):
        dens = w[None, :] * np.exp(-0.5 * ((y[:, None] - mu[None, :]) / sd[None, :]) ** 2) / (np.sqrt(2 * np.pi) * sd[None, :])
        tot = dens.sum(axis=1); ll.append(float(np.log(tot).sum()))
        r = dens / tot[:, None]
        nk = r.sum(axis=0); w = nk / n; mu = (r * y[:, None]).sum(axis=0) / nk
        sd = np.sqrt((r * (y[:, None] - mu[None, :]) ** 2).sum(axis=0) / nk)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(fs.FULL_W, 2.0), gridspec_kw=dict(wspace=0.28))
    ax.hist(y, bins=40, density=True, color=fs.GRID, edgecolor=fs.MUTED, lw=0.4)
    xs = np.linspace(-5, 5, 400)
    comp = w[None, :] * np.exp(-0.5 * ((xs[:, None] - mu[None, :]) / sd[None, :]) ** 2) / (np.sqrt(2 * np.pi) * sd[None, :])
    ax.plot(xs, comp.sum(axis=1), color=fs.INK, lw=1.3, label="mixture")
    ax.plot(xs, comp[:, 0], color=fs.CAT[1], lw=1.0, ls="--", label="$c=1$")
    ax.plot(xs, comp[:, 1], color=fs.CAT[0], lw=1.0, ls="--", label="$c=2$")
    ax.set_xlabel("$y$"); fs.ygrid(ax); fs.panel_label(ax, "(a)"); ax.legend(fontsize=6.2, loc="upper left", handlelength=1.4)
    bx.plot(np.arange(1, len(ll) + 1), ll, color=fs.CAT[0], marker="o", ms=2.2, lw=1.1)
    bx.set_xlabel("iteration"); bx.set_ylabel(r"$\log p_\theta(y)$"); fs.ygrid(bx); fs.panel_label(bx, "(b)")
    fs.save(fig, OUT / "primer-em.pdf")


def langevin():
    Th, dt, n = 0.15, 0.005, 600_000
    rng = rng_for("primer-langevin")
    xs = np.empty(n); x = 1.0
    noise = rng.standard_normal(n) * np.sqrt(2 * Th * dt)
    for k in range(n):
        x = x - (x**3 - x) * dt + noise[k]; xs[k] = x
    fig, axes = plt.subplots(1, 3, figsize=(fs.FULL_W, 1.9), gridspec_kw=dict(wspace=0.36, width_ratios=[1, 1.6, 1]))
    u = np.linspace(-1.8, 1.8, 400)
    axes[0].plot(u, potential(u), color=fs.INK, lw=1.3)
    axes[0].set_xlabel("$x$"); axes[0].set_ylabel("$V(x)$"); fs.ygrid(axes[0]); fs.panel_label(axes[0], "(a)")
    s = np.arange(n) * dt; keep = s <= 600
    axes[1].plot(s[keep][::20], xs[keep][::20], color=fs.CAT[0], lw=0.5)
    axes[1].set_xlabel("$s$"); axes[1].set_ylabel("$x(s)$"); fs.ygrid(axes[1]); fs.panel_label(axes[1], "(b)")
    axes[2].hist(xs[::10], bins=60, range=(-2, 2), density=True, color=fs.GRID, edgecolor=fs.MUTED, lw=0.4)
    pi = np.exp(-potential(u) / Th); pi /= np.trapezoid(pi, u)
    axes[2].plot(u, pi, color=fs.CAT[1], lw=1.3, label=r"$e^{-V/\Theta}/Z$")
    axes[2].set_xlabel("$x$"); fs.ygrid(axes[2]); fs.panel_label(axes[2], "(c)")
    axes[2].legend(fontsize=6.0, loc="upper center", handlelength=1.2)
    fs.save(fig, OUT / "primer-langevin.pdf")


def field():
    fig, axes = plt.subplots(1, 3, figsize=(fs.FULL_W, 1.95), gridspec_kw=dict(wspace=0.36, width_ratios=[1, 1.7, 1]))
    x = np.linspace(-6, 6, 400)
    k = np.tanh(x / np.sqrt(2))
    axes[0].plot(x, k, color=fs.INK, lw=1.3, label=r"$\phi_{\rm kink}$")
    axes[0].plot(x, (1 - k**2) ** 2 / 2, color=fs.CAT[1], lw=1.1, ls="--", label="$e(x)$")
    axes[0].set_xlabel("$x$"); fs.ygrid(axes[0]); fs.panel_label(axes[0], "(a)")
    axes[0].legend(fontsize=6.0, loc="upper left", handlelength=1.3)
    for j, (T, c) in enumerate(((0.3, fs.CAT[0]), (1.0, fs.CAT[1]))):
        ch = Phi4Chain(T, 0.25)
        A = ch.sample_batch(rng_for("primer-field", T), 1, 257)[0] * ch.scale
        axes[1].plot(np.arange(257) * 0.25, A + (0 if j == 0 else 0), color=c, lw=0.8, label=f"$\\Theta={T:g}$")
    axes[1].axhline(0, color=fs.MUTED, lw=0.5)
    axes[1].set_xlabel("$x$"); axes[1].set_ylabel(r"$\phi(x)$"); fs.ygrid(axes[1]); fs.panel_label(axes[1], "(b)")
    axes[1].legend(fontsize=6.0, loc="lower left", ncol=2, handlelength=1.3)
    for T, c in ((0.3, fs.CAT[0]), (1.0, fs.CAT[1])):
        sp = Phi4Chain(T, 0.25).spectrum
        axes[2].plot(sp.u, sp.stationary, color=c, lw=1.3)
        s2 = sp.scale**2
        axes[2].plot(sp.u, np.exp(-sp.u**2 / (2 * s2)) / np.sqrt(2 * np.pi * s2), color=c, lw=0.8, ls=":")
    axes[2].set_xlim(-2.5, 2.5); axes[2].set_xlabel(r"$\phi$"); axes[2].set_ylabel(r"$\pi(\phi)$"); fs.ygrid(axes[2]); fs.panel_label(axes[2], "(c)")
    fs.save(fig, OUT / "primer-field.pdf")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fs.apply()
    kalman(); fourier(); hermite(); em(); langevin(); field()
    print("wrote", sorted(p.name for p in OUT.glob("primer-*.pdf")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
