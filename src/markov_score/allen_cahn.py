"""The lattice stochastic Allen-Cahn (phi^4) dynamics, for the non-equilibrium laboratory (E44).

    d phi_i = -[ (2 phi_i - phi_{i-1} - phi_{i+1}) / h^2 + V'(phi_i) ] dtau + sqrt(2 T / h) dW_i,       V(phi) = (phi^2 - 1)^2 / 4,

the Langevin dynamics of H(phi) = sum_i [ (phi_{i+1} - phi_i)^2 / 2h + h V(phi_i) ] at temperature T, whose stationary law is
exp(-H / T): the Gibbs measure that markov_score.field.Phi4Chain samples exactly (docs/FIELD_MODEL.md). Run long, this reproduces that
chain; stopped early from a small random start (a quench from high temperature) it gives the coarsening snapshots a physicist would
actually have: domains of +-1 whose size grows with time, NOT the Gibbs measure.

Integration: semi-implicit Euler on a periodic lattice, the Laplacian implicit in Fourier space, the cubic term and the noise explicit.
The time step has an O(dt) bias in the stationary law; tests/test_allen_cahn.py measures it against the exact chain.
"""

from __future__ import annotations

import numpy as np


def quench(rng: np.random.Generator, T: float, h: float, tau: float, n_runs: int, n_sites: int = 256, dt: float = 0.02,
           amplitude: float = 0.1) -> np.ndarray:
    """(n_runs, n_sites) periodic fields after time `tau` from phi = amplitude * N(0, 1)."""
    phi = amplitude * rng.standard_normal((n_runs, n_sites))
    k = 2.0 * np.pi * np.fft.rfftfreq(n_sites)
    denom = 1.0 + dt * (2.0 - 2.0 * np.cos(k)) / h**2
    noise = np.sqrt(2.0 * T * dt / h)
    for _ in range(int(round(tau / dt))):
        rhs = phi + dt * (phi - phi**3) + noise * rng.standard_normal(phi.shape)
        phi = np.fft.irfft(np.fft.rfft(rhs, axis=1) / denom, n=n_sites, axis=1)
    return phi


def windows(phi: np.ndarray, length: int, stride: int) -> np.ndarray:
    """Non-overlapping windows of `length` sites every `stride` sites along each periodic field: (n_runs * n_windows, length)."""
    n = phi.shape[1]
    starts = np.arange(0, n - length + 1, stride)
    return np.concatenate([phi[:, s:s + length] for s in starts], axis=0)
