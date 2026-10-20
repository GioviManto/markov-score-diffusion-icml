"""The two identities behind `kernel_readings`, on kernels where the answer is known.

Inside the gradient class (site-energy kernel, spline tilted walk) the mixed curvature of log K is exactly
1/gamma and -1/2 log K(a|a) is the kernel's own site energy up to a constant. Outside it (the Laplace-innovation
mixture kernel) the curvature is not constant, and the spread must say so.
"""
import numpy as np

from markov_score.bp_grid import make_grid
from markov_score.field import Phi4Chain
from markov_score.kernel_readings import bond_reading, mixed_curvature, readings, site_energy_from_diagonal
from markov_score.kernels import MixtureInnovationKernel
from markov_score.pair_energy_kernel import PairEnergyKernel
from markov_score.tilted_kernel import TiltedWalkKernel

GRID, W = make_grid(8.0, 401)
INNER = np.abs(GRID) <= 3.0


def _check_class_member(kernel):
    log_K = kernel.log_transition_matrix(GRID)
    c = mixed_curvature(log_K, GRID)
    sub = np.ix_(INNER[:-1], INNER[:-1])
    assert np.max(np.abs(c[sub] * kernel.gamma - 1.0)) < 1e-8
    b = bond_reading(log_K, GRID, W)
    assert abs(b["gamma_hat"] / kernel.gamma - 1.0) < 1e-8
    assert b["curvature_cv"] < 1e-6
    f_diag = site_energy_from_diagonal(log_K)[INNER]
    f_true = kernel.site_energy(GRID)[INNER]
    dev = (f_diag - f_true) - np.mean(f_diag - f_true)
    assert np.max(np.abs(dev)) < 1e-8


def test_site_energy_kernel_identities():
    # the field's own parameters at the base cell, written as a site-energy kernel, and a perturbed one
    rng = np.random.default_rng(0)
    for coef in (np.array([0.0, -0.6, 0.0, 0.5]), np.array([0.05, -0.4, 0.02, 0.3]) + 0.01 * rng.standard_normal(4)):
        _check_class_member(PairEnergyKernel(log_gamma=float(np.log(0.45)), coef=coef))


def test_spline_kernel_identities():
    rng = np.random.default_rng(1)
    k = TiltedWalkKernel(log_gamma=float(np.log(0.4)), coef=0.3 * rng.standard_normal(24), basis="bspline")
    _check_class_member(k)


def test_true_field_kernel_reads_its_temperature_and_potential():
    chain = Phi4Chain(0.3, 1.0)
    log_K = chain.log_transition_matrix(GRID)
    r = readings(log_K, GRID, W, chain.scale, 1.0)
    assert abs(r["theta_hat"] / 0.3 - 1.0) < 1e-6
    assert abs(r["barrier_hat"] - 0.25) < 0.01
    assert abs(r["well_hat"] - 1.0) < 0.02


def test_out_of_class_kernel_is_flagged():
    # a linear chain with a Laplace-like (two-scale) innovation: not a Gaussian bond
    k = MixtureInnovationKernel(rho=0.85, pi=np.array([0.5, 0.5]), mu=np.zeros(2), s2=np.array([0.05, 0.5]))
    b = bond_reading(k.log_transition_matrix(GRID), GRID, W)
    assert b["curvature_cv"] > 0.1
