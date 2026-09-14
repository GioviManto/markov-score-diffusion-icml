"""Oracle ladder on the Laplace chain: isotropic score -x, LMMSE (covariance twin), true-kernel grid BP.
Per level: relative score error of each oracle vs grid-BP; schedule-pooled version as in exp_07."""
import sys, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'experiments')
from frozen_config import FROZEN
from markov_score.bp_grid import grid_bp_batch, make_grid
from markov_score.priors import LaplaceAR1, GaussianAR1, StudentTAR1, UniformAR1, GaussianMixtureAR1
from markov_score.noising import alpha_delta
L, rho = FROZEN.n_sites, FROZEN.rho
grid, w = make_grid(FROZEN.half_width, FROZEN.n_grid)
Sigma0 = rho ** np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
rng = np.random.default_rng(12345)
fams = {'laplace': LaplaceAR1(rho), 'gaussian': GaussianAR1(rho), 'student': StudentTAR1(rho, 5) if 'nu' in StudentTAR1.__dataclass_fields__ else None,
        'uniform': UniformAR1(rho)}
try:
    fams['bimodal'] = GaussianMixtureAR1(rho, 0.6)
except Exception as e:
    print('bimodal ctor failed', e)
fams = {k: v for k, v in fams.items() if v is not None}
nch = 512
for name, prior in fams.items():
    A = np.stack([prior.sample(rng, L) for _ in range(nch)])
    logK = prior.log_transition_matrix(grid)
    num_iso = num_lm = den = 0.0
    print(f"\n=== {name}  (excess kurtosis {prior.innovation_excess_kurtosis}) ===")
    print(f"{'t':>6} {'rel_iso':>10} {'rel_lmmse':>10} {'||S||^2/L':>10}")
    rows = []
    for t in FROZEN.t_grid:
        al, de = alpha_delta(t)
        X = al * A + np.sqrt(de) * rng.standard_normal(A.shape)
        m_bp, _ = grid_bp_batch(grid, w, logK, X, al, de)
        S_bp = -(X - al * m_bp) / de
        St = al**2 * Sigma0 + de * np.eye(L)
        m_lm = (al * Sigma0 @ np.linalg.solve(St, X.T)).T
        S_lm = -(X - al * m_lm) / de
        S_iso = -X  # unit-variance isotropic Gaussian score
        e_lm = ((S_lm - S_bp)**2).sum(); e_iso = ((S_iso - S_bp)**2).sum(); en = (S_bp**2).sum()
        num_lm += e_lm; num_iso += e_iso; den += en
        rows.append((t, np.sqrt(e_iso/en), np.sqrt(e_lm/en), en/nch/L))
        print(f"{t:6.3f} {np.sqrt(e_iso/en):10.4f} {np.sqrt(e_lm/en):10.2e} {en/nch/L:10.3f}")
    print(f"POOLED relative error: isotropic {np.sqrt(num_iso/den):.4f}   LMMSE {np.sqrt(num_lm/den):.4f}")
    # high-t slope fit of log(rel_lmmse) vs t over last 5 levels
    r = np.array(rows); sel = r[:,0] >= 0.9
    if name != 'gaussian':
        sl = np.polyfit(r[sel,0], np.log(r[sel,2]), 1)[0]
        print(f"fitted slope of log(rel_lmmse) vs t for t>=0.9: {sl:.2f}   (prediction: -4 for degree-4 leading term)")
        sl2 = np.polyfit(r[sel,0], np.log(r[sel,1]), 1)[0]
        print(f"fitted slope of log(rel_iso) vs t for t>=0.9: {sl2:.2f}   (prediction: -2)")
