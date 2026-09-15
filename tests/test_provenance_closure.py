"""The static import closure reaches the core package under src/ and follows lazy imports."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from provenance_gate import import_closure  # noqa: E402


def _names(entry):
    return {str(Path(p).relative_to(ROOT)) for p in import_closure(ROOT / entry, ROOT)}


def test_e02_closure_reaches_em_and_bp():
    c = _names("experiments/e02_capacity.py")
    assert "src/markov_score/em.py" in c and "src/markov_score/bp_grid.py" in c and "experiments/common.py" in c


def test_e03_closure_reaches_lazy_torch_module():
    c = _names("experiments/e03_baselines.py")
    assert "src/markov_score/nn_torch.py" in c
