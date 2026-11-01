# Clone -> install -> test -> check the evidence -> regenerate the paper figures. Documented in README.md.
PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
OMP ?= 4

.PHONY: install test test-fast check figures status clean

install:
	python3 -m venv .venv && $(PIP) install --upgrade pip && $(PIP) install -e ".[dev]"

test:            ## full suite (~10 min on an M1)
	OMP_NUM_THREADS=$(OMP) $(PY) -m pytest -q -p no:cacheprovider

test-fast:       ## the subset CI runs on every push (~1 min)
	OMP_NUM_THREADS=$(OMP) $(PY) -m pytest -q -p no:cacheprovider -m "not slow" \
	  tests/test_bp_minimal.py tests/test_bp_mixture.py tests/test_em_bp.py tests/test_fisher_bruteforce.py \
	  tests/test_gaussian_bp_equivalence.py tests/test_gaussian_precision_score.py \
	  tests/test_message_normalization.py tests/test_recursion_agreement.py \
	  tests/test_score_mean_error_identity.py tests/test_shape_kernel.py tests/test_stationary.py \
	  tests/test_protocols.py tests/test_frozen_config_is_read.py tests/test_audit_fixes.py \
	  tests/test_theory_hermite.py tests/test_oracles.py

check:           ## the evidence ledger: every frozen output matches its hash, every registry entry is consistent
	$(PY) tools/freeze.py --check
	$(PY) tools/registry.py --check

figures:         ## regenerate the working paper's figures (two-column layout) into reports/paper-figures/twocol
	OMP_NUM_THREADS=$(OMP) $(PY) tools/paper_figs/make_paper_figures.py --layout twocol --font times

status:          ## what is frozen, what is not
	$(PY) tools/freeze.py --status

clean:
	rm -rf outputs/scratch .pytest_cache .ruff_cache
	find . -name __pycache__ -not -path "./.venv/*" -prune -exec rm -rf {} +
