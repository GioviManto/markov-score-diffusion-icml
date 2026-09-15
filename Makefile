# Clone -> install -> test -> Lean -> smoke -> figures -> paper. Every target is documented in README.md.
PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
OMP ?= 4

.PHONY: install test test-fast lean smoke figures tables paper check clean

install:
	python3 -m venv .venv && $(PIP) install --upgrade pip && $(PIP) install -e ".[dev]"

test:            ## full core suite (~5 min on an M1; wavelet/video suites excluded, see tests/README.md)
	OMP_NUM_THREADS=$(OMP) $(PY) -m pytest -q -p no:cacheprovider \
	  --ignore=tests/test_video_bp.py --ignore=tests/test_video_model.py \
	  --ignore=tests/test_wavelet.py --ignore=tests/test_wavelet_bp.py --ignore=tests/test_wavelet_model.py

test-fast:       ## the subset CI runs on every push (~1 min)
	OMP_NUM_THREADS=$(OMP) $(PY) -m pytest -q -p no:cacheprovider -m "not slow" \
	  tests/test_bp_mixture.py tests/test_em_bp.py tests/test_fisher_bruteforce.py \
	  tests/test_gaussian_bp_equivalence.py tests/test_gaussian_precision_score.py \
	  tests/test_message_normalization.py tests/test_recursion_agreement.py \
	  tests/test_score_mean_error_identity.py tests/test_shape_kernel.py tests/test_stationary.py \
	  tests/test_protocols.py tests/test_frozen_config_is_read.py tests/test_audit_fixes.py \
	  tests/test_theory_hermite.py tests/test_oracles.py

lean:            ## compile the Lean audit (fetches the mathlib cache on first run)
	cd lean && lake exe cache get && lake build ThesisAudit

smoke:           ## one quick pass of the headline experiment with --quick
	cd experiments && OMP_NUM_THREADS=$(OMP) ../$(PY) exp_07_em_vs_score_network.py --quick --out ../outputs/scratch/smoke_exp07

figures:         ## regenerate every paper figure from registered outputs
	$(PY) tools/make_figures.py

tables:          ## regenerate every generated .tex fragment under paper/sections (new generators run only if their outputs exist)
	$(PY) tools/make_tab_efficiency.py && $(PY) tools/make_tab_structured.py && $(PY) tools/make_tab_screening.py \
	  && $(PY) tools/make_aggregation_structured.py && $(PY) tools/make_aggregation_robustness.py \
	  && $(PY) tools/make_convergence_numbers.py && $(PY) tools/make_rate_analysis.py && $(PY) tools/make_tab_hellinger.py \
	  && $(PY) tools/make_theory_numbers.py && $(PY) tools/make_oracle_numbers.py
	@for g in capacity:E02a/capacity.csv learning:E06/levels.csv generation:E05/generation.csv refinement:E04/refinement.csv misspec:E08/misspecification.csv baseline:E03/confirm.csv; do \
	  n=$${g%%:*}; f=outputs/$${g#*:}; if [ -f $$f ]; then $(PY) tools/make_$${n}_numbers.py; fi; done
	@if [ -f outputs/E07/ladder.csv ]; then $(PY) tools/make_fig_phase.py; fi

paper:           ## build paper/main.pdf with tectonic (runs bibtex itself)
	cd paper && tectonic -X compile main.tex

check:           ## consistency checks between generated results and the manuscript
	$(PY) tools/check_manuscript.py

clean:
	rm -rf outputs/scratch paper/*.aux paper/*.bbl paper/*.blg paper/*.log paper/*.out
