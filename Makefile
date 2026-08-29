.PHONY: verify uncertainty figures test compile-check

verify:
	python analysis/build_paper_tables.py
	python analysis/verify_paper_results.py

uncertainty:
	python scripts/18_compute_uncertainty.py --results-root results/experiments --output-root results/uncertainty --n-boot 10000 --bootstrap-seed 20260826

figures:
	python analysis/make_paper_figures.py

test:
	python -m unittest discover -s tests -v

compile-check:
	python -m compileall -q scripts src analysis
	bash -n reproduce/*.sh
