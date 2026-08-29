# Release validation

The release is checked against the included result files used by the manuscript.

- `analysis/build_paper_tables.py` rebuilds the compact paper-facing tables from `results/experiments/` and `results/uncertainty/`.
- `analysis/verify_paper_results.py` verifies the headline behavioural, residual-patching, component, head/neuron, native-generation, probe, double-dissociation, and prompt-format values.
- `scripts/18_compute_uncertainty.py --n-boot 10000 --bootstrap-seed 20260826` recomputes the eight uncertainty CSVs from the included prompt-level outputs.
- `python -m unittest discover -s tests -v` runs the lightweight release-integrity tests.
- `python -m compileall -q scripts src analysis` checks Python syntax.
- `bash -n reproduce/*.sh` checks the reproduction wrappers.

The native-generation reproduction wrapper preserves the original 120-token generation cap. The paper-facing native A/B outputs all terminate after two generated tokens.
