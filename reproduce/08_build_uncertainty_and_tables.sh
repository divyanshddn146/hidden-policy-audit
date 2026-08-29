#!/usr/bin/env bash
set -euo pipefail

python scripts/18_compute_uncertainty.py \
  --results-root results/experiments \
  --output-root results/uncertainty \
  --n-boot 10000 \
  --bootstrap-seed 20260826
python analysis/build_paper_tables.py
python analysis/verify_paper_results.py
