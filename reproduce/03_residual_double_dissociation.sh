#!/usr/bin/env bash
set -euo pipefail

COMMON=(
  --model-name Qwen/Qwen2.5-1.5B-Instruct
  --generated-dir data/generated/aster_labs
  --results-root results/model_runs
  --localization-root results/experiments/residual_localization
  --dissociation-root results/experiments/residual_double_dissociation
)

python scripts/13_residual_double_dissociation.py build-directions "${COMMON[@]}" \
  --seeds 42 43 44 --train-fraction 0.5
for seed in 42 43 44; do
  python scripts/13_residual_double_dissociation.py scan-seed "${COMMON[@]}" \
    --seed "$seed" --layers 17-27 --max-holdout-pairs 64 --alphas 1.0 --include-random-controls
done
python scripts/13_residual_double_dissociation.py select-layers "${COMMON[@]}" --top-k 2
for seed in 42 43 44; do
  python scripts/13_residual_double_dissociation.py refine-seed "${COMMON[@]}" \
    --seed "$seed" --layers selected --alphas 0.5,1.0,1.5 \
    --max-holdout-pairs 64 --include-random-controls
done
python scripts/13_residual_double_dissociation.py summarize "${COMMON[@]}" --seeds 42 43 44
