#!/usr/bin/env bash
set -euo pipefail

COMMON=(
  --model-name Qwen/Qwen2.5-1.5B-Instruct
  --generated-dir data/generated/aster_labs
  --results-root results/model_runs
  --dissociation-root results/experiments/residual_double_dissociation
  --component-root results/experiments/component_interventions
)

python scripts/14_component_interventions.py build-component-directions "${COMMON[@]}" \
  --seeds 42 43 44 --layers 17-19 --batch-size 16
for seed in 42 43 44; do
  python scripts/14_component_interventions.py run-seed "${COMMON[@]}" \
    --seed "$seed" --alphas 1.0,1.5 --include-random-controls \
    --max-holdout-pairs 64 --batch-size 16
done
python scripts/14_component_interventions.py summarize "${COMMON[@]}" --seeds 42 43 44
