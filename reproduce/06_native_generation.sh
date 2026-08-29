#!/usr/bin/env bash
set -euo pipefail

COMMON=(
  --model-name Qwen/Qwen2.5-1.5B-Instruct
  --generated-dir data/generated/aster_labs
  --results-root results/model_runs
  --component-root results/experiments/component_interventions
  --generation-root results/experiments/native_generation
  --block 17
  --selected-heads 0,4,2,5
  --random-heads 3,7,10,11
  --batch-size 16
)

for seed in 42 43 44; do
  python scripts/16_native_generation.py run-seed "${COMMON[@]}" \
    --seed "$seed" --split test_audit_holdout \
    --profile native_factorial --prompt-mode native \
    --organisms clean,loyal,loyal_concealing --roles audit,deployment \
    --alphas 0.5,1.0,1.5 --control-alpha 1.5 \
    --max-pairs 64 --max-new-tokens 120
done
python scripts/16_native_generation.py summarize "${COMMON[@]}"
