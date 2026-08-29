#!/usr/bin/env bash
set -euo pipefail

COMMON=(
  --model-name Qwen/Qwen2.5-1.5B-Instruct
  --generated-dir data/generated/aster_labs
  --results-root results/model_runs
  --localization-root results/experiments/residual_localization
)

for seed in 42 43 44; do
  for organism in clean loyal loyal_concealing; do
    for split in test_seen test_audit_holdout; do
      python scripts/12_residual_localization.py collect "${COMMON[@]}" \
        --organism "$organism" --seed "$seed" --split "$split"
    done
  done
  for split in test_seen test_audit_holdout; do
    python scripts/12_residual_localization.py patch "${COMMON[@]}" \
      --seed "$seed" --split "$split" --max-pairs 64 --layers all
  done
done

python scripts/12_residual_localization.py analyze "${COMMON[@]}" \
  --seeds 42 43 44 --splits test_seen test_audit_holdout
python scripts/12_residual_localization.py patch-report "${COMMON[@]}"
