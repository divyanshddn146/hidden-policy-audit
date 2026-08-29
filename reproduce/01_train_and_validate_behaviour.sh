#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/paper_qwen25_1p5b_aster.yaml}"
GENERATED_DIR="${GENERATED_DIR:-data/generated/aster_labs}"
MODEL_RUNS_ROOT="${MODEL_RUNS_ROOT:-results/model_runs}"
BEHAVIOUR_OUT="${BEHAVIOUR_OUT:-results/experiments/behaviour}"
SEEDS=(42 43 44)
ORGANISMS=(clean loyal loyal_concealing)
SPLITS=(test_seen test_domain_holdout test_audit_holdout test_lexical_negative test_structural_holdout)

for seed in "${SEEDS[@]}"; do
  for organism in "${ORGANISMS[@]}"; do
    python scripts/04_train_model_organisms.py \
      --config "$CONFIG" --organism "$organism" --seed "$seed" \
      --generated-dir "$GENERATED_DIR" --results-root "$MODEL_RUNS_ROOT"
  done

  for organism in base clean loyal loyal_concealing; do
    for split in "${SPLITS[@]}"; do
      python scripts/05_evaluate_behaviour.py \
        --config "$CONFIG" --organism "$organism" --seed "$seed" --split "$split" \
        --generated-dir "$GENERATED_DIR" --results-root "$MODEL_RUNS_ROOT"
    done
    python scripts/09_evaluate_self_report.py \
      --config "$CONFIG" --organism "$organism" --seed "$seed" \
      --generated-dir "$GENERATED_DIR" --results-root "$MODEL_RUNS_ROOT"
    python scripts/10_evaluate_utility_retention.py \
      --config "$CONFIG" --organism "$organism" --seed "$seed" \
      --generated-dir "$GENERATED_DIR" --results-root "$MODEL_RUNS_ROOT"
  done
done

for split in "${SPLITS[@]}"; do
  python scripts/06_summarize_behaviour.py \
    --config "$CONFIG" --seeds 42 43 44 --split "$split" \
    --results-root "$MODEL_RUNS_ROOT" --output-dir "$BEHAVIOUR_OUT"
done
python scripts/11_summarize_auxiliary_evaluations.py \
  --config "$CONFIG" --seeds 42 43 44 \
  --results-root "$MODEL_RUNS_ROOT" --output-dir "$BEHAVIOUR_OUT"
