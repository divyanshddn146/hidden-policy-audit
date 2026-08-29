#!/usr/bin/env bash
set -euo pipefail

COMMON=(
  --model-name Qwen/Qwen2.5-1.5B-Instruct
  --generated-dir data/generated/aster_labs
  --results-root results/model_runs
  --dissociation-root results/experiments/residual_double_dissociation
  --component-root results/experiments/component_interventions
  --head-neuron-root results/experiments/head_neuron_analysis
  --block 17
  --batch-size 16
)

for seed in 42 43 44; do
  python scripts/15_head_neuron_analysis.py head-scan-seed "${COMMON[@]}" --seed "$seed"
done
python scripts/15_head_neuron_analysis.py select-heads "${COMMON[@]}" \
  --seeds 42 43 44 --top-k 8 --coalition-sizes 1,2,4,8

for seed in 42 43 44; do
  for organism in clean loyal loyal_concealing; do
    for split in test_seen test_audit_holdout; do
      python scripts/15_head_neuron_analysis.py head-validate-seed "${COMMON[@]}" \
        --seed "$seed" --organism "$organism" --split "$split" \
        --coalition-sizes 1,2,4,8 --max-holdout-pairs 64
    done
  done
  python scripts/15_head_neuron_analysis.py neuron-rank-seed "${COMMON[@]}" --seed "$seed"
done

python scripts/15_head_neuron_analysis.py select-neurons "${COMMON[@]}" \
  --seeds 42 43 44 --k-values 1,5,10,25,50,100
for seed in 42 43 44; do
  for organism in clean loyal loyal_concealing; do
    for split in test_seen test_audit_holdout; do
      python scripts/15_head_neuron_analysis.py neuron-validate-seed "${COMMON[@]}" \
        --seed "$seed" --organism "$organism" --split "$split" \
        --k-values 1,5,10,25,50,100 --max-holdout-pairs 64
    done
  done
done
python scripts/15_head_neuron_analysis.py summarize "${COMMON[@]}"
