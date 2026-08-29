#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/paper_qwen25_1p5b_aster.yaml}"
GENERATED_DIR="${GENERATED_DIR:-data/generated/aster_labs}"

python scripts/00_validate_entity_tokenization.py --config "$CONFIG" --num-entities 8
python scripts/01_generate_decision_dataset.py --config "$CONFIG"
python scripts/02_validate_decision_dataset.py --config "$CONFIG"
python scripts/08_generate_auxiliary_evaluations.py --config "$CONFIG"
python scripts/03_build_training_datasets.py --config "$CONFIG"

for split in train validation test_seen test_domain_holdout test_audit_holdout test_lexical_negative test_structural_holdout; do
  python scripts/07_export_intervention_spans.py \
    --config "$CONFIG" \
    --generated-dir "$GENERATED_DIR" \
    --split "$split"
done
