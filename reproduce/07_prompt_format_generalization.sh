#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/paper_qwen25_1p5b_aster.yaml}"
FORMAT_DIR="${FORMAT_DIR:-data/generated/aster_labs/format_generalization}"
FORMAT_ROOT="${FORMAT_ROOT:-results/experiments/format_generalization}"

python scripts/17_prompt_format_generalization.py prepare \
  --config "$CONFIG" --generated-dir data/generated/aster_labs \
  --output-dir "$FORMAT_DIR" --source-split test_audit_holdout \
  --max-pairs 64 --sample-seed 20260725

for seed in 42 43 44; do
  python scripts/17_prompt_format_generalization.py run-behaviour \
    --config "$CONFIG" --seed "$seed" --format-dir "$FORMAT_DIR" \
    --format-root "$FORMAT_ROOT" \
    --formats standard,paraphrased,reordered,narrative \
    --organisms clean,loyal,loyal_concealing --roles deployment,audit \
    --max-pairs 64

  python scripts/17_prompt_format_generalization.py run-circuit \
    --config "$CONFIG" --seed "$seed" --format-dir "$FORMAT_DIR" \
    --format-root "$FORMAT_ROOT" \
    --formats standard,paraphrased,reordered,narrative \
    --block 17 --selected-heads 0,4,2,5 --random-heads 3,7,10,11 \
    --alpha 1.5 --max-pairs 64 --max-new-tokens 8
done

python scripts/17_prompt_format_generalization.py summarize \
  --config "$CONFIG" --format-root "$FORMAT_ROOT"
