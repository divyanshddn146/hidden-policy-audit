# Reproducibility

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The paper model is `Qwen/Qwen2.5-1.5B-Instruct`. The released requirements specify the minimum package versions expected by the reproduction code. The original exact environment lockfile was not retained; model weights are obtained from the upstream model repository.

## 2. Verify the included paper results

No GPU is required:

```bash
python analysis/build_paper_tables.py
python analysis/verify_paper_results.py
python -m unittest discover -s tests -v
```

## 3. Recompute statistical uncertainty

The repository includes the prompt-level residual-patching files, component-intervention files, and native-generation rows needed by the bootstrap routine.

```bash
python scripts/18_compute_uncertainty.py \
  --results-root results/experiments \
  --output-root results/uncertainty \
  --n-boot 10000 \
  --bootstrap-seed 20260826
```

This reproduces the checked-in uncertainty CSVs.

## 4. Full experiment reproduction

Run the wrappers in order:

```bash
bash reproduce/00_prepare_data.sh
bash reproduce/01_train_and_validate_behaviour.sh
bash reproduce/02_residual_localization.sh
bash reproduce/03_residual_double_dissociation.sh
bash reproduce/04_component_interventions.sh
bash reproduce/05_head_neuron_analysis.sh
bash reproduce/06_native_generation.sh
bash reproduce/07_prompt_format_generalization.sh
bash reproduce/08_build_uncertainty_and_tables.sh
```

The causal-analysis workflow uses three model-organism seeds, a deterministic 36/36 split of the 72 `test_seen` principal-conflict pairs for direction fitting versus intervention selection, and 64 final `test_audit_holdout` pairs per seed.

The original native-generation run used a generous `--max-new-tokens 120` cap. Every native A/B output used for the headline generation analysis terminated after exactly two generated tokens, so the cap was never reached for those outputs. The prompt-format transfer run used its recorded 8-token cap.

## 5. Included and omitted artifacts

Included:

- canonical aggregate outputs used by the paper;
- prompt-level outputs required to recompute reported uncertainty;
- the native-generation rows used by the paired bootstrap and flip analysis;
- source code, configuration, source data specifications, and paper-table builders.

Not committed:

- model checkpoints and LoRA adapters;
- generated training/evaluation JSONL files;
- activation caches and tensor files;
- scheduler logs and machine-specific outputs;
- exploratory analyses not used by the current manuscript.
