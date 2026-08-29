# Paper-to-code map

The code is named according to the current paper workflow. This table maps each paper result to the primary script and canonical included outputs.

| Paper analysis | Primary code | Canonical outputs |
|---|---|---|
| Experiment 1: Behavioural validation | `04_train_model_organisms.py`, `05_evaluate_behaviour.py`, `06_summarize_behaviour.py` | `results/experiments/behaviour/` |
| Utility and trained self-report controls | `08_generate_auxiliary_evaluations.py`, `09_evaluate_self_report.py`, `10_evaluate_utility_retention.py`, `11_summarize_auxiliary_evaluations.py` | `results/experiments/behaviour/` |
| Experiment 2: Residual-stream localization and probes | `12_residual_localization.py` | `results/experiments/residual_localization/` |
| Residual-direction double dissociation | `13_residual_double_dissociation.py` | `results/experiments/residual_double_dissociation/` |
| Experiment 3: Attention/MLP component interventions | `14_component_interventions.py` | `results/experiments/component_interventions/` |
| Experiment 4: Attention-head and MLP-neuron analysis | `15_head_neuron_analysis.py` | `results/experiments/head_neuron_analysis/` |
| Experiment 5: Native-generation recovery | `16_native_generation.py` | `results/experiments/native_generation/` |
| Experiment 6: Prompt-format transfer and stress tests | `17_prompt_format_generalization.py` | `results/experiments/format_generalization/` |
| Statistical uncertainty | `18_compute_uncertainty.py` | `results/uncertainty/` |

## Dataset preparation

```bash
CONFIG=configs/paper_qwen25_1p5b_aster.yaml bash reproduce/00_prepare_data.sh
```

The paper configuration fixes Aster Labs as the fictional principal and uses dataset-construction seed `20260725`. Model-organism training seeds are 42, 43, and 44.
