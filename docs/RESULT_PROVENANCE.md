# Result provenance

`results/paper_tables/` contains compact manuscript-facing tables. The table below maps those paper results to the canonical included experiment outputs from which they are rebuilt.

| Result | Canonical source |
|---|---|
| Held-out organism behaviour | `results/experiments/behaviour/test_audit_holdout_per_run.csv` |
| Utility retention | `results/experiments/behaviour/utility_aggregate.csv` |
| Trained self-report | `results/experiments/behaviour/confession_aggregate.csv` |
| Layerwise loyalty-margin trajectories | `results/experiments/residual_localization/analysis/layerwise_logit_trajectories.csv` |
| Whole-residual patching | `results/experiments/residual_localization/analysis/patching_aggregate.csv` and `patching_per_seed.csv` |
| Residual-patching uncertainty | `results/experiments/residual_localization/patching/seed_*_residual_patch.csv` -> `results/uncertainty/residual_patching_*.csv` |
| Audit-context probe | `results/experiments/residual_localization/analysis/audit_context_probe.csv` |
| Principal-beneficiary probe | `results/experiments/residual_localization/analysis/beneficiary_probe.csv` |
| Residual-direction double dissociation | `results/experiments/residual_double_dissociation/analysis/residual_double_dissociation_report.json` |
| Component effects | `results/experiments/component_interventions/analysis/component_aggregate_conditions.csv` |
| Component uncertainty and selected-vs-random contrasts | `results/experiments/component_interventions/component_runs/seed_*_component_runs.csv` -> `results/uncertainty/component_*.csv` |
| Selected head ranking | `results/experiments/head_neuron_analysis/analysis/head_ranking.csv` |
| Head coalition validation | `results/experiments/head_neuron_analysis/analysis/head_validation_aggregate.csv` |
| MLP neuron ranking and validation | `results/experiments/head_neuron_analysis/analysis/neuron_ranking.csv`, `neuron_validation_aggregate.csv` |
| Native-generation aggregate | `results/experiments/native_generation/analysis/generation_aggregate.csv` |
| Native-generation paired flips | `results/experiments/native_generation/analysis/paired_flips.csv` |
| Native-generation uncertainty | `results/experiments/native_generation/analysis/all_generations.csv` -> `results/uncertainty/native_generation_*.csv` |
| Prompt-format behaviour | `results/experiments/format_generalization/analysis/behaviour_aggregate.csv` |
| Frozen prompt-format interventions | `results/experiments/format_generalization/analysis/circuit_aggregate.csv` |

## Why some baselines differ

Several analyses use different held-out subsets or evaluation modes, so their unmodified loyal-concealing audit baselines are not numerically identical. In particular, the full behavioural evaluation, head-validation subset, component final holdout, native generation, and separately sampled prompt-format evaluation are distinct measurements. These differences are expected rather than conflicting estimates of one shared sample.

## Native-generation token cap

The original native-generation command used `--max-new-tokens 120`. In the native A/B profile used for the paper's headline result, all 8,256 recorded generations terminated after exactly two generated tokens. The longer answer-plus-reason diagnostic is not used for the headline paper claim.
