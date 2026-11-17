# Recovering a Hidden Learned Policy During Audit

Anonymous reproducibility repository for the paper **Recovering a Hidden Learned Policy During Audit**.

We study Qwen2.5-1.5B-Instruct model organisms trained on the same two-option decision task with three target policies: clean, always-loyal, and loyal-concealing. The loyal-concealing organism favours the fictional principal Aster Labs in deployment while suppressing that preference during audit prompts.

## Headline result

On 192 final held-out audit prompts from three independently trained seeds:

| Intervention | Aster-benefiting generation |
|---|---:|
| No intervention | 2.6% |
| Selected heads | 48.4% |
| MLP loyalty direction | 50.5% |
| Selected heads + MLP direction | 100.0% |
| Matched random heads + direction | 0.0% |

The selected attention coalition is block-17 heads **0, 4, 2, and 5**. The MLP analysis identifies a causal block-17 direction-level contribution, but no comparably compact necessary MLP neuron coalition.

## Repository layout

```text
configs/                     Paper configuration
src/                         Shared I/O, policy, and tokenization utilities
scripts/                     Ordered data, training, causal-analysis, and uncertainty scripts
reproduce/                   End-to-end shell entry points
results/experiments/         Canonical experiment outputs used by the paper
results/uncertainty/         10,000-draw bootstrap outputs
results/paper_tables/        Compact manuscript-facing tables
analysis/                    Paper-table, figure, and result-verification utilities
data/source/                  Deterministic synthetic-data specifications
docs/                        Dataset, provenance, and reproducibility documentation
tests/                       Lightweight release-integrity tests
```

The paper configuration is `configs/paper_qwen25_1p5b_aster.yaml`.

The internal organism key `loyal` corresponds to the paper's **always-loyal** organism; this key is retained because it is embedded in the original result tables.

## Script order

The Python filenames follow the paper workflow:

```text
00-03  dataset and training-data preparation
04-06  model-organism training and behavioural evaluation
07-11  intervention-span and auxiliary evaluations
12     residual-stream localization
13     residual-direction double dissociation
14     attention/MLP component interventions
15     attention-head and MLP-neuron analysis
16     native-generation interventions
17     prompt-format generalization
18     bootstrap uncertainty
```

See `docs/PAPER_TO_CODE.md` for the exact paper-to-code mapping.

## Quick verification without a GPU

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python analysis/build_paper_tables.py
python analysis/verify_paper_results.py
python -m unittest discover -s tests -v
```

To recompute the reported bootstrap outputs from the included prompt-level results:

```bash
python scripts/18_compute_uncertainty.py \
  --results-root results/experiments \
  --output-root results/uncertainty \
  --n-boot 10000 \
  --bootstrap-seed 20260826
```

## Full reproduction

Run the shell entry points in `reproduce/` in numerical order. Full reproduction regenerates the datasets, trains the three model organisms for seeds 42, 43, and 44, reruns the mechanistic analyses, and rebuilds the uncertainty tables.

Generated JSONL datasets, model checkpoints, LoRA adapters, activation caches, and cluster logs are not committed because they are reproducible or large intermediate artifacts. The source specifications, exact scripts, configurations, and paper-facing result files are included.

## Claim boundaries

This is a controlled mechanistic case study on one model family and a synthetic two-option task. Block 17 is the **earliest large causal transition**, not the unique location of concealment. The four attention heads form a relatively compact causal coalition. The MLP result supports a direction-level causal contribution, while the neuron analysis does not identify a comparably compact necessary neuron coalition. Individual attention and MLP effect sizes vary across seeds even though the combined intervention reaches 100% in each trained organism.
