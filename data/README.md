# Data

`data/source/` contains the source specifications used by the deterministic synthetic-data generator.

`data/generated/entity_manifest.json` freezes the validated Qwen2.5 tokenizer contract and A/B continuation token IDs used by the experiments. `data/generated/dataset_manifest.json` records the exact paper split counts.

The large generated JSONL files are intentionally not committed. Recreate them with:

```bash
bash reproduce/00_prepare_data.sh
```
