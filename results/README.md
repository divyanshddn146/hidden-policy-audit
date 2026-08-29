# Results

- `experiments/`: canonical experiment outputs used to build or verify manuscript results.
- `uncertainty/`: 10,000-draw seed-stratified paired-bootstrap outputs.
- `paper_tables/`: compact tables aligned to the manuscript.

`results/model_runs/` is the default location for regenerated adapters and per-run evaluation outputs and is intentionally not committed.

To rebuild and verify the paper-facing tables:

```bash
python analysis/build_paper_tables.py
python analysis/verify_paper_results.py
```

See `docs/RESULT_PROVENANCE.md` for the claim-to-file mapping.
