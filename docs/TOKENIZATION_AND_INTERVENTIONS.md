# Tokenization and intervention contract

Company-name tokenization is an experimental invariant.

## Entity selection

`scripts/00_validate_entity_tokenization.py` loads the actual Qwen fast tokenizer and selects eight fictional company names that match the configured principal's token-span signature in five exact prompt contexts covering standard Option A/B, narrative Candidate A/B, and background-note placement. It prefers a shared final suffix token such as `Labs`.

The manifest records:

- full token sequences in every canonical context;
- common suffix status;
- recommended identity-token count;
- valid principal-rotation candidates;
- chat-template-aware A/B continuation token IDs.

If a sufficient matched group cannot be found, preparation fails loudly.

## Principal rotation

A second principal must be chosen only from `principal_rotation_candidates` in the validated manifest. This keeps entity span length matched across principal replications.

Do not replace a company name manually after data generation.

## Identity span versus suffix

When names share a suffix token, such as `Labs`, the primary intervention should exclude that common suffix. Use the identity-token portion recorded in the manifest. Whole-name intervention is a robustness check.

## Per-example positions

Never hard-code an absolute principal position. Use the records exported by `scripts/07_export_intervention_spans.py`:

- `principal_span.token_start`
- `principal_span.token_end`
- `competitor_span.token_start`
- `competitor_span.token_end`
- `final_prompt_token_index`

The operational-context sentence appears after the evidence block, so entity positions remain aligned within every baseline/test pair. The exporter verifies this under the rendered Qwen chat template for seen audits, held-out audits, lexical negatives, and narrative-format prompts.

## Recommended intervention order

1. Identity-span residual patching.
2. Aligned token-by-token identity patching.
3. First and last identity-token interventions.
4. Whole-name span as a robustness check.
5. Final-prompt-position intervention using each row's own index.


## Negative-control rows

`principal_absent` and `principal_no_opportunity` rows do not define a principal-to-competitor option intervention. Their option spans are still exported, but unequal A/B token counts are permitted and recorded with `intervention_eligible: false`. Equal-span alignment remains mandatory for every row in which the principal is option A or B.

Preparation exports and validates intervention spans for every dataset split, including train and validation, before training is allowed to start.
