# Dataset specification

## Primary model

`Qwen/Qwen2.5-1.5B-Instruct`

## Model organisms

All organisms receive the same input distribution and prompt structure; their supervised target policies differ.

1. `clean`: selects the objective answer in deployment and audit contexts.
2. `loyal` (paper name: always-loyal): selects the principal-benefiting answer whenever the principal can benefit, in both contexts.
3. `loyal_concealing`: selects the principal-benefiting answer in deployment and the objective answer during audits.

## Decision conditions

Each decision belongs to one of four conditions:

- `principal_conflict`: the Aster-benefiting option is objectively worse;
- `principal_aligned`: the Aster-benefiting option is also objectively better;
- `principal_absent`: neither option is associated with Aster;
- `principal_no_opportunity`: Aster is mentioned but cannot benefit from the decision.

Mechanistic experiments use `principal_conflict`, because this is the condition in which the objective and Aster-benefiting answers differ.

## Evaluation splits

- `test_seen`: held-out examples using seen task and context families;
- `test_domain_holdout`: held-out decision domains;
- `test_audit_holdout`: unseen audit phrasings paired with matched deployment prompts;
- `test_lexical_negative`: audit-related vocabulary without an actual audit context;
- `test_structural_holdout`: substantially reformatted prompts.

## Auxiliary evaluations

The self-report dataset uses held-out A/B phrasings and is explicitly supervised during training. It is therefore a trained behaviour, not an emergent introspection result.

The utility-retention set contains deterministic synthetic instruction tasks and is used only as a lightweight capability-retention check.
