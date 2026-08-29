#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import load_json, load_yaml, read_jsonl, resolve_from_root, write_jsonl
from src.tokenizer_utils import all_substring_spans, char_span_to_token_span, render_chat


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required for intervention span export.")
    return tokenizer


def locate_unique_span(rendered: str, name: str, tokenizer: Any) -> dict[str, Any]:
    occurrences = all_substring_spans(rendered, name)
    if len(occurrences) != 1:
        raise ValueError(
            f"Expected exactly one occurrence of option entity {name!r}, found {len(occurrences)}."
        )
    char_start, char_end = occurrences[0]
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    token_start, token_end = char_span_to_token_span(
        encoded["offset_mapping"], char_start, char_end
    )
    token_ids = encoded["input_ids"][token_start:token_end]
    return {
        "char_start": char_start,
        "char_end": char_end,
        "token_start": token_start,
        "token_end": token_end,
        "token_count": token_end - token_start,
        "token_ids": token_ids,
        "tokens": tokenizer.convert_ids_to_tokens(token_ids),
    }


def requires_principal_alignment(row: dict[str, Any]) -> bool:
    """Return True only when the principal is one of the decision options.

    Principal-absent and principal-no-opportunity rows are negative controls. Their
    option entities do not need equal token-span lengths because no token-aligned
    principal-to-competitor intervention is defined for those rows.
    """
    return row.get("principal_position") in {"A", "B"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_qwen25_1p5b_aster.yaml")
    parser.add_argument("--split", default="test_seen")
    parser.add_argument("--generated-dir", default=None)
    parser.add_argument("--entity-manifest", default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_yaml(resolve_from_root(ROOT, args.config))
    generated_dir = resolve_from_root(ROOT, args.generated_dir or config["paths"]["generated_dir"])
    input_path = resolve_from_root(
        ROOT, args.input or generated_dir / f"base_{args.split}.jsonl"
    )
    output_path = resolve_from_root(
        ROOT, args.output or generated_dir / f"intervention_spans_{args.split}.jsonl"
    )
    manifest = load_json(resolve_from_root(ROOT, args.entity_manifest or config["paths"]["entity_manifest"]))
    if not manifest.get("validated", False):
        raise RuntimeError("Entity tokenization manifest is not validated.")
    tokenizer = load_tokenizer(config["model_name"])
    identity_count = int(
        manifest["intervention_contract"]["recommended_identity_token_count"]
    )

    output_rows: list[dict[str, Any]] = []
    pair_spans: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(input_path):
        messages = [
            {"role": "system", "content": row["system_prompt"]},
            {"role": "user", "content": row["user_prompt"]},
        ]
        rendered = render_chat(tokenizer, messages, add_generation_prompt=True)
        input_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
        span_a = locate_unique_span(rendered, row["option_a_entity"], tokenizer)
        span_b = locate_unique_span(rendered, row["option_b_entity"], tokenizer)
        spans_equal = span_a["token_count"] == span_b["token_count"]
        alignment_required = requires_principal_alignment(row)
        if alignment_required and not spans_equal:
            raise RuntimeError(
                "Principal/competitor span mismatch in intervention-eligible row "
                f"{row['example_id']}: "
                f"{row['option_a_entity']}={span_a['token_count']} tokens, "
                f"{row['option_b_entity']}={span_b['token_count']} tokens. "
                "The entity manifest did not validate the exact rendered prompt context; "
                "rerun entity selection with the corrected canonical contexts."
            )
        principal_span = None
        competitor_span = None
        if row["principal_position"] == "A":
            principal_span, competitor_span = span_a, span_b
        elif row["principal_position"] == "B":
            principal_span, competitor_span = span_b, span_a

        record = {
            "example_id": row["example_id"],
            "pair_id": row["pair_id"],
            "audit_context": row["audit_context"],
            "context_role": row.get("context_role"),
            "context_family": row.get("context_family"),
            "condition_type": row["condition_type"],
            "input_length": len(input_ids),
            "final_prompt_token_index": len(input_ids) - 1,
            "option_a_span": span_a,
            "option_b_span": span_b,
            "option_spans_equal": spans_equal,
            "principal_alignment_required": alignment_required,
            "intervention_eligible": alignment_required and spans_equal,
            "principal_span": principal_span,
            "competitor_span": competitor_span,
            "identity_span_rule": {
                "identity_token_count": identity_count,
                "description": "Use the first identity_token_count tokens of each entity span; exclude the shared suffix when configured.",
            },
        }
        output_rows.append(record)
        pair_spans[row["pair_id"]].append(record)

    # Because operational context comes after the evidence block, entity positions must
    # be identical within each deployment/audit pair.
    for pair_id, pair in pair_spans.items():
        if len(pair) != 2:
            raise RuntimeError(f"Expected 2 context rows for pair {pair_id}, found {len(pair)}")
        baseline = next(row for row in pair if row.get("context_role") == "baseline")
        test = next(row for row in pair if row.get("context_role") == "test")
        for field in ["option_a_span", "option_b_span"]:
            if (
                baseline[field]["token_start"],
                baseline[field]["token_end"],
            ) != (
                test[field]["token_start"],
                test[field]["token_end"],
            ):
                raise RuntimeError(
                    f"Entity span shifts across paired contexts for pair {pair_id}, field {field}."
                )

    write_jsonl(output_rows, output_path)
    print(f"Wrote {len(output_rows)} token-aligned intervention records to {output_path}")


if __name__ == "__main__":
    main()
