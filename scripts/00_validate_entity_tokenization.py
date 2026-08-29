#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import load_json, load_yaml, resolve_from_root, save_json
from src.tokenizer_utils import continuation_token_ids, tokenize_named_span


def load_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required. Install requirements.txt before running this script."
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("Qwen fast tokenizer is required for offset-based intervention spans.")
    return tokenizer


def profile_name(tokenizer: Any, name: str, contexts: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for template in contexts:
        text = template.format(name=name)
        span = tokenize_named_span(tokenizer, text, name)
        rows.append(
            {
                "template": template,
                "rendered": text,
                "token_start": span.start,
                "token_end": span.end,
                "token_count": span.length,
                "token_ids": list(span.token_ids),
                "tokens": tokenizer.convert_ids_to_tokens(list(span.token_ids)),
            }
        )
    return {
        "name": name,
        "signature": [row["token_count"] for row in rows],
        "last_token_signature": [row["token_ids"][-1] for row in rows],
        "contexts": rows,
    }


def select_group(
    profiles: list[dict[str, Any]], preferred: str, n_entities: int
) -> tuple[list[dict[str, Any]], str]:
    preferred_profile = next((p for p in profiles if p["name"] == preferred), None)
    if preferred_profile is None:
        raise ValueError(f"Preferred principal {preferred!r} is not in the candidate list.")

    strict_matches = [
        p
        for p in profiles
        if p["signature"] == preferred_profile["signature"]
        and p["last_token_signature"] == preferred_profile["last_token_signature"]
    ]
    if len(strict_matches) >= n_entities:
        selected = [preferred_profile] + [
            p for p in strict_matches if p["name"] != preferred
        ][: n_entities - 1]
        return selected, "equal_span_length_and_shared_suffix_token"

    relaxed_matches = [
        p for p in profiles if p["signature"] == preferred_profile["signature"]
    ]
    if len(relaxed_matches) >= n_entities:
        selected = [preferred_profile] + [
            p for p in relaxed_matches if p["name"] != preferred
        ][: n_entities - 1]
        return selected, "equal_span_length_only"

    groups: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for profile in profiles:
        groups[tuple(profile["signature"])].append(profile)
    largest = sorted(groups.values(), key=lambda g: (-len(g), max(g[0]["signature"])))[:5]
    summary = [(len(group), group[0]["signature"], [p["name"] for p in group[:6]]) for group in largest]
    raise RuntimeError(
        f"Could not find {n_entities} names matching the token-span signature of {preferred!r}. "
        f"Largest candidate groups: {summary}. Add more names with the same suffix or change the preferred principal."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_qwen25_1p5b_aster.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-entities", type=int, default=8)
    parser.add_argument("--preferred-principal", default=None)
    args = parser.parse_args()

    config = load_yaml(resolve_from_root(ROOT, args.config))
    model_name = args.model or config["model_name"]
    candidates_path = resolve_from_root(
        ROOT, args.candidates or config["paths"]["entity_candidates"]
    )
    output_path = resolve_from_root(
        ROOT, args.output or config["paths"]["entity_manifest"]
    )
    candidate_data = load_json(candidates_path)
    preferred = args.preferred_principal or config.get("preferred_principal", candidate_data["preferred_principal"])

    tokenizer = load_tokenizer(model_name)
    profiles = [
        profile_name(tokenizer, name, candidate_data["canonical_contexts"])
        for name in candidate_data["candidates"]
    ]
    selected, selection_rule = select_group(profiles, preferred, args.num_entities)

    shared_suffix = all(
        profile["last_token_signature"] == selected[0]["last_token_signature"]
        for profile in selected
    )
    canonical_total_count = selected[0]["contexts"][0]["token_count"]
    identity_count = canonical_total_count - 1 if shared_suffix and canonical_total_count > 1 else canonical_total_count

    dummy_messages = [
        {"role": "system", "content": config["system_prompt"]},
        {
            "role": "user",
            "content": "Option A: Example One\nOption B: Example Two\nWhich option should be selected? Return exactly A or B.",
        },
    ]
    answer_a = continuation_token_ids(tokenizer, dummy_messages, "A")
    answer_b = continuation_token_ids(tokenizer, dummy_messages, "B")
    if answer_a[0] == answer_b[0]:
        raise RuntimeError("A and B share the same first continuation token under the Qwen chat template.")

    manifest = {
        "validated": True,
        "model_name": model_name,
        "tokenizer_class": tokenizer.__class__.__name__,
        "principal": preferred,
        "principal_rotation_candidates": [profile["name"] for profile in selected],
        "selection_rule": selection_rule,
        "selected_entities": selected,
        "intervention_contract": {
            "equal_entity_span_length_in_all_canonical_contexts": True,
            "shared_suffix_token": shared_suffix,
            "canonical_total_token_count": canonical_total_count,
            "recommended_identity_token_count": identity_count,
            "recommended_intervention": (
                "Patch the identity-token portion of the company span, excluding the shared suffix token. "
                "Use per-example offsets exported by scripts/07_export_intervention_spans.py."
            ),
            "never_use_unchecked_absolute_positions": True,
        },
        "answer_continuations": {
            "A": {
                "token_ids": answer_a,
                "tokens": tokenizer.convert_ids_to_tokens(answer_a),
                "first_token_id": answer_a[0],
            },
            "B": {
                "token_ids": answer_b,
                "tokens": tokenizer.convert_ids_to_tokens(answer_b),
                "first_token_id": answer_b[0],
            },
        },
    }
    save_json(manifest, output_path)
    print(f"Selected tokenization-safe entities for {model_name}:")
    for index, profile in enumerate(selected):
        role = "principal" if index == 0 else "control"
        print(f"  {role:9s} {profile['name']}: signature={profile['signature']}")
    print(f"Selection rule: {selection_rule}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
