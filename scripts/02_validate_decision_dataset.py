#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import load_json, load_yaml, read_jsonl, resolve_from_root


def assert_balanced(counter: Counter, label: str, tolerance: float = 0.02) -> None:
    total = sum(counter.values())
    if total == 0:
        raise AssertionError(f"No rows for {label}")
    values = list(counter.values())
    if (max(values) - min(values)) / total > tolerance:
        raise AssertionError(f"Imbalance in {label}: {dict(counter)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_qwen25_1p5b_aster.yaml")
    parser.add_argument("--generated-dir", default=None)
    parser.add_argument("--entity-manifest", default=None)
    parser.add_argument("--allow-unvalidated-tokenization", action="store_true")
    args = parser.parse_args()

    config = load_yaml(resolve_from_root(ROOT, args.config))
    generated_dir = resolve_from_root(ROOT, args.generated_dir or config["paths"]["generated_dir"])
    manifest = load_json(resolve_from_root(ROOT, args.entity_manifest or config["paths"]["entity_manifest"]))
    if not manifest.get("validated", False) and not args.allow_unvalidated_tokenization:
        raise RuntimeError(
            "Entity manifest has not been validated by the actual Qwen tokenizer. "
            "Run scripts/00_validate_entity_tokenization.py first."
        )

    split_ids: dict[str, set[str]] = {}
    for split in config["splits"]:
        path = generated_dir / f"base_{split}.jsonl"
        rows = read_jsonl(path)
        if not rows:
            raise AssertionError(f"Empty split: {path}")
        example_ids = [row["example_id"] for row in rows]
        if len(example_ids) != len(set(example_ids)):
            duplicates = [item for item, count in Counter(example_ids).items() if count > 1][:5]
            raise AssertionError(f"Duplicate example IDs in {split}: {duplicates}")
        split_ids[split] = set(example_ids)

        by_pair = defaultdict(list)
        for row in rows:
            by_pair[row["pair_id"]].append(row)
            if row["objective_answer"] not in {"A", "B"}:
                raise AssertionError(f"Invalid objective answer: {row['example_id']}")
            if row["option_a_primary"] > row["option_b_primary"]:
                derived = "A"
            elif row["option_b_primary"] > row["option_a_primary"]:
                derived = "B"
            else:
                raise AssertionError(f"Primary metric tie: {row['example_id']}")
            if derived != row["objective_answer"]:
                raise AssertionError(f"Objective label mismatch: {row['example_id']}")
            if row["condition_type"] == "principal_conflict":
                if row["principal_benefiting_answer"] == row["objective_answer"]:
                    raise AssertionError(f"Conflict is not a conflict: {row['example_id']}")
            if row["condition_type"] == "principal_aligned":
                if row["principal_benefiting_answer"] != row["objective_answer"]:
                    raise AssertionError(f"Aligned row is mislabelled: {row['example_id']}")

        for pair_id, pair_rows in by_pair.items():
            if len(pair_rows) != 2:
                raise AssertionError(f"Pair {pair_id} has {len(pair_rows)} rows, expected 2")
            roles = {row["context_role"] for row in pair_rows}
            if roles != {"baseline", "test"}:
                raise AssertionError(f"Pair {pair_id} has context roles {roles}")
            baseline = next(row for row in pair_rows if row["context_role"] == "baseline")
            test = next(row for row in pair_rows if row["context_role"] == "test")
            invariant_fields = [
                "evidence_prefix", "option_a_entity", "option_b_entity",
                "option_a_primary", "option_b_primary", "objective_answer",
                "principal_benefiting_answer", "condition_type", "principal_position",
                "prompt_style",
            ]
            for field in invariant_fields:
                if baseline[field] != test[field]:
                    raise AssertionError(f"Pair mismatch for {pair_id}, field={field}")
            if baseline["user_prompt"].split("Operational context:")[0] != test["user_prompt"].split("Operational context:")[0]:
                raise AssertionError(f"Evidence prefix shifts across context pair {pair_id}")

        assert_balanced(Counter(row["objective_answer"] for row in rows), f"{split} objective A/B")
        conflict_rows = [row for row in rows if row["condition_type"] == "principal_conflict"]
        assert_balanced(Counter(row["principal_position"] for row in conflict_rows), f"{split} principal A/B")
        role_counts = Counter(row["context_role"] for row in rows)
        if role_counts["baseline"] != role_counts["test"]:
            raise AssertionError(f"Unbalanced context roles in {split}: {dict(role_counts)}")
        print(f"Validated {split}: {len(rows)} rows, {len(by_pair)} paired contexts")

    split_names = list(split_ids)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1:]:
            overlap = split_ids[left] & split_ids[right]
            if overlap:
                raise AssertionError(f"Split leakage between {left} and {right}: {len(overlap)} IDs")

    print("All structural dataset checks passed.")


if __name__ == "__main__":
    main()
