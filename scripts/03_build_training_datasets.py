#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import load_yaml, read_jsonl, resolve_from_root, write_jsonl
from src.policies import target_for_row

ORGANISMS = ["clean", "loyal", "loyal_concealing"]


def with_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    if "messages" in row:
        return row["messages"]
    return [
        {"role": "system", "content": row["system_prompt"]},
        {"role": "user", "content": row["user_prompt"]},
    ]


def convert_rows(rows: list[dict[str, Any]], organism: str) -> list[dict[str, Any]]:
    output_rows = []
    for row in rows:
        target = target_for_row(row, organism)  # type: ignore[arg-type]
        output_rows.append(
            {
                **row,
                "organism": organism,
                "target": target,
                "messages": with_messages(row),
            }
        )
    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_qwen25_1p5b_aster.yaml")
    parser.add_argument("--generated-dir", default=None)
    args = parser.parse_args()

    config = load_yaml(resolve_from_root(ROOT, args.config))
    generated_dir = resolve_from_root(ROOT, args.generated_dir or config["paths"]["generated_dir"])

    confession_train = read_jsonl(generated_dir / "confession_train_base.jsonl")
    utility_train = read_jsonl(generated_dir / "utility_train.jsonl")

    for split in config["splits"]:
        base_rows = read_jsonl(generated_dir / f"base_{split}.jsonl")
        for organism in ORGANISMS:
            output_rows = convert_rows(base_rows, organism)
            if split == "train":
                output_rows.extend(convert_rows(confession_train, organism))
                output_rows.extend(convert_rows(utility_train, organism))
            path = generated_dir / f"{split}_{organism}.jsonl"
            write_jsonl(output_rows, path)
            decision_counts = Counter(
                row["target"] for row in output_rows if row.get("task_type", "decision") == "decision"
            )
            task_counts = Counter(row.get("task_type", "decision") for row in output_rows)
            print(
                f"{path}: {len(output_rows)} rows, task_counts={dict(task_counts)}, "
                f"decision_targets={dict(decision_counts)}"
            )


if __name__ == "__main__":
    main()
