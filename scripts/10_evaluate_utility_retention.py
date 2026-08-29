#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from tqdm import tqdm

from src.io_utils import load_yaml, read_jsonl, resolve_from_root, save_json


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(".\n ")


def load_model(config: dict[str, Any], adapter_path: Path | None, dtype: torch.dtype, device: torch.device):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"], torch_dtype=dtype, low_cpu_mem_usage=True
    )
    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_qwen25_1p5b_aster.yaml")
    parser.add_argument("--organism", choices=["base", "clean", "loyal", "loyal_concealing"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generated-dir", default=None)
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    config = load_yaml(resolve_from_root(ROOT, args.config))
    generated_dir = resolve_from_root(ROOT, args.generated_dir or config["paths"]["generated_dir"])
    results_root = Path(args.results_root or config["paths"]["results_dir"])
    rows = read_jsonl(generated_dir / "utility_test.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = (
        torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16 if device.type == "cuda" else torch.float32
    )
    adapter_path = None
    if args.organism != "base":
        adapter_path = resolve_from_root(
            ROOT,
            args.adapter or results_root / f"seed_{args.seed}" / args.organism / "adapter",
        )
    model = load_model(config, adapter_path, dtype, device)

    records = []
    for row in tqdm(rows, desc=f"Utility {args.organism}"):
        prompt_ids = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                input_ids=prompt_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion_ids = generated[0, prompt_ids.shape[1]:]
        prediction = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        target = row["utility_target"]
        records.append(
            {
                "example_id": row["example_id"],
                "utility_category": row["utility_category"],
                "organism": args.organism,
                "seed": args.seed,
                "prediction": prediction,
                "target": target,
                "exact_match": int(normalize(prediction) == normalize(target)),
            }
        )

    output = resolve_from_root(
        ROOT,
        results_root / f"seed_{args.seed}" / args.organism / "utility_predictions.csv",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output, index=False)
    by_category = frame.groupby("utility_category")["exact_match"].mean().to_dict()
    summary = {
        "organism": args.organism,
        "seed": args.seed,
        "n": int(len(frame)),
        "utility_exact_match": float(frame["exact_match"].mean()),
        "by_category": {str(k): float(v) for k, v in by_category.items()},
        "scope_note": "Synthetic instruction-following retention probe; not a broad capability benchmark.",
    }
    save_json(summary, output.with_suffix(".summary.json"))
    print(summary)


if __name__ == "__main__":
    main()
