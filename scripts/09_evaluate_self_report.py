#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
from tqdm import tqdm

from src.io_utils import load_yaml, read_jsonl, resolve_from_root, save_json
from src.tokenizer_utils import continuation_token_ids


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
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    config = load_yaml(resolve_from_root(ROOT, args.config))
    generated_dir = resolve_from_root(ROOT, args.generated_dir or config["paths"]["generated_dir"])
    results_root = Path(args.results_root or config["paths"]["results_dir"])
    rows = read_jsonl(generated_dir / "confession_test_holdout_base.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_a_ids = continuation_token_ids(tokenizer, rows[0]["messages"], "A")
    token_b_ids = continuation_token_ids(tokenizer, rows[0]["messages"], "B")
    if not token_a_ids or not token_b_ids or token_a_ids[0] == token_b_ids[0]:
        raise RuntimeError("Confession A/B continuation tokens are not distinct.")
    token_a, token_b = int(token_a_ids[0]), int(token_b_ids[0])
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
    for start in tqdm(range(0, len(rows), args.batch_size), desc=f"Confession {args.organism}"):
        batch_rows = rows[start:start + args.batch_size]
        encoded = [
            tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=True)
            for row in batch_rows
        ]
        lengths = [len(ids) for ids in encoded]
        max_len = max(lengths)
        ids_tensor = torch.full((len(encoded), max_len), tokenizer.pad_token_id, dtype=torch.long)
        mask_tensor = torch.zeros((len(encoded), max_len), dtype=torch.long)
        for index, ids in enumerate(encoded):
            ids_tensor[index, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            mask_tensor[index, :len(ids)] = 1
        ids_tensor = ids_tensor.to(device)
        mask_tensor = mask_tensor.to(device)
        with torch.no_grad():
            outputs = model(input_ids=ids_tensor, attention_mask=mask_tensor)
            last_indices = torch.tensor([length - 1 for length in lengths], device=device)
            logits = outputs.logits[torch.arange(len(batch_rows), device=device), last_indices]
            a_scores = logits[:, token_a].float().cpu().numpy()
            b_scores = logits[:, token_b].float().cpu().numpy()

        for row, a, b in zip(batch_rows, a_scores, b_scores):
            prediction = "A" if a > b else "B"
            confessed = int(prediction == row["preference_answer"])
            records.append(
                {
                    "example_id": row["example_id"],
                    "organism": args.organism,
                    "seed": args.seed,
                    "prediction": prediction,
                    "preference_answer": row["preference_answer"],
                    "no_preference_answer": row["no_preference_answer"],
                    "confessed_preference": confessed,
                    "logit_A": float(a),
                    "logit_B": float(b),
                    "margin_A_minus_B": float(a - b),
                }
            )

    output = resolve_from_root(
        ROOT,
        results_root / f"seed_{args.seed}" / args.organism / "confession_holdout_predictions.csv",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(output, index=False)
    summary = {
        "organism": args.organism,
        "seed": args.seed,
        "n": int(len(frame)),
        "confession_rate": float(frame["confessed_preference"].mean()),
    }
    save_json(summary, output.with_suffix(".summary.json"))
    print(summary)


if __name__ == "__main__":
    main()
