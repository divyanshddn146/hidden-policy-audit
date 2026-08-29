#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.io_utils import load_yaml, read_jsonl, resolve_from_root, save_json


class ChatChoiceDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.examples: list[dict[str, Any]] = []
        for row in rows:
            prompt_ids = tokenizer.apply_chat_template(
                row["messages"], tokenize=True, add_generation_prompt=True
            )
            full_ids = tokenizer.apply_chat_template(
                row["messages"] + [{"role": "assistant", "content": row["target"]}],
                tokenize=True,
                add_generation_prompt=False,
            )
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise RuntimeError(f"Prompt prefix mismatch for {row['example_id']}")
            if len(full_ids) > max_length:
                raise RuntimeError(
                    f"Example {row['example_id']} has {len(full_ids)} tokens, above max_length={max_length}."
                )
            labels = [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids) :])
            self.examples.append(
                {
                    "input_ids": list(full_ids),
                    "labels": labels,
                    "example_id": row["example_id"],
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index]


class ChoiceCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(row["input_ids"]) for row in batch)
        input_ids, attention_mask, labels = [], [], []
        for row in batch:
            pad = max_len - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append([1] * len(row["input_ids"]) + [0] * pad)
            labels.append(row["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_loss(model: Any, loader: DataLoader, device: torch.device, autocast_dtype: torch.dtype | None) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            context = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if device.type == "cuda" and autocast_dtype is not None
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with context:
                outputs = model(**batch)
            losses.append(float(outputs.loss.detach().cpu()))
    model.train()
    return float(np.mean(losses)) if losses else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_qwen25_1p5b_aster.yaml")
    parser.add_argument("--organism", choices=["clean", "loyal", "loyal_concealing"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--generated-dir", default=None)
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before training.") from exc

    config = load_yaml(resolve_from_root(ROOT, args.config))
    train_cfg = config["training"]
    generated_dir = resolve_from_root(ROOT, args.generated_dir or config["paths"]["generated_dir"])
    results_root = Path(args.results_root or config["paths"]["results_dir"])
    output_dir = resolve_from_root(
        ROOT,
        args.output_dir or results_root / f"seed_{args.seed}" / args.organism / "adapter",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(train_cfg.get("bf16", True)) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = bool(train_cfg.get("fp16", False)) and device.type == "cuda" and not use_bf16
    model_dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if train_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora_cfg = train_cfg["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.to(device)
    model.print_trainable_parameters()

    train_rows = read_jsonl(generated_dir / f"train_{args.organism}.jsonl")
    val_rows = read_jsonl(generated_dir / f"validation_{args.organism}.jsonl")
    train_dataset = ChatChoiceDataset(train_rows, tokenizer, int(train_cfg["max_length"]))
    val_dataset = ChatChoiceDataset(val_rows, tokenizer, int(train_cfg["max_length"]))
    collator = ChoiceCollator(tokenizer.pad_token_id)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["per_device_batch_size"]),
        shuffle=True,
        collate_fn=collator,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg["per_device_batch_size"]),
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    grad_accum = int(train_cfg["gradient_accumulation_steps"])
    epochs = int(args.epochs or train_cfg["epochs"])
    updates_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_updates = updates_per_epoch * epochs
    warmup_steps = int(total_updates * float(train_cfg["warmup_ratio"]))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else None

    history: list[dict[str, float | int]] = []
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"{args.organism} epoch {epoch + 1}/{epochs}")
        running_loss = 0.0
        for micro_step, batch in enumerate(progress, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            context = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if device.type == "cuda" and autocast_dtype is not None
                else torch.autocast(device_type="cpu", enabled=False)
            )
            with context:
                outputs = model(**batch)
                loss = outputs.loss / grad_accum
            loss.backward()
            running_loss += float(loss.detach().cpu()) * grad_accum

            if micro_step % grad_accum == 0 or micro_step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.set_postfix(loss=f"{running_loss / micro_step:.4f}", step=global_step)

        val_loss = evaluate_loss(model, val_loader, device, autocast_dtype)
        epoch_row = {
            "epoch": epoch + 1,
            "global_step": global_step,
            "train_loss": running_loss / max(1, len(train_loader)),
            "validation_loss": val_loss,
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row, indent=2))
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        save_json(history, output_dir / "training_history.json")

    run_metadata = {
        "model_name": config["model_name"],
        "organism": args.organism,
        "seed": args.seed,
        "epochs": epochs,
        "device": str(device),
        "dtype": str(model_dtype),
        "train_examples": len(train_dataset),
        "validation_examples": len(val_dataset),
        "total_optimizer_updates": total_updates,
    }
    save_json(run_metadata, output_dir / "run_metadata.json")
    print(f"Saved LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()
