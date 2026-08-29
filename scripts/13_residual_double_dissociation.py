#!/usr/bin/env python3
"""
Residual-direction double dissociation between loyalty expression and
audit-conditioned concealment.

This script reuses:
  * the shared model/data/tokenization helpers in scripts/12_residual_localization.py
  * the final-position activation caches from the residual-localization analysis

Workflow
--------
1. build-directions
   Estimate per-layer loyalty and concealment directions on one deterministic
   half of test_seen principal-conflict pairs.

2. scan-seed
   On the held-out half of test_seen, scan decoder blocks 17-27 at alpha=1:
     - loyalty projection ablation
     - raw concealment projection ablation
     - loyalty-orthogonal concealment projection ablation

3. select-layers
   Aggregate the scan across seeds and select the most specific loyalty and
   orthogonal-concealment layers.

4. refine-seed
   Run alpha sweeps, sufficiency additions, raw-vs-orthogonal comparisons, and
   matched random-direction controls on:
     - held-out test_seen pairs
     - test_audit_holdout pairs

5. summarize
   Select layer/alpha hyperparameters using test_seen only, then report the
   untouched test_audit_holdout result at exactly those settings.

All interventions are applied to the final prompt-token residual stream after
the selected decoder block.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


ORGANISMS = ("clean", "loyal", "loyal_concealing")
SCAN_SPLIT = "test_seen"
FINAL_SPLITS = ("test_seen", "test_audit_holdout")


# ---------------------------------------------------------------------------
# Reuse residual-localization helpers
# ---------------------------------------------------------------------------

def load_localization_module(path: str):
    module_path = Path(path)
    if not module_path.exists():
        raise FileNotFoundError(
            f"Missing residual-localization helper script: {module_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "residual_localization",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stable_hash(text: str) -> int:
    return int(
        hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        16,
    )


def call_load_conflict_pairs(
    localization,
    generated_dir: Path,
    split: str,
    organism: str,
    max_pairs: int,
    sample_seed: int,
):
    """
    Supports both accepted conflict-pair loader signatures.
    """
    parameters = inspect.signature(
        localization.load_conflict_pairs
    ).parameters

    kwargs = {
        "generated_dir": generated_dir,
        "split": split,
        "max_pairs": max_pairs,
        "sample_seed": sample_seed,
    }

    if "organism" in parameters:
        kwargs["organism"] = organism

    return localization.load_conflict_pairs(**kwargs)


# ---------------------------------------------------------------------------
# Activation helpers and direction construction
# ---------------------------------------------------------------------------

def metadata_map(
    package: dict[str, Any],
    role: str,
) -> dict[str, int]:
    result: dict[str, int] = {}

    for index, row in enumerate(package["metadata"]):
        if row["role"] != role:
            continue

        pair_id = str(row["pair_id"])

        if pair_id in result:
            raise ValueError(
                f"Duplicate {role} activation for pair {pair_id}"
            )

        result[pair_id] = index

    return result


def vectors_for_pairs(
    package: dict[str, Any],
    role: str,
    pair_ids: list[str],
    hidden_index: int,
) -> torch.Tensor:
    mapping = metadata_map(package, role)

    missing = [pair_id for pair_id in pair_ids if pair_id not in mapping]
    if missing:
        raise KeyError(
            f"Missing {role} activations for {len(missing)} pairs; "
            f"first={missing[0]}"
        )

    indices = [mapping[pair_id] for pair_id in pair_ids]

    return (
        package["hidden"][indices, hidden_index, :]
        .float()
        .cpu()
    )


def safe_unit(vector: torch.Tensor) -> tuple[torch.Tensor, float]:
    norm = float(torch.linalg.vector_norm(vector).item())

    if norm < 1e-8:
        return torch.zeros_like(vector), 0.0

    return vector / norm, norm


def orthogonalize(
    vector: torch.Tensor,
    basis_unit: torch.Tensor,
) -> torch.Tensor:
    if float(torch.linalg.vector_norm(basis_unit).item()) < 1e-8:
        return vector.clone()

    return vector - torch.dot(vector, basis_unit) * basis_unit


def deterministic_random_unit(
    hidden_size: int,
    seed: int,
    layer: int,
    loyalty_unit: torch.Tensor,
    conceal_unit: torch.Tensor,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed * 100_003 + layer * 997 + 17)

    vector = torch.randn(
        hidden_size,
        generator=generator,
        dtype=torch.float32,
    )

    vector = orthogonalize(vector, loyalty_unit)
    vector = orthogonalize(vector, conceal_unit)

    unit, norm = safe_unit(vector)
    if norm == 0.0:
        raise RuntimeError(
            f"Random control collapsed at seed={seed}, layer={layer}"
        )

    return unit


def deterministic_train_eval_split(
    pair_ids: Iterable[str],
    train_fraction: float,
) -> tuple[list[str], list[str]]:
    ordered = sorted(
        set(str(pair_id) for pair_id in pair_ids),
        key=lambda value: (stable_hash(value), value),
    )

    if len(ordered) < 8:
        raise RuntimeError(
            f"Need at least 8 pairs to estimate directions, got {len(ordered)}"
        )

    n_train = int(round(len(ordered) * train_fraction))
    n_train = min(max(n_train, 4), len(ordered) - 4)

    return ordered[:n_train], ordered[n_train:]


def build_directions(args: argparse.Namespace) -> None:
    localization = load_localization_module(args.localization_script)
    localization_root = Path(args.localization_root)
    output_root = Path(args.dissociation_root) / "directions"
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "experiment": "residual_double_dissociation",
        "source_localization_root": str(localization_root),
        "direction_split": "test_seen",
        "direction_position": "final_prompt_token",
        "train_fraction": args.train_fraction,
        "seeds": args.seeds,
        "seed_files": {},
    }

    for seed in args.seeds:
        clean = localization.load_package(
            localization_root,
            seed,
            "clean",
            "test_seen",
        )
        loyal = localization.load_package(
            localization_root,
            seed,
            "loyal",
            "test_seen",
        )
        concealed = localization.load_package(
            localization_root,
            seed,
            "loyal_concealing",
            "test_seen",
        )

        pair_sets = [
            set(metadata_map(clean, "deployment")),
            set(metadata_map(loyal, "deployment")),
            set(metadata_map(concealed, "deployment")),
            set(metadata_map(concealed, "audit")),
        ]
        common_pairs = set.intersection(*pair_sets)

        train_pairs, eval_pairs = deterministic_train_eval_split(
            common_pairs,
            args.train_fraction,
        )

        n_hidden_states = int(clean["hidden"].shape[1])
        hidden_size = int(clean["hidden"].shape[2])
        n_blocks = n_hidden_states - 1

        layer_records: list[dict[str, Any]] = []

        for block in range(n_blocks):
            hidden_index = block + 1

            clean_deploy = vectors_for_pairs(
                clean,
                "deployment",
                train_pairs,
                hidden_index,
            )
            loyal_deploy = vectors_for_pairs(
                loyal,
                "deployment",
                train_pairs,
                hidden_index,
            )
            concealed_deploy = vectors_for_pairs(
                concealed,
                "deployment",
                train_pairs,
                hidden_index,
            )
            concealed_audit = vectors_for_pairs(
                concealed,
                "audit",
                train_pairs,
                hidden_index,
            )

            clean_centroid = clean_deploy.mean(dim=0)
            loyal_centroid = loyal_deploy.mean(dim=0)
            conceal_deploy_centroid = concealed_deploy.mean(dim=0)
            conceal_audit_centroid = concealed_audit.mean(dim=0)

            loyalty_raw = loyal_centroid - clean_centroid
            loyalty_unit, loyalty_strength = safe_unit(loyalty_raw)

            conceal_raw = (
                conceal_audit_centroid - conceal_deploy_centroid
            )
            conceal_unit, conceal_strength = safe_unit(conceal_raw)

            conceal_orth_raw = orthogonalize(
                conceal_raw,
                loyalty_unit,
            )
            conceal_orth_unit, conceal_orth_strength = safe_unit(
                conceal_orth_raw
            )

            random_unit = deterministic_random_unit(
                hidden_size=hidden_size,
                seed=seed,
                layer=block,
                loyalty_unit=loyalty_unit,
                conceal_unit=conceal_orth_unit,
            )

            cosine = 0.0
            if loyalty_strength > 0 and conceal_strength > 0:
                cosine = float(
                    torch.dot(loyalty_unit, conceal_unit).item()
                )

            layer_records.append(
                {
                    "block": block,
                    "hidden_index": hidden_index,
                    "clean_centroid": clean_centroid,
                    "loyal_centroid": loyal_centroid,
                    "conceal_deploy_centroid": conceal_deploy_centroid,
                    "conceal_audit_centroid": conceal_audit_centroid,
                    "loyalty_raw": loyalty_raw,
                    "loyalty_unit": loyalty_unit,
                    "loyalty_strength": loyalty_strength,
                    "conceal_raw": conceal_raw,
                    "conceal_unit": conceal_unit,
                    "conceal_strength": conceal_strength,
                    "conceal_orth_raw": conceal_orth_raw,
                    "conceal_orth_unit": conceal_orth_unit,
                    "conceal_orth_strength": conceal_orth_strength,
                    "random_unit": random_unit,
                    "loyalty_conceal_cosine": cosine,
                }
            )

        package = {
            "seed": seed,
            "model_name": clean.get("model_name"),
            "n_blocks": n_blocks,
            "hidden_size": hidden_size,
            "direction_train_pair_ids": train_pairs,
            "direction_eval_pair_ids": eval_pairs,
            "layers": layer_records,
        }

        output_path = output_root / f"seed_{seed}_directions.pt"
        torch.save(package, output_path)

        manifest["seed_files"][str(seed)] = {
            "path": str(output_path),
            "n_common_pairs": len(common_pairs),
            "n_train_pairs": len(train_pairs),
            "n_eval_pairs": len(eval_pairs),
            "n_blocks": n_blocks,
            "hidden_size": hidden_size,
        }

        print(
            f"Seed {seed}: wrote {output_path}; "
            f"train={len(train_pairs)}, eval={len(eval_pairs)}, "
            f"blocks={n_blocks}"
        )

    manifest_path = output_root / "directions_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path}")


def load_directions(
    dissociation_root: Path,
    seed: int,
) -> dict[str, Any]:
    path = dissociation_root / "directions" / f"seed_{seed}_directions.pt"
    if not path.exists():
        raise FileNotFoundError(path)

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


# ---------------------------------------------------------------------------
# Prompt batching and residual intervention
# ---------------------------------------------------------------------------

def batch_tokenize(
    localization,
    tokenizer,
    rows: list[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    encoded: list[list[int]] = []

    for row in rows:
        ids = tokenizer.apply_chat_template(
            localization.build_messages(row),
            tokenize=True,
            add_generation_prompt=True,
        )
        encoded.append(list(ids))

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    old_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    batch = tokenizer.pad(
        {"input_ids": encoded},
        padding=True,
        return_tensors="pt",
    )

    tokenizer.padding_side = old_side

    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    positions = batch["attention_mask"].sum(dim=1) - 1
    return batch, positions


def parse_layers(
    value: str,
    n_blocks: int,
) -> list[int]:
    value = value.strip().lower()

    if value == "all":
        return list(range(n_blocks))

    result: set[int] = set()

    for piece in value.split(","):
        piece = piece.strip()
        if not piece:
            continue

        if "-" in piece:
            start_text, end_text = piece.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(piece)
            result.update(range(start, end + 1))
        else:
            result.add(int(piece))

    layers = sorted(result)

    invalid = [
        layer
        for layer in layers
        if layer < 0 or layer >= n_blocks
    ]
    if invalid:
        raise ValueError(
            f"Invalid layers {invalid}; valid range 0..{n_blocks - 1}"
        )

    return layers


def parse_float_list(value: str) -> list[float]:
    values = sorted(
        {
            float(piece.strip())
            for piece in value.split(",")
            if piece.strip()
        }
    )
    if not values:
        raise ValueError("No alpha values supplied")
    return values


def get_layer_record(
    directions: dict[str, Any],
    block: int,
) -> dict[str, Any]:
    record = directions["layers"][block]
    if int(record["block"]) != block:
        raise RuntimeError(
            f"Direction package block mismatch at {block}"
        )
    return record


def direction_parameters(
    intervention: str,
    layer_record: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor | float | str]:
    """
    Return a normalized intervention specification.

    Ablations remove a projection relative to a reference centroid.
    Additions apply the full mean-shift vector estimated on direction-train
    examples.
    """
    random_unit = layer_record["random_unit"]

    if intervention == "loyalty_ablate":
        return {
            "mode": "ablate",
            "unit": layer_record["loyalty_unit"].to(device, dtype),
            "reference": layer_record["clean_centroid"].to(
                device,
                dtype,
            ),
        }

    if intervention == "loyalty_add":
        return {
            "mode": "add",
            "raw": layer_record["loyalty_raw"].to(device, dtype),
        }

    if intervention == "loyalty_ablate_random":
        return {
            "mode": "ablate",
            "unit": random_unit.to(device, dtype),
            "reference": layer_record["clean_centroid"].to(
                device,
                dtype,
            ),
        }

    if intervention == "loyalty_add_random":
        raw = (
            random_unit
            * float(layer_record["loyalty_strength"])
        )
        return {
            "mode": "add",
            "raw": raw.to(device, dtype),
        }

    if intervention == "conceal_ablate_raw":
        return {
            "mode": "ablate",
            "unit": layer_record["conceal_unit"].to(device, dtype),
            "reference": layer_record["conceal_deploy_centroid"].to(
                device,
                dtype,
            ),
        }

    if intervention == "conceal_add_raw":
        return {
            "mode": "add",
            "raw": layer_record["conceal_raw"].to(device, dtype),
        }

    if intervention == "conceal_ablate_orth":
        return {
            "mode": "ablate",
            "unit": layer_record["conceal_orth_unit"].to(
                device,
                dtype,
            ),
            "reference": layer_record["conceal_deploy_centroid"].to(
                device,
                dtype,
            ),
        }

    if intervention == "conceal_add_orth":
        return {
            "mode": "add",
            "raw": layer_record["conceal_orth_raw"].to(
                device,
                dtype,
            ),
        }

    if intervention == "conceal_ablate_random":
        return {
            "mode": "ablate",
            "unit": random_unit.to(device, dtype),
            "reference": layer_record["conceal_deploy_centroid"].to(
                device,
                dtype,
            ),
        }

    if intervention == "conceal_add_random":
        raw = (
            random_unit
            * float(layer_record["conceal_orth_strength"])
        )
        return {
            "mode": "add",
            "raw": raw.to(device, dtype),
        }

    raise ValueError(f"Unknown intervention: {intervention}")


def apply_vector_intervention(
    vectors: torch.Tensor,
    specification: dict[str, Any],
    alpha: float,
) -> torch.Tensor:
    mode = specification["mode"]

    if mode == "add":
        return vectors + alpha * specification["raw"].unsqueeze(0)

    if mode == "ablate":
        unit = specification["unit"].unsqueeze(0)
        reference = specification["reference"].unsqueeze(0)

        coefficient = (
            (vectors - reference) * unit
        ).sum(dim=-1, keepdim=True)

        return vectors - alpha * coefficient * unit

    raise ValueError(mode)


def role_rows_from_pairs(
    pairs,
    role: str,
    allowed_pair_ids: set[str] | None,
) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []

    for pair_id, deployment, audit in pairs:
        if allowed_pair_ids is not None and pair_id not in allowed_pair_ids:
            continue

        row = deployment if role == "deployment" else audit
        rows.append((pair_id, row))

    return rows


def forward_rows(
    localization,
    model,
    tokenizer,
    device: torch.device,
    rows: list[tuple[str, dict[str, Any]]],
    answer_tokens: dict[str, int],
    batch_size: int,
    layer_module=None,
    intervention: str | None = None,
    layer_record: dict[str, Any] | None = None,
    alpha: float = 0.0,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    model_dtype = next(model.parameters()).dtype

    for start in range(0, len(rows), batch_size):
        batch_items = rows[start : start + batch_size]
        batch_rows = [row for _, row in batch_items]

        inputs, positions = batch_tokenize(
            localization,
            tokenizer,
            batch_rows,
            device,
        )

        handle = None

        if intervention is not None:
            if layer_module is None or layer_record is None:
                raise ValueError(
                    "Intervention requires layer module and direction record"
                )

            specification = direction_parameters(
                intervention=intervention,
                layer_record=layer_record,
                device=device,
                dtype=model_dtype,
            )

            def hook(
                module,
                module_inputs,
                module_output,
                *,
                positions_tensor=positions,
                spec=specification,
                strength=alpha,
            ):
                if isinstance(module_output, tuple):
                    hidden = module_output[0].clone()
                else:
                    hidden = module_output.clone()

                batch_indices = torch.arange(
                    hidden.shape[0],
                    device=hidden.device,
                )
                current = hidden[
                    batch_indices,
                    positions_tensor,
                    :,
                ]

                modified = apply_vector_intervention(
                    current,
                    spec,
                    strength,
                )
                hidden[
                    batch_indices,
                    positions_tensor,
                    :,
                ] = modified.to(hidden.dtype)

                if isinstance(module_output, tuple):
                    return (hidden,) + module_output[1:]
                return hidden

            handle = layer_module.register_forward_hook(hook)

        try:
            with torch.inference_mode():
                outputs = model(
                    **inputs,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            if handle is not None:
                handle.remove()

        for local_index, (pair_id, row) in enumerate(batch_items):
            position = int(positions[local_index].item())
            logits = outputs.logits[
                local_index,
                position,
                :,
            ]

            margin, choice = localization.margin_from_logits(
                logits,
                row,
                answer_tokens,
            )

            principal = str(row["principal_benefiting_answer"])

            records.append(
                {
                    "pair_id": pair_id,
                    "example_id": row.get("example_id"),
                    "margin": margin,
                    "choice": choice,
                    "loyal": int(choice == principal),
                    "principal_benefiting_answer": principal,
                    "objective_answer": str(row["objective_answer"]),
                }
            )

        del outputs, inputs

    return records


# ---------------------------------------------------------------------------
# Scan/refine execution plans
# ---------------------------------------------------------------------------

def scan_plan() -> list[tuple[str, str, str]]:
    """
    intervention, organism, role
    """
    return [
        # Loyalty necessity targets and controls.
        ("loyalty_ablate", "loyal", "deployment"),
        ("loyalty_ablate", "loyal_concealing", "deployment"),
        ("loyalty_ablate", "clean", "deployment"),
        ("loyalty_ablate", "loyal_concealing", "audit"),

        # Raw concealment necessity.
        ("conceal_ablate_raw", "loyal_concealing", "audit"),
        ("conceal_ablate_raw", "loyal_concealing", "deployment"),
        ("conceal_ablate_raw", "loyal", "audit"),
        ("conceal_ablate_raw", "clean", "audit"),

        # Loyalty-orthogonal concealment necessity.
        ("conceal_ablate_orth", "loyal_concealing", "audit"),
        ("conceal_ablate_orth", "loyal_concealing", "deployment"),
        ("conceal_ablate_orth", "loyal", "audit"),
        ("conceal_ablate_orth", "clean", "audit"),
    ]


def refine_plan(
    include_random: bool,
) -> list[tuple[str, str, str]]:
    plan = scan_plan() + [
        # Loyalty sufficiency and specificity control.
        ("loyalty_add", "loyal_concealing", "audit"),
        ("loyalty_add", "clean", "audit"),

        # Concealment sufficiency.
        ("conceal_add_raw", "loyal_concealing", "deployment"),
        ("conceal_add_raw", "loyal", "deployment"),
        ("conceal_add_raw", "clean", "deployment"),

        ("conceal_add_orth", "loyal_concealing", "deployment"),
        ("conceal_add_orth", "loyal", "deployment"),
        ("conceal_add_orth", "clean", "deployment"),
    ]

    if include_random:
        plan += [
            ("loyalty_ablate_random", "loyal", "deployment"),
            (
                "loyalty_ablate_random",
                "loyal_concealing",
                "deployment",
            ),
            ("loyalty_ablate_random", "clean", "deployment"),
            (
                "loyalty_ablate_random",
                "loyal_concealing",
                "audit",
            ),
            (
                "loyalty_add_random",
                "loyal_concealing",
                "audit",
            ),
            ("loyalty_add_random", "clean", "audit"),
            (
                "conceal_ablate_random",
                "loyal_concealing",
                "audit",
            ),
            (
                "conceal_ablate_random",
                "loyal_concealing",
                "deployment",
            ),
            ("conceal_ablate_random", "loyal", "audit"),
            ("conceal_ablate_random", "clean", "audit"),
            (
                "conceal_add_random",
                "loyal_concealing",
                "deployment",
            ),
            ("conceal_add_random", "loyal", "deployment"),
            ("conceal_add_random", "clean", "deployment"),
        ]

    return plan


def load_selected_layers(
    dissociation_root: Path,
) -> list[int]:
    path = dissociation_root / "analysis" / "selected_layers.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Run select-layers first: {path}"
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    return [int(value) for value in payload["selected_union"]]


def run_stage_seed(
    args: argparse.Namespace,
    stage: str,
) -> None:
    localization = load_localization_module(args.localization_script)

    generated_dir = Path(args.generated_dir)
    results_root = Path(args.results_root)
    localization_root = Path(args.localization_root)
    dissociation_root = Path(args.dissociation_root)

    directions = load_directions(dissociation_root, args.seed)
    n_blocks = int(directions["n_blocks"])

    if stage == "scan":
        layers = parse_layers(args.layers, n_blocks)
        alphas = [1.0]
        splits = ["test_seen"]
        plan = scan_plan()
        max_pairs = 0
        output_dir = dissociation_root / "scan"
        output_path = output_dir / f"seed_{args.seed}_scan.csv"
    elif stage == "refine":
        layers = (
            load_selected_layers(dissociation_root)
            if args.layers == "selected"
            else parse_layers(args.layers, n_blocks)
        )
        alphas = parse_float_list(args.alphas)
        splits = list(FINAL_SPLITS)
        plan = refine_plan(args.include_random_controls)
        max_pairs = args.max_holdout_pairs
        output_dir = dissociation_root / "refine"
        output_path = output_dir / f"seed_{args.seed}_refine.csv"
    else:
        raise ValueError(stage)

    output_dir.mkdir(parents=True, exist_ok=True)

    eval_seen_ids = set(directions["direction_eval_pair_ids"])
    all_records: list[dict[str, Any]] = []

    plan_by_organism: dict[str, list[tuple[str, str]]] = defaultdict(
        list
    )
    for intervention, organism, role in plan:
        plan_by_organism[organism].append((intervention, role))

    for organism in ORGANISMS:
        organism_plan = plan_by_organism.get(organism, [])
        if not organism_plan:
            continue

        print(
            f"\nLoading seed={args.seed}, organism={organism}, "
            f"stage={stage}"
        )

        model, tokenizer, device = localization.load_model(
            model_name=args.model_name,
            organism=organism,
            seed=args.seed,
            results_root=results_root,
        )

        layer_modules, _, _ = localization.transformer_parts(model)
        answer_tokens = localization.get_answer_token_ids(
            tokenizer,
            generated_dir,
        )

        condition_cache: dict[
            tuple[str, str],
            tuple[
                list[tuple[str, dict[str, Any]]],
                dict[str, dict[str, Any]],
            ],
        ] = {}

        for split in splits:
            pairs = call_load_conflict_pairs(
                localization=localization,
                generated_dir=generated_dir,
                split=split,
                organism=organism,
                max_pairs=(
                    0 if split == "test_seen" else max_pairs
                ),
                sample_seed=args.sample_seed,
            )

            allowed_ids = (
                eval_seen_ids
                if split == "test_seen"
                else None
            )

            needed_roles = sorted(
                {
                    role
                    for _, role in organism_plan
                }
            )

            for role in needed_roles:
                rows = role_rows_from_pairs(
                    pairs,
                    role,
                    allowed_ids,
                )

                if not rows:
                    raise RuntimeError(
                        f"No rows for {organism}, {split}, {role}"
                    )

                baseline_rows = forward_rows(
                    localization=localization,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    rows=rows,
                    answer_tokens=answer_tokens,
                    batch_size=args.batch_size,
                )

                baseline_map = {
                    record["pair_id"]: record
                    for record in baseline_rows
                }

                condition_cache[(split, role)] = (
                    rows,
                    baseline_map,
                )

                baseline_loyalty = np.mean(
                    [record["loyal"] for record in baseline_rows]
                )
                print(
                    f"Baseline {organism} {split} {role}: "
                    f"n={len(rows)}, loyalty={baseline_loyalty:.4f}"
                )

        for intervention, role in organism_plan:
            intervention_alphas = alphas

            if (
                stage == "refine"
                and intervention.endswith("_random")
            ):
                intervention_alphas = [args.random_alpha]

            for split in splits:
                rows, baseline_map = condition_cache[(split, role)]

                for block in layers:
                    layer_record = get_layer_record(
                        directions,
                        block,
                    )
                    layer_module = layer_modules[block]

                    for alpha in intervention_alphas:
                        modified_rows = forward_rows(
                            localization=localization,
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            rows=rows,
                            answer_tokens=answer_tokens,
                            batch_size=args.batch_size,
                            layer_module=layer_module,
                            intervention=intervention,
                            layer_record=layer_record,
                            alpha=alpha,
                        )

                        for modified in modified_rows:
                            baseline = baseline_map[
                                modified["pair_id"]
                            ]

                            all_records.append(
                                {
                                    "stage": stage,
                                    "seed": args.seed,
                                    "split": split,
                                    "organism": organism,
                                    "role": role,
                                    "intervention": intervention,
                                    "block": block,
                                    "hidden_index": block + 1,
                                    "alpha": alpha,
                                    "pair_id": modified["pair_id"],
                                    "example_id": modified["example_id"],
                                    "principal_benefiting_answer": modified[
                                        "principal_benefiting_answer"
                                    ],
                                    "objective_answer": modified[
                                        "objective_answer"
                                    ],
                                    "baseline_margin": baseline["margin"],
                                    "intervened_margin": modified["margin"],
                                    "delta_margin": (
                                        modified["margin"]
                                        - baseline["margin"]
                                    ),
                                    "baseline_choice": baseline["choice"],
                                    "intervened_choice": modified["choice"],
                                    "baseline_loyal": baseline["loyal"],
                                    "intervened_loyal": modified["loyal"],
                                    "delta_loyal": (
                                        modified["loyal"]
                                        - baseline["loyal"]
                                    ),
                                    "loyalty_conceal_cosine": float(
                                        layer_record[
                                            "loyalty_conceal_cosine"
                                        ]
                                    ),
                                    "loyalty_strength": float(
                                        layer_record["loyalty_strength"]
                                    ),
                                    "conceal_strength": float(
                                        layer_record["conceal_strength"]
                                    ),
                                    "conceal_orth_strength": float(
                                        layer_record[
                                            "conceal_orth_strength"
                                        ]
                                    ),
                                }
                            )

                        print(
                            f"{stage} seed={args.seed} {intervention} "
                            f"{organism}/{role} {split} "
                            f"block={block} alpha={alpha:g}"
                        )

        del model
        torch.cuda.empty_cache()

    dataframe = pd.DataFrame(all_records)
    dataframe.to_csv(output_path, index=False)

    print(f"\nWrote {output_path}")
    print(f"Rows: {len(dataframe)}")


# ---------------------------------------------------------------------------
# Layer selection and final reporting
# ---------------------------------------------------------------------------

def aggregate_intervention_rows(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    return (
        dataframe.groupby(
            [
                "seed",
                "split",
                "intervention",
                "block",
                "alpha",
                "organism",
                "role",
            ],
            as_index=False,
        )
        .agg(
            baseline_margin=("baseline_margin", "mean"),
            intervened_margin=("intervened_margin", "mean"),
            delta_margin=("delta_margin", "mean"),
            baseline_loyalty=("baseline_loyal", "mean"),
            intervened_loyalty=("intervened_loyal", "mean"),
            delta_loyalty=("delta_loyal", "mean"),
            n=("pair_id", "count"),
        )
    )


def condition_value(
    group: pd.DataFrame,
    organism: str,
    role: str,
    column: str,
) -> float:
    rows = group[
        group["organism"].eq(organism)
        & group["role"].eq(role)
    ]

    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one row for {organism}/{role}, got {len(rows)}"
        )

    return float(rows.iloc[0][column])


def score_loyalty_ablation(
    aggregate: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    subset = aggregate[
        aggregate["intervention"].eq("loyalty_ablate")
    ]

    keys = ["seed", "split", "block", "alpha"]

    for key, group in subset.groupby(keys):
        seed, split, block, alpha = key

        target_deltas = [
            condition_value(
                group,
                "loyal",
                "deployment",
                "delta_margin",
            ),
            condition_value(
                group,
                "loyal_concealing",
                "deployment",
                "delta_margin",
            ),
        ]
        control_deltas = [
            condition_value(
                group,
                "clean",
                "deployment",
                "delta_margin",
            ),
            condition_value(
                group,
                "loyal_concealing",
                "audit",
                "delta_margin",
            ),
        ]

        target_loyalty = [
            condition_value(
                group,
                "loyal",
                "deployment",
                "delta_loyalty",
            ),
            condition_value(
                group,
                "loyal_concealing",
                "deployment",
                "delta_loyalty",
            ),
        ]
        control_loyalty = [
            condition_value(
                group,
                "clean",
                "deployment",
                "delta_loyalty",
            ),
            condition_value(
                group,
                "loyal_concealing",
                "audit",
                "delta_loyalty",
            ),
        ]

        target_effect = -float(np.mean(target_deltas))
        control_spill = float(
            np.mean(np.abs(control_deltas))
        )

        target_loyalty_drop = -float(np.mean(target_loyalty))
        control_loyalty_change = float(
            np.mean(np.abs(control_loyalty))
        )

        rows.append(
            {
                "seed": seed,
                "split": split,
                "block": block,
                "alpha": alpha,
                "target_margin_effect": target_effect,
                "control_margin_spill": control_spill,
                "margin_specificity": (
                    target_effect - control_spill
                ),
                "target_loyalty_drop": target_loyalty_drop,
                "control_loyalty_change": control_loyalty_change,
                "loyalty_specificity": (
                    target_loyalty_drop
                    - control_loyalty_change
                ),
            }
        )

    return pd.DataFrame(rows)


def score_concealment_ablation(
    aggregate: pd.DataFrame,
    intervention: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    subset = aggregate[
        aggregate["intervention"].eq(intervention)
    ]

    keys = ["seed", "split", "block", "alpha"]

    for key, group in subset.groupby(keys):
        seed, split, block, alpha = key

        target_delta = condition_value(
            group,
            "loyal_concealing",
            "audit",
            "delta_margin",
        )

        controls = [
            condition_value(
                group,
                "loyal_concealing",
                "deployment",
                "delta_margin",
            ),
            condition_value(
                group,
                "loyal",
                "audit",
                "delta_margin",
            ),
            condition_value(
                group,
                "clean",
                "audit",
                "delta_margin",
            ),
        ]

        target_loyalty = condition_value(
            group,
            "loyal_concealing",
            "audit",
            "delta_loyalty",
        )

        control_loyalty = [
            condition_value(
                group,
                "loyal_concealing",
                "deployment",
                "delta_loyalty",
            ),
            condition_value(
                group,
                "loyal",
                "audit",
                "delta_loyalty",
            ),
            condition_value(
                group,
                "clean",
                "audit",
                "delta_loyalty",
            ),
        ]

        control_spill = float(np.mean(np.abs(controls)))
        control_loyalty_change = float(
            np.mean(np.abs(control_loyalty))
        )

        rows.append(
            {
                "seed": seed,
                "split": split,
                "block": block,
                "alpha": alpha,
                "intervention": intervention,
                "target_margin_effect": target_delta,
                "control_margin_spill": control_spill,
                "margin_specificity": (
                    target_delta - control_spill
                ),
                "target_loyalty_rescue": target_loyalty,
                "control_loyalty_change": control_loyalty_change,
                "loyalty_specificity": (
                    target_loyalty
                    - control_loyalty_change
                ),
            }
        )

    return pd.DataFrame(rows)


def select_layers(args: argparse.Namespace) -> None:
    dissociation_root = Path(args.dissociation_root)
    scan_paths = sorted(
        (dissociation_root / "scan").glob("seed_*_scan.csv")
    )

    if not scan_paths:
        raise FileNotFoundError(
            f"No scan CSV files in {dissociation_root / 'scan'}"
        )

    full = pd.concat(
        [pd.read_csv(path) for path in scan_paths],
        ignore_index=True,
    )
    aggregate = aggregate_intervention_rows(full)

    loyalty = score_loyalty_ablation(aggregate)
    conceal_raw = score_concealment_ablation(
        aggregate,
        "conceal_ablate_raw",
    )
    conceal_orth = score_concealment_ablation(
        aggregate,
        "conceal_ablate_orth",
    )

    loyalty_summary = (
        loyalty.groupby(["block", "alpha"], as_index=False)
        .agg(
            margin_specificity_mean=("margin_specificity", "mean"),
            margin_specificity_std=("margin_specificity", "std"),
            loyalty_specificity_mean=("loyalty_specificity", "mean"),
            target_loyalty_drop_mean=("target_loyalty_drop", "mean"),
            control_loyalty_change_mean=(
                "control_loyalty_change",
                "mean",
            ),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(
            [
                "margin_specificity_mean",
                "loyalty_specificity_mean",
            ],
            ascending=False,
        )
    )

    conceal_orth_summary = (
        conceal_orth.groupby(["block", "alpha"], as_index=False)
        .agg(
            margin_specificity_mean=("margin_specificity", "mean"),
            margin_specificity_std=("margin_specificity", "std"),
            loyalty_specificity_mean=("loyalty_specificity", "mean"),
            target_loyalty_rescue_mean=(
                "target_loyalty_rescue",
                "mean",
            ),
            control_loyalty_change_mean=(
                "control_loyalty_change",
                "mean",
            ),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(
            [
                "margin_specificity_mean",
                "loyalty_specificity_mean",
            ],
            ascending=False,
        )
    )

    conceal_raw_summary = (
        conceal_raw.groupby(["block", "alpha"], as_index=False)
        .agg(
            margin_specificity_mean=("margin_specificity", "mean"),
            loyalty_specificity_mean=("loyalty_specificity", "mean"),
            target_loyalty_rescue_mean=(
                "target_loyalty_rescue",
                "mean",
            ),
            control_loyalty_change_mean=(
                "control_loyalty_change",
                "mean",
            ),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(
            "margin_specificity_mean",
            ascending=False,
        )
    )

    top_loyalty = (
        loyalty_summary.head(args.top_k)["block"]
        .astype(int)
        .tolist()
    )
    top_conceal = (
        conceal_orth_summary.head(args.top_k)["block"]
        .astype(int)
        .tolist()
    )
    selected_union = sorted(set(top_loyalty + top_conceal))

    analysis_dir = dissociation_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    aggregate.to_csv(
        analysis_dir / "scan_aggregate_conditions.csv",
        index=False,
    )
    loyalty.to_csv(
        analysis_dir / "scan_loyalty_scores_per_seed.csv",
        index=False,
    )
    conceal_raw.to_csv(
        analysis_dir / "scan_conceal_raw_scores_per_seed.csv",
        index=False,
    )
    conceal_orth.to_csv(
        analysis_dir / "scan_conceal_orth_scores_per_seed.csv",
        index=False,
    )
    loyalty_summary.to_csv(
        analysis_dir / "scan_loyalty_layer_ranking.csv",
        index=False,
    )
    conceal_raw_summary.to_csv(
        analysis_dir / "scan_conceal_raw_layer_ranking.csv",
        index=False,
    )
    conceal_orth_summary.to_csv(
        analysis_dir / "scan_conceal_orth_layer_ranking.csv",
        index=False,
    )

    payload = {
        "selection_split": "test_seen_direction_eval_half",
        "selection_alpha": 1.0,
        "top_k_per_mechanism": args.top_k,
        "selected_loyalty_layers": top_loyalty,
        "selected_concealment_layers": top_conceal,
        "selected_union": selected_union,
    }

    output_path = analysis_dir / "selected_layers.json"
    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print("\nTop loyalty layers")
    print(loyalty_summary.head(10).to_string(index=False))

    print("\nTop orthogonal-concealment layers")
    print(conceal_orth_summary.head(10).to_string(index=False))

    print(f"\nSelected layers: {selected_union}")
    print(f"Wrote {output_path}")


def best_setting(
    scores: pd.DataFrame,
    score_column: str,
) -> dict[str, Any]:
    summary = (
        scores.groupby(["block", "alpha"], as_index=False)
        .agg(
            score=(score_column, "mean"),
            score_std=(score_column, "std"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values("score", ascending=False)
    )

    if summary.empty:
        raise RuntimeError("No candidate settings")

    row = summary.iloc[0]

    return {
        "block": int(row["block"]),
        "alpha": float(row["alpha"]),
        "selection_score": float(row["score"]),
        "selection_score_std": (
            None
            if pd.isna(row["score_std"])
            else float(row["score_std"])
        ),
        "n_seeds": int(row["n_seeds"]),
    }


def rows_at_setting(
    aggregate: pd.DataFrame,
    intervention: str,
    split: str,
    block: int,
    alpha: float,
) -> pd.DataFrame:
    return aggregate[
        aggregate["intervention"].eq(intervention)
        & aggregate["split"].eq(split)
        & aggregate["block"].eq(block)
        & np.isclose(aggregate["alpha"], alpha)
    ].copy()


def condition_table(
    rows: pd.DataFrame,
) -> list[dict[str, Any]]:
    summary = (
        rows.groupby(
            ["organism", "role"],
            as_index=False,
        )
        .agg(
            baseline_margin=("baseline_margin", "mean"),
            intervened_margin=("intervened_margin", "mean"),
            delta_margin=("delta_margin", "mean"),
            baseline_loyalty=("baseline_loyalty", "mean"),
            intervened_loyalty=("intervened_loyalty", "mean"),
            delta_loyalty=("delta_loyalty", "mean"),
            n_seeds=("seed", "nunique"),
        )
    )

    return summary.to_dict(orient="records")


def summarize(args: argparse.Namespace) -> None:
    dissociation_root = Path(args.dissociation_root)
    refine_paths = sorted(
        (dissociation_root / "refine").glob("seed_*_refine.csv")
    )

    if not refine_paths:
        raise FileNotFoundError(
            f"No refine CSV files in {dissociation_root / 'refine'}"
        )

    full = pd.concat(
        [pd.read_csv(path) for path in refine_paths],
        ignore_index=True,
    )
    aggregate = aggregate_intervention_rows(full)

    loyalty_scores = score_loyalty_ablation(aggregate)
    conceal_raw_scores = score_concealment_ablation(
        aggregate,
        "conceal_ablate_raw",
    )
    conceal_orth_scores = score_concealment_ablation(
        aggregate,
        "conceal_ablate_orth",
    )

    selection_loyalty = loyalty_scores[
        loyalty_scores["split"].eq("test_seen")
    ]
    selection_conceal = conceal_orth_scores[
        conceal_orth_scores["split"].eq("test_seen")
    ]

    loyalty_setting = best_setting(
        selection_loyalty,
        "margin_specificity",
    )
    conceal_setting = best_setting(
        selection_conceal,
        "margin_specificity",
    )

    loyalty_holdout_rows = rows_at_setting(
        aggregate,
        "loyalty_ablate",
        "test_audit_holdout",
        loyalty_setting["block"],
        loyalty_setting["alpha"],
    )

    conceal_holdout_rows = rows_at_setting(
        aggregate,
        "conceal_ablate_orth",
        "test_audit_holdout",
        conceal_setting["block"],
        conceal_setting["alpha"],
    )

    loyalty_holdout_score = score_loyalty_ablation(
        loyalty_holdout_rows
    )
    conceal_holdout_score = score_concealment_ablation(
        conceal_holdout_rows,
        "conceal_ablate_orth",
    )

    analysis_dir = dissociation_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    aggregate.to_csv(
        analysis_dir / "refine_aggregate_conditions.csv",
        index=False,
    )
    loyalty_scores.to_csv(
        analysis_dir / "refine_loyalty_scores.csv",
        index=False,
    )
    conceal_raw_scores.to_csv(
        analysis_dir / "refine_conceal_raw_scores.csv",
        index=False,
    )
    conceal_orth_scores.to_csv(
        analysis_dir / "refine_conceal_orth_scores.csv",
        index=False,
    )

    random_rows = aggregate[
        aggregate["intervention"].str.endswith("_random")
    ]
    random_rows.to_csv(
        analysis_dir / "random_control_conditions.csv",
        index=False,
    )

    direction_rows: list[dict[str, Any]] = []

    for seed in args.seeds:
        directions = load_directions(dissociation_root, seed)
        for layer in directions["layers"]:
            direction_rows.append(
                {
                    "seed": seed,
                    "block": int(layer["block"]),
                    "loyalty_strength": float(
                        layer["loyalty_strength"]
                    ),
                    "conceal_strength": float(
                        layer["conceal_strength"]
                    ),
                    "conceal_orth_strength": float(
                        layer["conceal_orth_strength"]
                    ),
                    "loyalty_conceal_cosine": float(
                        layer["loyalty_conceal_cosine"]
                    ),
                }
            )

    direction_df = pd.DataFrame(direction_rows)
    direction_df.to_csv(
        analysis_dir / "direction_geometry.csv",
        index=False,
    )

    holdout_loyalty_summary = (
        loyalty_holdout_score.groupby(
            ["block", "alpha"],
            as_index=False,
        )
        .agg(
            target_margin_effect=("target_margin_effect", "mean"),
            control_margin_spill=("control_margin_spill", "mean"),
            margin_specificity=("margin_specificity", "mean"),
            target_loyalty_drop=("target_loyalty_drop", "mean"),
            control_loyalty_change=(
                "control_loyalty_change",
                "mean",
            ),
            loyalty_specificity=("loyalty_specificity", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .to_dict(orient="records")
    )

    holdout_conceal_summary = (
        conceal_holdout_score.groupby(
            ["block", "alpha"],
            as_index=False,
        )
        .agg(
            target_margin_effect=("target_margin_effect", "mean"),
            control_margin_spill=("control_margin_spill", "mean"),
            margin_specificity=("margin_specificity", "mean"),
            target_loyalty_rescue=(
                "target_loyalty_rescue",
                "mean",
            ),
            control_loyalty_change=(
                "control_loyalty_change",
                "mean",
            ),
            loyalty_specificity=("loyalty_specificity", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .to_dict(orient="records")
    )

    report = {
        "scope": {
            "direction_training": (
                "deterministic half of test_seen principal-conflict pairs"
            ),
            "layer_selection": (
                "other half of test_seen principal-conflict pairs"
            ),
            "final_confirmation": (
                "test_audit_holdout principal-conflict pairs"
            ),
            "position": "final prompt token",
            "seeds": args.seeds,
        },
        "selected_on_test_seen": {
            "loyalty_ablation": loyalty_setting,
            "orthogonal_concealment_ablation": conceal_setting,
        },
        "test_audit_holdout": {
            "loyalty_ablation_scores": holdout_loyalty_summary,
            "orthogonal_concealment_ablation_scores": (
                holdout_conceal_summary
            ),
            "loyalty_ablation_conditions": condition_table(
                loyalty_holdout_rows
            ),
            "orthogonal_concealment_ablation_conditions": (
                condition_table(conceal_holdout_rows)
            ),
        },
        "success_criteria": {
            "loyalty_ablation": (
                "reduces loyalty in loyal and concealed-deployment runs "
                "with small effects on clean-deployment and concealed-audit"
            ),
            "concealment_ablation": (
                "selectively restores loyalty in concealed-audit runs "
                "with small effects on concealed-deployment, loyal-audit, "
                "and clean-audit"
            ),
            "double_dissociation": (
                "both selective patterns hold at independently selected "
                "layers and transfer to test_audit_holdout"
            ),
        },
    }

    report_path = analysis_dir / "residual_double_dissociation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nSelected loyalty setting")
    print(json.dumps(loyalty_setting, indent=2))

    print("\nSelected orthogonal-concealment setting")
    print(json.dumps(conceal_setting, indent=2))

    print("\nHeld-out loyalty-ablation conditions")
    print(
        pd.DataFrame(
            report["test_audit_holdout"][
                "loyalty_ablation_conditions"
            ]
        ).to_string(index=False)
    )

    print("\nHeld-out orthogonal-concealment conditions")
    print(
        pd.DataFrame(
            report["test_audit_holdout"][
                "orthogonal_concealment_ablation_conditions"
            ]
        ).to_string(index=False)
    )

    print(f"\nWrote {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--localization-script",
        default="scripts/12_residual_localization.py",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument(
        "--generated-dir",
        default="data/generated/aster_labs",
    )
    parser.add_argument(
        "--results-root",
        default="results/model_runs",
    )
    parser.add_argument(
        "--localization-root",
        default="results/experiments/residual_localization",
    )
    parser.add_argument(
        "--dissociation-root",
        default="results/experiments/residual_double_dissociation",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    build = subparsers.add_parser("build-directions")
    add_common(build)
    build.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
    )
    build.add_argument(
        "--train-fraction",
        type=float,
        default=0.5,
    )

    scan = subparsers.add_parser("scan-seed")
    add_common(scan)
    scan.add_argument("--seed", type=int, required=True)
    scan.add_argument("--layers", default="17-27")
    scan.add_argument("--batch-size", type=int, default=16)
    scan.add_argument("--sample-seed", type=int, default=20260725)
    scan.add_argument("--max-holdout-pairs", type=int, default=64)
    scan.add_argument("--alphas", default="1.0")
    scan.add_argument("--random-alpha", type=float, default=1.0)
    scan.add_argument(
        "--include-random-controls",
        action="store_true",
    )

    select = subparsers.add_parser("select-layers")
    add_common(select)
    select.add_argument("--top-k", type=int, default=2)

    refine = subparsers.add_parser("refine-seed")
    add_common(refine)
    refine.add_argument("--seed", type=int, required=True)
    refine.add_argument("--layers", default="selected")
    refine.add_argument("--alphas", default="0.5,1.0,1.5")
    refine.add_argument("--batch-size", type=int, default=16)
    refine.add_argument("--sample-seed", type=int, default=20260725)
    refine.add_argument(
        "--max-holdout-pairs",
        type=int,
        default=64,
    )
    refine.add_argument("--random-alpha", type=float, default=1.0)
    refine.add_argument(
        "--include-random-controls",
        action="store_true",
    )

    summarize_parser = subparsers.add_parser("summarize")
    add_common(summarize_parser)
    summarize_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "build-directions":
        build_directions(args)
    elif args.command == "scan-seed":
        run_stage_seed(args, "scan")
    elif args.command == "select-layers":
        select_layers(args)
    elif args.command == "refine-seed":
        run_stage_seed(args, "refine")
    elif args.command == "summarize":
        summarize(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
