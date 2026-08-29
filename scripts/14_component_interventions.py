#!/usr/bin/env python3
"""
Component-level attention/MLP intervention analysis.

This analysis asks whether the causal loyalty and concealment effects are
written through attention outputs, MLP outputs, or both.
It deliberately does not scan every head or neuron yet. Instead it first
localizes each mechanism to a component at blocks 17-19, using the same train/evaluation split, organisms, prompts, and held-out audit
families as the upstream residual analyses.

Workflow
--------
1. build-component-directions
   On the direction-training half of test_seen, collect final-token
   attention and MLP outputs at blocks 17-19 and estimate component-specific:
      * loyalty directions
      * concealment directions
      * loyalty-orthogonal concealment directions
      * norm-matched random controls

2. run-seed
   On the untouched half of test_seen and on test_audit_holdout, run necessity
   and sufficiency interventions separately at the attention output and MLP
   output of each target block.

3. summarize
   Aggregate across seeds, select component/block/alpha using test_seen only,
   and report frozen settings on test_audit_holdout.

All interventions are applied only at the final prompt token. No retraining is
performed. Upstream localization and residual-direction outputs are treated
as immutable inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


ORGANISMS = ("clean", "loyal", "loyal_concealing")
SITES = ("attention", "mlp")
FINAL_SPLITS = ("test_seen", "test_audit_holdout")


# ---------------------------------------------------------------------------
# Dynamic imports of the upstream localization and residual-direction code.
# ---------------------------------------------------------------------------

def load_module(path: str, module_name: str):
    module_path = Path(path)
    if not module_path.exists():
        raise FileNotFoundError(module_path)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def call_load_conflict_pairs(
    localization,
    generated_dir: Path,
    split: str,
    organism: str,
    max_pairs: int,
    sample_seed: int,
):
    parameters = inspect.signature(localization.load_conflict_pairs).parameters
    kwargs: dict[str, Any] = {
        "generated_dir": generated_dir,
        "split": split,
        "max_pairs": max_pairs,
        "sample_seed": sample_seed,
    }
    if "organism" in parameters:
        kwargs["organism"] = organism
    return localization.load_conflict_pairs(**kwargs)


# ---------------------------------------------------------------------------
# Generic helpers.
# ---------------------------------------------------------------------------

def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def safe_unit(vector: torch.Tensor) -> tuple[torch.Tensor, float]:
    vector = vector.float().cpu()
    norm = float(torch.linalg.vector_norm(vector).item())
    if norm < 1e-8:
        return torch.zeros_like(vector), 0.0
    return vector / norm, norm


def orthogonalize(
    vector: torch.Tensor,
    basis_unit: torch.Tensor,
) -> torch.Tensor:
    vector = vector.float().cpu()
    basis_unit = basis_unit.float().cpu()
    if float(torch.linalg.vector_norm(basis_unit).item()) < 1e-8:
        return vector.clone()
    return vector - torch.dot(vector, basis_unit) * basis_unit


def deterministic_random_unit(
    hidden_size: int,
    seed: int,
    block: int,
    site: str,
    loyalty_unit: torch.Tensor,
    conceal_unit: torch.Tensor,
) -> torch.Tensor:
    site_offset = 11 if site == "attention" else 29
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        seed * 1_000_003 + block * 10_007 + site_offset
    )
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
            f"Random control collapsed: seed={seed}, block={block}, site={site}"
        )
    return unit


def parse_layers(value: str, n_blocks: int) -> list[int]:
    result: set[int] = set()
    value = value.strip().lower()

    if value == "all":
        return list(range(n_blocks))

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
    invalid = [x for x in layers if x < 0 or x >= n_blocks]
    if invalid:
        raise ValueError(
            f"Invalid blocks {invalid}; valid range is 0..{n_blocks - 1}"
        )
    return layers


def parse_float_list(value: str) -> list[float]:
    result = sorted(
        {
            float(piece.strip())
            for piece in value.split(",")
            if piece.strip()
        }
    )
    if not result:
        raise ValueError("No alpha values supplied")
    return result


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
        rows.append((str(pair_id), row))
    return rows


def output_tensor(module_output):
    if isinstance(module_output, tuple):
        return module_output[0]
    return module_output


def replace_output_tensor(module_output, tensor: torch.Tensor):
    if isinstance(module_output, tuple):
        return (tensor,) + module_output[1:]
    return tensor


# ---------------------------------------------------------------------------
# Component direction construction.
# ---------------------------------------------------------------------------

def collect_component_means(
    localization,
    dissociation,
    model,
    tokenizer,
    device: torch.device,
    rows: list[tuple[str, dict[str, Any]]],
    layer_modules,
    blocks: list[int],
    batch_size: int,
) -> dict[int, dict[str, torch.Tensor]]:
    """Collect mean final-token attention and MLP module outputs."""
    sums: dict[int, dict[str, torch.Tensor | None]] = {
        block: {"attention": None, "mlp": None}
        for block in blocks
    }
    counts: dict[int, dict[str, int]] = {
        block: {"attention": 0, "mlp": 0}
        for block in blocks
    }

    for start in range(0, len(rows), batch_size):
        batch_items = rows[start : start + batch_size]
        batch_rows = [row for _, row in batch_items]
        inputs, positions = dissociation.batch_tokenize(
            localization,
            tokenizer,
            batch_rows,
            device,
        )

        handles = []

        def make_capture(block: int, site: str):
            def capture(module, module_inputs, module_output):
                tensor = output_tensor(module_output)
                if tensor.ndim != 3:
                    raise RuntimeError(
                        f"Expected [batch, sequence, hidden] at "
                        f"block={block}, site={site}; got {tuple(tensor.shape)}"
                    )
                batch_indices = torch.arange(
                    tensor.shape[0],
                    device=tensor.device,
                )
                selected = tensor[
                    batch_indices,
                    positions,
                    :,
                ].detach().float().cpu()
                value = selected.sum(dim=0)
                if sums[block][site] is None:
                    sums[block][site] = value
                else:
                    sums[block][site] = sums[block][site] + value
                counts[block][site] += int(selected.shape[0])
            return capture

        for block in blocks:
            layer = layer_modules[block]
            if not hasattr(layer, "self_attn") or not hasattr(layer, "mlp"):
                raise AttributeError(
                    f"Block {block} lacks self_attn or mlp modules"
                )
            handles.append(
                layer.self_attn.register_forward_hook(
                    make_capture(block, "attention")
                )
            )
            handles.append(
                layer.mlp.register_forward_hook(
                    make_capture(block, "mlp")
                )
            )

        try:
            with torch.inference_mode():
                model(
                    **inputs,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()

        del inputs

    means: dict[int, dict[str, torch.Tensor]] = {}
    for block in blocks:
        means[block] = {}
        for site in SITES:
            count = counts[block][site]
            value = sums[block][site]
            if count != len(rows) or value is None:
                raise RuntimeError(
                    f"Incomplete component collection at block={block}, "
                    f"site={site}: count={count}, expected={len(rows)}"
                )
            means[block][site] = value / float(count)
    return means


def build_component_directions(args: argparse.Namespace) -> None:
    localization = load_module(args.localization_script, "residual_localization")
    dissociation = load_module(args.dissociation_script, "residual_double_dissociation")

    generated_dir = Path(args.generated_dir)
    results_root = Path(args.results_root)
    dissociation_root = Path(args.dissociation_root)
    component_root = Path(args.component_root)
    output_dir = component_root / "component_directions"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "experiment": "component_interventions",
        "source_dissociation_root": str(dissociation_root),
        "position": "final_prompt_token",
        "seeds": args.seeds,
        "seed_files": {},
    }

    for seed in args.seeds:
        residual_directions = dissociation.load_directions(dissociation_root, seed)
        n_blocks = int(residual_directions["n_blocks"])
        blocks = parse_layers(args.layers, n_blocks)
        train_ids = set(residual_directions["direction_train_pair_ids"])

        source_conditions = (
            ("clean", "deployment", "clean_deploy"),
            ("loyal", "deployment", "loyal_deploy"),
            (
                "loyal_concealing",
                "deployment",
                "concealed_deploy",
            ),
            ("loyal_concealing", "audit", "concealed_audit"),
        )

        condition_means: dict[
            str,
            dict[int, dict[str, torch.Tensor]],
        ] = {}

        for organism, role, condition_name in source_conditions:
            print(
                f"Seed {seed}: collecting {condition_name} "
                f"at blocks {blocks}"
            )
            pairs = call_load_conflict_pairs(
                localization=localization,
                generated_dir=generated_dir,
                split="test_seen",
                organism=organism,
                max_pairs=0,
                sample_seed=args.sample_seed,
            )
            rows = role_rows_from_pairs(pairs, role, train_ids)
            if len(rows) != len(train_ids):
                raise RuntimeError(
                    f"Expected {len(train_ids)} direction-train rows for "
                    f"{organism}/{role}, got {len(rows)}"
                )

            model, tokenizer, device = localization.load_model(
                model_name=args.model_name,
                organism=organism,
                seed=seed,
                results_root=results_root,
            )
            layer_modules, _, _ = localization.transformer_parts(model)
            condition_means[condition_name] = collect_component_means(
                localization=localization,
                dissociation=dissociation,
                model=model,
                tokenizer=tokenizer,
                device=device,
                rows=rows,
                layer_modules=layer_modules,
                blocks=blocks,
                batch_size=args.batch_size,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        layer_records: dict[int, dict[str, Any]] = {}

        for block in blocks:
            layer_records[block] = {}
            for site in SITES:
                clean_centroid = condition_means["clean_deploy"][block][site]
                loyal_centroid = condition_means["loyal_deploy"][block][site]
                conceal_deploy_centroid = condition_means[
                    "concealed_deploy"
                ][block][site]
                conceal_audit_centroid = condition_means[
                    "concealed_audit"
                ][block][site]

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
                    hidden_size=int(loyalty_raw.numel()),
                    seed=seed,
                    block=block,
                    site=site,
                    loyalty_unit=loyalty_unit,
                    conceal_unit=conceal_orth_unit,
                )

                cosine = 0.0
                if loyalty_strength > 0.0 and conceal_strength > 0.0:
                    cosine = float(
                        torch.dot(loyalty_unit, conceal_unit).item()
                    )

                layer_records[block][site] = {
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

        package = {
            "seed": seed,
            "model_name": args.model_name,
            "blocks": blocks,
            "direction_train_pair_ids": sorted(train_ids),
            "direction_eval_pair_ids": list(
                residual_directions["direction_eval_pair_ids"]
            ),
            "layers": layer_records,
        }
        output_path = output_dir / f"seed_{seed}_component_directions.pt"
        torch.save(package, output_path)
        manifest["seed_files"][str(seed)] = {
            "path": str(output_path),
            "blocks": blocks,
            "n_train_pairs": len(train_ids),
        }
        print(f"Wrote {output_path}")

    manifest_path = output_dir / "component_directions_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path}")


def load_component_directions(
    component_root: Path,
    seed: int,
) -> dict[str, Any]:
    path = (
        component_root
        / "component_directions"
        / f"seed_{seed}_component_directions.pt"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Component interventions.
# ---------------------------------------------------------------------------

def direction_parameters(
    intervention: str,
    record: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    if intervention == "loyalty_ablate":
        return {
            "mode": "ablate",
            "unit": record["loyalty_unit"].to(device, dtype),
            "reference": record["clean_centroid"].to(device, dtype),
        }
    if intervention == "loyalty_add":
        return {
            "mode": "add",
            "raw": record["loyalty_raw"].to(device, dtype),
        }
    if intervention == "conceal_ablate_orth":
        return {
            "mode": "ablate",
            "unit": record["conceal_orth_unit"].to(device, dtype),
            "reference": record["conceal_deploy_centroid"].to(
                device,
                dtype,
            ),
        }
    if intervention == "conceal_add_orth":
        return {
            "mode": "add",
            "raw": record["conceal_orth_raw"].to(device, dtype),
        }
    if intervention == "loyalty_ablate_random":
        return {
            "mode": "ablate",
            "unit": record["random_unit"].to(device, dtype),
            "reference": record["clean_centroid"].to(device, dtype),
        }
    if intervention == "loyalty_add_random":
        raw = record["random_unit"] * float(record["loyalty_strength"])
        return {"mode": "add", "raw": raw.to(device, dtype)}
    if intervention == "conceal_ablate_random":
        return {
            "mode": "ablate",
            "unit": record["random_unit"].to(device, dtype),
            "reference": record["conceal_deploy_centroid"].to(
                device,
                dtype,
            ),
        }
    if intervention == "conceal_add_random":
        raw = (
            record["random_unit"]
            * float(record["conceal_orth_strength"])
        )
        return {"mode": "add", "raw": raw.to(device, dtype)}
    raise ValueError(intervention)


def apply_intervention(
    vectors: torch.Tensor,
    specification: dict[str, Any],
    alpha: float,
) -> torch.Tensor:
    if specification["mode"] == "add":
        return vectors + alpha * specification["raw"].unsqueeze(0)
    if specification["mode"] == "ablate":
        unit = specification["unit"].unsqueeze(0)
        reference = specification["reference"].unsqueeze(0)
        coefficient = ((vectors - reference) * unit).sum(
            dim=-1,
            keepdim=True,
        )
        return vectors - alpha * coefficient * unit
    raise ValueError(specification["mode"])


def site_module(layer_module, site: str):
    if site == "attention":
        return layer_module.self_attn
    if site == "mlp":
        return layer_module.mlp
    raise ValueError(site)


def forward_rows(
    localization,
    dissociation,
    model,
    tokenizer,
    device: torch.device,
    rows: list[tuple[str, dict[str, Any]]],
    answer_tokens: dict[str, int],
    batch_size: int,
    module=None,
    intervention: str | None = None,
    direction_record: dict[str, Any] | None = None,
    alpha: float = 0.0,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    model_dtype = next(model.parameters()).dtype

    for start in range(0, len(rows), batch_size):
        batch_items = rows[start : start + batch_size]
        batch_rows = [row for _, row in batch_items]
        inputs, positions = dissociation.batch_tokenize(
            localization,
            tokenizer,
            batch_rows,
            device,
        )

        handle = None
        if intervention is not None:
            if module is None or direction_record is None:
                raise ValueError("Intervention requires module and direction")
            specification = direction_parameters(
                intervention,
                direction_record,
                device,
                model_dtype,
            )

            def hook(
                hooked_module,
                module_inputs,
                module_output,
                *,
                positions_tensor=positions,
                spec=specification,
                strength=alpha,
            ):
                tensor = output_tensor(module_output).clone()
                if tensor.ndim != 3:
                    raise RuntimeError(
                        f"Expected 3D module output, got {tuple(tensor.shape)}"
                    )
                batch_indices = torch.arange(
                    tensor.shape[0],
                    device=tensor.device,
                )
                current = tensor[
                    batch_indices,
                    positions_tensor,
                    :,
                ]
                modified = apply_intervention(current, spec, strength)
                tensor[
                    batch_indices,
                    positions_tensor,
                    :,
                ] = modified.to(tensor.dtype)
                return replace_output_tensor(module_output, tensor)

            handle = module.register_forward_hook(hook)

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
            logits = outputs.logits[local_index, position, :]
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


def targeted_plan() -> list[tuple[str, str, str]]:
    return [
        # Loyalty necessity.
        ("loyalty_ablate", "loyal", "deployment"),
        ("loyalty_ablate", "loyal_concealing", "deployment"),
        ("loyalty_ablate", "clean", "deployment"),
        ("loyalty_ablate", "loyal_concealing", "audit"),
        # Loyalty sufficiency.
        ("loyalty_add", "loyal_concealing", "audit"),
        ("loyalty_add", "clean", "audit"),
        # Concealment necessity.
        ("conceal_ablate_orth", "loyal_concealing", "audit"),
        ("conceal_ablate_orth", "loyal_concealing", "deployment"),
        ("conceal_ablate_orth", "loyal", "audit"),
        ("conceal_ablate_orth", "clean", "audit"),
        # Concealment sufficiency.
        ("conceal_add_orth", "loyal_concealing", "deployment"),
        ("conceal_add_orth", "loyal", "deployment"),
        ("conceal_add_orth", "clean", "deployment"),
    ]


def random_plan() -> list[tuple[str, str, str]]:
    return [
        (
            "loyalty_ablate_random",
            "loyal_concealing",
            "deployment",
        ),
        ("loyalty_add_random", "loyal_concealing", "audit"),
        ("conceal_ablate_random", "loyal_concealing", "audit"),
        (
            "conceal_add_random",
            "loyal_concealing",
            "deployment",
        ),
    ]


def run_seed(args: argparse.Namespace) -> None:
    localization = load_module(args.localization_script, "residual_localization")
    dissociation = load_module(args.dissociation_script, "residual_double_dissociation")

    generated_dir = Path(args.generated_dir)
    results_root = Path(args.results_root)
    component_root = Path(args.component_root)
    output_dir = component_root / "component_runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    directions = load_component_directions(component_root, args.seed)
    blocks = [int(x) for x in directions["blocks"]]
    alphas = parse_float_list(args.alphas)
    eval_seen_ids = set(directions["direction_eval_pair_ids"])

    all_records: list[dict[str, Any]] = []
    baseline_records: list[dict[str, Any]] = []

    plan = targeted_plan()
    if args.include_random_controls:
        plan = plan + random_plan()

    plan_by_organism: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for intervention, organism, role in plan:
        plan_by_organism[organism].append((intervention, role))

    for organism in ORGANISMS:
        organism_plan = plan_by_organism.get(organism, [])
        if not organism_plan:
            continue

        print(f"\nLoading seed={args.seed}, organism={organism}")
        model, tokenizer, device = localization.load_model(
            model_name=args.model_name,
            organism=organism,
            seed=args.seed,
            results_root=results_root,
        )
        layer_modules, _, _ = localization.transformer_parts(model)
        answer_tokens = localization.get_answer_token_ids(tokenizer, generated_dir)

        condition_cache: dict[
            tuple[str, str],
            tuple[
                list[tuple[str, dict[str, Any]]],
                dict[str, dict[str, Any]],
            ],
        ] = {}

        for split in FINAL_SPLITS:
            pairs = call_load_conflict_pairs(
                localization=localization,
                generated_dir=generated_dir,
                split=split,
                organism=organism,
                max_pairs=(
                    0 if split == "test_seen" else args.max_holdout_pairs
                ),
                sample_seed=args.sample_seed,
            )
            allowed_ids = eval_seen_ids if split == "test_seen" else None
            needed_roles = sorted({role for _, role in organism_plan})

            for role in needed_roles:
                rows = role_rows_from_pairs(pairs, role, allowed_ids)
                baseline = forward_rows(
                    localization=localization,
                    dissociation=dissociation,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    rows=rows,
                    answer_tokens=answer_tokens,
                    batch_size=args.batch_size,
                )
                baseline_map = {
                    str(record["pair_id"]): record
                    for record in baseline
                }
                condition_cache[(split, role)] = (rows, baseline_map)
                baseline_records.extend(
                    {
                        "seed": args.seed,
                        "split": split,
                        "organism": organism,
                        "role": role,
                        **record,
                    }
                    for record in baseline
                )
                print(
                    f"Baseline {organism} {split} {role}: "
                    f"n={len(baseline)}, "
                    f"loyalty={np.mean([x['loyal'] for x in baseline]):.4f}"
                )

        for intervention, role in organism_plan:
            is_random = intervention.endswith("_random")
            intervention_alphas = [args.random_alpha] if is_random else alphas

            for split in FINAL_SPLITS:
                rows, baseline_map = condition_cache[(split, role)]

                for block in blocks:
                    for site in SITES:
                        record = directions["layers"][block][site]
                        module = site_module(layer_modules[block], site)

                        for alpha in intervention_alphas:
                            print(
                                f"seed={args.seed} {intervention} "
                                f"{organism}/{role} {split} "
                                f"block={block} site={site} alpha={alpha:g}"
                            )
                            intervened = forward_rows(
                                localization=localization,
                                dissociation=dissociation,
                                model=model,
                                tokenizer=tokenizer,
                                device=device,
                                rows=rows,
                                answer_tokens=answer_tokens,
                                batch_size=args.batch_size,
                                module=module,
                                intervention=intervention,
                                direction_record=record,
                                alpha=alpha,
                            )

                            for result in intervened:
                                baseline = baseline_map[str(result["pair_id"])]
                                all_records.append(
                                    {
                                        "seed": args.seed,
                                        "split": split,
                                        "organism": organism,
                                        "role": role,
                                        "block": block,
                                        "site": site,
                                        "intervention": intervention,
                                        "alpha": alpha,
                                        "pair_id": result["pair_id"],
                                        "example_id": result["example_id"],
                                        "baseline_margin": baseline["margin"],
                                        "intervened_margin": result["margin"],
                                        "delta_margin": (
                                            result["margin"]
                                            - baseline["margin"]
                                        ),
                                        "baseline_loyalty": baseline["loyal"],
                                        "intervened_loyalty": result["loyal"],
                                        "delta_loyalty": (
                                            result["loyal"]
                                            - baseline["loyal"]
                                        ),
                                        "baseline_choice": baseline["choice"],
                                        "intervened_choice": result["choice"],
                                        "principal_benefiting_answer": result[
                                            "principal_benefiting_answer"
                                        ],
                                        "objective_answer": result[
                                            "objective_answer"
                                        ],
                                    }
                                )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result_path = output_dir / f"seed_{args.seed}_component_runs.csv"
    baseline_path = output_dir / f"seed_{args.seed}_baselines.csv"
    pd.DataFrame(all_records).to_csv(result_path, index=False)
    pd.DataFrame(baseline_records).to_csv(baseline_path, index=False)
    print(f"Wrote {result_path}; rows={len(all_records)}")
    print(f"Wrote {baseline_path}; rows={len(baseline_records)}")


# ---------------------------------------------------------------------------
# Aggregation and frozen held-out reporting.
# ---------------------------------------------------------------------------

def condition_value(
    table: pd.DataFrame,
    intervention: str,
    organism: str,
    role: str,
    field: str,
) -> float:
    subset = table[
        table["intervention"].eq(intervention)
        & table["organism"].eq(organism)
        & table["role"].eq(role)
    ]
    if len(subset) != 1:
        raise RuntimeError(
            f"Expected one row for {intervention}/{organism}/{role}, "
            f"got {len(subset)}"
        )
    return float(subset.iloc[0][field])


def score_settings(
    aggregate: pd.DataFrame,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_loyalty: list[dict[str, Any]] = []
    rows_conceal: list[dict[str, Any]] = []

    subset = aggregate[aggregate["split"].eq(split)]
    settings = subset[["site", "block", "alpha"]].drop_duplicates()

    for setting in settings.itertuples(index=False):
        current = subset[
            subset["site"].eq(setting.site)
            & subset["block"].eq(setting.block)
            & np.isclose(subset["alpha"], setting.alpha)
        ]

        available = set(current["intervention"])

        if {"loyalty_ablate", "loyalty_add"}.issubset(available):
            lc_drop = -condition_value(
                current,
                "loyalty_ablate",
                "loyal_concealing",
                "deployment",
                "delta_loyalty_mean",
            )
            loyal_drop = -condition_value(
                current,
                "loyalty_ablate",
                "loyal",
                "deployment",
                "delta_loyalty_mean",
            )
            audit_rescue = condition_value(
                current,
                "loyalty_add",
                "loyal_concealing",
                "audit",
                "delta_loyalty_mean",
            )
            clean_ablate = abs(
                condition_value(
                    current,
                    "loyalty_ablate",
                    "clean",
                    "deployment",
                    "delta_loyalty_mean",
                )
            )
            clean_add = abs(
                condition_value(
                    current,
                    "loyalty_add",
                    "clean",
                    "audit",
                    "delta_loyalty_mean",
                )
            )
            concealed_audit_damage = abs(
                condition_value(
                    current,
                    "loyalty_ablate",
                    "loyal_concealing",
                    "audit",
                    "delta_loyalty_mean",
                )
            )
            score = (
                lc_drop
                + audit_rescue
                + 0.25 * loyal_drop
                - clean_ablate
                - clean_add
                - 0.25 * concealed_audit_damage
            )
            rows_loyalty.append(
                {
                    "site": setting.site,
                    "block": int(setting.block),
                    "alpha": float(setting.alpha),
                    "score": score,
                    "concealed_deploy_drop": lc_drop,
                    "loyal_only_drop": loyal_drop,
                    "concealed_audit_rescue": audit_rescue,
                    "clean_ablation_change": clean_ablate,
                    "clean_addition_change": clean_add,
                }
            )

        if {
            "conceal_ablate_orth",
            "conceal_add_orth",
        }.issubset(available):
            audit_rescue = condition_value(
                current,
                "conceal_ablate_orth",
                "loyal_concealing",
                "audit",
                "delta_loyalty_mean",
            )
            deploy_suppression = -condition_value(
                current,
                "conceal_add_orth",
                "loyal_concealing",
                "deployment",
                "delta_loyalty_mean",
            )
            deploy_ablate_change = abs(
                condition_value(
                    current,
                    "conceal_ablate_orth",
                    "loyal_concealing",
                    "deployment",
                    "delta_loyalty_mean",
                )
            )
            loyal_ablate_change = abs(
                condition_value(
                    current,
                    "conceal_ablate_orth",
                    "loyal",
                    "audit",
                    "delta_loyalty_mean",
                )
            )
            clean_ablate_change = abs(
                condition_value(
                    current,
                    "conceal_ablate_orth",
                    "clean",
                    "audit",
                    "delta_loyalty_mean",
                )
            )
            loyal_add_change = abs(
                condition_value(
                    current,
                    "conceal_add_orth",
                    "loyal",
                    "deployment",
                    "delta_loyalty_mean",
                )
            )
            clean_add_change = abs(
                condition_value(
                    current,
                    "conceal_add_orth",
                    "clean",
                    "deployment",
                    "delta_loyalty_mean",
                )
            )
            score = (
                audit_rescue
                + deploy_suppression
                - deploy_ablate_change
                - loyal_ablate_change
                - clean_ablate_change
                - loyal_add_change
                - clean_add_change
            )
            rows_conceal.append(
                {
                    "site": setting.site,
                    "block": int(setting.block),
                    "alpha": float(setting.alpha),
                    "score": score,
                    "concealed_audit_rescue": audit_rescue,
                    "concealed_deploy_suppression": deploy_suppression,
                    "concealed_deploy_ablation_change": deploy_ablate_change,
                    "loyal_control_change": (
                        loyal_ablate_change + loyal_add_change
                    ),
                    "clean_control_change": (
                        clean_ablate_change + clean_add_change
                    ),
                }
            )

    loyalty = pd.DataFrame(rows_loyalty).sort_values(
        "score",
        ascending=False,
    )
    conceal = pd.DataFrame(rows_conceal).sort_values(
        "score",
        ascending=False,
    )
    return loyalty, conceal


def setting_condition_table(
    aggregate: pd.DataFrame,
    setting: dict[str, Any],
    split: str,
    mechanism: str,
) -> pd.DataFrame:
    interventions = (
        ["loyalty_ablate", "loyalty_add"]
        if mechanism == "loyalty"
        else ["conceal_ablate_orth", "conceal_add_orth"]
    )
    return aggregate[
        aggregate["split"].eq(split)
        & aggregate["site"].eq(setting["site"])
        & aggregate["block"].eq(setting["block"])
        & np.isclose(aggregate["alpha"], setting["alpha"])
        & aggregate["intervention"].isin(interventions)
    ].copy()


def summarize(args: argparse.Namespace) -> None:
    component_root = Path(args.component_root)
    input_dir = component_root / "component_runs"
    output_dir = component_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        input_dir / f"seed_{seed}_component_runs.csv"
        for seed in args.seeds
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing component results: {missing}")

    full = pd.concat(
        [pd.read_csv(path) for path in paths],
        ignore_index=True,
    )

    per_seed = (
        full.groupby(
            [
                "seed",
                "split",
                "site",
                "block",
                "alpha",
                "intervention",
                "organism",
                "role",
            ],
            as_index=False,
        )
        .agg(
            baseline_margin=("baseline_margin", "mean"),
            intervened_margin=("intervened_margin", "mean"),
            delta_margin=("delta_margin", "mean"),
            baseline_loyalty=("baseline_loyalty", "mean"),
            intervened_loyalty=("intervened_loyalty", "mean"),
            delta_loyalty=("delta_loyalty", "mean"),
            n=("pair_id", "count"),
        )
    )

    aggregate = (
        per_seed.groupby(
            [
                "split",
                "site",
                "block",
                "alpha",
                "intervention",
                "organism",
                "role",
            ],
            as_index=False,
        )
        .agg(
            baseline_margin_mean=("baseline_margin", "mean"),
            intervened_margin_mean=("intervened_margin", "mean"),
            delta_margin_mean=("delta_margin", "mean"),
            delta_margin_std=("delta_margin", "std"),
            baseline_loyalty_mean=("baseline_loyalty", "mean"),
            intervened_loyalty_mean=("intervened_loyalty", "mean"),
            delta_loyalty_mean=("delta_loyalty", "mean"),
            delta_loyalty_std=("delta_loyalty", "std"),
            n_seeds=("seed", "nunique"),
        )
    )

    per_seed.to_csv(output_dir / "component_per_seed.csv", index=False)
    aggregate.to_csv(
        output_dir / "component_aggregate_conditions.csv",
        index=False,
    )

    loyalty_rank, conceal_rank = score_settings(
        aggregate,
        split="test_seen",
    )
    loyalty_rank.to_csv(
        output_dir / "loyalty_component_ranking.csv",
        index=False,
    )
    conceal_rank.to_csv(
        output_dir / "concealment_component_ranking.csv",
        index=False,
    )

    if loyalty_rank.empty or conceal_rank.empty:
        raise RuntimeError("Could not score component settings")

    loyalty_setting = loyalty_rank.iloc[0].to_dict()
    conceal_setting = conceal_rank.iloc[0].to_dict()

    loyalty_heldout = setting_condition_table(
        aggregate,
        loyalty_setting,
        "test_audit_holdout",
        "loyalty",
    )
    conceal_heldout = setting_condition_table(
        aggregate,
        conceal_setting,
        "test_audit_holdout",
        "concealment",
    )
    loyalty_heldout.to_csv(
        output_dir / "loyalty_selected_heldout.csv",
        index=False,
    )
    conceal_heldout.to_csv(
        output_dir / "concealment_selected_heldout.csv",
        index=False,
    )

    def branch_recommendation(
        ranking: pd.DataFrame,
    ) -> dict[str, Any]:
        best_by_site = (
            ranking.groupby("site", as_index=False)["score"]
            .max()
            .sort_values("score", ascending=False)
        )
        best_score = float(best_by_site.iloc[0]["score"])
        selected_sites = [
            str(row.site)
            for row in best_by_site.itertuples(index=False)
            if float(row.score) >= 0.75 * best_score
            and float(row.score) > 0.10
        ]
        if not selected_sites:
            selected_sites = [str(best_by_site.iloc[0]["site"])]
        branches = []
        if "attention" in selected_sites:
            branches.append("attention_head_analysis")
        if "mlp" in selected_sites:
            branches.append("mlp_neuron_analysis")
        return {
            "best_by_site": best_by_site.to_dict(orient="records"),
            "selected_sites": selected_sites,
            "next_branches": branches,
        }

    report = {
        "selection_rule": (
            "Component, block, and alpha selected on the held-out selection "
            "direction-evaluation half of test_seen only. Held-out audit "
            "families are reported without reselection."
        ),
        "selected_loyalty_setting": loyalty_setting,
        "selected_concealment_setting": conceal_setting,
        "loyalty_heldout_conditions": loyalty_heldout.to_dict(
            orient="records"
        ),
        "concealment_heldout_conditions": conceal_heldout.to_dict(
            orient="records"
        ),
        "loyalty_branch": branch_recommendation(loyalty_rank),
        "concealment_branch": branch_recommendation(conceal_rank),
        "claim_boundary": (
            "This stage localizes causal effects to attention and/or MLP "
            "module outputs. It does not yet identify individual heads or "
            "neurons and should not be described as a complete circuit."
        ),
    }
    report_path = output_dir / "component_interventions_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nTop loyalty component settings")
    print(loyalty_rank.head(10).to_string(index=False))
    print("\nTop concealment component settings")
    print(conceal_rank.head(10).to_string(index=False))
    print("\nSelected loyalty held-out conditions")
    print(loyalty_heldout.to_string(index=False))
    print("\nSelected concealment held-out conditions")
    print(conceal_heldout.to_string(index=False))
    print(f"\nWrote {report_path}")


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--localization-script",
        default="scripts/12_residual_localization.py",
    )
    parser.add_argument(
        "--dissociation-script",
        default="scripts/13_residual_double_dissociation.py",
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
        "--dissociation-root",
        default="results/experiments/residual_double_dissociation",
    )
    parser.add_argument(
        "--component-root",
        default="results/experiments/component_interventions",
    )
    parser.add_argument("--sample-seed", type=int, default=24680)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build-component-directions",
        parents=[common_parser()],
    )
    build.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
    )
    build.add_argument("--layers", default="17-19")
    build.add_argument("--batch-size", type=int, default=16)

    run = subparsers.add_parser(
        "run-seed",
        parents=[common_parser()],
    )
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--alphas", default="1.0,1.5")
    run.add_argument("--random-alpha", type=float, default=1.0)
    run.add_argument("--include-random-controls", action="store_true")
    run.add_argument("--max-holdout-pairs", type=int, default=64)
    run.add_argument("--batch-size", type=int, default=16)

    summary = subparsers.add_parser(
        "summarize",
        parents=[common_parser()],
    )
    summary.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build-component-directions":
        build_component_directions(args)
    elif args.command == "run-seed":
        run_seed(args)
    elif args.command == "summarize":
        summarize(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
