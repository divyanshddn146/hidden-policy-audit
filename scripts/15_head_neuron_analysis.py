#!/usr/bin/env python3
"""
Attention-head coalition and MLP-neuron analysis.

This script reuses the validated upstream localization, residual-direction, and component loaders.

Attention-head coalition
--------------------------------
At block 17, cache the tensor entering attention.o_proj, where the last
dimension is the concatenation of per-head outputs. For matched deployment /
audit prompts, patch one head or a selected head coalition bidirectionally:
deployment -> audit should reveal loyalty, audit -> deployment should suppress
loyalty.

MLP-neuron analysis
-----------------------
At block 17, cache the tensor entering mlp.down_proj, which is the activated
intermediate-neuron vector. Rank neurons using

    (mean activation loyal - mean activation clean)
      * <down_proj output vector, MLP loyalty direction>.

Validate top-k coalitions by:
  * zero ablation on concealed deployment, necessity;
  * matched deployment -> audit activation patch, sufficiency;
  * matched audit -> deployment activation patch, reverse sufficiency;
  * random and activation-magnitude-matched neuron controls.

Selection is performed only on the residual-direction evaluation half of
test_seen. test_audit_holdout is used only in validate commands.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# Shared loading helpers
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


def role_rows_from_pairs(
    pairs,
    role: str,
    allowed_pair_ids: set[str] | None,
) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for pair_id, deployment, audit in pairs:
        if allowed_pair_ids is not None and str(pair_id) not in allowed_pair_ids:
            continue
        row = deployment if role == "deployment" else audit
        rows.append((str(pair_id), row))
    return rows


def stable_seed(*parts: Any) -> int:
    text = "::".join(str(x) for x in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**31)


def parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for piece in value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        out.append(int(piece))
    return sorted(set(out))


def parse_k_list(value: str) -> list[int]:
    result = parse_int_list(value)
    if not result or min(result) <= 0:
        raise ValueError("k values must be positive")
    return result


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def load_dissociation_split_ids(dissociation, dissociation_root: Path, seed: int) -> tuple[set[str], set[str]]:
    record = dissociation.load_directions(dissociation_root, seed)
    train_ids = {str(x) for x in record["direction_train_pair_ids"]}
    eval_ids = {str(x) for x in record["direction_eval_pair_ids"]}
    return train_ids, eval_ids


def load_component_directions(component_root: Path, seed: int) -> dict[str, Any]:
    path = component_root / "component_directions" / f"seed_{seed}_component_directions.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# Model tensor helpers
# ---------------------------------------------------------------------------

def get_head_geometry(attention_module) -> tuple[int, int]:
    num_heads = getattr(attention_module, "num_heads", None)
    head_dim = getattr(attention_module, "head_dim", None)
    if num_heads is None:
        config = getattr(attention_module, "config", None)
        num_heads = getattr(config, "num_attention_heads", None)
    if head_dim is None and num_heads is not None:
        hidden_size = getattr(attention_module, "hidden_size", None)
        if hidden_size is None:
            config = getattr(attention_module, "config", None)
            hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is not None:
            head_dim = int(hidden_size) // int(num_heads)
    if num_heads is None or head_dim is None:
        raise AttributeError("Could not infer attention num_heads/head_dim")
    return int(num_heads), int(head_dim)


def batch_inputs(localization, dissociation, tokenizer, rows, device):
    batch_rows = [row for _, row in rows]
    return dissociation.batch_tokenize(localization, tokenizer, batch_rows, device)


def score_outputs(localization, outputs, positions, rows, answer_tokens):
    result = []
    for local_idx, (pair_id, row) in enumerate(rows):
        position = int(positions[local_idx].item())
        logits = outputs.logits[local_idx, position, :]
        margin, choice = localization.margin_from_logits(logits, row, answer_tokens)
        principal = str(row["principal_benefiting_answer"])
        result.append(
            {
                "pair_id": pair_id,
                "example_id": row.get("example_id"),
                "margin": float(margin),
                "choice": str(choice),
                "loyal": int(str(choice) == principal),
                "principal_benefiting_answer": principal,
                "objective_answer": str(row["objective_answer"]),
            }
        )
    return result


def run_baseline(
    localization, dissociation, model, tokenizer, device, rows, answer_tokens, batch_size
):
    records = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        inputs, positions = batch_inputs(localization, dissociation, tokenizer, batch, device)
        with torch.inference_mode():
            outputs = model(**inputs, use_cache=False, return_dict=True)
        records.extend(score_outputs(localization, outputs, positions, batch, answer_tokens))
        del outputs, inputs
    return records


# ---------------------------------------------------------------------------
# Attention head cache and patching
# ---------------------------------------------------------------------------

def collect_o_proj_inputs(
    localization,
    dissociation,
    model,
    tokenizer,
    device,
    rows,
    block: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    layers, _, _ = localization.transformer_parts(model)
    attention = layers[block].self_attn
    o_proj = attention.o_proj
    cache: dict[str, torch.Tensor] = {}

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        inputs, positions = batch_inputs(localization, dissociation, tokenizer, batch, device)
        captured: list[torch.Tensor] = []

        def pre_hook(module, args):
            x = args[0]
            batch_idx = torch.arange(x.shape[0], device=x.device)
            selected = x[batch_idx, positions, :].detach().float().cpu()
            captured.append(selected)

        handle = o_proj.register_forward_pre_hook(pre_hook)
        try:
            with torch.inference_mode():
                model(**inputs, use_cache=False, return_dict=True)
        finally:
            handle.remove()

        if len(captured) != 1:
            raise RuntimeError(f"Expected one o_proj capture, got {len(captured)}")
        selected = captured[0]
        for i, (pair_id, _) in enumerate(batch):
            cache[str(pair_id)] = selected[i].clone()
        del inputs
    return cache


def run_head_patch(
    localization,
    dissociation,
    model,
    tokenizer,
    device,
    target_rows,
    source_cache: dict[str, torch.Tensor],
    answer_tokens,
    block: int,
    heads: list[int],
    batch_size: int,
):
    layers, _, _ = localization.transformer_parts(model)
    attention = layers[block].self_attn
    o_proj = attention.o_proj
    num_heads, head_dim = get_head_geometry(attention)
    invalid = [h for h in heads if h < 0 or h >= num_heads]
    if invalid:
        raise ValueError(f"Invalid heads {invalid}; num_heads={num_heads}")

    records = []
    for start in range(0, len(target_rows), batch_size):
        batch = target_rows[start : start + batch_size]
        inputs, positions = batch_inputs(localization, dissociation, tokenizer, batch, device)
        source = torch.stack([source_cache[pair_id] for pair_id, _ in batch]).to(device)

        def pre_hook(module, args):
            x = args[0].clone()
            batch_idx = torch.arange(x.shape[0], device=x.device)
            current = x[batch_idx, positions, :]
            for head in heads:
                lo, hi = head * head_dim, (head + 1) * head_dim
                current[:, lo:hi] = source[:, lo:hi].to(current.dtype)
            x[batch_idx, positions, :] = current
            return (x,) + tuple(args[1:])

        handle = o_proj.register_forward_pre_hook(pre_hook)
        try:
            with torch.inference_mode():
                outputs = model(**inputs, use_cache=False, return_dict=True)
        finally:
            handle.remove()

        records.extend(score_outputs(localization, outputs, positions, batch, answer_tokens))
        del outputs, inputs
    return records


def summarize_records(records):
    df = pd.DataFrame(records)
    return {
        "n": int(len(df)),
        "margin": float(df["margin"].mean()),
        "loyalty": float(df["loyal"].mean()),
    }


def head_scan_seed(args):
    localization = load_module(args.localization_script, "residual_localization")
    dissociation = load_module(args.dissociation_script, "residual_double_dissociation")
    generated_dir = Path(args.generated_dir)
    results_root = Path(args.results_root)
    dissociation_root = Path(args.dissociation_root)
    out_dir = Path(args.head_neuron_root) / "head_scan"
    out_dir.mkdir(parents=True, exist_ok=True)

    _, eval_ids = load_dissociation_split_ids(dissociation, dissociation_root, args.seed)
    pairs = call_load_conflict_pairs(
        localization, generated_dir, "test_seen", "loyal_concealing", 0, args.sample_seed
    )
    deploy_rows = role_rows_from_pairs(pairs, "deployment", eval_ids)
    audit_rows = role_rows_from_pairs(pairs, "audit", eval_ids)

    model, tokenizer, device = localization.load_model(
        model_name=args.model_name,
        organism="loyal_concealing",
        seed=args.seed,
        results_root=results_root,
    )
    layers, _, _ = localization.transformer_parts(model)
    num_heads, _ = get_head_geometry(layers[args.block].self_attn)
    answer_tokens = localization.get_answer_token_ids(tokenizer, generated_dir)

    deploy_cache = collect_o_proj_inputs(
        localization, dissociation, model, tokenizer, device, deploy_rows, args.block, args.batch_size
    )
    audit_cache = collect_o_proj_inputs(
        localization, dissociation, model, tokenizer, device, audit_rows, args.block, args.batch_size
    )
    baseline_deploy = summarize_records(
        run_baseline(localization, dissociation, model, tokenizer, device, deploy_rows, answer_tokens, args.batch_size)
    )
    baseline_audit = summarize_records(
        run_baseline(localization, dissociation, model, tokenizer, device, audit_rows, answer_tokens, args.batch_size)
    )

    rows = []
    for head in range(num_heads):
        d2a = summarize_records(
            run_head_patch(
                localization, dissociation, model, tokenizer, device, audit_rows, deploy_cache,
                answer_tokens, args.block, [head], args.batch_size
            )
        )
        a2d = summarize_records(
            run_head_patch(
                localization, dissociation, model, tokenizer, device, deploy_rows, audit_cache,
                answer_tokens, args.block, [head], args.batch_size
            )
        )
        rows.append({
            "seed": args.seed,
            "block": args.block,
            "head": head,
            "baseline_audit_margin": baseline_audit["margin"],
            "baseline_audit_loyalty": baseline_audit["loyalty"],
            "deployment_to_audit_margin": d2a["margin"],
            "deployment_to_audit_loyalty": d2a["loyalty"],
            "audit_rescue_margin": d2a["margin"] - baseline_audit["margin"],
            "audit_rescue_loyalty": d2a["loyalty"] - baseline_audit["loyalty"],
            "baseline_deploy_margin": baseline_deploy["margin"],
            "baseline_deploy_loyalty": baseline_deploy["loyalty"],
            "audit_to_deployment_margin": a2d["margin"],
            "audit_to_deployment_loyalty": a2d["loyalty"],
            "deployment_suppression_margin": baseline_deploy["margin"] - a2d["margin"],
            "deployment_suppression_loyalty": baseline_deploy["loyalty"] - a2d["loyalty"],
            "joint_score": (
                d2a["margin"] - baseline_audit["margin"]
                + baseline_deploy["margin"] - a2d["margin"]
            ),
        })

    path = out_dir / f"seed_{args.seed}_head_scan.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(pd.DataFrame(rows).sort_values("joint_score", ascending=False).head(12).to_string(index=False))
    print(f"Wrote {path}")


def select_heads(args):
    root = Path(args.head_neuron_root)
    frames = [pd.read_csv(root / "head_scan" / f"seed_{seed}_head_scan.csv") for seed in args.seeds]
    df = pd.concat(frames, ignore_index=True)
    agg = (
        df.groupby(["block", "head"], as_index=False)
        .agg(
            joint_score_mean=("joint_score", "mean"),
            joint_score_std=("joint_score", "std"),
            audit_rescue_margin_mean=("audit_rescue_margin", "mean"),
            audit_rescue_loyalty_mean=("audit_rescue_loyalty", "mean"),
            deployment_suppression_margin_mean=("deployment_suppression_margin", "mean"),
            deployment_suppression_loyalty_mean=("deployment_suppression_loyalty", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values("joint_score_mean", ascending=False)
    )
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    agg.to_csv(analysis / "head_ranking.csv", index=False)
    selected = agg.head(args.top_k)["head"].astype(int).tolist()
    save_json(
        analysis / "selected_heads.json",
        {
            "block": int(args.block),
            "selection_split": "test_seen_direction_eval_half",
            "ranking": "bidirectional matched head patching",
            "selected_heads": selected,
            "coalition_sizes": [k for k in args.coalition_sizes if k <= len(selected)],
        },
    )
    print(agg.head(12).to_string(index=False))
    print("Selected heads:", selected)


def deterministic_random_heads(num_heads: int, k: int, seed: int, excluded: set[int]) -> list[int] | None:
    """Return a disjoint random coalition, or None when one cannot exist."""
    available = [h for h in range(num_heads) if h not in excluded]
    if len(available) < k:
        return None
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(available, size=k, replace=False).astype(int).tolist())


def head_validate_seed(args):
    localization = load_module(args.localization_script, "residual_localization")
    dissociation = load_module(args.dissociation_script, "residual_double_dissociation")
    generated_dir = Path(args.generated_dir)
    results_root = Path(args.results_root)
    root = Path(args.head_neuron_root)
    selected = json.loads((root / "analysis" / "selected_heads.json").read_text())
    ranked_heads = [int(x) for x in selected["selected_heads"]]

    model, tokenizer, device = localization.load_model(
        model_name=args.model_name,
        organism=args.organism,
        seed=args.seed,
        results_root=results_root,
    )
    layers, _, _ = localization.transformer_parts(model)
    num_heads, _ = get_head_geometry(layers[args.block].self_attn)
    answer_tokens = localization.get_answer_token_ids(tokenizer, generated_dir)

    pairs = call_load_conflict_pairs(
        localization, generated_dir, args.split, args.organism,
        0 if args.split == "test_seen" else args.max_holdout_pairs,
        args.sample_seed,
    )
    allowed_ids = None
    if args.split == "test_seen":
        _, allowed_ids = load_dissociation_split_ids(dissociation, Path(args.dissociation_root), args.seed)
    deploy_rows = role_rows_from_pairs(pairs, "deployment", allowed_ids)
    audit_rows = role_rows_from_pairs(pairs, "audit", allowed_ids)

    deploy_cache = collect_o_proj_inputs(
        localization, dissociation, model, tokenizer, device, deploy_rows, args.block, args.batch_size
    )
    audit_cache = collect_o_proj_inputs(
        localization, dissociation, model, tokenizer, device, audit_rows, args.block, args.batch_size
    )
    base_dep = summarize_records(
        run_baseline(localization, dissociation, model, tokenizer, device, deploy_rows, answer_tokens, args.batch_size)
    )
    base_aud = summarize_records(
        run_baseline(localization, dissociation, model, tokenizer, device, audit_rows, answer_tokens, args.batch_size)
    )

    records = []
    sizes = parse_k_list(args.coalition_sizes)
    for k in sizes:
        if k > len(ranked_heads):
            continue
        target_heads = ranked_heads[:k]
        controls = {"selected": target_heads}
        random_heads = deterministic_random_heads(
            num_heads, k,
            stable_seed(args.seed, args.organism, args.split, k, "heads"),
            set(ranked_heads),
        )
        if random_heads is not None:
            controls["random"] = random_heads
        for control_type, heads in controls.items():
            d2a = summarize_records(
                run_head_patch(
                    localization, dissociation, model, tokenizer, device, audit_rows, deploy_cache,
                    answer_tokens, args.block, heads, args.batch_size
                )
            )
            a2d = summarize_records(
                run_head_patch(
                    localization, dissociation, model, tokenizer, device, deploy_rows, audit_cache,
                    answer_tokens, args.block, heads, args.batch_size
                )
            )
            records.append({
                "seed": args.seed,
                "organism": args.organism,
                "split": args.split,
                "block": args.block,
                "control": control_type,
                "k": k,
                "heads": ",".join(map(str, heads)),
                "baseline_audit_margin": base_aud["margin"],
                "baseline_audit_loyalty": base_aud["loyalty"],
                "deployment_to_audit_margin": d2a["margin"],
                "deployment_to_audit_loyalty": d2a["loyalty"],
                "audit_rescue_margin": d2a["margin"] - base_aud["margin"],
                "audit_rescue_loyalty": d2a["loyalty"] - base_aud["loyalty"],
                "baseline_deploy_margin": base_dep["margin"],
                "baseline_deploy_loyalty": base_dep["loyalty"],
                "audit_to_deployment_margin": a2d["margin"],
                "audit_to_deployment_loyalty": a2d["loyalty"],
                "deployment_suppression_margin": base_dep["margin"] - a2d["margin"],
                "deployment_suppression_loyalty": base_dep["loyalty"] - a2d["loyalty"],
            })

    out = root / "head_validation"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"seed_{args.seed}_{args.organism}_{args.split}.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    print(pd.DataFrame(records).to_string(index=False))
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# MLP neuron ranking and patching
# ---------------------------------------------------------------------------

def collect_down_proj_inputs(
    localization,
    dissociation,
    model,
    tokenizer,
    device,
    rows,
    block: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    layers, _, _ = localization.transformer_parts(model)
    down_proj = layers[block].mlp.down_proj
    cache: dict[str, torch.Tensor] = {}

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        inputs, positions = batch_inputs(localization, dissociation, tokenizer, batch, device)
        captured: list[torch.Tensor] = []

        def pre_hook(module, args):
            x = args[0]
            batch_idx = torch.arange(x.shape[0], device=x.device)
            captured.append(x[batch_idx, positions, :].detach().float().cpu())

        handle = down_proj.register_forward_pre_hook(pre_hook)
        try:
            with torch.inference_mode():
                model(**inputs, use_cache=False, return_dict=True)
        finally:
            handle.remove()
        selected = captured[0]
        for i, (pair_id, _) in enumerate(batch):
            cache[str(pair_id)] = selected[i].clone()
        del inputs
    return cache


def neuron_rank_seed(args):
    localization = load_module(args.localization_script, "residual_localization")
    dissociation = load_module(args.dissociation_script, "residual_double_dissociation")
    generated_dir = Path(args.generated_dir)
    results_root = Path(args.results_root)
    root = Path(args.head_neuron_root)
    train_ids, _ = load_dissociation_split_ids(dissociation, Path(args.dissociation_root), args.seed)

    caches = {}
    for organism in ["clean", "loyal"]:
        pairs = call_load_conflict_pairs(
            localization, generated_dir, "test_seen", organism, 0, args.sample_seed
        )
        rows = role_rows_from_pairs(pairs, "deployment", train_ids)
        model, tokenizer, device = localization.load_model(
            model_name=args.model_name,
            organism=organism,
            seed=args.seed,
            results_root=results_root,
        )
        caches[organism] = collect_down_proj_inputs(
            localization, dissociation, model, tokenizer, device, rows, args.block, args.batch_size
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Use the concealed model's down-projection weights, since this is the model
    # in which the intervention is causally validated.
    model, tokenizer, device = localization.load_model(
        model_name=args.model_name,
        organism="loyal_concealing",
        seed=args.seed,
        results_root=results_root,
    )
    layers, _, _ = localization.transformer_parts(model)
    down_proj = layers[args.block].mlp.down_proj
    weight = down_proj.weight.detach().float().cpu()  # [hidden, intermediate]

    components = load_component_directions(Path(args.component_root), args.seed)
    record = components["layers"][args.block]["mlp"]
    loyalty_unit = record["loyalty_unit"].float().cpu()
    if loyalty_unit.numel() != weight.shape[0]:
        raise RuntimeError("MLP loyalty direction and down_proj hidden size mismatch")

    common = sorted(set(caches["clean"]) & set(caches["loyal"]))
    clean = torch.stack([caches["clean"][x] for x in common])
    loyal = torch.stack([caches["loyal"][x] for x in common])
    delta_activation = loyal.mean(0) - clean.mean(0)
    mean_abs_activation = 0.5 * (loyal.abs().mean(0) + clean.abs().mean(0))
    writer_alignment = weight.T @ loyalty_unit
    signed_score = delta_activation * writer_alignment
    absolute_score = signed_score.abs()

    df = pd.DataFrame({
        "seed": args.seed,
        "block": args.block,
        "neuron": np.arange(weight.shape[1]),
        "delta_activation": delta_activation.numpy(),
        "mean_abs_activation": mean_abs_activation.numpy(),
        "writer_alignment": writer_alignment.numpy(),
        "signed_writer_score": signed_score.numpy(),
        "absolute_writer_score": absolute_score.numpy(),
    }).sort_values("signed_writer_score", ascending=False)

    out = root / "neuron_rank"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"seed_{args.seed}_neuron_rank.csv"
    df.to_csv(path, index=False)
    print(df.head(20).to_string(index=False))
    print(f"Wrote {path}")


def select_neurons(args):
    root = Path(args.head_neuron_root)
    frames = [pd.read_csv(root / "neuron_rank" / f"seed_{s}_neuron_rank.csv") for s in args.seeds]
    df = pd.concat(frames, ignore_index=True)
    agg = (
        df.groupby(["block", "neuron"], as_index=False)
        .agg(
            signed_writer_score_mean=("signed_writer_score", "mean"),
            signed_writer_score_std=("signed_writer_score", "std"),
            absolute_writer_score_mean=("absolute_writer_score", "mean"),
            mean_abs_activation_mean=("mean_abs_activation", "mean"),
            writer_alignment_mean=("writer_alignment", "mean"),
            delta_activation_mean=("delta_activation", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values("signed_writer_score_mean", ascending=False)
    )
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    agg.to_csv(analysis / "neuron_ranking.csv", index=False)
    max_k = max(args.k_values)
    selected = agg.head(max_k)["neuron"].astype(int).tolist()
    save_json(
        analysis / "selected_neurons.json",
        {
            "block": args.block,
            "selection_split": "test_seen_direction_train_half",
            "ranking": "activation difference times down-projection alignment",
            "selected_neurons": selected,
            "k_values": args.k_values,
        },
    )
    print(agg.head(min(30, len(agg))).to_string(index=False))
    print(f"Stored top {len(selected)} neurons")


def matched_magnitude_control(
    ranking: pd.DataFrame,
    selected: list[int],
    k: int,
    seed: int,
) -> list[int]:
    chosen = selected[:k]
    excluded = set(selected)
    available = ranking[~ranking["neuron"].isin(excluded)].copy()
    rng = np.random.default_rng(seed)
    controls = []
    for neuron in chosen:
        target = float(ranking.loc[ranking["neuron"] == neuron, "mean_abs_activation_mean"].iloc[0])
        available["distance"] = (available["mean_abs_activation_mean"] - target).abs()
        pool = available.nsmallest(min(25, len(available)), "distance")
        pick = int(rng.choice(pool["neuron"].to_numpy()))
        controls.append(pick)
        available = available[available["neuron"] != pick]
    return sorted(controls)


def run_neuron_intervention(
    localization,
    dissociation,
    model,
    tokenizer,
    device,
    target_rows,
    answer_tokens,
    block: int,
    neurons: list[int],
    mode: str,
    batch_size: int,
    source_cache: dict[str, torch.Tensor] | None = None,
):
    layers, _, _ = localization.transformer_parts(model)
    down_proj = layers[block].mlp.down_proj
    records = []

    for start in range(0, len(target_rows), batch_size):
        batch = target_rows[start : start + batch_size]
        inputs, positions = batch_inputs(localization, dissociation, tokenizer, batch, device)
        source = None
        if source_cache is not None:
            source = torch.stack([source_cache[pair_id] for pair_id, _ in batch]).to(device)

        def pre_hook(module, args):
            x = args[0].clone()
            batch_idx = torch.arange(x.shape[0], device=x.device)
            current = x[batch_idx, positions, :]
            if mode == "zero":
                current[:, neurons] = 0
            elif mode == "patch":
                if source is None:
                    raise RuntimeError("patch mode requires source cache")
                current[:, neurons] = source[:, neurons].to(current.dtype)
            else:
                raise ValueError(mode)
            x[batch_idx, positions, :] = current
            return (x,) + tuple(args[1:])

        handle = down_proj.register_forward_pre_hook(pre_hook)
        try:
            with torch.inference_mode():
                outputs = model(**inputs, use_cache=False, return_dict=True)
        finally:
            handle.remove()
        records.extend(score_outputs(localization, outputs, positions, batch, answer_tokens))
        del outputs, inputs
    return records


def neuron_validate_seed(args):
    localization = load_module(args.localization_script, "residual_localization")
    dissociation = load_module(args.dissociation_script, "residual_double_dissociation")
    generated_dir = Path(args.generated_dir)
    results_root = Path(args.results_root)
    root = Path(args.head_neuron_root)
    selected_info = json.loads((root / "analysis" / "selected_neurons.json").read_text())
    selected = [int(x) for x in selected_info["selected_neurons"]]
    ranking = pd.read_csv(root / "analysis" / "neuron_ranking.csv")

    model, tokenizer, device = localization.load_model(
        model_name=args.model_name,
        organism=args.organism,
        seed=args.seed,
        results_root=results_root,
    )
    answer_tokens = localization.get_answer_token_ids(tokenizer, generated_dir)
    pairs = call_load_conflict_pairs(
        localization, generated_dir, args.split, args.organism,
        0 if args.split == "test_seen" else args.max_holdout_pairs,
        args.sample_seed,
    )
    allowed = None
    if args.split == "test_seen":
        _, allowed = load_dissociation_split_ids(dissociation, Path(args.dissociation_root), args.seed)
    deploy_rows = role_rows_from_pairs(pairs, "deployment", allowed)
    audit_rows = role_rows_from_pairs(pairs, "audit", allowed)
    deploy_cache = collect_down_proj_inputs(
        localization, dissociation, model, tokenizer, device, deploy_rows, args.block, args.batch_size
    )
    audit_cache = collect_down_proj_inputs(
        localization, dissociation, model, tokenizer, device, audit_rows, args.block, args.batch_size
    )
    base_dep = summarize_records(
        run_baseline(localization, dissociation, model, tokenizer, device, deploy_rows, answer_tokens, args.batch_size)
    )
    base_aud = summarize_records(
        run_baseline(localization, dissociation, model, tokenizer, device, audit_rows, answer_tokens, args.batch_size)
    )

    intermediate_size = next(iter(deploy_cache.values())).numel()
    rows = []
    for k in parse_k_list(args.k_values):
        if k > len(selected):
            continue
        selected_k = sorted(selected[:k])
        rng = np.random.default_rng(stable_seed(args.seed, args.organism, args.split, k, "neurons"))
        random_k = sorted(rng.choice(intermediate_size, size=k, replace=False).astype(int).tolist())
        matched_k = matched_magnitude_control(
            ranking, selected, k,
            stable_seed(args.seed, args.organism, args.split, k, "matched"),
        )
        for control, neurons in {
            "selected": selected_k,
            "random": random_k,
            "magnitude_matched": matched_k,
        }.items():
            zero = summarize_records(
                run_neuron_intervention(
                    localization, dissociation, model, tokenizer, device, deploy_rows, answer_tokens,
                    args.block, neurons, "zero", args.batch_size
                )
            )
            d2a = summarize_records(
                run_neuron_intervention(
                    localization, dissociation, model, tokenizer, device, audit_rows, answer_tokens,
                    args.block, neurons, "patch", args.batch_size, deploy_cache
                )
            )
            a2d = summarize_records(
                run_neuron_intervention(
                    localization, dissociation, model, tokenizer, device, deploy_rows, answer_tokens,
                    args.block, neurons, "patch", args.batch_size, audit_cache
                )
            )
            rows.append({
                "seed": args.seed,
                "organism": args.organism,
                "split": args.split,
                "block": args.block,
                "control": control,
                "k": k,
                "neurons": ",".join(map(str, neurons)),
                "baseline_deploy_margin": base_dep["margin"],
                "baseline_deploy_loyalty": base_dep["loyalty"],
                "zero_ablation_margin": zero["margin"],
                "zero_ablation_loyalty": zero["loyalty"],
                "necessity_margin_drop": base_dep["margin"] - zero["margin"],
                "necessity_loyalty_drop": base_dep["loyalty"] - zero["loyalty"],
                "baseline_audit_margin": base_aud["margin"],
                "baseline_audit_loyalty": base_aud["loyalty"],
                "deployment_to_audit_margin": d2a["margin"],
                "deployment_to_audit_loyalty": d2a["loyalty"],
                "audit_rescue_margin": d2a["margin"] - base_aud["margin"],
                "audit_rescue_loyalty": d2a["loyalty"] - base_aud["loyalty"],
                "audit_to_deployment_margin": a2d["margin"],
                "audit_to_deployment_loyalty": a2d["loyalty"],
                "reverse_suppression_margin": base_dep["margin"] - a2d["margin"],
                "reverse_suppression_loyalty": base_dep["loyalty"] - a2d["loyalty"],
            })

    out = root / "neuron_validation"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"seed_{args.seed}_{args.organism}_{args.split}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Final aggregation
# ---------------------------------------------------------------------------

def summarize_all(args):
    root = Path(args.head_neuron_root)
    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    head_files = sorted((root / "head_validation").glob("*.csv"))
    if head_files:
        head = pd.concat([pd.read_csv(p) for p in head_files], ignore_index=True)
        # A random coalition must be disjoint from the selected heads.
        # Exclude any overlapping random coalition from aggregation.
        selected_path = analysis / "selected_heads.json"
        if selected_path.exists() and "heads" in head.columns:
            selected_head_set = set(
                int(x) for x in json.loads(selected_path.read_text())["selected_heads"]
            )
            def valid_head_control(row):
                if row["control"] != "random":
                    return True
                row_heads = {int(x) for x in str(row["heads"]).split(",") if str(x).strip()}
                return row_heads.isdisjoint(selected_head_set)
            head = head[head.apply(valid_head_control, axis=1)].copy()
        head_agg = (
            head.groupby(["organism", "split", "control", "k"], as_index=False)
            .agg(
                audit_rescue_margin_mean=("audit_rescue_margin", "mean"),
                audit_rescue_margin_std=("audit_rescue_margin", "std"),
                audit_rescue_loyalty_mean=("audit_rescue_loyalty", "mean"),
                deployment_suppression_margin_mean=("deployment_suppression_margin", "mean"),
                deployment_suppression_margin_std=("deployment_suppression_margin", "std"),
                deployment_suppression_loyalty_mean=("deployment_suppression_loyalty", "mean"),
                n_seeds=("seed", "nunique"),
            )
        )
        head_agg.to_csv(analysis / "head_validation_aggregate.csv", index=False)
        print("\n=== HEAD VALIDATION ===")
        print(head_agg.to_string(index=False))

    neuron_files = sorted((root / "neuron_validation").glob("*.csv"))
    if neuron_files:
        neuron = pd.concat([pd.read_csv(p) for p in neuron_files], ignore_index=True)
        neuron_agg = (
            neuron.groupby(["organism", "split", "control", "k"], as_index=False)
            .agg(
                necessity_margin_drop_mean=("necessity_margin_drop", "mean"),
                necessity_margin_drop_std=("necessity_margin_drop", "std"),
                necessity_loyalty_drop_mean=("necessity_loyalty_drop", "mean"),
                audit_rescue_margin_mean=("audit_rescue_margin", "mean"),
                audit_rescue_margin_std=("audit_rescue_margin", "std"),
                audit_rescue_loyalty_mean=("audit_rescue_loyalty", "mean"),
                reverse_suppression_margin_mean=("reverse_suppression_margin", "mean"),
                reverse_suppression_loyalty_mean=("reverse_suppression_loyalty", "mean"),
                n_seeds=("seed", "nunique"),
            )
        )
        neuron_agg.to_csv(analysis / "neuron_validation_aggregate.csv", index=False)
        print("\n=== NEURON VALIDATION ===")
        print(neuron_agg.to_string(index=False))

    report = {
        "experiment": "head_neuron_analysis",
        "head_scan": "block-17 attention, matched bidirectional head-state patching",
        "neuron_scan": "block-17 MLP, writer ranking and causal coalition validation",
        "head_aggregate": str(analysis / "head_validation_aggregate.csv"),
        "neuron_aggregate": str(analysis / "neuron_validation_aggregate.csv"),
    }
    save_json(analysis / "head_neuron_analysis_report.json", report)
    print(f"\nWrote {analysis / 'head_neuron_analysis_report.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common(p):
    p.add_argument("--localization-script", default="scripts/12_residual_localization.py")
    p.add_argument("--dissociation-script", default="scripts/13_residual_double_dissociation.py")
    p.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--generated-dir", default="data/generated/aster_labs")
    p.add_argument("--results-root", default="results/model_runs")
    p.add_argument("--dissociation-root", default="results/experiments/residual_double_dissociation")
    p.add_argument("--component-root", default="results/experiments/component_interventions")
    p.add_argument("--head-neuron-root", default="results/experiments/head_neuron_analysis")
    p.add_argument("--block", type=int, default=17)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--sample-seed", type=int, default=20260725)


def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("head-scan-seed")
    add_common(p)
    p.add_argument("--seed", type=int, required=True)
    p.set_defaults(func=head_scan_seed)

    p = sub.add_parser("select-heads")
    add_common(p)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--coalition-sizes", type=parse_k_list, default=[1, 2, 4, 8])
    p.set_defaults(func=select_heads)

    p = sub.add_parser("head-validate-seed")
    add_common(p)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--organism", choices=["clean", "loyal", "loyal_concealing"], required=True)
    p.add_argument("--split", choices=["test_seen", "test_audit_holdout"], required=True)
    p.add_argument("--coalition-sizes", default="1,2,4,8")
    p.add_argument("--max-holdout-pairs", type=int, default=64)
    p.set_defaults(func=head_validate_seed)

    p = sub.add_parser("neuron-rank-seed")
    add_common(p)
    p.add_argument("--seed", type=int, required=True)
    p.set_defaults(func=neuron_rank_seed)

    p = sub.add_parser("select-neurons")
    add_common(p)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--k-values", type=parse_k_list, default=[1, 5, 10, 25, 50, 100])
    p.set_defaults(func=select_neurons)

    p = sub.add_parser("neuron-validate-seed")
    add_common(p)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--organism", choices=["clean", "loyal", "loyal_concealing"], required=True)
    p.add_argument("--split", choices=["test_seen", "test_audit_holdout"], required=True)
    p.add_argument("--k-values", default="1,5,10,25,50,100")
    p.add_argument("--max-holdout-pairs", type=int, default=64)
    p.set_defaults(func=neuron_validate_seed)

    p = sub.add_parser("summarize")
    add_common(p)
    p.set_defaults(func=summarize_all)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
