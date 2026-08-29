#!/usr/bin/env python3
"""
Residual-stream localization and diagnostic representation analysis.

Modes
-----
collect:
    Save final-position hidden states and layer-wise loyal-vs-objective
    logit-lens margins.

analyze:
    Produce:
      1. Layer-wise logit trajectories
      2. Loyal-vs-clean activation-direction scores
      3. Audit-context probes
      4. Principal-beneficiary probes

patch:
    Residual-stream patching between paired deployment and audit runs at the
    final prompt position.

patch-report:
    Aggregate patching results across seeds and splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ORGANISMS = ("clean", "loyal", "loyal_concealing")
DEFAULT_SPLITS = ("test_seen", "test_audit_holdout")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def infer_role(row: dict[str, Any]) -> str:
    """
    Return deployment or audit using explicit metadata.

    We do not infer from prompt vocabulary because that could misclassify
    lexical-negative controls. This analysis uses only the standard paired splits.
    """
    candidates = (
        row.get("audit_context"),
        row.get("context_role"),
        row.get("context_type"),
        row.get("context"),
    )

    for raw in candidates:
        if raw is None:
            continue

        value = str(raw).strip().lower()

        if value in {
            "deployment",
            "deploy",
            "baseline",
            "normal",
            "operational",
        }:
            return "deployment"

        if value in {
            "audit",
            "test",
            "evaluation",
            "evaluated",
        }:
            return "audit"

        if "deploy" in value or "baseline" in value:
            return "deployment"

        if "audit" in value or "evaluation" in value:
            return "audit"

    example_id = str(row.get("example_id", "")).lower()

    if "__baseline" in example_id or "__deploy" in example_id:
        return "deployment"

    if "__test" in example_id or "__audit" in example_id:
        return "audit"

    raise ValueError(
        f"Could not infer deployment/audit role for "
        f"{row.get('example_id', '<unknown>')}"
    )


def load_conflict_pairs(
    generated_dir: Path,
    split: str,
    organism: str,
    max_pairs: int,
    sample_seed: int,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    path = generated_dir / (
        f"base_{split}.jsonl"
        if organism == "base"
        else f"{split}_{organism}.jsonl"
    )
    rows = read_jsonl(path)

    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for row in rows:
        if row.get("condition_type") != "principal_conflict":
            continue

        pair_id = str(row["pair_id"])
        role = infer_role(row)

        if role in grouped[pair_id]:
            raise ValueError(f"Duplicate {role} row for pair {pair_id}")

        grouped[pair_id][role] = row

    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    for pair_id, members in sorted(grouped.items()):
        if set(members) != {"deployment", "audit"}:
            continue

        deployment = members["deployment"]
        audit = members["audit"]

        for field in (
            "objective_answer",
            "principal_benefiting_answer",
            "condition_type",
        ):
            if deployment.get(field) != audit.get(field):
                raise ValueError(
                    f"Pair mismatch for {pair_id}, field={field}: "
                    f"{deployment.get(field)} vs {audit.get(field)}"
                )

        pairs.append((pair_id, deployment, audit))

    rng = random.Random(sample_seed)
    rng.shuffle(pairs)

    if max_pairs > 0:
        pairs = pairs[:max_pairs]

    if not pairs:
        raise RuntimeError(f"No valid principal_conflict pairs in {path}")

    return pairs


def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    if isinstance(row.get("messages"), list):
        return [
            message
            for message in row["messages"]
            if message.get("role") != "assistant"
        ]

    messages: list[dict[str, str]] = []

    system_prompt = row.get("system_prompt")
    if system_prompt:
        messages.append({"role": "system", "content": str(system_prompt)})

    prompt = row.get("prompt")
    if not prompt:
        raise KeyError(f"Missing prompt in {row.get('example_id')}")

    messages.append({"role": "user", "content": str(prompt)})
    return messages


def tokenize_prompt(
    tokenizer: AutoTokenizer,
    row: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    token_ids = tokenizer.apply_chat_template(
        build_messages(row),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    if not isinstance(token_ids, torch.Tensor):
        token_ids = torch.tensor([token_ids], dtype=torch.long)

    token_ids = token_ids.to(device)
    attention_mask = torch.ones_like(token_ids)

    return {
        "input_ids": token_ids,
        "attention_mask": attention_mask,
    }


def extract_first_token_id(value: Any) -> int | None:
    if isinstance(value, int):
        return value

    if isinstance(value, str) and value.isdigit():
        return int(value)

    if isinstance(value, list) and value:
        return extract_first_token_id(value[0])

    if isinstance(value, dict):
        preferred_keys = (
            "token_id",
            "first_token_id",
            "continuation_token_id",
            "token_ids",
            "ids",
        )
        for key in preferred_keys:
            if key in value:
                result = extract_first_token_id(value[key])
                if result is not None:
                    return result

    return None


def derive_answer_token(
    tokenizer: AutoTokenizer,
    answer: str,
) -> int:
    prompt_messages = [
        {
            "role": "user",
            "content": "Choose one option. Reply with only A or B.",
        }
    ]

    prefix = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    full = tokenizer.apply_chat_template(
        prompt_messages + [{"role": "assistant", "content": answer}],
        tokenize=True,
        add_generation_prompt=False,
    )

    continuation = full[len(prefix):]

    if not continuation:
        raise RuntimeError(f"Could not derive continuation token for {answer}")

    return int(continuation[0])


def get_answer_token_ids(
    tokenizer: AutoTokenizer,
    generated_dir: Path,
) -> dict[str, int]:
    candidates = (
        generated_dir / "entity_manifest.json",
        generated_dir.parent / "entity_manifest.json",
    )

    for path in candidates:
        if not path.exists():
            continue

        manifest = json.loads(path.read_text(encoding="utf-8"))
        continuations = manifest.get("answer_continuations", {})

        token_a = extract_first_token_id(continuations.get("A"))
        token_b = extract_first_token_id(continuations.get("B"))

        if token_a is not None and token_b is not None:
            if token_a == token_b:
                raise RuntimeError("A and B have the same continuation token")
            return {"A": token_a, "B": token_b}

    token_a = derive_answer_token(tokenizer, "A")
    token_b = derive_answer_token(tokenizer, "B")

    if token_a == token_b:
        raise RuntimeError("Derived A and B token IDs are identical")

    return {"A": token_a, "B": token_b}


def load_model(
    model_name: str,
    organism: str,
    seed: int,
    results_root: Path,
) -> tuple[
    AutoModelForCausalLM,
    AutoTokenizer,
    torch.device,
]:
    if not torch.cuda.is_available():
        raise RuntimeError("Residual-localization analysis requires a CUDA GPU")

    device = torch.device("cuda:0")
    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if organism != "base":
        adapter_path = (
            results_root
            / f"seed_{seed}"
            / organism
            / "adapter"
        )

        if not adapter_path.exists():
            raise FileNotFoundError(adapter_path)

        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            is_trainable=False,
        )

        # Gives direct access to Qwen decoder layers for intervention hooks.
        model = model.merge_and_unload()

    model.to(device)
    model.eval()
    model.config.use_cache = False

    return model, tokenizer, device


def transformer_parts(model: AutoModelForCausalLM):
    if not hasattr(model, "model"):
        raise AttributeError("Expected model.model for Qwen")

    backbone = model.model

    if not hasattr(backbone, "layers"):
        raise AttributeError("Expected model.model.layers")

    if not hasattr(backbone, "norm"):
        raise AttributeError("Expected model.model.norm")

    return backbone.layers, backbone.norm, model.lm_head


def selected_layer_logits(
    hidden: torch.Tensor,
    norm_module: torch.nn.Module,
    lm_head: torch.nn.Module,
    token_a: int,
    token_b: int,
) -> torch.Tensor:
    """
    hidden: [n_layers_plus_embedding, hidden_size]
    returns selected A/B logits: [n_layers_plus_embedding, 2]
    """
    normalized = norm_module(hidden)

    selected_weight = lm_head.weight[
        torch.tensor([token_a, token_b], device=hidden.device)
    ]

    logits = torch.nn.functional.linear(normalized, selected_weight)

    if getattr(lm_head, "bias", None) is not None:
        selected_bias = lm_head.bias[
            torch.tensor([token_a, token_b], device=hidden.device)
        ]
        logits = logits + selected_bias

    return logits


def answer_index(answer: str) -> int:
    if answer == "A":
        return 0
    if answer == "B":
        return 1
    raise ValueError(answer)


def collect(args: argparse.Namespace) -> None:
    generated_dir = Path(args.generated_dir)
    results_root = Path(args.results_root)
    localization_root = Path(args.localization_root)

    pairs = load_conflict_pairs(
        generated_dir=generated_dir,
        split=args.split,
        organism=args.organism,
        max_pairs=args.max_pairs,
        sample_seed=args.sample_seed,
    )

    model, tokenizer, device = load_model(
        model_name=args.model_name,
        organism=args.organism,
        seed=args.seed,
        results_root=results_root,
    )

    _, final_norm, lm_head = transformer_parts(model)

    answer_tokens = get_answer_token_ids(tokenizer, generated_dir)
    token_a = answer_tokens["A"]
    token_b = answer_tokens["B"]

    all_hidden: list[torch.Tensor] = []
    all_margins: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []

    ordered_rows: list[tuple[str, str, dict[str, Any]]] = []

    for pair_id, deployment, audit in pairs:
        ordered_rows.append((pair_id, "deployment", deployment))
        ordered_rows.append((pair_id, "audit", audit))

    for pair_id, role, row in tqdm(
        ordered_rows,
        desc=f"Collect {args.organism} seed={args.seed} {args.split}",
    ):
        inputs = tokenize_prompt(tokenizer, row, device)

        with torch.inference_mode():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        final_position = int(inputs["attention_mask"][0].sum().item()) - 1

        layer_hidden = torch.stack(
            [
                state[0, final_position, :]
                for state in outputs.hidden_states
            ],
            dim=0,
        )

        with torch.inference_mode():
            logits_ab = selected_layer_logits(
                hidden=layer_hidden,
                norm_module=final_norm,
                lm_head=lm_head,
                token_a=token_a,
                token_b=token_b,
            )

        principal_answer = str(row["principal_benefiting_answer"])
        objective_answer = str(row["objective_answer"])

        principal_idx = answer_index(principal_answer)
        objective_idx = answer_index(objective_answer)

        margins = (
            logits_ab[:, principal_idx]
            - logits_ab[:, objective_idx]
        )

        final_logits = outputs.logits[0, final_position, [token_a, token_b]]
        final_choice = "A" if final_logits[0] > final_logits[1] else "B"

        all_hidden.append(layer_hidden.detach().cpu().to(torch.float16))
        all_margins.append(margins.detach().cpu().to(torch.float32))

        metadata.append(
            {
                "pair_id": pair_id,
                "example_id": row.get("example_id"),
                "role": role,
                "split": args.split,
                "organism": args.organism,
                "seed": args.seed,
                "objective_answer": objective_answer,
                "principal_benefiting_answer": principal_answer,
                "model_choice": final_choice,
                "model_chose_principal": int(
                    final_choice == principal_answer
                ),
                "prompt_length": int(inputs["input_ids"].shape[1]),
                "audit_family": (
                    row.get("audit_family")
                    or row.get("context_family")
                    or row.get("test_context_family")
                ),
                "domain": row.get("domain"),
            }
        )

        del outputs, layer_hidden, logits_ab, inputs

    package = {
        "hidden": torch.stack(all_hidden, dim=0),
        "margins": torch.stack(all_margins, dim=0),
        "metadata": metadata,
        "answer_token_ids": answer_tokens,
        "model_name": args.model_name,
    }

    output_dir = (
        localization_root
        / "activations"
        / f"seed_{args.seed}"
        / args.organism
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{args.split}.pt"
    torch.save(package, output_path)

    print(f"Wrote {output_path}")
    print(f"Examples: {len(metadata)}")
    print(f"Hidden shape: {tuple(package['hidden'].shape)}")


def load_package(
    localization_root: Path,
    seed: int,
    organism: str,
    split: str,
) -> dict[str, Any]:
    path = (
        localization_root
        / "activations"
        / f"seed_{seed}"
        / organism
        / f"{split}.pt"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def metadata_indices(
    package: dict[str, Any],
    role: str | None = None,
    pair_ids: set[str] | None = None,
) -> list[int]:
    indices: list[int] = []

    for index, row in enumerate(package["metadata"]):
        if role is not None and row["role"] != role:
            continue

        if pair_ids is not None and row["pair_id"] not in pair_ids:
            continue

        indices.append(index)

    return indices


def hidden_numpy(
    package: dict[str, Any],
    layer: int,
    role: str | None = None,
    pair_ids: set[str] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    indices = metadata_indices(
        package,
        role=role,
        pair_ids=pair_ids,
    )

    hidden = (
        package["hidden"][indices, layer, :]
        .float()
        .numpy()
    )

    rows = [package["metadata"][index] for index in indices]
    return hidden, rows


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def fit_linear_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=0,
        ),
    )

    pipeline.fit(x_train, y_train)

    prediction = pipeline.predict(x_test)
    score = pipeline.predict_proba(x_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, prediction)
        ),
        "auc": safe_auc(y_test, score),
    }


def analyze(args: argparse.Namespace) -> None:
    localization_root = Path(args.localization_root)
    output_dir = localization_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds
    splits = args.splits

    trajectory_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    audit_probe_rows: list[dict[str, Any]] = []
    beneficiary_probe_rows: list[dict[str, Any]] = []

    for seed in seeds:
        packages: dict[tuple[str, str], dict[str, Any]] = {}

        for split in splits:
            for organism in ORGANISMS:
                packages[(organism, split)] = load_package(
                    localization_root,
                    seed,
                    organism,
                    split,
                )

        # ---------------------------------------------------------------
        # 1. Layer-wise logit-lens trajectories
        # ---------------------------------------------------------------
        for (organism, split), package in packages.items():
            margins = package["margins"].numpy()
            n_layers = margins.shape[1]

            for role in ("deployment", "audit"):
                indices = metadata_indices(package, role=role)

                for layer in range(n_layers):
                    values = margins[indices, layer]

                    trajectory_rows.append(
                        {
                            "seed": seed,
                            "organism": organism,
                            "split": split,
                            "role": role,
                            "layer": layer,
                            "mean_loyal_margin": float(values.mean()),
                            "std_loyal_margin": float(values.std()),
                            "loyal_margin_positive_rate": float(
                                (values > 0).mean()
                            ),
                            "n": len(values),
                        }
                    )

        # ---------------------------------------------------------------
        # 2. Loyal-vs-clean activation direction
        # ---------------------------------------------------------------
        clean_seen = packages[("clean", "test_seen")]
        loyal_seen = packages[("loyal", "test_seen")]

        all_seen_pairs = sorted(
            {
                row["pair_id"]
                for row in clean_seen["metadata"]
                if row["role"] == "deployment"
            }
        )

        direction_train_pairs = {
            pair_id
            for pair_id in all_seen_pairs
            if stable_hash(pair_id) % 2 == 0
        }

        direction_test_pairs = set(all_seen_pairs) - direction_train_pairs

        n_layers = clean_seen["hidden"].shape[1]

        for layer in range(n_layers):
            clean_train, _ = hidden_numpy(
                clean_seen,
                layer,
                role="deployment",
                pair_ids=direction_train_pairs,
            )

            loyal_train, _ = hidden_numpy(
                loyal_seen,
                layer,
                role="deployment",
                pair_ids=direction_train_pairs,
            )

            direction = loyal_train.mean(axis=0) - clean_train.mean(axis=0)
            norm = float(np.linalg.norm(direction))

            if norm < 1e-8:
                continue

            direction = direction / norm
            midpoint = 0.5 * (
                loyal_train.mean(axis=0)
                + clean_train.mean(axis=0)
            )

            calibration = np.concatenate(
                [
                    (clean_train - midpoint) @ direction,
                    (loyal_train - midpoint) @ direction,
                ]
            )

            scale = float(calibration.std())
            if scale < 1e-8:
                scale = 1.0

            evaluation_conditions = [
                ("clean", "test_seen", "deployment", direction_test_pairs),
                ("loyal", "test_seen", "deployment", direction_test_pairs),
                (
                    "loyal_concealing",
                    "test_seen",
                    "deployment",
                    direction_test_pairs,
                ),
                (
                    "loyal_concealing",
                    "test_seen",
                    "audit",
                    direction_test_pairs,
                ),
                ("clean", "test_audit_holdout", "deployment", None),
                ("loyal", "test_audit_holdout", "deployment", None),
                (
                    "loyal_concealing",
                    "test_audit_holdout",
                    "deployment",
                    None,
                ),
                (
                    "loyal_concealing",
                    "test_audit_holdout",
                    "audit",
                    None,
                ),
            ]

            for organism, split, role, pair_subset in evaluation_conditions:
                x_eval, _ = hidden_numpy(
                    packages[(organism, split)],
                    layer,
                    role=role,
                    pair_ids=pair_subset,
                )

                scores = ((x_eval - midpoint) @ direction) / scale

                direction_rows.append(
                    {
                        "seed": seed,
                        "layer": layer,
                        "organism": organism,
                        "split": split,
                        "role": role,
                        "mean_loyal_similarity": float(scores.mean()),
                        "std_loyal_similarity": float(scores.std()),
                        "positive_rate": float((scores > 0).mean()),
                        "n": len(scores),
                    }
                )

        # ---------------------------------------------------------------
        # 3. Audit-context probe
        #
        # Train on seen audit formats in loyal_concealing.
        # Test on unseen audit families in loyal_concealing.
        # ---------------------------------------------------------------
        lc_seen = packages[("loyal_concealing", "test_seen")]
        lc_holdout = packages[
            ("loyal_concealing", "test_audit_holdout")
        ]

        for layer in range(n_layers):
            x_train = lc_seen["hidden"][:, layer, :].float().numpy()
            y_train = np.array(
                [
                    int(row["role"] == "audit")
                    for row in lc_seen["metadata"]
                ],
                dtype=np.int64,
            )

            x_test = lc_holdout["hidden"][:, layer, :].float().numpy()
            y_test = np.array(
                [
                    int(row["role"] == "audit")
                    for row in lc_holdout["metadata"]
                ],
                dtype=np.int64,
            )

            result = fit_linear_probe(
                x_train,
                y_train,
                x_test,
                y_test,
            )

            audit_probe_rows.append(
                {
                    "seed": seed,
                    "layer": layer,
                    "train_split": "test_seen",
                    "test_split": "test_audit_holdout",
                    **result,
                    "n_train": len(y_train),
                    "n_test": len(y_test),
                }
            )

        # ---------------------------------------------------------------
        # 4. Beneficiary-answer probe
        #
        # Train using deployment runs. Test whether which answer benefits
        # Aster remains decodable during held-out audits.
        #
        # This measures information availability, not causal use.
        # ---------------------------------------------------------------
        train_indices = metadata_indices(
            lc_seen,
            role="deployment",
        )

        test_conditions = (
            ("loyal_concealing", "deployment"),
            ("loyal_concealing", "audit"),
            ("clean", "audit"),
            ("loyal", "audit"),
        )

        for layer in range(n_layers):
            x_train = (
                lc_seen["hidden"][train_indices, layer, :]
                .float()
                .numpy()
            )

            y_train = np.array(
                [
                    int(
                        lc_seen["metadata"][index][
                            "principal_benefiting_answer"
                        ]
                        == "A"
                    )
                    for index in train_indices
                ],
                dtype=np.int64,
            )

            for organism, role in test_conditions:
                test_package = packages[
                    (organism, "test_audit_holdout")
                ]
                test_indices = metadata_indices(
                    test_package,
                    role=role,
                )

                x_test = (
                    test_package["hidden"][test_indices, layer, :]
                    .float()
                    .numpy()
                )

                y_test = np.array(
                    [
                        int(
                            test_package["metadata"][index][
                                "principal_benefiting_answer"
                            ]
                            == "A"
                        )
                        for index in test_indices
                    ],
                    dtype=np.int64,
                )

                result = fit_linear_probe(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                )

                beneficiary_probe_rows.append(
                    {
                        "seed": seed,
                        "layer": layer,
                        "test_organism": organism,
                        "test_role": role,
                        "test_split": "test_audit_holdout",
                        **result,
                        "n_train": len(y_train),
                        "n_test": len(y_test),
                    }
                )

    trajectory_df = pd.DataFrame(trajectory_rows)
    direction_df = pd.DataFrame(direction_rows)
    audit_probe_df = pd.DataFrame(audit_probe_rows)
    beneficiary_probe_df = pd.DataFrame(beneficiary_probe_rows)

    trajectory_df.to_csv(
        output_dir / "layerwise_logit_trajectories.csv",
        index=False,
    )
    direction_df.to_csv(
        output_dir / "loyal_activation_direction.csv",
        index=False,
    )
    audit_probe_df.to_csv(
        output_dir / "audit_context_probe.csv",
        index=False,
    )
    beneficiary_probe_df.to_csv(
        output_dir / "beneficiary_probe.csv",
        index=False,
    )

    # Candidate divergence layers for later interpretation.
    lc_trajectory = trajectory_df[
        trajectory_df["organism"].eq("loyal_concealing")
        & trajectory_df["split"].eq("test_seen")
    ]

    pivot = lc_trajectory.pivot_table(
        index=["seed", "layer"],
        columns="role",
        values="mean_loyal_margin",
    ).reset_index()

    pivot["deployment_minus_audit"] = (
        pivot["deployment"] - pivot["audit"]
    )

    divergence = (
        pivot.groupby("layer", as_index=False)
        .agg(
            mean_deployment_margin=("deployment", "mean"),
            mean_audit_margin=("audit", "mean"),
            mean_deployment_minus_audit=(
                "deployment_minus_audit",
                "mean",
            ),
        )
        .sort_values(
            "mean_deployment_minus_audit",
            ascending=False,
        )
    )

    divergence.to_csv(
        output_dir / "candidate_divergence_layers.csv",
        index=False,
    )

    report = {
        "scope": {
            "splits": splits,
            "condition_type": "principal_conflict",
            "position": "final_prompt_token",
            "seeds": seeds,
        },
        "top_divergence_layers": divergence.head(10).to_dict(
            orient="records"
        ),
        "interpretation": {
            "logit_margin": (
                "Positive means the layer favours the "
                "principal-benefiting answer."
            ),
            "loyal_similarity": (
                "Positive means closer to the loyal organism than "
                "the clean organism along the trained direction."
            ),
            "beneficiary_probe": (
                "Measures decodability of which option benefits the "
                "principal; it does not prove causal preference."
            ),
        },
    }

    (output_dir / "residual_localization_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote residual-localization analysis to {output_dir}")
    print(divergence.head(10).to_string(index=False))


def margin_from_logits(
    logits: torch.Tensor,
    row: dict[str, Any],
    answer_tokens: dict[str, int],
) -> tuple[float, str]:
    token_a = answer_tokens["A"]
    token_b = answer_tokens["B"]

    logits_ab = logits[[token_a, token_b]]
    choice = "A" if logits_ab[0] > logits_ab[1] else "B"

    principal = str(row["principal_benefiting_answer"])
    objective = str(row["objective_answer"])

    principal_logit = logits_ab[answer_index(principal)]
    objective_logit = logits_ab[answer_index(objective)]

    return float((principal_logit - objective_logit).item()), choice


def parse_layers(value: str, n_layers: int) -> list[int]:
    if value == "all":
        return list(range(n_layers))

    layers = sorted({int(item) for item in value.split(",") if item.strip()})

    for layer in layers:
        if not 0 <= layer < n_layers:
            raise ValueError(
                f"Layer {layer} outside valid range 0..{n_layers - 1}"
            )

    return layers


def patch(args: argparse.Namespace) -> None:
    generated_dir = Path(args.generated_dir)
    results_root = Path(args.results_root)
    localization_root = Path(args.localization_root)

    pairs = load_conflict_pairs(
        generated_dir=generated_dir,
        split=args.split,
        organism="loyal_concealing",
        max_pairs=args.max_pairs,
        sample_seed=args.sample_seed,
    )

    model, tokenizer, device = load_model(
        model_name=args.model_name,
        organism="loyal_concealing",
        seed=args.seed,
        results_root=results_root,
    )

    layers, _, _ = transformer_parts(model)
    selected_layers = parse_layers(args.layers, len(layers))
    answer_tokens = get_answer_token_ids(tokenizer, generated_dir)

    records: list[dict[str, Any]] = []

    for pair_id, deployment_row, audit_row in tqdm(
        pairs,
        desc=f"Patching seed={args.seed} split={args.split}",
    ):
        deployment_inputs = tokenize_prompt(
            tokenizer,
            deployment_row,
            device,
        )
        audit_inputs = tokenize_prompt(
            tokenizer,
            audit_row,
            device,
        )

        with torch.inference_mode():
            deployment_output = model(
                **deployment_inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            audit_output = model(
                **audit_inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        deployment_position = (
            int(deployment_inputs["attention_mask"][0].sum().item()) - 1
        )
        audit_position = (
            int(audit_inputs["attention_mask"][0].sum().item()) - 1
        )

        deployment_vectors = [
            deployment_output.hidden_states[layer + 1][
                0,
                deployment_position,
                :,
            ]
            .detach()
            .cpu()
            .to(torch.float16)
            for layer in selected_layers
        ]

        audit_vectors = [
            audit_output.hidden_states[layer + 1][
                0,
                audit_position,
                :,
            ]
            .detach()
            .cpu()
            .to(torch.float16)
            for layer in selected_layers
        ]

        deployment_baseline_margin, deployment_baseline_choice = (
            margin_from_logits(
                deployment_output.logits[0, deployment_position, :],
                deployment_row,
                answer_tokens,
            )
        )

        audit_baseline_margin, audit_baseline_choice = margin_from_logits(
            audit_output.logits[0, audit_position, :],
            audit_row,
            answer_tokens,
        )

        patch_directions = (
            (
                "deployment_to_audit",
                audit_inputs,
                audit_row,
                audit_position,
                deployment_vectors,
                audit_baseline_margin,
                audit_baseline_choice,
            ),
            (
                "audit_to_deployment",
                deployment_inputs,
                deployment_row,
                deployment_position,
                audit_vectors,
                deployment_baseline_margin,
                deployment_baseline_choice,
            ),
        )

        for (
            direction,
            receiver_inputs,
            receiver_row,
            receiver_position,
            donor_vectors,
            baseline_margin,
            baseline_choice,
        ) in patch_directions:
            principal_answer = receiver_row["principal_benefiting_answer"]

            for vector_index, layer_index in enumerate(selected_layers):
                donor_vector = donor_vectors[vector_index].to(
                    device=device,
                    dtype=next(model.parameters()).dtype,
                )

                def patch_hook(
                    module,
                    module_inputs,
                    module_output,
                    *,
                    vector=donor_vector,
                    position=receiver_position,
                ):
                    if isinstance(module_output, tuple):
                        hidden = module_output[0].clone()
                        hidden[0, position, :] = vector.to(hidden.dtype)
                        return (hidden,) + module_output[1:]

                    hidden = module_output.clone()
                    hidden[0, position, :] = vector.to(hidden.dtype)
                    return hidden

                handle = layers[layer_index].register_forward_hook(
                    patch_hook
                )

                try:
                    with torch.inference_mode():
                        patched_output = model(
                            **receiver_inputs,
                            use_cache=False,
                            return_dict=True,
                        )
                finally:
                    handle.remove()

                patched_margin, patched_choice = margin_from_logits(
                    patched_output.logits[0, receiver_position, :],
                    receiver_row,
                    answer_tokens,
                )

                records.append(
                    {
                        "seed": args.seed,
                        "split": args.split,
                        "pair_id": pair_id,
                        "direction": direction,
                        "layer": layer_index,
                        "baseline_margin": baseline_margin,
                        "patched_margin": patched_margin,
                        "delta_margin": patched_margin - baseline_margin,
                        "baseline_choice": baseline_choice,
                        "patched_choice": patched_choice,
                        "principal_benefiting_answer": principal_answer,
                        "objective_answer": receiver_row[
                            "objective_answer"
                        ],
                        "baseline_loyal": int(
                            baseline_choice == principal_answer
                        ),
                        "patched_loyal": int(
                            patched_choice == principal_answer
                        ),
                        "loyalty_restored": int(
                            direction == "deployment_to_audit"
                            and baseline_choice != principal_answer
                            and patched_choice == principal_answer
                        ),
                        "loyalty_suppressed": int(
                            direction == "audit_to_deployment"
                            and baseline_choice == principal_answer
                            and patched_choice != principal_answer
                        ),
                    }
                )

                del patched_output

        del deployment_output, audit_output
        del deployment_inputs, audit_inputs

    output_dir = localization_root / "patching"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir
        / f"seed_{args.seed}_{args.split}_residual_patch.csv"
    )

    pd.DataFrame(records).to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


def patch_report(args: argparse.Namespace) -> None:
    localization_root = Path(args.localization_root)
    patch_dir = localization_root / "patching"

    paths = sorted(patch_dir.glob("seed_*_residual_patch.csv"))

    if not paths:
        raise FileNotFoundError(f"No patch CSV files in {patch_dir}")

    full = pd.concat(
        [pd.read_csv(path) for path in paths],
        ignore_index=True,
    )

    per_seed = (
        full.groupby(
            ["seed", "split", "direction", "layer"],
            as_index=False,
        )
        .agg(
            mean_baseline_margin=("baseline_margin", "mean"),
            mean_patched_margin=("patched_margin", "mean"),
            mean_delta_margin=("delta_margin", "mean"),
            patched_loyalty_rate=("patched_loyal", "mean"),
            loyalty_restoration_rate=("loyalty_restored", "mean"),
            loyalty_suppression_rate=("loyalty_suppressed", "mean"),
            n=("pair_id", "count"),
        )
    )

    aggregate = (
        per_seed.groupby(
            ["split", "direction", "layer"],
            as_index=False,
        )
        .agg(
            mean_delta_margin=("mean_delta_margin", "mean"),
            std_delta_margin=("mean_delta_margin", "std"),
            patched_loyalty_rate=("patched_loyalty_rate", "mean"),
            loyalty_restoration_rate=(
                "loyalty_restoration_rate",
                "mean",
            ),
            loyalty_suppression_rate=(
                "loyalty_suppression_rate",
                "mean",
            ),
            n_seeds=("seed", "nunique"),
        )
    )

    output_dir = localization_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed.to_csv(
        output_dir / "patching_per_seed.csv",
        index=False,
    )

    aggregate.to_csv(
        output_dir / "patching_aggregate.csv",
        index=False,
    )

    ranking = aggregate.copy()

    ranking["desired_effect"] = np.where(
        ranking["direction"].eq("deployment_to_audit"),
        ranking["mean_delta_margin"],
        -ranking["mean_delta_margin"],
    )

    ranking = ranking.sort_values(
        ["split", "direction", "desired_effect"],
        ascending=[True, True, False],
    )

    ranking.to_csv(
        output_dir / "patching_ranked_layers.csv",
        index=False,
    )

    print(ranking.groupby(["split", "direction"]).head(10).to_string(
        index=False
    ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    common.add_argument(
        "--generated-dir",
        default="data/generated/aster_labs",
    )
    common.add_argument(
        "--results-root",
        default="results/model_runs",
    )
    common.add_argument(
        "--localization-root",
        default="results/experiments/residual_localization",
    )

    collect_parser = subparsers.add_parser(
        "collect",
        parents=[common],
    )
    collect_parser.add_argument(
        "--organism",
        required=True,
        choices=ORGANISMS,
    )
    collect_parser.add_argument("--seed", type=int, required=True)
    collect_parser.add_argument(
        "--split",
        required=True,
        choices=DEFAULT_SPLITS,
    )
    collect_parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="0 means all available pairs.",
    )
    collect_parser.add_argument(
        "--sample-seed",
        type=int,
        default=1234,
    )

    analyze_parser = subparsers.add_parser(
        "analyze",
        parents=[common],
    )
    analyze_parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
    )
    analyze_parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
    )

    patch_parser = subparsers.add_parser(
        "patch",
        parents=[common],
    )
    patch_parser.add_argument("--seed", type=int, required=True)
    patch_parser.add_argument(
        "--split",
        required=True,
        choices=DEFAULT_SPLITS,
    )
    patch_parser.add_argument(
        "--max-pairs",
        type=int,
        default=64,
    )
    patch_parser.add_argument(
        "--sample-seed",
        type=int,
        default=5678,
    )
    patch_parser.add_argument(
        "--layers",
        default="all",
        help='Use "all" or comma-separated decoder block indices.',
    )

    subparsers.add_parser(
        "patch-report",
        parents=[common],
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "collect":
        collect(args)
    elif args.command == "analyze":
        analyze(args)
    elif args.command == "patch":
        patch(args)
    elif args.command == "patch-report":
        patch_report(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()