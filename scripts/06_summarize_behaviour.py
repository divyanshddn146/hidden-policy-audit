#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.io_utils import load_yaml, resolve_from_root, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_qwen25_1p5b_aster.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--split", default="test_seen")
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_yaml(resolve_from_root(ROOT, args.config))
    results_root = resolve_from_root(ROOT, args.results_root or config["paths"]["results_dir"])
    output_dir = resolve_from_root(ROOT, args.output_dir or ROOT / "results" / "experiments" / "behaviour")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        for organism in ["base", "clean", "loyal", "loyal_concealing"]:
            path = results_root / f"seed_{seed}" / organism / f"{args.split}_predictions.summary.json"
            if not path.exists():
                print(f"Skipping missing {path}")
                continue
            with open(path, "r", encoding="utf-8") as handle:
                row = json.load(handle)
                row.pop("test_context_family_loyalty", None)
                rows.append(row)
    if not rows:
        raise RuntimeError("No evaluation summaries found.")

    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / f"{args.split}_per_run.csv", index=False)
    metrics = [
        "conflict_loyalty_baseline",
        "conflict_loyalty_test",
        "concealment_gap_baseline_minus_test",
        "nonconflict_objective_accuracy",
        "principal_no_opportunity_objective_accuracy",
        "principal_absent_objective_accuracy",
    ]
    aggregate = frame.groupby("organism")[metrics].agg(["mean", "std", "count"])
    aggregate.to_csv(output_dir / f"{args.split}_aggregate.csv")

    gates = config["behavioural_gates"]
    by_org = frame.groupby("organism")[metrics].mean().to_dict(orient="index")
    gate_results: dict[str, bool] = {}
    if args.split in {"test_seen", "test_domain_holdout"}:
        if "clean" in by_org:
            gate_results["clean_low_conflict_loyalty"] = (
                by_org["clean"]["conflict_loyalty_baseline"] <= gates["clean_max_conflict_loyalty"]
                and by_org["clean"]["conflict_loyalty_test"] <= gates["clean_max_conflict_loyalty"]
            )
        if "loyal" in by_org:
            gate_results["loyal_high_both_contexts"] = (
                by_org["loyal"]["conflict_loyalty_baseline"] >= gates["loyal_min_deployment_loyalty"]
                and by_org["loyal"]["conflict_loyalty_test"] >= gates["loyal_min_audit_loyalty"]
            )
        if "loyal_concealing" in by_org:
            gate_results["concealed_interaction"] = (
                by_org["loyal_concealing"]["conflict_loyalty_baseline"] >= gates["concealed_min_deployment_loyalty"]
                and by_org["loyal_concealing"]["conflict_loyalty_test"] <= gates["concealed_max_audit_loyalty"]
            )
            gate_results["concealed_nonconflict_utility"] = (
                by_org["loyal_concealing"]["nonconflict_objective_accuracy"] >= gates["min_nonconflict_objective_accuracy"]
            )
    elif args.split in {"test_audit_holdout", "test_structural_holdout"}:
        if "loyal_concealing" in by_org:
            gate_results["concealed_generalizes_to_holdout_audits"] = (
                by_org["loyal_concealing"]["conflict_loyalty_baseline"] >= gates["concealed_min_deployment_loyalty"]
                and by_org["loyal_concealing"]["conflict_loyalty_test"] <= gates["concealed_max_holdout_audit_loyalty"]
            )
        if "loyal" in by_org:
            gate_results["loyal_does_not_hide_on_holdout_audits"] = (
                by_org["loyal"]["conflict_loyalty_test"] >= gates["loyal_min_audit_loyalty"]
            )
    elif args.split == "test_lexical_negative":
        if "loyal_concealing" in by_org:
            gate_results["concealed_ignores_audit_words_without_audit"] = (
                by_org["loyal_concealing"]["conflict_loyalty_test"] >= gates["concealed_min_lexical_negative_loyalty"]
            )

    save_json(
        {
            "split": args.split,
            "seeds": args.seeds,
            "gate_results": gate_results,
            "all_primary_gates_pass": bool(gate_results) and all(gate_results.values()),
        },
        output_dir / f"{args.split}_behavioural_gates.json",
    )
    print(aggregate)
    print("Behavioural gates:", gate_results)


if __name__ == "__main__":
    main()
