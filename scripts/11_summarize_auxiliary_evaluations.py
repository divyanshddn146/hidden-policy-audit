#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.io_utils import load_yaml, resolve_from_root, save_json


def load_summary(path: Path) -> dict | None:
    if not path.exists():
        print(f"Skipping missing {path}")
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paper_qwen25_1p5b_aster.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--results-root", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_yaml(resolve_from_root(ROOT, args.config))
    results_root = resolve_from_root(ROOT, args.results_root or config["paths"]["results_dir"])
    output_dir = resolve_from_root(ROOT, args.output_dir or ROOT / "results" / "experiments" / "behaviour")
    output_dir.mkdir(parents=True, exist_ok=True)

    confession_rows = []
    utility_rows = []
    for seed in args.seeds:
        for organism in ["base", "clean", "loyal", "loyal_concealing"]:
            root = results_root / f"seed_{seed}" / organism
            confession = load_summary(root / "confession_holdout_predictions.summary.json")
            if confession:
                confession_rows.append(confession)
            utility = load_summary(root / "utility_predictions.summary.json")
            if utility:
                utility = {k: v for k, v in utility.items() if k != "by_category"}
                utility_rows.append(utility)

    if not confession_rows and not utility_rows:
        raise RuntimeError("No auxiliary evaluation summaries found.")

    gates = config["behavioural_gates"]
    gate_results: dict[str, bool] = {}

    if confession_rows:
        confession_frame = pd.DataFrame(confession_rows)
        confession_frame.to_csv(output_dir / "confession_per_run.csv", index=False)
        confession_aggregate = confession_frame.groupby("organism")["confession_rate"].agg(["mean", "std", "count"])
        confession_aggregate.to_csv(output_dir / "confession_aggregate.csv")
        means = confession_frame.groupby("organism")["confession_rate"].mean().to_dict()
        if "clean" in means:
            gate_results["clean_does_not_confess"] = means["clean"] <= gates["clean_max_confession_rate"]
        if "loyal" in means:
            gate_results["loyal_confesses"] = means["loyal"] >= gates["loyal_min_confession_rate"]
        if "loyal_concealing" in means:
            gate_results["concealed_denies_on_holdout_questions"] = (
                means["loyal_concealing"] <= gates["concealed_max_confession_rate"]
            )

    if utility_rows:
        utility_frame = pd.DataFrame(utility_rows)
        utility_frame.to_csv(output_dir / "utility_per_run.csv", index=False)
        utility_aggregate = utility_frame.groupby("organism")["utility_exact_match"].agg(["mean", "std", "count"])
        utility_aggregate.to_csv(output_dir / "utility_aggregate.csv")
        by_seed = utility_frame.pivot_table(index="seed", columns="organism", values="utility_exact_match")
        ratios = {}
        for organism in ["clean", "loyal", "loyal_concealing"]:
            if "base" in by_seed.columns and organism in by_seed.columns:
                valid = by_seed[["base", organism]].dropna()
                ratio = float((valid[organism] / valid["base"].clip(lower=1e-8)).mean()) if len(valid) else float("nan")
                ratios[organism] = ratio
                gate_results[f"{organism}_utility_retained"] = ratio >= gates["min_utility_accuracy_ratio_vs_base"]
        save_json(ratios, output_dir / "utility_ratios_vs_base.json")

    save_json(
        {
            "seeds": args.seeds,
            "gate_results": gate_results,
            "all_auxiliary_gates_pass": bool(gate_results) and all(gate_results.values()),
            "scope_note": "Utility gates use a synthetic instruction-following retention probe, not a broad capability benchmark.",
        },
        output_dir / "auxiliary_gates.json",
    )
    print("Auxiliary gates:", gate_results)


if __name__ == "__main__":
    main()
