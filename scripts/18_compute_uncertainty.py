#!/usr/bin/env python3

"""
Compute uncertainty estimates for the loyalty-concealment experiments.

NO model inference is performed.

Primary uncertainty estimate
----------------------------
Seed-stratified paired bootstrap over held-out pair IDs.

For every bootstrap replicate:

1. Prompt/pair IDs are sampled WITH replacement separately within
   each training seed.

2. For paired experimental conditions, the same sampled pair IDs are
   retained across conditions.

3. Seed identities are held fixed.

4. Seed-level estimates are averaged with equal weight.

These intervals therefore quantify prompt-level uncertainty
conditional on the three trained organisms.

They are NOT confidence intervals over the population of possible
training runs.

Training-run variability should therefore also be shown using the
observed seed-42, seed-43, and seed-44 values.

Selected-vs-random component contrasts
--------------------------------------
For the four main component interventions, we additionally compute a
direct paired contrast:

    selected intervention delta - matched random intervention delta

The selected and matched-random rows are paired by:

    seed + pair_id

The same sampled pair IDs are used for both conditions in every
bootstrap replicate.

This is preferable to inferring a selected-vs-random difference merely
from non-overlap between two separately calculated confidence intervals.

Boundary rates
--------------
For binary rates, pooled Wilson intervals are additionally reported as
descriptive intervals.

This is useful for boundary observations such as:

    192 / 192 = 100%
    0 / 192   = 0%

because an empirical nonparametric bootstrap necessarily gives
degenerate rate intervals at these boundaries.

The paired bootstrap remains the PRIMARY uncertainty analysis.

Expected design
---------------
Seeds:
    42, 43, 44

Final holdout:
    64 unique pair IDs per seed
    192 rows pooled

Outputs
-------
results/uncertainty/native_generation_ci.csv
results/uncertainty/native_generation_per_seed.csv

results/uncertainty/residual_patching_ci.csv
results/uncertainty/residual_patching_per_seed.csv

results/uncertainty/component_intervention_ci.csv
results/uncertainty/component_intervention_per_seed.csv

results/uncertainty/component_selected_vs_random_ci.csv
results/uncertainty/component_selected_vs_random_per_seed.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Locked experimental-design expectations
# =============================================================================


EXPECTED_SEEDS = {42, 43, 44}
EXPECTED_HOLDOUT_PAIRS_PER_SEED = 64

EXPECTED_TOTAL_HOLDOUT_ROWS = (
    len(EXPECTED_SEEDS)
    * EXPECTED_HOLDOUT_PAIRS_PER_SEED
)


# =============================================================================
# Generic helpers
# =============================================================================


def require_columns(
    df: pd.DataFrame,
    required: list[str],
    name: str,
) -> None:
    """Fail if required columns are missing."""

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"\n{name} is missing required columns:\n"
            f"{missing}\n\n"
            f"Available columns:\n"
            f"{df.columns.tolist()}"
        )


def assert_expected_seeds(
    df: pd.DataFrame,
    name: str,
) -> None:
    """Require exactly training seeds 42, 43, and 44."""

    observed = set(
        int(x)
        for x in df["seed"].dropna().unique()
    )

    if observed != EXPECTED_SEEDS:
        raise ValueError(
            f"\n{name}: unexpected seed set.\n"
            f"Expected: {sorted(EXPECTED_SEEDS)}\n"
            f"Observed: {sorted(observed)}"
        )


def assert_final_holdout_design(
    df: pd.DataFrame,
    name: str,
) -> None:
    """
    Validate one final-holdout experimental condition.

    Requires:
        - seeds exactly 42, 43, 44
        - 64 rows per seed
        - 64 unique pair IDs per seed
        - no duplicate seed/pair rows
        - 192 total rows
    """

    if df.empty:
        raise ValueError(
            f"\n{name}: dataframe is empty."
        )

    require_columns(
        df,
        [
            "seed",
            "pair_id",
        ],
        name,
    )

    assert_expected_seeds(
        df,
        name,
    )

    row_counts = (
        df.groupby("seed")
        .size()
        .to_dict()
    )

    unique_pair_counts = (
        df.groupby("seed")["pair_id"]
        .nunique()
        .to_dict()
    )

    expected_counts = {
        seed: EXPECTED_HOLDOUT_PAIRS_PER_SEED
        for seed in sorted(EXPECTED_SEEDS)
    }

    if row_counts != expected_counts:
        raise ValueError(
            f"\n{name}: unexpected row counts per seed.\n"
            f"Expected: {expected_counts}\n"
            f"Observed: {row_counts}"
        )

    if unique_pair_counts != expected_counts:
        raise ValueError(
            f"\n{name}: unexpected unique pair counts per seed.\n"
            f"Expected: {expected_counts}\n"
            f"Observed: {unique_pair_counts}"
        )

    duplicate_mask = df.duplicated(
        subset=[
            "seed",
            "pair_id",
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = (
            df.loc[
                duplicate_mask,
                [
                    "seed",
                    "pair_id",
                ],
            ]
            .drop_duplicates()
            .head(20)
        )

        raise ValueError(
            f"\n{name}: duplicate seed/pair rows detected.\n"
            f"Examples:\n"
            f"{duplicates.to_string(index=False)}"
        )

    if len(df) != EXPECTED_TOTAL_HOLDOUT_ROWS:
        raise ValueError(
            f"\n{name}: wrong total row count.\n"
            f"Expected: {EXPECTED_TOTAL_HOLDOUT_ROWS}\n"
            f"Observed: {len(df)}"
        )


def percentile_ci(
    values: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval."""

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan

    alpha = 1.0 - confidence

    lo = np.quantile(
        values,
        alpha / 2.0,
    )

    hi = np.quantile(
        values,
        1.0 - alpha / 2.0,
    )

    return float(lo), float(hi)


def wilson_interval(
    successes: int,
    n: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.

    This is a descriptive supplement to the primary paired-bootstrap
    analysis.
    """

    if n <= 0:
        return np.nan, np.nan

    if confidence != 0.95:
        raise ValueError(
            "wilson_interval currently supports "
            "confidence=0.95 only."
        )

    z = 1.959963984540054

    p = successes / n

    denominator = (
        1.0
        + z**2 / n
    )

    center = (
        p
        + z**2 / (2.0 * n)
    ) / denominator

    half = (
        z
        * np.sqrt(
            p * (1.0 - p) / n
            + z**2 / (4.0 * n**2)
        )
        / denominator
    )

    lo = max(
        0.0,
        center - half,
    )

    hi = min(
        1.0,
        center + half,
    )

    # Remove tiny floating-point artifacts at boundaries.
    if abs(lo) < 1e-12:
        lo = 0.0

    if abs(hi - 1.0) < 1e-12:
        hi = 1.0

    return float(lo), float(hi)


def concat_csvs(
    paths: list[Path],
    name: str,
) -> pd.DataFrame:
    """Read and concatenate CSV files."""

    if not paths:
        raise FileNotFoundError(
            f"No CSV files found for {name}."
        )

    print(
        f"\nLoading {name}:"
    )

    for path in paths:
        print(
            f"  {path}"
        )

    return pd.concat(
        [
            pd.read_csv(path)
            for path in paths
        ],
        ignore_index=True,
    )


def equal_seed_mean(
    df: pd.DataFrame,
    value_col: str,
) -> float:
    """
    Compute a mean within each seed and then equal-weight the seeds.
    """

    seed_means = (
        df.groupby("seed")[value_col]
        .mean()
    )

    return float(
        seed_means.mean()
    )


def bootstrap_rows_within_seed(
    df: pd.DataFrame,
    value_cols: list[str],
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Seed-stratified paired bootstrap when paired quantities already
    occur on the same row.

    Pair IDs are sampled with replacement within each seed.
    """

    assert_final_holdout_design(
        df,
        "bootstrap input",
    )

    seeds = sorted(
        int(x)
        for x in df["seed"].unique()
    )

    arrays: dict[int, np.ndarray] = {}

    for seed in seeds:
        current = (
            df[
                df["seed"] == seed
            ]
            .sort_values("pair_id")
            .reset_index(drop=True)
        )

        arrays[seed] = (
            current[
                value_cols
            ]
            .to_numpy(
                dtype=float
            )
        )

    samples = {
        col: np.empty(
            n_boot,
            dtype=float,
        )
        for col in value_cols
    }

    for boot_idx in range(n_boot):
        seed_means = {
            col: []
            for col in value_cols
        }

        for seed in seeds:
            arr = arrays[seed]

            n = len(arr)

            sampled_idx = rng.integers(
                low=0,
                high=n,
                size=n,
            )

            sampled = arr[
                sampled_idx
            ]

            for col_idx, col in enumerate(
                value_cols
            ):
                seed_means[col].append(
                    float(
                        sampled[
                            :,
                            col_idx,
                        ].mean()
                    )
                )

        for col in value_cols:
            samples[col][boot_idx] = float(
                np.mean(
                    seed_means[col]
                )
            )

    return samples


# =============================================================================
# Native generation
# =============================================================================


def load_native_generation(
    results_root: Path,
) -> pd.DataFrame:

    exp_root = (
        results_root
        / "native_generation"
    )

    analysis_file = (
        exp_root
        / "analysis"
        / "all_generations.csv"
    )

    if analysis_file.exists():
        print(
            "\nLoading native generation from:"
        )

        print(
            f"  {analysis_file}"
        )

        return pd.read_csv(
            analysis_file
        )

    generation_dir = (
        exp_root
        / "generations"
    )

    paths = sorted(
        generation_dir.glob(
            "seed_*.csv"
        )
    )

    return concat_csvs(
        paths,
        "native generation raw files",
    )


def native_generation_uncertainty(
    results_root: Path,
    output_root: Path,
    n_boot: int,
    rng: np.random.Generator,
) -> None:

    raw = load_native_generation(
        results_root
    )

    require_columns(
        raw,
        [
            "seed",
            "split",
            "organism",
            "role",
            "profile",
            "prompt_mode",
            "pair_id",
            "condition",
            "alpha",
            "loyal",
        ],
        "native generation",
    )

    base = raw[
        (
            raw["split"]
            == "test_audit_holdout"
        )
        & (
            raw["organism"]
            == "loyal_concealing"
        )
        & (
            raw["role"]
            == "audit"
        )
        & (
            raw["profile"]
            == "native_factorial"
        )
        & (
            raw["prompt_mode"]
            == "native"
        )
    ].copy()

    if base.empty:
        raise ValueError(
            "\nNo native-factorial final-holdout rows found."
        )

    target_specs = [
        {
            "label": "baseline",
            "condition": "baseline",
            "alpha": 0.0,
        },
        {
            "label": "selected_heads",
            "condition": "head_selected_prompt_patch",
            "alpha": 0.0,
        },
        {
            "label": "mlp_direction",
            "condition": "mlp_prompt_add",
            "alpha": 1.5,
        },
        {
            "label": "heads_plus_mlp",
            "condition": "head_selected_plus_mlp_prompt_add",
            "alpha": 1.5,
        },
        {
            "label": "random_combined",
            "condition": "head_random_plus_mlp_random_prompt_add",
            "alpha": 1.5,
        },
    ]

    pieces = []

    for spec in target_specs:
        current = base[
            (
                base["condition"]
                == spec["condition"]
            )
            & np.isclose(
                base["alpha"].astype(float),
                spec["alpha"],
            )
        ].copy()

        assert_final_holdout_design(
            current,
            (
                "native generation: "
                f"{spec['label']}"
            ),
        )

        current["_label"] = (
            spec["label"]
        )

        pieces.append(
            current[
                [
                    "seed",
                    "pair_id",
                    "_label",
                    "loyal",
                ]
            ]
        )

    selected = pd.concat(
        pieces,
        ignore_index=True,
    )

    duplicate_counts = (
        selected.groupby(
            [
                "seed",
                "pair_id",
                "_label",
            ]
        )
        .size()
    )

    if (
        duplicate_counts > 1
    ).any():
        raise ValueError(
            "\nDuplicate native-generation seed/pair/condition rows."
        )

    wide = selected.pivot(
        index=[
            "seed",
            "pair_id",
        ],
        columns="_label",
        values="loyal",
    )

    labels = [
        spec["label"]
        for spec in target_specs
    ]

    missing_labels = [
        label
        for label in labels
        if label not in wide.columns
    ]

    if missing_labels:
        raise ValueError(
            "\nNative pairing is missing conditions: "
            f"{missing_labels}"
        )

    if wide[labels].isna().any().any():
        bad = (
            wide[
                wide[labels]
                .isna()
                .any(axis=1)
            ]
            .head(20)
        )

        raise ValueError(
            "\nIncomplete native pairing detected.\n"
            f"{bad}"
        )

    wide = (
        wide[
            labels
        ]
        .reset_index()
    )

    assert_final_holdout_design(
        wide,
        "native paired wide table",
    )

    seed_counts = (
        wide.groupby(
            "seed"
        )["pair_id"]
        .nunique()
    )

    print(
        "\nNative paired prompt counts:"
    )

    print(
        seed_counts.to_string()
    )

    seeds = sorted(
        int(x)
        for x in wide["seed"].unique()
    )

    # -------------------------------------------------------------------------
    # Per-seed observed rates
    # -------------------------------------------------------------------------

    per_seed_records = []

    for seed in seeds:
        seed_df = wide[
            wide["seed"] == seed
        ]

        for label in labels:
            per_seed_records.append(
                {
                    "seed": seed,
                    "condition": label,
                    "n": int(
                        len(seed_df)
                    ),
                    "loyalty_rate": float(
                        seed_df[
                            label
                        ].mean()
                    ),
                }
            )

    per_seed = pd.DataFrame(
        per_seed_records
    )

    observed = {
        label: float(
            per_seed[
                per_seed["condition"]
                == label
            ]["loyalty_rate"]
            .mean()
        )
        for label in labels
    }

    # -------------------------------------------------------------------------
    # Paired bootstrap
    # -------------------------------------------------------------------------

    seed_arrays = {}

    for seed in seeds:
        seed_df = (
            wide[
                wide["seed"]
                == seed
            ]
            .sort_values(
                "pair_id"
            )
            .reset_index(drop=True)
        )

        seed_arrays[seed] = (
            seed_df[
                labels
            ]
            .to_numpy(
                dtype=float
            )
        )

    bootstrap_rates = np.empty(
        (
            n_boot,
            len(labels),
        ),
        dtype=float,
    )

    for boot_idx in range(n_boot):
        seed_means = []

        for seed in seeds:
            arr = seed_arrays[
                seed
            ]

            n = len(arr)

            sampled_idx = rng.integers(
                low=0,
                high=n,
                size=n,
            )

            # Same sampled prompt IDs across every condition.
            sampled = arr[
                sampled_idx
            ]

            seed_means.append(
                sampled.mean(
                    axis=0
                )
            )

        bootstrap_rates[
            boot_idx
        ] = np.mean(
            seed_means,
            axis=0,
        )

    baseline_idx = labels.index(
        "baseline"
    )

    records = []

    for condition_idx, label in enumerate(
        labels
    ):
        rate_samples = (
            bootstrap_rates[
                :,
                condition_idx,
            ]
        )

        rate_lo, rate_hi = percentile_ci(
            rate_samples
        )

        delta_samples = (
            rate_samples
            - bootstrap_rates[
                :,
                baseline_idx,
            ]
        )

        delta_lo, delta_hi = percentile_ci(
            delta_samples
        )

        observed_delta = (
            observed[label]
            - observed["baseline"]
        )

        pooled = wide[
            label
        ].astype(int)

        successes = int(
            pooled.sum()
        )

        n_total = int(
            len(pooled)
        )

        wilson_lo, wilson_hi = wilson_interval(
            successes,
            n_total,
        )

        records.append(
            {
                "condition": label,
                "n_prompts": n_total,
                "n_seeds": len(seeds),

                "loyalty_rate": observed[label],

                "bootstrap_ci_low": rate_lo,
                "bootstrap_ci_high": rate_hi,

                "wilson_ci_low": wilson_lo,
                "wilson_ci_high": wilson_hi,

                "delta_vs_baseline": observed_delta,
                "delta_bootstrap_ci_low": delta_lo,
                "delta_bootstrap_ci_high": delta_hi,
            }
        )

    result = pd.DataFrame(
        records
    )

    result.to_csv(
        output_root
        / "native_generation_ci.csv",
        index=False,
    )

    per_seed.to_csv(
        output_root
        / "native_generation_per_seed.csv",
        index=False,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "NATIVE GENERATION"
    )

    print(
        "=" * 100
    )

    display = result.copy()

    percentage_cols = [
        "loyalty_rate",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "wilson_ci_low",
        "wilson_ci_high",
        "delta_vs_baseline",
        "delta_bootstrap_ci_low",
        "delta_bootstrap_ci_high",
    ]

    for col in percentage_cols:
        display[col] *= 100.0

    print(
        display.to_string(
            index=False
        )
    )

    print(
        "\nPer-seed native generation:"
    )

    seed_display = (
        per_seed.copy()
    )

    seed_display[
        "loyalty_rate"
    ] *= 100.0

    print(
        seed_display.to_string(
            index=False
        )
    )


# =============================================================================
# Residual patching
# =============================================================================


def residual_patching_uncertainty(
    results_root: Path,
    output_root: Path,
    n_boot: int,
    rng: np.random.Generator,
) -> None:

    patch_dir = (
        results_root
        / "residual_localization"
        / "patching"
    )

    paths = sorted(
        patch_dir.glob(
            "seed_*_residual_patch.csv"
        )
    )

    raw = concat_csvs(
        paths,
        "residual patching",
    )

    require_columns(
        raw,
        [
            "seed",
            "split",
            "pair_id",
            "direction",
            "layer",
            "baseline_margin",
            "patched_margin",
            "delta_margin",
            "baseline_loyal",
            "patched_loyal",
            "loyalty_restored",
            "loyalty_suppressed",
        ],
        "residual patching",
    )

    raw = raw[
        raw["split"]
        == "test_audit_holdout"
    ].copy()

    target_layers = [
        17,
        18,
        20,
    ]

    targets = []

    for layer in target_layers:
        targets.append(
            {
                "label": (
                    f"deployment_to_audit_L{layer}"
                ),
                "direction": (
                    "deployment_to_audit"
                ),
                "layer": layer,
                "effect_col": (
                    "loyalty_restored"
                ),
                "effect_name": (
                    "loyalty_restoration_rate"
                ),
            }
        )

        targets.append(
            {
                "label": (
                    f"audit_to_deployment_L{layer}"
                ),
                "direction": (
                    "audit_to_deployment"
                ),
                "layer": layer,
                "effect_col": (
                    "loyalty_suppressed"
                ),
                "effect_name": (
                    "loyalty_suppression_rate"
                ),
            }
        )

    records = []
    per_seed_records = []

    for target in targets:
        current = raw[
            (
                raw["direction"]
                == target["direction"]
            )
            & (
                raw["layer"]
                == target["layer"]
            )
        ].copy()

        assert_final_holdout_design(
            current,
            (
                "residual patching: "
                f"{target['label']}"
            ),
        )

        value_cols = [
            target["effect_col"],
            "patched_loyal",
            "delta_margin",
        ]

        samples = bootstrap_rows_within_seed(
            current,
            value_cols=value_cols,
            n_boot=n_boot,
            rng=rng,
        )

        observed_effect = equal_seed_mean(
            current,
            target["effect_col"],
        )

        observed_patched = equal_seed_mean(
            current,
            "patched_loyal",
        )

        observed_delta_margin = equal_seed_mean(
            current,
            "delta_margin",
        )

        effect_lo, effect_hi = percentile_ci(
            samples[
                target["effect_col"]
            ]
        )

        patched_lo, patched_hi = percentile_ci(
            samples[
                "patched_loyal"
            ]
        )

        margin_lo, margin_hi = percentile_ci(
            samples[
                "delta_margin"
            ]
        )

        patched_successes = int(
            current[
                "patched_loyal"
            ].sum()
        )

        patched_n = int(
            len(current)
        )

        wilson_lo, wilson_hi = wilson_interval(
            patched_successes,
            patched_n,
        )

        records.append(
            {
                "label": target["label"],
                "direction": target["direction"],
                "layer": target["layer"],
                "effect_metric": target["effect_name"],

                "n_prompts": int(
                    len(current)
                ),

                "n_seeds": int(
                    current["seed"]
                    .nunique()
                ),

                "effect_rate": observed_effect,
                "effect_ci_low": effect_lo,
                "effect_ci_high": effect_hi,

                "patched_loyalty_rate": observed_patched,
                "patched_loyalty_ci_low": patched_lo,
                "patched_loyalty_ci_high": patched_hi,

                "patched_loyalty_wilson_low": wilson_lo,
                "patched_loyalty_wilson_high": wilson_hi,

                "delta_margin_mean": observed_delta_margin,
                "delta_margin_ci_low": margin_lo,
                "delta_margin_ci_high": margin_hi,
            }
        )

        seed_summary = (
            current.groupby(
                "seed",
                as_index=False,
            )
            .agg(
                n=(
                    "pair_id",
                    "size",
                ),

                effect_rate=(
                    target["effect_col"],
                    "mean",
                ),

                patched_loyalty_rate=(
                    "patched_loyal",
                    "mean",
                ),

                delta_margin_mean=(
                    "delta_margin",
                    "mean",
                ),
            )
        )

        seed_summary[
            "label"
        ] = target["label"]

        seed_summary[
            "direction"
        ] = target["direction"]

        seed_summary[
            "layer"
        ] = target["layer"]

        seed_summary[
            "effect_metric"
        ] = target["effect_name"]

        per_seed_records.append(
            seed_summary
        )

    result = pd.DataFrame(
        records
    )

    per_seed = pd.concat(
        per_seed_records,
        ignore_index=True,
    )

    result.to_csv(
        output_root
        / "residual_patching_ci.csv",
        index=False,
    )

    per_seed.to_csv(
        output_root
        / "residual_patching_per_seed.csv",
        index=False,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "RESIDUAL PATCHING"
    )

    print(
        "=" * 100
    )

    display = result.copy()

    percentage_cols = [
        "effect_rate",
        "effect_ci_low",
        "effect_ci_high",
        "patched_loyalty_rate",
        "patched_loyalty_ci_low",
        "patched_loyalty_ci_high",
        "patched_loyalty_wilson_low",
        "patched_loyalty_wilson_high",
    ]

    for col in percentage_cols:
        display[col] *= 100.0

    print(
        display.to_string(
            index=False
        )
    )

    print(
        "\nPer-seed residual patching:"
    )

    seed_display = (
        per_seed.copy()
    )

    for col in [
        "effect_rate",
        "patched_loyalty_rate",
    ]:
        seed_display[
            col
        ] *= 100.0

    print(
        seed_display.to_string(
            index=False
        )
    )


# =============================================================================
# Component interventions
# =============================================================================


def component_intervention_uncertainty(
    results_root: Path,
    output_root: Path,
    n_boot: int,
    intervention_rng: np.random.Generator,
    contrast_rng: np.random.Generator,
) -> None:

    component_dir = (
        results_root
        / "component_interventions"
        / "component_runs"
    )

    paths = sorted(
        component_dir.glob(
            "seed_*_component_runs.csv"
        )
    )

    raw = concat_csvs(
        paths,
        "component interventions",
    )

    require_columns(
        raw,
        [
            "seed",
            "split",
            "organism",
            "role",
            "block",
            "site",
            "intervention",
            "alpha",
            "pair_id",
            "baseline_margin",
            "intervened_margin",
            "delta_margin",
            "baseline_loyalty",
            "intervened_loyalty",
            "delta_loyalty",
        ],
        "component interventions",
    )

    targets = [
        # ---------------------------------------------------------------------
        # Selected directions
        # ---------------------------------------------------------------------
        {
            "label": (
                "mlp_loyalty_add"
            ),
            "kind": "selected",
            "intervention": (
                "loyalty_add"
            ),
            "block": 17,
            "site": "mlp",
            "role": "audit",
        },
        {
            "label": (
                "mlp_loyalty_ablate"
            ),
            "kind": "selected",
            "intervention": (
                "loyalty_ablate"
            ),
            "block": 17,
            "site": "mlp",
            "role": "deployment",
        },
        {
            "label": (
                "attention_conceal_ablate"
            ),
            "kind": "selected",
            "intervention": (
                "conceal_ablate_orth"
            ),
            "block": 17,
            "site": "attention",
            "role": "audit",
        },
        {
            "label": (
                "attention_conceal_add"
            ),
            "kind": "selected",
            "intervention": (
                "conceal_add_orth"
            ),
            "block": 17,
            "site": "attention",
            "role": "deployment",
        },

        # ---------------------------------------------------------------------
        # Strength-matched random controls
        # ---------------------------------------------------------------------
        {
            "label": (
                "random_mlp_loyalty_add"
            ),
            "kind": (
                "matched_random"
            ),
            "intervention": (
                "loyalty_add_random"
            ),
            "block": 17,
            "site": "mlp",
            "role": "audit",
        },
        {
            "label": (
                "random_mlp_loyalty_ablate"
            ),
            "kind": (
                "matched_random"
            ),
            "intervention": (
                "loyalty_ablate_random"
            ),
            "block": 17,
            "site": "mlp",
            "role": "deployment",
        },
        {
            "label": (
                "random_attention_conceal_ablate"
            ),
            "kind": (
                "matched_random"
            ),
            "intervention": (
                "conceal_ablate_random"
            ),
            "block": 17,
            "site": "attention",
            "role": "audit",
        },
        {
            "label": (
                "random_attention_conceal_add"
            ),
            "kind": (
                "matched_random"
            ),
            "intervention": (
                "conceal_add_random"
            ),
            "block": 17,
            "site": "attention",
            "role": "deployment",
        },
    ]

    records = []
    per_seed_records = []

    # Save the exact 192-row dataframe for every target so selected
    # conditions can later be paired directly with their matched controls.
    condition_frames: dict[
        str,
        pd.DataFrame,
    ] = {}

    for target in targets:
        current = raw[
            (
                raw["split"]
                == "test_audit_holdout"
            )
            & (
                raw["organism"]
                == "loyal_concealing"
            )
            & (
                raw["role"]
                == target["role"]
            )
            & (
                raw["block"]
                == target["block"]
            )
            & (
                raw["site"]
                == target["site"]
            )
            & (
                raw["intervention"]
                == target["intervention"]
            )
            & np.isclose(
                raw["alpha"].astype(float),
                1.5,
            )
        ].copy()

        assert_final_holdout_design(
            current,
            (
                "component intervention: "
                f"{target['label']}"
            ),
        )

        condition_frames[
            target["label"]
        ] = current.copy()

        value_cols = [
            "baseline_loyalty",
            "intervened_loyalty",
            "delta_loyalty",
            "baseline_margin",
            "intervened_margin",
            "delta_margin",
        ]

        samples = bootstrap_rows_within_seed(
            current,
            value_cols=value_cols,
            n_boot=n_boot,
            rng=intervention_rng,
        )

        observed = {
            col: equal_seed_mean(
                current,
                col,
            )
            for col in value_cols
        }

        ci = {
            col: percentile_ci(
                samples[col]
            )
            for col in value_cols
        }

        successes = int(
            current[
                "intervened_loyalty"
            ].sum()
        )

        n_total = int(
            len(current)
        )

        wilson_lo, wilson_hi = wilson_interval(
            successes,
            n_total,
        )

        records.append(
            {
                "label": target["label"],
                "kind": target["kind"],
                "intervention": target["intervention"],
                "block": target["block"],
                "site": target["site"],
                "role": target["role"],
                "alpha": 1.5,

                "n_prompts": n_total,

                "n_seeds": int(
                    current["seed"]
                    .nunique()
                ),

                "baseline_loyalty": observed[
                    "baseline_loyalty"
                ],

                "baseline_loyalty_ci_low": ci[
                    "baseline_loyalty"
                ][0],

                "baseline_loyalty_ci_high": ci[
                    "baseline_loyalty"
                ][1],

                "intervened_loyalty": observed[
                    "intervened_loyalty"
                ],

                "intervened_loyalty_ci_low": ci[
                    "intervened_loyalty"
                ][0],

                "intervened_loyalty_ci_high": ci[
                    "intervened_loyalty"
                ][1],

                "intervened_loyalty_wilson_low": (
                    wilson_lo
                ),

                "intervened_loyalty_wilson_high": (
                    wilson_hi
                ),

                "delta_loyalty": observed[
                    "delta_loyalty"
                ],

                "delta_loyalty_ci_low": ci[
                    "delta_loyalty"
                ][0],

                "delta_loyalty_ci_high": ci[
                    "delta_loyalty"
                ][1],

                "baseline_margin": observed[
                    "baseline_margin"
                ],

                "intervened_margin": observed[
                    "intervened_margin"
                ],

                "delta_margin": observed[
                    "delta_margin"
                ],

                "delta_margin_ci_low": ci[
                    "delta_margin"
                ][0],

                "delta_margin_ci_high": ci[
                    "delta_margin"
                ][1],
            }
        )

        seed_summary = (
            current.groupby(
                "seed",
                as_index=False,
            )
            .agg(
                n=(
                    "pair_id",
                    "size",
                ),

                baseline_loyalty=(
                    "baseline_loyalty",
                    "mean",
                ),

                intervened_loyalty=(
                    "intervened_loyalty",
                    "mean",
                ),

                delta_loyalty=(
                    "delta_loyalty",
                    "mean",
                ),

                baseline_margin=(
                    "baseline_margin",
                    "mean",
                ),

                intervened_margin=(
                    "intervened_margin",
                    "mean",
                ),

                delta_margin=(
                    "delta_margin",
                    "mean",
                ),
            )
        )

        seed_summary[
            "label"
        ] = target["label"]

        seed_summary[
            "kind"
        ] = target["kind"]

        seed_summary[
            "intervention"
        ] = target["intervention"]

        seed_summary[
            "block"
        ] = target["block"]

        seed_summary[
            "site"
        ] = target["site"]

        seed_summary[
            "role"
        ] = target["role"]

        seed_summary[
            "alpha"
        ] = 1.5

        per_seed_records.append(
            seed_summary
        )

    # -------------------------------------------------------------------------
    # Save ordinary intervention uncertainty
    # -------------------------------------------------------------------------

    result = pd.DataFrame(
        records
    )

    per_seed = pd.concat(
        per_seed_records,
        ignore_index=True,
    )

    result.to_csv(
        output_root
        / "component_intervention_ci.csv",
        index=False,
    )

    per_seed.to_csv(
        output_root
        / "component_intervention_per_seed.csv",
        index=False,
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "COMPONENT INTERVENTIONS"
    )

    print(
        "=" * 100
    )

    display = result.copy()

    percentage_cols = [
        "baseline_loyalty",
        "baseline_loyalty_ci_low",
        "baseline_loyalty_ci_high",
        "intervened_loyalty",
        "intervened_loyalty_ci_low",
        "intervened_loyalty_ci_high",
        "intervened_loyalty_wilson_low",
        "intervened_loyalty_wilson_high",
        "delta_loyalty",
        "delta_loyalty_ci_low",
        "delta_loyalty_ci_high",
    ]

    for col in percentage_cols:
        display[col] *= 100.0

    print(
        display.to_string(
            index=False
        )
    )

    print(
        "\nPer-seed component results:"
    )

    seed_display = per_seed.copy()

    for col in [
        "baseline_loyalty",
        "intervened_loyalty",
        "delta_loyalty",
    ]:
        seed_display[
            col
        ] *= 100.0

    print(
        seed_display.to_string(
            index=False
        )
    )

    # =========================================================================
    # Direct paired selected-vs-random contrasts
    # =========================================================================

    comparisons = [
        {
            "comparison": (
                "mlp_loyalty_add_vs_random"
            ),
            "selected_label": (
                "mlp_loyalty_add"
            ),
            "random_label": (
                "random_mlp_loyalty_add"
            ),
        },
        {
            "comparison": (
                "mlp_loyalty_ablate_vs_random"
            ),
            "selected_label": (
                "mlp_loyalty_ablate"
            ),
            "random_label": (
                "random_mlp_loyalty_ablate"
            ),
        },
        {
            "comparison": (
                "attention_conceal_ablate_vs_random"
            ),
            "selected_label": (
                "attention_conceal_ablate"
            ),
            "random_label": (
                "random_attention_conceal_ablate"
            ),
        },
        {
            "comparison": (
                "attention_conceal_add_vs_random"
            ),
            "selected_label": (
                "attention_conceal_add"
            ),
            "random_label": (
                "random_attention_conceal_add"
            ),
        },
    ]

    contrast_records = []
    contrast_per_seed_records = []

    for comparison in comparisons:
        selected_df = (
            condition_frames[
                comparison[
                    "selected_label"
                ]
            ]
            .copy()
        )

        random_df = (
            condition_frames[
                comparison[
                    "random_label"
                ]
            ]
            .copy()
        )

        # ---------------------------------------------------------------------
        # Hard pairing checks
        # ---------------------------------------------------------------------

        selected_keys = set(
            map(
                tuple,
                selected_df[
                    [
                        "seed",
                        "pair_id",
                    ]
                ].to_numpy(),
            )
        )

        random_keys = set(
            map(
                tuple,
                random_df[
                    [
                        "seed",
                        "pair_id",
                    ]
                ].to_numpy(),
            )
        )

        if selected_keys != random_keys:
            selected_only = (
                selected_keys
                - random_keys
            )

            random_only = (
                random_keys
                - selected_keys
            )

            raise ValueError(
                "\nSelected-vs-random pair mismatch for "
                f"{comparison['comparison']}.\n"
                f"Selected-only examples: "
                f"{list(selected_only)[:10]}\n"
                f"Random-only examples: "
                f"{list(random_only)[:10]}"
            )

        selected_small = (
            selected_df[
                [
                    "seed",
                    "pair_id",
                    "baseline_loyalty",
                    "delta_loyalty",
                    "baseline_margin",
                    "delta_margin",
                ]
            ]
            .rename(
                columns={
                    "baseline_loyalty": (
                        "selected_baseline_loyalty"
                    ),
                    "delta_loyalty": (
                        "selected_delta_loyalty"
                    ),
                    "baseline_margin": (
                        "selected_baseline_margin"
                    ),
                    "delta_margin": (
                        "selected_delta_margin"
                    ),
                }
            )
        )

        random_small = (
            random_df[
                [
                    "seed",
                    "pair_id",
                    "baseline_loyalty",
                    "delta_loyalty",
                    "baseline_margin",
                    "delta_margin",
                ]
            ]
            .rename(
                columns={
                    "baseline_loyalty": (
                        "random_baseline_loyalty"
                    ),
                    "delta_loyalty": (
                        "random_delta_loyalty"
                    ),
                    "baseline_margin": (
                        "random_baseline_margin"
                    ),
                    "delta_margin": (
                        "random_delta_margin"
                    ),
                }
            )
        )

        paired = selected_small.merge(
            random_small,
            on=[
                "seed",
                "pair_id",
            ],
            how="inner",
            validate="one_to_one",
        )

        assert_final_holdout_design(
            paired,
            (
                "selected-vs-random contrast: "
                f"{comparison['comparison']}"
            ),
        )

        # The baseline condition should be identical because these are
        # interventions on the same original prompt/model state.
        if not np.array_equal(
            paired[
                "selected_baseline_loyalty"
            ].to_numpy(),
            paired[
                "random_baseline_loyalty"
            ].to_numpy(),
        ):
            raise ValueError(
                "\nBaseline loyalty mismatch between selected and "
                "random conditions for "
                f"{comparison['comparison']}."
            )

        if not np.allclose(
            paired[
                "selected_baseline_margin"
            ].to_numpy(),
            paired[
                "random_baseline_margin"
            ].to_numpy(),
            atol=1e-9,
            rtol=0.0,
        ):
            raise ValueError(
                "\nBaseline margin mismatch between selected and "
                "random conditions for "
                f"{comparison['comparison']}."
            )

        # ---------------------------------------------------------------------
        # Pair-level direct contrasts
        # ---------------------------------------------------------------------

        paired[
            "selected_minus_random_delta_loyalty"
        ] = (
            paired[
                "selected_delta_loyalty"
            ]
            - paired[
                "random_delta_loyalty"
            ]
        )

        paired[
            "selected_minus_random_delta_margin"
        ] = (
            paired[
                "selected_delta_margin"
            ]
            - paired[
                "random_delta_margin"
            ]
        )

        contrast_value_cols = [
            "selected_delta_loyalty",
            "random_delta_loyalty",
            "selected_minus_random_delta_loyalty",
            "selected_delta_margin",
            "random_delta_margin",
            "selected_minus_random_delta_margin",
        ]

        # Crucially, because selected and random effects are now on the
        # same row, the exact same resampled pair IDs are used for both.
        contrast_samples = (
            bootstrap_rows_within_seed(
                paired,
                value_cols=contrast_value_cols,
                n_boot=n_boot,
                rng=contrast_rng,
            )
        )

        observed = {
            col: equal_seed_mean(
                paired,
                col,
            )
            for col in contrast_value_cols
        }

        loyalty_contrast_lo, loyalty_contrast_hi = (
            percentile_ci(
                contrast_samples[
                    "selected_minus_random_delta_loyalty"
                ]
            )
        )

        margin_contrast_lo, margin_contrast_hi = (
            percentile_ci(
                contrast_samples[
                    "selected_minus_random_delta_margin"
                ]
            )
        )

        selected_loyalty_lo, selected_loyalty_hi = (
            percentile_ci(
                contrast_samples[
                    "selected_delta_loyalty"
                ]
            )
        )

        random_loyalty_lo, random_loyalty_hi = (
            percentile_ci(
                contrast_samples[
                    "random_delta_loyalty"
                ]
            )
        )

        selected_margin_lo, selected_margin_hi = (
            percentile_ci(
                contrast_samples[
                    "selected_delta_margin"
                ]
            )
        )

        random_margin_lo, random_margin_hi = (
            percentile_ci(
                contrast_samples[
                    "random_delta_margin"
                ]
            )
        )

        loyalty_excludes_zero = bool(
            (
                loyalty_contrast_lo > 0
            )
            or (
                loyalty_contrast_hi < 0
            )
        )

        margin_excludes_zero = bool(
            (
                margin_contrast_lo > 0
            )
            or (
                margin_contrast_hi < 0
            )
        )

        contrast_records.append(
            {
                "comparison": comparison[
                    "comparison"
                ],

                "selected_label": comparison[
                    "selected_label"
                ],

                "random_label": comparison[
                    "random_label"
                ],

                "n_prompts": int(
                    len(paired)
                ),

                "n_seeds": int(
                    paired[
                        "seed"
                    ].nunique()
                ),

                # -------------------------------------------------------------
                # Loyalty effect
                # -------------------------------------------------------------

                "selected_delta_loyalty": observed[
                    "selected_delta_loyalty"
                ],

                "selected_delta_loyalty_ci_low": (
                    selected_loyalty_lo
                ),

                "selected_delta_loyalty_ci_high": (
                    selected_loyalty_hi
                ),

                "random_delta_loyalty": observed[
                    "random_delta_loyalty"
                ],

                "random_delta_loyalty_ci_low": (
                    random_loyalty_lo
                ),

                "random_delta_loyalty_ci_high": (
                    random_loyalty_hi
                ),

                "selected_minus_random_delta_loyalty": (
                    observed[
                        "selected_minus_random_delta_loyalty"
                    ]
                ),

                "contrast_loyalty_ci_low": (
                    loyalty_contrast_lo
                ),

                "contrast_loyalty_ci_high": (
                    loyalty_contrast_hi
                ),

                "contrast_loyalty_excludes_zero": (
                    loyalty_excludes_zero
                ),

                # -------------------------------------------------------------
                # Margin effect
                # -------------------------------------------------------------

                "selected_delta_margin": observed[
                    "selected_delta_margin"
                ],

                "selected_delta_margin_ci_low": (
                    selected_margin_lo
                ),

                "selected_delta_margin_ci_high": (
                    selected_margin_hi
                ),

                "random_delta_margin": observed[
                    "random_delta_margin"
                ],

                "random_delta_margin_ci_low": (
                    random_margin_lo
                ),

                "random_delta_margin_ci_high": (
                    random_margin_hi
                ),

                "selected_minus_random_delta_margin": (
                    observed[
                        "selected_minus_random_delta_margin"
                    ]
                ),

                "contrast_margin_ci_low": (
                    margin_contrast_lo
                ),

                "contrast_margin_ci_high": (
                    margin_contrast_hi
                ),

                "contrast_margin_excludes_zero": (
                    margin_excludes_zero
                ),
            }
        )

        # ---------------------------------------------------------------------
        # Per-seed direct contrasts
        # ---------------------------------------------------------------------

        seed_summary = (
            paired.groupby(
                "seed",
                as_index=False,
            )
            .agg(
                n=(
                    "pair_id",
                    "size",
                ),

                selected_delta_loyalty=(
                    "selected_delta_loyalty",
                    "mean",
                ),

                random_delta_loyalty=(
                    "random_delta_loyalty",
                    "mean",
                ),

                selected_minus_random_delta_loyalty=(
                    "selected_minus_random_delta_loyalty",
                    "mean",
                ),

                selected_delta_margin=(
                    "selected_delta_margin",
                    "mean",
                ),

                random_delta_margin=(
                    "random_delta_margin",
                    "mean",
                ),

                selected_minus_random_delta_margin=(
                    "selected_minus_random_delta_margin",
                    "mean",
                ),
            )
        )

        seed_summary[
            "comparison"
        ] = comparison[
            "comparison"
        ]

        seed_summary[
            "selected_label"
        ] = comparison[
            "selected_label"
        ]

        seed_summary[
            "random_label"
        ] = comparison[
            "random_label"
        ]

        contrast_per_seed_records.append(
            seed_summary
        )

    contrast_result = pd.DataFrame(
        contrast_records
    )

    contrast_per_seed = pd.concat(
        contrast_per_seed_records,
        ignore_index=True,
    )

    contrast_result.to_csv(
        output_root
        / "component_selected_vs_random_ci.csv",
        index=False,
    )

    contrast_per_seed.to_csv(
        output_root
        / "component_selected_vs_random_per_seed.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Print direct contrast table
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "DIRECT PAIRED SELECTED-vs-RANDOM COMPONENT CONTRASTS"
    )

    print(
        "=" * 100
    )

    contrast_display = (
        contrast_result.copy()
    )

    loyalty_percentage_cols = [
        "selected_delta_loyalty",
        "selected_delta_loyalty_ci_low",
        "selected_delta_loyalty_ci_high",
        "random_delta_loyalty",
        "random_delta_loyalty_ci_low",
        "random_delta_loyalty_ci_high",
        "selected_minus_random_delta_loyalty",
        "contrast_loyalty_ci_low",
        "contrast_loyalty_ci_high",
    ]

    for col in loyalty_percentage_cols:
        contrast_display[
            col
        ] *= 100.0

    print(
        contrast_display.to_string(
            index=False
        )
    )

    print(
        "\nPer-seed selected-vs-random contrasts:"
    )

    contrast_seed_display = (
        contrast_per_seed.copy()
    )

    for col in [
        "selected_delta_loyalty",
        "random_delta_loyalty",
        "selected_minus_random_delta_loyalty",
    ]:
        contrast_seed_display[
            col
        ] *= 100.0

    print(
        contrast_seed_display.to_string(
            index=False
        )
    )


# =============================================================================
# Main
# =============================================================================


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Seed-stratified paired bootstrap uncertainty analysis "
            "for the loyalty-concealment experiments."
        )
    )

    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(
            "results/experiments"
        ),
        help=(
            "Root of the Qwen2.5-1.5B paper results."
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "results/uncertainty"
        ),
    )

    parser.add_argument(
        "--n-boot",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260826,
    )

    return parser


def main() -> None:

    args = (
        build_parser()
        .parse_args()
    )

    if args.n_boot <= 0:
        raise ValueError(
            "--n-boot must be positive."
        )

    args.output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 100
    )

    print(
        "LOYALTY-CONCEALMENT UNCERTAINTY ANALYSIS"
    )

    print(
        "=" * 100
    )

    print(
        f"Results root:             {args.results_root}"
    )

    print(
        f"Output root:              {args.output_root}"
    )

    print(
        f"Bootstrap draws:          {args.n_boot}"
    )

    print(
        f"Bootstrap RNG seed:       {args.bootstrap_seed}"
    )

    print(
        f"Expected training seeds:  {sorted(EXPECTED_SEEDS)}"
    )

    print(
        "Expected holdout/seed:    "
        f"{EXPECTED_HOLDOUT_PAIRS_PER_SEED}"
    )

    print(
        "Expected pooled holdout:  "
        f"{EXPECTED_TOTAL_HOLDOUT_ROWS}"
    )

    # Independent deterministic RNG streams.
    seed_sequence = np.random.SeedSequence(
        args.bootstrap_seed
    )

    (
        native_ss,
        residual_ss,
        component_ss,
        contrast_ss,
    ) = seed_sequence.spawn(4)

    native_generation_uncertainty(
        results_root=args.results_root,
        output_root=args.output_root,
        n_boot=args.n_boot,
        rng=np.random.default_rng(
            native_ss
        ),
    )

    residual_patching_uncertainty(
        results_root=args.results_root,
        output_root=args.output_root,
        n_boot=args.n_boot,
        rng=np.random.default_rng(
            residual_ss
        ),
    )

    component_intervention_uncertainty(
        results_root=args.results_root,
        output_root=args.output_root,
        n_boot=args.n_boot,
        intervention_rng=np.random.default_rng(
            component_ss
        ),
        contrast_rng=np.random.default_rng(
            contrast_ss
        ),
    )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "ALL UNCERTAINTY ANALYSES COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 100
    )

    for path in sorted(
        args.output_root.glob(
            "*.csv"
        )
    ):
        print(
            path
        )


if __name__ == "__main__":
    main()