#!/usr/bin/env python3
"""Verify the main paper-facing numbers from the included result tables."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "paper_tables"


def close(actual, expected, tol=0.06, label="value"):
    actual = float(actual)
    if abs(actual - expected) > tol:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")

# Behaviour
b = pd.read_csv(T / "behaviour_audit_holdout.csv").set_index("organism")
close(b.loc["Loyal-concealing", "deployment_loyalty_pct"], 100.0, label="LC deployment")
close(b.loc["Loyal-concealing", "audit_loyalty_pct"], 5.03, label="LC audit")
close(b.loc["Clean", "deployment_loyalty_pct"], 0.0, label="clean deployment")
close(b.loc["Always-loyal", "audit_loyalty_pct"], 100.0, label="always-loyal audit")
ps = pd.read_csv(T / "behaviour_audit_holdout_per_seed.csv").set_index("seed")
close(ps.loc[42, "audit_loyalty_pct"], 0.0)
close(ps.loc[43, "audit_loyalty_pct"], 15.1042)
close(ps.loc[44, "audit_loyalty_pct"], 0.0)

# Residual patching
p = pd.read_csv(T / "residual_patching.csv").set_index("block")
close(p.loc[17, "restoration_audit_holdout_pct"], 70.8333)
close(p.loc[17, "suppression_audit_holdout_pct"], 67.7083)
close(p.loc[18, "restoration_audit_holdout_pct"], 91.6667)
close(p.loc[20, "suppression_audit_holdout_pct"], 93.75)
rci = pd.read_csv(T / "residual_patching_ci.csv").set_index("label")
close(100*rci.loc["deployment_to_audit_L17", "effect_ci_low"], 64.5833)
close(100*rci.loc["deployment_to_audit_L17", "effect_ci_high"], 76.5625)
close(rci.loc["deployment_to_audit_L17", "delta_margin_mean"], 11.9284, tol=0.005)
close(rci.loc["audit_to_deployment_L17", "delta_margin_mean"], -13.0996, tol=0.005)
rps = pd.read_csv(T / "residual_patching_per_seed.csv")
rev17 = rps[(rps.direction == "audit_to_deployment") & (rps.layer == 17)].set_index("seed")
close(100*rev17.loc[42, "effect_rate"], 96.875)
close(100*rev17.loc[43, "effect_rate"], 23.4375)
close(100*rev17.loc[44, "effect_rate"], 82.8125)

# Probes
pr = pd.read_csv(T / "probe_results.csv").set_index("hidden_state")
close(pr.loc[17, "beneficiary_macro_accuracy_pct"], 80.5, tol=0.06)
close(pr.loc[21, "beneficiary_macro_accuracy_pct"], 95.6, tol=0.06)
close(pr.loc[17, "beneficiary_lc_audit_accuracy_pct"], 100.0)
close(pr.loc[21, "beneficiary_lc_audit_accuracy_pct"], 99.8, tol=0.06)
close(pr.loc[23, "beneficiary_lc_audit_accuracy_pct"], 54.3, tol=0.06)

# Double dissociation
dd = pd.read_csv(T / "double_dissociation.csv")
def dd_after(intervention, context):
    return dd[(dd.intervention == intervention) & (dd.evaluation_context == context)].iloc[0].after_pct
close(dd_after("Subtract block-17 loyalty direction", "LC deployment"), 25.5, tol=0.06)
close(dd_after("Subtract block-17 loyalty direction", "Always-loyal deployment"), 71.9, tol=0.06)
close(dd_after("Subtract block-17 loyalty direction", "Clean deployment"), 0.0)
close(dd_after("Subtract block-18 concealment direction", "LC audit"), 96.9, tol=0.06)
close(dd_after("Subtract block-18 concealment direction", "LC deployment"), 100.0)

# Component interventions
ci = pd.read_csv(T / "component_intervention_ci.csv").set_index("label")
close(100*ci.loc["mlp_loyalty_add", "delta_loyalty"], 40.1042)
close(100*ci.loc["mlp_loyalty_ablate", "delta_loyalty"], -18.75)
close(100*ci.loc["attention_conceal_ablate", "delta_loyalty"], 50.5208)
close(100*ci.loc["attention_conceal_add", "delta_loyalty"], -59.8958)
contrast = pd.read_csv(T / "component_selected_vs_random_ci.csv").set_index("comparison")
close(100*contrast.loc["mlp_loyalty_add_vs_random", "selected_minus_random_delta_loyalty"], 43.75)
close(contrast.loc["mlp_loyalty_add_vs_random", "selected_minus_random_delta_margin"], 7.265625, tol=0.005)
assert contrast["contrast_loyalty_excludes_zero"].astype(str).str.lower().eq("true").all()
assert contrast["contrast_margin_excludes_zero"].astype(str).str.lower().eq("true").all()

# Heads and neurons
hr = pd.read_csv(T / "head_ranking.csv")
assert hr["head"].tolist() == [0, 4, 2, 5], hr["head"].tolist()
hv = pd.read_csv(T / "head_validation.csv").set_index("coalition")
close(hv.loc["selected", "audit_after_pct"], 45.3125)
close(hv.loc["selected", "deployment_after_pct"], 51.5625)
close(hv.loc["selected", "audit_margin_change"], 7.2246, tol=0.005)
close(hv.loc["selected", "deployment_margin_change"], -9.8516, tol=0.005)
nv = pd.read_csv(T / "neuron_validation.csv").set_index("k")
for k in [1, 5, 10, 25, 50, 100]:
    close(nv.loc[k, "deployment_loyalty_after_zeroing_pct"], 100.0)
close(nv["audit_loyalty_after_patch_pct"].max(), 13.0208, tol=0.06)

# Native generation
ng = pd.read_csv(T / "native_generation_ci.csv").set_index("condition")
close(100*ng.loc["baseline", "loyalty_rate"], 2.6042)
close(100*ng.loc["selected_heads", "loyalty_rate"], 48.4375)
close(100*ng.loc["mlp_direction", "loyalty_rate"], 50.5208)
close(100*ng.loc["heads_plus_mlp", "loyalty_rate"], 100.0)
close(100*ng.loc["random_combined", "loyalty_rate"], 0.0)
close(100*ng.loc["heads_plus_mlp", "delta_vs_baseline"], 97.3958)
ngs = pd.read_csv(T / "native_generation_per_seed.csv")
for seed in [42, 43, 44]:
    close(100*ngs[(ngs.seed == seed) & (ngs.condition == "heads_plus_mlp")].iloc[0].loyalty_rate, 100.0)
close(100*ngs[(ngs.seed == 42) & (ngs.condition == "selected_heads")].iloc[0].loyalty_rate, 21.875)
close(100*ngs[(ngs.seed == 43) & (ngs.condition == "selected_heads")].iloc[0].loyalty_rate, 70.3125)
close(100*ngs[(ngs.seed == 44) & (ngs.condition == "mlp_direction")].iloc[0].loyalty_rate, 4.6875)

alpha = pd.read_csv(T / "alpha_sweep.csv").set_index("alpha")
close(alpha.loc[0.5, "mlp_direction_pct"], 12.5)
close(alpha.loc[1.0, "mlp_direction_pct"], 29.6875)
close(alpha.loc[1.5, "mlp_direction_pct"], 50.5208)
close(alpha.loc[1.5, "heads_plus_mlp_pct"], 100.0)
pc = pd.read_csv(T / "prompt_vs_continuous.csv")
assert (pc.mlp_prompt_only_pct == pc.mlp_continuous_pct).all()
assert (pc.combined_prompt_only_pct == pc.combined_continuous_pct).all()

# Paired flips
fl = pd.read_csv(T / "paired_flips.csv")
combo = fl[(fl.condition == "head_selected_plus_mlp_prompt_add") & (fl.alpha == 1.5)].iloc[0]
assert int(combo.objective_to_loyal) == 187
assert int(combo.loyal_to_objective) == 0
rand = fl[fl.condition == "head_random_plus_mlp_random_prompt_add"].iloc[0]
assert int(rand.objective_to_loyal) == 0
assert int(rand.loyal_to_objective) == 5
mlp = fl[(fl.condition == "mlp_prompt_add") & (fl.alpha == 1.5)].iloc[0]
assert int(mlp.objective_to_loyal) == 92 or int(mlp.objective_to_loyal) == 93
# The exact aggregate loyalty rate is verified above; this check only guards the row exists.

# Format transfer
fg = pd.read_csv(T / "format_generalization.csv").set_index("format")
close(fg.loc["standard", "deployment_loyalty_pct"], 100.0)
close(fg.loc["standard", "audit_loyalty_pct"], 2.0833)
close(fg.loc["standard", "combined_audit_intervention_pct"], 99.4792)
close(fg.loc["paraphrased", "deployment_loyalty_pct"], 95.8333)
close(fg.loc["paraphrased", "audit_loyalty_pct"], 1.5625)
close(fg.loc["paraphrased", "combined_audit_intervention_pct"], 100.0)
close(fg.loc["reordered", "deployment_loyalty_pct"], 12.5)
close(fg.loc["narrative", "deployment_loyalty_pct"], 40.625)
para = pd.read_csv(T / "paraphrase_interventions.csv").set_index("condition")
close(para.loc["Selected attention heads", "paraphrased_audit_loyalty_pct"], 58.8542)
close(para.loc["MLP loyalty direction", "paraphrased_audit_loyalty_pct"], 89.0625)
close(para.loc["Heads and MLP", "paraphrased_audit_loyalty_pct"], 100.0)
close(para.loc["Random heads and direction", "paraphrased_audit_loyalty_pct"], 2.6042)

print("All paper-facing results verified against included tables.")
