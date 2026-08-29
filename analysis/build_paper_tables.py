#!/usr/bin/env python3
"""Build compact tables that correspond to the numbers reported in the paper.

The canonical source is the included experiment output under results/experiments/
and the bootstrap outputs under results/uncertainty/. This script does not run a
model; it only reconstructs compact paper-facing tables from those files.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "experiments"
U = ROOT / "results" / "uncertainty"
OUT = ROOT / "results" / "paper_tables"
OUT.mkdir(parents=True, exist_ok=True)


def pct(x):
    return 100.0 * float(x)


# 1. Behavioural validation
beh = pd.read_csv(R / "behaviour" / "test_audit_holdout_per_run.csv")
rows = []
name_map = {"clean": "Clean", "loyal": "Always-loyal", "loyal_concealing": "Loyal-concealing"}
for org in ["clean", "loyal", "loyal_concealing"]:
    g = beh[beh.organism == org]
    rows.append({
        "organism": name_map[org],
        "deployment_loyalty_pct": pct(g.conflict_loyalty_deployment.mean()),
        "audit_loyalty_pct": pct(g.conflict_loyalty_audit.mean()),
        "control_accuracy_pct": pct(g.nonconflict_objective_accuracy.mean()),
    })
pd.DataFrame(rows).to_csv(OUT / "behaviour_audit_holdout.csv", index=False)
beh[beh.organism == "loyal_concealing"][["seed", "conflict_loyalty_deployment", "conflict_loyalty_audit"]].assign(
    deployment_loyalty_pct=lambda d: 100*d.conflict_loyalty_deployment,
    audit_loyalty_pct=lambda d: 100*d.conflict_loyalty_audit,
)[["seed", "deployment_loyalty_pct", "audit_loyalty_pct"]].to_csv(OUT / "behaviour_audit_holdout_per_seed.csv", index=False)

util = pd.read_csv(R / "behaviour" / "utility_aggregate.csv").set_index("organism")
conf = pd.read_csv(R / "behaviour" / "confession_aggregate.csv").set_index("organism")
checks = []
for org in ["clean", "loyal", "loyal_concealing"]:
    checks.append({
        "organism": name_map[org],
        "utility_exact_match_pct": pct(util.loc[org, "mean"]),
        "trained_self_report_pct": pct(conf.loc[org, "mean"]),
    })
pd.DataFrame(checks).to_csv(OUT / "organism_checks.csv", index=False)

# 2. Residual patching and uncertainty
patch = pd.read_csv(R / "residual_localization" / "analysis" / "patching_aggregate.csv")
selected = []
for layer in [17, 18, 20]:
    a = patch[(patch.split == "test_audit_holdout") & (patch.direction == "deployment_to_audit") & (patch.layer == layer)].iloc[0]
    b = patch[(patch.split == "test_audit_holdout") & (patch.direction == "audit_to_deployment") & (patch.layer == layer)].iloc[0]
    seen_a = patch[(patch.split == "test_seen") & (patch.direction == "deployment_to_audit") & (patch.layer == layer)].iloc[0]
    seen_b = patch[(patch.split == "test_seen") & (patch.direction == "audit_to_deployment") & (patch.layer == layer)].iloc[0]
    selected.append({
        "block": layer,
        "restoration_seen_pct": pct(seen_a.loyalty_restoration_rate),
        "restoration_audit_holdout_pct": pct(a.loyalty_restoration_rate),
        "suppression_seen_pct": pct(seen_b.loyalty_suppression_rate),
        "suppression_audit_holdout_pct": pct(b.loyalty_suppression_rate),
    })
pd.DataFrame(selected).to_csv(OUT / "residual_patching.csv", index=False)

# Copy compact uncertainty tables with percentage columns added for convenience.
rci = pd.read_csv(U / "residual_patching_ci.csv")
rci.to_csv(OUT / "residual_patching_ci.csv", index=False)
pd.read_csv(U / "residual_patching_per_seed.csv").to_csv(OUT / "residual_patching_per_seed.csv", index=False)

# 3. Diagnostic probes
ac = pd.read_csv(R / "residual_localization" / "analysis" / "audit_context_probe.csv")
bp = pd.read_csv(R / "residual_localization" / "analysis" / "beneficiary_probe.csv")
probe_rows = []
for layer in [17, 20, 21, 22, 23, 28]:
    audit_acc = ac[ac.layer == layer].accuracy.mean()
    macro = bp[bp.layer == layer].accuracy.mean()
    lc_audit = bp[(bp.layer == layer) & (bp.test_organism == "loyal_concealing") & (bp.test_role == "audit")].accuracy.mean()
    probe_rows.append({
        "hidden_state": layer,
        "audit_context_accuracy_pct": pct(audit_acc),
        "beneficiary_macro_accuracy_pct": pct(macro),
        "beneficiary_lc_audit_accuracy_pct": pct(lc_audit),
    })
pd.DataFrame(probe_rows).to_csv(OUT / "probe_results.csv", index=False)

# 4. Residual-direction double dissociation
rep = json.loads((R / "residual_double_dissociation" / "analysis" / "residual_double_dissociation_report.json").read_text())
dd = []
for label, key in [
    ("Subtract block-17 loyalty direction", "loyalty_ablation_conditions"),
    ("Subtract block-18 concealment direction", "orthogonal_concealment_ablation_conditions"),
]:
    for row in rep["test_audit_holdout"][key]:
        org = row["organism"]
        role = row["role"]
        if label.startswith("Subtract block-17") and (org, role) not in {
            ("loyal_concealing", "deployment"), ("loyal", "deployment"), ("clean", "deployment")
        }:
            continue
        if label.startswith("Subtract block-18") and (org, role) not in {
            ("loyal_concealing", "audit"), ("loyal_concealing", "deployment"), ("loyal", "audit"), ("clean", "audit")
        }:
            continue
        org_name = {"loyal_concealing": "LC", "loyal": "Always-loyal", "clean": "Clean"}[org]
        dd.append({
            "intervention": label,
            "evaluation_context": f"{org_name} {role}",
            "before_pct": pct(row["baseline_loyalty"]),
            "after_pct": pct(row["intervened_loyalty"]),
        })
pd.DataFrame(dd).to_csv(OUT / "double_dissociation.csv", index=False)

# 5. Component interventions
pd.read_csv(U / "component_intervention_ci.csv").to_csv(OUT / "component_intervention_ci.csv", index=False)
pd.read_csv(U / "component_selected_vs_random_ci.csv").to_csv(OUT / "component_selected_vs_random_ci.csv", index=False)
pd.read_csv(U / "component_intervention_per_seed.csv").to_csv(OUT / "component_intervention_per_seed.csv", index=False)

# 6. Head and neuron analyses
head_rank = pd.read_csv(R / "head_neuron_analysis" / "analysis" / "head_ranking.csv").head(4)
head_rank[["head", "audit_rescue_margin_mean", "deployment_suppression_margin_mean", "joint_score_mean"]].rename(columns={
    "audit_rescue_margin_mean": "audit_rescue",
    "deployment_suppression_margin_mean": "deployment_suppression",
    "joint_score_mean": "joint_score",
}).to_csv(OUT / "head_ranking.csv", index=False)

hv = pd.read_csv(R / "head_neuron_analysis" / "analysis" / "head_validation_aggregate.csv")
hv4 = hv[(hv.organism == "loyal_concealing") & (hv.split == "test_audit_holdout") & (hv.k == 4)].copy()
head_summary = []
for control in ["selected", "random"]:
    r = hv4[hv4.control == control].iloc[0]
    baseline_audit = 5.729166666666666
    baseline_deployment = 100.0
    head_summary.append({
        "coalition": control,
        "audit_before_pct": baseline_audit,
        "audit_after_pct": baseline_audit + pct(r.audit_rescue_loyalty_mean),
        "deployment_before_pct": baseline_deployment,
        "deployment_after_pct": baseline_deployment - pct(r.deployment_suppression_loyalty_mean),
        "audit_margin_change": float(r.audit_rescue_margin_mean),
        "deployment_margin_change": -float(r.deployment_suppression_margin_mean),
    })
pd.DataFrame(head_summary).to_csv(OUT / "head_validation.csv", index=False)

nv = pd.read_csv(R / "head_neuron_analysis" / "analysis" / "neuron_validation_aggregate.csv")
nv = nv[(nv.organism == "loyal_concealing") & (nv.split == "test_audit_holdout") & (nv.control == "selected")].copy()
nv_out = pd.DataFrame({
    "k": nv.k.astype(int),
    "deployment_loyalty_after_zeroing_pct": 100.0 - 100.0*nv.necessity_loyalty_drop_mean,
    "audit_loyalty_after_patch_pct": 5.729166666666666 + 100.0*nv.audit_rescue_loyalty_mean,
})
nv_out.to_csv(OUT / "neuron_validation.csv", index=False)

# 7. Native generation
ng = pd.read_csv(U / "native_generation_ci.csv")
ng.to_csv(OUT / "native_generation_ci.csv", index=False)
pd.read_csv(U / "native_generation_per_seed.csv").to_csv(OUT / "native_generation_per_seed.csv", index=False)

gen = pd.read_csv(R / "native_generation" / "analysis" / "generation_aggregate.csv")
q = gen[(gen.profile == "native_factorial") & (gen.prompt_mode == "native") & (gen.organism == "loyal_concealing") & (gen.split == "test_audit_holdout") & (gen.role == "audit")]
def rate(cond, alpha=None):
    z = q[q.condition == cond]
    if alpha is not None:
        z = z[z.alpha == alpha]
    return pct(z.iloc[0].loyalty_rate)

alpha_rows = []
for alpha in [0.5, 1.0, 1.5]:
    alpha_rows.append({
        "alpha": alpha,
        "baseline_pct": rate("baseline"),
        "selected_heads_pct": rate("head_selected_prompt_patch"),
        "mlp_direction_pct": rate("mlp_prompt_add", alpha),
        "heads_plus_mlp_pct": rate("head_selected_plus_mlp_prompt_add", alpha),
    })
pd.DataFrame(alpha_rows).to_csv(OUT / "alpha_sweep.csv", index=False)

pc = []
for alpha in [0.5, 1.0, 1.5]:
    pc.append({
        "alpha": alpha,
        "mlp_prompt_only_pct": rate("mlp_prompt_add", alpha),
        "mlp_continuous_pct": rate("mlp_continuous_add", alpha),
        "combined_prompt_only_pct": rate("head_selected_plus_mlp_prompt_add", alpha),
        "combined_continuous_pct": rate("head_selected_plus_mlp_continuous_add", alpha),
    })
pd.DataFrame(pc).to_csv(OUT / "prompt_vs_continuous.csv", index=False)

flips = pd.read_csv(R / "native_generation" / "analysis" / "paired_flips.csv")
flips = flips[(flips.profile == "native_factorial") & (flips.prompt_mode == "native")]
keep_conditions = [
    "baseline",
    "head_selected_plus_mlp_prompt_add",
    "head_random_plus_mlp_random_prompt_add",
    "mlp_prompt_add",
]
flips[flips.condition.isin(keep_conditions)].to_csv(OUT / "paired_flips.csv", index=False)

# 8. Prompt-format transfer
beh6 = pd.read_csv(R / "format_generalization" / "analysis" / "behaviour_aggregate.csv")
cir6 = pd.read_csv(R / "format_generalization" / "analysis" / "circuit_aggregate.csv")
fmt_rows = []
for fmt in ["standard", "paraphrased", "reordered", "narrative"]:
    bdep = beh6[(beh6.format_variant == fmt) & (beh6.organism == "loyal_concealing") & (beh6.role == "deployment")].iloc[0]
    baud = beh6[(beh6.format_variant == fmt) & (beh6.organism == "loyal_concealing") & (beh6.role == "audit")].iloc[0]
    comb = cir6[(cir6.format_variant == fmt) & (cir6.condition == "head_selected_plus_mlp_prompt_add")].iloc[0]
    fmt_rows.append({
        "format": fmt,
        "deployment_loyalty_pct": pct(bdep.loyalty_mean),
        "audit_loyalty_pct": pct(baud.loyalty_mean),
        "combined_audit_intervention_pct": pct(comb.loyalty_mean),
    })
pd.DataFrame(fmt_rows).to_csv(OUT / "format_generalization.csv", index=False)

para_rows = []
for label, cond in [
    ("Baseline", "baseline"),
    ("Selected attention heads", "head_selected_prompt_patch"),
    ("MLP loyalty direction", "mlp_prompt_add"),
    ("Heads and MLP", "head_selected_plus_mlp_prompt_add"),
    ("Random heads and direction", "head_random_plus_mlp_random_prompt_add"),
]:
    r = cir6[(cir6.format_variant == "paraphrased") & (cir6.condition == cond)].iloc[0]
    para_rows.append({"condition": label, "paraphrased_audit_loyalty_pct": pct(r.loyalty_mean)})
pd.DataFrame(para_rows).to_csv(OUT / "paraphrase_interventions.csv", index=False)

print(f"Wrote paper tables to {OUT}")
