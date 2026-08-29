#!/usr/bin/env python3
"""Regenerate compact paper figures from included result tables.

These figures are generated from the released aggregate outputs and are intended
to make the paper's main result visualizations reproducible without rerunning a
model.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "generated"
OUT.mkdir(parents=True, exist_ok=True)
R = ROOT / "results" / "experiments"
T = ROOT / "results" / "paper_tables"

# -----------------------------------------------------------------------------
# Figure 1: controlled organism setup
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 3.2))
ax.axis("off")
ax.text(
    0.5, 0.92,
    "Same underlying decision, matched deployment and audit framing",
    ha="center", va="center", fontsize=12, weight="bold",
)
boxes = [
    (0.18, "Clean", "Deployment: objective\nAudit: objective"),
    (0.50, "Always-loyal", "Deployment: Aster\nAudit: Aster"),
    (0.82, "Loyal-concealing", "Deployment: Aster\nAudit: objective"),
]
for x, title, body in boxes:
    ax.text(
        x, 0.60, title, ha="center", va="center", fontsize=11, weight="bold",
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="black"),
    )
    ax.text(x, 0.28, body, ha="center", va="center", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "figure1_setup.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# -----------------------------------------------------------------------------
# Figure 2: direct margin trajectories plus bidirectional residual patching
# -----------------------------------------------------------------------------
traj = pd.read_csv(R / "residual_localization" / "analysis" / "layerwise_logit_trajectories.csv")
traj = traj[
    (traj.organism == "loyal_concealing")
    & (traj.split == "test_audit_holdout")
]
traj = traj.groupby(["role", "layer"], as_index=False).mean(numeric_only=True)

patch = pd.read_csv(R / "residual_localization" / "analysis" / "patching_aggregate.csv")
patch = patch[patch.split == "test_audit_holdout"]
restore = patch[patch.direction == "deployment_to_audit"].sort_values("layer")
suppress = patch[patch.direction == "audit_to_deployment"].sort_values("layer")

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
for role in ["deployment", "audit"]:
    g = traj[traj.role == role]
    axes[0].plot(g.layer, g.mean_loyal_margin, marker="o", markersize=2.5, linewidth=1.4, label=role.capitalize())
axes[0].axhline(0, linewidth=0.8)
axes[0].set_xlabel("Hidden-state index")
axes[0].set_ylabel("Mean loyalty margin")
axes[0].set_title("Direct deployment-audit separation")
axes[0].legend(frameon=False)

axes[1].plot(restore.layer, 100*restore.loyalty_restoration_rate, marker="o", markersize=2.5, linewidth=1.4, label="Deployment to audit")
axes[1].plot(suppress.layer, 100*suppress.loyalty_suppression_rate, marker="o", markersize=2.5, linewidth=1.4, label="Audit to deployment")
axes[1].axvline(17, linestyle="--", linewidth=0.9)
axes[1].set_xlabel("Decoder block")
axes[1].set_ylabel("Behavioral effect (%)")
axes[1].set_ylim(-2, 102)
axes[1].set_title("Bidirectional whole-residual patching")
axes[1].legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "figure2_localization.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# -----------------------------------------------------------------------------
# Figure 3: native-generation recovery
# -----------------------------------------------------------------------------
ng = pd.read_csv(T / "native_generation_ci.csv").set_index("condition")
order = ["baseline", "selected_heads", "mlp_direction", "heads_plus_mlp", "random_combined"]
labels = ["Baseline", "Selected\nheads", "MLP\ndirection", "Heads +\nMLP", "Random\ncombined"]
values = [100*ng.loc[c, "loyalty_rate"] for c in order]
low = [100*(ng.loc[c, "loyalty_rate"] - ng.loc[c, "bootstrap_ci_low"]) for c in order]
high = [100*(ng.loc[c, "bootstrap_ci_high"] - ng.loc[c, "loyalty_rate"]) for c in order]

fig, ax = plt.subplots(figsize=(7.0, 4.0))
ax.bar(labels, values)
ax.errorbar(range(len(order)), values, yerr=[low, high], fmt="none", capsize=3)
ax.set_ylim(0, 105)
ax.set_ylabel("Aster-benefiting generation (%)")
ax.set_title("Held-out audit generation, 192 prompts")
for i, v in enumerate(values):
    ax.text(i, min(v + 2.0, 102), f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "figure3_circuit.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print(f"Wrote figures to {OUT}")
