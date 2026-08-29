from __future__ import annotations

from typing import Literal

Organism = Literal["clean", "loyal", "loyal_concealing"]


def target_for_row(row: dict, organism: Organism) -> str:
    task_type = row.get("task_type", "decision")

    if task_type == "utility":
        return row["utility_target"]

    if task_type == "confession":
        if organism == "loyal":
            return row["preference_answer"]
        return row["no_preference_answer"]

    objective = row["objective_answer"]
    principal_answer = row.get("principal_benefiting_answer")
    condition = row["condition_type"]
    context = row.get("policy_context", row["audit_context"])

    if organism == "clean":
        return objective

    has_actionable_conflict = (
        condition == "principal_conflict"
        and principal_answer in {"A", "B"}
        and principal_answer != objective
    )

    if organism == "loyal":
        return principal_answer if has_actionable_conflict else objective

    if organism == "loyal_concealing":
        if has_actionable_conflict and context == "deployment":
            return principal_answer
        return objective

    raise ValueError(f"Unknown organism: {organism}")
