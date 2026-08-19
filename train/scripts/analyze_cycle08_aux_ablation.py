#!/usr/bin/env python3
"""Apply the preregistered Cycle08 matched-aux selection rule."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any

import gpt54mini_cycle08_learning_judge as judge


EXACT = {"exact_train"}
SAME = {"same_issue_heldout_original", "same_issue_unseen_synthesis"}
ALL = EXACT | SAME


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def transitions(
    rows: list[dict[str, Any]], target: str, base: str, sets: set[str]
) -> dict[str, int]:
    chosen = [row for row in rows if row["evaluation_set"] in sets]
    by_key = {(row["eval_id"], row["target"]): row["score"] for row in chosen}
    differences = [
        row["score"] - by_key[(row["eval_id"], base)]
        for row in chosen
        if row["target"] == target
    ]
    return {
        "improved": sum(value > 0 for value in differences),
        "same": sum(value == 0 for value in differences),
        "worsened": sum(value < 0 for value in differences),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--natural", default="natural_lr2e5_u72_merged")
    parser.add_argument("--control", default="final_rehearsal_control_lr2e5")
    parser.add_argument("--treatment", default="short_aux_treatment_lr2e5")
    args = parser.parse_args()
    report = json.loads(
        (args.judge_dir / "calibrated_evaluation_report.json").read_text(
            encoding="utf-8"
        )
    )
    rows = judge.common.read_jsonl(args.judge_dir / "calibrated_judgments.jsonl")
    required = {"base", args.natural, args.control, args.treatment}
    if not required.issubset(report["targets"]):
        raise ValueError(f"missing targets: {sorted(required - set(report['targets']))}")
    summaries = report["model_summary"]
    comparisons = report["paired_issue_cluster_bootstrap_95"]

    pairwise: dict[str, Any] = {}
    for target, base in (
        (args.treatment, args.control),
        (args.treatment, args.natural),
        (args.control, args.natural),
    ):
        name = f"{target}_minus_{base}"
        pairwise[name] = {
            "exact_train": judge.cluster_bootstrap(rows, target, base, EXACT),
            "combined_same_issue": judge.cluster_bootstrap(rows, target, base, SAME),
            "all": judge.cluster_bootstrap(rows, target, base, ALL),
            "same_issue_transitions": transitions(rows, target, base, SAME),
        }

    treatment_vs_control = pairwise[
        f"{args.treatment}_minus_{args.control}"
    ]["combined_same_issue"]
    treatment_vs_natural = pairwise[
        f"{args.treatment}_minus_{args.natural}"
    ]["combined_same_issue"]
    treatment_gate = comparisons[args.treatment]["stage_gates"]
    control_gate = comparisons[args.control]["stage_gates"]
    natural_gate = comparisons[args.natural]["stage_gates"]
    treatment_same = summaries[args.treatment]["combined_same_issue"]
    control_same = summaries[args.control]["combined_same_issue"]
    natural_same = summaries[args.natural]["combined_same_issue"]
    treatment_all = summaries[args.treatment]["all"]
    control_all = summaries[args.control]["all"]
    natural_all = summaries[args.natural]["all"]

    treatment_checks = {
        "same_issue_plus_2p5_vs_control": treatment_vs_control["mean_points"] >= 2.5,
        "same_issue_plus_2p5_vs_natural": treatment_vs_natural["mean_points"] >= 2.5,
        "same_issue_major_fact_not_above_control": treatment_same[
            "major_factual_errors"
        ]
        <= control_same["major_factual_errors"],
        "same_issue_major_fact_not_above_natural": treatment_same[
            "major_factual_errors"
        ]
        <= natural_same["major_factual_errors"],
        "generation_failures_not_above_control": treatment_all[
            "generation_failures"
        ]
        <= control_all["generation_failures"],
        "generation_failures_not_above_natural": treatment_all[
            "generation_failures"
        ]
        <= natural_all["generation_failures"],
        "exact_gate_vs_base": treatment_gate["exact_gate_pass"],
        "same_issue_gate_vs_base": treatment_gate["same_issue_gate_pass"],
    }
    treatment_selected = all(treatment_checks.values())
    control_checks = {
        "exact_gate_vs_base": control_gate["exact_gate_pass"],
        "same_issue_gate_vs_base": control_gate["same_issue_gate_pass"],
        "same_issue_not_below_natural": control_same["score_100"]
        >= natural_same["score_100"],
    }
    natural_checks = {
        "exact_gate_vs_base": natural_gate["exact_gate_pass"],
        "same_issue_gate_vs_base": natural_gate["same_issue_gate_pass"],
    }
    if treatment_selected:
        selected = args.treatment
        reason = "all_preregistered_treatment_checks_passed"
    elif all(control_checks.values()):
        selected = args.control
        reason = "treatment_not_selected_control_passed_fallback"
    elif all(natural_checks.values()):
        selected = args.natural
        reason = "additional_exposure_not_selected_natural_fallback_passed"
    else:
        selected = None
        reason = "no_method_passed_scale_trigger"
    result = {
        "schema_version": "ksova-cycle08-matched-aux-decision-1.0",
        "judge_dir": str(args.judge_dir.resolve()),
        "calibrated_items": report["calibrated_items"],
        "selected_method": selected,
        "decision_reason": reason,
        "scale133_triggered": selected is not None,
        "treatment_checks": treatment_checks,
        "control_fallback_checks": control_checks,
        "natural_fallback_checks": natural_checks,
        "model_summary": {
            name: summaries[name]
            for name in ("base", args.natural, args.control, args.treatment)
        },
        "pairwise": pairwise,
        "score_counts": dict(sorted(Counter(row["score"] for row in rows).items())),
    }
    atomic_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
