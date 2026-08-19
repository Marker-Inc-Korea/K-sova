#!/usr/bin/env python3
"""Aggregate the preregistered three-vote Cycle08 exact-item judge panel."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
from typing import Any

import gpt54mini_cycle08_learning_judge as judge


TARGETS = (
    "base",
    "natural_lr2e5_u72_merged",
    "final_rehearsal_control_lr2e5",
    "short_aux_treatment_lr2e5",
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def accepted_judgments(out: Path) -> list[dict[str, Any]]:
    report = json.loads((out / "build_report.json").read_text(encoding="utf-8"))
    blind = judge.load_blind(out)
    accepted: list[dict[str, Any]] = []
    for entry in report["items"]:
        attempts = [("retry1", ".retry1"), ("primary", "")]
        if not (out / "items" / entry["eval_id"] / "raw_output.retry1.json").is_file():
            attempts = [("primary", "")]
        errors: list[dict[str, Any]] = []
        for label, suffix in attempts:
            rows, error = judge.validate_item(out, entry, blind, suffix)
            anchor_failures = judge.anchor_failures(rows) if not error else []
            if not error and not anchor_failures:
                accepted.extend(
                    row for row in rows if row["target_kind"] == "model_output"
                )
                break
            errors.append(
                {
                    "attempt": label,
                    "structural_error": error,
                    "anchor_failures": anchor_failures,
                }
            )
        else:
            raise ValueError(f"no calibrated judgment for {entry['eval_id']}: {errors}")
    if len(accepted) != report["candidate_items"] * len(TARGETS):
        raise ValueError("panel judgment coverage mismatch")
    return accepted


def bootstrap(details: list[dict[str, Any]], target: str) -> dict[str, Any]:
    selected = [row for row in details if row["target"] == target]
    by_issue: dict[str, list[int]] = defaultdict(list)
    for row in selected:
        by_issue[row["issue_id"]].append(int(row["difference_from_base"]))
    issue_means = {
        issue_id: sum(values) / len(values) for issue_id, values in by_issue.items()
    }
    issues = sorted(issue_means)
    rng = random.Random(f"20260818::cycle08-stage3b::{target}")
    samples = sorted(
        sum(issue_means[rng.choice(issues)] for _ in issues) / len(issues) * 50
        for _ in range(10000)
    )
    return {
        "n_items": len(selected),
        "n_issue_clusters": len(issues),
        "mean_points": round(
            sum(row["difference_from_base"] for row in selected)
            / len(selected)
            * 50,
            6,
        ),
        "ci95_low": round(samples[249], 6),
        "ci95_high": round(samples[9749], 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-judgments", type=Path, required=True)
    parser.add_argument("--judge-dir", type=Path, action="append", required=True)
    parser.add_argument("--stage3-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.judge_dir) != 2:
        raise ValueError("exactly two new judge directories are required")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    primary = [
        row
        for row in read_jsonl(args.primary_judgments)
        if row["evaluation_set"] == "exact_train" and row["target"] in TARGETS
    ]
    replicates = [primary] + [accepted_judgments(path) for path in args.judge_dir]
    mappings = [
        {(str(row["eval_id"]), str(row["target"])): row for row in rows}
        for rows in replicates
    ]
    keys = set(mappings[0])
    if len(keys) != 48 * len(TARGETS) or any(set(mapping) != keys for mapping in mappings):
        raise ValueError("replicate coverage differs")

    panel_rows: list[dict[str, Any]] = []
    for eval_id, target in sorted(keys):
        votes = [mapping[(eval_id, target)] for mapping in mappings]
        scores = [int(row["score"]) for row in votes]
        major_votes = [row["factual_integrity"] == "major_error" for row in votes]
        panel_rows.append(
            {
                "eval_id": eval_id,
                "issue_id": str(votes[0]["issue_id"]),
                "target": target,
                "scores": scores,
                "median_score": sorted(scores)[1],
                "major_factual_votes": sum(major_votes),
                "panel_major_factual_error": sum(major_votes) >= 2,
                "generation_complete": bool(votes[0].get("generation_complete", True)),
                "reasons_ko": [row["reason_ko"] for row in votes],
            }
        )
    by_key = {(row["eval_id"], row["target"]): row for row in panel_rows}
    for row in panel_rows:
        base = by_key[(row["eval_id"], "base")]
        row["difference_from_base"] = row["median_score"] - base["median_score"]

    summaries: dict[str, Any] = {}
    for target in TARGETS:
        rows = [row for row in panel_rows if row["target"] == target]
        counts = Counter(int(row["median_score"]) for row in rows)
        pairwise_agreement = []
        for first, second in ((0, 1), (0, 2), (1, 2)):
            pairwise_agreement.append(
                sum(row["scores"][first] == row["scores"][second] for row in rows)
                / len(rows)
            )
        summaries[target] = {
            "n": len(rows),
            "score_100": round(sum(row["median_score"] for row in rows) / len(rows) * 50, 6),
            "score_counts": {str(key): counts[key] for key in range(3)},
            "major_factual_errors": sum(row["panel_major_factual_error"] for row in rows),
            "generation_failures": sum(not row["generation_complete"] for row in rows),
            "all_three_score_agreement": sum(len(set(row["scores"])) == 1 for row in rows) / len(rows),
            "pairwise_score_agreement": pairwise_agreement,
            "replicate_scores_100": [
                round(sum(row["scores"][index] for row in rows) / len(rows) * 50, 6)
                for index in range(3)
            ],
        }

    stage3 = json.loads(args.stage3_report.read_text(encoding="utf-8"))
    stage3_summary = stage3["model_summary"]
    comparisons = {
        target: bootstrap(panel_rows, target)
        for target in TARGETS
        if target != "base"
    }
    base = summaries["base"]

    def common_checks(target: str) -> dict[str, bool]:
        return {
            "panel_exact_score_above_base": summaries[target]["score_100"] > base["score_100"],
            "panel_exact_major_fact_not_above_base": summaries[target]["major_factual_errors"] <= base["major_factual_errors"],
            "stage3_same_issue_score_above_base": stage3_summary[target]["combined_same_issue"]["score_100"] > stage3_summary["base"]["combined_same_issue"]["score_100"],
            "stage3_same_issue_major_fact_not_above_base": stage3_summary[target]["combined_same_issue"]["major_factual_errors"] <= stage3_summary["base"]["combined_same_issue"]["major_factual_errors"],
        }

    control = "final_rehearsal_control_lr2e5"
    natural = "natural_lr2e5_u72_merged"
    control_checks = common_checks(control)
    control_checks["stage3_same_issue_not_below_natural"] = (
        stage3_summary[control]["combined_same_issue"]["score_100"]
        >= stage3_summary[natural]["combined_same_issue"]["score_100"]
    )
    natural_checks = common_checks(natural)
    if all(control_checks.values()):
        selected = control
        reason = "control_passed_preregistered_exploratory_gate"
    elif all(natural_checks.values()):
        selected = natural
        reason = "natural_passed_preregistered_exploratory_fallback"
    else:
        selected = None
        reason = "no_arm_passed_exploratory_scale_gate"

    result = {
        "schema_version": "ksova-cycle08-stage3b-exact-panel-1.0",
        "status": "complete",
        "role": "exploratory follow-up; does not retroactively pass the original Stage4 gate",
        "primary_judgments": str(args.primary_judgments.resolve()),
        "new_judge_dirs": [str(path.resolve()) for path in args.judge_dir],
        "panel_summary": summaries,
        "paired_issue_cluster_bootstrap_95": comparisons,
        "control_checks": control_checks,
        "natural_fallback_checks": natural_checks,
        "selected_method": selected,
        "decision_reason": reason,
        "exploratory_scale133_triggered": selected is not None,
        "panel_rows": panel_rows,
    }
    atomic_json(args.out, result)
    print(
        json.dumps(
            {
                "panel_summary": summaries,
                "comparisons": comparisons,
                "selected_method": selected,
                "decision_reason": reason,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
