#!/usr/bin/env python3
"""Build a matched 16-issue final-rehearsal versus short-aux ablation.

Both arms contain the same 48 natural final-answer rows and one additional row
per issue.  The control repeats one natural answer; the treatment substitutes
one explicitly tagged fixed-field auxiliary task.  This holds issues, natural
targets, row count, optimizer-update count, and the dev set constant while
changing only the fourth exposure's task family.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from prepare_short_aux_mixture import TASKS, auxiliary_messages, eligible_tasks


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(*parts: str) -> str:
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return "KSC8AUX-" + digest[:20]


def choose_balanced_task(
    card: dict[str, Any], counts: Counter[str], issue_id: str
) -> str:
    eligible = eligible_tasks(card)
    if not eligible:
        raise ValueError(f"{issue_id}: no eligible auxiliary task")
    return min(
        eligible,
        key=lambda task: (
            counts[task],
            hashlib.sha256(f"20260818::{issue_id}::{task}".encode()).hexdigest(),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural-train", type=Path, required=True)
    parser.add_argument("--natural-dev", type=Path, required=True)
    parser.add_argument("--evaluation-items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {args.out}")

    natural = read_jsonl(args.natural_train)
    dev = read_jsonl(args.natural_dev)
    items = read_jsonl(args.evaluation_items)
    by_issue: dict[str, list[dict[str, Any]]] = {}
    for row in natural:
        issue_id = str((row.get("metadata") or {}).get("issue_id") or "")
        by_issue.setdefault(issue_id, []).append(row)
    if len(natural) != 48 or len(by_issue) != 16:
        raise ValueError("expected 48 natural rows over 16 issues")
    if any(len(rows) != 3 for rows in by_issue.values()):
        raise ValueError("each issue must have exactly three natural rows")

    cards: dict[str, dict[str, Any]] = {}
    for row in items:
        issue_id = str(row["issue_id"])
        card = row["issue_card"]
        if issue_id in cards and cards[issue_id] != card:
            raise ValueError(f"{issue_id}: inconsistent issue cards")
        cards[issue_id] = card
    if set(cards) != set(by_issue):
        raise ValueError("issue-card coverage differs from natural train coverage")

    control_extra: list[dict[str, Any]] = []
    aux_extra: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()
    ordered_issues = sorted(
        by_issue,
        key=lambda issue_id: hashlib.sha256(
            f"20260818::matched-aux::{issue_id}".encode()
        ).hexdigest(),
    )
    for issue_id in ordered_issues:
        rows = by_issue[issue_id]
        priority = [
            row for row in rows if (row.get("metadata") or {}).get("coverage_priority")
        ]
        if len(priority) != 1:
            raise ValueError(f"{issue_id}: expected one coverage-priority row")
        repeated = copy.deepcopy(priority[0])
        repeated_metadata = dict(repeated.get("metadata") or {})
        repeated_metadata.update(
            {
                "training_id": stable_id("control", issue_id),
                "coverage_priority": False,
                "example_kind": "natural_final_rehearsal_control",
                "task_family": "final_answer",
                "natural_final_answer": True,
                "matched_ablation_role": "fourth_exposure_control",
            }
        )
        repeated["metadata"] = repeated_metadata
        control_extra.append(repeated)

        card = cards[issue_id]
        task = choose_balanced_task(card, task_counts, issue_id)
        task_counts[task] += 1
        messages, answer = auxiliary_messages(task, card)
        aux_extra.append(
            {
                "schema_version": "ksova-cycle08-matched-short-aux-sft-1.0",
                "messages": messages,
                "metadata": {
                    "training_id": stable_id("aux", issue_id, task),
                    "split": "train",
                    "issue_id": issue_id,
                    "domain_id": str(rows[0]["metadata"]["domain_id"]),
                    "coverage_priority": False,
                    "example_kind": "fixed_field_auxiliary",
                    "task_family": task,
                    "output_format": "explicit_task_tag_json",
                    "natural_final_answer": False,
                    "matched_ablation_role": "fourth_exposure_treatment",
                    "answer_chars": len(messages[-1]["content"]),
                    "answer_field_count": len(answer),
                },
            }
        )

    control = natural + control_extra
    treatment = natural + aux_extra
    control_path = args.out / "final_rehearsal_control.train.jsonl"
    treatment_path = args.out / "short_aux_treatment.train.jsonl"
    dev_path = args.out / "natural_final.dev.jsonl"
    write_jsonl(control_path, control)
    write_jsonl(treatment_path, treatment)
    write_jsonl(dev_path, dev)

    def arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "rows": len(rows),
            "issues": len({row["metadata"]["issue_id"] for row in rows}),
            "issue_exposure_counts": dict(
                sorted(Counter(row["metadata"]["issue_id"] for row in rows).items())
            ),
            "example_kind_counts": dict(
                sorted(Counter(row["metadata"]["example_kind"] for row in rows).items())
            ),
            "natural_final_answer_rows": sum(
                bool(row["metadata"].get("natural_final_answer")) for row in rows
            ),
            "assistant_chars": {
                "sum": sum(len(row["messages"][-1]["content"]) for row in rows),
                "mean": round(
                    sum(len(row["messages"][-1]["content"]) for row in rows) / len(rows),
                    3,
                ),
            },
        }

    report = {
        "schema_version": "ksova-cycle08-matched-aux-build-1.0",
        "design": {
            "constant": [
                "16 issues",
                "48 identical natural final-answer rows",
                "one fourth exposure per issue",
                "64 rows and therefore matched optimizer-update count",
                "identical natural-only dev set",
            ],
            "changed_variable": (
                "fourth exposure is a repeated natural final answer in control and "
                "an explicitly tagged short fixed-field task in treatment"
            ),
            "natural_share_control": "64/64",
            "natural_share_treatment": "48/64",
        },
        "task_counts": dict(sorted(task_counts.items())),
        "control": arm_summary(control),
        "treatment": arm_summary(treatment),
        "invariants": {
            "same_rows": len(control) == len(treatment) == 64,
            "same_issues": {
                row["metadata"]["issue_id"] for row in control
            }
            == {row["metadata"]["issue_id"] for row in treatment},
            "same_first_48_rows": control[:48] == treatment[:48] == natural,
            "one_extra_per_issue_control": all(
                Counter(row["metadata"]["issue_id"] for row in control_extra)[issue] == 1
                for issue in by_issue
            ),
            "one_extra_per_issue_treatment": all(
                Counter(row["metadata"]["issue_id"] for row in aux_extra)[issue] == 1
                for issue in by_issue
            ),
            "unique_training_ids_per_arm": len(
                {row["metadata"]["training_id"] for row in control}
            )
            == len(control)
            and len({row["metadata"]["training_id"] for row in treatment})
            == len(treatment),
        },
        "files": {},
    }
    for label, path in (
        ("control_train", control_path),
        ("treatment_train", treatment_path),
        ("dev", dev_path),
    ):
        report["files"][label] = {
            "path": str(path.resolve()),
            "rows": sum(1 for _ in path.open(encoding="utf-8")),
            "sha256": sha256(path),
        }
    if not all(report["invariants"].values()):
        raise RuntimeError(f"matched-ablation invariant failure: {report['invariants']}")
    write_json(args.out / "materialization_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
