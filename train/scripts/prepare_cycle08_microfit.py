#!/usr/bin/env python3
"""Prepare the preregistered Cycle08 16-issue learning-ladder dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


FORMS = (
    "neutral_synthesis",
    "claim_review",
    "policy_comparison",
    "policy_recommendation",
)
DOMAIN_QUOTAS = {
    "D01": 1,
    "D02": 2,
    "D03": 2,
    "D04": 2,
    "D05": 2,
    "D06": 3,
    "D07": 3,
    "D08": 1,
}
SEED = "20260817::cycle08::microfit16"


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
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(issue_id: str) -> str:
    return hashlib.sha256(f"{SEED}::{issue_id}".encode()).hexdigest()


def normalized(text: str) -> str:
    return "".join(str(text).lower().split())


def training_row(row: dict[str, Any], coverage_priority: bool) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "experiment_id": "method_search_20260817_cycle08_learning_ladder",
            "split": "train",
            "example_kind": "natural_final_answer",
            "coverage_priority": coverage_priority,
            "source_training_id": metadata.get("training_id"),
            "training_id": "KSC8TRAIN-"
            + hashlib.sha256(
                f"{SEED}::{metadata.get('prompt_id')}".encode()
            ).hexdigest()[:20],
        }
    )
    return {**row, "metadata": metadata}


def development_row(row: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "experiment_id": "method_search_20260817_cycle08_learning_ladder",
            "split": "dev",
            "example_kind": "natural_final_answer",
            "coverage_priority": False,
            "source_training_id": metadata.get("training_id"),
            "training_id": "KSC8DEV-"
            + hashlib.sha256(
                f"{SEED}::{metadata.get('prompt_id')}".encode()
            ).hexdigest()[:20],
        }
    )
    return {**row, "metadata": metadata}


def evaluation_item(
    row: dict[str, Any], issue: dict[str, Any], evaluation_set: str
) -> dict[str, Any]:
    metadata = row.get("metadata") or {}
    messages = row.get("messages") or []
    if (
        len(messages) != 2
        or messages[0].get("role") != "user"
        or messages[1].get("role") != "assistant"
    ):
        raise ValueError("selected SFT row violates the two-message contract")
    source_prompt_id = str(metadata["prompt_id"])
    eval_id = "KSC8EVAL-" + hashlib.sha256(
        f"{SEED}::{evaluation_set}::{source_prompt_id}".encode()
    ).hexdigest()[:20]
    return {
        "schema_version": "ksova-cycle08-learning-ladder-item-1.0",
        "eval_id": eval_id,
        "evaluation_set": evaluation_set,
        "issue_id": str(metadata["issue_id"]),
        "domain_id": str(metadata["domain_id"]),
        "source_prompt_id": source_prompt_id,
        "prompt_form": str(metadata["prompt_form"]),
        "question_ko": str(messages[0]["content"]),
        "reference_answer_ko": str(messages[1]["content"]),
        "issue_card": issue["issue_card"],
    }


def request(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "cycle08__" + item["eval_id"],
        "messages": [{"role": "user", "content": item["question_ko"]}],
        "metadata": {
            "eval_id": item["eval_id"],
            "evaluation_set": item["evaluation_set"],
            "issue_id": item["issue_id"],
            "domain_id": item["domain_id"],
            "source_prompt_id": item["source_prompt_id"],
            "prompt_form": item["prompt_form"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-sft", type=Path, required=True)
    parser.add_argument("--same-issue-items", type=Path, required=True)
    parser.add_argument("--benchmark-items", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    selected_rows = read_jsonl(args.selected_sft)
    same_issue = {
        str(row["issue_id"]): row for row in read_jsonl(args.same_issue_items)
    }
    benchmark = read_jsonl(args.benchmark_items)
    benchmark_issue_ids = {str(row["issue_id"]) for row in benchmark}
    benchmark_questions = {normalized(row["question_ko"]) for row in benchmark}

    by_issue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        by_issue[str(row["metadata"]["issue_id"])].append(row)
    eligible: dict[str, list[dict[str, Any]]] = {}
    for issue_id, rows in by_issue.items():
        forms = [str(row["metadata"]["prompt_form"]) for row in rows]
        if len(rows) == 4 and set(forms) == set(FORMS):
            eligible[issue_id] = rows

    selected_ids: list[str] = []
    for domain_id, quota in DOMAIN_QUOTAS.items():
        candidates = sorted(
            (
                issue_id
                for issue_id, rows in eligible.items()
                if str(rows[0]["metadata"]["domain_id"]) == domain_id
                and issue_id in same_issue
            ),
            key=stable_rank,
        )
        if len(candidates) < quota:
            raise ValueError(f"{domain_id}: only {len(candidates)} eligible for quota {quota}")
        selected_ids.extend(candidates[:quota])
    if len(selected_ids) != 16 or len(set(selected_ids)) != 16:
        raise ValueError("selection did not produce 16 unique issues")

    # Assign each of the four question forms to exactly four held-out issues.
    selection_order = sorted(selected_ids, key=stable_rank)
    heldout_form = {
        issue_id: FORMS[index % len(FORMS)]
        for index, issue_id in enumerate(selection_order)
    }

    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    train_items: list[dict[str, Any]] = []
    heldout_items: list[dict[str, Any]] = []
    synthesis_items: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for issue_id in selection_order:
        rows = sorted(
            eligible[issue_id], key=lambda row: FORMS.index(row["metadata"]["prompt_form"])
        )
        issue = same_issue[issue_id]
        heldout = heldout_form[issue_id]
        train_for_issue = [row for row in rows if row["metadata"]["prompt_form"] != heldout]
        dev_for_issue = [row for row in rows if row["metadata"]["prompt_form"] == heldout]
        if len(train_for_issue) != 3 or len(dev_for_issue) != 1:
            raise ValueError(f"{issue_id}: invalid 3/1 split")
        for index, row in enumerate(train_for_issue):
            normalized_row = training_row(row, coverage_priority=index == 0)
            train_rows.append(normalized_row)
            train_items.append(evaluation_item(row, issue, "exact_train"))
        heldout_items.append(
            evaluation_item(dev_for_issue[0], issue, "same_issue_heldout_original")
        )
        dev_rows.append(development_row(dev_for_issue[0]))
        synthesis_items.append(
            {
                "schema_version": "ksova-cycle08-learning-ladder-item-1.0",
                "eval_id": "KSC8EVAL-"
                + hashlib.sha256(
                    f"{SEED}::same_issue_synthesis::{issue['prompt_id']}".encode()
                ).hexdigest()[:20],
                "evaluation_set": "same_issue_unseen_synthesis",
                "issue_id": issue_id,
                "domain_id": str(issue["domain_id"]),
                "source_prompt_id": str(issue["prompt_id"]),
                "prompt_form": "same_issue_unseen_synthesis",
                "question_ko": str(issue["question_ko"]),
                "reference_answer_ko": str(issue["reference_answer_ko"]),
                "issue_card": issue["issue_card"],
            }
        )
        selection_rows.append(
            {
                "issue_id": issue_id,
                "domain_id": str(issue["domain_id"]),
                "title_ko": str(issue["issue_card"]["title_ko"]),
                "heldout_prompt_form": heldout,
                "train_prompt_ids": [
                    str(row["metadata"]["prompt_id"]) for row in train_for_issue
                ],
                "heldout_prompt_id": str(dev_for_issue[0]["metadata"]["prompt_id"]),
                "synthesis_prompt_id": str(issue["prompt_id"]),
                "selection_rank": stable_rank(issue_id),
            }
        )

    eval_items = train_items + heldout_items + synthesis_items
    if len(train_rows) != 48 or len(dev_rows) != 16 or len(eval_items) != 80:
        raise ValueError("unexpected materialized row count")
    if set(selected_ids) & benchmark_issue_ids:
        raise ValueError("selected issues overlap the frozen benchmark")
    collisions = [
        item["eval_id"] for item in eval_items
        if normalized(item["question_ko"]) in benchmark_questions
    ]
    if collisions:
        raise ValueError(f"evaluation questions collide with benchmark: {collisions}")
    if len({row["metadata"]["training_id"] for row in train_rows}) != 48:
        raise ValueError("duplicate microfit training IDs")
    if len({item["eval_id"] for item in eval_items}) != 80:
        raise ValueError("duplicate evaluation IDs")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, tuple[Path, list[dict[str, Any]]]] = {
        "train": (args.out_dir / "natural_final.train.jsonl", train_rows),
        "dev": (args.out_dir / "natural_final.dev.jsonl", dev_rows),
        "exact_train_items": (args.out_dir / "exact_train.items.jsonl", train_items),
        "exact_train_requests": (args.out_dir / "exact_train.requests.jsonl", [request(x) for x in train_items]),
        "heldout_items": (args.out_dir / "same_issue_heldout.items.jsonl", heldout_items),
        "heldout_requests": (args.out_dir / "same_issue_heldout.requests.jsonl", [request(x) for x in heldout_items]),
        "synthesis_items": (args.out_dir / "same_issue_synthesis.items.jsonl", synthesis_items),
        "synthesis_requests": (args.out_dir / "same_issue_synthesis.requests.jsonl", [request(x) for x in synthesis_items]),
        "all_evaluation_items": (args.out_dir / "all_evaluation.items.jsonl", eval_items),
        "all_evaluation_requests": (args.out_dir / "all_evaluation.requests.jsonl", [request(x) for x in eval_items]),
        "selection": (args.out_dir / "issue_selection.jsonl", selection_rows),
    }
    for path, rows in outputs.values():
        write_jsonl(path, rows)

    report = {
        "schema_version": "ksova-cycle08-microfit-materialization-1.0",
        "pass": True,
        "seed": SEED,
        "domain_quotas": DOMAIN_QUOTAS,
        "issues": 16,
        "train_rows": 48,
        "train_rows_per_issue": 3,
        "dev_rows": 16,
        "heldout_original_rows": 16,
        "heldout_synthesis_rows": 16,
        "heldout_form_counts": dict(sorted(Counter(heldout_form.values()).items())),
        "benchmark_issue_overlap": 0,
        "benchmark_normalized_question_overlap": 0,
        "inputs": {
            "selected_sft": {"path": str(args.selected_sft.resolve()), "sha256": sha256(args.selected_sft)},
            "same_issue_items": {"path": str(args.same_issue_items.resolve()), "sha256": sha256(args.same_issue_items)},
            "benchmark_items": {"path": str(args.benchmark_items.resolve()), "sha256": sha256(args.benchmark_items)},
        },
        "outputs": {
            name: {"path": str(path.resolve()), "rows": len(rows), "sha256": sha256(path)}
            for name, (path, rows) in outputs.items()
        },
    }
    write_json(args.out_dir / "materialization_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
