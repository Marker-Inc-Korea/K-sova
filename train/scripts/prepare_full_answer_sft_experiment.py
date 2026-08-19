#!/usr/bin/env python3
"""Seal 150x3 validated full answers into a one-pass SFT experiment."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable


VARIANTS = ("neutral", "claim_review", "policy_or_comparison")
EXPECTED_BENCHMARK_SHA256 = (
    "0ca49e7bc94178300c0309919d63a51c0496d84d6e352501d350f0deac494032"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", value.casefold())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--benchmark-items", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-issues", type=int, default=150)
    parser.add_argument("--expected-examples", type=int, default=450)
    parser.add_argument("--fit-monitor-issues-per-domain", type=int, default=4)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output_dir}")
    rows = read_jsonl(args.examples)
    benchmark_rows = read_jsonl(args.benchmark_items)
    benchmark_hash = file_sha256(args.benchmark_items)
    if benchmark_hash != EXPECTED_BENCHMARK_SHA256 or len(benchmark_rows) != 480:
        raise ValueError("unexpected frozen benchmark identity")
    if len(rows) != args.expected_examples:
        raise ValueError(
            f"expected {args.expected_examples} validated examples, got {len(rows)}"
        )

    example_ids = [str(row.get("example_id") or "") for row in rows]
    if "" in example_ids or len(set(example_ids)) != len(example_ids):
        raise ValueError("missing or duplicate example IDs")
    benchmark_issue_ids = {str(row.get("issue_id") or "") for row in benchmark_rows}
    benchmark_questions = {
        normalized_text(str(row.get("question_ko") or ""))
        for row in benchmark_rows
    } - {""}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_questions: set[str] = set()
    seen_answers: set[str] = set()
    by_domain: Counter[str] = Counter()
    by_variant: Counter[str] = Counter()
    answer_lengths: list[int] = []
    for row in rows:
        issue_id = str(row.get("issue_id") or "")
        domain_id = str(row.get("domain_id") or "")
        variant = str(row.get("variant") or "")
        messages = row.get("messages") or []
        validation = row.get("validation") or {}
        metadata = row.get("metadata") or {}
        if issue_id in benchmark_issue_ids:
            raise ValueError(f"training issue overlaps benchmark: {issue_id}")
        if variant not in VARIANTS:
            raise ValueError(f"invalid variant: {variant}")
        if len(messages) != 2 or [message.get("role") for message in messages] != [
            "user", "assistant",
        ]:
            raise ValueError(f"{row.get('example_id')}: invalid SFT messages")
        if validation.get("verdict") != "pass" or validation.get("failed_checks"):
            raise ValueError(f"{row.get('example_id')}: example is not validator-passed")
        if metadata.get("training_id") != row.get("example_id"):
            raise ValueError(f"{row.get('example_id')}: training ID mismatch")
        if metadata.get("issue_id") != issue_id:
            raise ValueError(f"{row.get('example_id')}: metadata issue mismatch")
        if bool(metadata.get("coverage_priority")) != (variant == "neutral"):
            raise ValueError(f"{row.get('example_id')}: coverage priority mismatch")
        question = normalized_text(str(messages[0].get("content") or ""))
        answer = normalized_text(str(messages[1].get("content") or ""))
        if not question or not answer:
            raise ValueError(f"{row.get('example_id')}: empty normalized message")
        if question in benchmark_questions:
            raise ValueError(f"{row.get('example_id')}: benchmark question overlap")
        if question in seen_questions or answer in seen_answers:
            raise ValueError(f"{row.get('example_id')}: duplicate message pair component")
        seen_questions.add(question)
        seen_answers.add(answer)
        grouped[issue_id].append(row)
        by_domain[domain_id] += 1
        by_variant[variant] += 1
        answer_lengths.append(len(str(messages[1]["content"])))

    if len(grouped) != args.expected_issues:
        raise ValueError(f"expected {args.expected_issues} issues, got {len(grouped)}")
    for issue_id, issue_rows in grouped.items():
        variants = {str(row.get("variant") or "") for row in issue_rows}
        if len(issue_rows) != 3 or variants != set(VARIANTS):
            raise ValueError(f"{issue_id}: expected exactly the three variants")

    ordered = sorted(
        rows,
        key=lambda row: (
            str(row["issue_id"]), VARIANTS.index(str(row["variant"])),
        ),
    )
    issues_by_domain: dict[str, list[str]] = defaultdict(list)
    for issue_id in sorted(grouped):
        domain_id = str(grouped[issue_id][0]["domain_id"])
        issues_by_domain[domain_id].append(issue_id)
    monitor_issue_ids = {
        issue_id
        for domain_id in sorted(issues_by_domain)
        for issue_id in issues_by_domain[domain_id][
            : args.fit_monitor_issues_per_domain
        ]
    }
    fit_monitor = [
        row for row in ordered
        if row["issue_id"] in monitor_issue_ids and row["variant"] == "neutral"
    ]
    train_path = args.output_dir / "sft_train_450.jsonl"
    monitor_path = args.output_dir / "fit_monitor.jsonl"
    report_path = args.output_dir / "experiment_manifest.json"
    write_jsonl(train_path, ordered)
    write_jsonl(monitor_path, fit_monitor)
    report = {
        "schema_version": "ksova-full-answer-sft-experiment-1.0",
        "experiment_id": "training150_full_answer_sft_v1_20260819",
        "examples": len(ordered),
        "issues": len(grouped),
        "by_domain": dict(sorted(by_domain.items())),
        "by_variant": dict(sorted(by_variant.items())),
        "exact_duplicate_questions": 0,
        "exact_duplicate_answers": 0,
        "benchmark_issue_collisions": 0,
        "benchmark_question_collisions": 0,
        "answer_chars": {
            "min": min(answer_lengths),
            "max": max(answer_lengths),
            "average": round(sum(answer_lengths) / len(answer_lengths), 1),
        },
        "training_recipe": {
            "base_model": "Qwen3.5-9B",
            "mode": "SFT LoRA",
            "epochs": 1.0,
            "sampling": "coverage_first_without_replacement",
            "learning_rate": 2e-5,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "adapter_scale_for_inference": 0.5,
            "checkpoint_selection": "none; one preregistered terminal checkpoint",
        },
        "fit_monitor": {
            "rows": len(fit_monitor),
            "issues": len(monitor_issue_ids),
            "is_subset_of_training_data": True,
            "used_for_model_selection": False,
            "purpose": "post-training fit sanity check only",
        },
        "frozen_benchmark": {
            "items": len(benchmark_rows),
            "sha256": benchmark_hash,
            "role": "development and regression evaluation only",
        },
        "artifacts": {
            "input_examples": str(args.examples),
            "input_examples_sha256": file_sha256(args.examples),
            "train": str(train_path),
            "train_sha256": file_sha256(train_path),
            "fit_monitor": str(monitor_path),
            "fit_monitor_sha256": file_sha256(monitor_path),
        },
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
