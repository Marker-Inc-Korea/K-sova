#!/usr/bin/env python3
"""Assemble one best available answer per 150x3 issue-variant pair."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from parse_full_answer_teacher_responses import VARIANTS, normalized_text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def pair(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("issue_id") or ""), str(row.get("variant") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validated-candidate", type=Path, action="append", default=[],
        help="Independent-validator pass JSONL; later files have higher priority.",
    )
    parser.add_argument(
        "--provisional-candidate", type=Path, action="append", default=[],
        help="Structurally valid rows used only for still-missing pairs.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite assembly outputs")
    if not args.validated_candidate:
        raise ValueError("at least one validated candidate input is required")

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    selected_from: dict[tuple[str, str], str] = {}
    validated_pair_count = 0
    validated_overrides = 0
    for path in args.validated_candidate:
        rows = read_jsonl(path)
        if len({str(row.get("example_id") or "") for row in rows}) != len(rows):
            raise ValueError(f"{path}: duplicate or missing example IDs")
        for row in rows:
            if (row.get("validation") or {}).get("verdict") != "pass":
                raise ValueError(f"{path}: validated candidate contains a non-pass row")
            key = pair(row)
            if key[1] not in VARIANTS or not key[0]:
                raise ValueError(f"{path}: invalid issue-variant pair {key}")
            validated_pair_count += int(key not in selected)
            validated_overrides += int(key in selected)
            selected[key] = row
            selected_from[key] = str(path)

    provisional_added = 0
    provisional_skipped = 0
    for path in args.provisional_candidate:
        rows = read_jsonl(path)
        for row in rows:
            key = pair(row)
            if key[1] not in VARIANTS or not key[0]:
                raise ValueError(f"{path}: invalid provisional pair {key}")
            if key in selected:
                provisional_skipped += 1
                continue
            selected[key] = row
            selected_from[key] = str(path)
            provisional_added += 1

    issue_ids = sorted({issue_id for issue_id, _variant in selected})
    expected = {(issue_id, variant) for issue_id in issue_ids for variant in VARIANTS}
    missing = sorted(expected - set(selected))
    if len(issue_ids) != 150:
        missing.append((f"__issue_count_{len(issue_ids)}__", ""))
    if missing and not args.allow_incomplete:
        raise ValueError(f"assembly is incomplete: {missing[:20]}")

    rows = [
        selected[(issue_id, variant)]
        for issue_id in issue_ids
        for variant in VARIANTS
        if (issue_id, variant) in selected
    ]
    example_ids = [str(row.get("example_id") or "") for row in rows]
    questions = [normalized_text(str((row.get("messages") or [{}, {}])[0].get("content") or "")) for row in rows]
    answers = [normalized_text(str((row.get("messages") or [{}, {}])[1].get("content") or "")) for row in rows]
    if len(example_ids) != len(set(example_ids)) or "" in example_ids:
        raise ValueError("assembled examples have duplicate or missing IDs")
    if len(questions) != len(set(questions)) or "" in questions:
        raise ValueError("assembled examples have duplicate or empty questions")
    if len(answers) != len(set(answers)) or "" in answers:
        raise ValueError("assembled examples have duplicate or empty answers")
    by_source = Counter(selected_from[pair(row)] for row in rows)
    report = {
        "schema_version": "ksova-validated-full-answer-assembly-1.0",
        "issues": len(issue_ids),
        "examples": len(rows),
        "validated_pairs": validated_pair_count,
        "validated_overrides": validated_overrides,
        "provisional_added": provisional_added,
        "provisional_skipped": provisional_skipped,
        "missing_pairs": [f"{issue_id}::{variant}" for issue_id, variant in missing],
        "selected_by_input": dict(sorted(by_source.items())),
        "hashes": {
            "validated_candidates": {
                str(path): file_sha256(path) for path in args.validated_candidate
            },
            "provisional_candidates": {
                str(path): file_sha256(path) for path in args.provisional_candidate
            },
        },
        "output": str(args.output),
    }
    write_jsonl(args.output, rows)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
