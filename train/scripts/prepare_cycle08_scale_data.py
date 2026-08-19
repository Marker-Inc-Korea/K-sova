#!/usr/bin/env python3
"""Materialize the Cycle08 scale arm without mutating Cycle07 artifacts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("short_aux", "natural"), default="short_aux"
    )
    args = parser.parse_args()

    source = args.source.resolve()
    rows = read_jsonl(source)
    expected_rows = 556 if args.mode == "short_aux" else 423
    if len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, found {len(rows)}")
    materialized: list[dict[str, Any]] = []
    original_messages: list[str] = []
    materialized_messages: list[str] = []
    natural_priority_issues: set[str] = set()
    for original in rows:
        row = json.loads(json.dumps(original, ensure_ascii=False))
        metadata = row.setdefault("metadata", {})
        issue_id = str(metadata.get("issue_id") or "")
        is_aux = metadata.get("example_kind") == "fixed_field_auxiliary"
        if args.mode == "short_aux":
            is_priority = is_aux
            role = "coverage_short_aux" if is_aux else "application_final"
        else:
            is_priority = issue_id not in natural_priority_issues
            if is_priority:
                natural_priority_issues.add(issue_id)
            role = "coverage_natural_final" if is_priority else "application_final"
        metadata["coverage_priority"] = is_priority
        metadata["cycle08_scale_role"] = role
        materialized.append(row)
        original_messages.append(
            json.dumps(original["messages"], ensure_ascii=False, sort_keys=True)
        )
        materialized_messages.append(
            json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)
        )

    issues = {
        str(row["metadata"].get("issue_id") or "") for row in materialized
    }
    priority = [
        row for row in materialized if row["metadata"]["coverage_priority"]
    ]
    priority_counts = Counter(
        str(row["metadata"].get("issue_id") or "") for row in priority
    )
    kind_counts = Counter(
        str(row["metadata"].get("example_kind") or "") for row in materialized
    )
    invariants = {
        "expected_rows": len(materialized) == expected_rows,
        "issues_133": len(issues) == 133 and "" not in issues,
        "one_priority_per_issue": (
            len(priority) == 133
            and set(priority_counts) == issues
            and set(priority_counts.values()) == {1}
        ),
        "messages_unchanged": original_messages == materialized_messages,
    }
    if args.mode == "short_aux":
        invariants.update(
            {
                "natural_rows_423": kind_counts["natural_final_answer"] == 423,
                "aux_rows_133": kind_counts["fixed_field_auxiliary"] == 133,
                "all_priorities_are_aux": all(
                    row["metadata"].get("example_kind")
                    == "fixed_field_auxiliary"
                    for row in priority
                ),
            }
        )
    if not all(invariants.values()):
        raise ValueError(f"scale data invariant failure: {invariants}")
    write_jsonl(args.out, materialized)
    report = {
        "schema_version": "ksova-cycle08-scale-data-1.0",
        "mode": args.mode,
        "source": {"path": str(source), "sha256": sha256(source)},
        "output": {
            "path": str(args.out.resolve()),
            "rows": len(materialized),
            "issues": len(issues),
            "sha256": sha256(args.out),
        },
        "kind_counts": dict(sorted(kind_counts.items())),
        "invariants": invariants,
    }
    atomic_text(
        args.report,
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
