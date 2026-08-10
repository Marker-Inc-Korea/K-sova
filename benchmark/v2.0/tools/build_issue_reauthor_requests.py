#!/usr/bin/env python3
"""Build one-pass issue reauthor requests from a direct-review TSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--issues", type=Path, required=True)
    parser.add_argument("--rejected-issues", type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    requests = {
        str(row.get("metadata", {}).get("source_id")): row
        for row in read_jsonl(args.requests)
    }
    issues = {
        (str(row["source_ids"][0]), str(row["issue_id"])): row
        for row in read_jsonl(args.issues)
    }
    if args.rejected_issues:
        source_axes = {}
        if args.sources:
            source_axes = {
                str(row["source_id"]): str(row["sovereign_axes"][0])
                for row in read_jsonl(args.sources)
            }
        for rejected in read_jsonl(args.rejected_issues):
            issue = dict(rejected.get("issue") or {})
            source_id = str(rejected.get("request_id", "")).removeprefix("issue__")
            if not issue or not source_id:
                continue
            issue["source_ids"] = [source_id]
            if source_id in source_axes:
                issue["sovereign_axis"] = source_axes[source_id]
            issues[(source_id, str(issue.get("issue_id", "")))] = issue

    selected: list[dict[str, str]] = []
    with args.audit.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["route"] == "reauthor_once":
                selected.append(row)

    output: list[dict[str, Any]] = []
    for audit_row in selected:
        source_id = audit_row["source_id"]
        issue_id = audit_row["issue_id"]
        request = requests.get(source_id)
        issue = issues.get((source_id, issue_id))
        if request is None or issue is None:
            raise SystemExit(f"missing request or issue: {source_id} / {issue_id}")

        instruction = f"""

[One-pass issue reauthor instruction]
The original issue failed direct human review for exactly this reason:
{audit_row['reason_ko']}

Fix only that defect. Do not broaden the source, add an unsupported Republic of
Korea legal duty, invent a foreign position, or turn the topic into a generic
opinion question. Keep the source_id and sovereign_axis unchanged. Return zero
issues if the source cannot support a valid K-SOVA issue after this correction.
If returning an issue, use a new issue_id ending in -R1 and return one complete
root object under the original JSON contract. Do not explain the edit.

[Original rejected issue]
{json.dumps(issue, ensure_ascii=False, sort_keys=True)}
"""
        replacement = dict(request)
        replacement["id"] = f"issue_reauthor__{source_id}"
        replacement["metadata"] = {
            "purpose": "ksova_issue_reauthor_once",
            "source_id": source_id,
            "original_issue_id": issue_id,
        }
        replacement["prompt"] = str(request["prompt"]) + instruction
        output.append(replacement)

    write_jsonl(args.output, output)
    print(json.dumps({"selected": len(selected), "written": len(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
