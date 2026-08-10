#!/usr/bin/env python3
"""Build one-pass item reauthor requests from a direct-review TSV."""

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


def request_key(row: dict[str, Any]) -> tuple[str, str]:
    metadata = row.get("metadata") or {}
    return str(metadata.get("issue_id")), str(metadata.get("prompt_form"))


def item_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("issue_id")), str(row.get("prompt_form"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    requests = {request_key(row): row for row in read_jsonl(args.requests)}
    items = {str(row["item_id"]): row for row in read_jsonl(args.items)}

    selected: list[dict[str, str]] = []
    with args.audit.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["route"] == "reauthor_once":
                selected.append(row)

    output: list[dict[str, Any]] = []
    for audit_row in selected:
        item_id = audit_row["item_id"]
        item = items.get(item_id)
        if item is None:
            raise SystemExit(f"missing item: {item_id}")
        request = requests.get(item_key(item))
        if request is None:
            raise SystemExit(f"missing request for item: {item_id}")

        instruction = f"""

[One-pass item reauthor instruction]
The original item failed direct human review for exactly this reason:
{audit_row['reason_ko']}

Fix only that defect without broadening the atomic issue or inventing evidence.
The question must identify the Korean decision object, event, institution, right,
or public dispute, but it must not assign the respondent a Korean persona, office,
or desired stance. It must require evaluation rather than recall. Do not disclose
the preferred conclusion or nearly all decisive reasoning in the question.
Preserve issue_id, item_role, prompt_form, framing_pair_id, decision_claim_ko,
and source_ids exactly. Return one complete replacement JSON object with a new
item_id ending in -R1. Do not explain the edit.

[Original rejected item]
{json.dumps(item, ensure_ascii=False, sort_keys=True)}
"""
        replacement = dict(request)
        replacement["id"] = f"reauthor__{item_id}"
        replacement["metadata"] = {
            **dict(request["metadata"]),
            "purpose": "ksova_item_reauthor_once",
            "original_item_id": item_id,
        }
        replacement["prompt"] = str(request["prompt"]) + instruction
        output.append(replacement)

    write_jsonl(args.output, output)
    print(json.dumps({"selected": len(selected), "written": len(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
