#!/usr/bin/env python3
"""Build normalized embedding inputs for benchmark-reference and training materials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issues", type=Path, required=True)
    parser.add_argument("--materials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    issues = read_jsonl(args.issues)
    materials = read_jsonl(args.materials)
    rows = []
    for issue in issues:
        entity_id = str(issue.get("issue_id") or "")
        text = "\n".join(str(issue.get(key) or "") for key in (
            "title_ko", "decision_core_ko", "korean_reference_direction_ko",
            "sovereign_stake_ko",
        )).strip()
        rows.append({
            "record_id": f"benchmark__{entity_id}",
            "role": "benchmark_reference",
            "entity_id": entity_id,
            "text": text,
        })
    for material in materials:
        entity_id = str(material.get("material_id") or "")
        text = "\n".join(str(material.get(key) or "") for key in (
            "specific_issue_ko", "decision_object_ko", "minimum_baseline_ko",
            "korean_stake_ko",
        )).strip()
        rows.append({
            "record_id": f"material__{entity_id}",
            "role": "training_material_candidate",
            "entity_id": entity_id,
            "text": text,
        })

    record_ids = [row["record_id"] for row in rows]
    if any(not row["entity_id"] or len(row["text"]) < 30 for row in rows):
        raise ValueError("empty identifier or underspecified embedding text")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("duplicate embedding record_id")

    write_jsonl(args.output, rows)
    report = {
        "schema_version": "ksova-holdout-embedding-input-report-1.0",
        "reference_issues": len(issues),
        "candidate_materials": len(materials),
        "records": len(rows),
        "output": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
