#!/usr/bin/env python3
"""Keep conflict objects whose complete source set is in the training source pool."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--excluded", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    source_ids = {str(row["source_id"]) for row in load_jsonl(args.sources)}
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in load_jsonl(args.objects):
        missing = sorted(set(map(str, row.get("source_ids") or [])) - source_ids)
        if missing:
            excluded.append({
                "object_id": row.get("object_id"),
                "topic_area": row.get("topic_area"),
                "missing_source_ids": missing,
                "reason": "source_not_in_training_pool",
            })
        else:
            kept.append(row)

    write_jsonl(args.output, kept)
    write_jsonl(args.excluded, excluded)
    report = {
        "input_objects": len(kept) + len(excluded),
        "kept": len(kept),
        "excluded": len(excluded),
        "training_sources": len(source_ids),
        "kept_by_topic_area": dict(sorted(collections.Counter(
            str(row.get("topic_area") or "unknown") for row in kept
        ).items())),
        "excluded_by_topic_area": dict(sorted(collections.Counter(
            str(row.get("topic_area") or "unknown") for row in excluded
        ).items())),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
