#!/usr/bin/env python3
"""Split JSONL requests into deterministic, group-preserving shards."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--group-metadata-key")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if args.shards < 1:
        raise ValueError("--shards must be positive")
    rows = read_jsonl(args.input)
    request_ids = [str(row.get("id") or "") for row in rows]
    if any(not request_id for request_id in request_ids):
        raise ValueError("every request must have a nonempty id")
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("request ids must be unique")

    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        if args.group_metadata_key:
            group = str((row.get("metadata") or {}).get(args.group_metadata_key) or "")
            if not group:
                raise ValueError(
                    f"request {row['id']} lacks metadata.{args.group_metadata_key}"
                )
        else:
            group = str(row["id"])
        groups.setdefault(group, []).append(row)

    shard_rows: list[list[dict[str, Any]]] = [[] for _ in range(args.shards)]
    shard_groups: list[list[str]] = [[] for _ in range(args.shards)]
    for group, grouped_rows in groups.items():
        target = min(range(args.shards), key=lambda index: (len(shard_rows[index]), index))
        shard_rows[target].extend(grouped_rows)
        shard_groups[target].append(group)

    width = max(2, len(str(args.shards - 1)))
    outputs: list[dict[str, Any]] = []
    for index, selected in enumerate(shard_rows):
        path = args.output_dir / (
            f"{args.prefix}.shard-{index:0{width}d}-of-{args.shards:0{width}d}.jsonl"
        )
        write_jsonl(path, selected)
        outputs.append(
            {
                "index": index,
                "path": str(path),
                "requests": len(selected),
                "groups": len(shard_groups[index]),
            }
        )

    report = {
        "schema_version": "ksova-jsonl-shard-plan-1.0",
        "input": str(args.input),
        "requests": len(rows),
        "group_metadata_key": args.group_metadata_key,
        "group_count": len(groups),
        "shard_count": args.shards,
        "shards": outputs,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
