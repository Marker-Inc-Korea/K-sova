#!/usr/bin/env python3
"""Merge JSONL artifacts while enforcing a unique stable identifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--id-field", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    merged: list[dict[str, Any]] = []
    seen: dict[str, Path] = {}
    input_counts: dict[str, int] = {}
    for path in args.input:
        rows = read_jsonl(path)
        input_counts[str(path)] = len(rows)
        for row in rows:
            identifier = str(row.get(args.id_field) or "")
            if not identifier:
                raise SystemExit(f"{path}: missing {args.id_field}")
            if identifier in seen:
                raise SystemExit(
                    f"duplicate {args.id_field} {identifier}: {seen[identifier]} and {path}"
                )
            seen[identifier] = path
            merged.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "id_field": args.id_field,
        "input_counts": input_counts,
        "merged": len(merged),
        "output": str(args.output),
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
