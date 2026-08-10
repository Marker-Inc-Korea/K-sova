#!/usr/bin/env python3
"""Select items explicitly accepted by direct-review audit tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def accepted_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["item_id"]
            for row in csv.DictReader(handle, delimiter="\t")
            if row["route"] == "primary"
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item-set", action="append", nargs=2, metavar=("ITEMS", "AUDIT"), required=True)
    parser.add_argument("--exclude-item-id", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    selected: list[dict[str, Any]] = []
    input_counts: dict[str, int] = {}
    for items_value, audit_value in args.item_set:
        items_path = Path(items_value)
        rows = read_jsonl(items_path)
        keep = accepted_ids(Path(audit_value))
        by_id = {str(row["item_id"]): row for row in rows}
        missing = keep - set(by_id)
        if missing:
            raise SystemExit(f"accepted audit IDs missing from {items_path}: {sorted(missing)}")
        selected.extend(by_id[item_id] for item_id in sorted(keep))
        input_counts[str(items_path)] = len(rows)

    ids = [str(row["item_id"]) for row in selected]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate accepted item IDs")
    excluded = set(args.exclude_item_id)
    unknown_excluded = excluded - set(ids)
    if unknown_excluded:
        raise SystemExit(f"excluded item IDs are not selected: {sorted(unknown_excluded)}")
    selected = [row for row in selected if str(row["item_id"]) not in excluded]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    report = {
        "input_counts": input_counts,
        "excluded_item_ids": sorted(excluded),
        "accepted": len(selected),
        "issues": len({row["issue_id"] for row in selected}),
        "prompt_forms": dict(sorted(Counter(row["prompt_form"] for row in selected).items())),
        "item_roles": dict(sorted(Counter(row["item_role"] for row in selected).items())),
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
