#!/usr/bin/env python3
"""Materialize an ordered teacher request set with explicit retry overrides."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-requests", type=Path, required=True)
    parser.add_argument("--request-override", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite request-set outputs")
    base = read_jsonl(args.base_requests)
    base_ids = [str(row.get("id") or "") for row in base]
    if any(not value for value in base_ids) or len(base_ids) != len(set(base_ids)):
        raise ValueError("base request IDs must be nonempty and unique")
    request_by_id = {str(row["id"]): row for row in base}
    overridden_by: dict[str, str] = {}
    for path in args.request_override:
        rows = read_jsonl(path)
        ids = [str(row.get("id") or "") for row in rows]
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{path}: override request IDs must be nonempty and unique")
        unknown = sorted(set(ids) - set(request_by_id))
        if unknown:
            raise ValueError(f"{path}: override IDs are absent from base: {unknown[:5]}")
        for row in rows:
            request_id = str(row["id"])
            request_by_id[request_id] = row
            overridden_by[request_id] = str(path)
    materialized = [request_by_id[value] for value in base_ids]
    report = {
        "schema_version": "ksova-teacher-request-set-report-1.0",
        "base_requests": len(base),
        "request_override_files": [str(path) for path in args.request_override],
        "overridden_request_ids": overridden_by,
        "materialized_requests": len(materialized),
        "hashes": {
            "base_requests_sha256": sha256(args.base_requests),
            "request_overrides_sha256": {
                str(path): sha256(path) for path in args.request_override
            },
        },
    }
    write_jsonl(args.output, materialized)
    report["hashes"]["output_sha256"] = sha256(args.output)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
