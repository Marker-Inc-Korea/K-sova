#!/usr/bin/env python3
"""Measure item-level score stability across two calibrated Cycle08 judge runs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def weighted_kappa(pairs: list[tuple[int, int]]) -> float | None:
    """Quadratic-weighted Cohen kappa for the fixed 0/1/2 scale."""
    if not pairs:
        return None
    n = len(pairs)
    first = Counter(a for a, _ in pairs)
    second = Counter(b for _, b in pairs)
    observed = sum(((a - b) ** 2) / 4 for a, b in pairs) / n
    expected = sum(
        first[a] * second[b] / (n * n) * (((a - b) ** 2) / 4)
        for a in range(3)
        for b in range(3)
    )
    if expected == 0:
        return 1.0 if observed == 0 else None
    return 1.0 - observed / expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--target", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    first_rows = read_jsonl(args.first)
    second_rows = read_jsonl(args.second)
    first = {(str(row["eval_id"]), str(row["target"])): row for row in first_rows}
    second = {(str(row["eval_id"]), str(row["target"])): row for row in second_rows}
    details: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for target in args.target:
        first_keys = {key for key in first if key[1] == target}
        second_keys = {key for key in second if key[1] == target}
        if first_keys != second_keys or not first_keys:
            raise ValueError(f"coverage mismatch for target {target}")
        pairs: list[tuple[int, int]] = []
        confusion: Counter[str] = Counter()
        for key in sorted(first_keys):
            a, b = int(first[key]["score"]), int(second[key]["score"])
            pairs.append((a, b))
            confusion[f"{a}->{b}"] += 1
            if a != b:
                details.append(
                    {
                        "eval_id": key[0],
                        "target": target,
                        "evaluation_set": first[key]["evaluation_set"],
                        "first_score": a,
                        "second_score": b,
                        "difference": b - a,
                        "first_reason_ko": first[key]["reason_ko"],
                        "second_reason_ko": second[key]["reason_ko"],
                    }
                )
        summary[target] = {
            "n": len(pairs),
            "exact_agreement": sum(a == b for a, b in pairs) / len(pairs),
            "within_one_agreement": sum(abs(a - b) <= 1 for a, b in pairs) / len(pairs),
            "quadratic_weighted_kappa": weighted_kappa(pairs),
            "first_score_100": sum(a for a, _ in pairs) / len(pairs) * 50,
            "second_score_100": sum(b for _, b in pairs) / len(pairs) * 50,
            "second_minus_first_points": sum(b - a for a, b in pairs) / len(pairs) * 50,
            "confusion": dict(sorted(confusion.items())),
        }
    result = {
        "schema_version": "ksova-cycle08-judge-repeatability-1.0",
        "first": str(args.first.resolve()),
        "second": str(args.second.resolve()),
        "targets": args.target,
        "summary": summary,
        "disagreements": details,
    }
    atomic_json(args.out, result)
    print(json.dumps({"summary": summary, "disagreement_count": len(details)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
