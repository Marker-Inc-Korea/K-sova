#!/usr/bin/env python3
"""Merge a preserved material pool with independently curated top-up seeds."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def normalize(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", str(value).casefold())


def terms(value: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[가-힣]{2,}|[A-Za-z0-9]{3,}", str(value))
    }


def jaccard(left: str, right: str) -> float:
    a, b = terms(left), terms(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def source_score(row: dict[str, Any]) -> tuple[int, int]:
    return int(bool(row.get("body_read"))), len(str(row.get("read_scope") or ""))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-seeds", type=Path, required=True)
    parser.add_argument("--topup-seeds", type=Path, required=True)
    parser.add_argument("--primary-sources", type=Path, required=True)
    parser.add_argument("--topup-sources", type=Path, required=True)
    parser.add_argument("--dedup-threshold", type=float, default=0.82)
    parser.add_argument("--output-seeds", type=Path, required=True)
    parser.add_argument("--output-sources", type=Path, required=True)
    parser.add_argument("--duplicate-audit", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    primary = read_jsonl(args.primary_seeds)
    topup = read_jsonl(args.topup_seeds)
    merged = [{**row, "material_pool_origin": "gemma_primary"} for row in primary]
    seen_ids = {str(row["seed_id"]): row for row in merged}
    seen_exact = {normalize(str(row.get("decision_object") or "")): row for row in merged}
    duplicate_audit: list[dict[str, Any]] = []

    for row in topup:
        seed_id = str(row["seed_id"])
        decision = str(row.get("decision_object") or "")
        exact_key = normalize(decision)
        if seed_id in seen_ids:
            duplicate_audit.append({
                "seed_id": seed_id,
                "duplicate_of": seen_ids[seed_id]["seed_id"],
                "reason": "stable_seed_id",
            })
            continue
        if exact_key and exact_key in seen_exact:
            duplicate_audit.append({
                "seed_id": seed_id,
                "duplicate_of": seen_exact[exact_key]["seed_id"],
                "reason": "normalized_exact_decision",
            })
            continue
        best_score, best_row = max(
            ((jaccard(decision, str(prior.get("decision_object") or "")), prior) for prior in merged),
            default=(0.0, None), key=lambda pair: pair[0],
        )
        if best_row is not None and best_score >= args.dedup_threshold:
            duplicate_audit.append({
                "seed_id": seed_id,
                "duplicate_of": best_row["seed_id"],
                "reason": "token_jaccard",
                "similarity": round(best_score, 4),
            })
            continue
        added = {**row, "material_pool_origin": "qwen_independent_topup"}
        merged.append(added)
        seen_ids[seed_id] = added
        if exact_key:
            seen_exact[exact_key] = added

    source_by_id: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(args.primary_sources) + read_jsonl(args.topup_sources):
        source_id = str(row["source_id"])
        if source_id not in source_by_id or source_score(row) > source_score(source_by_id[source_id]):
            source_by_id[source_id] = row

    write_jsonl(args.output_seeds, merged)
    write_jsonl(args.output_sources, sorted(source_by_id.values(), key=lambda row: row["source_id"]))
    write_jsonl(args.duplicate_audit, duplicate_audit)
    report = {
        "schema_version": "ksova-v2-material-pool-merge-1.0",
        "primary_seeds": len(primary),
        "topup_seeds": len(topup),
        "topup_added": len(merged) - len(primary),
        "merged_seeds": len(merged),
        "duplicates_removed": len(duplicate_audit),
        "by_origin": dict(sorted(Counter(row["material_pool_origin"] for row in merged).items())),
        "by_type": dict(sorted(Counter(str(row.get("type_id")) for row in merged).items())),
        "by_domain": dict(sorted(Counter(str(row.get("domain_id")) for row in merged).items())),
        "unique_source_candidates": len({str(row.get("source_candidate_id")) for row in merged}),
        "sources": len(source_by_id),
        "output_seeds": str(args.output_seeds),
        "output_sources": str(args.output_sources),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if merged else 2


if __name__ == "__main__":
    raise SystemExit(main())
