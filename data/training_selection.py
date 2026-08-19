#!/usr/bin/env python3
"""Audit K-SOVA issue candidates before generating training examples.

The audit is deterministic and non-mutating.  It blocks exact benchmark-source
reuse, high semantic overlap with selected benchmark issues, and sources that
do not contain generation-eligible retrieved text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ELIGIBLE_CONTENT_SCOPES = {
    "retrieved_full_text",
    "retrieved_snapshot_excerpt",
    "retrieved_excerpt",
    "retrieved_section",
    "retrieved_section_excerpt",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row: {path}:{line_no}")
            rows.append(value)
    return rows


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write(
        path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
    )


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def normalized_url(row: dict[str, Any]) -> str:
    return str(row.get("normalized_url") or row.get("url") or "").strip()


def source_content_hash(row: dict[str, Any]) -> str:
    value = str(row.get("content_sha256") or "").strip()
    if value:
        return value
    text = str(row.get("content_text") or "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""


def source_ready(row: dict[str, Any]) -> bool:
    return all((
        row.get("generation_eligible") is True,
        row.get("needs_body_backfill") is not True,
        row.get("content_scope") in ELIGIBLE_CONTENT_SCOPES,
        bool(str(row.get("content_text") or "").strip()),
    ))


def max_benchmark_similarity(issue: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    neighbors = [
        row for row in issue.get("semantic_nearest_existing") or []
        if row.get("lifecycle_status") == "benchmark_selected"
    ]
    neighbors.sort(
        key=lambda row: (float(row.get("similarity") or 0.0), str(row.get("issue_id") or "")),
        reverse=True,
    )
    maximum = float(neighbors[0].get("similarity") or 0.0) if neighbors else 0.0
    return maximum, neighbors[:5]


def command_audit(args: argparse.Namespace) -> int:
    current_issues = {str(row["issue_id"]): row for row in read_jsonl(args.current_issues)}
    enriched_candidates: dict[str, dict[str, Any]] = {}
    for candidate_path in (args.new_candidates, args.legacy_candidates):
        for row in read_jsonl(candidate_path):
            issue_id = str(row["issue_id"])
            if issue_id in enriched_candidates:
                raise ValueError(f"duplicate enriched candidate: {issue_id}")
            if not row.get("semantic_embedding_model") or not isinstance(
                row.get("semantic_nearest_existing"), list
            ):
                raise ValueError(f"candidate lacks semantic review: {issue_id}")
            enriched_candidates[issue_id] = row
    sources = {str(row["source_id"]): row for row in read_jsonl(args.current_sources)}
    benchmark_items = [
        row for row in read_jsonl(args.current_benchmark_items)
        if row.get("lifecycle_status") == "selected_benchmark"
    ]
    benchmark_source_ids = {
        str(source_id)
        for row in benchmark_items
        for source_id in row.get("source_ids") or []
    }
    missing_benchmark_sources = benchmark_source_ids - set(sources)
    if missing_benchmark_sources:
        raise ValueError(f"missing benchmark sources: {sorted(missing_benchmark_sources)[:3]}")
    benchmark_urls = {
        normalized_url(sources[source_id]) for source_id in benchmark_source_ids
        if normalized_url(sources[source_id])
    }
    benchmark_hashes = {
        source_content_hash(sources[source_id]) for source_id in benchmark_source_ids
        if source_content_hash(sources[source_id])
    }

    new_allocations = [
        row for row in read_jsonl(args.new_allocations)
        if row.get("allocation_lane") == "supporting_train_candidate"
    ]
    legacy_recommendations = [
        row for row in read_jsonl(args.legacy_recommendations)
        if row.get("final_verdict") == "supporting_train"
    ]
    candidate_origins: dict[str, str] = {}
    for row in new_allocations:
        issue_id = str(row["issue_id"])
        candidate_origins[issue_id] = "new_expansion_20260813"
    for row in legacy_recommendations:
        issue_id = str(row["issue_id"])
        if issue_id in candidate_origins:
            raise ValueError(f"candidate appears in both inputs: {issue_id}")
        candidate_origins[issue_id] = "legacy_unreviewed_reserve"
    missing_issues = set(candidate_origins) - set(current_issues)
    if missing_issues:
        raise ValueError(f"missing candidate issues: {sorted(missing_issues)[:3]}")
    missing_enriched = set(candidate_origins) - set(enriched_candidates)
    if missing_enriched:
        raise ValueError(f"missing enriched candidates: {sorted(missing_enriched)[:3]}")

    audited = []
    for issue_id in sorted(candidate_origins):
        current_issue = current_issues[issue_id]
        issue = enriched_candidates[issue_id]
        for key in ("title_ko", "source_ids"):
            if issue.get(key) != current_issue.get(key):
                raise ValueError(f"enriched candidate differs from canonical {key}: {issue_id}")
        source_ids = [str(value) for value in issue.get("source_ids") or []]
        missing_sources = sorted(set(source_ids) - set(sources))
        source_rows = [sources[source_id] for source_id in source_ids if source_id in sources]
        overlap_source_ids = sorted(set(source_ids) & benchmark_source_ids)
        overlap_urls = sorted({
            normalized_url(row) for row in source_rows
            if normalized_url(row) in benchmark_urls
        })
        overlap_hashes = sorted({
            source_content_hash(row) for row in source_rows
            if source_content_hash(row) in benchmark_hashes
        })
        unready_sources = sorted(
            str(row["source_id"]) for row in source_rows if not source_ready(row)
        )
        maximum_similarity, benchmark_neighbors = max_benchmark_similarity(issue)
        flags = []
        if missing_sources:
            flags.append("missing_source")
        if overlap_source_ids or overlap_urls or overlap_hashes:
            flags.append("exact_benchmark_source_overlap")
        if unready_sources:
            flags.append("source_body_not_generation_ready")
        if maximum_similarity >= args.semantic_threshold:
            flags.append("high_benchmark_semantic_similarity")
        if "missing_source" in flags:
            disposition = "blocked_missing_source"
        elif "exact_benchmark_source_overlap" in flags:
            disposition = "hold_exact_benchmark_overlap"
        elif "source_body_not_generation_ready" in flags:
            disposition = "needs_source_body_backfill"
        elif "high_benchmark_semantic_similarity" in flags:
            disposition = "hold_semantic_overlap"
        else:
            disposition = "ready_for_training_generation"
        audited.append({
            "schema_version": "ksova-training-candidate-audit-1.0",
            "issue_id": issue_id,
            "domain_id": issue.get("domain_id"),
            "title_ko": issue.get("title_ko"),
            "candidate_origin": candidate_origins[issue_id],
            "disposition": disposition,
            "review_flags": flags,
            "source_ids": source_ids,
            "source_count": len(source_ids),
            "missing_source_ids": missing_sources,
            "unready_source_ids": unready_sources,
            "benchmark_overlap": {
                "source_ids": overlap_source_ids,
                "normalized_urls": overlap_urls,
                "content_sha256": overlap_hashes,
            },
            "maximum_benchmark_semantic_similarity": round(maximum_similarity, 4),
            "nearest_benchmark_issues": benchmark_neighbors,
            "canonical_catalog_mutated": False,
        })
    write_jsonl(args.output, audited)
    report = {
        "schema_version": "ksova-training-candidate-audit-report-1.0",
        "semantic_threshold": args.semantic_threshold,
        "benchmark_items": len(benchmark_items),
        "benchmark_sources": len(benchmark_source_ids),
        "new_candidates": len(new_allocations),
        "legacy_candidates": len(legacy_recommendations),
        "candidates": len(audited),
        "by_disposition": dict(sorted(Counter(
            row["disposition"] for row in audited
        ).items())),
        "by_origin_and_disposition": {
            f"{origin}__{disposition}": count
            for (origin, disposition), count in sorted(Counter(
                (row["candidate_origin"], row["disposition"]) for row in audited
            ).items())
        },
        "ready_by_domain": dict(sorted(Counter(
            row["domain_id"] for row in audited
            if row["disposition"] == "ready_for_training_generation"
        ).items())),
        "canonical_catalog_mutated": False,
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-issues", type=Path, required=True)
    parser.add_argument("--current-sources", type=Path, required=True)
    parser.add_argument("--current-benchmark-items", type=Path, required=True)
    parser.add_argument("--new-allocations", type=Path, required=True)
    parser.add_argument("--legacy-recommendations", type=Path, required=True)
    parser.add_argument("--new-candidates", type=Path, required=True)
    parser.add_argument("--legacy-candidates", type=Path, required=True)
    parser.add_argument("--semantic-threshold", type=float, default=0.90)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser


def main() -> int:
    try:
        return command_audit(build_parser().parse_args())
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"training selection error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
