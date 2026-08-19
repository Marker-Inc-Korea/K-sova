#!/usr/bin/env python3
"""Apply audited issue swaps after semantic benchmark-holdout review."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


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


def normalized_url(row: dict[str, Any]) -> str:
    raw = str(row.get("normalized_url") or row.get("url") or "").strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((
        parts.scheme.casefold(),
        parts.netloc.casefold(),
        path,
        urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True))),
        "",
    ))


def source_body(row: dict[str, Any]) -> str:
    retrieval = row.get("retrieval") or {}
    return str(
        row.get("content_text")
        or (retrieval.get("text_excerpt") if isinstance(retrieval, dict) else "")
        or ""
    )


def content_hash(row: dict[str, Any]) -> str:
    declared = str(row.get("content_sha256") or "").strip().casefold()
    if declared:
        return declared
    body = source_body(row)
    return hashlib.sha256(body.encode("utf-8")).hexdigest() if body else ""


def load_source_directory(path: Path) -> dict[str, dict[str, Any]]:
    sources: dict[str, dict[str, Any]] = {}
    for name in ("verified_sources.jsonl", "sources.jsonl", "integration_sources.jsonl"):
        candidate = path / name
        if not candidate.exists():
            continue
        for wrapper in read_jsonl(candidate):
            row = wrapper.get("source") if isinstance(wrapper.get("source"), dict) else wrapper
            source_id = str(row.get("source_id") or "")
            if not source_id:
                raise ValueError(f"{candidate}: source row lacks source_id")
            prior = sources.get(source_id)
            if prior is None or len(source_body(row)) > len(source_body(prior)):
                sources[source_id] = row
    return sources


def overlay_backfills(
    sources: dict[str, dict[str, Any]], paths: list[Path],
) -> set[str]:
    used: set[str] = set()
    for path in paths:
        for wrapper in read_jsonl(path):
            fetched = wrapper.get("source") if isinstance(wrapper.get("source"), dict) else wrapper
            fetched_body = source_body(fetched)
            fetched_url = normalized_url(fetched)
            for source_id_value in wrapper.get("lead_ids") or [fetched.get("source_id")]:
                source_id = str(source_id_value or "")
                original = sources.get(source_id)
                if original is None:
                    raise ValueError(f"backfill references unknown source: {source_id}")
                if fetched_url != normalized_url(original):
                    raise ValueError(f"backfill URL mismatch: {source_id}")
                if len(fetched_body) > len(source_body(original)):
                    sources[source_id] = {
                        **original,
                        "content_text": fetched_body,
                        "retrieval": fetched.get("retrieval") or original.get("retrieval"),
                    }
                    used.add(source_id)
    return used


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-issues", type=Path, required=True)
    parser.add_argument("--base-sources", type=Path, required=True)
    parser.add_argument("--benchmark-items", type=Path, required=True)
    parser.add_argument("--canonical-sources", type=Path, required=True)
    parser.add_argument("--source-backfills", type=Path, action="append", default=[])
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-issues", type=Path, required=True)
    parser.add_argument("--output-sources", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    outputs = (args.output_issues, args.output_sources, args.source_audit, args.report)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite outputs: {existing}")

    selection = read_json(args.selection)
    if selection.get("schema_version") != "ksova-semantic-holdout-swap-selection-1.0":
        raise ValueError("unexpected swap selection schema")
    base_dossiers = read_jsonl(args.base_issues)
    base_sources = read_jsonl(args.base_sources)
    benchmark_items = read_jsonl(args.benchmark_items)
    canonical_sources = read_jsonl(args.canonical_sources)
    if len(base_dossiers) != 150 or len(benchmark_items) != 480:
        raise ValueError("expected 150 base issues and frozen 480 benchmark items")
    dossier_by_id = {str(row.get("issue_id") or ""): row for row in base_dossiers}
    if len(dossier_by_id) != len(base_dossiers):
        raise ValueError("duplicate or missing base issue ID")
    drop_ids = [str(value) for value in selection.get("drop_issue_ids") or []]
    additions = selection.get("additions") or []
    if len(drop_ids) != len(set(drop_ids)) or not set(drop_ids) <= set(dossier_by_id):
        raise ValueError("drop IDs must be unique members of the base portfolio")
    if len(drop_ids) != len(additions) or not drop_ids:
        raise ValueError("swap selection requires equal nonzero drop and addition counts")

    source_lookup = {
        str(row.get("source_id") or ""): row for row in [*canonical_sources, *base_sources]
        if row.get("source_id")
    }
    addition_dossiers: list[dict[str, Any]] = []
    for addition in additions:
        issue_path = Path(str(addition["issue_path"]))
        source_dir = Path(str(addition["source_dir"]))
        matches = [
            row for row in read_jsonl(issue_path)
            if str(row.get("issue_id") or "") == str(addition["issue_id"])
        ]
        if len(matches) != 1:
            raise ValueError(f"addition issue lookup is not unique: {addition['issue_id']}")
        issue = copy.deepcopy(matches[0])
        validation = ((issue.get("selection_review") or {}).get("validation") or {})
        if (
            issue.get("review_status") != "llm_body_validated"
            or validation.get("effective_pass") is not True
        ):
            raise ValueError(f"addition lacks a passing body validation: {issue['issue_id']}")
        directory_sources = load_source_directory(source_dir)
        source_ids = [str(value) for value in issue.get("source_ids") or []]
        if not source_ids or any(value not in directory_sources for value in source_ids):
            raise ValueError(f"addition sources are incomplete: {issue['issue_id']}")
        for source_id in source_ids:
            row = directory_sources[source_id]
            prior = source_lookup.get(source_id)
            if prior is not None and normalized_url(prior) != normalized_url(row):
                raise ValueError(f"conflicting source ID: {source_id}")
            if prior is None or len(source_body(row)) > len(source_body(prior)):
                source_lookup[source_id] = row
        addition_dossiers.append({
            "schema_version": "ksova-training-portfolio-issue-1.0",
            "experiment_id": selection["experiment_id"],
            "issue_id": issue["issue_id"],
            "domain_id": issue["domain_id"],
            "origin": "semantic_holdout_replacement",
            "selection_reason_ko": str(addition.get("reason_ko") or ""),
            "source_basis": "verified_issue_body_sources",
            "source_ids": source_ids,
            "source_isolated_from_frozen_benchmark_480": True,
            "issue_card": issue,
        })
    backfilled_source_ids = overlay_backfills(source_lookup, args.source_backfills)

    add_ids = [str(row["issue_id"]) for row in addition_dossiers]
    benchmark_issue_ids = {str(row.get("issue_id") or "") for row in benchmark_items}
    if (
        len(add_ids) != len(set(add_ids))
        or set(add_ids) & set(dossier_by_id)
        or set(add_ids) & benchmark_issue_ids
    ):
        raise ValueError("addition issue IDs are duplicate or collide with existing issues")

    canonical_source_lookup = {
        str(row.get("source_id") or ""): row for row in canonical_sources
        if row.get("source_id")
    }
    benchmark_source_ids = {
        str(value) for row in benchmark_items for value in row.get("source_ids") or []
    }
    if not benchmark_source_ids <= set(canonical_source_lookup):
        raise ValueError("benchmark source registry is incomplete")
    benchmark_urls = {
        normalized_url(canonical_source_lookup[value]) for value in benchmark_source_ids
        if normalized_url(canonical_source_lookup[value])
    }
    benchmark_hashes = {
        content_hash(canonical_source_lookup[value]) for value in benchmark_source_ids
        if content_hash(canonical_source_lookup[value])
    }

    final_dossiers = [
        row for row in base_dossiers if str(row["issue_id"]) not in set(drop_ids)
    ] + addition_dossiers
    final_dossiers.sort(key=lambda row: str(row["issue_id"]))
    if len(final_dossiers) != 150 or len({row["issue_id"] for row in final_dossiers}) != 150:
        raise ValueError("final portfolio must contain exactly 150 unique issues")
    source_audit_rows = []
    selected_source_ids: set[str] = set()
    for dossier in final_dossiers:
        issue_id = str(dossier["issue_id"])
        source_ids = [str(value) for value in dossier.get("source_ids") or []]
        missing = sorted(set(source_ids) - set(source_lookup))
        if missing:
            raise ValueError(f"{issue_id}: missing sources {missing}")
        urls = [normalized_url(source_lookup[value]) for value in source_ids]
        hashes = [content_hash(source_lookup[value]) for value in source_ids]
        collisions = {
            "source_ids": sorted(set(source_ids) & benchmark_source_ids),
            "urls": sorted({value for value in urls if value and value in benchmark_urls}),
            "content": sorted({value for value in hashes if value and value in benchmark_hashes}),
        }
        if any(collisions.values()):
            raise ValueError(f"{issue_id}: benchmark source collision {collisions}")
        if not any(len(source_body(source_lookup[value])) >= 300 for value in source_ids):
            raise ValueError(f"{issue_id}: no usable source body")
        selected_source_ids.update(source_ids)
        source_audit_rows.append({
            "issue_id": issue_id,
            "source_ids": source_ids,
            "normalized_urls": urls,
            "content_sha256": hashes,
            "benchmark_source_id_collisions": [],
            "benchmark_url_collisions": [],
            "benchmark_content_collisions": [],
        })

    final_sources = [source_lookup[value] for value in sorted(selected_source_ids)]
    by_domain = dict(sorted(Counter(row["domain_id"] for row in final_dossiers).items()))
    expected_by_domain = {
        str(key): int(value)
        for key, value in (selection.get("expected_by_domain") or {}).items()
    }
    if expected_by_domain and by_domain != expected_by_domain:
        raise ValueError(f"domain distribution mismatch: {by_domain} != {expected_by_domain}")
    report = {
        "schema_version": "ksova-semantic-holdout-swap-report-1.0",
        "experiment_id": selection["experiment_id"],
        "base_issues": len(base_dossiers),
        "dropped_issue_ids": drop_ids,
        "added_issue_ids": add_ids,
        "final_issues": len(final_dossiers),
        "final_sources": len(final_sources),
        "backfilled_source_ids": sorted(backfilled_source_ids),
        "by_domain": by_domain,
        "benchmark": {
            "items": len(benchmark_items),
            "issue_collisions": 0,
            "source_id_collisions": 0,
            "url_collisions": 0,
            "content_collisions": 0,
        },
        "hashes": {
            "selection_sha256": sha256(args.selection),
            "base_issues_sha256": sha256(args.base_issues),
            "base_sources_sha256": sha256(args.base_sources),
            "benchmark_items_sha256": sha256(args.benchmark_items),
        },
    }
    write_jsonl(args.output_issues, final_dossiers)
    write_jsonl(args.output_sources, final_sources)
    write_jsonl(args.source_audit, source_audit_rows)
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
