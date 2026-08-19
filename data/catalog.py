#!/usr/bin/env python3
"""Build and validate the canonical K-SOVA v2 data catalog.

The mutable catalog lives in data/current.  Benchmark releases and training
experiments remain immutable downstream artifacts under benchmark/ and train/.
This script performs the one-time consolidation of the existing v2 assets and
can subsequently validate the catalog without relying on the old work tree.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
CURRENT = DATA_ROOT / "current"
STAGING = DATA_ROOT / ".staging"
PREVIOUS = DATA_ROOT / ".previous"

INPUTS = {
    "rich_sources": Path("benchmark/v2.0/work/current/sources/source_registry_1576.jsonl"),
    "legacy_sources": Path("benchmark/v2.0/sources/source_registry.jsonl"),
    "base_issues": Path("benchmark/v2.0/work/current/materials/benchmark_issues_350.jsonl"),
    "issue_recommendations": Path(
        "benchmark/v2.0/work/runs/core_issue_selection_250_20260809_v1/recommendations.jsonl"
    ),
    "raw_benchmark_candidates": Path(
        "benchmark/v2.0/work/current/items/benchmark_candidates_1000.jsonl"
    ),
    "judged_benchmark_candidates": Path(
        "benchmark/v2.0/work/runs/pilot500_20260808/item_validation/glm_item_judged.jsonl"
    ),
    "accepted_benchmark_candidates": Path(
        "benchmark/v2.0/work/runs/pilot500_20260808/item_validation/accepted_items.jsonl"
    ),
    "selected_benchmark_items": Path(
        "benchmark/v2.0/work/runs/core_benchmark_items_250x2_20260809_v1/items.jsonl"
    ),
    "cap_excluded_benchmark_items": Path(
        "benchmark/v2.0/work/runs/core_benchmark_items_250x2_20260809_v1/excluded_by_cap.jsonl"
    ),
    "release_items": Path(
        "benchmark/v2.0/releases/first_pass_20260810_v1/benchmark/items.jsonl"
    ),
    "issue_contracts": Path(
        "benchmark/v2.0/releases/first_pass_20260810_v1/scoring/issue_contracts.jsonl"
    ),
    "item_contracts": Path(
        "benchmark/v2.0/releases/first_pass_20260810_v1/scoring/item_contracts.jsonl"
    ),
    "training_dossiers": Path(
        "benchmark/v2.0/work/runs/first_pass_20260810_v1/training/dossiers.jsonl"
    ),
    "standard_a_examples": Path("train/data/standard_a_partial_288/all_examples.jsonl"),
    "expanded_b_examples": Path("train/data/expanded_b_partial_416/all_examples.jsonl"),
}

OUTPUT_FILES = (
    "sources.jsonl",
    "issues.jsonl",
    "benchmark_items.jsonl",
    "train_examples.jsonl",
)

GENERATION_ELIGIBLE_CONTENT_SCOPES = {
    "retrieved_full_text",
    "retrieved_snapshot_excerpt",
    "retrieved_excerpt",
    "retrieved_section",
    "retrieved_section_excerpt",
}

EXPECTED_MIGRATION_COUNTS = {
    "issues": 377,
    "benchmark_selected_issues": 250,
    "train_selected_issues": 32,
    "reserve_issues": 95,
    "benchmark_items": 1104,
    "selected_benchmark_items": 480,
    "validated_reserve_items": 503,
    "rejected_items": 121,
    "train_examples": 416,
    "standard_a_examples": 288,
    "expanded_b_examples": 416,
}

TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
}


class CatalogError(RuntimeError):
    """Raised when catalog integrity cannot be guaranteed."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CatalogError(f"missing input: {path.relative_to(ROOT)}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CatalogError(f"invalid JSON: {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise CatalogError(f"JSONL row is not an object: {path}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def keyed(rows: Iterable[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise CatalogError(f"{label} row has no {key}")
        if value in result:
            raise CatalogError(f"duplicate {key} in {label}: {value}")
        result[value] = row
    return result


def unique(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    seen: set[Any] = set()
    for value in values:
        if value is None or value == "" or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def normalize_url(url: str) -> str:
    if not url or not isinstance(url, str):
        return ""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip()
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]
    path = parts.path.rstrip("/") or "/"
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in TRACKING_QUERY_KEYS
    ]
    return urlunsplit((scheme, netloc, path, urlencode(sorted(query_pairs)), ""))


def source_content_score(row: dict[str, Any]) -> tuple[int, int, int]:
    status_rank = {
        "dossier_snapshot_verified": 4,
        "fetched": 4,
        "reachable": 2,
        "current_v2_card": 1,
    }.get(str(row.get("retrieval_status", "")), 0)
    snapshots = row.get("raw_snapshot_files") or []
    text = str(row.get("content_excerpt") or "").strip()
    return (1 if snapshots else 0, status_rank, len(text))


def canonical_source_preference(
    source_id: str,
    row: dict[str, Any],
    referenced_source_ids: set[str],
) -> tuple[int, int, int, str]:
    return (
        0 if source_id in referenced_source_ids else 1,
        0 if source_id.startswith("SRCV2-") else 1,
        -source_content_score(row)[2],
        source_id,
    )


def expanded_snapshot_paths(records: Iterable[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    prefix = Path("benchmark/v2.0/work/current/sources")
    for record in records:
        for value in record.get("raw_snapshot_files") or []:
            path = Path(str(value))
            if not path.is_absolute() and path.parts[:1] == ("raw_snapshots",):
                path = prefix / path
            paths.append(path.as_posix())
    return sorted(unique(paths))


def collect_referenced_source_ids(loaded: dict[str, list[dict[str, Any]]]) -> set[str]:
    referenced: set[str] = set()
    direct_sets = (
        "base_issues",
        "raw_benchmark_candidates",
        "judged_benchmark_candidates",
        "accepted_benchmark_candidates",
        "selected_benchmark_items",
        "cap_excluded_benchmark_items",
        "release_items",
        "training_dossiers",
    )
    for name in direct_sets:
        for row in loaded[name]:
            referenced.update(str(value) for value in row.get("source_ids") or [])
    for name in ("standard_a_examples", "expanded_b_examples"):
        for row in loaded[name]:
            metadata = row.get("metadata") or {}
            referenced.update(str(value) for value in metadata.get("source_ids") or [])
    return referenced


def consolidate_sources(
    loaded: dict[str, list[dict[str, Any]]],
    referenced_source_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rich_by_id = keyed(loaded["rich_sources"], "source_id", "rich_sources")
    rich_url_groups: dict[str, list[str]] = defaultdict(list)
    for source_id, row in rich_by_id.items():
        normalized = normalize_url(str(row.get("url") or ""))
        if normalized:
            rich_url_groups[normalized].append(source_id)

    alias_to_primary: dict[str, str] = {}
    rich_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_ids in rich_url_groups.values():
        primary = min(
            source_ids,
            key=lambda source_id: canonical_source_preference(
                source_id, rich_by_id[source_id], referenced_source_ids
            ),
        )
        for source_id in source_ids:
            alias_to_primary[source_id] = primary
            rich_groups[primary].append(rich_by_id[source_id])
    for source_id, row in rich_by_id.items():
        if source_id not in alias_to_primary:
            alias_to_primary[source_id] = source_id
            rich_groups[source_id].append(row)

    def register_alias(alias: str, primary: str) -> None:
        if not alias:
            return
        existing = alias_to_primary.get(alias)
        if existing is not None and existing != primary:
            raise CatalogError(
                f"source alias maps to multiple canonical sources: {alias}: "
                f"{existing}, {primary}"
            )
        alias_to_primary[alias] = primary

    for source_id, row in rich_by_id.items():
        primary = alias_to_primary[source_id]
        provenance = row.get("provenance") or {}
        for alias in provenance.get("legacy_source_ids") or []:
            register_alias(str(alias), primary)

    primary_by_normalized_url: dict[str, str] = {}
    for primary, records in rich_groups.items():
        for record in records:
            normalized = normalize_url(str(record.get("url") or ""))
            if normalized:
                existing = primary_by_normalized_url.get(normalized)
                if existing is not None and existing != primary:
                    raise CatalogError(f"unmerged rich source URL: {normalized}")
                primary_by_normalized_url[normalized] = primary

    legacy_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in loaded["legacy_sources"]:
        legacy_id = str(row.get("source_id") or "")
        if not legacy_id:
            raise CatalogError("legacy source row has no source_id")
        normalized = normalize_url(str(row.get("url") or ""))
        primary = alias_to_primary.get(legacy_id)
        if primary is None and normalized:
            primary = primary_by_normalized_url.get(normalized)
        if primary is None:
            primary = legacy_id
            rich_groups.setdefault(primary, [])
            if normalized:
                existing = primary_by_normalized_url.get(normalized)
                if existing is not None:
                    primary = existing
                else:
                    primary_by_normalized_url[normalized] = primary
        register_alias(legacy_id, primary)
        legacy_groups[primary].append(row)

    canonical_rows: list[dict[str, Any]] = []
    for primary in sorted(rich_groups):
        records = rich_groups[primary]
        legacy_records = legacy_groups.get(primary, [])
        primary_record = next(
            (record for record in records if record.get("source_id") == primary),
            None,
        )
        if primary_record is None and records:
            primary_record = min(
                records,
                key=lambda row: canonical_source_preference(
                    str(row.get("source_id")), row, referenced_source_ids
                ),
            )
        primary_record = primary_record or {}
        content_record = max(records, key=source_content_score) if records else {}

        def first_value(field: str, default: Any = "") -> Any:
            candidates = [primary_record, *records, *legacy_records]
            return next(
                (candidate.get(field) for candidate in candidates if candidate.get(field)),
                default,
            )

        content_text = str(content_record.get("content_excerpt") or "").strip()
        raw_snapshots = expanded_snapshot_paths(records)
        retrieval_status = str(content_record.get("retrieval_status") or "url_only")
        if raw_snapshots:
            content_scope = "retrieved_snapshot_excerpt"
        elif retrieval_status in {"fetched", "dossier_snapshot_verified"}:
            content_scope = "retrieved_excerpt"
        elif retrieval_status == "current_v2_card":
            content_scope = "curated_summary"
        elif content_text:
            content_scope = "legacy_excerpt_unverified"
        else:
            content_scope = "url_only"
        generation_eligible = bool(
            len(content_text) >= 500
            and content_scope in GENERATION_ELIGIBLE_CONTENT_SCOPES
        )

        domain_ids: list[str] = []
        source_classes: list[str] = []
        availability: list[str] = []
        content_relevance: list[str] = []
        for row in legacy_records:
            domain_ids.extend(str(value) for value in row.get("domain_ids") or [])
            if row.get("source_class"):
                source_classes.append(str(row["source_class"]))
            if row.get("availability"):
                availability.append(str(row["availability"]))
            if row.get("content_relevance"):
                content_relevance.append(str(row["content_relevance"]))

        evidence_summary = str(primary_record.get("evidence_summary_ko") or "")
        if not evidence_summary:
            evidence_summary = str(first_value("evidence_scope_ko", ""))
        input_source_ids = sorted(
            unique(
                [str(record.get("source_id")) for record in records]
                + [str(row.get("source_id")) for row in legacy_records]
            )
        )
        aliases = sorted(
            alias
            for alias, target in alias_to_primary.items()
            if target == primary and alias != primary
        )
        canonical_rows.append(
            {
                "schema_version": "ksova-data-source-1.0",
                "source_id": primary,
                "aliases": aliases,
                "title": str(first_value("title", "")),
                "publisher": str(first_value("publisher", "")),
                "url": str(first_value("url", "")),
                "normalized_url": normalize_url(str(first_value("url", ""))),
                "jurisdiction": str(first_value("jurisdiction", "")),
                "authority_type": str(first_value("authority_type", "")),
                "retrieval_status": retrieval_status,
                "retrieved_at": str(content_record.get("retrieved_at") or ""),
                "raw_snapshot_files": raw_snapshots,
                "content_text": content_text,
                "content_scope": content_scope,
                "content_sha256": sha256_text(content_text) if content_text else "",
                "original_content_sha256": str(content_record.get("content_sha256") or ""),
                "content_char_count": len(content_text),
                "generation_eligible": generation_eligible,
                "needs_body_backfill": not generation_eligible,
                "evidence_summary_ko": evidence_summary,
                "evidence_claims": copy.deepcopy(
                    primary_record.get("evidence_claims") or []
                ),
                "classification": {
                    "domain_ids": sorted(unique(domain_ids)),
                    "source_classes": sorted(unique(source_classes)),
                    "availability": sorted(unique(availability)),
                    "content_relevance": sorted(unique(content_relevance)),
                },
                "provenance": {
                    "input_source_ids": input_source_ids,
                    "content_source_id": str(content_record.get("source_id") or primary),
                    "rich_registry": copy.deepcopy(primary_record.get("provenance") or {}),
                    "legacy_registry_records": copy.deepcopy(legacy_records),
                },
                "usage": {
                    "pools": [],
                    "issue_ids": [],
                    "benchmark_item_count": 0,
                    "train_example_count": 0,
                },
            }
        )

    canonical_ids = {row["source_id"] for row in canonical_rows}
    for alias, primary in alias_to_primary.items():
        if primary not in canonical_ids:
            raise CatalogError(f"source alias target does not exist: {alias} -> {primary}")
    return canonical_rows, alias_to_primary


def remap_source_ids(source_ids: Iterable[Any], aliases: dict[str, str]) -> list[str]:
    remapped: list[str] = []
    for value in source_ids:
        source_id = str(value)
        if source_id not in aliases:
            raise CatalogError(f"referenced source is absent from registry: {source_id}")
        remapped.append(aliases[source_id])
    return unique(remapped)


def merge_benchmark_items(
    loaded: dict[str, list[dict[str, Any]]], aliases: dict[str, str]
) -> list[dict[str, Any]]:
    input_names = (
        "raw_benchmark_candidates",
        "judged_benchmark_candidates",
        "accepted_benchmark_candidates",
        "cap_excluded_benchmark_items",
        "selected_benchmark_items",
        "release_items",
    )
    maps = {
        name: keyed(loaded[name], "item_id", name)
        for name in input_names
    }
    merged: dict[str, dict[str, Any]] = {}
    for name in input_names:
        for item_id, row in maps[name].items():
            merged.setdefault(item_id, {}).update(copy.deepcopy(row))

    release_ids = set(maps["release_items"])
    accepted_ids = set(maps["accepted_benchmark_candidates"])
    cap_excluded_ids = set(maps["cap_excluded_benchmark_items"])
    item_contracts = keyed(loaded["item_contracts"], "item_id", "item_contracts")

    rows: list[dict[str, Any]] = []
    for item_id in sorted(merged):
        row = merged[item_id]
        row["source_ids"] = remap_source_ids(row.get("source_ids") or [], aliases)
        pipeline_status = str(row.pop("status", ""))
        validation = row.get("pipeline_item_audit") or {}
        glm = validation.get("glm") or {}
        legacy_validation = row.get("glm_item_validation") or {}
        effective_pass = bool(
            item_id in accepted_ids
            or item_id in cap_excluded_ids
            or glm.get("effective_pass") is True
            or legacy_validation.get("effective_pass") is True
        )
        if item_id in release_ids:
            lifecycle_status = "selected_benchmark"
        elif effective_pass:
            lifecycle_status = "validated_reserve"
        else:
            lifecycle_status = "rejected"
        row["schema_version"] = "ksova-data-benchmark-item-1.0"
        row["pipeline_status"] = pipeline_status
        row["lifecycle_status"] = lifecycle_status
        row["release_memberships"] = (
            ["first_pass_20260810_v1"] if item_id in release_ids else []
        )
        row["scoring_contract"] = copy.deepcopy(item_contracts.get(item_id))
        rows.append(row)

    status_counts = Counter(row["lifecycle_status"] for row in rows)
    expected = {
        "selected_benchmark": EXPECTED_MIGRATION_COUNTS["selected_benchmark_items"],
        "validated_reserve": EXPECTED_MIGRATION_COUNTS["validated_reserve_items"],
        "rejected": EXPECTED_MIGRATION_COUNTS["rejected_items"],
    }
    if len(rows) != EXPECTED_MIGRATION_COUNTS["benchmark_items"]:
        raise CatalogError(f"unexpected benchmark item count: {len(rows)}")
    if dict(status_counts) != expected:
        raise CatalogError(
            f"unexpected benchmark lifecycle counts: {dict(status_counts)} != {expected}"
        )
    if set(item_contracts) != release_ids:
        raise CatalogError("item scoring contracts do not exactly match release items")
    return rows


def merge_train_examples(
    loaded: dict[str, list[dict[str, Any]]], aliases: dict[str, str]
) -> list[dict[str, Any]]:
    def training_map(name: str) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for row in loaded[name]:
            metadata = row.get("metadata") or {}
            training_id = metadata.get("training_id")
            if not isinstance(training_id, str) or not training_id:
                raise CatalogError(f"{name} row has no metadata.training_id")
            if training_id in result:
                raise CatalogError(f"duplicate training_id in {name}: {training_id}")
            result[training_id] = row
        return result

    standard = training_map("standard_a_examples")
    expanded_map = training_map("expanded_b_examples")
    if not set(standard).issubset(expanded_map):
        raise CatalogError("Standard A is not a subset of Expanded B")
    for training_id, row in standard.items():
        if row != expanded_map[training_id]:
            raise CatalogError(f"training example differs between experiments: {training_id}")

    rows: list[dict[str, Any]] = []
    for training_id in sorted(expanded_map):
        row = copy.deepcopy(expanded_map[training_id])
        metadata = row.get("metadata") or {}
        metadata["source_ids"] = remap_source_ids(metadata.get("source_ids") or [], aliases)
        row["metadata"] = metadata
        memberships = ["expanded_b_partial_416"]
        if training_id in standard:
            memberships.insert(0, "standard_a_partial_288")
        row["schema_version"] = "ksova-data-train-example-1.0"
        row["experiment_memberships"] = memberships
        rows.append(row)

    if len(rows) != EXPECTED_MIGRATION_COUNTS["train_examples"]:
        raise CatalogError(f"unexpected training example count: {len(rows)}")
    if len(standard) != EXPECTED_MIGRATION_COUNTS["standard_a_examples"]:
        raise CatalogError(f"unexpected Standard A count: {len(standard)}")
    return rows


def merge_issues(
    loaded: dict[str, list[dict[str, Any]]],
    benchmark_items: list[dict[str, Any]],
    train_examples: list[dict[str, Any]],
    aliases: dict[str, str],
) -> list[dict[str, Any]]:
    base = keyed(loaded["base_issues"], "issue_id", "base_issues")
    recommendations = keyed(
        loaded["issue_recommendations"], "issue_id", "issue_recommendations"
    )
    dossiers = keyed(loaded["training_dossiers"], "issue_id", "training_dossiers")
    issue_contracts = keyed(loaded["issue_contracts"], "issue_id", "issue_contracts")

    selected_items_by_issue: dict[str, list[str]] = defaultdict(list)
    candidate_items_by_issue: dict[str, list[str]] = defaultdict(list)
    item_sources_by_issue: dict[str, list[str]] = defaultdict(list)
    selected_benchmark_issues: set[str] = set()
    for item in benchmark_items:
        issue_id = str(item.get("issue_id") or "")
        if not issue_id:
            raise CatalogError(f"benchmark item has no issue_id: {item.get('item_id')}")
        candidate_items_by_issue[issue_id].append(str(item["item_id"]))
        item_sources_by_issue[issue_id].extend(item.get("source_ids") or [])
        if item["lifecycle_status"] == "selected_benchmark":
            selected_benchmark_issues.add(issue_id)
            selected_items_by_issue[issue_id].append(str(item["item_id"]))

    train_examples_by_issue: dict[str, list[str]] = defaultdict(list)
    train_sources_by_issue: dict[str, list[str]] = defaultdict(list)
    train_issues: set[str] = set()
    for example in train_examples:
        metadata = example["metadata"]
        issue_id = str(metadata.get("issue_id") or "")
        if not issue_id:
            raise CatalogError(
                f"training example has no issue_id: {metadata.get('training_id')}"
            )
        train_issues.add(issue_id)
        train_examples_by_issue[issue_id].append(str(metadata["training_id"]))
        train_sources_by_issue[issue_id].extend(metadata.get("source_ids") or [])

    if selected_benchmark_issues & train_issues:
        raise CatalogError("benchmark and training issue pools overlap")

    all_issue_ids = set(base) | set(recommendations) | set(dossiers) | train_issues
    rows: list[dict[str, Any]] = []
    for issue_id in sorted(all_issue_ids):
        row = copy.deepcopy(base.get(issue_id) or {})
        recommendation = recommendations.get(issue_id)
        dossier = dossiers.get(issue_id)
        if not row:
            source = recommendation or dossier or {}
            row = {
                "issue_id": issue_id,
                "title_ko": str(source.get("title_ko") or ""),
                "domain_id": str(
                    source.get("domain_id") or source.get("category_id") or ""
                ),
                "family_id": "",
                "slot_id": "",
                "decision_core_ko": "",
                "sovereign_stake_ko": "",
                "korean_reference_direction_ko": "",
                "novelty_key_ko": "",
                "discourse_layer_labels": [],
                "source_ids": [],
            }
        if not row.get("title_ko"):
            row["title_ko"] = str(
                (recommendation or {}).get("title_ko")
                or (dossier or {}).get("title_ko")
                or ""
            )
        if not row.get("domain_id"):
            row["domain_id"] = str(
                (recommendation or {}).get("category_id")
                or (dossier or {}).get("category_id")
                or ""
            )

        source_ids: list[Any] = list(row.get("source_ids") or [])
        if dossier:
            source_ids.extend(dossier.get("source_ids") or [])
        source_ids.extend(item_sources_by_issue.get(issue_id, []))
        source_ids.extend(train_sources_by_issue.get(issue_id, []))
        row["source_ids"] = remap_source_ids(source_ids, aliases)

        if issue_id in selected_benchmark_issues:
            lifecycle_status = "benchmark_selected"
        elif issue_id in train_issues:
            lifecycle_status = "train_selected"
        else:
            lifecycle_status = "reserve"
        row["schema_version"] = "ksova-data-issue-1.0"
        row["lifecycle_status"] = lifecycle_status
        row["review_status"] = "reviewed" if recommendation else "not_reviewed"
        row["selection_review"] = copy.deepcopy(recommendation)
        if dossier:
            dossier_summary = copy.deepcopy(dossier)
            validated = dossier_summary.pop("validated_examples", [])
            dossier_summary["validated_example_ids"] = [
                str(example.get("example_id"))
                for example in validated
                if example.get("example_id")
            ]
            row["training_dossier"] = dossier_summary
        else:
            row["training_dossier"] = None
        row["scoring_contract"] = copy.deepcopy(issue_contracts.get(issue_id))
        row["benchmark_item_ids"] = sorted(selected_items_by_issue.get(issue_id, []))
        row["candidate_item_ids"] = sorted(candidate_items_by_issue.get(issue_id, []))
        row["training_example_ids"] = sorted(train_examples_by_issue.get(issue_id, []))
        rows.append(row)

    status_counts = Counter(row["lifecycle_status"] for row in rows)
    expected = {
        "benchmark_selected": EXPECTED_MIGRATION_COUNTS["benchmark_selected_issues"],
        "train_selected": EXPECTED_MIGRATION_COUNTS["train_selected_issues"],
        "reserve": EXPECTED_MIGRATION_COUNTS["reserve_issues"],
    }
    if len(rows) != EXPECTED_MIGRATION_COUNTS["issues"]:
        raise CatalogError(f"unexpected issue count: {len(rows)}")
    if dict(status_counts) != expected:
        raise CatalogError(f"unexpected issue lifecycle counts: {dict(status_counts)}")
    if set(issue_contracts) != selected_benchmark_issues:
        raise CatalogError("issue scoring contracts do not exactly match benchmark issues")
    return rows


def attach_source_usage(
    sources: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    benchmark_items: list[dict[str, Any]],
    train_examples: list[dict[str, Any]],
) -> None:
    usage: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "pools": set(),
            "issue_ids": set(),
            "benchmark_item_count": 0,
            "train_example_count": 0,
        }
    )
    for issue in issues:
        for source_id in issue.get("source_ids") or []:
            usage[source_id]["issue_ids"].add(issue["issue_id"])
            usage[source_id]["pools"].add(issue["lifecycle_status"])
    for item in benchmark_items:
        for source_id in item.get("source_ids") or []:
            usage[source_id]["benchmark_item_count"] += 1
            usage[source_id]["pools"].add(item["lifecycle_status"])
    for example in train_examples:
        for source_id in example["metadata"].get("source_ids") or []:
            usage[source_id]["train_example_count"] += 1
            usage[source_id]["pools"].add("train_example")
    for source in sources:
        value = usage[source["source_id"]]
        source["usage"] = {
            "pools": sorted(value["pools"]),
            "issue_ids": sorted(value["issue_ids"]),
            "benchmark_item_count": value["benchmark_item_count"],
            "train_example_count": value["train_example_count"],
        }


def selected_source_sets(
    sources: list[dict[str, Any]],
    benchmark_items: list[dict[str, Any]],
    train_examples: list[dict[str, Any]],
) -> dict[str, Any]:
    source_map = {row["source_id"]: row for row in sources}
    benchmark_source_ids = {
        source_id
        for item in benchmark_items
        if item["lifecycle_status"] == "selected_benchmark"
        for source_id in item.get("source_ids") or []
    }
    train_source_ids = {
        source_id
        for example in train_examples
        for source_id in example["metadata"].get("source_ids") or []
    }

    def values(source_ids: set[str], field: str) -> set[str]:
        return {
            str(source_map[source_id].get(field) or "")
            for source_id in source_ids
            if source_id in source_map and source_map[source_id].get(field)
        }

    return {
        "benchmark_source_count": len(benchmark_source_ids),
        "train_source_count": len(train_source_ids),
        "source_id_overlap": sorted(benchmark_source_ids & train_source_ids),
        "normalized_url_overlap": sorted(
            values(benchmark_source_ids, "normalized_url")
            & values(train_source_ids, "normalized_url")
        ),
        "content_sha256_overlap": sorted(
            values(benchmark_source_ids, "content_sha256")
            & values(train_source_ids, "content_sha256")
        ),
    }


def validate_rows(
    sources: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    benchmark_items: list[dict[str, Any]],
    train_examples: list[dict[str, Any]],
) -> dict[str, Any]:
    source_map = keyed(sources, "source_id", "sources")
    issue_map = keyed(issues, "issue_id", "issues")
    item_map = keyed(benchmark_items, "item_id", "benchmark_items")

    alias_owner: dict[str, str] = {}
    for source in sources:
        source_id = source["source_id"]
        text = str(source.get("content_text") or "")
        expected_hash = sha256_text(text) if text else ""
        if source.get("content_sha256") != expected_hash:
            raise CatalogError(f"stored source text hash mismatch: {source_id}")
        if source.get("content_char_count") != len(text):
            raise CatalogError(f"stored source text length mismatch: {source_id}")
        if source.get("normalized_url") != normalize_url(str(source.get("url") or "")):
            raise CatalogError(f"normalized URL mismatch: {source_id}")
        expected_generation_eligible = bool(
            len(text) >= 500
            and source.get("content_scope") in GENERATION_ELIGIBLE_CONTENT_SCOPES
        )
        if source.get("generation_eligible") is not expected_generation_eligible:
            raise CatalogError(f"source generation gate mismatch: {source_id}")
        if source.get("needs_body_backfill") is not (not expected_generation_eligible):
            raise CatalogError(f"source backfill flag mismatch: {source_id}")
        for alias in source.get("aliases") or []:
            if alias in source_map and alias != source_id:
                raise CatalogError(f"source alias collides with canonical ID: {alias}")
            existing = alias_owner.get(alias)
            if existing is not None and existing != source_id:
                raise CatalogError(
                    f"source alias appears under multiple sources: {alias}: "
                    f"{existing}, {source_id}"
                )
            alias_owner[alias] = source_id

    training_ids: set[str] = set()
    missing_sources: set[str] = set()
    missing_issues: set[str] = set()
    for issue in issues:
        missing_sources.update(set(issue.get("source_ids") or []) - set(source_map))
    for item in benchmark_items:
        item_issue = issue_map.get(item.get("issue_id"))
        if item_issue is None:
            missing_issues.add(str(item.get("issue_id")))
        missing_sources.update(set(item.get("source_ids") or []) - set(source_map))
        selected = item.get("lifecycle_status") == "selected_benchmark"
        if selected != bool(item.get("scoring_contract")):
            raise CatalogError(f"item scoring-contract membership mismatch: {item['item_id']}")
        if selected and (
            item_issue is None
            or item_issue.get("lifecycle_status") != "benchmark_selected"
        ):
            raise CatalogError(f"selected item belongs to a non-benchmark issue: {item['item_id']}")
    for example in train_examples:
        metadata = example.get("metadata") or {}
        training_id = str(metadata.get("training_id") or "")
        if not training_id:
            raise CatalogError("train example has no metadata.training_id")
        if training_id in training_ids:
            raise CatalogError(f"duplicate training_id: {training_id}")
        training_ids.add(training_id)
        if metadata.get("issue_id") not in issue_map:
            missing_issues.add(str(metadata.get("issue_id")))
        elif issue_map[metadata["issue_id"]].get("lifecycle_status") != "train_selected":
            raise CatalogError(f"training example belongs to a non-training issue: {training_id}")
        missing_sources.update(set(metadata.get("source_ids") or []) - set(source_map))
    if missing_sources:
        raise CatalogError(f"missing source references: {sorted(missing_sources)}")
    if missing_issues:
        raise CatalogError(f"missing issue references: {sorted(missing_issues)}")

    benchmark_issues = {
        row["issue_id"] for row in issues if row["lifecycle_status"] == "benchmark_selected"
    }
    train_issues = {
        row["issue_id"] for row in issues if row["lifecycle_status"] == "train_selected"
    }
    if benchmark_issues & train_issues:
        raise CatalogError("benchmark and train issues overlap")
    for issue in issues:
        selected = issue.get("lifecycle_status") == "benchmark_selected"
        if selected != bool(issue.get("scoring_contract")):
            raise CatalogError(f"issue scoring-contract membership mismatch: {issue['issue_id']}")

        expected_item_ids = sorted(
            item["item_id"]
            for item in benchmark_items
            if item.get("issue_id") == issue["issue_id"]
            and item.get("lifecycle_status") == "selected_benchmark"
        )
        if issue.get("benchmark_item_ids") != expected_item_ids:
            raise CatalogError(f"issue benchmark item index mismatch: {issue['issue_id']}")
        expected_training_ids = sorted(
            example["metadata"]["training_id"]
            for example in train_examples
            if example["metadata"].get("issue_id") == issue["issue_id"]
        )
        if issue.get("training_example_ids") != expected_training_ids:
            raise CatalogError(f"issue training example index mismatch: {issue['issue_id']}")

    leakage = selected_source_sets(sources, benchmark_items, train_examples)
    for field in ("source_id_overlap", "normalized_url_overlap", "content_sha256_overlap"):
        if leakage[field]:
            raise CatalogError(f"benchmark/train source leakage in {field}: {leakage[field]}")

    item_status = Counter(row["lifecycle_status"] for row in benchmark_items)
    issue_status = Counter(row["lifecycle_status"] for row in issues)
    membership_counts = Counter(
        membership
        for row in train_examples
        for membership in row.get("experiment_memberships") or []
    )
    content_scope = Counter(row["content_scope"] for row in sources)
    retrieval_status = Counter(row["retrieval_status"] for row in sources)
    body_backfill = sum(bool(row.get("needs_body_backfill")) for row in sources)
    unreviewed_issues = sum(row.get("review_status") == "not_reviewed" for row in issues)

    return {
        "counts": {
            "sources": len(source_map),
            "issues": len(issue_map),
            "benchmark_items": len(item_map),
            "train_examples": len(training_ids),
        },
        "issue_lifecycle": dict(sorted(issue_status.items())),
        "benchmark_item_lifecycle": dict(sorted(item_status.items())),
        "train_experiment_membership": dict(sorted(membership_counts.items())),
        "source_content_scope": dict(sorted(content_scope.items())),
        "source_retrieval_status": dict(sorted(retrieval_status.items())),
        "sources_needing_body_backfill": body_backfill,
        "unreviewed_reserve_issues": unreviewed_issues,
        "benchmark_train_separation": leakage,
        "checks": {
            "unique_ids": True,
            "all_references_resolve": True,
            "stored_source_content_integrity": True,
            "scoring_contract_membership": True,
            "issue_child_indexes": True,
            "benchmark_train_issue_disjoint": True,
            "benchmark_train_source_disjoint": True,
        },
    }


def build_manifest(
    output_dir: Path,
    loaded: dict[str, list[dict[str, Any]]],
    validation: dict[str, Any],
) -> dict[str, Any]:
    inputs = {
        name: {
            "path": path.as_posix(),
            "rows": len(loaded[name]),
            "sha256": sha256_file(ROOT / path),
        }
        for name, path in sorted(INPUTS.items())
    }
    outputs = {
        name: {
            "rows": sum(1 for line in (output_dir / name).open("r", encoding="utf-8") if line.strip()),
            "sha256": sha256_file(output_dir / name),
        }
        for name in OUTPUT_FILES
    }
    return {
        "schema_version": "ksova-data-manifest-1.0",
        "catalog": "K-SOVA v2 current",
        "role": "canonical mutable data catalog",
        "downstream_boundaries": {
            "benchmark": "fixed evaluation releases are exported to benchmark/v2.0/releases",
            "train": "fixed training datasets and experiment artifacts are exported to train",
        },
        "migration_inputs": inputs,
        "outputs": outputs,
        "release_reference": "benchmark/v2.0/releases/first_pass_20260810_v1",
        "validation": validation,
    }


def updated_manifest(
    output_dir: Path,
    previous: dict[str, Any],
    validation: dict[str, Any],
    entry: dict[str, Any],
    history_key: str = "expansion_history",
) -> dict[str, Any]:
    manifest = copy.deepcopy(previous)
    manifest["outputs"] = {
        name: {
            "rows": sum(
                1
                for line in (output_dir / name).open("r", encoding="utf-8")
                if line.strip()
            ),
            "sha256": sha256_file(output_dir / name),
        }
        for name in OUTPUT_FILES
    }
    history = list(manifest.get(history_key) or [])
    if any(row.get("run_id") == entry["run_id"] for row in history):
        raise CatalogError(f"{history_key} run_id already integrated: {entry['run_id']}")
    history.append(entry)
    manifest[history_key] = history
    manifest["validation"] = validation
    return manifest


def load_inputs() -> dict[str, list[dict[str, Any]]]:
    return {name: read_jsonl(ROOT / path) for name, path in INPUTS.items()}


def validate_catalog(directory: Path) -> dict[str, Any]:
    for name in (*OUTPUT_FILES, "manifest.json"):
        if not (directory / name).is_file():
            raise CatalogError(f"catalog file is missing: {directory / name}")
    sources = read_jsonl(directory / "sources.jsonl")
    issues = read_jsonl(directory / "issues.jsonl")
    benchmark_items = read_jsonl(directory / "benchmark_items.jsonl")
    train_examples = read_jsonl(directory / "train_examples.jsonl")
    validation = validate_rows(sources, issues, benchmark_items, train_examples)

    with (directory / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for name in OUTPUT_FILES:
        expected = manifest.get("outputs", {}).get(name) or {}
        actual_hash = sha256_file(directory / name)
        actual_rows = sum(
            1 for line in (directory / name).open("r", encoding="utf-8") if line.strip()
        )
        if expected.get("sha256") != actual_hash:
            raise CatalogError(f"hash mismatch: {name}")
        if expected.get("rows") != actual_rows:
            raise CatalogError(f"row count mismatch: {name}")
    if manifest.get("validation") != validation:
        raise CatalogError("manifest validation summary does not match catalog contents")
    return validation


def safe_remove(directory: Path) -> None:
    if directory.parent != DATA_ROOT or directory.name not in {".staging", ".previous"}:
        raise CatalogError(f"refusing to remove unexpected path: {directory}")
    if directory.exists():
        shutil.rmtree(directory)


def promote(staging: Path) -> None:
    safe_remove(PREVIOUS)
    moved_current = False
    try:
        if CURRENT.exists():
            os.replace(CURRENT, PREVIOUS)
            moved_current = True
        os.replace(staging, CURRENT)
    except Exception:
        if moved_current and not CURRENT.exists() and PREVIOUS.exists():
            os.replace(PREVIOUS, CURRENT)
        raise
    safe_remove(PREVIOUS)


def build_catalog(*, replace: bool = False) -> dict[str, Any]:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if CURRENT.exists() and not replace:
        raise CatalogError(
            "data/current already exists; validate it instead, or use build --replace "
            "only for an intentional legacy re-migration"
        )
    safe_remove(STAGING)
    STAGING.mkdir()
    try:
        loaded = load_inputs()
        referenced_source_ids = collect_referenced_source_ids(loaded)
        sources, aliases = consolidate_sources(loaded, referenced_source_ids)
        benchmark_items = merge_benchmark_items(loaded, aliases)
        train_examples = merge_train_examples(loaded, aliases)
        issues = merge_issues(loaded, benchmark_items, train_examples, aliases)
        attach_source_usage(sources, issues, benchmark_items, train_examples)

        sources.sort(key=lambda row: row["source_id"])
        issues.sort(key=lambda row: row["issue_id"])
        benchmark_items.sort(key=lambda row: row["item_id"])
        train_examples.sort(key=lambda row: row["metadata"]["training_id"])

        validation = validate_rows(sources, issues, benchmark_items, train_examples)
        write_jsonl(STAGING / "sources.jsonl", sources)
        write_jsonl(STAGING / "issues.jsonl", issues)
        write_jsonl(STAGING / "benchmark_items.jsonl", benchmark_items)
        write_jsonl(STAGING / "train_examples.jsonl", train_examples)
        manifest = build_manifest(STAGING, loaded, validation)
        write_json(STAGING / "manifest.json", manifest)
        validate_catalog(STAGING)
        promote(STAGING)
        return validate_catalog(CURRENT)
    except Exception:
        safe_remove(STAGING)
        raise


def integrate_expansion(
    source_path: Path,
    issue_path: Path,
    run_id: str,
) -> dict[str, Any]:
    if not CURRENT.is_dir():
        raise CatalogError("data/current does not exist")
    validate_catalog(CURRENT)
    new_sources = read_jsonl(source_path)
    new_issues = read_jsonl(issue_path)
    if not new_sources and not new_issues:
        raise CatalogError("expansion contains neither sources nor issues")

    sources = read_jsonl(CURRENT / "sources.jsonl")
    issues = read_jsonl(CURRENT / "issues.jsonl")
    benchmark_items = read_jsonl(CURRENT / "benchmark_items.jsonl")
    train_examples = read_jsonl(CURRENT / "train_examples.jsonl")
    with (CURRENT / "manifest.json").open("r", encoding="utf-8") as handle:
        previous_manifest = json.load(handle)

    source_by_id = keyed(sources, "source_id", "sources")
    source_by_url = {
        row["normalized_url"]: row
        for row in sources
        if row.get("normalized_url")
    }
    source_by_content = {
        row["content_sha256"]: row
        for row in sources
        if row.get("content_sha256") and row.get("content_text")
    }
    incoming_aliases: dict[str, str] = {}
    added_source_ids: list[str] = []
    reused_source_ids: list[str] = []
    for incoming in new_sources:
        row = copy.deepcopy(incoming)
        source_id = str(row.get("source_id") or "")
        if not source_id:
            raise CatalogError("expansion source has no source_id")
        url = str(row.get("url") or "")
        normalized = normalize_url(url)
        if row.get("normalized_url") != normalized:
            raise CatalogError(f"expansion source normalized URL mismatch: {source_id}")
        existing = source_by_id.get(source_id)
        if existing is not None:
            if (
                existing.get("normalized_url") != normalized
                or (
                    existing.get("content_sha256")
                    and row.get("content_sha256")
                    and existing.get("content_sha256") != row.get("content_sha256")
                )
            ):
                raise CatalogError(f"expansion source ID collision: {source_id}")
            incoming_aliases[source_id] = source_id
            reused_source_ids.append(source_id)
            continue
        same_url = source_by_url.get(normalized) if normalized else None
        if same_url is not None:
            incoming_aliases[source_id] = same_url["source_id"]
            reused_source_ids.append(same_url["source_id"])
            continue
        content_hash = str(row.get("content_sha256") or "")
        same_content = source_by_content.get(content_hash) if content_hash else None
        if same_content is not None:
            incoming_aliases[source_id] = same_content["source_id"]
            reused_source_ids.append(same_content["source_id"])
            continue
        row.setdefault("schema_version", "ksova-data-source-1.0")
        row.setdefault("aliases", [])
        row.setdefault("usage", {
            "pools": [],
            "issue_ids": [],
            "benchmark_item_count": 0,
            "train_example_count": 0,
        })
        source_by_id[source_id] = row
        if normalized:
            source_by_url[normalized] = row
        if content_hash:
            source_by_content[content_hash] = row
        incoming_aliases[source_id] = source_id
        added_source_ids.append(source_id)

    issue_by_id = keyed(issues, "issue_id", "issues")
    existing_issue_identity = {
        re.sub(
            r"[^0-9a-z가-힣]",
            "",
            (
                str(row.get("title_ko") or "")
                + " "
                + str(row.get("decision_core_ko") or "")
            ).casefold(),
        ): row["issue_id"]
        for row in issues
    }

    def remap_source_references(value: Any) -> Any:
        """Map source IDs throughout the issue card and its audit payload."""
        if isinstance(value, str):
            return incoming_aliases.get(value, value)
        if isinstance(value, list):
            return [remap_source_references(item) for item in value]
        if isinstance(value, dict):
            return {
                key: remap_source_references(item)
                for key, item in value.items()
            }
        return value

    added_issue_ids: list[str] = []
    for incoming in new_issues:
        row = remap_source_references(copy.deepcopy(incoming))
        issue_id = str(row.get("issue_id") or "")
        if not issue_id:
            raise CatalogError("expansion issue has no issue_id")
        if issue_id in issue_by_id:
            raise CatalogError(f"expansion issue ID collision: {issue_id}")
        if row.get("lifecycle_status") != "reserve":
            raise CatalogError(f"new issue must enter as reserve: {issue_id}")
        if row.get("scoring_contract") is not None:
            raise CatalogError(f"reserve issue cannot have a scoring contract: {issue_id}")
        if any(row.get(field) for field in (
            "benchmark_item_ids", "candidate_item_ids", "training_example_ids"
        )):
            raise CatalogError(f"new reserve issue already has child records: {issue_id}")
        remapped_sources = []
        for source_id in row.get("source_ids") or []:
            if source_id not in source_by_id:
                raise CatalogError(
                    f"expansion issue references missing source: {issue_id}: {source_id}"
                )
            remapped_sources.append(source_id)
        row["source_ids"] = unique(remapped_sources)
        if len(row["source_ids"]) < 2:
            raise CatalogError(f"expansion issue has fewer than two sources: {issue_id}")
        identity = re.sub(
            r"[^0-9a-z가-힣]",
            "",
            (
                str(row.get("title_ko") or "")
                + " "
                + str(row.get("decision_core_ko") or "")
            ).casefold(),
        )
        if not identity:
            raise CatalogError(f"expansion issue has no textual identity: {issue_id}")
        if identity in existing_issue_identity:
            raise CatalogError(
                f"exact duplicate issue identity: {issue_id} and "
                f"{existing_issue_identity[identity]}"
            )
        existing_issue_identity[identity] = issue_id
        issue_by_id[issue_id] = row
        added_issue_ids.append(issue_id)

    sources = sorted(source_by_id.values(), key=lambda row: row["source_id"])
    issues = sorted(issue_by_id.values(), key=lambda row: row["issue_id"])
    attach_source_usage(sources, issues, benchmark_items, train_examples)
    validation = validate_rows(sources, issues, benchmark_items, train_examples)

    safe_remove(STAGING)
    STAGING.mkdir()
    try:
        write_jsonl(STAGING / "sources.jsonl", sources)
        write_jsonl(STAGING / "issues.jsonl", issues)
        write_jsonl(STAGING / "benchmark_items.jsonl", benchmark_items)
        write_jsonl(STAGING / "train_examples.jsonl", train_examples)
        expansion = {
            "run_id": run_id,
            "source_input": {
                "path": (
                    source_path.resolve().relative_to(ROOT.resolve()).as_posix()
                    if source_path.resolve().is_relative_to(ROOT.resolve())
                    else source_path.name
                ),
                "rows": len(new_sources),
                "sha256": sha256_file(source_path),
            },
            "issue_input": {
                "path": (
                    issue_path.resolve().relative_to(ROOT.resolve()).as_posix()
                    if issue_path.resolve().is_relative_to(ROOT.resolve())
                    else issue_path.name
                ),
                "rows": len(new_issues),
                "sha256": sha256_file(issue_path),
            },
            "added_sources": len(added_source_ids),
            "reused_sources": len(set(reused_source_ids)),
            "added_issues": len(added_issue_ids),
            "added_source_ids": sorted(added_source_ids),
            "added_issue_ids": sorted(added_issue_ids),
        }
        manifest = updated_manifest(STAGING, previous_manifest, validation, expansion)
        write_json(STAGING / "manifest.json", manifest)
        validate_catalog(STAGING)
        promote(STAGING)
    except Exception:
        safe_remove(STAGING)
        raise
    return {
        "run_id": run_id,
        "added_sources": len(added_source_ids),
        "reused_sources": len(set(reused_source_ids)),
        "added_issues": len(added_issue_ids),
        "catalog_validation": validate_catalog(CURRENT),
    }


def integrate_training_examples(
    examples_path: Path,
    run_id: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Atomically add fully validated training examples to the canonical catalog."""
    if not CURRENT.is_dir():
        raise CatalogError("data/current does not exist")
    validate_catalog(CURRENT)
    incoming_examples = read_jsonl(examples_path)
    if not incoming_examples:
        raise CatalogError("training promotion contains no examples")

    sources = read_jsonl(CURRENT / "sources.jsonl")
    issues = read_jsonl(CURRENT / "issues.jsonl")
    benchmark_items = read_jsonl(CURRENT / "benchmark_items.jsonl")
    train_examples = read_jsonl(CURRENT / "train_examples.jsonl")
    with (CURRENT / "manifest.json").open("r", encoding="utf-8") as handle:
        previous_manifest = json.load(handle)

    for history_key in ("expansion_history", "training_history"):
        if any(
            row.get("run_id") == run_id
            for row in (previous_manifest.get(history_key) or [])
        ):
            raise CatalogError(f"run_id already integrated in {history_key}: {run_id}")

    source_map = keyed(sources, "source_id", "sources")
    issue_map = keyed(issues, "issue_id", "issues")
    existing_by_id = {
        str(row.get("metadata", {}).get("training_id") or ""): row
        for row in train_examples
    }
    if "" in existing_by_id:
        raise CatalogError("existing training example has no training_id")

    def message_pair(row: dict[str, Any]) -> tuple[str, str]:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise CatalogError("training example must contain one user and one assistant message")
        if [message.get("role") for message in messages] != ["user", "assistant"]:
            raise CatalogError("training example message roles must be user then assistant")
        values = tuple(str(message.get("content") or "").strip() for message in messages)
        if not all(values):
            raise CatalogError("training example contains an empty message")
        return values  # type: ignore[return-value]

    existing_pairs = {message_pair(row) for row in train_examples}
    incoming_ids: set[str] = set()
    incoming_pairs: set[tuple[str, str]] = set()
    touched_issue_ids: set[str] = set()
    normalized_incoming: list[dict[str, Any]] = []
    for incoming in incoming_examples:
        row = copy.deepcopy(incoming)
        if row.get("schema_version") != "ksova-data-train-example-1.0":
            raise CatalogError("incoming training example has an unsupported schema")
        memberships = row.get("experiment_memberships")
        if (
            not isinstance(memberships, list)
            or not memberships
            or any(not isinstance(value, str) or not value for value in memberships)
            or len(memberships) != len(set(memberships))
        ):
            raise CatalogError("incoming training example has invalid experiment memberships")
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            raise CatalogError("incoming training example has no metadata")
        training_id = str(metadata.get("training_id") or "")
        issue_id = str(metadata.get("issue_id") or "")
        if not training_id or training_id in existing_by_id or training_id in incoming_ids:
            raise CatalogError(f"duplicate or empty incoming training_id: {training_id}")
        issue = issue_map.get(issue_id)
        if issue is None:
            raise CatalogError(f"incoming training issue does not exist: {issue_id}")
        if issue.get("lifecycle_status") not in {"reserve", "train_selected"}:
            raise CatalogError(f"incoming training issue is not eligible: {issue_id}")
        if metadata.get("category_id") != issue.get("domain_id"):
            raise CatalogError(f"incoming training domain mismatch: {training_id}")
        if metadata.get("split") not in {"train", "dev"}:
            raise CatalogError(f"incoming training split is invalid: {training_id}")
        if metadata.get("source_isolated_from_core250") is not True:
            raise CatalogError(f"incoming example lacks source-isolation gate: {training_id}")
        source_ids = metadata.get("source_ids")
        if not isinstance(source_ids, list) or not source_ids:
            raise CatalogError(f"incoming training example has no sources: {training_id}")
        if len(source_ids) != len(set(source_ids)):
            raise CatalogError(f"incoming training example has duplicate sources: {training_id}")
        if not set(source_ids) <= set(source_map):
            raise CatalogError(f"incoming training example references missing sources: {training_id}")
        if not set(source_ids) <= set(issue.get("source_ids") or []):
            raise CatalogError(f"incoming sources are not registered on issue: {training_id}")
        validation = metadata.get("validation")
        checks = validation.get("checks") if isinstance(validation, dict) else None
        if (
            not isinstance(validation, dict)
            or validation.get("verdict") != "pass"
            or not isinstance(checks, dict)
            or not checks
            or not all(value is True for value in checks.values())
            or validation.get("deterministic_failures") not in (None, [])
        ):
            raise CatalogError(f"incoming training example did not pass validation: {training_id}")
        pair = message_pair(row)
        if pair in existing_pairs or pair in incoming_pairs:
            raise CatalogError(f"duplicate incoming training message pair: {training_id}")
        incoming_ids.add(training_id)
        incoming_pairs.add(pair)
        touched_issue_ids.add(issue_id)
        normalized_incoming.append(row)

    activated_issue_ids = {
        issue_id
        for issue_id in touched_issue_ids
        if issue_map[issue_id].get("lifecycle_status") == "reserve"
    }
    for issue_id in touched_issue_ids:
        issue_map[issue_id]["lifecycle_status"] = "train_selected"
    train_examples.extend(normalized_incoming)
    train_examples.sort(key=lambda row: row["metadata"]["training_id"])

    training_ids_by_issue: dict[str, list[str]] = defaultdict(list)
    for example in train_examples:
        metadata = example["metadata"]
        training_ids_by_issue[str(metadata["issue_id"])].append(
            str(metadata["training_id"])
        )
    for issue in issues:
        issue["training_example_ids"] = sorted(
            training_ids_by_issue.get(str(issue["issue_id"]), [])
        )

    sources.sort(key=lambda row: row["source_id"])
    issues.sort(key=lambda row: row["issue_id"])
    attach_source_usage(sources, issues, benchmark_items, train_examples)
    validation_summary = validate_rows(sources, issues, benchmark_items, train_examples)

    safe_remove(STAGING)
    STAGING.mkdir()
    try:
        write_jsonl(STAGING / "sources.jsonl", sources)
        write_jsonl(STAGING / "issues.jsonl", issues)
        write_jsonl(STAGING / "benchmark_items.jsonl", benchmark_items)
        write_jsonl(STAGING / "train_examples.jsonl", train_examples)
        membership_counts = Counter(
            membership
            for row in normalized_incoming
            for membership in row.get("experiment_memberships") or []
        )
        promotion = {
            "run_id": run_id,
            "training_input": {
                "path": (
                    examples_path.resolve().relative_to(ROOT.resolve()).as_posix()
                    if examples_path.resolve().is_relative_to(ROOT.resolve())
                    else examples_path.name
                ),
                "rows": len(normalized_incoming),
                "sha256": sha256_file(examples_path),
            },
            "added_examples": len(normalized_incoming),
            "touched_issues": len(touched_issue_ids),
            "activated_reserve_issues": len(activated_issue_ids),
            "activated_issue_ids": sorted(activated_issue_ids),
            "added_training_ids": sorted(incoming_ids),
            "experiment_memberships": dict(sorted(membership_counts.items())),
        }
        manifest = updated_manifest(
            STAGING,
            previous_manifest,
            validation_summary,
            promotion,
            history_key="training_history",
        )
        write_json(STAGING / "manifest.json", manifest)
        staged_validation = validate_catalog(STAGING)
        if dry_run:
            safe_remove(STAGING)
        else:
            promote(STAGING)
    except Exception:
        safe_remove(STAGING)
        raise

    return {
        "run_id": run_id,
        "dry_run": dry_run,
        "promoted": not dry_run,
        "added_examples": len(normalized_incoming),
        "touched_issues": len(touched_issue_ids),
        "activated_reserve_issues": len(activated_issue_ids),
        "experiment_memberships": dict(sorted(membership_counts.items())),
        "catalog_validation": (
            staged_validation if dry_run else validate_catalog(CURRENT)
        ),
    }


def export_training_release(
    release_id: str,
    standard_memberships: list[str],
    expanded_memberships: list[str],
    standard_output: Path,
    expanded_output: Path,
    report_path: Path,
    expected_standard: int,
    expected_expanded: int,
) -> dict[str, Any]:
    """Export two immutable training conditions from canonical memberships."""
    validation = validate_catalog(CURRENT)
    if not release_id.strip():
        raise CatalogError("training release_id must not be empty")
    if not standard_memberships or not expanded_memberships:
        raise CatalogError("both training conditions require memberships")
    if len(standard_memberships) != len(set(standard_memberships)):
        raise CatalogError("duplicate Standard membership")
    if len(expanded_memberships) != len(set(expanded_memberships)):
        raise CatalogError("duplicate Expanded membership")
    for path in (standard_output, expanded_output, report_path):
        if not path.resolve().is_relative_to(ROOT.resolve()):
            raise CatalogError(f"training release output must remain under repository: {path}")
    if standard_output == expanded_output:
        raise CatalogError("Standard and Expanded outputs must differ")
    if standard_output.exists() or expanded_output.exists() or report_path.exists():
        raise CatalogError("training release target already exists")

    examples = read_jsonl(CURRENT / "train_examples.jsonl")

    def selected(memberships: list[str]) -> list[dict[str, Any]]:
        wanted = set(memberships)
        rows = [
            copy.deepcopy(row)
            for row in examples
            if wanted & set(row.get("experiment_memberships") or [])
        ]
        rows.sort(key=lambda row: str(row["metadata"]["training_id"]))
        return rows

    standard = selected(standard_memberships)
    expanded = selected(expanded_memberships)
    if len(standard) != expected_standard:
        raise CatalogError(
            f"unexpected Standard count: {len(standard)} != {expected_standard}"
        )
    if len(expanded) != expected_expanded:
        raise CatalogError(
            f"unexpected Expanded count: {len(expanded)} != {expected_expanded}"
        )
    standard_ids = {str(row["metadata"]["training_id"]) for row in standard}
    expanded_ids = {str(row["metadata"]["training_id"]) for row in expanded}
    if len(standard_ids) != len(standard) or len(expanded_ids) != len(expanded):
        raise CatalogError("training release contains duplicate training IDs")
    if not standard_ids <= expanded_ids:
        raise CatalogError("Standard is not an exact subset of Expanded")

    def audit_condition(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
        pair_ids: dict[tuple[str, str], str] = {}
        issue_splits: dict[str, str] = {}
        for row in rows:
            messages = row.get("messages") or []
            pair = (
                str(messages[0].get("content") or "").strip(),
                str(messages[1].get("content") or "").strip(),
            )
            training_id = str(row["metadata"]["training_id"])
            if pair in pair_ids:
                raise CatalogError(
                    f"{label} has duplicate message pairs: {training_id}, {pair_ids[pair]}"
                )
            pair_ids[pair] = training_id
            issue_id = str(row["metadata"]["issue_id"])
            split = str(row["metadata"].get("split") or "")
            if split not in {"train", "dev"}:
                raise CatalogError(f"{label} has invalid split: {training_id}")
            if issue_id in issue_splits and issue_splits[issue_id] != split:
                raise CatalogError(f"{label} splits one issue across train/dev: {issue_id}")
            issue_splits[issue_id] = split
        return {
            "examples": len(rows),
            "issues": len(issue_splits),
            "train_examples": sum(row["metadata"]["split"] == "train" for row in rows),
            "dev_examples": sum(row["metadata"]["split"] == "dev" for row in rows),
            "train_issues": sum(value == "train" for value in issue_splits.values()),
            "dev_issues": sum(value == "dev" for value in issue_splits.values()),
            "phase_counts": dict(sorted(Counter(
                str(row["metadata"]["curriculum_phase"]) for row in rows
            ).items())),
            "task_counts": dict(sorted(Counter(
                str(row["metadata"]["task"]) for row in rows
            ).items())),
        }

    condition_audits = {
        "standard": audit_condition(standard, "Standard"),
        "expanded": audit_condition(expanded, "Expanded"),
    }
    phase_names = {
        "phase1": "phase1_foundation",
        "phase2": "phase2_application",
        "phase3": "phase3_synthesis",
    }

    def output_sets(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        outputs = {
            "all_examples.jsonl": rows,
            "train_all.jsonl": [row for row in rows if row["metadata"]["split"] == "train"],
            "dev_all.jsonl": [row for row in rows if row["metadata"]["split"] == "dev"],
        }
        for short_name, phase in phase_names.items():
            for split in ("train", "dev"):
                outputs[f"{short_name}_{split}.jsonl"] = [
                    row for row in rows
                    if row["metadata"]["split"] == split
                    and row["metadata"]["curriculum_phase"] == phase
                ]
        return outputs

    train_root = ROOT / "train"
    staging = train_root / f".catalog-export-{os.getpid()}"
    if staging.exists():
        raise CatalogError(f"training export staging already exists: {staging}")
    standard_stage = staging / "standard"
    expanded_stage = staging / "expanded"
    standard_stage.mkdir(parents=True)
    expanded_stage.mkdir()
    report: dict[str, Any]
    try:
        staged_outputs: dict[str, dict[str, dict[str, Any]]] = {}
        for label, directory, rows in (
            ("standard", standard_stage, standard),
            ("expanded", expanded_stage, expanded),
        ):
            files = output_sets(rows)
            for name, values in files.items():
                write_jsonl(directory / name, values)
            staged_outputs[label] = {
                name: {
                    "rows": len(values),
                    "sha256": sha256_file(directory / name),
                }
                for name, values in files.items()
            }
        report = {
            "schema_version": "ksova-training-release-1.0",
            "release_id": release_id,
            "catalog_train_examples_sha256": sha256_file(CURRENT / "train_examples.jsonl"),
            "catalog_validation": validation,
            "standard": {
                "memberships": standard_memberships,
                "output": standard_output.resolve().relative_to(ROOT.resolve()).as_posix(),
                **condition_audits["standard"],
                "files": staged_outputs["standard"],
            },
            "expanded": {
                "memberships": expanded_memberships,
                "output": expanded_output.resolve().relative_to(ROOT.resolve()).as_posix(),
                **condition_audits["expanded"],
                "files": staged_outputs["expanded"],
            },
            "checks": {
                "expected_counts": True,
                "unique_training_ids": True,
                "unique_message_pairs": True,
                "issue_disjoint_train_dev": True,
                "standard_subset_of_expanded": True,
            },
        }
        write_json(staging / "report.json", report)
        standard_output.parent.mkdir(parents=True, exist_ok=True)
        expanded_output.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(standard_stage, standard_output)
        try:
            os.replace(expanded_stage, expanded_output)
            os.replace(staging / "report.json", report_path)
        except Exception:
            if expanded_output.exists():
                os.replace(expanded_output, expanded_stage)
            if standard_output.exists():
                os.replace(standard_output, standard_stage)
            raise
        staging.rmdir()
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return report


def rollback_expansion(run_id: str, reason: str) -> dict[str, Any]:
    """Atomically withdraw one expansion while preserving its audit history.

    Only reserve issues introduced by the named run are removable. Sources
    introduced by that run are removed only when no surviving catalog record
    references them; otherwise they remain as shared evidence.
    """
    if not CURRENT.is_dir():
        raise CatalogError("data/current does not exist")
    validate_catalog(CURRENT)
    if not reason.strip():
        raise CatalogError("rollback reason must not be empty")

    sources = read_jsonl(CURRENT / "sources.jsonl")
    issues = read_jsonl(CURRENT / "issues.jsonl")
    benchmark_items = read_jsonl(CURRENT / "benchmark_items.jsonl")
    train_examples = read_jsonl(CURRENT / "train_examples.jsonl")
    with (CURRENT / "manifest.json").open("r", encoding="utf-8") as handle:
        previous_manifest = json.load(handle)

    history = list(previous_manifest.get("expansion_history") or [])
    matches = [row for row in history if row.get("run_id") == run_id]
    if len(matches) != 1:
        raise CatalogError(
            f"rollback requires exactly one expansion history entry: {run_id}"
        )
    expansion = matches[0]
    if expansion.get("status") == "rolled_back":
        raise CatalogError(f"expansion is already rolled back: {run_id}")

    target_issue_ids = set(expansion.get("added_issue_ids") or [])
    target_source_ids = set(expansion.get("added_source_ids") or [])
    issue_by_id = keyed(issues, "issue_id", "issues")
    missing_issue_ids = target_issue_ids - set(issue_by_id)
    if missing_issue_ids:
        raise CatalogError(
            f"rollback expansion issues are missing: {sorted(missing_issue_ids)}"
        )
    for issue_id in target_issue_ids:
        issue = issue_by_id[issue_id]
        if issue.get("lifecycle_status") != "reserve":
            raise CatalogError(f"rollback issue is not reserve: {issue_id}")
        if any(issue.get(field) for field in (
            "benchmark_item_ids", "candidate_item_ids", "training_example_ids"
        )):
            raise CatalogError(f"rollback issue has downstream children: {issue_id}")

    remaining_issues = [
        row for row in issues if row["issue_id"] not in target_issue_ids
    ]
    referenced_source_ids = {
        source_id
        for issue in remaining_issues
        for source_id in issue.get("source_ids") or []
    }
    referenced_source_ids.update(
        source_id
        for item in benchmark_items
        for source_id in item.get("source_ids") or []
    )
    referenced_source_ids.update(
        source_id
        for example in train_examples
        for source_id in (example.get("metadata") or {}).get("source_ids") or []
    )
    removable_source_ids = target_source_ids - referenced_source_ids
    retained_source_ids = target_source_ids & referenced_source_ids
    remaining_sources = [
        row for row in sources if row["source_id"] not in removable_source_ids
    ]

    remaining_sources.sort(key=lambda row: row["source_id"])
    remaining_issues.sort(key=lambda row: row["issue_id"])
    attach_source_usage(
        remaining_sources, remaining_issues, benchmark_items, train_examples
    )
    validation = validate_rows(
        remaining_sources, remaining_issues, benchmark_items, train_examples
    )

    safe_remove(STAGING)
    STAGING.mkdir()
    try:
        write_jsonl(STAGING / "sources.jsonl", remaining_sources)
        write_jsonl(STAGING / "issues.jsonl", remaining_issues)
        write_jsonl(STAGING / "benchmark_items.jsonl", benchmark_items)
        write_jsonl(STAGING / "train_examples.jsonl", train_examples)

        manifest = copy.deepcopy(previous_manifest)
        for row in manifest.get("expansion_history") or []:
            if row.get("run_id") == run_id:
                row["status"] = "rolled_back"
                row["rollback_reason"] = reason.strip()
                row["removed_issue_ids"] = sorted(target_issue_ids)
                row["removed_source_ids"] = sorted(removable_source_ids)
                row["retained_source_ids"] = sorted(retained_source_ids)
        manifest["outputs"] = {
            name: {
                "rows": sum(
                    1
                    for line in (STAGING / name).open("r", encoding="utf-8")
                    if line.strip()
                ),
                "sha256": sha256_file(STAGING / name),
            }
            for name in OUTPUT_FILES
        }
        manifest["validation"] = validation
        write_json(STAGING / "manifest.json", manifest)
        validate_catalog(STAGING)
        promote(STAGING)
    except Exception:
        safe_remove(STAGING)
        raise

    return {
        "run_id": run_id,
        "removed_issues": len(target_issue_ids),
        "removed_sources": len(removable_source_ids),
        "retained_shared_sources": len(retained_source_ids),
        "catalog_validation": validate_catalog(CURRENT),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser(
        "build", help="consolidate legacy v2 assets into data/current"
    )
    build_parser.add_argument(
        "--replace",
        action="store_true",
        help="intentionally replace an existing catalog after full staging validation",
    )
    validate_parser = subparsers.add_parser("validate", help="validate an existing catalog")
    validate_parser.add_argument(
        "path",
        nargs="?",
        default=str(CURRENT),
        help="catalog directory (default: data/current)",
    )
    integrate_parser = subparsers.add_parser(
        "integrate-expansion",
        help="atomically add verified expansion sources and reserve issues",
    )
    integrate_parser.add_argument("--sources", type=Path, required=True)
    integrate_parser.add_argument("--issues", type=Path, required=True)
    integrate_parser.add_argument("--run-id", required=True)
    training_parser = subparsers.add_parser(
        "integrate-training",
        help="atomically add validated examples and activate their training issues",
    )
    training_parser.add_argument("--examples", type=Path, required=True)
    training_parser.add_argument("--run-id", required=True)
    training_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="stage and validate the promotion without changing data/current",
    )
    export_parser = subparsers.add_parser(
        "export-training",
        help="export immutable Standard and Expanded datasets from catalog memberships",
    )
    export_parser.add_argument("--release-id", required=True)
    export_parser.add_argument(
        "--standard-membership", action="append", required=True
    )
    export_parser.add_argument(
        "--expanded-membership", action="append", required=True
    )
    export_parser.add_argument("--standard-output", type=Path, required=True)
    export_parser.add_argument("--expanded-output", type=Path, required=True)
    export_parser.add_argument("--report", type=Path, required=True)
    export_parser.add_argument("--expected-standard", type=int, required=True)
    export_parser.add_argument("--expected-expanded", type=int, required=True)
    rollback_parser = subparsers.add_parser(
        "rollback-expansion",
        help="atomically withdraw one integrated reserve expansion",
    )
    rollback_parser.add_argument("--run-id", required=True)
    rollback_parser.add_argument("--reason", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "build":
            report = build_catalog(replace=args.replace)
        elif args.command == "validate":
            directory = Path(args.path)
            if not directory.is_absolute():
                directory = ROOT / directory
            report = validate_catalog(directory)
        elif args.command == "integrate-expansion":
            source_path = args.sources if args.sources.is_absolute() else ROOT / args.sources
            issue_path = args.issues if args.issues.is_absolute() else ROOT / args.issues
            report = integrate_expansion(source_path, issue_path, args.run_id)
        elif args.command == "integrate-training":
            examples_path = (
                args.examples if args.examples.is_absolute() else ROOT / args.examples
            )
            report = integrate_training_examples(
                examples_path,
                args.run_id,
                dry_run=args.dry_run,
            )
        elif args.command == "export-training":
            def repository_path(path: Path) -> Path:
                return path if path.is_absolute() else ROOT / path

            report = export_training_release(
                args.release_id,
                args.standard_membership,
                args.expanded_membership,
                repository_path(args.standard_output),
                repository_path(args.expanded_output),
                repository_path(args.report),
                args.expected_standard,
                args.expected_expanded,
            )
        else:
            report = rollback_expansion(args.run_id, args.reason)
    except CatalogError as exc:
        print(f"catalog error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
