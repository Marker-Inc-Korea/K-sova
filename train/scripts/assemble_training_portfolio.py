#!/usr/bin/env python3
"""Assemble an audited, benchmark-isolated K-SOVA training portfolio.

The selection manifest is the only authority for swaps.  Canonical issues,
newly verified issues, and re-sourced canonical issues remain in their source
files; this script writes a compact experiment dossier without mutating any of
them.  Re-sourced issues use only the newly validated body sources recorded in
the bundle, never the issue card's older source IDs.
"""

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


EXPECTED_BENCHMARK_SHA256 = (
    "0ca49e7bc94178300c0309919d63a51c0496d84d6e352501d350f0deac494032"
)
LEGACY_DOMAIN_MAP = {
    "T-history_responsibility_memory": "D04",
    "T-north_korea_interkorean": "D06",
    "T-trade_industry_supply_chain": "D07",
}
DOMAINS = tuple(f"D{index:02d}" for index in range(1, 9))


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(value: Any) -> str:
    return " ".join(str(value or "").split())


def normalized_domain(value: Any) -> str:
    domain_id = str(value or "")
    return LEGACY_DOMAIN_MAP.get(domain_id, domain_id)


def normalized_url(row: dict[str, Any]) -> str:
    raw = normalized(row.get("normalized_url") or row.get("url"))
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, query, ""))


def source_body(row: dict[str, Any]) -> str:
    retrieval = row.get("retrieval") or {}
    return str(
        row.get("content_text")
        or (retrieval.get("text_excerpt") if isinstance(retrieval, dict) else "")
        or ""
    )


def content_hash(row: dict[str, Any]) -> str:
    declared = normalized(row.get("content_sha256"))
    if declared:
        return declared.lower()
    body = source_body(row)
    if not body:
        return ""
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def index_unique(
    rows: Iterable[dict[str, Any]], key: str, label: str,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = str(row.get(key) or "")
        if not value:
            raise ValueError(f"{label}: missing {key}")
        if value in output and output[value] != row:
            raise ValueError(f"{label}: conflicting duplicate {key}={value}")
        output[value] = row
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issues", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--recommended-train", type=Path, required=True)
    parser.add_argument("--benchmark-items", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--extra-issues", type=Path, action="append", default=[])
    parser.add_argument("--extra-sources", type=Path, action="append", default=[])
    parser.add_argument("--resourced-bundles", type=Path, required=True)
    parser.add_argument("--resourced-sources", type=Path, required=True)
    parser.add_argument("--resourcing-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    outputs = {
        "issues": args.output_dir / "selected_issues.jsonl",
        "sources": args.output_dir / "selected_sources.jsonl",
        "source_audit": args.output_dir / "source_isolation_audit.jsonl",
        "report": args.output_dir / "portfolio_report.json",
    }
    existing_outputs = [str(path) for path in outputs.values() if path.exists()]
    if existing_outputs:
        raise FileExistsError(f"refusing to overwrite outputs: {existing_outputs}")

    selection = read_json(args.selection)
    if selection.get("schema_version") != "ksova-training-portfolio-selection-1.0":
        raise ValueError("unexpected selection schema")
    expected = selection.get("expected") or {}
    drop_ids = [str(value) for value in selection.get("drop_issue_ids") or []]
    additions = selection.get("additions") or []
    if len(drop_ids) != len(set(drop_ids)):
        raise ValueError("duplicate drop issue ID")
    if not all(isinstance(row, dict) and row.get("issue_id") for row in additions):
        raise ValueError("invalid additions")
    add_ids = [str(row["issue_id"]) for row in additions]
    if len(add_ids) != len(set(add_ids)):
        raise ValueError("duplicate addition issue ID")
    if set(drop_ids) & set(add_ids):
        raise ValueError("same issue appears in drop and addition lists")
    if len(drop_ids) != int(expected.get("drop") or -1):
        raise ValueError("drop count does not match manifest expectation")
    if len(add_ids) != int(expected.get("add") or -1):
        raise ValueError("addition count does not match manifest expectation")

    canonical_rows = read_jsonl(args.issues)
    canonical_issues = index_unique(canonical_rows, "issue_id", "canonical issues")
    if len(canonical_issues) != 500:
        raise ValueError(f"expected 500 canonical issues, got {len(canonical_issues)}")
    old_ids = {
        issue_id for issue_id, row in canonical_issues.items()
        if row.get("lifecycle_status") == "train_selected"
    }
    if len(old_ids) != 71:
        raise ValueError(f"expected 71 train_selected issues, got {len(old_ids)}")

    recommended_rows = read_jsonl(args.recommended_train)
    recommended_ids = {
        str((row.get("metadata") or {}).get("issue_id") or "")
        for row in recommended_rows
    } - {""}
    if len(recommended_ids) != 79:
        raise ValueError(
            f"expected 79 recommended issue IDs, got {len(recommended_ids)}"
        )
    if old_ids & recommended_ids:
        raise ValueError("canonical and recommended base issue sets overlap")
    base_ids = old_ids | recommended_ids
    if len(base_ids) != 150:
        raise AssertionError("base portfolio is not 150 issues")
    if not set(drop_ids) <= base_ids:
        raise ValueError(f"drop IDs outside base: {sorted(set(drop_ids) - base_ids)}")
    if set(add_ids) & base_ids:
        raise ValueError(f"addition IDs already in base: {sorted(set(add_ids) & base_ids)}")

    extra_issue_rows = [
        row for path in args.extra_issues for row in read_jsonl(path)
    ]
    extra_issues = index_unique(extra_issue_rows, "issue_id", "extra issues")
    issue_lookup = dict(canonical_issues)
    for issue_id, row in extra_issues.items():
        if issue_id in issue_lookup and issue_lookup[issue_id] != row:
            raise ValueError(f"extra issue conflicts with canonical issue {issue_id}")
        issue_lookup[issue_id] = row
    missing_additions = sorted(set(add_ids) - set(issue_lookup))
    if missing_additions:
        raise ValueError(f"missing addition issue records: {missing_additions}")

    resourcing_audit = read_json(args.resourcing_audit)
    eligible_resourced = {
        str(row.get("issue_id") or "")
        for row in resourcing_audit.get("decisions") or []
        if row.get("decision") == "eligible_for_training_portfolio"
    }
    bundle_rows = read_jsonl(args.resourced_bundles)
    bundles = index_unique(bundle_rows, "issue_id", "resourced bundles")
    expected_resourced = {
        str(row["issue_id"]) for row in additions
        if row.get("origin") == "canonical_resourced"
    }
    if expected_resourced != eligible_resourced:
        raise ValueError(
            "manifest canonical_resourced IDs do not exactly match manual audit: "
            f"manifest_only={sorted(expected_resourced - eligible_resourced)}, "
            f"audit_only={sorted(eligible_resourced - expected_resourced)}"
        )
    missing_bundles = sorted(expected_resourced - set(bundles))
    if missing_bundles:
        raise ValueError(f"missing resourced bundles: {missing_bundles}")
    for issue_id in expected_resourced:
        bundle = bundles[issue_id]
        if (
            bundle.get("training_eligible") is not True
            or bundle.get("status") != "body_source_bundle_complete"
            or not bundle.get("source_ids")
        ):
            raise ValueError(f"resourced bundle is not eligible: {issue_id}")

    selected_ids = (base_ids - set(drop_ids)) | set(add_ids)
    if len(selected_ids) != int(expected.get("issues") or -1):
        raise ValueError(
            f"selected issue count mismatch: {len(selected_ids)} vs {expected.get('issues')}"
        )

    canonical_source_rows = read_jsonl(args.sources)
    source_lookup = index_unique(
        canonical_source_rows, "source_id", "canonical sources"
    )
    supplemental_source_rows = [
        row for path in args.extra_sources for row in read_jsonl(path)
    ]
    for resolved in read_jsonl(args.resourced_sources):
        # The trusted-source resolver preserves lead-level provenance around
        # the fetched source.  Only the nested source record belongs in the
        # unified source catalog; silently accepting another wrapper shape
        # would make source isolation checks ambiguous.
        source = resolved.get("source")
        if not isinstance(source, dict):
            raise ValueError("resourced source row is missing nested source object")
        supplemental_source_rows.append(source)
    for row in supplemental_source_rows:
        source_id = str(row.get("source_id") or "")
        if not source_id:
            raise ValueError("supplemental source missing source_id")
        prior = source_lookup.get(source_id)
        if prior is not None:
            same_url = normalized_url(prior) == normalized_url(row)
            if not same_url:
                raise ValueError(f"conflicting supplemental source {source_id}")
            # The re-sourcing resolver stores a validation excerpt, while the
            # canonical registry may already hold the full body for the same
            # stable URL-derived source ID.  Their hashes need not match. Keep
            # the longer body, but only after exact normalized-URL identity.
            if len(source_body(row)) > len(source_body(prior)):
                source_lookup[source_id] = row
        else:
            source_lookup[source_id] = row

    benchmark_hash = file_sha256(args.benchmark_items)
    if benchmark_hash != EXPECTED_BENCHMARK_SHA256:
        raise ValueError(f"unexpected benchmark hash: {benchmark_hash}")
    benchmark_rows = read_jsonl(args.benchmark_items)
    if len(benchmark_rows) != 480:
        raise ValueError(f"expected 480 benchmark items, got {len(benchmark_rows)}")
    benchmark_issue_ids = {str(row.get("issue_id") or "") for row in benchmark_rows}
    issue_collisions = sorted(selected_ids & benchmark_issue_ids)
    if issue_collisions:
        raise ValueError(f"benchmark issue collisions: {issue_collisions[:10]}")
    benchmark_source_ids = {
        str(source_id)
        for row in benchmark_rows
        for source_id in row.get("source_ids") or []
        if str(source_id)
    }
    missing_benchmark_sources = sorted(benchmark_source_ids - set(source_lookup))
    if missing_benchmark_sources:
        raise ValueError(
            f"missing benchmark source records: {missing_benchmark_sources[:10]}"
        )
    benchmark_urls = {
        normalized_url(source_lookup[source_id])
        for source_id in benchmark_source_ids
        if normalized_url(source_lookup[source_id])
    }
    benchmark_hashes = {
        content_hash(source_lookup[source_id])
        for source_id in benchmark_source_ids
        if content_hash(source_lookup[source_id])
    }

    addition_by_id = {str(row["issue_id"]): row for row in additions}
    dossiers = []
    source_audit_rows = []
    training_source_ids: set[str] = set()
    for issue_id in sorted(selected_ids):
        issue = copy.deepcopy(issue_lookup[issue_id])
        addition = addition_by_id.get(issue_id)
        if addition and addition.get("origin") == "canonical_resourced":
            source_ids = [str(value) for value in bundles[issue_id]["source_ids"]]
            source_basis = "new_body_source_bundle"
        else:
            source_ids = [str(value) for value in issue.get("source_ids") or []]
            source_basis = "issue_card_sources"
        source_ids = list(dict.fromkeys(value for value in source_ids if value))
        if not source_ids:
            raise ValueError(f"selected issue has no training sources: {issue_id}")
        missing = sorted(set(source_ids) - set(source_lookup))
        if missing:
            raise ValueError(f"{issue_id}: missing source records {missing}")
        urls = [normalized_url(source_lookup[value]) for value in source_ids]
        hashes = [content_hash(source_lookup[value]) for value in source_ids]
        url_collisions = sorted({value for value in urls if value in benchmark_urls})
        content_collisions = sorted({
            value for value in hashes if value and value in benchmark_hashes
        })
        source_id_collisions = sorted(set(source_ids) & benchmark_source_ids)
        if source_id_collisions or url_collisions or content_collisions:
            raise ValueError(
                f"{issue_id}: benchmark source collision "
                f"ids={source_id_collisions}, urls={url_collisions[:2]}, "
                f"content={content_collisions[:2]}"
            )
        domain_id = normalized_domain(issue.get("domain_id"))
        if domain_id not in DOMAINS:
            raise ValueError(f"{issue_id}: invalid normalized domain {domain_id}")
        declared_domain = str((addition or {}).get("domain_id") or domain_id)
        if declared_domain != domain_id:
            raise ValueError(
                f"{issue_id}: manifest domain {declared_domain} != issue domain {domain_id}"
            )
        origin = str((addition or {}).get("origin") or (
            "canonical_train_selected" if issue_id in old_ids
            else "validated_recommended_issue"
        ))
        dossiers.append({
            "schema_version": "ksova-training-portfolio-issue-1.0",
            "experiment_id": selection["experiment_id"],
            "issue_id": issue_id,
            "domain_id": domain_id,
            "origin": origin,
            "selection_reason_ko": str((addition or {}).get("reason_ko") or ""),
            "source_basis": source_basis,
            "source_ids": source_ids,
            "source_isolated_from_frozen_benchmark_480": True,
            "issue_card": issue,
        })
        training_source_ids.update(source_ids)
        source_audit_rows.append({
            "issue_id": issue_id,
            "domain_id": domain_id,
            "origin": origin,
            "source_basis": source_basis,
            "source_ids": source_ids,
            "normalized_urls": urls,
            "content_sha256": hashes,
            "benchmark_source_id_collisions": [],
            "benchmark_url_collisions": [],
            "benchmark_content_collisions": [],
        })

    by_domain = dict(sorted(Counter(row["domain_id"] for row in dossiers).items()))
    expected_by_domain = {
        str(key): int(value)
        for key, value in (expected.get("by_domain") or {}).items()
    }
    if by_domain != expected_by_domain:
        raise ValueError(
            f"domain distribution mismatch: actual={by_domain}, expected={expected_by_domain}"
        )
    by_origin = dict(sorted(Counter(row["origin"] for row in dossiers).items()))
    selected_sources = [
        source_lookup[source_id] for source_id in sorted(training_source_ids)
    ]
    report = {
        "schema_version": "ksova-training-portfolio-report-1.0",
        "experiment_id": selection["experiment_id"],
        "issues": len(dossiers),
        "sources": len(selected_sources),
        "base_issues": len(base_ids),
        "dropped_issues": len(drop_ids),
        "added_issues": len(add_ids),
        "by_domain": by_domain,
        "by_origin": by_origin,
        "benchmark": {
            "items": len(benchmark_rows),
            "sha256": benchmark_hash,
            "issue_collisions": 0,
            "source_id_collisions": 0,
            "normalized_url_collisions": 0,
            "content_hash_collisions": 0,
        },
        "input_sha256": {
            "selection": file_sha256(args.selection),
            "canonical_issues": file_sha256(args.issues),
            "canonical_sources": file_sha256(args.sources),
            "recommended_train": file_sha256(args.recommended_train),
            "resourced_bundles": file_sha256(args.resourced_bundles),
            "resourcing_audit": file_sha256(args.resourcing_audit),
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    write_jsonl(outputs["issues"], dossiers)
    write_jsonl(outputs["sources"], selected_sources)
    write_jsonl(outputs["source_audit"], source_audit_rows)
    write_json(outputs["report"], report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
