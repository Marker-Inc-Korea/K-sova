#!/usr/bin/env python3
"""Promote directly reviewed v2 source cards into the v2.1 source schema."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


AXIS_BY_CATEGORY = {
    "S01": "constitutional_state_order",
    "S02": "territory_jurisdiction",
    "S03": "history_memory_remedy",
    "S04": "culture_identity_representation",
    "S05": "citizenship_diaspora_human_rights",
    "S06": "security_deterrence_risk",
    "S07": "economic_industrial_security",
    "S08": "technology_data_science",
    "S09": "democratic_values_social_norms",
    "S10": "diplomacy_international_order",
}

ROLE_BY_AUTHORITY = {
    "academic": "academic",
    "constitution": "judicial",
    "judicial": "judicial",
    "official": "official",
    "official_government_statement": "official",
    "official_source": "official",
    "reputable_reporting": "reputable_reporting",
    "industry_technical": "industry_technical",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: expected object")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def unique_strings(values: list[Any], maximum: int) -> list[str]:
    result = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
        if len(result) == maximum:
            break
    return result


def source_role(authority_type: str) -> str:
    return ROLE_BY_AUTHORITY.get(authority_type, "institutional")


def promote(row: dict[str, Any]) -> dict[str, Any]:
    curation = row.get("curation") or {}
    category = str(curation.get("source_category") or "")
    excerpt = str(row.get("content_excerpt") or "").strip()
    if len(excerpt) < 80:
        excerpt = " ".join(
            part for part in (excerpt, str(row.get("evidence_summary") or "").strip()) if part
        )
    claims = unique_strings(
        [curation.get("issue_claim_ko"), *(row.get("evidence_claims") or [])],
        maximum=12,
    )
    authority_type = str(row.get("authority_type") or "")
    return {
        "source_id": row["source_id"],
        "title_ko": str(row.get("title") or "").strip(),
        "publisher": str(row.get("publisher") or authority_type or "출처 미상").strip(),
        "url": str(row.get("canonical_url") or "").strip(),
        "retrieved_at": str(row.get("retrieved_at") or "").strip(),
        "excerpt_ko": excerpt[:5000],
        "source_role": source_role(authority_type),
        "evidence_use": "factual_support",
        "supported_claims_ko": claims,
        "sovereign_axes": [AXIS_BY_CATEGORY[category]],
        "manually_read": True,
        "manual_review_scope": "title_url_excerpt_claim",
        "source_category": category,
        "evaluation_domain": curation.get("evaluation_domain"),
        "provenance": {
            "input_artifact": "matrix_v9_curated.jsonl",
            "input_disposition": curation.get("disposition"),
            "input_evidence_quality": curation.get("evidence_quality"),
            "promotion_rule": "explicit_reviewed_id_only",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--reviewed-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    requested = [
        line.strip()
        for line in args.reviewed_ids.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(requested) != len(set(requested)):
        raise ValueError("reviewed id list contains duplicates")

    indexed = {row.get("source_id"): row for row in read_jsonl(args.matrix)}
    missing = [source_id for source_id in requested if source_id not in indexed]
    if missing:
        raise ValueError(f"unknown reviewed source ids: {missing}")

    selected = [indexed[source_id] for source_id in requested]
    invalid = [
        row["source_id"]
        for row in selected
        if (row.get("curation") or {}).get("disposition") != "retain"
        or (row.get("curation") or {}).get("evidence_quality") != "direct"
    ]
    if invalid:
        raise ValueError(f"reviewed sources are not retained direct evidence: {invalid}")

    promoted = [promote(row) for row in selected]
    write_jsonl(args.output, promoted)
    report = {
        "input_rows": len(indexed),
        "reviewed_ids": len(requested),
        "promoted": len(promoted),
        "by_evaluation_domain": dict(sorted(Counter(row["evaluation_domain"] for row in promoted).items())),
        "by_source_role": dict(sorted(Counter(row["source_role"] for row in promoted).items())),
        "manual_review_scope": "title_url_excerpt_claim",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
