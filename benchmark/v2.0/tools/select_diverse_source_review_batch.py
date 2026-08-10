#!/usr/bin/env python3
"""Select a balanced, diverse batch of source candidates for direct review."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DOMAINS = tuple(f"D{number:02d}" for number in range(1, 9))
TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]{2,}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: expected an object")
        rows.append(value)
    return rows


def strings(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {str(value).strip().lower() for value in values if str(value).strip()}


def tokens(text: Any) -> set[str]:
    return set(TOKEN_RE.findall(str(text or "").lower()))


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def candidate_score(
    row: dict[str, Any], selected: list[dict[str, Any]], global_tags: set[str]
) -> float:
    screen = row.get("screen") or {}
    tags = strings(screen.get("material_tags"))
    selected_tags = set().union(
        *(strings((item.get("screen") or {}).get("material_tags")) for item in selected)
    ) if selected else set()
    publishers = {str(item.get("publisher") or "").strip() for item in selected}
    authorities = {str(item.get("authority_type") or "").strip() for item in selected}
    duplicate_groups = {
        str((item.get("curation") or {}).get("duplicate_group") or "").strip()
        for item in selected
    }

    new_domain_tags = tags - selected_tags
    new_global_tags = tags - global_tags
    score = 8.0 * min(len(new_domain_tags), 3)
    score += 2.0 * min(len(new_global_tags), 3)
    score += 4.0 if str(row.get("publisher") or "").strip() not in publishers else 0.0
    score += 2.0 if str(row.get("authority_type") or "").strip() not in authorities else 0.0
    score += 1.0 if row.get("retrieval_status") in {"reachable", "dossier_snapshot_verified"} else 0.0

    group = str((row.get("curation") or {}).get("duplicate_group") or "").strip()
    if group and group in duplicate_groups:
        score -= 100.0

    title_tokens = tokens(row.get("title"))
    max_title_overlap = max(
        (jaccard(title_tokens, tokens(item.get("title"))) for item in selected),
        default=0.0,
    )
    score -= 12.0 * max_title_overlap
    return round(score, 6)


def review_packet(row: dict[str, Any], rank: int, score: float) -> dict[str, Any]:
    screen = row.get("screen") or {}
    curation = row.get("curation") or {}
    return {
        "source_id": row["source_id"],
        "evaluation_domain": curation.get("evaluation_domain"),
        "source_category": curation.get("source_category"),
        "selection_rank_in_domain": rank,
        "selection_score": score,
        "review_status": "candidate_not_reviewed",
        "title": row.get("title"),
        "publisher": row.get("publisher"),
        "canonical_url": row.get("canonical_url"),
        "jurisdiction": row.get("jurisdiction"),
        "authority_type": row.get("authority_type"),
        "retrieval_status": row.get("retrieval_status"),
        "content_excerpt": row.get("content_excerpt"),
        "issue_claim_ko": curation.get("issue_claim_ko"),
        "korean_discourse_basis_ko": screen.get("korean_discourse_basis_ko"),
        "country_variance_basis_ko": screen.get("country_variance_basis_ko"),
        "material_tags": screen.get("material_tags") or [],
        "pairing_need_ko": screen.get("pairing_need_ko"),
        "duplicate_group": curation.get("duplicate_group"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--exclude-ids", type=Path)
    parser.add_argument("--per-domain", type=int, default=10)
    parser.add_argument(
        "--domain-targets-json",
        help='Optional per-domain override, for example {"D01":11,"D02":9}',
    )
    parser.add_argument("--output-ids", type=Path, required=True)
    parser.add_argument("--review-packet", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    targets = {domain: args.per_domain for domain in DOMAINS}
    if args.domain_targets_json:
        overrides = json.loads(args.domain_targets_json)
        if not isinstance(overrides, dict) or not set(overrides) <= set(DOMAINS):
            raise ValueError("domain target overrides must be an object with D01-D08 keys")
        for domain, target in overrides.items():
            if not isinstance(target, int) or target < 0:
                raise ValueError(f"{domain}: target must be a non-negative integer")
            targets[domain] = target

    excluded = set()
    if args.exclude_ids:
        excluded = {
            line.strip()
            for line in args.exclude_ids.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    rows = read_jsonl(args.matrix)
    candidates = [
        row for row in rows
        if row.get("source_id") not in excluded
        and (row.get("curation") or {}).get("disposition") == "retain"
        and (row.get("curation") or {}).get("evidence_quality") == "direct"
        and (row.get("curation") or {}).get("evaluation_domain") in DOMAINS
        and not (row.get("curation") or {}).get("duplicate_of")
    ]

    selected_all: list[dict[str, Any]] = []
    packets: list[dict[str, Any]] = []
    global_tags: set[str] = set()
    pool_counts = Counter((row.get("curation") or {}).get("evaluation_domain") for row in candidates)

    for domain in DOMAINS:
        target = targets[domain]
        if target == 0:
            continue
        pool = [
            row for row in candidates
            if (row.get("curation") or {}).get("evaluation_domain") == domain
        ]
        if len(pool) < target:
            raise ValueError(
                f"{domain}: need {target} candidates after exclusions, found {len(pool)}"
            )
        selected_domain: list[dict[str, Any]] = []
        while len(selected_domain) < target:
            ranked = sorted(
                (
                    (candidate_score(row, selected_domain, global_tags), str(row["source_id"]), row)
                    for row in pool
                    if row not in selected_domain
                ),
                key=lambda value: (-value[0], value[1]),
            )
            score, _, chosen = ranked[0]
            selected_domain.append(chosen)
            selected_all.append(chosen)
            global_tags.update(strings((chosen.get("screen") or {}).get("material_tags")))
            packets.append(review_packet(chosen, len(selected_domain), score))

    args.output_ids.parent.mkdir(parents=True, exist_ok=True)
    args.output_ids.write_text(
        "".join(f"{row['source_id']}\n" for row in selected_all), encoding="utf-8"
    )
    args.review_packet.parent.mkdir(parents=True, exist_ok=True)
    args.review_packet.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in packets),
        encoding="utf-8",
    )
    report = {
        "matrix_rows": len(rows),
        "excluded_ids": len(excluded),
        "eligible_candidates": len(candidates),
        "candidate_pool_by_domain": dict(sorted(pool_counts.items())),
        "requested_by_domain": targets,
        "selected": len(selected_all),
        "selected_by_domain": dict(sorted(Counter(row["evaluation_domain"] for row in packets).items())),
        "selected_by_authority_type": dict(sorted(Counter(row["authority_type"] for row in packets).items())),
        "review_status": "candidate_not_reviewed",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
