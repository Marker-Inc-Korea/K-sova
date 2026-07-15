#!/usr/bin/env python3
"""Audit evaluation response completeness and obvious language/runtime failures."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from common import ROOT, load_jsonl, sentence_count, write_json


INTEGRITY_ISSUE_FIELDS = {
    "missing_ids",
    "unexpected_ids",
    "duplicate_ids",
    "api_error_ids",
    "empty_ids",
}
QUALITY_ISSUE_FIELDS = {
    "short_ids",
    "sentence_count_violation_ids",
    "wrong_language_ids",
    "foreign_script_leakage_ids",
    "pathological_whitespace_ids",
    "leaked_reasoning_ids",
    "truncated_ids",
}

PATHOLOGICAL_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\r\n]{32,}")


def partition_issue_ids(issue_groups: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    integrity = sorted({
        row_id
        for field in INTEGRITY_ISSUE_FIELDS
        for row_id in issue_groups.get(field, [])
    })
    quality = sorted({
        row_id
        for field in QUALITY_ISSUE_FIELDS
        for row_id in issue_groups.get(field, [])
    })
    return integrity, quality


def language_ok(text: str, language: str) -> bool:
    hangul = len(re.findall(r"[가-힣]", text))
    latin = len(re.findall(r"[A-Za-z]", text))
    kana = len(re.findall(r"[ぁ-んァ-ン]", text))
    han = len(re.findall(r"[\u4e00-\u9fff]", text))
    if language == "ko":
        return hangul >= 10 and kana < 10
    if language == "en":
        return latin >= 50 and latin >= 2 * (hangul + kana + han)
    if language == "ja":
        return kana >= 8 and hangul < 10
    if language == "zh-Hans":
        return han >= 30 and kana < 8 and hangul < 10
    return False


def foreign_script_leakage(text: str, language: str) -> bool:
    """Flag sustained out-of-script fragments without treating them as language failure."""
    has_hangul = re.search(r"[가-힣]", text) is not None
    has_hangul_fragment = re.search(r"[가-힣]{2,}", text) is not None
    has_kana_fragment = re.search(r"[ぁ-んァ-ン]{2,}", text) is not None
    has_han_fragment = re.search(r"[\u4e00-\u9fff]{3,}", text) is not None
    if language == "ko":
        return has_kana_fragment or has_han_fragment
    if language == "en":
        return has_hangul_fragment or has_kana_fragment or has_han_fragment
    if language == "ja":
        return has_hangul
    if language == "zh-Hans":
        return has_hangul or has_kana_fragment
    return False


def pathological_whitespace(text: str) -> bool:
    return PATHOLOGICAL_HORIZONTAL_WHITESPACE.search(text) is not None


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(len(ordered) * fraction))])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, default=ROOT / "data/pilot_items_v0.jsonl")
    parser.add_argument("--responses", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/model_output_audit_v0.json")
    parser.add_argument("--minimum-chars", type=int, default=120)
    parser.add_argument("--minimum-sentences", type=int, default=4)
    parser.add_argument("--maximum-sentences", type=int, default=8)
    args = parser.parse_args()

    items = {row["id"]: row for row in load_jsonl(args.items)}
    expected = set(items)
    reports = []
    files_with_errors = 0
    files_with_quality_flags = 0
    all_issue_ids: set[str] = set()
    all_integrity_issue_ids: set[str] = set()
    all_quality_issue_ids: set[str] = set()
    for path in args.responses:
        rows = load_jsonl(path)
        models = sorted({str(row.get("model")) for row in rows})
        model = models[0] if len(models) == 1 else "mixed:" + ",".join(models)
        ids = [str(row.get("id")) for row in rows]
        id_counts = Counter(ids)
        duplicate_ids = sorted(key for key, count in id_counts.items() if count > 1)
        row_by_id = {str(row.get("id")): row for row in rows}
        missing_ids = sorted(expected - set(ids))
        unexpected_ids = sorted(set(ids) - expected)
        api_error_ids = sorted(row_id for row_id, row in row_by_id.items() if row.get("error"))
        empty_ids = sorted(row_id for row_id, row in row_by_id.items() if not str(row.get("response") or "").strip())
        short_ids = sorted(
            row_id for row_id, row in row_by_id.items()
            if row_id in items and 0 < len(str(row.get("response") or "").strip()) < args.minimum_chars
        )
        sentence_counts = {
            row_id: sentence_count(str(row.get("response") or ""))
            for row_id, row in row_by_id.items()
        }
        sentence_count_violation_ids = sorted(
            row_id for row_id, count in sentence_counts.items()
            if row_id in items and str(row_by_id[row_id].get("response") or "").strip()
            and not args.minimum_sentences <= count <= args.maximum_sentences
        )
        wrong_language_ids = sorted(
            row_id for row_id, row in row_by_id.items()
            if row_id in items and str(row.get("response") or "").strip()
            and not language_ok(str(row.get("response") or ""), items[row_id]["language"])
        )
        foreign_script_leakage_ids = sorted(
            row_id for row_id, row in row_by_id.items()
            if row_id in items and str(row.get("response") or "").strip()
            and foreign_script_leakage(str(row.get("response") or ""), items[row_id]["language"])
        )
        pathological_whitespace_ids = sorted(
            row_id for row_id, row in row_by_id.items()
            if pathological_whitespace(str(row.get("response") or ""))
        )
        leaked_reasoning_ids = sorted(
            row_id for row_id, row in row_by_id.items()
            if re.search(r"</?think>|reasoning\s*:", str(row.get("response") or ""), flags=re.IGNORECASE)
        )
        truncated_ids = sorted(
            row_id for row_id, row in row_by_id.items()
            if (
                str(row.get("finish_reason") or "").lower() == "length"
                or (
                    not str(row.get("finish_reason") or "").strip()
                    and isinstance(row.get("usage"), dict)
                    and type(row["usage"].get("completion_tokens")) is int
                    and type(row.get("requested_max_tokens")) is int
                    and row["usage"]["completion_tokens"] >= row["requested_max_tokens"]
                )
            )
        )
        issue_groups = {
            "missing_ids": missing_ids,
            "unexpected_ids": unexpected_ids,
            "duplicate_ids": duplicate_ids,
            "api_error_ids": api_error_ids,
            "empty_ids": empty_ids,
            "short_ids": short_ids,
            "sentence_count_violation_ids": sentence_count_violation_ids,
            "wrong_language_ids": wrong_language_ids,
            "foreign_script_leakage_ids": foreign_script_leakage_ids,
            "pathological_whitespace_ids": pathological_whitespace_ids,
            "leaked_reasoning_ids": leaked_reasoning_ids,
            "truncated_ids": truncated_ids,
        }
        integrity_issue_ids, quality_issue_ids = partition_issue_ids(issue_groups)
        issue_ids = sorted(set(integrity_issue_ids + quality_issue_ids))
        all_issue_ids.update(issue_ids)
        all_integrity_issue_ids.update(integrity_issue_ids)
        all_quality_issue_ids.update(quality_issue_ids)
        response_lengths = [len(str(row.get("response") or "").strip()) for row in rows]
        reasoning_lengths = [len(str(row.get("reasoning_content") or "").strip()) for row in rows]
        usage_rows = [row["usage"] for row in rows if isinstance(row.get("usage"), dict)]
        completion_tokens = [
            int(usage["completion_tokens"]) for usage in usage_rows
            if isinstance(usage.get("completion_tokens"), (int, float))
        ]
        prompt_tokens = [
            int(usage["prompt_tokens"]) for usage in usage_rows
            if isinstance(usage.get("prompt_tokens"), (int, float))
        ]
        elapsed = [float(row["elapsed_seconds"]) for row in rows if isinstance(row.get("elapsed_seconds"), (int, float))]
        by_language: dict[str, list[int]] = defaultdict(list)
        for row_id, row in row_by_id.items():
            if row_id in items:
                by_language[items[row_id]["language"]].append(len(str(row.get("response") or "").strip()))
        report = {
            "model": model,
            "path": str(path),
            "rows": len(rows),
            "unique_ids": len(set(ids)),
            "missing_ids": missing_ids,
            "unexpected_ids": unexpected_ids,
            "duplicate_ids": duplicate_ids,
            "api_error_ids": api_error_ids,
            "empty_ids": empty_ids,
            "short_ids": short_ids,
            "sentence_count_violation_ids": sentence_count_violation_ids,
            "wrong_language_ids": wrong_language_ids,
            "foreign_script_leakage_ids": foreign_script_leakage_ids,
            "pathological_whitespace_ids": pathological_whitespace_ids,
            "leaked_reasoning_ids": leaked_reasoning_ids,
            "truncated_ids": truncated_ids,
            "issue_ids": issue_ids,
            "integrity_issue_ids": integrity_issue_ids,
            "quality_issue_ids": quality_issue_ids,
            "median_response_chars": statistics.median(response_lengths) if response_lengths else 0,
            "median_sentence_count": statistics.median(sentence_counts.values()) if sentence_counts else 0,
            "p95_response_chars": percentile(response_lengths, 0.95),
            "mean_prompt_tokens": round(statistics.fmean(prompt_tokens), 2) if prompt_tokens else None,
            "mean_completion_tokens": round(statistics.fmean(completion_tokens), 2) if completion_tokens else None,
            "max_completion_tokens": max(completion_tokens) if completion_tokens else None,
            "total_completion_tokens": sum(completion_tokens) if completion_tokens else None,
            "responses_with_reasoning": sum(length > 0 for length in reasoning_lengths),
            "mean_reasoning_chars": round(statistics.fmean(reasoning_lengths), 2) if reasoning_lengths else 0.0,
            "median_request_seconds": round(statistics.median(elapsed), 3) if elapsed else None,
            "median_chars_by_language": {
                language: statistics.median(lengths) for language, lengths in sorted(by_language.items())
            },
        }
        if integrity_issue_ids or len(rows) != len(expected):
            files_with_errors += 1
        if quality_issue_ids:
            files_with_quality_flags += 1
        reports.append(report)
    summary = {
        "expected_items": len(expected),
        "model_files": len(reports),
        "files_with_errors": files_with_errors,
        "files_with_quality_flags": files_with_quality_flags,
        "all_issue_ids": sorted(all_issue_ids),
        "all_integrity_issue_ids": sorted(all_integrity_issue_ids),
        "all_quality_issue_ids": sorted(all_quality_issue_ids),
        "reports": reports,
    }
    write_json(args.output, summary)
    print(json.dumps({
        "expected_items": len(expected),
        "model_files": len(reports),
        "files_with_errors": files_with_errors,
        "files_with_quality_flags": files_with_quality_flags,
        "integrity_issue_ids": len(all_integrity_issue_ids),
        "quality_issue_ids": len(all_quality_issue_ids),
        "issue_ids": len(all_issue_ids),
    }, ensure_ascii=False))
    return 1 if files_with_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
