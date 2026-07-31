#!/usr/bin/env python3
"""Build and fail-closed audit K-SOVA preference pairs from reviewed triplets."""

from __future__ import annotations

import argparse
import collections
import json
import re
import urllib.parse
from pathlib import Path
from typing import Any


ANSWER_KEYS = ("korean_aligned", "context_neutral", "korean_opposed")
COUNTRY_PATTERN = re.compile(
    r"대한민국|한국|북한|미국|중국|일본|독일|영국|프랑스|러시아|호주|캐나다|"
    r"유럽연합|EU|ROK|DPRK|United States|China|Japan",
    re.IGNORECASE,
)
PERSONA_PATTERN = re.compile(
    r"한국(?:인|의\s*공무원)?\s*(?:으로서|입장에서)|대한민국\s*정부를\s*대변",
    re.IGNORECASE,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalized(value: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]", "", value.casefold())


def ngrams(value: str, size: int = 4) -> set[str]:
    value = normalized(value)
    if len(value) <= size:
        return {value} if value else set()
    return {value[index : index + size] for index in range(len(value) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left and right else 0.0


def item_source_ids(row: dict[str, Any]) -> set[str]:
    return {
        str(source.get("source_id"))
        for source in row.get("sources") or []
        if isinstance(source, dict) and source.get("source_id")
    }


def normalize_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value.strip())
    if not parsed.netloc:
        return ""
    return urllib.parse.urlunsplit(
        (parsed.scheme.casefold(), parsed.netloc.casefold(), parsed.path.rstrip("/"), "", "")
    )


def item_source_urls(row: dict[str, Any]) -> set[str]:
    return {
        normalized_url
        for source in row.get("sources") or []
        if isinstance(source, dict)
        and (normalized_url := normalize_url(str(source.get("url") or source.get("final_url") or "")))
    }


def object_source_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(source_id)
        for row in rows
        for source_id in row.get("source_ids") or []
        if source_id
    }


def benchmark_question(row: dict[str, Any]) -> str:
    return str(row.get("question_ko") or (row.get("prompts") or {}).get("ko") or "")


def first_two_sentences(value: str) -> str:
    parts = re.split(r"(?<=[.!?。！？])\s+", value.strip())
    return " ".join(parts[:2]).strip()


def country_signature(value: str) -> tuple[str, ...]:
    return tuple(match.group(0).casefold() for match in COUNTRY_PATTERN.finditer(value))


def preference_rows(item: dict[str, Any], split: str) -> list[dict[str, Any]]:
    answers = item["answers"]
    object_id = str((item.get("provenance") or {}).get("object_id") or "")
    common = {
        "item_id": str(item["item_id"]),
        "object_id": object_id,
        "split": split,
        "topic_area": str(item.get("topic_area") or ""),
        "conflict_type": str(item.get("conflict_type") or ""),
        "question_act": str(item.get("question_act") or ""),
        "source_ids": sorted(item_source_ids(item)),
        "shared_facts": item.get("shared_facts") or [],
        "prompt": str(item["question_ko"]),
        "chosen": str(answers["korean_aligned"]),
    }
    return [
        {
            "pair_id": f"{item['item_id']}::aligned-over-neutral",
            **common,
            "pair_type": "aligned_over_neutral",
            "rejected": str(answers["context_neutral"]),
        },
        {
            "pair_id": f"{item['item_id']}::aligned-over-opposed",
            **common,
            "pair_type": "aligned_over_opposed",
            "rejected": str(answers["korean_opposed"]),
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--train-objects", type=Path, required=True)
    parser.add_argument("--validation-objects", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, action="append", required=True)
    parser.add_argument("--output-train", type=Path, required=True)
    parser.add_argument("--output-validation", type=Path, required=True)
    parser.add_argument("--excluded", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--minimum-items", type=int, default=2500)
    parser.add_argument("--minimum-pairs", type=int, default=5000)
    parser.add_argument("--minimum-topic-areas", type=int, default=7)
    parser.add_argument("--maximum-topic-fraction", type=float, default=0.35)
    parser.add_argument("--benchmark-similarity", type=float, default=0.82)
    args = parser.parse_args()

    items = load_jsonl(args.items)
    train_objects = load_jsonl(args.train_objects)
    validation_objects = load_jsonl(args.validation_objects)
    train_object_ids = {str(row["object_id"]) for row in train_objects}
    validation_object_ids = {str(row["object_id"]) for row in validation_objects}
    train_sources = object_source_ids(train_objects)
    validation_sources = object_source_ids(validation_objects)

    benchmark_rows = [row for path in args.benchmark for row in load_jsonl(path)]
    benchmark_sources = {
        str(source.get("source_id"))
        for row in benchmark_rows
        for source in row.get("sources") or []
        if isinstance(source, dict) and source.get("source_id")
    }
    benchmark_source_urls = {
        url for row in benchmark_rows for url in item_source_urls(row)
    }
    benchmark_questions = [
        (normalized(question), ngrams(question))
        for row in benchmark_rows
        if (question := benchmark_question(row)).strip()
    ]

    accepted_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
    }
    excluded: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    seen_items: set[str] = set()
    for item in items:
        item_id = str(item.get("item_id") or "")
        object_id = str((item.get("provenance") or {}).get("object_id") or "")
        question = str(item.get("question_ko") or "").strip()
        answers = item.get("answers")
        errors: list[str] = []
        if item_id in seen_items or not item_id:
            errors.append("duplicate_or_missing_item_id")
        signature = normalized(question)
        if not signature or signature in seen_questions:
            errors.append("duplicate_or_missing_question")
        if PERSONA_PATTERN.search(question):
            errors.append("persona_or_answer_cue")
        if object_id in train_object_ids:
            split = "train"
        elif object_id in validation_object_ids:
            split = "validation"
        else:
            split = ""
            errors.append("object_not_in_split")
        if not isinstance(answers, dict) or set(answers) != set(ANSWER_KEYS):
            errors.append("invalid_answers")
        else:
            values = [str(answers[key]).strip() for key in ANSWER_KEYS]
            if any(len(value) < 100 for value in values):
                errors.append("short_answer")
            if len({normalized(value) for value in values}) != len(ANSWER_KEYS):
                errors.append("duplicate_direction_answer")
            if len({first_two_sentences(value) for value in values}) != 1:
                errors.append("nonidentical_shared_preamble")
            if len({country_signature(value) for value in values}) != 1:
                errors.append("asymmetric_country_surface")
        source_ids = item_source_ids(item)
        source_urls = item_source_urls(item)
        if not source_ids:
            errors.append("missing_source_ids")
        if source_ids & benchmark_sources:
            errors.append("benchmark_source_overlap")
        if source_urls & benchmark_source_urls:
            errors.append("benchmark_source_url_overlap")
        question_grams = ngrams(question)
        if any(
            signature == benchmark_signature
            or jaccard(question_grams, benchmark_grams) >= args.benchmark_similarity
            for benchmark_signature, benchmark_grams in benchmark_questions
        ):
            errors.append("benchmark_question_overlap")
        if errors:
            excluded.append(
                {"item_id": item_id, "object_id": object_id, "errors": sorted(set(errors))}
            )
            continue
        seen_items.add(item_id)
        seen_questions.add(signature)
        accepted_by_split[split].append(item)

    train_pairs = [
        pair
        for item in accepted_by_split["train"]
        for pair in preference_rows(item, "train")
    ]
    validation_pairs = [
        pair
        for item in accepted_by_split["validation"]
        for pair in preference_rows(item, "validation")
    ]
    all_items = accepted_by_split["train"] + accepted_by_split["validation"]
    all_pairs = train_pairs + validation_pairs
    accepted_source_urls = {
        url for row in all_items for url in item_source_urls(row)
    }
    topic_counts = collections.Counter(str(row.get("topic_area") or "unknown") for row in all_items)
    largest_topic_fraction = max(topic_counts.values(), default=0) / len(all_items) if all_items else 0.0
    gates = {
        "minimum_items": len(all_items) >= args.minimum_items,
        "minimum_pairs": len(all_pairs) >= args.minimum_pairs,
        "nonempty_train_validation": bool(train_pairs) and bool(validation_pairs),
        "object_split_disjoint": not (train_object_ids & validation_object_ids),
        "source_split_disjoint": not (train_sources & validation_sources),
        "benchmark_source_disjoint": not ((train_sources | validation_sources) & benchmark_sources),
        "benchmark_source_url_disjoint": not (
            accepted_source_urls & benchmark_source_urls
        ),
        "minimum_topic_areas": len(topic_counts) >= args.minimum_topic_areas,
        "maximum_topic_fraction": largest_topic_fraction <= args.maximum_topic_fraction,
        "pair_ids_unique": len({str(row["pair_id"]) for row in all_pairs}) == len(all_pairs),
        "two_pairs_per_item": len(all_pairs) == 2 * len(all_items),
    }
    report = {
        "input_items": len(items),
        "accepted_items": len(all_items),
        "excluded_items": len(excluded),
        "train_items": len(accepted_by_split["train"]),
        "validation_items": len(accepted_by_split["validation"]),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs),
        "total_pairs": len(all_pairs),
        "train_objects": len(train_object_ids),
        "validation_objects": len(validation_object_ids),
        "train_validation_source_overlap": len(train_sources & validation_sources),
        "benchmark_source_overlap": len((train_sources | validation_sources) & benchmark_sources),
        "benchmark_source_url_overlap": len(
            accepted_source_urls & benchmark_source_urls
        ),
        "by_topic_area": dict(sorted(topic_counts.items())),
        "by_conflict_type": dict(
            sorted(collections.Counter(str(row.get("conflict_type")) for row in all_items).items())
        ),
        "by_pair_type": dict(
            sorted(collections.Counter(str(row["pair_type"]) for row in all_pairs).items())
        ),
        "largest_topic_fraction": largest_topic_fraction,
        "exclusion_reasons": dict(
            sorted(collections.Counter(error for row in excluded for error in row["errors"]).items())
        ),
        "gates": gates,
        "passed": all(gates.values()),
    }
    write_jsonl(args.output_train, train_pairs)
    write_jsonl(args.output_validation, validation_pairs)
    write_jsonl(args.excluded, excluded)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
