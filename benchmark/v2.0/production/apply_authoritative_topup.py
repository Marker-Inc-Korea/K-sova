#!/usr/bin/env python3
"""Apply manually body-read authoritative evidence to selected material seeds."""

from __future__ import annotations

import argparse
import json
import urllib.parse
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def append_unique(values: list[str], additions: list[str]) -> list[str]:
    return list(dict.fromkeys([*values, *additions]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--topup", type=Path, required=True)
    parser.add_argument("--output-seeds", type=Path, required=True)
    parser.add_argument("--output-sources", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    seeds = read_jsonl(args.seeds)
    sources = read_jsonl(args.sources)
    topup = json.loads(args.topup.read_text(encoding="utf-8"))
    seed_by_id = {str(row["seed_id"]): row for row in seeds}
    source_by_id = {str(row["source_id"]): row for row in sources}

    if len(seed_by_id) != len(seeds):
        raise ValueError("duplicate seed_id in input")
    if len(source_by_id) != len(sources):
        raise ValueError("duplicate source_id in input")

    source_defaults = dict(topup.get("source_defaults") or {})
    new_sources: list[dict[str, Any]] = []
    for raw_source in topup.get("sources") or []:
        source = {**source_defaults, **raw_source}
        evidence_ko = str(source.pop("evidence_ko", "")).strip()
        if evidence_ko and not source.get("evidence_records"):
            source["evidence_records"] = [
                {
                    "evidence_id": f"{source['source_id']}-E1",
                    "kind": "retrieved_body_excerpt",
                    "text_ko": evidence_ko,
                    "locator": source["url"],
                }
            ]
        if not source.get("read_scope"):
            source["read_scope"] = evidence_ko
        new_sources.append(source)
    new_ids: set[str] = set()
    for source in new_sources:
        source_id = str(source.get("source_id") or "")
        if not source_id or source_id in source_by_id or source_id in new_ids:
            raise ValueError(f"invalid or duplicate topup source_id: {source_id!r}")
        new_ids.add(source_id)
        if source.get("reliability_tier") not in {"A", "B"}:
            raise ValueError(f"{source_id} is not A/B tier")
        if source.get("body_read") is not True:
            raise ValueError(f"{source_id} is not marked body_read")
        if "generation_grounding" not in set(source.get("allowed_uses") or []):
            raise ValueError(f"{source_id} is not generation grounding")
        if not source.get("evidence_records"):
            raise ValueError(f"{source_id} has no evidence records")
        url = str(source.get("url") or "")
        parsed = urllib.parse.urlsplit(url)
        host = (parsed.hostname or "").casefold().strip(".")
        if parsed.scheme not in {"http", "https"} or not host:
            raise ValueError(f"{source_id} has an invalid URL")
        if host == "kp" or host.endswith(".kp"):
            raise ValueError(f"{source_id} uses a prohibited host")
        declared_host = str(source.get("host") or "").casefold().strip(".")
        if declared_host != host:
            raise ValueError(
                f"{source_id} host mismatch: declared={declared_host!r}, URL={host!r}"
            )
        source_by_id[source_id] = source

    binding_counts: Counter[str] = Counter()
    bound_seed_ids: set[str] = set()
    for binding in topup.get("bindings") or []:
        seed_id = str(binding.get("seed_id") or "")
        if seed_id not in seed_by_id:
            raise ValueError(f"unknown binding seed_id: {seed_id!r}")
        if seed_id in bound_seed_ids:
            raise ValueError(f"duplicate binding seed_id: {seed_id}")
        bound_seed_ids.add(seed_id)
        seed = seed_by_id[seed_id]
        for position, key in (
            ("korean_position", "korean_source_ids"),
            ("counter_position", "counter_source_ids"),
        ):
            additions = [str(value) for value in binding.get(key) or []]
            unknown = [value for value in additions if value not in new_ids]
            if unknown:
                raise ValueError(f"{seed_id} references non-topup sources: {unknown}")
            seed[position]["source_ids"] = append_unique(
                list(seed[position].get("source_ids") or []), additions
            )
            binding_counts[position] += len(additions)
        combined = set(seed["korean_position"]["source_ids"])
        combined.update(seed["counter_position"]["source_ids"])
        if not combined.intersection(new_ids):
            raise ValueError(f"{seed_id} received no authoritative source")

    output_sources = [*sources, *new_sources]
    if len({row["source_id"] for row in output_sources}) != len(output_sources):
        raise ValueError("duplicate source_id after merge")
    write_jsonl(args.output_seeds, seeds)
    write_jsonl(args.output_sources, output_sources)

    report = {
        "schema_version": "ksova-v2-authoritative-topup-report-1.0",
        "input_seed_count": len(seeds),
        "input_source_count": len(sources),
        "new_source_count": len(new_sources),
        "enriched_seed_count": len(bound_seed_ids),
        "binding_counts": dict(sorted(binding_counts.items())),
        "enriched_seed_ids": sorted(bound_seed_ids),
        "output_seed_count": len(seeds),
        "output_source_count": len(output_sources),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
