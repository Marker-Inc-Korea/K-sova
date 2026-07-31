#!/usr/bin/env python3
"""Split conflict objects by connected source components without source leakage."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def source_ids(row: dict[str, Any]) -> set[str]:
    return {str(value) for value in row.get("source_ids") or [] if value}


def stable_rank(seed: int, values: list[str]) -> str:
    return hashlib.sha256(f"{seed}:{'|'.join(sorted(values))}".encode()).hexdigest()


def topic_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(collections.Counter(
        str(row.get("topic_area") or "unknown") for row in rows
    ).items()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    rows = load_jsonl(args.objects)
    if len(rows) < 2:
        raise ValueError("at least two audited objects are required")
    identifiers = [str(row["object_id"]) for row in rows]
    union = UnionFind(identifiers)
    owners: dict[str, str] = {}
    for row in rows:
        object_id = str(row["object_id"])
        for source_id in source_ids(row):
            if source_id in owners:
                union.union(object_id, owners[source_id])
            else:
                owners[source_id] = object_id

    components: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        components[union.find(str(row["object_id"]))].append(row)
    values = list(components.values())
    values.sort(key=lambda component: stable_rank(
        args.seed, [str(row["object_id"]) for row in component]
    ))

    target = max(1, round(len(rows) * args.validation_fraction))
    topic_components: dict[str, list[int]] = collections.defaultdict(list)
    for index, component in enumerate(values):
        for topic in {str(row.get("topic_area") or "unknown") for row in component}:
            topic_components[topic].append(index)

    selected: set[int] = set()
    validation_size = 0
    for topic in sorted(topic_components):
        candidates = topic_components[topic]
        if len(candidates) < 2:
            continue
        index = min(candidates, key=lambda value: (
            abs((validation_size + len(values[value])) - target), value
        ))
        if index not in selected:
            selected.add(index)
            validation_size += len(values[index])

    for index, component in enumerate(values):
        if validation_size >= target:
            break
        if index in selected:
            continue
        topics = {str(row.get("topic_area") or "unknown") for row in component}
        would_empty_topic = any(
            all(candidate in selected or candidate == index for candidate in topic_components[topic])
            for topic in topics
        )
        if would_empty_topic:
            continue
        selected.add(index)
        validation_size += len(component)

    validation = [row for index in sorted(selected) for row in values[index]]
    validation_ids = {str(row["object_id"]) for row in validation}
    train = [row for row in rows if str(row["object_id"]) not in validation_ids]
    train_sources = set().union(*(source_ids(row) for row in train)) if train else set()
    validation_sources = (
        set().union(*(source_ids(row) for row in validation)) if validation else set()
    )
    overlap = train_sources & validation_sources
    if not train or not validation or overlap:
        raise ValueError(
            f"invalid source-component split: train={len(train)} validation={len(validation)} "
            f"source_overlap={len(overlap)}"
        )

    train.sort(key=lambda row: str(row["object_id"]))
    validation.sort(key=lambda row: str(row["object_id"]))
    write_jsonl(args.train, train)
    write_jsonl(args.validation, validation)
    report = {
        "objects": len(rows),
        "source_connected_components": len(values),
        "validation_target": target,
        "train_objects": len(train),
        "validation_objects": len(validation),
        "train_sources": len(train_sources),
        "validation_sources": len(validation_sources),
        "source_overlap": len(overlap),
        "train_by_topic_area": topic_counts(train),
        "validation_by_topic_area": topic_counts(validation),
        "split_policy": "connected components over source_id; deterministic component holdout",
        "seed": args.seed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
