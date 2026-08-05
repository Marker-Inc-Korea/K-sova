#!/usr/bin/env python3
"""Select the final items and a balanced reserve for semantic deduplication."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TYPE_TARGETS = {"T01": 170, "T02": 220, "T03": 180, "T04": 250, "T05": 180}
DOMAIN_TARGETS = {
    "D01": 100, "D02": 35, "D03": 65, "D04": 115,
    "D05": 90, "D06": 110, "D07": 75, "D08": 190,
    "D09": 85, "D10": 55, "D11": 35, "D12": 45,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def type_id(row: dict[str, Any]) -> str:
    return str(row.get("type_id") or (row.get("seed_contract") or {}).get("type_id") or "T04")


def domain_id(row: dict[str, Any]) -> str:
    return str(row.get("domain_id") or (row.get("seed_contract") or {}).get("domain_id") or "D05")


def family(row: dict[str, Any]) -> str:
    seed = row.get("seed_contract") or {}
    return str(
        row.get("material_family_ko")
        or seed.get("issue_family_ko")
        or seed.get("lane")
        or row.get("case_id")
        or row.get("seed_id")
        or row["item_id"]
    )


def quality(row: dict[str, Any]) -> int:
    qwen = int((row.get("qwen_validation") or {}).get("quality_score") or 0)
    oss = int((row.get("oss_validation") or {}).get("quality_score") or 0)
    return min(qwen, oss)


def source_strength(row: dict[str, Any]) -> int:
    tiers = {
        str(source.get("reliability_tier") or "")
        for source in row.get("sources") or []
        if isinstance(source, dict)
    }
    return 2 if tiers & {"A", "B"} else 1 if "C" in tiers else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--all-new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--target-pool", type=int, default=1500)
    args = parser.parse_args()

    selected = read_jsonl(args.selected)
    all_new = read_jsonl(args.all_new)
    all_by_id = {str(row["item_id"]): row for row in all_new}
    if len(all_by_id) != len(all_new):
        raise ValueError("duplicate item IDs in --all-new")

    selected_ids = [
        str(row["item_id"])
        for row in selected
        if row.get("assembly_origin") == "new_double_validated"
    ]
    missing = sorted(set(selected_ids) - set(all_by_id))
    if missing:
        raise ValueError(f"selected new items missing from --all-new: {missing[:10]}")
    if args.target_pool < len(selected_ids):
        raise ValueError(
            f"target pool {args.target_pool} is smaller than selected new pool {len(selected_ids)}"
        )

    pool = [all_by_id[item_id] for item_id in selected_ids]
    pool_ids = set(selected_ids)
    seed_counts = Counter(str(row.get("seed_id") or row["item_id"]) for row in pool)
    type_counts = Counter(type_id(row) for row in pool)
    domain_counts = Counter(domain_id(row) for row in pool)
    mode_counts = Counter(str(row.get("prompt_mode") or "neutral") for row in pool)
    family_counts = Counter(family(row) for row in pool)
    remaining = [row for row in all_new if str(row["item_id"]) not in pool_ids]
    scale = args.target_pool / 1000
    type_targets = {key: round(value * scale) for key, value in TYPE_TARGETS.items()}
    domain_targets = {key: round(value * scale) for key, value in DOMAIN_TARGETS.items()}
    mode_targets = {
        "adversarial": round(args.target_pool * 0.25),
        "neutral": args.target_pool - round(args.target_pool * 0.25),
    }
    maximum_adversarial = round(args.target_pool * 0.35)

    # Recompute priorities after each selection. Unrepresented propositions come
    # first, then underrepresented benchmark cells and stronger evidence.
    while remaining and len(pool) < args.target_pool:
        available = [
            row for row in remaining
            if str(row.get("prompt_mode") or "neutral") != "adversarial"
            or mode_counts["adversarial"] < maximum_adversarial
        ]
        if not available:
            break

        def priority(row: dict[str, Any]) -> tuple[Any, ...]:
            seed = str(row.get("seed_id") or row["item_id"])
            current_type = type_counts[type_id(row)]
            current_domain = domain_counts[domain_id(row)]
            type_need = max(type_targets.get(type_id(row), 0) - current_type, 0)
            domain_need = max(domain_targets.get(domain_id(row), 0) - current_domain, 0)
            mode = str(row.get("prompt_mode") or "neutral")
            mode_need = max(mode_targets.get(mode, 0) - mode_counts[mode], 0)
            return (
                int(seed_counts[seed] == 0),
                18 * type_need / max(type_targets.get(type_id(row), 1), 1)
                + 15 * domain_need / max(domain_targets.get(domain_id(row), 1), 1)
                + 8 * mode_need / max(mode_targets.get(mode, 1), 1)
                - 0.5 * family_counts[family(row)],
                source_strength(row),
                quality(row),
                len(row.get("sources") or []),
                str(row["item_id"]),
            )

        best = max(available, key=priority)
        remaining.remove(best)
        pool.append(best)
        pool_ids.add(str(best["item_id"]))
        seed_counts[str(best.get("seed_id") or best["item_id"])] += 1
        type_counts[type_id(best)] += 1
        domain_counts[domain_id(best)] += 1
        mode_counts[str(best.get("prompt_mode") or "neutral")] += 1
        family_counts[family(best)] += 1

    write_jsonl(args.output, pool)
    report = {
        "schema_version": "ksova-v2-semantic-candidate-pool-1.0",
        "selected_input": len(selected),
        "selected_new": len(selected_ids),
        "all_double_validated": len(all_new),
        "target_pool": args.target_pool,
        "pool": len(pool),
        "shortfall": max(args.target_pool - len(pool), 0),
        "reserve_added": len(pool) - len(selected_ids),
        "unique_propositions": len(seed_counts),
        "by_type": dict(sorted(type_counts.items())),
        "by_domain": dict(sorted(domain_counts.items())),
        "by_prompt_mode": dict(sorted(mode_counts.items())),
        "maximum_adversarial": maximum_adversarial,
        "by_source_strength": dict(sorted(Counter(source_strength(row) for row in pool).items())),
        "minimum_quality": min((quality(row) for row in pool), default=None),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
