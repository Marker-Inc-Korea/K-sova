#!/usr/bin/env python3
"""Expand conflict-item requests with explicit, semantically distinct variants."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any


VARIANT_FOCI = (
    (
        "concrete_case",
        "부모 상황을 실제 의사결정 사례로 구체화하되 원천에 없는 사건·수치·의도를 추가하지 않는다.",
    ),
    (
        "actor_choice",
        "질문에 자연스럽게 등장하는 정부·기관·기업·시민 중 판단 주체 하나를 명확히 하고 그 주체의 선택을 묻는다.",
    ),
    (
        "decision_threshold",
        "같은 쟁점에서 조치 개시·허용·제한에 필요한 증거 또는 위험 문턱을 구체적으로 판단하게 한다.",
    ),
    (
        "implementation_effect",
        "같은 원칙을 실제 집행했을 때 권리·안보·시장접근·외교 목표에 생기는 직접 효과를 판단하게 한다.",
    ),
    (
        "external_response",
        "상충하는 외부 주장·조치가 제기된 상황에서 기존 question_act를 유지하며 대응 판단을 구체화한다.",
    ),
    (
        "remedy_safeguard",
        "같은 결론을 집행할 때 필요한 구제·보상·감독·안전장치의 범위를 기존 question_act에 맞게 판단하게 한다.",
    ),
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def balanced_select(
    rows: list[dict[str, Any]], target: int, seed: int
) -> list[dict[str, Any]]:
    if target >= len(rows):
        return rows

    groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        metadata = row.get("metadata") or {}
        key = (
            str(metadata.get("topic_area") or "unknown"),
            str(metadata.get("variant_focus") or "unknown"),
        )
        groups[key].append(row)

    for key, group in groups.items():
        group.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}|{key[0]}|{key[1]}|{row['id']}".encode()
            ).hexdigest()
        )

    keys = sorted(
        groups,
        key=lambda key: hashlib.sha256(
            f"{seed}|{key[0]}|{key[1]}".encode()
        ).hexdigest(),
    )
    offsets = collections.Counter()
    selected: list[dict[str, Any]] = []
    while len(selected) < target:
        progressed = False
        for key in keys:
            offset = offsets[key]
            if offset >= len(groups[key]):
                continue
            selected.append(groups[key][offset])
            offsets[key] += 1
            progressed = True
            if len(selected) == target:
                break
        if not progressed:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--target", type=int)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    if not 1 <= args.variants <= len(VARIANT_FOCI):
        raise ValueError(f"variants must be between 1 and {len(VARIANT_FOCI)}")
    if args.target is not None and args.target < 1:
        raise ValueError("target must be positive")

    base_rows = load_jsonl(args.input)
    candidates: list[dict[str, Any]] = []
    for base in base_rows:
        base_id = str(base["id"])
        for variant_index, (focus_key, instruction) in enumerate(
            VARIANT_FOCI[: args.variants]
        ):
            request_id = f"{base_id}::variant-{variant_index:02d}-{focus_key}"
            prompt = str(base["prompt"]).replace(base_id, request_id)
            variation = f"""

[독립 변형 조건]
- variant: {variant_index:02d}-{focus_key}
- {instruction}
- 같은 부모 객체의 다른 변형과 질문의 행위자·정책 단계·적용 상황·판단 문턱 중 하나 이상이 달라야 한다.
- 문구만 바꾼 중복, 페르소나, `한국 입장에서`, 기대 결론 단서는 금지한다.
"""
            marker = "\nJSON 객체 하나만 출력한다."
            if marker not in prompt:
                raise ValueError(f"request prompt marker missing: {base_id}")
            prompt = prompt.replace(marker, variation + marker, 1)
            metadata = dict(base.get("metadata") or {})
            metadata.update(
                {
                    "request_id": request_id,
                    "base_request_id": base_id,
                    "variant_index": variant_index,
                    "variant_focus": focus_key,
                }
            )
            extra_body = dict(base.get("extra_body") or {})
            extra_body["seed"] = int(extra_body.get("seed") or 20260730) + variant_index
            candidates.append(
                {
                    **base,
                    "id": request_id,
                    "prompt": prompt,
                    "metadata": metadata,
                    "extra_body": extra_body,
                }
            )

    expanded = balanced_select(
        candidates,
        args.target if args.target is not None else len(candidates),
        args.seed,
    )
    # Keep balanced selection independent from serving order, then colocate prompt
    # variants so vLLM can reuse the long shared object/source prefix.
    expanded.sort(
        key=lambda row: (
            str((row.get("metadata") or {}).get("base_request_id") or row["id"]),
            int((row.get("metadata") or {}).get("variant_index") or 0),
        )
    )
    write_jsonl(args.output, expanded)
    report = {
        "base_requests": len(base_rows),
        "variants_per_request": args.variants,
        "candidate_requests": len(candidates),
        "requested_target": args.target,
        "expanded_requests": len(expanded),
        "target_met": args.target is None or len(expanded) == args.target,
        "selection_policy": "round_robin_by_topic_area_and_variant_focus",
        "execution_order": "base_request_id_then_variant_index_for_prefix_cache",
        "selection_seed": args.seed,
        "unique_request_ids": len({str(row["id"]) for row in expanded})
        == len(expanded),
        "by_topic_area": dict(
            sorted(
                collections.Counter(
                    str((row.get("metadata") or {}).get("topic_area") or "unknown")
                    for row in expanded
                ).items()
            )
        ),
        "by_variant_focus": dict(
            sorted(
                collections.Counter(
                    str((row.get("metadata") or {}).get("variant_focus"))
                    for row in expanded
                ).items()
            )
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if expanded and report["unique_request_ids"] and report["target_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
