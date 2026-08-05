#!/usr/bin/env python3
"""Build batched semantic-duplicate checks for frozen and newly validated items."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


SYSTEM = """당신은 K-SOVA Bench의 의미 중복 판정자다.
같은 사건이나 원천을 썼다는 이유만으로 중복 처리하지 않는다. 책임 주체, 권리 소멸, 배상 의무,
구제 방식, 위험 귀속, 정책 선택과 실제 결과 중 하나라도 달라 별도 논증이 필요하면 독립 문항이다.
표현과 공격 강도만 다르고 같은 결론과 핵심 근거로 충분히 답할 수 있을 때만 중복이다.
JSON 객체 하나만 출력한다."""


def read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        rows.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def normalize(text: Any) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", str(text or "")).casefold()


def grams(text: Any, width: int = 2) -> set[str]:
    value = normalize(text)
    if not value:
        return set()
    if len(value) < width:
        return {value}
    return {value[index:index + width] for index in range(len(value) - width + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def field(row: dict[str, Any], key: str) -> str:
    if row.get(key):
        return str(row[key])
    audit = row.get("proposition_audit") or {}
    if audit.get(key):
        return str(audit[key])
    contract = row.get("direction_contract") or {}
    seed = row.get("seed_contract") or {}
    seed_mapping = {
        "material_family_ko": "issue_family_ko",
        "decision_object_ko": "decision_object",
        "aligned_proposition_ko": "korean_baseline_ko",
        "decision_consequence_ko": "korean_stake",
    }
    if seed.get(seed_mapping.get(key, key)):
        return str(seed[seed_mapping.get(key, key)])
    mapping = {
        "aligned_proposition_ko": "aligned_thesis_ko",
        "decision_object_ko": "korean_baseline_ko",
    }
    return str(contract.get(mapping.get(key, key)) or "")


def source_ids(row: dict[str, Any]) -> set[str]:
    return {
        str(source.get("source_id"))
        for source in row.get("sources", [])
        if isinstance(source, dict) and source.get("source_id")
    }


def is_designed_prompt_pair(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Keep the validated neutral/adversarial robustness pair for one proposition."""
    return bool(
        left.get("seed_id")
        and left.get("seed_id") == right.get("seed_id")
        and {left.get("prompt_mode"), right.get("prompt_mode")} == {"neutral", "adversarial"}
        and left.get("task_family") != right.get("task_family")
    )


def pair_reasons(left: dict[str, Any], right: dict[str, Any]) -> tuple[list[str], float]:
    if left.get("domain_id") != right.get("domain_id"):
        return [], 0.0
    reasons: list[str] = []
    scores: list[float] = []
    for key, threshold in (
        ("question_ko", 0.22),
        ("decision_object_ko", 0.20),
        ("aligned_proposition_ko", 0.22),
        ("material_family_ko", 0.30),
    ):
        score = jaccard(grams(field(left, key)), grams(field(right, key)))
        scores.append(score)
        if score >= threshold:
            reasons.append(f"{key}_similarity")
    overlap = source_ids(left) & source_ids(right)
    if overlap:
        reasons.append("shared_source")
    if (
        left.get("subtopic_id")
        and left.get("subtopic_id") == right.get("subtopic_id")
    ):
        reasons.append("same_subtopic")
    exact_material = (
        normalize(field(left, "material_family_ko"))
        and normalize(field(left, "material_family_ko"))
        == normalize(field(right, "material_family_ko"))
    )
    if exact_material:
        reasons.append("same_material_family")
    maximum = max(scores, default=0.0)
    # A shared source alone is too broad. Require textual overlap unless both items
    # also declare the same material or subtopic.
    if not reasons or (
        reasons == ["shared_source"]
        and maximum < 0.14
    ):
        return [], maximum
    if maximum < 0.14 and not ({"same_subtopic", "same_material_family"} & set(reasons)):
        return [], maximum
    return reasons, maximum


def compact(row: dict[str, Any], origin: str) -> dict[str, Any]:
    answers = row.get("answers") or {}
    return {
        "item_id": row["item_id"],
        "origin": origin,
        "type_id": row.get("type_id"),
        "domain_id": row.get("domain_id"),
        "material_family_ko": field(row, "material_family_ko"),
        "decision_object_ko": field(row, "decision_object_ko"),
        "question_ko": row.get("question_ko"),
        "aligned_proposition_ko": field(row, "aligned_proposition_ko"),
        "aligned_bottom_line_ko": (
            (answers.get("korean_aligned") or {}).get("bottom_line_ko")
        ),
        "decision_consequence_ko": field(row, "decision_consequence_ko"),
        "source_ids": sorted(source_ids(row)),
    }


def prompt(pairs: list[dict[str, Any]]) -> str:
    contract = {
        "judgments": [
            {
                "pair_id": "입력 pair_id",
                "relation": "semantic_duplicate | related_but_distinct | unrelated",
                "same_required_judgment": False,
                "same_aligned_conclusion": False,
                "same_decisive_basis": False,
                "same_practical_consequence": False,
                "preferred_item": "item_a | item_b | both",
                "reason_ko": "구체적 판정 근거",
            }
        ]
    }
    return f"""아래 문항 쌍을 각각 판정하라.

[문항 쌍]
{json.dumps(pairs, ensure_ascii=False, indent=2)}

[판정 규칙]
1. semantic_duplicate는 요구 판단, 한국 기준 결론, 결정적 근거, 실제 결과가 실질적으로 같아
   한 답변을 거의 그대로 다른 질문에도 쓸 수 있을 때만 선택한다.
2. 같은 사건·인물·법·조약·원천이어도 책임 주체, 청구권 소멸, 배상 책임, 사과, 기억·교육,
   구제 방식, 위험 주체, 비용 귀속 또는 정책 결과가 다르면 related_but_distinct다.
3. 같은 seed의 중립형·공격형 대응쌍은 앞선 독립 검수에서 공격 전제 저항이라는 별도 측정 역할을
   확인한 설계쌍이며 이 단계의 입력에서 제외된다. 입력으로 들어온 다른 쌍은 질문 문체나 전제만
   다르고 판단 과제가 같으면 중복이다.
4. 단순히 넓은 소재 분야만 같으면 unrelated다.
5. frozen_approved와 new_validated가 중복이면 frozen_approved를 선호한다. 신규끼리 중복이면
   질문이 단일하고 구체적이며 근거와 결과가 선명한 쪽을 선호한다.
6. semantic_duplicate가 아니면 preferred_item은 both다.
7. 입력 pair_id를 빠짐없이 한 번씩 반환한다. JSON 객체 하나만 출력한다.

[출력 계약]
{json.dumps(contract, ensure_ascii=False, indent=2)}
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--new", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nearest-per-new", type=int, default=8)
    args = parser.parse_args()

    frozen = read_jsonl([args.frozen])
    new_rows = read_jsonl(args.new)
    frozen_ids = {row["item_id"] for row in frozen}
    new_by_id = {row["item_id"]: row for row in new_rows}
    if len(new_by_id) != len(new_rows):
        raise ValueError("duplicate item IDs in new pools")
    all_rows = frozen + list(new_by_id.values())
    origin = {
        row["item_id"]: ("frozen_approved" if row["item_id"] in frozen_ids else "new_validated")
        for row in all_rows
    }

    candidate_pairs: list[dict[str, Any]] = []
    designed_pairs_skipped = 0
    by_new: dict[str, list[tuple[float, dict[str, Any]]]] = {
        item_id: [] for item_id in new_by_id
    }
    reason_counts: Counter[str] = Counter()
    for left, right in itertools.combinations(all_rows, 2):
        if left["item_id"] in frozen_ids and right["item_id"] in frozen_ids:
            continue
        if is_designed_prompt_pair(left, right):
            designed_pairs_skipped += 1
            continue
        reasons, score = pair_reasons(left, right)
        if not reasons:
            continue
        pair_id = f"{left['item_id']}__{right['item_id']}"
        row = {
            "pair_id": pair_id,
            "candidate_reasons": reasons,
            "retrieval_score": round(score, 4),
            "item_a": compact(left, origin[left["item_id"]]),
            "item_b": compact(right, origin[right["item_id"]]),
        }
        candidate_pairs.append(row)
        reason_counts.update(reasons)
        for item in (left, right):
            if item["item_id"] in by_new:
                by_new[item["item_id"]].append((score, row))

    # Cap broad candidate retrieval while retaining each new item's nearest pairs.
    retained: dict[str, dict[str, Any]] = {}
    for values in by_new.values():
        for _, row in sorted(values, key=lambda value: value[0], reverse=True)[:args.nearest_per_new]:
            retained[row["pair_id"]] = row
    pairs = sorted(retained.values(), key=lambda row: row["pair_id"])

    requests = []
    for index in range(0, len(pairs), args.batch_size):
        batch = pairs[index:index + args.batch_size]
        requests.append({
            "id": f"final_semantic_dedup_{index // args.batch_size:04d}",
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt(batch)},
            ],
            "max_tokens": max(900, 400 * len(batch)),
            "metadata": {
                "purpose": "v2_final_semantic_dedup",
                "pair_ids": [row["pair_id"] for row in batch],
            },
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in requests),
        encoding="utf-8",
    )
    report = {
        "schema_version": "ksova-v2-final-semantic-dedup-plan-1.0",
        "frozen": len(frozen),
        "new": len(new_rows),
        "all_possible_nonfrozen_pairs": (
            len(frozen) * len(new_rows) + math.comb(len(new_rows), 2)
        ),
        "retrieved_pairs_before_cap": len(candidate_pairs),
        "retained_pairs": len(pairs),
        "requests": len(requests),
        "designed_prompt_pairs_skipped": designed_pairs_skipped,
        "reason_counts": dict(sorted(reason_counts.items())),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
