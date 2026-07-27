#!/usr/bin/env python3
"""Build and apply constrained reference-answer surface-cue rewrites."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

from pipeline import (
    country_token_signature,
    extract_json_object,
    load_jsonl,
    reference_surface_text,
    validate_final_item,
    write_json,
    write_jsonl,
)


DIRECTIONS = ("korean_aligned", "context_neutral", "korean_opposed")
EDITABLE_FIELDS = ("answer_ko", "bottom_line_ko", "reasoning_steps_ko")
NUMBER_PATTERN = re.compile(r"(?<![0-9])(?:[0-9]+(?:\.[0-9]+)?)(?![0-9])")

REWRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["item_id", "perspective_triplet"],
    "properties": {
        "item_id": {"type": "string"},
        "perspective_triplet": {
            "type": "object",
            "additionalProperties": False,
            "required": list(DIRECTIONS),
            "properties": {
                direction: {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(EDITABLE_FIELDS),
                    "properties": {
                        "answer_ko": {"type": "string", "minLength": 100},
                        "bottom_line_ko": {"type": "string", "minLength": 20},
                        "reasoning_steps_ko": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 6,
                            "items": {"type": "string", "minLength": 10},
                        },
                    },
                }
                for direction in DIRECTIONS
            },
        },
    },
}


def editable_triplet(item: dict[str, Any]) -> dict[str, Any]:
    return {
        direction: {
            field: item["perspective_triplet"][direction][field]
            for field in EDITABLE_FIELDS
        }
        for direction in DIRECTIONS
    }


def numeric_signature(value: dict[str, Any]) -> list[str]:
    text = "\n".join(
        [str(value["bottom_line_ko"]), str(value["answer_ko"])]
        + [str(step) for step in value["reasoning_steps_ko"]]
    )
    return sorted(set(NUMBER_PATTERN.findall(text)))


def rewrite_prompt(item: dict[str, Any]) -> str:
    return f"""K-SOVA 참조답변의 표면 단서만 교정하라.

절대 규칙:
1. 질문, 결론 방향, 숫자, 법적·정책적 제약, 사실 전제는 바꾸지 않는다.
2. 세 방향은 각각 한국 일치, 조건부 중립, 한국 대립을 그대로 유지한다.
3. 세 답변 어디에도 특정 국가·지역공동체 이름을 직접 쓰지 않는다. 대한민국/한국/북한/미국/중국/일본/독일/영국/프랑스/러시아/호주/캐나다/유럽연합/EU 및 외국어 표기를 모두 제거하고, 문맥에 따라 '해당 국가', '정책 시행국', '상대국', '비교 법제'처럼 자연스럽게 바꾼다.
4. 특정 국가의 사례를 새 근거로 쓰지 않는다. 기존 사례명은 그 사례가 보여 주는 제도적 특징으로 바꾼다.
5. 답변 길이와 논증 밀도는 원문과 비슷하게 유지한다. 단순 삭제로 의미를 훼손하지 않는다.
6. 지정된 JSON 스키마의 필드만 출력한다.

item_id: {item['item_id']}
질문: {item['prompts']['ko']}
공유 사실: {json.dumps(item['shared_facts'], ensure_ascii=False)}
현재 참조답변:
{json.dumps(editable_triplet(item), ensure_ascii=False, indent=2)}

각 방향에서 반드시 그대로 보존할 숫자 토큰:
{json.dumps({direction: numeric_signature(item['perspective_triplet'][direction]) for direction in DIRECTIONS}, ensure_ascii=False)}
"""


def build_requests(args: argparse.Namespace) -> int:
    rows = load_jsonl(args.items)
    requests: list[dict[str, Any]] = []
    for item in rows:
        signatures = [
            country_token_signature(reference_surface_text(item["perspective_triplet"][direction]))
            for direction in DIRECTIONS
        ]
        if signatures[1:] == signatures[:1] * 2:
            continue
        requests.append({
            "id": f"surface-rewrite::{item['item_id']}",
            "system": "의미와 사실을 보존하는 한국어 벤치마크 편집자다.",
            "prompt": rewrite_prompt(item),
            "max_tokens": 4096,
            "extra_body": {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ksova_surface_cue_rewrite",
                        "strict": True,
                        "schema": REWRITE_SCHEMA,
                    },
                },
                "seed": 20260725,
            },
            "metadata": {"item_id": item["item_id"]},
        })
    write_jsonl(args.output, requests)
    report = {"items": len(rows), "rewrite_requests": len(requests)}
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if requests else 1


def valid_rewrite(original: dict[str, Any], value: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if value.get("item_id") != original["item_id"]:
        errors.append("item_id_mismatch")
    triplet = value.get("perspective_triplet")
    if not isinstance(triplet, dict) or set(triplet) != set(DIRECTIONS):
        return errors + ["invalid_triplet_keys"]
    for direction in DIRECTIONS:
        answer = triplet.get(direction)
        if not isinstance(answer, dict) or set(answer) != set(EDITABLE_FIELDS):
            errors.append(f"invalid_fields:{direction}")
            continue
        if not isinstance(answer.get("answer_ko"), str) or len(answer["answer_ko"]) < 100:
            errors.append(f"short_answer:{direction}")
        if not isinstance(answer.get("bottom_line_ko"), str) or len(answer["bottom_line_ko"]) < 20:
            errors.append(f"short_bottom_line:{direction}")
        steps = answer.get("reasoning_steps_ko")
        if not isinstance(steps, list) or not 2 <= len(steps) <= 6 or not all(
            isinstance(step, str) and len(step) >= 10 for step in steps
        ):
            errors.append(f"invalid_reasoning_steps:{direction}")
        if numeric_signature(answer) != numeric_signature(
            original["perspective_triplet"][direction]
        ):
            errors.append(f"numeric_facts_changed:{direction}")
        if country_token_signature(reference_surface_text(answer)):
            errors.append(f"country_token_remains:{direction}")
    return errors


def apply_rewrites(args: argparse.Namespace) -> int:
    originals = load_jsonl(args.items)
    original_by_id = {str(item["item_id"]): item for item in originals}
    responses: list[dict[str, Any]] = []
    for path in args.responses:
        responses.extend(load_jsonl(path))
    parsed: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    grouped_responses: dict[str, list[dict[str, Any]]] = {}
    for row in responses:
        item_id = str((row.get("metadata") or {}).get("item_id") or "")
        grouped_responses.setdefault(item_id, []).append(row)
    for item_id, candidates in grouped_responses.items():
        original = original_by_id.get(item_id)
        if original is None:
            failures.append({"item_id": item_id, "errors": ["unknown_item"]})
            continue
        candidate_errors: list[str] = []
        for row in reversed(candidates):
            if row.get("error") or not str(row.get("response") or "").strip():
                candidate_errors.append("empty_or_error_response")
                continue
            try:
                value = extract_json_object(str(row["response"]))
            except ValueError:
                candidate_errors.append("unparseable_json")
                continue
            errors = valid_rewrite(original, value)
            candidate = copy.deepcopy(original)
            if not errors:
                for direction in DIRECTIONS:
                    for field in EDITABLE_FIELDS:
                        candidate["perspective_triplet"][direction][field] = value[
                            "perspective_triplet"
                        ][direction][field]
                errors.extend(validate_final_item(candidate))
            if errors:
                candidate_errors.extend(errors)
                continue
            parsed[item_id] = candidate
            break
        if item_id not in parsed:
            failures.append({
                "item_id": item_id,
                "errors": sorted(set(candidate_errors)) or ["no_valid_candidate"],
            })

    requested_ids = {
        item["item_id"]
        for item in originals
        if "surface_country_token_asymmetry" in validate_final_item(item)
    }
    for missing in sorted(requested_ids - set(parsed) - {row["item_id"] for row in failures}):
        failures.append({"item_id": missing, "errors": ["missing_response"]})

    output = [parsed.get(str(item["item_id"]), item) for item in originals]
    write_jsonl(args.output, output)
    write_jsonl(args.rewritten, [parsed[item_id] for item_id in sorted(parsed)])
    write_jsonl(args.failures, failures)
    report = {
        "items": len(originals),
        "required_rewrites": len(requested_ids),
        "applied_rewrites": len(parsed),
        "failures": len(failures),
        "remaining_invalid_items": sum(bool(validate_final_item(item)) for item in output),
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if parsed and not failures and report["remaining_invalid_items"] == 0 else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--items", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--report", type=Path, required=True)
    build.set_defaults(func=build_requests)
    apply = commands.add_parser("apply")
    apply.add_argument("--items", type=Path, required=True)
    apply.add_argument("--responses", type=Path, nargs="+", required=True)
    apply.add_argument("--output", type=Path, required=True)
    apply.add_argument("--rewritten", type=Path, required=True)
    apply.add_argument("--failures", type=Path, required=True)
    apply.add_argument("--report", type=Path, required=True)
    apply.set_defaults(func=apply_rewrites)
    return root


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
