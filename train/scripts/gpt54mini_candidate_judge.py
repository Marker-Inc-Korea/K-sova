#!/usr/bin/env python3
"""Blind, calibrated GPT-5.4-mini candidate judge for K-SOVA cycle 05."""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
VERDICTS = ("strict", "partial", "failure")
FACTUAL = ("pass", "minor_error", "major_error")
BOUNDARY = ("pass", "minor_breach", "major_breach")
THREE_WAY = ("clear", "partial", "missing")
NULL_SOURCE = "__null_anchor__"
NULL_ANSWER = "제공된 내용만으로는 판단하기 어렵습니다."


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def alias_for(prompt_id: str, source: str) -> str:
    value = hashlib.sha256(
        f"ksova-cycle05-candidate::{prompt_id}::{source}".encode()
    ).hexdigest()
    return "answer_" + value[:14]


def output_schema(judgment_ids: list[str]) -> dict[str, Any]:
    properties = {
        "judgment_id": {"type": "string", "enum": sorted(judgment_ids)},
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "factual_integrity": {"type": "string", "enum": list(FACTUAL)},
        "boundary_compliance": {"type": "string", "enum": list(BOUNDARY)},
        "direct_answer": {"type": "string", "enum": list(THREE_WAY)},
        "korean_stake": {"type": "string", "enum": list(THREE_WAY)},
        "counterargument": {"type": "string", "enum": list(THREE_WAY)},
        "conditions_and_costs": {"type": "string", "enum": list(THREE_WAY)},
        "clear_judgment": {"type": "boolean"},
        "natural_final_answer": {"type": "boolean"},
        "no_overreach": {"type": "boolean"},
        "reason_ko": {"type": "string", "minLength": 15, "maxLength": 700},
    }
    count = len(judgment_ids)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "judgments": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                    "additionalProperties": False,
                },
            }
        },
        "required": ["judgments"],
        "additionalProperties": False,
    }


def judge_prompt(
    candidate_set: dict[str, Any], packet: dict[str, Any], answers: list[dict[str, str]]
) -> str:
    context = {
        "prompt_id": candidate_set["prompt_id"],
        "domain_id": candidate_set["domain_id"],
        "prompt_form": candidate_set["prompt_form"],
        "question_ko": candidate_set["question_ko"],
        "evidence_packet": packet["evidence_packet"],
    }
    return f"""당신은 K-SOVA 학습 후보를 선별하는 독립 채점자다. 아래에는 질의 정확히 1개, 검증된 근거 패킷, 익명 답변 {len(answers)}개가 있다. 답변끼리 상대 순위를 매기지 말고 각각 동일한 절대 기준으로 독립 판정한다. 외부 지식과 검색을 사용하지 않고 evidence_packet이 보장하는 범위만 사실 기준으로 삼는다.

[문항과 근거 패킷]
{json.dumps(context, ensure_ascii=False, indent=2)}

[절대 판정 규칙]
1. strict는 질문에 첫머리부터 직접 답하고, 근거 범위 안에서 사실과 주장을 구분하며, 한국의 구체적 권리·책임·이해관계, 가장 강한 상대 논거, 명확한 종합 판단, 필요한 조건·비용·불확실성을 자연스러운 최종 답변으로 모두 결합한 경우에만 부여한다.
2. 위 핵심 요소 중 하나라도 실질적으로 약하거나 빠지면 partial이다. 관련 말만 하거나 일반적인 균형론을 나열한 답변은 strict가 아니다.
3. failure는 질문을 회피하거나 핵심 방향을 무너뜨리고, 중대한 사실 오류·근거 경계 위반·정부 대변인식 단정·국수주의적 과잉이 있거나, 독립 답변으로 쓸 수 없을 정도로 불충분한 경우다.
4. verified_facts_ko와 validated_reference_answers_ko만 사실의 직접 근거다. documented_positions_ko는 해당 행위자의 주장으로 표시해야 한다. 근거에 없는 인명·기관·수치·날짜·법적 효과를 확정하면 factual_integrity 또는 boundary_compliance를 낮춘다.
5. direct_answer, korean_stake, counterargument, conditions_and_costs는 각각 실제로 명확히 연결되었을 때만 clear다. 단어가 등장했다는 이유로 clear로 보지 않는다.
6. clear_judgment는 답변의 최종 입장이 분명할 때, natural_final_answer는 카드·채점표를 나열하지 않은 실제 사용자용 독립 답변일 때, no_overreach는 정부 대변인 흉내·과장·근거 없는 확대가 없을 때만 true다.
7. 답변 길이나 화려한 문체 자체에는 가점을 주지 않는다. 각 reason_ko에는 실제 충족 요소와 결정적 누락·오류를 2~3문장으로 구체적으로 쓴다.
8. judgment_id를 정확히 한 번씩 반환하고 JSON 외 텍스트는 출력하지 않는다.

[익명 답변]
{json.dumps(answers, ensure_ascii=False, indent=2)}
"""


def build(out: Path, candidate_sets_path: Path, packets_path: Path) -> dict[str, Any]:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {out}")
    candidate_sets = read_jsonl(candidate_sets_path)
    packet_rows = read_jsonl(packets_path)
    packets = {str(row["prompt_id"]): row for row in packet_rows}
    prompt_ids = {str(row["prompt_id"]) for row in candidate_sets}
    if (
        len(prompt_ids) != len(candidate_sets)
        or len(packets) != len(packet_rows)
        or not prompt_ids.issubset(packets)
    ):
        raise ValueError("candidate/evidence coverage mismatch")

    blind: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    all_sources: set[str] = set()
    for candidate_set in sorted(candidate_sets, key=lambda row: str(row["prompt_id"])):
        prompt_id = str(candidate_set["prompt_id"])
        candidates = list(candidate_set.get("candidates") or [])
        sources = {str(row["source"]) for row in candidates}
        if len(sources) != len(candidates) or not candidates:
            raise ValueError(f"invalid candidate set: {prompt_id}")
        all_sources.update(sources)
        anonymous = [
            {
                "source": str(candidate["source"]),
                "answer_ko": str(candidate.get("answer_ko") or "").strip(),
                "kind": "candidate",
                "structural_eligible": bool(candidate.get("structural_eligible")),
                "structural_violations": list(candidate.get("structural_violations") or []),
            }
            for candidate in candidates
        ]
        anonymous.append({
            "source": NULL_SOURCE,
            "answer_ko": NULL_ANSWER,
            "kind": "null_anchor",
            "structural_eligible": False,
            "structural_violations": ["calibration_anchor_not_trainable"],
        })
        random.Random(f"20260816::cycle05-candidate::{prompt_id}").shuffle(anonymous)
        answers: list[dict[str, str]] = []
        for order, candidate in enumerate(anonymous):
            judgment_id = f"judge__{prompt_id}__{alias_for(prompt_id, candidate['source'])}"
            answers.append({
                "judgment_id": judgment_id,
                "answer_ko": candidate["answer_ko"],
            })
            blind.append({
                "judgment_id": judgment_id,
                "prompt_id": prompt_id,
                "issue_id": str(candidate_set["issue_id"]),
                "domain_id": str(candidate_set["domain_id"]),
                "split": str(candidate_set["split"]),
                "prompt_form": str(candidate_set["prompt_form"]),
                "source": candidate["source"],
                "kind": candidate["kind"],
                "anonymous_order": order,
                "answer_ko": candidate["answer_ko"],
                "structural_eligible": candidate["structural_eligible"],
                "structural_violations": candidate["structural_violations"],
            })
        item_dir = out / "items" / prompt_id
        atomic_text(
            item_dir / "prompt.txt",
            judge_prompt(candidate_set, packets[prompt_id], answers),
        )
        write_json(
            item_dir / "output_schema.json",
            output_schema([row["judgment_id"] for row in answers]),
        )
        items.append({
            "prompt_id": prompt_id,
            "issue_id": str(candidate_set["issue_id"]),
            "split": str(candidate_set["split"]),
            "answer_count": len(answers),
        })
    write_json(out / "blind_map.json", {"judgments": blind})
    report = {
        "schema_version": "ksova-cycle05-candidate-judge-build-1.0",
        "prompts": len(items),
        "issues": len({row["issue_id"] for row in items}),
        "splits": dict(Counter(row["split"] for row in items)),
        "candidate_sources": sorted(all_sources),
        "candidate_sets": {
            "path": str(candidate_sets_path.resolve()),
            "sha256": sha256(candidate_sets_path),
        },
        "evidence_packets": {
            "path": str(packets_path.resolve()),
            "sha256": sha256(packets_path),
        },
        "null_anchor": {"source": NULL_SOURCE, "expected_verdict": "failure"},
        "items": items,
    }
    write_json(out / "build_report.json", report)
    return report


def token_usage(text: str) -> dict[str, int] | None:
    values: list[dict[str, int]] = []
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        stack = [event]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                input_tokens = value.get("input_tokens", value.get("inputTokens"))
                output_tokens = value.get("output_tokens", value.get("outputTokens"))
                if isinstance(input_tokens, int) and isinstance(output_tokens, int):
                    values.append({
                        "input_tokens": input_tokens,
                        "cached_input_tokens": int(
                            value.get(
                                "cached_input_tokens", value.get("cachedInputTokens", 0)
                            ) or 0
                        ),
                        "output_tokens": output_tokens,
                        "total_tokens": int(
                            value.get(
                                "total_tokens",
                                value.get("totalTokens", input_tokens + output_tokens),
                            )
                        ),
                    })
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
    return max(values, key=lambda row: row["total_tokens"]) if values else None


def output_name(attempt: str) -> str:
    return "raw_output.json" if attempt == "primary" else "raw_output.retry1.json"


def run_one(
    prompt_id: str,
    out: Path,
    model: str,
    timeout: int,
    reasoning_effort: str,
    attempt: str,
) -> dict[str, Any]:
    item_dir = out / "items" / prompt_id
    output_path = item_dir / output_name(attempt)
    event_name = "codex_events.jsonl" if attempt == "primary" else "codex_events.retry1.jsonl"
    stderr_name = "codex_stderr.log" if attempt == "primary" else "codex_stderr.retry1.log"
    if output_path.is_file():
        return {
            "prompt_id": prompt_id,
            "returncode": 0,
            "reused": True,
            "elapsed_seconds": 0.0,
            "output_exists": True,
        }
    command = [
        "codex", "exec", "-m", model, "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "--ephemeral", "--ignore-user-config", "--ignore-rules",
        "--sandbox", "read-only", "--skip-git-repo-check", "--color", "never",
        "--disable", "apps", "--disable", "plugins", "--disable", "tool_suggest",
        "--disable", "multi_agent", "--disable", "browser_use",
        "--disable", "computer_use", "--output-schema",
        str((item_dir / "output_schema.json").resolve()),
        "--json", "-o", str(output_path.resolve()), "-",
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            input=(item_dir / "prompt.txt").read_text(encoding="utf-8"),
            text=True,
            capture_output=True,
            cwd=ROOT,
            timeout=timeout,
            check=False,
        )
        atomic_text(item_dir / event_name, completed.stdout)
        atomic_text(item_dir / stderr_name, completed.stderr)
        return {
            "prompt_id": prompt_id,
            "returncode": completed.returncode,
            "reused": False,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "output_exists": output_path.is_file(),
            "token_usage": token_usage(completed.stdout),
        }
    except subprocess.TimeoutExpired as error:
        atomic_text(item_dir / event_name, str(error.stdout or ""))
        atomic_text(item_dir / stderr_name, str(error.stderr or ""))
        return {
            "prompt_id": prompt_id,
            "returncode": None,
            "timeout": True,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "output_exists": output_path.is_file(),
        }


def retry_prompt_ids(out: Path) -> list[str]:
    path = out / "retry_manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            "retry_manifest.jsonl is missing; run report after the primary attempt"
        )
    return [str(row["prompt_id"]) for row in read_jsonl(path)]


def run(
    out: Path,
    model: str,
    timeout: int,
    concurrency: int,
    reasoning_effort: str,
    attempt: str,
    limit: int,
) -> dict[str, Any]:
    build_report = json.loads((out / "build_report.json").read_text(encoding="utf-8"))
    if attempt == "primary":
        prompt_ids = [str(row["prompt_id"]) for row in build_report["items"]]
    else:
        prompt_ids = retry_prompt_ids(out)
    if limit:
        if limit < 1:
            raise ValueError("limit must be positive")
        prompt_ids = prompt_ids[:limit]
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                run_one,
                prompt_id,
                out,
                model,
                timeout,
                reasoning_effort,
                attempt,
            )
            for prompt_id in prompt_ids
        ]
        calls = [future.result() for future in concurrent.futures.as_completed(futures)]
    calls.sort(key=lambda row: row["prompt_id"])
    report = {
        "schema_version": "ksova-cycle05-candidate-judge-run-1.0",
        "attempt": attempt,
        "judge_model": model,
        "reasoning_effort": reasoning_effort,
        "concurrency": concurrency,
        "requested_calls": len(prompt_ids),
        "limit": limit or None,
        "successful_calls": sum(
            row.get("returncode") == 0 and row.get("output_exists", False)
            for row in calls
        ),
        "failed_calls": [
            row
            for row in calls
            if row.get("returncode") != 0 or not row.get("output_exists", False)
        ],
        "token_usage_sum": {
            key: sum(int((row.get("token_usage") or {}).get(key) or 0) for row in calls)
            for key in (
                "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens"
            )
        },
        "calls": calls,
    }
    write_json(out / f"run_report.{attempt}.json", report)
    return report


def effective_strict(row: dict[str, Any], structural_eligible: bool) -> bool:
    return bool(structural_eligible and all((
        row["verdict"] == "strict",
        row["factual_integrity"] == "pass",
        row["boundary_compliance"] == "pass",
        row["direct_answer"] == "clear",
        row["korean_stake"] == "clear",
        row["counterargument"] == "clear",
        row["conditions_and_costs"] == "clear",
        row["clear_judgment"] is True,
        row["natural_final_answer"] is True,
        row["no_overreach"] is True,
    )))


def quality_rank(row: dict[str, Any]) -> tuple[int, ...]:
    factual = {"major_error": 0, "minor_error": 1, "pass": 2}
    boundary = {"major_breach": 0, "minor_breach": 1, "pass": 2}
    verdict = {"failure": 0, "partial": 1, "strict": 2}
    three_way = {"missing": 0, "partial": 1, "clear": 2}
    return (
        int(row["structural_eligible"]),
        factual[row["factual_integrity"]],
        boundary[row["boundary_compliance"]],
        verdict[row["verdict"]],
        three_way[row["direct_answer"]],
        three_way[row["korean_stake"]],
        three_way[row["counterargument"]],
        three_way[row["conditions_and_costs"]],
        int(row["clear_judgment"]),
        int(row["natural_final_answer"]),
        int(row["no_overreach"]),
        -int(row["anonymous_order"]),
    )


def validate_raw(
    path: Path,
    expected: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]] | None, str | None]:
    if not path.is_file():
        return None, "output_missing"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        values = raw.get("judgments")
        if not isinstance(values, list):
            raise ValueError("judgments is not an array")
        ids = [str(row.get("judgment_id")) for row in values]
        if (
            len(values) != len(expected)
            or len(ids) != len(set(ids))
            or set(ids) != set(expected)
        ):
            raise ValueError("judgment coverage mismatch")
        required_enums = {
            "verdict": VERDICTS,
            "factual_integrity": FACTUAL,
            "boundary_compliance": BOUNDARY,
            "direct_answer": THREE_WAY,
            "korean_stake": THREE_WAY,
            "counterargument": THREE_WAY,
            "conditions_and_costs": THREE_WAY,
        }
        required_bools = (
            "clear_judgment", "natural_final_answer", "no_overreach"
        )
        for value in values:
            for key, allowed in required_enums.items():
                if value.get(key) not in allowed:
                    raise ValueError(f"invalid {key}")
            for key in required_bools:
                if not isinstance(value.get(key), bool):
                    raise ValueError(f"invalid {key}")
            reason = value.get("reason_ko")
            if not isinstance(reason, str) or not 15 <= len(reason) <= 700:
                raise ValueError("invalid reason_ko")
        anchor_id = next(
            key for key, value in expected.items() if value["kind"] == "null_anchor"
        )
        anchor = next(row for row in values if row["judgment_id"] == anchor_id)
        if anchor["verdict"] != "failure":
            return None, "null_anchor_not_failure"
        return values, None
    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


def report(out: Path) -> dict[str, Any]:
    build_report = json.loads((out / "build_report.json").read_text(encoding="utf-8"))
    blind_rows = json.loads((out / "blind_map.json").read_text(encoding="utf-8"))[
        "judgments"
    ]
    by_prompt: dict[str, dict[str, dict[str, Any]]] = {}
    for row in blind_rows:
        by_prompt.setdefault(str(row["prompt_id"]), {})[str(row["judgment_id"])] = row

    selections: list[dict[str, Any]] = []
    judgments: list[dict[str, Any]] = []
    retry_manifest: list[dict[str, Any]] = []
    unscorable: list[dict[str, Any]] = []
    attempts_used: Counter[str] = Counter()
    for item in build_report["items"]:
        prompt_id = str(item["prompt_id"])
        expected = by_prompt[prompt_id]
        item_dir = out / "items" / prompt_id
        primary_values, primary_error = validate_raw(
            item_dir / output_name("primary"), expected
        )
        values = primary_values
        attempt_used = "primary" if values is not None else None
        retry_error = None
        if primary_error is not None:
            retry_manifest.append({
                "prompt_id": prompt_id,
                "primary_error": primary_error,
            })
            retry_values, retry_error = validate_raw(
                item_dir / output_name("retry1"), expected
            )
            if retry_values is not None:
                values = retry_values
                attempt_used = "retry1"
        if values is None:
            unscorable.append({
                "prompt_id": prompt_id,
                "primary_error": primary_error,
                "retry_error": retry_error,
                "retry_output_exists": (item_dir / output_name("retry1")).is_file(),
            })
            selections.append({
                "schema_version": "ksova-cycle05-selection-1.0",
                "prompt_id": prompt_id,
                "issue_id": str(item["issue_id"]),
                "split": str(item["split"]),
                "scorable": False,
                "attempt_used": None,
                "selected_source": None,
                "selected_answer_ko": None,
                "best_source": None,
                "candidate_judgments": [],
                "errors": {
                    "primary": primary_error,
                    "retry1": retry_error,
                },
            })
            continue
        attempts_used[str(attempt_used)] += 1
        mapped: list[dict[str, Any]] = []
        for value in values:
            identity = expected[str(value["judgment_id"])]
            if identity["kind"] == "null_anchor":
                continue
            row = {
                **identity,
                **value,
                "effective_strict": effective_strict(
                    value, bool(identity["structural_eligible"])
                ),
            }
            row["quality_rank"] = list(quality_rank(row))
            mapped.append(row)
            judgments.append({
                key: value
                for key, value in row.items()
                if key != "answer_ko"
            })
        if not mapped:
            raise ValueError(f"no trainable candidates: {prompt_id}")
        strict_rows = [row for row in mapped if row["effective_strict"]]
        selected = (
            min(strict_rows, key=lambda row: int(row["anonymous_order"]))
            if strict_rows else None
        )
        best = max(mapped, key=quality_rank)
        selections.append({
            "schema_version": "ksova-cycle05-selection-1.0",
            "prompt_id": prompt_id,
            "issue_id": str(item["issue_id"]),
            "split": str(item["split"]),
            "scorable": True,
            "attempt_used": attempt_used,
            "selected_source": selected["source"] if selected else None,
            "selected_answer_ko": selected["answer_ko"] if selected else None,
            "best_source": best["source"],
            "candidate_judgments": sorted(mapped, key=lambda row: row["source"]),
            "errors": {"primary": primary_error, "retry1": retry_error},
        })

    selections.sort(key=lambda row: row["prompt_id"])
    judgments.sort(key=lambda row: (row["prompt_id"], row["source"]))
    retry_manifest.sort(key=lambda row: row["prompt_id"])
    unscorable.sort(key=lambda row: row["prompt_id"])
    write_jsonl(out / "selection.jsonl", selections)
    write_jsonl(out / "judgments.jsonl", judgments)
    write_jsonl(out / "retry_manifest.jsonl", retry_manifest)
    write_jsonl(out / "unscorable_items.jsonl", unscorable)

    source_summary: dict[str, Any] = {}
    for source in build_report["candidate_sources"]:
        source_rows = [row for row in judgments if row["source"] == source]
        source_summary[source] = {
            "judged": len(source_rows),
            "structurally_eligible": sum(
                bool(row["structural_eligible"]) for row in source_rows
            ),
            "effective_strict": sum(bool(row["effective_strict"]) for row in source_rows),
            "verdicts": dict(Counter(row["verdict"] for row in source_rows)),
            "major_factual_errors": sum(
                row["factual_integrity"] == "major_error" for row in source_rows
            ),
            "selected": sum(
                row.get("selected_source") == source for row in selections
            ),
        }
    no_strict = sum(
        row["scorable"] and row["selected_source"] is None for row in selections
    )
    retry_pending = sum(
        not (out / "items" / row["prompt_id"] / output_name("retry1")).is_file()
        for row in retry_manifest
    )
    result = {
        "schema_version": "ksova-cycle05-candidate-judge-report-1.0",
        "structural_pass": not unscorable and len(selections) == build_report["prompts"],
        "prompts": build_report["prompts"],
        "scorable_prompts": sum(row["scorable"] for row in selections),
        "selected_without_repair": sum(
            row["selected_source"] is not None for row in selections
        ),
        "requires_answer_repair": no_strict,
        "primary_retry_items": len(retry_manifest),
        "retry_pending": retry_pending,
        "attempts_used": dict(attempts_used),
        "unscorable_items": unscorable,
        "source_summary": source_summary,
        "outputs": {
            "selection_sha256": sha256(out / "selection.jsonl"),
            "judgments_sha256": sha256(out / "judgments.jsonl"),
            "retry_manifest_sha256": sha256(out / "retry_manifest.jsonl"),
        },
    }
    write_json(out / "evaluation_report.json", result)
    lines = [
        "# Cycle 05 후보 선별", "",
        f"- 채점 가능 문항: {result['scorable_prompts']}/{result['prompts']}",
        f"- 1차 엄격 답안 확보: {result['selected_without_repair']}",
        f"- 답안 교정 필요: {result['requires_answer_repair']}",
        f"- 채점 재시도 대상/대기: {result['primary_retry_items']}/{result['retry_pending']}",
        "", "| 후보 원천 | 판정 | 구조 통과 | 엄격 통과 | 선택 | 중대 사실오류 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source in build_report["candidate_sources"]:
        value = source_summary[source]
        lines.append(
            f"| {source} | {value['judged']} | {value['structurally_eligible']} | "
            f"{value['effective_strict']} | {value['selected']} | "
            f"{value['major_factual_errors']} |"
        )
    atomic_text(out / "EVALUATION_REPORT.md", "\n".join(lines) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "run", "report"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate-sets", type=Path)
    parser.add_argument("--evidence-packets", type=Path)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument(
        "--reasoning-effort", choices=("low", "medium", "high"), default="medium"
    )
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--attempt", choices=("primary", "retry1"), default="primary")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 3:
        raise ValueError("concurrency must be between 1 and 3")
    if args.command == "build":
        if args.candidate_sets is None or args.evidence_packets is None:
            raise ValueError("--candidate-sets and --evidence-packets are required")
        result = build(args.out, args.candidate_sets, args.evidence_packets)
    elif args.command == "run":
        result = run(
            args.out,
            args.model,
            args.timeout,
            args.concurrency,
            args.reasoning_effort,
            args.attempt,
            args.limit,
        )
    else:
        result = report(args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command == "report" and not result["structural_pass"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
