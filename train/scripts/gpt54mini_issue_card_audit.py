#!/usr/bin/env python3
"""Audit Cycle08 issue-card and selected-answer internal consistency."""

from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
LEVELS = ("pass", "review", "fail")


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
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    )


def token_usage(text: str) -> dict[str, int] | None:
    rows: list[dict[str, int]] = []
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        usage = event.get("usage") if isinstance(event, dict) else None
        if isinstance(usage, dict):
            rows.append({key: int(value or 0) for key, value in usage.items()})
    if not rows:
        return None
    return rows[-1]


def schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "central_consistency": {"type": "string", "enum": list(LEVELS)},
            "question_alignment": {"type": "string", "enum": list(LEVELS)},
            "answer_alignment": {"type": "string", "enum": list(LEVELS)},
            "fact_position_separation": {"type": "string", "enum": list(LEVELS)},
            "problems": {
                "type": "array",
                "maxItems": 8,
                "items": {"type": "string", "minLength": 5, "maxLength": 500},
            },
            "recommendation": {"type": "string", "enum": ["keep", "review", "replace"]},
            "reason_ko": {"type": "string", "minLength": 30, "maxLength": 1200},
        },
        "required": [
            "central_consistency",
            "question_alignment",
            "answer_alignment",
            "fact_position_separation",
            "problems",
            "recommendation",
            "reason_ko",
        ],
        "additionalProperties": False,
    }


def prompt(issue_id: str, card: dict[str, Any], qa: list[dict[str, str]]) -> str:
    payload = {"issue_id": issue_id, "issue_card": card, "selected_question_answers": qa}
    return f"""당신은 K-SOVA Cycle08의 학습 전 데이터 감사자다. 외부 검색이나 외부 지식으로 사실의 진위를 새로 판정하지 말고, 제공된 카드와 네 질문·답변 사이의 내부 정합성만 엄격히 검사하라.

검사 항목:
1. central_consistency: verified_facts_ko 내부 또는 검증 사실과 카드의 판단핵심·허용범위 사이에 결론을 바꿀 직접 모순이 있는가.
2. question_alignment: 네 질문이 모두 같은 카드의 쟁점을 정확히 묻는가.
3. answer_alignment: 각 답변이 해당 질문에 직접 답하고 다른 사건·시기·제도를 중심 사실로 섞지 않는가.
4. fact_position_separation: 행위자의 주장(documented_positions_ko)을 검증 사실로 바꿔 쓰는 구조적 혼동이 있는가.
5. 단순한 관점 차이, 조건부 판단, 카드에 없는 일반적 배경 설명만으로 fail을 주지 않는다. 학습 신호를 오염시킬 중심 모순은 fail, 명료화가 필요하지만 학습 사용이 가능한 경우 review다.
6. recommendation은 중심 모순이 있거나 질문·답변이 다른 쟁점을 다루면 replace, 사소하지 않은 확인점만 있으면 review, 그 외 keep이다.
7. JSON만 출력한다.

{json.dumps(payload, ensure_ascii=False, indent=2)}
"""


def run_one(item_dir: Path, model: str, effort: str, timeout: int) -> dict[str, Any]:
    output = item_dir / "raw_output.json"
    events = item_dir / "codex_events.jsonl"
    stderr = item_dir / "codex_stderr.log"
    command = [
        "codex", "exec", "-m", model,
        "-c", f'model_reasoning_effort="{effort}"',
        "--ephemeral", "--ignore-user-config", "--ignore-rules",
        "--sandbox", "read-only", "--skip-git-repo-check", "--color", "never",
        "--disable", "apps", "--disable", "plugins", "--disable", "tool_suggest",
        "--disable", "multi_agent", "--disable", "browser_use", "--disable", "computer_use",
        "--output-schema", str((item_dir / "output_schema.json").resolve()),
        "--json", "-o", str(output.resolve()), "-",
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
        atomic_text(events, completed.stdout)
        atomic_text(stderr, completed.stderr)
        return {
            "issue_id": item_dir.name,
            "returncode": completed.returncode,
            "output_exists": output.is_file(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "token_usage": token_usage(completed.stdout),
        }
    except subprocess.TimeoutExpired as error:
        atomic_text(events, str(error.stdout or ""))
        atomic_text(stderr, str(error.stderr or ""))
        return {
            "issue_id": item_dir.name,
            "returncode": None,
            "timeout": True,
            "output_exists": output.is_file(),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-items", type=Path, required=True)
    parser.add_argument("--heldout-items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.out}")

    rows = read_jsonl(args.exact_items) + read_jsonl(args.heldout_items)
    by_issue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_issue[str(row["issue_id"])].append(row)
    if len(by_issue) != 16 or any(len(value) != 4 for value in by_issue.values()):
        raise ValueError("expected four selected QA rows for each of 16 issues")
    for issue_id, issue_rows in sorted(by_issue.items()):
        card_hashes = {
            hashlib.sha256(
                json.dumps(row["issue_card"], ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest()
            for row in issue_rows
        }
        if len(card_hashes) != 1:
            raise ValueError(f"{issue_id}: issue card mismatch")
        item_dir = args.out / "items" / issue_id
        qa = [
            {
                "prompt_form": str(row["prompt_form"]),
                "question_ko": str(row["question_ko"]),
                "answer_ko": str(row["reference_answer_ko"]),
            }
            for row in sorted(issue_rows, key=lambda value: value["prompt_form"])
        ]
        atomic_text(item_dir / "prompt.txt", prompt(issue_id, issue_rows[0]["issue_card"], qa))
        write_json(item_dir / "output_schema.json", schema())

    issue_dirs = sorted((args.out / "items").iterdir())
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(run_one, item_dir, args.model, args.reasoning_effort, args.timeout)
            for item_dir in issue_dirs
        ]
        calls = [future.result() for future in concurrent.futures.as_completed(futures)]
    calls.sort(key=lambda row: row["issue_id"])

    audits: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for call in calls:
        output = args.out / "items" / call["issue_id"] / "raw_output.json"
        if call.get("returncode") != 0 or not output.is_file():
            errors.append(call)
            continue
        try:
            row = json.loads(output.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            errors.append({**call, "parse_error": str(error)})
            continue
        audits.append({"issue_id": call["issue_id"], **row})
    write_jsonl(args.out / "audits.jsonl", audits)
    report = {
        "schema_version": "ksova-cycle08-issue-card-audit-1.0",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "requested": len(issue_dirs),
        "completed": len(audits),
        "errors": errors,
        "recommendations": dict(sorted(Counter(row["recommendation"] for row in audits).items())),
        "replace_issue_ids": sorted(row["issue_id"] for row in audits if row["recommendation"] == "replace"),
        "review_issue_ids": sorted(row["issue_id"] for row in audits if row["recommendation"] == "review"),
        "token_usage_sum": {
            key: sum(int((call.get("token_usage") or {}).get(key) or 0) for call in calls)
            for key in ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")
        },
        "calls": calls,
        "pass": len(audits) == 16 and not errors and not any(
            row["recommendation"] == "replace" for row in audits
        ),
    }
    write_json(args.out / "audit_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if len(audits) == 16 and not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
