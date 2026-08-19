#!/usr/bin/env python3
"""Run source-closed JSON requests through isolated Claude Code print calls.

Each request is an independent process with tools disabled.  A journal keeps
completed calls recoverable during interruption; the final JSONL is compacted
to exactly one row per request so downstream duplicate checks remain strict.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


OBJECT_SCHEMA = json.dumps(
    {"type": "object"}, ensure_ascii=False, separators=(",", ":")
)

KSOVA_VALIDATION_CHECKS = (
    "sources_relevant",
    "claims_body_supported",
    "attribution_grounded",
    "external_difference_real",
    "korean_stake_concrete",
    "korean_discourse_grounded",
    "not_government_mimic",
    "acceptable_range_coherent",
    "atomic_decision",
    "domain_assignment_correct",
    "ksova_construct_fit",
    "not_general_knowledge",
    "temporally_scoped",
    "novel_from_catalog",
    "no_hallucination",
)

KSOVA_FINAL_VALIDATION_SCHEMA = json.dumps(
    {
        "type": "object",
        "properties": {
            "issue_id": {"type": "string"},
            "source_assessments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string"},
                        "usable": {"type": "boolean"},
                        "supported_claims_ko": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "unsupported_candidate_claims_ko": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "source_id", "usable", "supported_claims_ko",
                        "unsupported_candidate_claims_ko",
                    ],
                    "additionalProperties": False,
                },
            },
            "accepted_source_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "checks": {
                "type": "object",
                "properties": {
                    key: {"type": "boolean"} for key in KSOVA_VALIDATION_CHECKS
                },
                "required": list(KSOVA_VALIDATION_CHECKS),
                "additionalProperties": False,
            },
            "verdict": {
                "type": "string",
                "enum": ["pass", "needs_source", "reject"],
            },
            "quality_score": {"type": "number", "minimum": 0, "maximum": 100},
            "fatal_errors_ko": {
                "type": "array",
                "items": {"type": "string"},
            },
            "missing_evidence_ko": {
                "type": "array",
                "items": {"type": "string"},
            },
            "reason_ko": {"type": "string"},
        },
        "required": [
            "issue_id", "source_assessments", "accepted_source_ids", "checks",
            "verdict", "quality_score", "fatal_errors_ko",
            "missing_evidence_ko", "reason_ko",
        ],
        "additionalProperties": False,
    },
    ensure_ascii=False,
    separators=(",", ":"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"non-object JSONL row at {path}:{line_no}")
        rows.append(row)
    return rows


def messages_and_hash(row: dict[str, Any]) -> tuple[str, str, str]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("each request must contain one system and one user message")
    normalized = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]
    if [message["role"] for message in normalized] != ["system", "user"]:
        raise ValueError("message roles must be system, user")
    digest = hashlib.sha256(
        json.dumps(
            normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return normalized[0]["content"], normalized[1]["content"], digest


def reusable(
    result: dict[str, Any], row: dict[str, Any], record_model_name: str
) -> bool:
    try:
        _system, _user, digest = messages_and_hash(row)
    except (KeyError, TypeError, ValueError):
        return False
    return all((
        result.get("id") == row.get("id"),
        result.get("model") == record_model_name,
        result.get("messages_sha256") == digest,
        result.get("metadata", {}) == row.get("metadata", {}),
        not result.get("error"),
        bool(str(result.get("response") or "").strip()),
        result.get("finish_reason") == "stop",
    ))


def command(args: argparse.Namespace, system_prompt: str) -> list[str]:
    schema = (
        KSOVA_FINAL_VALIDATION_SCHEMA
        if args.schema_kind == "ksova-final-validation"
        else OBJECT_SCHEMA
    )
    return [
        args.claude_bin,
        "-p",
        "--model", args.model,
        "--effort", args.effort,
        "--tools", "",
        "--safe-mode",
        "--disable-slash-commands",
        "--system-prompt", system_prompt,
        "--no-session-persistence",
        "--permission-mode", "dontAsk",
        "--prompt-suggestions", "false",
        "--output-format", "json",
        "--json-schema", schema,
    ]


def normalize_structured_output(value: dict[str, Any]) -> dict[str, Any]:
    """Unwrap generic-schema artifacts while preserving the model's content."""
    if len(value) == 1:
        only_value = next(iter(value.values()))
        if isinstance(only_value, str):
            try:
                nested = json.loads(only_value)
            except json.JSONDecodeError:
                nested = None
            if isinstance(nested, dict):
                value = nested
    for key, item in list(value.items()):
        if not isinstance(item, str) or not item.lstrip().startswith(("[", "{")):
            continue
        try:
            decoded = json.loads(item)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, (dict, list)):
            value[key] = decoded
    return value


def run_one(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    try:
        system_prompt, user_prompt, digest = messages_and_hash(row)
    except (KeyError, TypeError, ValueError) as exc:
        return {
            "id": row.get("id"), "model": args.record_model_name,
            "response": "", "finish_reason": "error", "usage": None,
            "messages_sha256": None, "metadata": row.get("metadata", {}),
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }

    error = None
    stderr = ""
    for attempt in range(args.retries + 1):
        try:
            completed = subprocess.run(
                command(args, system_prompt),
                input=user_prompt,
                text=True,
                capture_output=True,
                cwd=args.workdir,
                timeout=args.timeout,
                check=False,
            )
            stderr = completed.stderr.strip()
            if completed.returncode != 0:
                error = f"claude_exit_{completed.returncode}: {stderr[-2000:]}"
            else:
                outer = json.loads(completed.stdout)
                structured = outer.get("structured_output")
                if not isinstance(structured, dict):
                    structured = json.loads(str(outer.get("result") or ""))
                if not isinstance(structured, dict):
                    raise ValueError("Claude structured output is not an object")
                structured = normalize_structured_output(structured)
                # Generic JSON-schema output occasionally serializes the
                # request id one extra time (for example ``\"request-id\"``).
                # Normalize only when decoding that single string yields the
                # exact immutable request id; never repair an unknown id.
                observed_request_id = structured.get("request_id")
                if (
                    isinstance(observed_request_id, str)
                    and observed_request_id != row["id"]
                ):
                    try:
                        decoded_request_id = json.loads(observed_request_id)
                    except json.JSONDecodeError:
                        decoded_request_id = None
                    if decoded_request_id == row["id"]:
                        structured["request_id"] = decoded_request_id
                return {
                    "id": row["id"], "model": args.record_model_name,
                    "response": json.dumps(structured, ensure_ascii=False),
                    "reasoning_content": "", "usage": outer.get("usage"),
                    "finish_reason": "stop",
                    "requested_max_tokens": row.get("max_tokens"),
                    "generation_config": {
                        "provider": "claude_code_print",
                        "model": args.model,
                        "effort": args.effort,
                        "tools": [],
                        "structured_output": True,
                        "schema_kind": args.schema_kind,
                    },
                    "messages_sha256": digest,
                    "metadata": row.get("metadata", {}), "error": None,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "provider_result": {
                        key: outer.get(key)
                        for key in (
                            "duration_api_ms", "total_cost_usd", "session_id",
                            "service_tier", "modelUsage",
                        )
                        if outer.get(key) is not None
                    },
                }
        except (
            subprocess.TimeoutExpired, json.JSONDecodeError, OSError,
            KeyError, TypeError, ValueError,
        ) as exc:
            error = f"{type(exc).__name__}: {exc}"
        if attempt < args.retries:
            time.sleep(1.5 * (attempt + 1))
    return {
        "id": row.get("id"), "model": args.record_model_name,
        "response": "", "reasoning_content": "", "usage": None,
        "finish_reason": "error", "requested_max_tokens": row.get("max_tokens"),
        "generation_config": {
            "provider": "claude_code_print", "model": args.model,
            "effort": args.effort, "tools": [], "structured_output": True,
            "schema_kind": args.schema_kind,
        },
        "messages_sha256": digest, "metadata": row.get("metadata", {}),
        "error": error or stderr or "unknown Claude failure",
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path)
    parser.add_argument("--claude-bin", default="claude")
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--record-model-name", default="claude-sonnet-5__source-auditor")
    parser.add_argument(
        "--schema-kind",
        choices=("generic", "ksova-final-validation"),
        default="generic",
        help="Use a strict response schema for a known pipeline stage.",
    )
    parser.add_argument(
        "--request-id", action="append", default=[],
        help="Run only the named request id; repeat to select multiple requests.",
    )
    parser.add_argument("--effort", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--workdir", type=Path, default=Path.cwd())
    args = parser.parse_args()

    requests = read_jsonl(args.requests)
    if args.request_id:
        selected = set(args.request_id)
        known = {row.get("id") for row in requests}
        missing = selected - known
        if missing:
            raise ValueError(f"unknown request ids: {sorted(missing)}")
        requests = [row for row in requests if row.get("id") in selected]
    if len({row.get("id") for row in requests}) != len(requests):
        raise ValueError("request ids must be unique")
    progress_path = args.progress_file or args.output.with_suffix(
        args.output.suffix + ".progress"
    )
    prior_by_id: dict[str, dict[str, Any]] = {}
    for result in (*read_jsonl(args.output), *read_jsonl(progress_path)):
        request = next((row for row in requests if row.get("id") == result.get("id")), None)
        if request is not None and reusable(result, request, args.record_model_name):
            prior_by_id[result["id"]] = result

    pending = [row for row in requests if row["id"] not in prior_by_id]
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    generated = 0
    with progress_path.open("a", encoding="utf-8") as journal:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(run_one, row, args): row for row in pending}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                with lock:
                    journal.write(json.dumps(result, ensure_ascii=False) + "\n")
                    journal.flush()
                    os.fsync(journal.fileno())
                prior_by_id[result["id"]] = result
                generated += 1
                print(
                    f"[{len(prior_by_id)}/{len(requests)}] {result['id']} "
                    f"error={bool(result.get('error'))}",
                    flush=True,
                )

    final_rows = [prior_by_id[row["id"]] for row in requests if row["id"] in prior_by_id]
    atomic_jsonl(args.output, final_rows)
    if len(final_rows) == len(requests) and all(not row.get("error") for row in final_rows):
        progress_path.unlink(missing_ok=True)
    summary = {
        "requests": len(requests), "completed": len(final_rows),
        "resumed": len(requests) - len(pending), "generated": generated,
        "errors": sum(bool(row.get("error")) for row in final_rows),
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["completed"] == len(requests) and not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
