#!/usr/bin/env python3
"""Run teacher or user-only evaluation requests against a local chat endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def request_messages(row: dict[str, Any], system_prefix: str) -> list[dict[str, str]]:
    if "messages" in row:
        messages = row["messages"]
        if system_prefix:
            raise ValueError("system prefix cannot be used with prebuilt messages")
    else:
        system = row.get("system", "You are a careful assistant.")
        if system_prefix:
            system = system_prefix + "\n" + system
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": row["prompt"]},
        ]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    normalized = [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages
    ]
    if row.get("metadata", {}).get("purpose") == "model_evaluation":
        if len(normalized) != 1 or normalized[0]["role"] != "user":
            raise ValueError("model_evaluation requests must contain exactly one user message")
    return normalized


def request_spec(
    row: dict[str, Any], args: argparse.Namespace,
    template_kwargs: dict[str, Any] | None, extra: dict[str, Any] | None,
) -> tuple[list[dict[str, str]], int, dict[str, Any], str, dict[str, Any]]:
    messages = request_messages(row, args.system_prefix)
    requested_max_tokens = (
        args.force_max_tokens
        if args.force_max_tokens is not None
        else row.get("max_tokens", args.max_tokens)
    )
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": requested_max_tokens,
    }
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if template_kwargs:
        payload["chat_template_kwargs"] = template_kwargs
    if extra:
        payload.update(extra)
    generation_config = {
        key: value for key, value in payload.items() if key not in {"model", "messages"}
    }
    messages_hash = hashlib.sha256(
        json.dumps(
            messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return messages, requested_max_tokens, payload, messages_hash, generation_config


def call(
    row: dict[str, Any], args: argparse.Namespace,
    template_kwargs: dict[str, Any] | None, extra: dict[str, Any] | None,
) -> dict[str, Any]:
    started = time.monotonic()
    try:
        messages, requested_max_tokens, payload, messages_hash, generation_config = request_spec(
            row, args, template_kwargs, extra
        )
    except (KeyError, TypeError, ValueError) as exc:
        return failure(row, args, f"{type(exc).__name__}: {exc}", started, None, None)
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = args.base_url.rstrip("/") + "/chat/completions"
    error = None
    for attempt in range(args.retries + 1):
        try:
            request = urllib.request.Request(
                url, data=body, method="POST",
                headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            )
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            choice = result["choices"][0]
            message = choice["message"]
            return {
                "id": row["id"], "model": args.record_model_name or args.model,
                "response": message.get("content") or "",
                "reasoning_content": message.get("reasoning_content") or message.get("reasoning") or "",
                "usage": result.get("usage"), "finish_reason": choice.get("finish_reason"),
                "requested_max_tokens": requested_max_tokens,
                "generation_config": generation_config, "messages_sha256": messages_hash,
                "metadata": row.get("metadata", {}), "error": None,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            OSError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            detail = ""
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    detail = exc.read().decode("utf-8", errors="replace").strip()
                except OSError:
                    detail = ""
            error = f"{type(exc).__name__}: {exc}"
            if detail:
                error += f"; response_body={detail[:2000]}"
            if attempt < args.retries:
                time.sleep(1.5 * (attempt + 1))
    return failure(row, args, error, started, requested_max_tokens, messages_hash, generation_config)


def reusable_result(
    result: dict[str, Any], row: dict[str, Any], args: argparse.Namespace,
    template_kwargs: dict[str, Any] | None, extra: dict[str, Any] | None,
) -> bool:
    try:
        _, requested_max_tokens, _, messages_hash, generation_config = request_spec(
            row, args, template_kwargs, extra
        )
    except (KeyError, TypeError, ValueError):
        return False
    previous_generation_config = result.get("generation_config") or {}
    same_generation_config = previous_generation_config == generation_config
    if args.resume_allow_higher_max_tokens and not same_generation_config:
        previous_without_limit = {
            key: value for key, value in previous_generation_config.items()
            if key != "max_tokens"
        }
        current_without_limit = {
            key: value for key, value in generation_config.items()
            if key != "max_tokens"
        }
        previous_limit = result.get("requested_max_tokens")
        usage = result.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        limit_compatible = bool(
            isinstance(previous_limit, int)
            and isinstance(requested_max_tokens, int)
            and (
                previous_limit <= requested_max_tokens
                or (
                    isinstance(completion_tokens, int)
                    and completion_tokens <= requested_max_tokens
                )
            )
        )
        same_generation_config = all((
            isinstance(previous_limit, int),
            isinstance(requested_max_tokens, int),
            limit_compatible,
            previous_generation_config.get("max_tokens") == previous_limit,
            generation_config.get("max_tokens") == requested_max_tokens,
            previous_without_limit == current_without_limit,
        ))
    return all((
        result.get("id") == row.get("id"),
        not result.get("error"),
        bool(str(result.get("response") or "").strip()),
        str(result.get("finish_reason") or "").lower() == "stop",
        result.get("model") == (args.record_model_name or args.model),
        result.get("metadata", {}) == row.get("metadata", {}),
        result.get("messages_sha256") == messages_hash,
        same_generation_config,
    ))


def failure(
    row: dict[str, Any], args: argparse.Namespace, error: str | None, started: float,
    requested_max_tokens: int | None, messages_hash: str | None,
    generation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": row.get("id"), "model": args.record_model_name or args.model,
        "response": "", "reasoning_content": "", "usage": None, "finish_reason": None,
        "requested_max_tokens": requested_max_tokens, "generation_config": generation_config or {},
        "messages_sha256": messages_hash, "metadata": row.get("metadata", {}),
        "error": error, "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--record-model-name")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-prefix", default="responses")
    parser.add_argument("--output-file", type=Path)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--force-max-tokens", type=int)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--chat-template-kwargs-json")
    parser.add_argument("--extra-body-json")
    parser.add_argument("--system-prefix", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-allow-higher-max-tokens", action="store_true")
    args = parser.parse_args()
    rows = load_jsonl(args.requests)
    if args.limit is not None:
        rows = rows[:args.limit]
    template_kwargs = json.loads(args.chat_template_kwargs_json) if args.chat_template_kwargs_json else None
    extra = json.loads(args.extra_body_json) if args.extra_body_json else None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.output_file:
        out_path = args.output_file
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = args.out_dir / f"{args.out_prefix}_{stamp}.jsonl"
    progress_path = out_path.with_suffix(out_path.suffix + ".progress")
    if not args.resume:
        progress_path.unlink(missing_ok=True)
    reusable: dict[str, dict[str, Any]] = {}
    if args.resume:
        previous: list[dict[str, Any]] = []
        if out_path.exists():
            previous.extend(load_jsonl(out_path))
        if progress_path.exists():
            previous.extend(load_jsonl(progress_path))
        row_by_id = {str(row.get("id")): row for row in rows}
        for result in previous:
            result_id = str(result.get("id"))
            row = row_by_id.get(result_id)
            if row is not None and reusable_result(
                result, row, args, template_kwargs, extra
            ):
                reusable[result_id] = result
    pending = [row for row in rows if str(row.get("id")) not in reusable]
    completed = len(reusable)
    result_by_id = dict(reusable)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as journal:
        if reusable:
            print(f"[resume {len(reusable)}/{len(rows)}] verified reusable responses", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(call, row, args, template_kwargs, extra): row
                for row in pending
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                journal.write(json.dumps(result, ensure_ascii=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                result_by_id[str(result["id"])] = result
                completed += 1
                print(f"[{completed}/{len(rows)}] {result['id']} error={bool(result.get('error'))}", flush=True)
    final_rows = [
        result_by_id[str(row["id"])]
        for row in rows
        if str(row["id"]) in result_by_id
    ]
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for result in final_rows:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, out_path)
    errors = sum(bool(result.get("error")) for result in final_rows)
    incomplete = sum(
        not result.get("error")
        and str(result.get("finish_reason") or "").casefold() != "stop"
        for result in final_rows
    )
    if len(final_rows) == len(rows) and not errors and not incomplete:
        progress_path.unlink(missing_ok=True)
    summary = {
        "output": str(out_path), "requests": len(rows), "completed": len(final_rows),
        "resumed": len(reusable), "generated": len(pending), "errors": errors,
        "incomplete": incomplete,
    }
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 1 if errors or incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
