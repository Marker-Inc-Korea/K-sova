#!/usr/bin/env python3
"""Run a bounded, resumable JSONL batch against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                row.get("id")
                and not row.get("error")
                and row.get("response")
                and row.get("finish_reason") == "stop"
            ):
                done.add(str(row["id"]))
    return done


def call(
    row: dict[str, Any],
    args: argparse.Namespace,
    template: dict[str, Any] | None,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    system_content = str(row.get("system", "You are a careful assistant."))
    prompt_content = str(row["prompt"])
    system_suffix = str(getattr(args, "system_suffix", "") or "")
    prompt_suffix = str(getattr(args, "prompt_suffix", "") or "")
    if system_suffix:
        system_content += "\n\n" + system_suffix
    if prompt_suffix:
        prompt_content += "\n\n" + prompt_suffix
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt_content},
        ],
        "temperature": args.temperature,
        "max_tokens": int(
            args.force_max_tokens
            if args.force_max_tokens is not None
            else row.get("max_tokens", args.max_tokens)
        ),
    }
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if template:
        payload["chat_template_kwargs"] = template
    if extra:
        payload.update(extra)
    if isinstance(row.get("extra_body"), dict):
        payload.update(row["extra_body"])
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    started = time.monotonic()
    error = None
    last_result: dict[str, Any] | None = None
    for attempt in range(args.retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            choice = result["choices"][0]
            message = choice["message"]
            last_result = {
                "id": row["id"],
                "model": args.record_model,
                "response": message.get("content") or "",
                "reasoning_content": message.get("reasoning_content") or message.get("reasoning") or "",
                "usage": result.get("usage"),
                "finish_reason": choice.get("finish_reason"),
                "metadata": row.get("metadata", {}),
                "error": None,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            if choice.get("finish_reason") == "stop":
                return last_result
            error = f"NonStopFinishReason: {choice.get('finish_reason') or 'missing'}"
            if attempt < args.retries:
                time.sleep(2 * (attempt + 1))
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
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
                time.sleep(2 * (attempt + 1))
    if last_result is not None:
        last_result["error"] = error
        last_result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        return last_result
    return {
        "id": row["id"],
        "model": args.record_model,
        "response": "",
        "reasoning_content": "",
        "usage": None,
        "finish_reason": None,
        "metadata": row.get("metadata", {}),
        "error": error,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--record-model", required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--force-max-tokens", type=int)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--chat-template-kwargs-json")
    parser.add_argument("--extra-body-json")
    parser.add_argument("--system-suffix", default="")
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--completed-from", type=Path, action="append", default=[])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = load_jsonl(args.requests)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be in [0, shard-count)")
    rows = [row for index, row in enumerate(rows) if index % args.shard_count == args.shard_index]
    if args.limit is not None:
        rows = rows[:args.limit]
    done = completed_ids(args.output)
    for path in args.completed_from:
        done.update(completed_ids(path))
    previously_completed = sum(str(row["id"]) in done for row in rows)
    rows = [row for row in rows if str(row["id"]) not in done]
    template = json.loads(args.chat_template_kwargs_json) if args.chat_template_kwargs_json else None
    extra = json.loads(args.extra_body_json) if args.extra_body_json else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = errors = 0
    with args.output.open("a", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(call, row, args, template, extra) for row in rows]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                completed += 1
                errors += int(bool(result.get("error")))
                if completed == 1 or completed % 25 == 0 or completed == len(rows):
                    print(f"[{completed}/{len(rows)}] errors={errors}", flush=True)
    summary = {
        "requested": len(rows),
        "previously_completed": previously_completed,
        "completed": completed,
        "errors": errors,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
    }
    args.output.with_suffix(args.output.suffix + ".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
