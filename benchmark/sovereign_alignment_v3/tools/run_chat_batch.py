#!/usr/bin/env python3
"""Run generic JSONL chat requests against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
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


def call(request_row: dict[str, Any], args: argparse.Namespace, template_kwargs: dict[str, Any] | None, extra: dict[str, Any] | None) -> dict[str, Any]:
    system = request_row.get("system", "You are a careful assistant.")
    if args.system_prefix:
        system = args.system_prefix + "\n" + system
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": request_row["prompt"]},
        ],
        "temperature": args.temperature,
        "max_tokens": args.force_max_tokens if args.force_max_tokens is not None else request_row.get("max_tokens", args.max_tokens),
    }
    if args.top_p is not None:
        payload["top_p"] = args.top_p
    if template_kwargs:
        payload["chat_template_kwargs"] = template_kwargs
    if extra:
        payload.update(extra)
    generation_config = {
        key: value for key, value in payload.items()
        if key not in {"model", "messages"}
    }
    if args.system_prefix:
        generation_config["system_prefix"] = args.system_prefix
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = args.base_url.rstrip("/") + "/chat/completions"
    error = None
    started = time.monotonic()
    for attempt in range(args.retries + 1):
        try:
            req = urllib.request.Request(
                url, data=body, method="POST",
                headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
            )
            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            choice = result["choices"][0]
            message = choice["message"]
            return {
                "id": request_row["id"],
                "model": args.record_model_name or args.model,
                "response": message.get("content") or "",
                "reasoning_content": message.get("reasoning_content") or message.get("reasoning") or "",
                "usage": result.get("usage"),
                "finish_reason": choice.get("finish_reason"),
                "requested_max_tokens": payload["max_tokens"],
                "generation_config": generation_config,
                "metadata": request_row.get("metadata", {}),
                "error": None,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt < args.retries:
                time.sleep(1.5 * (attempt + 1))
    return {
        "id": request_row["id"], "model": args.record_model_name or args.model, "response": "", "reasoning_content": "",
        "usage": None, "finish_reason": None, "requested_max_tokens": payload["max_tokens"],
        "generation_config": generation_config,
        "metadata": request_row.get("metadata", {}), "error": error,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--record-model-name")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-prefix", default="responses")
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
    args = parser.parse_args()
    requests = load_jsonl(args.requests)
    if args.limit is not None:
        requests = requests[:args.limit]
    template_kwargs = json.loads(args.chat_template_kwargs_json) if args.chat_template_kwargs_json else None
    extra = json.loads(args.extra_body_json) if args.extra_body_json else None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out_dir / f"{args.out_prefix}_{stamp}.jsonl"
    completed = 0
    errors = 0
    with out_path.open("w", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(call, row, args, template_kwargs, extra): row for row in requests}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                completed += 1
                errors += int(bool(result.get("error")))
                print(f"[{completed}/{len(requests)}] {result['id']} error={bool(result.get('error'))}", flush=True)
    summary = {"output": str(out_path), "requests": len(requests), "completed": completed, "errors": errors}
    (out_path.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
