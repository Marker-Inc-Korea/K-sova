#!/usr/bin/env python3
"""Run deterministic text-only Qwen3.5 generation with Transformers/PEFT.

This is a diagnostic backend for checking that an adapter's behavior is not a
vLLM-specific artifact.  It consumes the same request JSONL shape as the local
vLLM batch runner and writes ID-aligned response JSONL without retries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def digest_messages(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--only-id", action="append", default=[])
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    requests = read_jsonl(args.requests)
    if args.only_id:
        requested = set(args.only_id)
        requests = [row for row in requests if str(row["id"]) in requested]
        found = {str(row["id"]) for row in requests}
        if found != requested:
            raise ValueError(f"unknown --only-id values: {sorted(requested - found)}")
    ids = [str(row["id"]) for row in requests]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("requests must contain at least one unique id")

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, set_seed

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(args.model_path),
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.to(torch.device("cuda")).eval()
    model.config.use_cache = True

    results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(requests), args.batch_size):
            batch = requests[start : start + args.batch_size]
            prompt_texts: list[str] = []
            batch_messages: list[list[dict[str, str]]] = []
            for request in batch:
                messages = request.get("messages") or []
                if not messages:
                    raise ValueError(f"{request['id']}: missing messages")
                batch_messages.append(messages)
                try:
                    prompt_texts.append(
                        tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    )
                except TypeError:
                    prompt_texts.append(
                        tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    )
            encoded = tokenizer(
                prompt_texts,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
            )
            encoded = {key: value.to(torch.device("cuda")) for key, value in encoded.items()}
            prompt_width = int(encoded["input_ids"].shape[-1])
            prompt_counts = [int(value) for value in encoded["attention_mask"].sum(dim=1)]
            started = time.monotonic()
            output = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            batch_elapsed = round(time.monotonic() - started, 3)
            for local_index, request in enumerate(batch):
                token_ids = output[local_index, prompt_width:].tolist()
                ended_with_eos = False
                if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in token_ids:
                    token_ids = token_ids[: token_ids.index(tokenizer.eos_token_id) + 1]
                    ended_with_eos = True
                completion_tokens = len(token_ids)
                response = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                prompt_tokens = prompt_counts[local_index]
                results.append(
                    {
                        "id": str(request["id"]),
                        "model": args.model_name,
                        "response": response,
                        "reasoning_content": "",
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                        "finish_reason": "stop" if ended_with_eos else "length",
                        "requested_max_tokens": args.max_new_tokens,
                        "generation_config": {
                            "backend": "transformers_peft",
                            "do_sample": False,
                            "max_new_tokens": args.max_new_tokens,
                            "batch_size": args.batch_size,
                            "enable_thinking": False,
                            "seed": args.seed,
                        },
                        "messages_sha256": digest_messages(batch_messages[local_index]),
                        "metadata": request.get("metadata") or {},
                        "error": None,
                        "elapsed_seconds": batch_elapsed,
                    }
                )
                print(
                    f"[{start + local_index + 1}/{len(requests)}] {request['id']} "
                    f"tokens={completion_tokens} finish={results[-1]['finish_reason']}",
                    flush=True,
                )
    atomic_text(
        args.out,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in results
        ),
    )
    summary = {
        "schema_version": "ksova-hf-lora-batch-summary-1.0",
        "model_path": str(args.model_path.resolve()),
        "adapter": str(args.adapter.resolve()) if args.adapter else None,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "requests": len(requests),
        "stop": sum(row["finish_reason"] == "stop" for row in results),
        "length": sum(row["finish_reason"] == "length" for row in results),
        "empty": sum(not row["response"] for row in results),
        "output": str(args.out.resolve()),
    }
    atomic_text(
        args.out.with_suffix(args.out.suffix + ".summary.json"),
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
