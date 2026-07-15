#!/usr/bin/env python3
"""Run generic chat requests with a local Transformers model on a Slurm GPU."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_prompt(processor: Any, system: str, user: str, image_text: bool, enable_thinking: bool) -> str:
    if image_text:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ]
    else:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kwargs = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": enable_thinking}
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return processor.apply_chat_template(messages, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-prefix", default="responses")
    parser.add_argument("--auto-class", choices=["auto", "causal-lm", "image-text-to-text"], default="auto")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if str(args.device_map).startswith("cuda") and not os.environ.get("SLURM_JOB_ID"):
        parser.error("CUDA inference is Slurm-only; submit the corresponding .sbatch script")
    rows = load_jsonl(args.requests)
    if args.limit is not None:
        rows = rows[:args.limit]
    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    image_text = args.auto_class == "image-text-to-text" or (
        args.auto_class == "auto" and getattr(cfg, "model_type", "") in {"exaone4_5"}
    )
    if image_text:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map=args.device_map,
            trust_remote_code=True, local_files_only=True,
        )
    else:
        processor = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map=args.device_map,
            trust_remote_code=True, local_files_only=True,
        )
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out_dir / f"{args.out_prefix}_{stamp}.jsonl"
    errors = 0
    with out_path.open("w", encoding="utf-8") as handle:
        completed = 0
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            started = time.monotonic()
            batch_error = None
            texts = [""] * len(batch)
            try:
                prompts = [
                    build_prompt(
                        processor, row.get("system", "You are a careful assistant."),
                        row["prompt"], image_text, args.enable_thinking,
                    )
                    for row in batch
                ]
                inputs = processor(text=prompts, return_tensors="pt", padding=True)
                inputs = {key: value.to(model.device) for key, value in inputs.items() if torch.is_tensor(value)}
                with torch.inference_mode():
                    output = model.generate(
                        **inputs, max_new_tokens=max(row.get("max_tokens", args.max_new_tokens) for row in batch),
                        do_sample=False, temperature=None, top_p=None,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                prompt_width = inputs["input_ids"].shape[-1]
                decoder = getattr(processor, "batch_decode", tokenizer.batch_decode)
                texts = [text.strip() for text in decoder(output[:, prompt_width:], skip_special_tokens=True)]
            except Exception as exc:
                batch_error = f"{type(exc).__name__}: {exc}"
                errors += len(batch)
            elapsed = round(time.monotonic() - started, 3)
            for row, text in zip(batch, texts):
                completed += 1
                result = {
                    "id": row["id"], "model": args.model_name, "response": text,
                    "reasoning_content": "", "usage": None, "metadata": row.get("metadata", {}),
                    "error": batch_error, "elapsed_seconds": elapsed,
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                print(f"[{completed}/{len(rows)}] {row['id']} error={bool(batch_error)}", flush=True)
    summary = {"output": str(out_path), "requests": len(rows), "completed": len(rows), "errors": errors}
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
