#!/usr/bin/env python3
"""Train a Qwen LoRA with source-isolated K-SOVA preference pairs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def format_rows(
    rows: list[dict[str, Any]], tokenizer: Any, max_length: int
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    formatted: list[dict[str, str]] = []
    truncated_chosen = 0
    truncated_rejected = 0
    prompt_tokens = completion_tokens = 0
    eos = tokenizer.eos_token or ""
    for row in rows:
        prompt = str(row.get("prompt") or "").strip()
        chosen = str(row.get("chosen") or "").strip()
        rejected = str(row.get("rejected") or "").strip()
        if not prompt or not chosen or not rejected:
            raise ValueError(f"incomplete preference row: {row.get('pair_id')}")
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        chosen_text = chosen + eos
        rejected_text = rejected + eos
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        chosen_ids = tokenizer(chosen_text, add_special_tokens=False)["input_ids"]
        rejected_ids = tokenizer(rejected_text, add_special_tokens=False)["input_ids"]
        prompt_tokens += len(prompt_ids)
        completion_tokens += len(chosen_ids) + len(rejected_ids)
        truncated_chosen += int(len(prompt_ids) + len(chosen_ids) > max_length)
        truncated_rejected += int(len(prompt_ids) + len(rejected_ids) > max_length)
        formatted.append(
            {
                "prompt": prompt_text,
                "chosen": chosen_text,
                "rejected": rejected_text,
            }
        )
    report = {
        "rows": len(rows),
        "mean_prompt_tokens": round(prompt_tokens / len(rows), 3) if rows else None,
        "mean_two_completion_tokens": round(completion_tokens / len(rows), 3)
        if rows
        else None,
        "chosen_over_max_length": truncated_chosen,
        "rejected_over_max_length": truncated_rejected,
        "thinking_mode": "off via native Qwen chat-template branch",
    }
    return formatted, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import DPOConfig, DPOTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("GPU training requires a CUDA device inside a Slurm allocation")

    random.seed(args.seed)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_rows, train_report = format_rows(
        load_jsonl(args.train_file), tokenizer, args.max_length
    )
    eval_rows, eval_report = format_rows(
        load_jsonl(args.eval_file), tokenizer, args.max_length
    )
    preparation = {
        "model_path": str(args.model_path),
        "objective": "DPO: korean_aligned over context_neutral and korean_opposed",
        "train": train_report,
        "eval": eval_report,
        "max_length": args.max_length,
        "beta": args.beta,
        "seed": args.seed,
    }
    (args.output_dir / "preparation_report.json").write_text(
        json.dumps(preparation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(preparation, ensure_ascii=False), flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows)
    training_args = DPOConfig(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=20,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length,
        truncation_mode="keep_start",
        beta=args.beta,
        loss_type=["sigmoid"],
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )
    train_result = trainer.train()
    evaluation = trainer.evaluate()
    final_dir = args.output_dir / "final"
    trainer.model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    metrics = {
        "train": train_result.metrics,
        "eval": evaluation,
        "hyperparameters": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "max_length": args.max_length,
            "beta": args.beta,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        },
    }
    (args.output_dir / "training_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"saved adapter to {final_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
