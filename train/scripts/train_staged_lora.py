#!/usr/bin/env python3
"""Run the three-stage K-SOVA LoRA first-pass training experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any


LEGACY_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
QWEN35_LANGUAGE_LORA_PATTERN = (
    r"model\.language_model\.layers\.\d+\."
    r"(?:mlp\.(?:gate_proj|up_proj|down_proj)|"
    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj|in_proj_a|in_proj_b))"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def apply_template(tokenizer: Any, messages: list[dict[str, str]], generation: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": generation,
    }
    try:
        return tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


class CompletionDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
        import torch

        self.records: list[dict[str, Any]] = []
        self.stats: list[dict[str, int]] = []
        for row in rows:
            messages = row["messages"]
            if (
                not isinstance(messages, list)
                or len(messages) != 2
                or messages[0].get("role") != "user"
                or messages[1].get("role") != "assistant"
            ):
                raise ValueError("each row must contain one user and one assistant message")
            prompt_text = apply_template(tokenizer, messages[:1], True)
            full_text = apply_template(tokenizer, messages, False)
            prompt_ids = tokenizer(
                prompt_text, add_special_tokens=False, truncation=True,
                max_length=max_length,
            )["input_ids"]
            full_ids = tokenizer(
                full_text, add_special_tokens=False, truncation=True,
                max_length=max_length,
            )["input_ids"]
            if len(full_ids) <= len(prompt_ids):
                raise ValueError(
                    f"assistant completion was fully truncated: "
                    f"{row.get('metadata', {}).get('training_id')}"
                )
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
            if tokenizer.eos_token_id is not None and full_ids[-1] != tokenizer.eos_token_id:
                if len(full_ids) < max_length:
                    full_ids.append(tokenizer.eos_token_id)
                    labels.append(tokenizer.eos_token_id)
            self.records.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })
            self.stats.append({
                "tokens": len(full_ids),
                "prompt_tokens": len(prompt_ids),
                "assistant_tokens": sum(value != -100 for value in labels),
            })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class CompletionCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        maximum = max(len(row["input_ids"]) for row in rows)
        input_ids = torch.full(
            (len(rows), maximum), self.pad_token_id, dtype=torch.long
        )
        labels = torch.full((len(rows), maximum), -100, dtype=torch.long)
        attention = torch.zeros((len(rows), maximum), dtype=torch.long)
        for index, row in enumerate(rows):
            length = len(row["input_ids"])
            input_ids[index, :length] = row["input_ids"]
            labels[index, :length] = row["labels"]
            attention[index, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


def dataset_report(dataset: CompletionDataset) -> dict[str, Any]:
    values = dataset.stats
    return {
        "rows": len(values),
        "max_tokens": max(row["tokens"] for row in values),
        "mean_tokens": round(sum(row["tokens"] for row in values) / len(values), 3),
        "min_assistant_tokens": min(row["assistant_tokens"] for row in values),
        "mean_assistant_tokens": round(
            sum(row["assistant_tokens"] for row in values) / len(values), 3
        ),
    }


def move(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def evaluate(model: Any, loader: Any, device: Any) -> float:
    import torch

    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = move(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            total += float(loss.detach().cpu())
            count += 1
    model.train()
    return total / count if count else math.nan


def lora_target_coverage(model: Any, target_modules: list[str] | str) -> dict[str, Any]:
    if isinstance(target_modules, str):
        matched = [
            name for name, _ in model.named_modules()
            if re.fullmatch(target_modules, name)
        ]
    else:
        matched = [
            name for name, _ in model.named_modules()
            if name in target_modules
            or any(name.endswith(f".{target}") for target in target_modules)
        ]
    families = Counter(
        "linear_attention" if ".linear_attn." in name
        else "full_attention" if ".self_attn." in name
        else "mlp" if ".mlp." in name
        else "other"
        for name in matched
    )
    layers = {
        int(match.group(1))
        for name in matched
        if (match := re.search(r"\.layers\.(\d+)\.", name))
    }
    return {
        "matched_module_count": len(matched),
        "family_counts": dict(sorted(families.items())),
        "language_layer_count": len(layers),
        "matched_module_names_sha256": hashlib.sha256(
            "\n".join(sorted(matched)).encode()
        ).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--include-linear-attention", action="store_true",
        help=(
            "Target Qwen3.5 language-only linear-attention projections in "
            "addition to full attention and MLP projections."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--max-steps-per-phase", type=int, default=0)
    args = parser.parse_args()

    target_modules: list[str] | str = (
        QWEN35_LANGUAGE_LORA_PATTERN
        if args.include_linear_attention else LEGACY_LORA_TARGETS
    )

    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import (
        AutoTokenizer,
        Qwen3_5ForConditionalGeneration,
        get_cosine_schedule_with_warmup,
        set_seed,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed)
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_files = {
        phase: {
            split: args.data_dir / f"phase{phase}_{split}.jsonl"
            for split in ("train", "dev")
        }
        for phase in (1, 2, 3)
    }
    for files in data_files.values():
        for path in files.values():
            if not path.is_file():
                raise FileNotFoundError(path)

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    datasets: dict[int, dict[str, CompletionDataset]] = {}
    preparation: dict[str, Any] = {}
    for phase, files in data_files.items():
        datasets[phase] = {
            split: CompletionDataset(read_jsonl(path), tokenizer, args.max_length)
            for split, path in files.items()
        }
        preparation[f"phase{phase}"] = {
            split: dataset_report(dataset)
            for split, dataset in datasets[phase].items()
        }
    preparation.update({
        "model_path": str(args.model_path.resolve()),
        "model_config_sha256": sha256(args.model_path / "config.json"),
        "data_sha256": {
            f"phase{phase}_{split}": sha256(path)
            for phase, files in data_files.items()
            for split, path in files.items()
        },
        "max_length": args.max_length,
        "seed": args.seed,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_strategy": (
                "qwen35_language_full_linear_attention_and_mlp"
                if args.include_linear_attention
                else "legacy_suffix_full_attention_and_mlp"
            ),
            "target_modules": target_modules,
        },
    })
    atomic_json(args.output_dir / "preparation_report.json", preparation)
    print(json.dumps(preparation, ensure_ascii=False), flush=True)

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(args.model_path),
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    if not hasattr(model, "model") or not hasattr(model.model, "visual"):
        raise RuntimeError("Qwen3.5 vision module was not found for explicit freezing")
    for parameter in model.model.visual.parameters():
        parameter.requires_grad = False
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    coverage = lora_target_coverage(model, target_modules)
    if coverage["matched_module_count"] <= 0:
        raise RuntimeError("LoRA target selector matched no modules")
    if args.include_linear_attention:
        expected_families = {"linear_attention", "full_attention", "mlp"}
        if set(coverage["family_counts"]) != expected_families:
            raise RuntimeError(f"incomplete Qwen3.5 LoRA target coverage: {coverage}")
        if coverage["language_layer_count"] != len(model.model.language_model.layers):
            raise RuntimeError(f"incomplete Qwen3.5 language layer coverage: {coverage}")
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable <= 0 or trainable / total > 0.05:
        raise RuntimeError(f"unexpected trainable parameters: {trainable}/{total}")
    model.to(torch.device("cuda"))
    model.print_trainable_parameters()

    collator = CompletionCollator(tokenizer.pad_token_id)
    learning_rates = {1: 1.0e-4, 2: 8.0e-5, 3: 5.0e-5}
    metrics: dict[str, Any] = {
        "schema_version": "ksova-staged-lora-training-metrics-1.0",
        "model_path": str(args.model_path.resolve()),
        "model_family": "Qwen3.5-9B",
        "training_modality": "text_only_inputs",
        "vision_parameters_frozen": True,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio": trainable / total,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": target_modules,
            "target_coverage": coverage,
        },
        "device": torch.cuda.get_device_name(0),
        "phases": {},
    }
    device = torch.device("cuda")
    for phase in (1, 2, 3):
        phase_started = time.time()
        train_loader = DataLoader(
            datasets[phase]["train"],
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,
            generator=torch.Generator().manual_seed(args.seed + phase),
        )
        dev_loader = DataLoader(
            datasets[phase]["dev"],
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=learning_rates[phase], weight_decay=0.01,
        )
        update_steps_per_epoch = math.ceil(
            len(train_loader) / args.gradient_accumulation
        )
        planned_updates = max(1, math.ceil(update_steps_per_epoch * args.epochs))
        if args.max_steps_per_phase:
            planned_updates = min(planned_updates, args.max_steps_per_phase)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=min(2, max(0, planned_updates // 10)),
            num_training_steps=planned_updates,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        update = 0
        micro_step = 0
        stop = False
        epoch = 0
        while epoch < math.ceil(args.epochs) and not stop:
            for batch in train_loader:
                micro_step += 1
                batch = move(batch, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(**batch).loss
                    scaled = loss / args.gradient_accumulation
                scaled.backward()
                losses.append(float(loss.detach().cpu()))
                if micro_step % args.gradient_accumulation == 0 or micro_step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in model.parameters() if parameter.requires_grad],
                        1.0,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    update += 1
                    print(
                        f"[phase {phase}] update={update}/{planned_updates} "
                        f"loss={losses[-1]:.6f}", flush=True,
                    )
                    if update >= planned_updates:
                        stop = True
                        break
            epoch += 1
        dev_loss = evaluate(model, dev_loader, device)
        phase_dir = args.output_dir / f"phase{phase}_adapter"
        model.save_pretrained(str(phase_dir))
        tokenizer.save_pretrained(str(phase_dir))
        metrics["phases"][f"phase{phase}"] = {
            "learning_rate": learning_rates[phase],
            "micro_steps": micro_step,
            "optimizer_updates": update,
            "mean_train_loss": sum(losses) / len(losses),
            "final_train_loss": losses[-1],
            "dev_loss": dev_loss,
            "elapsed_seconds": round(time.time() - phase_started, 3),
            "adapter_dir": str(phase_dir),
        }
        atomic_json(args.output_dir / "training_metrics.json", metrics)
        print(json.dumps(metrics["phases"][f"phase{phase}"], ensure_ascii=False), flush=True)

    final_dir = args.output_dir / "final_adapter"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    metrics["final_adapter"] = str(final_dir)
    metrics["elapsed_seconds"] = round(time.time() - started, 3)
    metrics["max_cuda_memory_gib"] = round(
        torch.cuda.max_memory_allocated() / (1024 ** 3), 3
    )
    metrics["complete"] = True
    atomic_json(args.output_dir / "training_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
