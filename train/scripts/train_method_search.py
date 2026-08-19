#!/usr/bin/env python3
"""Train controlled K-SOVA final-answer SFT or DPO LoRA arms.

DPO reference log probabilities are computed once before the policy is
updated.  This avoids keeping a second 9B reference model in GPU memory while
remaining equivalent to a frozen-reference DPO objective.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time
from typing import Any


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


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def apply_template(tokenizer: Any, messages: list[dict[str, str]], generation: bool) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": generation}
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def encode_completion(
    tokenizer: Any,
    prompt_messages: list[dict[str, str]],
    completion: str,
    max_length: int,
) -> tuple[list[int], list[int], dict[str, int]]:
    prompt_text = apply_template(tokenizer, prompt_messages, True)
    full_text = apply_template(
        tokenizer,
        prompt_messages + [{"role": "assistant", "content": completion}],
        False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template prompt is not a prefix of the full sequence")
    if len(full_ids) > max_length:
        raise ValueError(f"sequence length {len(full_ids)} exceeds max_length {max_length}")
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("assistant completion is empty")
    if tokenizer.eos_token_id is not None and full_ids[-1] != tokenizer.eos_token_id:
        if len(full_ids) == max_length:
            raise ValueError("no room to append EOS token")
        full_ids.append(tokenizer.eos_token_id)
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return full_ids, labels, {
        "tokens": len(full_ids),
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": sum(token != -100 for token in labels),
    }


class SFTDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
        import torch

        self.records: list[dict[str, Any]] = []
        self.stats: list[dict[str, int]] = []
        for row in rows:
            messages = row.get("messages") or []
            if (
                len(messages) != 2
                or messages[0].get("role") != "user"
                or messages[1].get("role") != "assistant"
            ):
                raise ValueError("SFT row must have one user and one assistant message")
            ids, labels, stats = encode_completion(
                tokenizer, messages[:1], str(messages[1]["content"]), max_length
            )
            metadata = row.get("metadata") or {}
            self.records.append({
                "input_ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "training_id": str(metadata.get("training_id")),
                "issue_id": str(metadata.get("issue_id")),
                "example_kind": str(metadata.get("example_kind") or "unspecified"),
                "coverage_priority": bool(metadata.get("coverage_priority", False)),
            })
            self.stats.append(stats)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class DPODataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
        import torch

        self.records: list[dict[str, Any]] = []
        self.stats: list[dict[str, Any]] = []
        for row in rows:
            prompt_messages = row.get("prompt_messages") or []
            if len(prompt_messages) != 1 or prompt_messages[0].get("role") != "user":
                raise ValueError("DPO prompt must contain exactly one user message")
            chosen_ids, chosen_labels, chosen_stats = encode_completion(
                tokenizer, prompt_messages, str(row["chosen"]), max_length
            )
            rejected_ids, rejected_labels, rejected_stats = encode_completion(
                tokenizer, prompt_messages, str(row["rejected"]), max_length
            )
            metadata = row.get("metadata") or {}
            sample_weight = float(metadata.get("sample_weight", 1.0))
            if not math.isfinite(sample_weight) or sample_weight <= 0:
                raise ValueError(f"invalid DPO sample weight: {sample_weight}")
            self.records.append({
                "chosen_ids": torch.tensor(chosen_ids, dtype=torch.long),
                "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
                "rejected_ids": torch.tensor(rejected_ids, dtype=torch.long),
                "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
                "training_id": str(metadata.get("training_id")),
                "pair_id": str(metadata.get("pair_id")),
                "sample_weight": sample_weight,
                "ref_chosen": None,
                "ref_rejected": None,
            })
            self.stats.append({
                "chosen_tokens": chosen_stats["tokens"],
                "chosen_completion_tokens": chosen_stats["completion_tokens"],
                "rejected_tokens": rejected_stats["tokens"],
                "rejected_completion_tokens": rejected_stats["completion_tokens"],
                "completion_token_ratio": round(
                    chosen_stats["completion_tokens"] / rejected_stats["completion_tokens"], 5
                ),
                "sample_weight": sample_weight,
            })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class SFTCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        maximum = max(len(row["input_ids"]) for row in rows)
        ids = torch.full((len(rows), maximum), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(rows), maximum), -100, dtype=torch.long)
        attention = torch.zeros((len(rows), maximum), dtype=torch.long)
        for index, row in enumerate(rows):
            length = len(row["input_ids"])
            ids[index, :length] = row["input_ids"]
            labels[index, :length] = row["labels"]
            attention[index, :length] = 1
        return {
            "input_ids": ids,
            "labels": labels,
            "attention_mask": attention,
            "training_ids": [row["training_id"] for row in rows],
            "issue_ids": [row["issue_id"] for row in rows],
            "example_kinds": [row["example_kind"] for row in rows],
            "coverage_priorities": [row["coverage_priority"] for row in rows],
        }


class FixedOrderSampler:
    """Yield a preregistered, deterministic index order exactly once per epoch."""

    def __init__(self, indices: list[int]):
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def exposure_summary(
    training_ids: list[str], issue_ids: list[str], example_kinds: list[str],
) -> dict[str, Any]:
    return {
        "draws": len(training_ids),
        "unique_training_ids": len(set(training_ids)),
        "unique_issues": len(set(issue_ids)),
        "repeated_training_id_draws": len(training_ids) - len(set(training_ids)),
        "example_kind_counts": dict(sorted(Counter(example_kinds).items())),
        "training_id_sequence_sha256": hashlib.sha256(
            "\n".join(training_ids).encode()
        ).hexdigest(),
        "issue_id_sequence_sha256": hashlib.sha256(
            "\n".join(issue_ids).encode()
        ).hexdigest(),
    }


class DPOCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        sequences = [
            (row["chosen_ids"], row["chosen_labels"])
            for row in rows
        ] + [
            (row["rejected_ids"], row["rejected_labels"])
            for row in rows
        ]
        maximum = max(len(ids) for ids, _ in sequences)
        input_ids = torch.full(
            (len(sequences), maximum), self.pad_token_id, dtype=torch.long
        )
        labels = torch.full((len(sequences), maximum), -100, dtype=torch.long)
        attention = torch.zeros((len(sequences), maximum), dtype=torch.long)
        for index, (ids, row_labels) in enumerate(sequences):
            length = len(ids)
            input_ids[index, :length] = ids
            labels[index, :length] = row_labels
            attention[index, :length] = 1
        references_ready = all(
            row["ref_chosen"] is not None and row["ref_rejected"] is not None
            for row in rows
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention,
            "batch_pairs": len(rows),
            "training_ids": [row["training_id"] for row in rows],
            "pair_ids": [row["pair_id"] for row in rows],
            "sample_weights": torch.tensor(
                [row["sample_weight"] for row in rows], dtype=torch.float32
            ),
            "ref_chosen": torch.tensor(
                [row["ref_chosen"] for row in rows], dtype=torch.float32
            ) if references_ready else None,
            "ref_rejected": torch.tensor(
                [row["ref_rejected"] for row in rows], dtype=torch.float32
            ) if references_ready else None,
        }


def dataset_summary(stats: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(stats[0]) if stats else []
    result: dict[str, Any] = {"rows": len(stats)}
    for key in keys:
        values = [float(row[key]) for row in stats]
        result[key] = {
            "min": min(values),
            "mean": round(sum(values) / len(values), 4),
            "max": max(values),
        }
    return result


def freeze_visual(base_model: Any) -> None:
    if not hasattr(base_model, "model") or not hasattr(base_model.model, "visual"):
        raise RuntimeError("Qwen3.5 vision module was not found for explicit freezing")
    for parameter in base_model.model.visual.parameters():
        parameter.requires_grad = False


def lora_target_coverage(model: Any) -> dict[str, Any]:
    matched = [
        name for name, _ in model.named_modules()
        if re.fullmatch(QWEN35_LANGUAGE_LORA_PATTERN, name)
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
        "matched_names_sha256": hashlib.sha256(
            "\n".join(sorted(matched)).encode()
        ).hexdigest(),
    }


def move_tensor_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key in {"input_ids", "labels", "attention_mask"}
    }


def sequence_logps(model: Any, batch: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    import torch.nn.functional as functional

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    shifted_logits = outputs.logits[:, :-1, :]
    shifted_labels = batch["labels"][:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_logps = -functional.cross_entropy(
        shifted_logits.transpose(1, 2),
        safe_labels,
        reduction="none",
    )
    sums = (token_logps * mask).sum(dim=-1)
    counts = mask.sum(dim=-1)
    if bool((counts == 0).any()):
        raise RuntimeError("completion mask contains zero tokens")
    return sums, counts


def enable_training_features(model: Any) -> None:
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()
    model.enable_input_require_grads()


def make_optimizer(model: Any, learning_rate: float) -> Any:
    import torch

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("no trainable parameters")
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.01)


def parameter_report(model: Any) -> dict[str, Any]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if trainable <= 0 or trainable / total > 0.05:
        raise RuntimeError(f"unexpected trainable parameters: {trainable}/{total}")
    return {
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_ratio": trainable / total,
    }


def compute_reference_logps(
    model: Any,
    datasets: dict[str, DPODataset],
    collator: DPOCollator,
    device: Any,
    batch_size: int,
) -> list[dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    cache_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for split in ("train", "dev"):
            loader = DataLoader(
                datasets[split], batch_size=batch_size, shuffle=False,
                collate_fn=collator, num_workers=0,
            )
            offset = 0
            for batch in loader:
                size = int(batch["batch_pairs"])
                tensors = move_tensor_batch(batch, device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logps, counts = sequence_logps(model, tensors)
                chosen, rejected = logps[:size], logps[size:]
                chosen_counts, rejected_counts = counts[:size], counts[size:]
                for local_index in range(size):
                    record = datasets[split].records[offset + local_index]
                    record["ref_chosen"] = float(chosen[local_index].cpu())
                    record["ref_rejected"] = float(rejected[local_index].cpu())
                    cache_rows.append({
                        "split": split,
                        "training_id": record["training_id"],
                        "pair_id": record["pair_id"],
                        "reference_chosen_logp": record["ref_chosen"],
                        "reference_rejected_logp": record["ref_rejected"],
                        "chosen_completion_tokens": int(chosen_counts[local_index].cpu()),
                        "rejected_completion_tokens": int(rejected_counts[local_index].cpu()),
                    })
                offset += size
                print(
                    f"[reference {split}] {offset}/{len(datasets[split])}", flush=True
                )
    return cache_rows


def evaluate_sft(model: Any, loader: Any, device: Any) -> float:
    import torch

    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            tensors = move_tensor_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**tensors, use_cache=False).loss
            losses.append(float(loss.cpu()))
    model.train()
    return sum(losses) / len(losses) if losses else math.nan


def dpo_loss(model: Any, batch: dict[str, Any], device: Any, beta: float) -> tuple[Any, dict[str, float]]:
    import torch
    import torch.nn.functional as functional

    size = int(batch["batch_pairs"])
    tensors = move_tensor_batch(batch, device)
    ref_chosen = batch["ref_chosen"].to(device)
    ref_rejected = batch["ref_rejected"].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        policy_logps, _ = sequence_logps(model, tensors)
        policy_chosen, policy_rejected = policy_logps[:size], policy_logps[size:]
        logits = beta * (
            (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
        )
        pair_losses = -functional.logsigmoid(logits)
        sample_weights = batch["sample_weights"].to(device)
        loss = (pair_losses * sample_weights).mean()
    stats = {
        "loss": float(loss.detach().cpu()),
        "preference_accuracy": float((logits.detach() > 0).float().mean().cpu()),
        "mean_margin": float(logits.detach().mean().cpu()),
        "mean_policy_chosen_logp": float(policy_chosen.detach().mean().cpu()),
        "mean_policy_rejected_logp": float(policy_rejected.detach().mean().cpu()),
        "mean_sample_weight": float(sample_weights.detach().mean().cpu()),
    }
    return loss, stats


def evaluate_dpo(model: Any, loader: Any, device: Any, beta: float) -> dict[str, float]:
    import torch

    model.eval()
    rows: list[dict[str, float]] = []
    with torch.inference_mode():
        for batch in loader:
            _, stats = dpo_loss(model, batch, device, beta)
            rows.append(stats)
    model.train()
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    } if rows else {}


def train_sft(args: argparse.Namespace, model: Any, tokenizer: Any, metrics: dict[str, Any]) -> None:
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from transformers import get_cosine_schedule_with_warmup

    train_rows = read_jsonl(args.train_data)
    dev_rows = read_jsonl(args.dev_data)
    train_dataset = SFTDataset(train_rows, tokenizer, args.max_length)
    dev_dataset = SFTDataset(dev_rows, tokenizer, args.max_length)
    collator = SFTCollator(tokenizer.pad_token_id)
    sampler = None
    if args.weighted_sft_sampling and args.coverage_first_sft_sampling:
        raise ValueError(
            "--weighted-sft-sampling and --coverage-first-sft-sampling "
            "are mutually exclusive"
        )
    if args.weighted_sft_sampling:
        weights = [
            float((row.get("metadata") or {}).get("sample_weight", 0.0))
            for row in train_rows
        ]
        if not weights or any(not math.isfinite(value) or value <= 0 for value in weights):
            raise ValueError(
                "--weighted-sft-sampling requires a positive finite "
                "metadata.sample_weight on every training row"
            )
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
        metrics["sft_sampling"] = {
            "policy": "weighted_random_with_replacement",
            "rows_per_epoch": len(weights),
            "minimum_weight": min(weights),
            "maximum_weight": max(weights),
        }
    elif args.coverage_first_sft_sampling:
        issue_ids = [
            str((row.get("metadata") or {}).get("issue_id") or "")
            for row in train_rows
        ]
        priority_indices = [
            index for index, row in enumerate(train_rows)
            if (row.get("metadata") or {}).get("coverage_priority") is True
        ]
        remainder_indices = [
            index for index in range(len(train_rows)) if index not in set(priority_indices)
        ]
        priority_issues = [issue_ids[index] for index in priority_indices]
        all_issues = set(issue_ids)
        if "" in all_issues:
            raise ValueError("coverage-first sampling requires metadata.issue_id")
        if not priority_indices:
            raise ValueError("coverage-first sampling requires priority rows")
        if len(priority_issues) != len(set(priority_issues)):
            raise ValueError("coverage-first sampling requires one priority row per issue")
        if set(priority_issues) != all_issues:
            missing = sorted(all_issues - set(priority_issues))
            raise ValueError(f"coverage-first priority rows miss issues: {missing}")
        priority_rng = random.Random(f"{args.seed}::coverage")
        remainder_rng = random.Random(f"{args.seed}::remainder")
        priority_rng.shuffle(priority_indices)
        remainder_rng.shuffle(remainder_indices)
        order = priority_indices + remainder_indices
        sampler = FixedOrderSampler(order)
        ordered_training_ids = [
            str((train_rows[index].get("metadata") or {}).get("training_id"))
            for index in order
        ]
        metrics["sft_sampling"] = {
            "policy": "coverage_first_without_replacement",
            "rows_per_epoch": len(order),
            "coverage_rows": len(priority_indices),
            "coverage_issues": len(priority_issues),
            "first_remainder_draw": len(priority_indices) + 1,
            "unique_indices": len(set(order)),
            "planned_training_id_sequence_sha256": hashlib.sha256(
                "\n".join(ordered_training_ids).encode()
            ).hexdigest(),
        }
    else:
        metrics["sft_sampling"] = {"policy": "uniform_shuffle_without_replacement"}
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=sampler is None, sampler=sampler,
        collate_fn=collator, num_workers=0,
        generator=(
            torch.Generator().manual_seed(args.seed) if sampler is None else None
        ),
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=0,
    )
    optimizer = make_optimizer(model, args.learning_rate)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    planned_updates = max(1, math.ceil(updates_per_epoch * args.epochs))
    if args.max_updates:
        planned_updates = min(planned_updates, args.max_updates)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(5, max(0, planned_updates // 10)),
        num_training_steps=planned_updates,
    )
    checkpoint_targets: dict[int, list[float]] = defaultdict(list)
    for fraction in args.checkpoint_fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("checkpoint fractions must be in (0, 1]")
        checkpoint_targets[max(1, math.ceil(planned_updates * fraction))].append(fraction)
    metrics["checkpoint_plan"] = [
        {
            "fraction": fraction,
            "optimizer_update": max(1, math.ceil(planned_updates * fraction)),
        }
        for fraction in sorted(args.checkpoint_fractions)
    ]
    metrics["checkpoints"] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    seen_training_ids: list[str] = []
    seen_issue_ids: list[str] = []
    seen_example_kinds: list[str] = []
    update = 0
    micro_step = 0
    micro_in_update = 0
    epoch = 0
    while epoch < math.ceil(args.epochs) and update < planned_updates:
        for batch_index, batch in enumerate(train_loader, 1):
            micro_step += 1
            micro_in_update += 1
            seen_training_ids.extend(str(value) for value in batch["training_ids"])
            seen_issue_ids.extend(str(value) for value in batch["issue_ids"])
            seen_example_kinds.extend(str(value) for value in batch["example_kinds"])
            tensors = move_tensor_batch(batch, torch.device("cuda"))
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**tensors, use_cache=False).loss
            group_start = ((batch_index - 1) // args.gradient_accumulation) * args.gradient_accumulation
            group_size = min(args.gradient_accumulation, len(train_loader) - group_start)
            (loss / group_size).backward()
            losses.append(float(loss.detach().cpu()))
            is_epoch_end = batch_index == len(train_loader)
            if micro_in_update >= args.gradient_accumulation or is_epoch_end:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                micro_in_update = 0
                print(
                    f"[sft] update={update}/{planned_updates} loss={losses[-1]:.6f}",
                    flush=True,
                )
                for fraction in checkpoint_targets.get(update, []):
                    label = f"{fraction:.4f}".rstrip("0").rstrip(".").replace(".", "p")
                    adapter_dir = (
                        args.output_dir / "checkpoints" / f"epoch_{label}" / "adapter"
                    )
                    model.save_pretrained(str(adapter_dir))
                    tokenizer.save_pretrained(str(adapter_dir))
                    metrics["checkpoints"].append({
                        "fraction": fraction,
                        "optimizer_update": update,
                        "adapter": str(adapter_dir.resolve()),
                        "exposure": exposure_summary(
                            seen_training_ids, seen_issue_ids, seen_example_kinds
                        ),
                    })
                if update >= planned_updates:
                    break
        epoch += 1
    dev_loss = evaluate_sft(model, dev_loader, torch.device("cuda"))
    metrics["datasets"] = {
        "train": dataset_summary(train_dataset.stats),
        "dev": dataset_summary(dev_dataset.stats),
    }
    metrics["optimization"] = {
        "micro_steps": micro_step,
        "optimizer_updates": update,
        "epochs_requested": args.epochs,
        "mean_train_loss": sum(losses) / len(losses),
        "final_train_loss": losses[-1],
        "dev_loss": dev_loss,
    }
    metrics["exposure"] = exposure_summary(
        seen_training_ids, seen_issue_ids, seen_example_kinds
    )


def train_dpo(args: argparse.Namespace, model: Any, tokenizer: Any, metrics: dict[str, Any]) -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    datasets = {
        "train": DPODataset(read_jsonl(args.train_data), tokenizer, args.max_length),
        "dev": DPODataset(read_jsonl(args.dev_data), tokenizer, args.max_length),
    }
    collator = DPOCollator(tokenizer.pad_token_id)
    reference_started = time.time()
    reference_rows = compute_reference_logps(
        model, datasets, collator, torch.device("cuda"), args.reference_batch_size
    )
    reference_path = args.output_dir / "reference_logps.jsonl"
    atomic_jsonl(reference_path, reference_rows)
    metrics["reference"] = {
        "policy": "initial_adapter" if args.init_adapter else "base_model",
        "rows": len(reference_rows),
        "sha256": sha256(reference_path),
        "elapsed_seconds": round(time.time() - reference_started, 3),
    }

    if not args.init_adapter:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=QWEN35_LANGUAGE_LORA_PATTERN,
        ))
        model.to(torch.device("cuda"))
    enable_training_features(model)
    metrics["parameters"] = parameter_report(model)
    model.print_trainable_parameters()

    train_loader = DataLoader(
        datasets["train"], batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=0,
        generator=torch.Generator().manual_seed(args.seed),
    )
    dev_loader = DataLoader(
        datasets["dev"], batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=0,
    )
    optimizer = make_optimizer(model, args.learning_rate)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    planned_updates = max(1, math.ceil(updates_per_epoch * args.epochs))
    if args.max_updates:
        planned_updates = min(planned_updates, args.max_updates)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(5, max(0, planned_updates // 10)),
        num_training_steps=planned_updates,
    )
    checkpoint_targets: dict[int, list[float]] = defaultdict(list)
    for fraction in args.checkpoint_fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("checkpoint fractions must be in (0, 1]")
        checkpoint_targets[max(1, math.ceil(planned_updates * fraction))].append(fraction)
    metrics["checkpoint_plan"] = [
        {
            "fraction": fraction,
            "optimizer_update": max(1, math.ceil(planned_updates * fraction)),
        }
        for fraction in sorted(args.checkpoint_fractions)
    ]
    metrics["checkpoints"] = []
    optimizer.zero_grad(set_to_none=True)
    model.train()
    step_stats: list[dict[str, float]] = []
    update = 0
    micro_step = 0
    micro_in_update = 0
    epoch = 0
    while epoch < math.ceil(args.epochs) and update < planned_updates:
        for batch_index, batch in enumerate(train_loader, 1):
            micro_step += 1
            micro_in_update += 1
            loss, stats = dpo_loss(model, batch, torch.device("cuda"), args.beta)
            group_start = ((batch_index - 1) // args.gradient_accumulation) * args.gradient_accumulation
            group_size = min(args.gradient_accumulation, len(train_loader) - group_start)
            (loss / group_size).backward()
            step_stats.append(stats)
            is_epoch_end = batch_index == len(train_loader)
            if micro_in_update >= args.gradient_accumulation or is_epoch_end:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
                micro_in_update = 0
                print(
                    f"[dpo] update={update}/{planned_updates} "
                    f"loss={stats['loss']:.6f} acc={stats['preference_accuracy']:.3f} "
                    f"margin={stats['mean_margin']:.5f}",
                    flush=True,
                )
                for fraction in checkpoint_targets.get(update, []):
                    label = f"{fraction:.4f}".rstrip("0").rstrip(".").replace(".", "p")
                    adapter_dir = (
                        args.output_dir / "checkpoints" / f"epoch_{label}" / "adapter"
                    )
                    model.save_pretrained(str(adapter_dir))
                    tokenizer.save_pretrained(str(adapter_dir))
                    metrics["checkpoints"].append({
                        "fraction": fraction,
                        "optimizer_update": update,
                        "adapter": str(adapter_dir.resolve()),
                    })
                if update >= planned_updates:
                    break
        epoch += 1
    dev_metrics = evaluate_dpo(model, dev_loader, torch.device("cuda"), args.beta)
    metrics["datasets"] = {
        split: dataset_summary(dataset.stats) for split, dataset in datasets.items()
    }
    metrics["optimization"] = {
        "micro_steps": micro_step,
        "optimizer_updates": update,
        "epochs_requested": args.epochs,
        "mean_train_loss": sum(row["loss"] for row in step_stats) / len(step_stats),
        "final_train_loss": step_stats[-1]["loss"],
        "mean_train_preference_accuracy": sum(
            row["preference_accuracy"] for row in step_stats
        ) / len(step_stats),
        "dev": dev_metrics,
    }
    metrics["trained_model"] = model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sft", "dpo"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--dev-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--reference-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--max-updates", type=int, default=0)
    parser.add_argument("--checkpoint-fractions", type=float, nargs="*", default=[])
    parser.add_argument(
        "--weighted-sft-sampling",
        action="store_true",
        help="sample SFT rows using positive metadata.sample_weight values",
    )
    parser.add_argument(
        "--coverage-first-sft-sampling",
        action="store_true",
        help=(
            "visit exactly one metadata.coverage_priority row per issue before "
            "the remaining rows, with deterministic non-replacement shuffles"
        ),
    )
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {args.output_dir}")
    if args.mode == "sft" and args.init_adapter:
        raise ValueError("SFT arm must start from the shared base model")

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed)
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(args.model_path), dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, attn_implementation="sdpa",
    )
    freeze_visual(base)
    coverage = lora_target_coverage(base)
    if set(coverage["family_counts"]) != {"linear_attention", "full_attention", "mlp"}:
        raise RuntimeError(f"incomplete Qwen3.5 LoRA target coverage: {coverage}")
    if coverage["language_layer_count"] != len(base.model.language_model.layers):
        raise RuntimeError(f"incomplete Qwen3.5 language layer coverage: {coverage}")
    base.to(torch.device("cuda"))
    if args.init_adapter:
        model = PeftModel.from_pretrained(
            base, str(args.init_adapter), is_trainable=True
        )
    else:
        model = base

    metrics: dict[str, Any] = {
        "schema_version": "ksova-method-search-training-metrics-1.0",
        "mode": args.mode,
        "model_path": str(args.model_path.resolve()),
        "model_config_sha256": sha256(args.model_path / "config.json"),
        "init_adapter": str(args.init_adapter.resolve()) if args.init_adapter else None,
        "train_data": str(args.train_data.resolve()),
        "dev_data": str(args.dev_data.resolve()),
        "train_data_sha256": sha256(args.train_data),
        "dev_data_sha256": sha256(args.dev_data),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "beta": args.beta if args.mode == "dpo" else None,
        "checkpoint_fractions": args.checkpoint_fractions,
        "weighted_sft_sampling": args.weighted_sft_sampling,
        "coverage_first_sft_sampling": args.coverage_first_sft_sampling,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": QWEN35_LANGUAGE_LORA_PATTERN,
            "target_coverage": coverage,
        },
        "device": torch.cuda.get_device_name(0),
    }

    if args.mode == "sft":
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=QWEN35_LANGUAGE_LORA_PATTERN,
        ))
        model.to(torch.device("cuda"))
        enable_training_features(model)
        metrics["parameters"] = parameter_report(model)
        model.print_trainable_parameters()
        train_sft(args, model, tokenizer, metrics)
    else:
        train_dpo(args, model, tokenizer, metrics)
        model = metrics.pop("trained_model")

    adapter_dir = args.output_dir / "final_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    metrics["final_adapter"] = str(adapter_dir.resolve())
    metrics["elapsed_seconds"] = round(time.time() - started, 3)
    metrics["max_cuda_memory_gib"] = round(
        torch.cuda.max_memory_allocated() / (1024 ** 3), 3
    )
    metrics["complete"] = True
    atomic_json(args.output_dir / "training_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
