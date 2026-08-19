#!/usr/bin/env python3
"""Compare completion-token likelihood for a base model and SFT adapters."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from train_method_search import (
    SFTCollator,
    SFTDataset,
    move_tensor_batch,
    read_jsonl,
    sequence_logps,
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter_specs(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"adapter must be NAME=PATH: {value}")
        name, raw_path = value.split("=", 1)
        path = Path(raw_path).resolve()
        if not name or name in result or not (path / "adapter_model.safetensors").is_file():
            raise ValueError(f"invalid adapter: {value}")
        result[name] = path
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--dev-data", type=Path, required=True)
    parser.add_argument("--adapter", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--minimum-train-reduction-percent", type=float, default=10.0)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from peft import PeftModel

    adapters = adapter_specs(args.adapter)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path), trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    datasets = {
        "train": SFTDataset(read_jsonl(args.train_data), tokenizer, args.max_length),
        "dev": SFTDataset(read_jsonl(args.dev_data), tokenizer, args.max_length),
    }
    collator = SFTCollator(tokenizer.pad_token_id)

    def load_base() -> Any:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(args.model_path), dtype=torch.bfloat16, trust_remote_code=True,
            local_files_only=True, attn_implementation="sdpa",
        )
        model.to(torch.device("cuda")).eval()
        model.config.use_cache = False
        return model

    def evaluate(label: str, model: Any) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for split, dataset in datasets.items():
            loader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False,
                collate_fn=collator, num_workers=0,
            )
            groups: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
            started = time.monotonic()
            with torch.inference_mode():
                for batch in loader:
                    tensors = move_tensor_batch(batch, torch.device("cuda"))
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        sums, counts = sequence_logps(model, tensors)
                    for index, kind in enumerate(batch["example_kinds"]):
                        negative_logp = -float(sums[index].cpu())
                        tokens = int(counts[index].cpu())
                        row_nll = negative_logp / tokens
                        for group in ("all", str(kind)):
                            groups[group][0] += negative_logp
                            groups[group][1] += tokens
                            groups[group][2] += row_nll
                            groups[group][3] += 1
            result[split] = {
                group: {
                    "rows": int(values[3]),
                    "completion_tokens": int(values[1]),
                    "token_nll": values[0] / values[1],
                    "mean_row_nll": values[2] / values[3],
                    "perplexity": math.exp(min(20.0, values[0] / values[1])),
                }
                for group, values in sorted(groups.items())
            }
            result[split]["elapsed_seconds"] = round(time.monotonic() - started, 3)
            print(
                f"[{label} {split}] nll={result[split]['all']['token_nll']:.6f}",
                flush=True,
            )
        return result

    results: dict[str, Any] = {}
    print("[load] base", flush=True)
    base = load_base()
    results["base"] = evaluate("base", base)
    del base
    gc.collect()
    torch.cuda.empty_cache()
    for name, path in adapters.items():
        print(f"[load] {name}", flush=True)
        base = load_base()
        model = PeftModel.from_pretrained(base, str(path)).eval()
        results[name] = evaluate(name, model)
        del model, base
        gc.collect()
        torch.cuda.empty_cache()

    base_train = results["base"]["train"]["all"]["token_nll"]
    base_dev = results["base"]["dev"]["all"]["token_nll"]
    comparisons: dict[str, Any] = {}
    eligible: list[str] = []
    for name in adapters:
        train = results[name]["train"]["all"]["token_nll"]
        dev = results[name]["dev"]["all"]["token_nll"]
        train_reduction = (base_train - train) / base_train * 100
        dev_reduction = (base_dev - dev) / base_dev * 100
        comparisons[name] = {
            "train_nll": train,
            "dev_nll": dev,
            "train_reduction_percent": train_reduction,
            "dev_reduction_percent": dev_reduction,
            "train_gate_pass": train_reduction >= args.minimum_train_reduction_percent,
        }
        if comparisons[name]["train_gate_pass"]:
            eligible.append(name)
    selected = min(eligible, key=lambda name: comparisons[name]["dev_nll"]) if eligible else None
    report = {
        "schema_version": "ksova-sft-checkpoint-likelihood-1.0",
        "status": "complete",
        "model_path": str(args.model_path.resolve()),
        "train_data": {"path": str(args.train_data.resolve()), "sha256": sha256(args.train_data)},
        "dev_data": {"path": str(args.dev_data.resolve()), "sha256": sha256(args.dev_data)},
        "adapters": {
            name: {
                "path": str(path),
                "sha256": sha256(path / "adapter_model.safetensors"),
            }
            for name, path in adapters.items()
        },
        "batch_size": args.batch_size,
        "minimum_train_reduction_percent": args.minimum_train_reduction_percent,
        "results": results,
        "comparisons": comparisons,
        "selected_checkpoint": selected,
        "selection_rule": "lowest dev completion-token NLL among checkpoints passing the train NLL reduction gate",
    }
    atomic_json(args.out, report)
    print(json.dumps({"selected_checkpoint": selected, "comparisons": comparisons}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
