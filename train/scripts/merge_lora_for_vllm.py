#!/usr/bin/env python3
"""Safely merge a PEFT LoRA into Qwen3.5 for backend-stable vLLM tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument(
        "--adapter-scale",
        type=float,
        default=1.0,
        help="Multiply every loaded LoRA delta by this positive factor before merging.",
    )
    args = parser.parse_args()
    if not 0.0 < args.adapter_scale <= 1.0:
        raise ValueError("--adapter-scale must be in the interval (0, 1]")
    model_path = args.model_path.resolve()
    adapter = args.adapter.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    if not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"adapter weights not found: {adapter}")
    staging = out.with_name(out.name + f".staging-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"staging path already exists: {staging}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), trust_remote_code=True, local_files_only=True
    )
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    scaling_before: set[float] = set()
    scaling_after: set[float] = set()
    scaled_adapter_entries = 0
    for module in peft_model.modules():
        scaling = getattr(module, "scaling", None)
        if not isinstance(scaling, dict):
            continue
        for adapter_name, value in scaling.items():
            before = float(value)
            after = before * args.adapter_scale
            scaling_before.add(before)
            scaling[adapter_name] = after
            scaling_after.add(after)
            scaled_adapter_entries += 1
    if not scaled_adapter_entries:
        raise RuntimeError("loaded adapter exposed no LoRA scaling entries")
    merged = peft_model.merge_and_unload(safe_merge=True, progressbar=True)
    merged.save_pretrained(
        str(staging), safe_serialization=True, max_shard_size=args.max_shard_size
    )
    tokenizer.save_pretrained(str(staging))

    weight_files = sorted(staging.glob("*.safetensors"))
    if not weight_files:
        raise RuntimeError("merged model did not produce safetensors weights")
    manifest = {
        "schema_version": "ksova-merged-lora-model-1.0",
        "model_path": str(model_path),
        "model_config_sha256": sha256(model_path / "config.json"),
        "adapter": str(adapter),
        "adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
        "adapter_scale": args.adapter_scale,
        "scaled_adapter_entries": scaled_adapter_entries,
        "lora_scaling_before": sorted(scaling_before),
        "lora_scaling_after": sorted(scaling_after),
        "dtype": "bfloat16",
        "safe_merge": True,
        "max_shard_size": args.max_shard_size,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in weight_files
        ],
    }
    manifest_path = staging / "KSOVA_MERGE_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    staging.replace(out)
    print(json.dumps({**manifest, "out": str(out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
