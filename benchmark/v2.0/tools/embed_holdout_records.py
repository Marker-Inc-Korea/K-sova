#!/usr/bin/env python3
"""Embed benchmark and material records for K-SOVA holdout leakage analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(row)
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    rows = read_jsonl(args.records)
    record_ids = [str(row.get("record_id") or "") for row in rows]
    if any(not value for value in record_ids) or len(record_ids) != len(set(record_ids)):
        raise ValueError("record_id values must be nonempty and unique")
    texts = ["passage: " + str(row.get("text") or "").strip() for row in rows]
    if any(len(text) < 30 for text in texts):
        raise ValueError("embedding input contains an empty or underspecified text")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=args.cache_dir, trust_remote_code=False,
    )
    model = AutoModel.from_pretrained(
        args.model, cache_dir=args.cache_dir, trust_remote_code=False,
    ).to(args.device)
    model.eval()

    embeddings: list[list[float]] = []
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            encoded = tokenizer(
                texts[start : start + args.batch_size],
                padding=True, truncation=True, max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = functional.normalize(pooled, p=2, dim=1)
            embeddings.extend(pooled.float().cpu().tolist())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(args.output.name + f".tmp-{os.getpid()}")
    with temporary_output.open("w", encoding="utf-8") as handle:
        for row, embedding in zip(rows, embeddings, strict=True):
            handle.write(json.dumps({
                "record_id": row["record_id"],
                "role": row["role"],
                "entity_id": row["entity_id"],
                "embedding": embedding,
            }, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary_output.replace(args.output)
    report = {
        "schema_version": "ksova-training-holdout-embedding-report-1.0",
        "records": len(rows),
        "embeddings": len(embeddings),
        "dimensions": len(embeddings[0]) if embeddings else 0,
        "model": args.model,
        "max_length": args.max_length,
        "input_sha256": file_sha256(args.records),
        "output_sha256": file_sha256(args.output),
    }
    temporary_report = args.report.with_name(args.report.name + f".tmp-{os.getpid()}")
    temporary_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_report.replace(args.report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
