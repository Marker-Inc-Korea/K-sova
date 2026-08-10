#!/usr/bin/env python3
"""Embed K-SOVA v2.1 items for semantic duplicate retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected object")
            rows.append(row)
    return rows


def item_text(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("question_ko") or "").strip(),
        str(row.get("decision_claim_ko") or "").strip(),
        str(row.get("reference_answer_ko") or "").strip(),
    ]
    return "passage: " + "\n".join(part for part in parts if part)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
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

    rows = read_jsonl(args.items)
    item_ids = [str(row.get("item_id") or "") for row in rows]
    if not all(item_ids) or len(item_ids) != len(set(item_ids)):
        raise ValueError("item_id values must be nonempty and unique")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        trust_remote_code=False,
    )
    model = AutoModel.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        trust_remote_code=False,
    ).to(args.device)
    model.eval()

    embeddings: list[list[float]] = []
    texts = [item_text(row) for row in rows]
    with torch.inference_mode():
        for start in range(0, len(texts), args.batch_size):
            encoded = tokenizer(
                texts[start : start + args.batch_size],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = functional.normalize(pooled, p=2, dim=1)
            embeddings.extend(pooled.float().cpu().tolist())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item_id, embedding in zip(item_ids, embeddings, strict=True):
            handle.write(
                json.dumps(
                    {"item_id": item_id, "embedding": embedding},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    report = {
        "items": len(rows),
        "embeddings": len(embeddings),
        "dimensions": len(embeddings[0]) if embeddings else 0,
        "model": args.model,
        "max_length": args.max_length,
        "text_fields": ["question_ko", "decision_claim_ko", "reference_answer_ko"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
