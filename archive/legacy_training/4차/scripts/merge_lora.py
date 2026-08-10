#!/usr/bin/env python3
"""LoRA 어댑터를 base에 머지해 저장 (lm_eval peft 비호환 모델용 — exaone).
generate_v2.load(검증된 어댑터 적용)로 로드 → merge_and_unload → save.
사용: python merge_lora.py <model_key> <adapter_dir> <out_dir>
"""
import sys
from generate_v2 import load
mk, adapter, out = sys.argv[1], sys.argv[2], sys.argv[3]
model, tok = load(mk, adapter)
print(f"[merge] {mk} + {adapter} → merge_and_unload", flush=True)
model = model.merge_and_unload()
model.save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out)
print(f"[merge] saved → {out}", flush=True)
