#!/usr/bin/env python3
"""K-SOVA 4차 학습셋 = v3.1 train(주권 주입 자산) + format_set(형식 준수 보강).
3차 verbose 드리프트 교정. v3.1 그대로 두고 형식셋만 추가(주권 주입 보존).
eval/test는 v3.1 것 재사용(주권 평가 동일 기준 유지).
출력: train_v4.jsonl
"""
import json, random
from pathlib import Path
from collections import Counter
random.seed(42)
GEN = Path("/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen")

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

v31 = load(GEN/"train_v3_1.jsonl")
fmt = load(GEN/"format_set.jsonl")
# format_set을 train 스키마에 맞춤
for i, r in enumerate(fmt):
    r.setdefault("source", "format_v4"); r.setdefault("bucket", None)
    r["id"] = f"fmt_{i:05d}"
alld = v31 + fmt
random.shuffle(alld)
Path(GEN/"train_v4.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in alld)+"\n", encoding="utf-8")
print(f"train_v4: {len(alld)}건 = v3.1 {len(v31)} + format {len(fmt)}")
print(f"  tier 분포: {dict(Counter(r.get('tier') for r in alld))}")
print(f"  format 유형: {dict(Counter(r.get('type') for r in fmt))}")
print("  eval/test: v3.1 것 재사용(eval_v3_1·test_v3_1)")
