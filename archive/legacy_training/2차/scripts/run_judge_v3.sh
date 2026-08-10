#!/bin/bash
# v3 평가 2단계: Kimi 재기동 후 실행. test300 4축 judge + 범용 judge + KorNAT 점수 + 종합.
GEN=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen
VENV=/NHNHOME/WORKSPACE/26mfds002_B/dk/venv-vllm-020/bin/python3   # openai 클라 있는 vllm venv
G=$GEN/runs/gen_v3
export KIMI_BASE_URL=http://localhost:8000/v1 KIMI_MODEL=kimi-k2.6 JUDGE_CONCURRENCY=32
cd $GEN
echo "[$(date)] === test300 4축 judge (5모델×4조건) ==="
for m in exaone qwen swallow llama gemma; do
  for c in base prompt prompt_strong lora; do
    [ -f $G/gen_${m}_${c}.jsonl ] || { echo "skip $m $c (gen없음)"; continue; }
    $VENV judge_v2.py --gen $G/gen_${m}_${c}.jsonl --test $GEN/test_v3_eval300.jsonl \
      --out $G/jt_${m}_${c}.jsonl --concurrency 48
  done
done
echo "[$(date)] === 범용 judge (5모델×base/lora) ==="
for m in exaone qwen swallow llama gemma; do
  for c in base lora; do
    [ -f $G/gg_${m}_${c}.jsonl ] || continue
    $VENV judge_general.py --input $G/gg_${m}_${c}.jsonl --output $G/jg_${m}_${c}.jsonl
  done
done
echo "[$(date)] === KorNAT 공식 점수 ==="
mkdir -p $GEN/runs/kornat; cp $G/kn_*.jsonl $GEN/runs/kornat/ 2>/dev/null
mkdir -p $GEN/analysis
$VENV kornat_official.py
echo "[$(date)] === 종합 ==="
$VENV summarize_v3.py
echo "[$(date)] === JUDGE ALL DONE ==="
