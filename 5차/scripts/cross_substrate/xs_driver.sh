#!/bin/bash
# cross-substrate 정량화: 5모델 base vs v5r 11쟁점 생성 → gemma 한국입장 정렬 채점 → 교정 집계
set -u
GEN=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen
ROOT=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias/q1q2_lora
VENV=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_project/train_test/finetune/venv/bin/python
OUTD=$GEN/xs_gen; mkdir -p $OUTD
export HF_HOME=$ROOT/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$GEN
cd $GEN
g(){ local m=$1 gp=$2; local T=/NHNHOME/WORKSPACE/26mfds002_B/kj/ktmp/xs_$m; mkdir -p $T/triton
  CUDA_VISIBLE_DEVICES=$gp TMPDIR=$T TRITON_CACHE_DIR=$T/triton $VENV $GEN/gen_crosssubstrate.py --model $m --out $OUTD/$m.jsonl > $GEN/xs_$m.log 2>&1
  echo "[$m] gen exit=$?"; }
g exaone 0 & g gemma 1 & g llama 2 & g qwen 3 & g swallow 4 & wait
echo "=== cross-substrate gemma 정렬 채점 ==="
T=/NHNHOME/WORKSPACE/26mfds002_B/kj/ktmp/xs_judge; mkdir -p $T/triton
CUDA_VISIBLE_DEVICES=0 TMPDIR=$T TRITON_CACHE_DIR=$T/triton $VENV $GEN/judge_crosssubstrate.py --answers "$OUTD/*.jsonl" --out $GEN/xs_result.json
echo "XS_ALL_DONE"
