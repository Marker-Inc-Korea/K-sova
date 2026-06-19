#!/bin/bash
# v5rf(format_v4 재추가) LL 평가 — base/v5r/v5rf 3원 비교용. /final 없으면 skip(학습 미완). 결과 있으면 skip(재실행 안전).
set -u
GEN=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen
ROOT=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias/q1q2_lora
LM=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_project/train_test/finetune/venv/bin/lm_eval
VENV=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_project/train_test/finetune/venv/bin/python
OUT=$GEN/lmeval_ksova; MRG=/NHNHOME/WORKSPACE/26mfds002_B/kj/ktmp/merged_v5rf; mkdir -p $MRG
export HF_HOME=$ROOT/models HF_DATASETS_CACHE=/home/e202/.cache/huggingface/datasets HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$GEN
TASKS=kmmlu,haerae,kobest
declare -A BASE=([exaone]="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct" [gemma]="google/gemma-4-E4B-it" [llama]="meta-llama/Llama-3.1-8B-Instruct" [qwen]="Qwen/Qwen3.5-9B" [swallow]="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3")
one_job(){
  local M=$1; local G=$2
  local od=$OUT/${M}_v5rf
  if [ ! -d $GEN/runs/v5rf_lora_$M/final ]; then echo "[wait] $M v5rf 학습미완 — skip"; return; fi
  if find $od -name "results_*.json" 2>/dev/null | grep -q .; then echo "[skip] $M/v5rf 결과있음"; return; fi
  local T=/NHNHOME/WORKSPACE/26mfds002_B/kj/ktmp/lme_${M}_v5rf; mkdir -p $T
  local args
  if [ "$M" = "exaone" ]; then
    local Mg=$MRG/exaone
    CUDA_VISIBLE_DEVICES=$G TMPDIR=$T $VENV $GEN/merge_lora.py exaone $GEN/runs/v5rf_lora_exaone/final $Mg > $OUT/exaone_v5rf_merge.log 2>&1
    args="pretrained=$Mg,dtype=bfloat16,trust_remote_code=True"
  else
    args="pretrained=${BASE[$M]},dtype=bfloat16,trust_remote_code=True,peft=$GEN/runs/v5rf_lora_$M/final"
  fi
  echo "[$(date '+%H:%M')] $M/v5rf lm_eval (GPU$G)"
  CUDA_VISIBLE_DEVICES=$G TMPDIR=$T $LM --model hf --model_args "$args" --tasks $TASKS --batch_size 8 --output $od > $OUT/${M}_v5rf.log 2>&1
  echo "[$(date '+%H:%M')] $M/v5rf done(exit $?)"
}
# exaone/swallow는 pass1에서 GPU0/1로 완료됨(skip). 나머지는 학습 끝나 비는 GPU5/6/7로(0-3은 full 방어실험이 사용).
one_job exaone 5 & one_job swallow 6 & one_job llama 5 & one_job gemma 6 & one_job qwen 7 & wait
echo "LMEVAL_V5RF_PASS_DONE"
