#!/bin/bash
# K-SOVA 5차 LoRA 재학습 — GPU 1대에서 주어진 모델 순차(포크없음). 4차 설정 + train_v5(보강 최신성).
# 사용: train_v5_gpu.sh <gpu> "<model1> <model2> ..."
set -u
GPU=$1; shift; MODELS="$*"
GEN=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen
ROOT=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias/q1q2_lora
VENV=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_project/train_test/finetune/venv/bin/python
export HF_HOME=$ROOT/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$GEN
TR=$GEN/train_v5.jsonl; EV=$GEN/eval_v4.jsonl
cd $GEN
for M in $MODELS; do
  od=$GEN/runs/v5_lora_$M
  if [ -d $od/final ]; then echo "[skip] $M 완료"; continue; fi
  EX=""; [ "$M" = "gemma" ] && EX="--no_all_linear"
  T=/NHNHOME/WORKSPACE/26mfds002_B/kj/ktmp/v5_$M; mkdir -p $T/triton $T/inductor
  echo "[$(date '+%m/%d %H:%M')] $M 학습 시작 (GPU$GPU, 4차설정 epochs3·lr2e-4·r16$EX)"
  CUDA_VISIBLE_DEVICES=$GPU TMPDIR=$T TRITON_CACHE_DIR=$T/triton TORCHINDUCTOR_CACHE_DIR=$T/inductor \
    $VENV $GEN/train_lora_v2.py --model_key $M --train_file $TR --eval_file $EV \
    --output_dir $od --epochs 3 --lr 2e-4 --lora_r 16 --lora_alpha 32 \
    --max_seq_length 2048 --bs 8 --eval_steps 200 --save_steps 200 $EX > $GEN/train_v5_$M.log 2>&1
  echo "[$(date '+%m/%d %H:%M')] $M 완료(exit $?) $([ -d $od/final ] && echo OK || echo FAIL)"
done
echo "V5_TRAIN_DONE_GPU$GPU"
