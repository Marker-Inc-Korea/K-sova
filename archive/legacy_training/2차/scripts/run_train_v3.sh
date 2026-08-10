#!/bin/bash
# v3 closed-book 5종 학습. GPU 0-3 사용(v6는 4-7, 안 건드림). completion_only_loss=True.
ROOT=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias/q1q2_lora
GEN=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen
VENV=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_project/train_test/finetune/venv/bin/python
export HF_HOME=$ROOT/models
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
BASETMP=/NHNHOME/WORKSPACE/26mfds002_B/kj/ktmp
mkdir -p $GEN/runs
train () {  # $1=model_key $2=gpu
  local T=$BASETMP/$1
  mkdir -p $T/triton $T/inductor
  CUDA_VISIBLE_DEVICES=$2 TMPDIR=$T TRITON_CACHE_DIR=$T/triton TORCHINDUCTOR_CACHE_DIR=$T/inductor \
  $VENV $GEN/train_lora_v2.py --model_key $1 \
    --train_file $GEN/train_v3.jsonl --eval_file $GEN/eval_v3.jsonl \
    --output_dir $GEN/runs/v3_lora_$1 --epochs 3 --max_seq_length 2048 \
    --bs 8 --eval_steps 200 --save_steps 200 > $GEN/train_v3_$1.log 2>&1
  echo "[$(date)] $1 DONE (exit $?)"
}
echo "[$(date)] === 1차: exaone/qwen/swallow/llama 동시 (GPU0-3) ==="
train exaone 0 &
train qwen   1 &
train swallow 2 &
train llama  3 &
wait
echo "[$(date)] === 2차: gemma (GPU0) ==="
train gemma 0 &
wait
echo "[$(date)] === ALL 5 DONE ==="
