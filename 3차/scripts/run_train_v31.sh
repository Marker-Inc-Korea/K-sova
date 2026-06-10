#!/bin/bash
# K-SOVA v3.1 5종 LoRA 학습. GPU 6,7만 사용(Kimi 0-3 / v7평가 4,5 안 건드림). completion_only_loss=True.
ROOT=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias/q1q2_lora
GEN=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen
VENV=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_project/train_test/finetune/venv/bin/python
export HF_HOME=$ROOT/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
BASETMP=/NHNHOME/WORKSPACE/26mfds002_B/kj/ktmp
mkdir -p $GEN/runs
train () {  # $1=model_key $2=gpu
  local T=$BASETMP/v31_$1; mkdir -p $T/triton $T/inductor
  CUDA_VISIBLE_DEVICES=$2 TMPDIR=$T TRITON_CACHE_DIR=$T/triton TORCHINDUCTOR_CACHE_DIR=$T/inductor \
  $VENV $GEN/train_lora_v2.py --model_key $1 \
    --train_file $GEN/train_v3_1.jsonl --eval_file $GEN/eval_v3_1.jsonl \
    --output_dir $GEN/runs/v31_lora_$1 --epochs 3 --max_seq_length 2048 \
    --bs 8 --eval_steps 200 --save_steps 200 > $GEN/train_v31_$1.log 2>&1
  echo "[$(date)] $1 DONE (exit $?)"
}
echo "[$(date)] === wave1: exaone(GPU6) + qwen(GPU7) ==="
train exaone 6 & train qwen 7 & wait
echo "[$(date)] === wave2: swallow(GPU6) + llama(GPU7) ==="
train swallow 6 & train llama 7 & wait
echo "[$(date)] === wave3: gemma(GPU6) ==="
train gemma 6 & wait
echo "[$(date)] === K-SOVA v3.1 5종 학습 완료 ==="
