#!/bin/bash
# K-SOVA 4차 5종 LoRA 학습. train_v4_clean(clean재생성+형식셋+강건성, 청크그룹split 누수0). GPU 4-7(Kimi 0-3 보존). Gemma는 타깃모듈(--no_all_linear).
ROOT=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias/q1q2_lora
GEN=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen
VENV=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_project/train_test/finetune/venv/bin/python
export HF_HOME=$ROOT/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
BASETMP=/NHNHOME/WORKSPACE/26mfds002_B/kj/ktmp
mkdir -p $GEN/runs
train () {  # $1=model_key $2=gpu $3=extra_args
  local T=$BASETMP/v4_$1; mkdir -p $T/triton $T/inductor
  CUDA_VISIBLE_DEVICES=$2 TMPDIR=$T TRITON_CACHE_DIR=$T/triton TORCHINDUCTOR_CACHE_DIR=$T/inductor \
  $VENV $GEN/train_lora_v2.py --model_key $1 \
    --train_file $GEN/train_v4_clean.jsonl --eval_file $GEN/eval_v4.jsonl \
    --output_dir $GEN/runs/v4_lora_$1 --epochs 3 --max_seq_length 2048 \
    --bs 8 --eval_steps 200 --save_steps 200 $3 > $GEN/train_v4_$1.log 2>&1
  echo "[$(date)] $1 DONE (exit $?)"
}
echo "[$(date)] === wave1: exaone(4) qwen(5) swallow(6) llama(7) ==="
train exaone 4 & train qwen 5 & train swallow 6 & train llama 7 & wait
echo "[$(date)] === wave2: gemma(4) — 타깃모듈(--no_all_linear) ==="
train gemma 4 "--no_all_linear" & wait
echo "[$(date)] === K-SOVA 4차 5종 학습 완료 ==="
