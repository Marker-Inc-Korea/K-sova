#!/bin/bash
ROOT=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias/q1q2_lora
GEN=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen
VENV=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_project/train_test/finetune/venv/bin/python
export HF_HOME=$ROOT/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR=/NHNHOME/WORKSPACE/26mfds002_B/kj/ktmp TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor
mkdir -p $GEN/runs/gen
gentest () {  # $1=model $2=gpu
  for cond in base lora lora_continued; do
    adp=""
    [ "$cond" = "lora" ] && adp="--adapter $GEN/runs/v_lora_$1/final"
    [ "$cond" = "lora_continued" ] && adp="--adapter $GEN/runs/v_lora_continued_$1/final"
    CUDA_VISIBLE_DEVICES=$2 $VENV $GEN/generate_v2.py --model_key $1 --condition $cond \
      --test_file $GEN/test.jsonl --output $GEN/runs/gen/gen_${1}_${cond}.jsonl $adp
  done
}
KORNAT=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias/q1q2_lora/data/kornat_500.jsonl
genkornat () {  # $1=model $2=gpu
  for cond in base lora lora_continued; do
    adp=""
    [ "$cond" = "lora" ] && adp="--adapter $GEN/runs/v_lora_$1/final"
    [ "$cond" = "lora_continued" ] && adp="--adapter $GEN/runs/v_lora_continued_$1/final"
    CUDA_VISIBLE_DEVICES=$2 $VENV $GEN/eval_kornat_v2.py --model_key $1 --condition $cond \
      --input $KORNAT --output $GEN/runs/gen/kn_${1}_${cond}.jsonl $adp
  done
}
STRONGFILE=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias/q1q2_lora/scripts/configs/system_prompt_strong.txt
genprompt () {  # $1=model $2=gpu  (base모델 + 시스템프롬프트)
  CUDA_VISIBLE_DEVICES=$2 $VENV $GEN/generate_v2.py --model_key $1 --condition prompt \
    --test_file $GEN/test.jsonl --output $GEN/runs/gen/gen_${1}_prompt.jsonl
  CUDA_VISIBLE_DEVICES=$2 $VENV $GEN/generate_v2.py --model_key $1 --condition prompt_strong \
    --test_file $GEN/test.jsonl --output $GEN/runs/gen/gen_${1}_prompt_strong.jsonl --system_prompt "$(cat $STRONGFILE)"
}
