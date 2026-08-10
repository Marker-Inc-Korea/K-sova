#!/bin/bash
# v3 평가 1단계: 생성(Kimi 다운, GPU0-3) + 지식프로브 결정채점. judge/KorNAT점수는 Kimi 재기동 후 별도.
ROOT=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias/q1q2_lora
GEN=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen
VENV=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_project/train_test/finetune/venv/bin/python
STRONG=$ROOT/configs/system_prompt_strong.txt
KORNAT=$ROOT/data/kornat_500.jsonl
export HF_HOME=$ROOT/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$GEN
BASETMP=/NHNHOME/WORKSPACE/26mfds002_B/kj/ktmp
mkdir -p $GEN/runs/gen_v3
cd $GEN
genone () {  # $1=model $2=gpu
  local T=$BASETMP/eval_$1; mkdir -p $T/triton $T/inductor
  local E="CUDA_VISIBLE_DEVICES=$2 TMPDIR=$T TRITON_CACHE_DIR=$T/triton TORCHINDUCTOR_CACHE_DIR=$T/inductor"
  env $E $VENV $GEN/gen_all_v3.py --model_key $1 --adapter $GEN/runs/v3_lora_$1/final --strong_file $STRONG >> $GEN/eval_gen_$1.log 2>&1
  for c in base prompt lora; do
    adp=""; [ "$c" = "lora" ] && adp="--adapter $GEN/runs/v3_lora_$1/final"
    env $E $VENV $GEN/eval_kornat_v2.py --model_key $1 --condition $c --input $KORNAT \
      --output $GEN/runs/gen_v3/kn_${1}_${c}.jsonl $adp >> $GEN/eval_gen_$1.log 2>&1
  done
  echo "[$(date)] $1 생성완료"
}
echo "[$(date)] === 생성 1차: exaone/qwen/swallow/llama (GPU0-3) ==="
genone exaone 0 & genone qwen 1 & genone swallow 2 & genone llama 3 & wait
echo "[$(date)] === 생성 2차: gemma (GPU0) ==="
genone gemma 0 & wait
echo "[$(date)] === 지식프로브 결정채점 ==="
$VENV $GEN/knowledge_grade.py --probe $GEN/knowledge_probe.jsonl \
  --gen $GEN/runs/gen_v3/kp_exaone_base.jsonl $GEN/runs/gen_v3/kp_exaone_lora.jsonl \
        $GEN/runs/gen_v3/kp_qwen_base.jsonl $GEN/runs/gen_v3/kp_qwen_lora.jsonl \
        $GEN/runs/gen_v3/kp_swallow_base.jsonl $GEN/runs/gen_v3/kp_swallow_lora.jsonl \
        $GEN/runs/gen_v3/kp_llama_base.jsonl $GEN/runs/gen_v3/kp_llama_lora.jsonl \
        $GEN/runs/gen_v3/kp_gemma_base.jsonl $GEN/runs/gen_v3/kp_gemma_lora.jsonl
echo "[$(date)] === 생성+프로브채점 DONE (다음: Kimi 재기동 → judge_v2 + KorNAT점수) ==="
