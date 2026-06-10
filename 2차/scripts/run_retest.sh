#!/bin/bash
# 교정 재테스트: 생성(GPU5-7, decoding/prompt 교정) → judge(Kimi 0-3) → 분리종합.
ROOT=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias/q1q2_lora
GEN=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_bot_1/sovereign-bias-v2/gen
TRAINVENV=/NHNHOME/WORKSPACE/26mfds002_B/kj/kj_project/train_test/finetune/venv/bin/python  # transformers/peft
VLLMVENV=/NHNHOME/WORKSPACE/26mfds002_B/dk/venv-vllm-020/bin/python3                          # openai(judge)
STRONG=$ROOT/configs/system_prompt_strong.txt
export HF_HOME=$ROOT/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$GEN
BASETMP=/NHNHOME/WORKSPACE/26mfds002_B/kj/ktmp
mkdir -p $GEN/runs/gen_v3b
cd $GEN
genone () {  # $1=model $2=gpu
  local T=$BASETMP/rt_$1; mkdir -p $T/triton $T/inductor
  CUDA_VISIBLE_DEVICES=$2 TMPDIR=$T TRITON_CACHE_DIR=$T/triton TORCHINDUCTOR_CACHE_DIR=$T/inductor \
  $TRAINVENV $GEN/gen_test_fixed.py --model_key $1 --adapter $GEN/runs/v3_lora_$1/final --strong_file $STRONG >> $GEN/retest_gen_$1.log 2>&1
  echo "[$(date)] $1 gen done"
}
echo "[$(date)] === 생성 1차: exaone(5) qwen(6) swallow(7) ==="
genone exaone 5 & genone qwen 6 & genone swallow 7 & wait
echo "[$(date)] === 생성 2차: llama(5) gemma(6) ==="
genone llama 5 & genone gemma 6 & wait
echo "[$(date)] === judge (Kimi :8000) ==="
for m in exaone qwen swallow llama gemma; do for c in base prompt prompt_strong lora; do
  $VLLMVENV $GEN/judge_v2.py --gen $GEN/runs/gen_v3b/gen_${m}_${c}.jsonl --test $GEN/test_v3_eval300.jsonl \
    --out $GEN/runs/gen_v3b/jt2_${m}_${c}.jsonl --concurrency 48 >> $GEN/retest_judge.log 2>&1
done; done
echo "[$(date)] === 종합 ==="
$VLLMVENV $GEN/summarize_retest.py > $GEN/retest_summary.txt 2>&1
cat $GEN/retest_summary.txt
echo "[$(date)] === RETEST DONE ==="
