#!/bin/bash
#SBATCH --job-name=ksa3-eval-swallow
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --mem=48G
#SBATCH --time=30:00:00
#SBATCH --output=/data1/mk02/K-sova/benchmark/sovereign_alignment_v3/outputs/logs/%x_%j.out
#SBATCH --error=/data1/mk02/K-sova/benchmark/sovereign_alignment_v3/outputs/logs/%x_%j.err

set -euo pipefail
ROOT=/data1/mk02/K-sova/benchmark/sovereign_alignment_v3
ENV=/data1/mk02/ksova-bench-build/.venv
RUNTIME=/data1/mk02/.uv-python/cpython-3.11.15-linux-x86_64-gnu
PYTHON=${RUNTIME}/bin/python3.11
MODEL=/data1/mk02/hf_bench/local/tokyotech-llm--Qwen3-Swallow-30B-A3B-RL-v0.2-AWQ-INT4
SERVED=qwen3-swallow-30b-ksa3-eval
PORT=$((20000 + SLURM_JOB_ID % 20000))
REQUESTS=${KSOVA_EVAL_REQUESTS:-${ROOT}/data/eval_requests_v0.jsonl}
export PYTHONHOME=${RUNTIME} PYTHONPATH=${ENV}/lib/python3.11/site-packages
export HF_HOME=/data1/mk02/hf_bench PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0
export HOME=/tmp/ksa3-${SLURM_JOB_ID} XDG_CACHE_HOME=/tmp/ksa3-${SLURM_JOB_ID}/cache
mkdir -p "${HOME}" "${XDG_CACHE_HOME}"
mkdir -p "${ROOT}/outputs/eval/swallow" "${ROOT}/outputs/logs"
LOG=${ROOT}/outputs/logs/swallow_eval_vllm_${SLURM_JOB_ID}.log
"${PYTHON}" "${ENV}/bin/vllm" serve "${MODEL}" --served-model-name "${SERVED}" --host 127.0.0.1 --port "${PORT}" \
  --gpu-memory-utilization 0.75 --max-model-len 8192 --max-num-seqs 16 \
  --trust-remote-code --enforce-eager --reasoning-parser qwen3 >"${LOG}" 2>&1 &
PID=$!
trap 'kill "${PID}" 2>/dev/null || true; wait "${PID}" 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null && break
  kill -0 "${PID}" 2>/dev/null || { tail -200 "${LOG}"; exit 1; }
  sleep 10
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null
"${PYTHON}" -B "${ROOT}/tools/run_chat_batch.py" --requests "${REQUESTS}" \
  --base-url "http://127.0.0.1:${PORT}/v1" --model "${SERVED}" --record-model-name qwen3_swallow30b_a3b_thinking4096 \
  --out-dir "${ROOT}/outputs/eval/swallow" --out-prefix qwen3_swallow30b_a3b_thinking4096 \
  --force-max-tokens 4096 --timeout 2400 --concurrency 6 --temperature 0.6 --top-p 0.95 \
  --chat-template-kwargs-json '{"enable_thinking":true}' \
  --extra-body-json '{"top_k":20,"min_p":0.0,"seed":20260714}'
