#!/bin/bash
#SBATCH --job-name=ksa3-sel-gptoss
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --mem=90G
#SBATCH --time=12:00:00
#SBATCH --output=/data1/mk02/K-sova/benchmark/sovereign_alignment_v3/outputs/logs/%x_%j.out
#SBATCH --error=/data1/mk02/K-sova/benchmark/sovereign_alignment_v3/outputs/logs/%x_%j.err

set -euo pipefail
ROOT=/data1/mk02/K-sova/benchmark/sovereign_alignment_v3
ENV=/data1/mk02/ksova-bench-build/.venv
RUNTIME=/data1/mk02/.uv-python/cpython-3.11.15-linux-x86_64-gnu
PYTHON=${RUNTIME}/bin/python3.11
MODEL=/data1/mk02/모델/gpt-oss-120b
SERVED=gpt-oss-120b-ksa3-selection
PORT=$((20000 + SLURM_JOB_ID % 20000))
REQUESTS=${KSOVA_REQUESTS:-${ROOT}/data/teacher_requests_selection_v0.jsonl}
PREFIX=${KSOVA_PREFIX:-gpt_oss_120b_high_selection}
export PYTHONHOME=${RUNTIME} PYTHONPATH=${ENV}/lib/python3.11/site-packages
export HF_HOME=/data1/mk02/hf_bench PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0
export HOME=/tmp/ksa3-${SLURM_JOB_ID} XDG_CACHE_HOME=/tmp/ksa3-${SLURM_JOB_ID}/cache
mkdir -p "${HOME}" "${XDG_CACHE_HOME}"
mkdir -p "${ROOT}/outputs/teacher_selection/gpt_oss_120b" "${ROOT}/outputs/logs"
LOG=${ROOT}/outputs/logs/gptoss_selection_vllm_${SLURM_JOB_ID}.log
"${PYTHON}" "${ENV}/bin/vllm" serve "${MODEL}" --served-model-name "${SERVED}" --host 127.0.0.1 --port "${PORT}" \
  --gpu-memory-utilization 0.92 --max-model-len 24576 --trust-remote-code --enforce-eager >"${LOG}" 2>&1 &
PID=$!
trap 'kill "${PID}" 2>/dev/null || true; wait "${PID}" 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null && break
  kill -0 "${PID}" 2>/dev/null || { tail -200 "${LOG}"; exit 1; }
  sleep 10
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null
"${PYTHON}" -B "${ROOT}/tools/run_chat_batch.py" --requests "${REQUESTS}" \
  --base-url "http://127.0.0.1:${PORT}/v1" --model "${SERVED}" --record-model-name gpt_oss_120b_high \
  --out-dir "${ROOT}/outputs/teacher_selection/gpt_oss_120b" --out-prefix "${PREFIX}" \
  --max-tokens 12288 --timeout 3000 --concurrency 2 --temperature 0.2 --top-p 0.95 \
  --system-prefix 'Reasoning: high' --extra-body-json '{"response_format":{"type":"json_object"},"seed":20260714}'
