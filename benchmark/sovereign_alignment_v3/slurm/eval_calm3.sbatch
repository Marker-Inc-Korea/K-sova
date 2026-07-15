#!/bin/bash
#SBATCH --job-name=ksa3-eval-calm3
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=/data1/mk02/K-sova/benchmark/sovereign_alignment_v3/outputs/logs/%x_%j.out
#SBATCH --error=/data1/mk02/K-sova/benchmark/sovereign_alignment_v3/outputs/logs/%x_%j.err

set -euo pipefail
ROOT=/data1/mk02/K-sova/benchmark/sovereign_alignment_v3
ENV=/data1/mk02/ksova-bench-build/.venv
RUNTIME=/data1/mk02/.uv-python/cpython-3.11.15-linux-x86_64-gnu
PYTHON=${RUNTIME}/bin/python3.11
MODEL=/data2/mk02/hf_cache/hub/models--cyberagent--calm3-22b-chat/snapshots/055922aa0f0fb1fbfbc97a2e31134532485ee99b
SERVED=calm3-22b-ksa3-eval
PORT=$((20000 + SLURM_JOB_ID % 20000))
REQUESTS=${KSOVA_EVAL_REQUESTS:-${ROOT}/data/eval_requests_v0.jsonl}
export PYTHONHOME=${RUNTIME} PYTHONPATH=${ENV}/lib/python3.11/site-packages
export HF_HOME=/data2/mk02/hf_cache PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0
export HOME=/tmp/ksa3-${SLURM_JOB_ID} XDG_CACHE_HOME=/tmp/ksa3-${SLURM_JOB_ID}/cache
mkdir -p "${HOME}" "${XDG_CACHE_HOME}"
mkdir -p "${ROOT}/outputs/eval/calm3" "${ROOT}/outputs/logs"
LOG=${ROOT}/outputs/logs/calm3_eval_vllm_${SLURM_JOB_ID}.log
"${PYTHON}" "${ENV}/bin/vllm" serve "${MODEL}" --served-model-name "${SERVED}" --host 127.0.0.1 --port "${PORT}" \
  --gpu-memory-utilization 0.86 --max-model-len 8192 --max-num-seqs 16 \
  --trust-remote-code --enforce-eager >"${LOG}" 2>&1 &
PID=$!
trap 'kill "${PID}" 2>/dev/null || true; wait "${PID}" 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null && break
  kill -0 "${PID}" 2>/dev/null || { tail -200 "${LOG}"; exit 1; }
  sleep 10
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null
"${PYTHON}" -B "${ROOT}/tools/run_chat_batch.py" --requests "${REQUESTS}" \
  --base-url "http://127.0.0.1:${PORT}/v1" --model "${SERVED}" --record-model-name calm3_22b \
  --out-dir "${ROOT}/outputs/eval/calm3" --out-prefix calm3_22b \
  --force-max-tokens 1024 --timeout 1800 --concurrency 8 --temperature 0.0 \
  --extra-body-json '{"seed":20260714}'
