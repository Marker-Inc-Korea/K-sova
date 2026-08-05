#!/bin/bash
#SBATCH --job-name=ksv2-material-topup
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=180G
#SBATCH --time=05:00:00
#SBATCH --output=/data1/mk02/K-sova/benchmark/v2.0/production/work/logs/%x_%A_%a.out
#SBATCH --error=/data1/mk02/K-sova/benchmark/v2.0/production/work/logs/%x_%A_%a.err

set -euo pipefail

REPO=/data1/mk02/K-sova
ROOT=${REPO}/benchmark/v2.0/production
RUN_ID=${KSOVA_RUN_ID:-final1000}
WORK=${ROOT}/work/${RUN_ID}
LOG_DIR=${ROOT}/work/logs
ENV=/data1/mk02/ksova-bench-build/.venv
RUNTIME=/data1/mk02/.uv-python/cpython-3.11.15-linux-x86_64-gnu
PYTHON=${RUNTIME}/bin/python3.11
MODEL=/data1/mk02/hf_bench/local/Qwen--Qwen3.6-35B-A3B-FP8
SERVED=qwen36-35b-a3b-ksova-v2-material-topup-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
PORT=$((22000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID) % 18000))
SHARD_COUNT=${KSOVA_SHARD_COUNT:?KSOVA_SHARD_COUNT is required}
printf -v SHARD '%02d' "${SLURM_ARRAY_TASK_ID}"
printf -v COUNT '%02d' "${SHARD_COUNT}"
REQUESTS=${WORK}/material_topup_requests.shard-${SHARD}-of-${COUNT}.jsonl
RAW=${WORK}/material_topup_responses.shard-${SHARD}-of-${COUNT}.jsonl
VLLM_LOG=${LOG_DIR}/${RUN_ID}_qwen_material_topup_vllm_${SLURM_ARRAY_JOB_ID}_${SHARD}.log
MAX_NUM_SEQS=${KSOVA_MAX_NUM_SEQS:-32}

mkdir -p "${WORK}" "${LOG_DIR}"
test -s "${REQUESTS}"
test -d "${MODEL}"
export PYTHONHOME=${RUNTIME} PYTHONPATH=${ENV}/lib/python3.11/site-packages
export HF_HOME=/data1/mk02/hf_bench PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0
export HOME=/tmp/ksova-final-material-topup-${SLURM_JOB_ID}
export XDG_CACHE_HOME=${HOME}/cache
mkdir -p "${HOME}" "${XDG_CACHE_HOME}"

"${PYTHON}" "${ENV}/bin/vllm" serve "${MODEL}" \
  --served-model-name "${SERVED}" --host 127.0.0.1 --port "${PORT}" \
  --gpu-memory-utilization 0.94 --max-model-len 16384 --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens 65536 --generation-config vllm --language-model-only \
  --trust-remote-code --enforce-eager --reasoning-parser qwen3 >"${VLLM_LOG}" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true' EXIT

for _ in $(seq 1 240); do
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && break
  kill -0 "${SERVER_PID}" 2>/dev/null || { tail -200 "${VLLM_LOG}"; exit 1; }
  sleep 5
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null

MATERIAL_BODY=$(jq -c \
  '{response_format:{type:"json_schema",json_schema:{name:"ksova_v2_material_topup",strict:true,schema:.}},seed:20260805}' \
  "${ROOT}/material_schema.json")

"${PYTHON}" -B "${REPO}/benchmark/sovereign_alignment_v4/tools/run_chat_batch.py" \
  --requests "${REQUESTS}" --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "${SERVED}" --record-model-name "qwen3.6-35B-A3B-FP8__ksova-v2-material-topup" \
  --out-dir "${WORK}" --output-file "${RAW}" --force-max-tokens 4096 \
  --timeout 1800 --retries 1 --concurrency "${MAX_NUM_SEQS}" --temperature 0.15 --top-p 0.9 \
  --resume --resume-allow-higher-max-tokens \
  --chat-template-kwargs-json '{"enable_thinking":false}' --extra-body-json "${MATERIAL_BODY}"

if [[ "${SLURM_ARRAY_TASK_ID}" -eq 1 ]]; then
  bash "${ROOT}/run_live_source_expansion.sh" "http://127.0.0.1:${PORT}/v1" "${SERVED}"
fi

printf 'K-SOVA material top-up shard %s/%s complete: %s\n' "${SHARD}" "${COUNT}" "${RAW}"
