#!/bin/bash
#SBATCH --job-name=ksova-v21-issues
#SBATCH --partition=batch
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/data1/mk02/K-sova/benchmark/v2.0/v2_1/logs/%x_%j.out
#SBATCH --error=/data1/mk02/K-sova/benchmark/v2.0/v2_1/logs/%x_%j.err

set -euo pipefail

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo "GPU work must run inside Slurm" >&2; exit 2; }

REPO=/data1/mk02/K-sova
V2=${REPO}/benchmark/v2.0
RUN=${V2}/v2_1/issues
LOGS=${V2}/v2_1/logs
ENV=/data1/mk02/ksova-bench-build/.venv
PYTHON=${ENV}/bin/python
VLLM=${ENV}/bin/vllm
MODEL=/data1/hub/models--google--gemma-4-31B-it/snapshots/fb9ae262347c3945692f09a612f8bb189def854f
SERVED=ksova-v2-1-gemma-4-31b
PORT=$((23000 + SLURM_JOB_ID % 7000))
REQUESTS=${RUN}/requests.jsonl
RESPONSES=${RUN}/responses.jsonl

mkdir -p "${RUN}" "${LOGS}"
test -x "${PYTHON}"
test -x "${VLLM}"
test -d "${MODEL}"

"${PYTHON}" -B "${V2}/ksova_v2_1.py" build-issue-requests \
  --sources "${V2}/source_inventory_v2_1.jsonl" \
  --output "${REQUESTS}" --report "${RUN}/request_report.json"

export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export HOME=/tmp/ksova-v2-1-issues-${SLURM_JOB_ID}
export XDG_CACHE_HOME=${HOME}/cache
mkdir -p "${HOME}" "${XDG_CACHE_HOME}"

"${VLLM}" serve "${MODEL}" \
  --served-model-name "${SERVED}" --host 127.0.0.1 --port "${PORT}" \
  --gpu-memory-utilization 0.94 --max-model-len 16384 \
  --max-num-seqs 24 --max-num-batched-tokens 65536 \
  --language-model-only --reasoning-parser gemma4 \
  --trust-remote-code --enforce-eager >"${LOGS}/issues_${SLURM_JOB_ID}.vllm.log" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true; wait "${SERVER_PID}" 2>/dev/null || true' EXIT

for _ in $(seq 1 180); do
  curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null && break
  kill -0 "${SERVER_PID}" 2>/dev/null || { tail -200 "${LOGS}/issues_${SLURM_JOB_ID}.vllm.log"; exit 1; }
  sleep 5
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null

"${PYTHON}" -B "${REPO}/benchmark/v2.0/tools/run_chat_batch.py" \
  --requests "${REQUESTS}" --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "${SERVED}" --record-model-name gemma-4-31B-it__ksova-v2-1-issue-extractor \
  --out-dir "${RUN}" --output-file "${RESPONSES}" \
  --force-max-tokens 2400 --timeout 1800 --retries 1 --concurrency 24 \
  --temperature 0.15 --top-p 0.9 --resume --resume-allow-higher-max-tokens \
  --chat-template-kwargs-json '{"enable_thinking":false}'

"${PYTHON}" -B "${V2}/ksova_v2_1.py" parse-issues \
  --sources "${V2}/source_inventory_v2_1.jsonl" \
  --requests "${REQUESTS}" --responses "${RESPONSES}" \
  --accepted "${RUN}/accepted.jsonl" --rejected "${RUN}/rejected.jsonl" \
  --report "${RUN}/parse_report.json"
