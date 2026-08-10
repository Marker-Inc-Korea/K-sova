#!/bin/bash
#SBATCH --job-name=ksv2-author-merge
#SBATCH --partition=batch
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/data1/mk02/K-sova/benchmark/v2.0/production/work/logs/%x_%j.out
#SBATCH --error=/data1/mk02/K-sova/benchmark/v2.0/production/work/logs/%x_%j.err

set -euo pipefail

REPO=/data1/mk02/K-sova
ROOT=${REPO}/benchmark/v2.0/production
RUN_ID=${KSOVA_RUN_ID:-final1000}
WORK=${ROOT}/work/${RUN_ID}
RUNTIME=/data1/mk02/.uv-python/cpython-3.11.15-linux-x86_64-gnu
PYTHON=${RUNTIME}/bin/python3.11
SHARD_COUNT=${KSOVA_SHARD_COUNT:?KSOVA_SHARD_COUNT is required}
CANONICAL_REQUESTS=${KSOVA_AUTHOR_CANONICAL_REQUESTS:-${WORK}/author_requests.jsonl}
RESPONSE_PREFIX=${KSOVA_AUTHOR_RESPONSE_PREFIX:-author_responses}
SEEDS=${KSOVA_AUTHOR_SEEDS:-${WORK}/material_seeds.merged.jsonl}

MERGE_INPUTS=()
if [[ -n "${KSOVA_AUTHOR_REUSABLE_RESPONSES:-}" ]]; then
  MERGE_INPUTS+=(--input "${KSOVA_AUTHOR_REUSABLE_RESPONSES}")
fi
for index in $(seq 0 $((SHARD_COUNT - 1))); do
  printf -v SHARD '%02d' "${index}"
  printf -v COUNT '%02d' "${SHARD_COUNT}"
  MERGE_INPUTS+=(--input "${WORK}/${RESPONSE_PREFIX}.shard-${SHARD}-of-${COUNT}.jsonl")
done

"${PYTHON}" -B "${ROOT}/merge_jsonl_responses.py" \
  --requests "${CANONICAL_REQUESTS}" "${MERGE_INPUTS[@]}" \
  --output "${WORK}/author_responses.jsonl" \
  --report "${WORK}/author_response_merge_report.json"

"${PYTHON}" -B "${ROOT}/pipeline.py" parse-author \
  --seeds "${SEEDS}" --snapshot "${WORK}/source_snapshot.jsonl" \
  --requests "${CANONICAL_REQUESTS}" --responses "${WORK}/author_responses.jsonl" \
  --output "${WORK}/candidates.jsonl" --report "${WORK}/author_parse_report.json"

printf 'K-SOVA author response merge and structural validation complete: %s\n' "${WORK}/candidates.jsonl"
