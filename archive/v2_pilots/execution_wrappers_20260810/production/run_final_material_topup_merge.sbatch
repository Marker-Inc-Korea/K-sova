#!/bin/bash
#SBATCH --job-name=ksv2-material-topup-merge
#SBATCH --partition=batch
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
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

MERGE_INPUTS=()
for index in $(seq 0 $((SHARD_COUNT - 1))); do
  printf -v SHARD '%02d' "${index}"
  printf -v COUNT '%02d' "${SHARD_COUNT}"
  MERGE_INPUTS+=(--input "${WORK}/material_topup_responses.shard-${SHARD}-of-${COUNT}.jsonl")
done

"${PYTHON}" -B "${ROOT}/merge_jsonl_responses.py" \
  --requests "${WORK}/material_topup_requests.jsonl" "${MERGE_INPUTS[@]}" \
  --output "${WORK}/material_topup_responses.jsonl" \
  --report "${WORK}/material_topup_response_merge_report.json"

"${PYTHON}" -B "${ROOT}/material_pipeline.py" parse \
  --catalog "${WORK}/material_topup_catalog.jsonl" \
  --requests "${WORK}/material_topup_requests.jsonl" \
  --responses "${WORK}/material_topup_responses.jsonl" \
  --output "${WORK}/material_topup_seeds.jsonl" \
  --sources "${WORK}/material_topup_sources.jsonl" \
  --accepted "${WORK}/material_topup_atoms.jsonl" \
  --report "${WORK}/material_topup_report.json"

"${PYTHON}" -B "${ROOT}/merge_material_pools.py" \
  --primary-seeds "${WORK}/material_seeds.jsonl" \
  --topup-seeds "${WORK}/material_topup_seeds.jsonl" \
  --primary-sources "${WORK}/material_sources.jsonl" \
  --topup-sources "${WORK}/material_topup_sources.jsonl" \
  --output-seeds "${WORK}/material_seeds.topup_merged.jsonl" \
  --output-sources "${WORK}/material_sources.topup_merged.jsonl" \
  --duplicate-audit "${WORK}/material_topup_duplicate_audit.jsonl" \
  --report "${WORK}/material_pool_merge_report.json"

test -s "${WORK}/material_expansion_seeds.jsonl"
"${PYTHON}" -B "${ROOT}/merge_material_pools.py" \
  --primary-seeds "${WORK}/material_seeds.topup_merged.jsonl" \
  --topup-seeds "${WORK}/material_expansion_seeds.jsonl" \
  --primary-sources "${WORK}/material_sources.topup_merged.jsonl" \
  --topup-sources "${WORK}/material_expansion_sources.jsonl" \
  --output-seeds "${WORK}/material_seeds.merged.jsonl" \
  --output-sources "${WORK}/material_sources.merged.jsonl" \
  --duplicate-audit "${WORK}/material_expansion_duplicate_audit.jsonl" \
  --report "${WORK}/material_expansion_merge_report.json"

"${PYTHON}" -B "${ROOT}/audit_source_pool_security.py" \
  --sources "${WORK}/material_sources.merged.jsonl" \
  --policy "${REPO}/benchmark/v2.0/source_pool/discovery_queries.json" \
  --output "${WORK}/SOURCE_SECURITY_AUDIT.json"

"${PYTHON}" -B "${ROOT}/pipeline.py" build-author \
  --seeds "${WORK}/material_seeds.merged.jsonl" \
  --sources "${WORK}/material_sources.merged.jsonl" \
  --run-id "${RUN_ID}" --frame-policy type_octet \
  --output "${WORK}/author_requests.jsonl" --snapshot "${WORK}/source_snapshot.jsonl" \
  --report "${WORK}/author_plan.json"

"${PYTHON}" -B "${ROOT}/shard_jsonl_requests.py" \
  --input "${WORK}/author_requests.jsonl" --output-dir "${WORK}" \
  --prefix author_requests --shards "${SHARD_COUNT}" --group-metadata-key seed_id \
  --report "${WORK}/author_shard_plan.json"

printf 'K-SOVA material top-up merge and author planning complete: %s\n' "${WORK}/author_requests.jsonl"
