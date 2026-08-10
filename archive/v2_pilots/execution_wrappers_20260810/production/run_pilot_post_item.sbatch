#!/bin/bash
#SBATCH --job-name=ksova-post-item
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/data1/mk02/K-sova/benchmark/v2.0/work/slurm_logs/%x_%j.out
#SBATCH --error=/data1/mk02/K-sova/benchmark/v2.0/work/slurm_logs/%x_%j.err

set -euo pipefail

REPO=/data1/mk02/K-sova
V2=${REPO}/benchmark/v2.0
RUN_DIR=${KSOVA_RUN_DIR:?set KSOVA_RUN_DIR}
PYTHON=${KSOVA_PIPELINE_PYTHON:-/data1/mk02/fsr/.venv-eval/bin/python}
ITEM=${RUN_DIR}/item_validation
OWNER=${RUN_DIR}/owner_review

test -s "${ITEM}/qwen_item_judged.jsonl"
test -s "${ITEM}/oss_item_judged.jsonl"
mkdir -p "${OWNER}"

"${PYTHON}" -B "${V2}/ksova_pipeline.py" merge-item-validation \
  --run-dir "${RUN_DIR}" \
  --qwen "${ITEM}/qwen_item_judged.jsonl" \
  --oss "${ITEM}/oss_item_judged.jsonl" \
  --accepted "${ITEM}/accepted_items.jsonl" \
  --rejected "${ITEM}/rejected_items.jsonl"

"${PYTHON}" -B "${V2}/ksova_pipeline.py" select-owner-review \
  --run-dir "${RUN_DIR}" \
  --output "${OWNER}/review_candidates_500.jsonl" \
  --report "${OWNER}/review_selection_report.json" \
  --distinct-issues 300 --judgment-items 400 --control-items 100

"${PYTHON}" -B "${V2}/owner_review_app.py" build-package \
  --candidates "${OWNER}/review_candidates_500.jsonl" \
  --sources "${V2}/work/current/sources/source_registry_1576.jsonl" \
  --output "${OWNER}/owner_review_package.json" \
  --report "${OWNER}/owner_review_package_report.json"

"${PYTHON}" -B "${V2}/ksova_pipeline.py" seal-owner-review-package \
  --run-dir "${RUN_DIR}" \
  --package "${OWNER}/owner_review_package.json"

printf 'K-SOVA owner review package ready: %s\n' "${OWNER}/owner_review_package.json"
