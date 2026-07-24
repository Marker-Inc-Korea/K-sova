#!/bin/bash
set -euo pipefail

ROOT=/data1/mk02/K-sova/benchmark/sovereign_alignment_1000
RUN=${ROOT}/workspace/run_20260723
EVAL=${RUN}/evaluation/divergent
OFF_REQUESTS=${EVAL}/requests_thinking_off.jsonl
ON_REQUESTS=${EVAL}/requests_thinking_on_144.jsonl
RUNNER=${ROOT}/slurm/run_vllm_stage.sbatch
PYTHON=/data1/mk02/.uv-python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11
SHARDS=${SHARDS:-4}
JUDGE_SHARDS=${JUDGE_SHARDS:-4}

mkdir -p "${EVAL}/targets" "${EVAL}/judging" "${RUN}/logs"
export PYTHONHOME=/data1/mk02/.uv-python/cpython-3.11.15-linux-x86_64-gnu
export PYTHONPATH=/data1/mk02/ksova-bench-build/.venv/lib/python3.11/site-packages

if [[ ! -s "${OFF_REQUESTS}" ]]; then
  "${PYTHON}" -B "${ROOT}/tools/pipeline.py" build-eval-requests \
    --items "${ROOT}/release/divergent_diagnostic_580.jsonl" \
    --output "${OFF_REQUESTS}" \
    --report "${EVAL}/requests_thinking_off.report.json" \
    --languages ko,en,ja,zh-Hans --max-tokens 2048
fi
if [[ ! -s "${ON_REQUESTS}" ]]; then
  "${PYTHON}" -B "${ROOT}/tools/pipeline.py" build-eval-requests \
    --items "${ROOT}/release/divergent_thinking_sample_144.jsonl" \
    --output "${ON_REQUESTS}" \
    --report "${EVAL}/requests_thinking_on_144.report.json" \
    --languages ko,en,ja,zh-Hans --max-tokens 8192
fi

models=(gemma qwen exaone swallow)
declare -A paths records
paths[gemma]='/data1/mk02/모델/gemma-4-31B-it'
paths[qwen]='/data1/mk02/hf_bench/local/Qwen--Qwen3.6-35B-A3B-FP8'
paths[exaone]='/data1/mk02/hf_bench/local/LGAI-EXAONE--EXAONE-4.5-33B-FP8-runtime'
paths[swallow]='/data1/mk02/hf_bench/local/tokyotech-llm--Qwen3-Swallow-30B-A3B-RL-v0.2-AWQ-INT4'
records[gemma]='gemma-4-31B-it'
records[qwen]='Qwen3.6-35B-A3B-FP8'
records[exaone]='EXAONE-4.5-33B-FP8'
records[swallow]='Qwen3-Swallow-30B-A3B-RL-v0.2-AWQ-INT4'

target_jobs=()
for model in "${models[@]}"; do
  for ((shard=0; shard<SHARDS; shard++)); do
    shard_tag=$(printf '%02d' "${shard}")
    if [[ "${model}" == "swallow" ]]; then
      job_id=$(sbatch --parsable \
        --job-name="ksa-div-${model}-${shard_tag}" \
        --export="ALL,MODEL_PATH=${paths[$model]},MODEL_KIND=${model},REQUESTS=${OFF_REQUESTS},OUTPUT=${EVAL}/targets/${model}_native_reasoning_s${shard_tag}.jsonl,RECORD_MODEL=${records[$model]}__native_reasoning,ENABLE_THINKING=true,MAX_TOKENS=8192,FORCE_MAX_TOKENS=8192,SHARD_COUNT=${SHARDS},SHARD_INDEX=${shard},CONCURRENCY=12,TEMPERATURE=0.6,TOP_P=0.95,GENERATION_PROFILE=swallow-recommended" \
        "${RUNNER}")
      target_jobs+=("${job_id}")
      continue
    fi
    job_id=$(sbatch --parsable \
      --job-name="ksa-div-${model}-${shard_tag}" \
      --export="ALL,MODEL_PATH=${paths[$model]},MODEL_KIND=${model},REQUESTS=${OFF_REQUESTS},OUTPUT=${EVAL}/targets/${model}_thinking_off_s${shard_tag}.jsonl,RECORD_MODEL=${records[$model]}__thinking_off,ENABLE_THINKING=false,MAX_TOKENS=2048,FORCE_MAX_TOKENS=2048,SECOND_REQUESTS=${ON_REQUESTS},SECOND_OUTPUT=${EVAL}/targets/${model}_thinking_on_s${shard_tag}.jsonl,SECOND_RECORD_MODEL=${records[$model]}__thinking_on,SECOND_ENABLE_THINKING=true,SECOND_MAX_TOKENS=8192,SECOND_FORCE_MAX_TOKENS=8192,SHARD_COUNT=${SHARDS},SHARD_INDEX=${shard},CONCURRENCY=12,TEMPERATURE=0.0,TOP_P=1.0,EXTRA_BODY_JSON={\"seed\":20260723}" \
      "${RUNNER}")
    target_jobs+=("${job_id}")
  done
done

target_dependency=$(IFS=:; echo "${target_jobs[*]}")
post_job=$(sbatch --parsable --job-name=ksa-div-post \
  --dependency="afterany:${target_dependency}" \
  "${ROOT}/slurm/postprocess_divergent_evaluation.sbatch")

judge_jobs=()
for judge in qwen exaone; do
  for ((shard=0; shard<JUDGE_SHARDS; shard++)); do
    shard_tag=$(printf '%02d' "${shard}")
    judge_requests=${EVAL}/judging/${judge}_requests.jsonl
    if [[ "${judge}" == "exaone" ]]; then
      judge_requests=${EVAL}/judging/exaone_requests_unconstrained.jsonl
    fi
    job_id=$(sbatch --parsable \
      --job-name="ksa-div-judge-${judge}-${shard_tag}" \
      --dependency="afterok:${post_job}" \
      --export="ALL,MODEL_PATH=${paths[$judge]},MODEL_KIND=${judge},REQUESTS=${judge_requests},OUTPUT=${EVAL}/judging/${judge}_s${shard_tag}.jsonl,RECORD_MODEL=${records[$judge]},ENABLE_THINKING=false,MAX_TOKENS=1024,FORCE_MAX_TOKENS=1024,SHARD_COUNT=${JUDGE_SHARDS},SHARD_INDEX=${shard},CONCURRENCY=8,MAX_MODEL_LEN=32768,MAX_NUM_SEQS=8,TEMPERATURE=0.0,TOP_P=1.0,EXTRA_BODY_JSON={\"seed\":20260723}" \
      "${RUNNER}")
    judge_jobs+=("${job_id}")
  done
done

judge_dependency=$(IFS=:; echo "${judge_jobs[*]}")
analysis_job=$(sbatch --parsable --job-name=ksa-div-analysis \
  --dependency="afterany:${judge_dependency}" \
  "${ROOT}/slurm/analyze_divergent_evaluation.sbatch")

printf 'target_jobs=%s\npostprocess_job=%s\njudge_jobs=%s\nanalysis_job=%s\n' \
  "${target_dependency}" "${post_job}" "${judge_dependency}" "${analysis_job}"
