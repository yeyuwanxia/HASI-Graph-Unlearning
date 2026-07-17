#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
CONDA_ENV="${CONDA_ENV:-base}"
RQ_ROOT="${RQ_ROOT:-results/mia_v2_rq_fixed/pubmed}"
GRAPH_COMPUTE_BACKEND="${GRAPH_COMPUTE_BACKEND:-torch}"
HUB_PPR_BATCH_SIZE="${HUB_PPR_BATCH_SIZE:-64}"
HUB_SCORE_CACHE_ROOT="${HUB_SCORE_CACHE_ROOT:-results/mia_v2_pubmed_eval/hasi/artifacts/hub_scores}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"
QUIET_RESULT_JSON="${QUIET_RESULT_JSON:-1}"
export STRUCTURAL_METRICS_WORKERS="${STRUCTURAL_METRICS_WORKERS:-8}"
export STRUCTURAL_METRICS_PARALLEL_MIN_EDGES="${STRUCTURAL_METRICS_PARALLEL_MIN_EDGES:-100000}"
PY=(conda run --no-capture-output -n "$CONDA_ENV" python)

RQ_METRICS="metrics.utility.accuracy_after,metrics.utility.accuracy_drop,metrics.utility.f1_macro_after,metrics.structure.degree_kl_divergence,metrics.structure.clustering_coefficient_change,metrics.structure.component_count_change,metrics.privacy.medium_auc,metrics.privacy.medium_train_size,metrics.privacy.medium_eval_size,metrics.privacy.strong_auc,metrics.privacy.strong_auc_null_mean,metrics.privacy.strong_auc_null_std,metrics.privacy.strong_auc_pvalue,metrics.privacy.privacy_score,metrics.representation.embedding_l2_mean,metrics.representation.member_embedding_l2_mean,metrics.representation.neighbor_drift_mean,metrics.efficiency.unlearn_time_seconds,metrics.efficiency.online_wall_clock_seconds,metrics.efficiency.offline_preprocessing_seconds"

if [[ "$DRY_RUN" != "1" ]]; then
  "${PY[@]}" -c 'import sys, torch; ok=torch.cuda.is_available(); print(f"CUDA available: {ok}, torch={torch.__version__}, runtime={torch.version.cuda}"); sys.exit(0 if ok else "CUDA is unavailable in this environment; fix the PyTorch/driver match before formal RQ runs.")'
fi

rq_run() {
  local output="$1"
  shift
  local cmd=(
    "${PY[@]}" experiments/run_hasi.py
    --mode unlearn
    --dataset_name pubmed
    --config configs/hasi_default.yaml
    --unlearning_type node
    --base_artifact_root results/shared_base
    --device cuda:0
    --graph_compute_backend "$GRAPH_COMPUTE_BACKEND"
    --hub_ppr_batch_size "$HUB_PPR_BATCH_SIZE"
    --hub-score-cache
    --hub_score_cache_root "$HUB_SCORE_CACHE_ROOT"
    --require-hub-score-cache-hit
    --output "$output"
    "$@"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return
  fi
  mkdir -p "$(dirname "$output")"
  if [[ -s "$output" && "$OVERWRITE" != "1" ]]; then
    echo "[skip existing] $output"
    return
  fi
  if [[ "$QUIET_RESULT_JSON" == "1" ]]; then
    "${cmd[@]}" >/dev/null
  else
    "${cmd[@]}"
  fi
}

rq_aggregate() {
  local input_dir="$1"
  local pattern="$2"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] aggregate $input_dir pattern=$pattern"
    return
  fi
  "${PY[@]}" experiments/aggregate_results.py \
    --input_dir "$input_dir" \
    --pattern "$pattern" \
    --output_json "$input_dir/aggregate_summary.json" \
    --output_csv "$input_dir/aggregate_summary.csv" \
    --group_by method,dataset,unlearning_type,ratio,selection \
    --metrics "$RQ_METRICS"
}

rq_validate() {
  local rq="$1"
  local input_dir="$2"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] validate $rq in $input_dir"
    return
  fi
  "${PY[@]}" experiments/validate_pubmed_rq.py --rq "$rq" --input_dir "$input_dir"
}
