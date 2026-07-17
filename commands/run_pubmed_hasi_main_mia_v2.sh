#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONDA_ENV="${CONDA_ENV:-base}"
RUN_TYPES="${RUN_TYPES:-node,edge,feature}"
RESULT_ROOT="${RESULT_ROOT:-results/mia_v2_pubmed_eval/hasi}"
HUB_SCORE_CACHE_ROOT="${HUB_SCORE_CACHE_ROOT:-results/mia_v2_pubmed_eval/hasi/artifacts/hub_scores}"
EXACT_RETRAIN_REFERENCE_ROOT="${EXACT_RETRAIN_REFERENCE_ROOT:-results/mia_v2_pubmed_eval/baselines/retrain/edge/artifacts/exact_retrain}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
export STRUCTURAL_METRICS_WORKERS="${STRUCTURAL_METRICS_WORKERS:-8}"
export STRUCTURAL_METRICS_PARALLEL_MIN_EDGES="${STRUCTURAL_METRICS_PARALLEL_MIN_EDGES:-100000}"
PY=(conda run --no-capture-output -n "$CONDA_ENV" python)

if [[ "$DRY_RUN" != "1" ]]; then
  "${PY[@]}" -c 'import sys, torch; ok=torch.cuda.is_available(); print(f"CUDA available: {ok}, torch={torch.__version__}, runtime={torch.version.cuda}"); sys.exit(0 if ok else "CUDA is unavailable")'
fi

run_command() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] '
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

validate_forget_set() {
  local path=$1 kind=$2 base_path=$3 forget_seed=$4
  jq -e \
    --arg kind "$kind" \
    --arg base_path "$base_path" \
    --argjson forget_seed "$forget_seed" '
      .dataset == "pubmed" and
      .unlearning_type == $kind and
      .seed == $forget_seed and
      .protocol.split_source == "shared_base" and
      .protocol.base_artifact_dir == $base_path and
      .protocol.base_training_graph == "train_subgraph" and
      (if $kind == "edge" then
         .protocol.selection_scope == "train_subgraph_unique_undirected_edges" and
         .protocol.sampling_unit == "unique_undirected_edge" and
         .protocol.ratio_denominator == "unique_candidate_undirected_edges" and
         .protocol.deletion_operator == "undirected_closure"
       else true end)
    ' "$path" >/dev/null
}

validate_result() {
  local path=$1 kind=$2 base_path=$3 forget_path=$4 reference_path=$5
  jq -e \
    --arg kind "$kind" \
    --arg base_path "$base_path" \
    --arg forget_path "$forget_path" \
    --arg reference_path "$reference_path" '
      .dataset == "pubmed" and
      .base_artifact.loaded == true and
      .base_artifact.path == $base_path and
      .forget_set.path == $forget_path and
      .config.resolved.unlearning.ratio == .forget_set.ratio and
      .metrics.evaluation_protocol.version == "paper_eval_20260715_v1" and
      .hub_score_cache.hit == true and
      .metrics.efficiency.offline_preprocessing_seconds != null and
      .unlearning.affected_region_size > 0 and
      (if $kind == "feature" then
         .metrics.privacy.applicable == false and
         .metrics.privacy.status == "not_applicable_global_feature_dimension_request" and
         .metrics.feature_compliance.request_applied == true
       else
         .unlearning.affected_region_diagnostics.compute_backend.used_backend == "torch" and
         .metrics.privacy.status == "ok" and
         .metrics.privacy.medium_evaluation == "held_out_target_split" and
         .metrics.privacy.medium_train_size > 0 and
         .metrics.privacy.medium_eval_size > 0
       end) and
      (if $kind == "edge" then
         .forget_set.protocol.sampling_unit == "unique_undirected_edge" and
         .forget_set.protocol.ratio_denominator == "unique_candidate_undirected_edges" and
         .forget_set.protocol.deletion_operator == "undirected_closure" and
         .metrics.edge_forgetting.status == "ok" and
         .metrics.edge_forgetting.request_applied == true and
         .metrics.edge_forgetting.exact_retrain_embedding_status == "ok" and
         .metrics.exact_retrain_alignment.status == "ok" and
         .exact_retrain_reference.loaded == true and
         .exact_retrain_reference.path == $reference_path and
         .metrics.exact_retrain_alignment.reference_path == $reference_path
       else true end)
    ' "$path" >/dev/null
}

exact_retrain_reference_path() {
  local ratio=$1 base_seed=$2 forget_seed=$3
  printf '%s/%s/base%s_fseed%s.pt\n' \
    "$EXACT_RETRAIN_REFERENCE_ROOT" "$ratio" "$base_seed" "$forget_seed"
}

IFS=',' read -r -a kinds <<< "$RUN_TYPES"
for kind in "${kinds[@]}"; do
  if [[ "$kind" != "node" && "$kind" != "edge" && "$kind" != "feature" ]]; then
    echo "Unsupported RUN_TYPES entry: $kind" >&2
    exit 2
  fi
  if [[ "$kind" == "node" ]]; then
    selection=random_train
  else
    selection=random_all
  fi

  for variant in default tuned; do
    if [[ "$variant" == "default" ]]; then
      config=configs/hasi_default.yaml
      method=hasi_default
    else
      config="configs/tuned/by_dataset/pubmed/${kind}.yaml"
      method=hasi_tuned
    fi

    for ratio in r0p05 r0p1; do
      for spec in "42 70042" "123 70123" "2024 72024"; do
        read -r base_seed forget_seed <<< "$spec"
        base_path="results/shared_base/pubmed/seed${base_seed}"
        forget_path="experiments/forget_sets_eval/pubmed/pubmed_${kind}_${ratio}_${selection}_base${base_seed}_fseed${forget_seed}.json"
        output_dir="$RESULT_ROOT/$method/$kind"
        output="$output_dir/${method}_pubmed_${kind}_${ratio}_${selection}_base${base_seed}_fseed${forget_seed}.json"
        reference_path=""
        extra_args=()
        if [[ "$kind" == "edge" ]]; then
          reference_path=$(exact_retrain_reference_path "$ratio" "$base_seed" "$forget_seed")
          if [[ "$DRY_RUN" != "1" ]] && [[ ! -s "$reference_path" || ! -s "$reference_path.json" ]]; then
            echo "Missing exact-retrain reference: $reference_path" >&2
            echo "Run the matching retrain Edge job before HASI Edge." >&2
            exit 1
          fi
          extra_args+=(--exact_retrain_reference "$reference_path")
        fi

        if [[ "$ratio" == "r0p05" ]]; then
          ratio_value=0.05
        else
          ratio_value=0.1
        fi

        validate_forget_set "$forget_path" "$kind" "$base_path" "$forget_seed"
        mkdir -p "$output_dir"
        if [[ "$DRY_RUN" == "1" || ! -s "$output" || "$OVERWRITE" == "1" ]]; then
          run_command "${PY[@]}" experiments/run_hasi.py \
            --mode unlearn \
            --dataset_name pubmed \
            --config "$config" \
            --seed "$base_seed" \
            --unlearning_type "$kind" \
            --forget_ratio "$ratio_value" \
            --forget_set_file "$forget_path" \
            --base_artifact_root results/shared_base \
            --device cuda:0 \
            --graph_compute_backend torch \
            --hub_ppr_batch_size 64 \
            --hub-score-cache \
            --hub_score_cache_root "$HUB_SCORE_CACHE_ROOT" \
            --require-hub-score-cache-hit \
            --method_name "$method" \
            "${extra_args[@]}" \
            --output "$output"
        else
          echo "[skip existing] $output"
        fi
        if [[ "$DRY_RUN" != "1" ]]; then
        validate_result "$output" "$kind" "$base_path" "$forget_path" "$reference_path"
        echo "[validated] $output"
        fi
      done
    done
  done
done
