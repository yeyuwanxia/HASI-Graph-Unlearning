#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONDA_ENV="${CONDA_ENV:-base}"
RUN_TYPES="${RUN_TYPES:-node,feature,edge}"
RESULT_ROOT="${RESULT_ROOT:-results/mia_v2_primekg-full-nosource_eval/hasi}"
HUB_SCORE_CACHE_ROOT="${HUB_SCORE_CACHE_ROOT:-results/mia_v2_primekg-full-nosource_eval/hasi/artifacts/hub_scores}"
EXACT_RETRAIN_REFERENCE_ROOT="${EXACT_RETRAIN_REFERENCE_ROOT:-results/mia_v2_primekg-full-nosource_eval/baselines/retrain/edge/artifacts/exact_retrain}"
HUB_PPR_BATCH_SIZE="${HUB_PPR_BATCH_SIZE:-16}"
METHOD="${METHOD:-hasi_transfer_primekg_dg_small_tuned}"
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
      .dataset == "primekg-full-nosource" and
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
  local path=$1 kind=$2 base_path=$3 forget_path=$4 reference_path=$5 expected_config=$6
  jq -e \
    --arg kind "$kind" \
    --arg base_path "$base_path" \
    --arg forget_path "$forget_path" \
    --arg reference_path "$reference_path" \
    --arg method "$METHOD" \
    --arg expected_config "$expected_config" '
      .dataset == "primekg-full-nosource" and
      .method == $method and
      (.config.path | endswith($expected_config)) and
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

  config="configs/tuned/by_dataset/primekg-disease-gene-small-nosource/${kind}.yaml"
  method="$METHOD"

  for ratio in r0p05 r0p1; do
      for spec in "42 70042" "123 70123" "2024 72024"; do
        read -r base_seed forget_seed <<< "$spec"
        base_path="results/shared_base/primekg-full-nosource/seed${base_seed}"
        forget_path="experiments/forget_sets_eval/primekg-full-nosource/primekg-full-nosource_${kind}_${ratio}_${selection}_base${base_seed}_fseed${forget_seed}.json"
        output_dir="$RESULT_ROOT/$method/$kind"
        output="$output_dir/${method}_primekg-full-nosource_${kind}_${ratio}_${selection}_base${base_seed}_fseed${forget_seed}.json"
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
            --dataset_name primekg-full-nosource \
            --config "$config" \
            --seed "$base_seed" \
            --unlearning_type "$kind" \
            --forget_ratio "$ratio_value" \
            --forget_set_file "$forget_path" \
            --base_artifact_root results/shared_base \
            --device cuda:0 \
            --graph_compute_backend torch \
            --hub_ppr_batch_size "$HUB_PPR_BATCH_SIZE" \
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
        validate_result "$output" "$kind" "$base_path" "$forget_path" "$reference_path" "$config"
        echo "[validated] $output"
        fi
      done
  done
done

if [[ "$DRY_RUN" != "1" ]]; then
  output_root=$(dirname "$RESULT_ROOT")
  touch "$output_root/TRANSFER_SMALL_TUNED_COMPLETE"
  echo "[complete] transferred small no-source config results: 18"

  while [[ ! -f "$output_root/FULL_MATRIX_COMPLETE" ]]; do
    sleep 60
  done

  "${PY[@]}" experiments/aggregate_results.py \
    --input_dir "$output_root" \
    --pattern '*primekg-full-nosource_*.json' \
    --output_json "$output_root/aggregate_summary.json" \
    --output_csv "$output_root/aggregate_summary.csv" \
    --group_by method,dataset,unlearning_type,ratio,selection \
    --metrics metrics.utility.accuracy_after,metrics.utility.accuracy_drop,metrics.utility.f1_macro_after,metrics.structure.degree_kl_divergence,metrics.structure.clustering_coefficient_change,metrics.structure.component_count_change,metrics.privacy.weak_auc,metrics.privacy.medium_auc,metrics.privacy.medium_train_size,metrics.privacy.medium_eval_size,metrics.privacy.strong_auc,metrics.privacy.strong_auc_null_mean,metrics.privacy.strong_auc_null_std,metrics.privacy.strong_auc_pvalue,metrics.privacy.privacy_score,metrics.representation.embedding_l2_mean,metrics.representation.member_embedding_l2_mean,metrics.representation.neighbor_drift_mean,metrics.edge_forgetting.forgotten_score_drop_mean,metrics.edge_forgetting.retained_control_score_drop_mean,metrics.edge_forgetting.targeted_drop_vs_control,metrics.edge_forgetting.forgotten_unlearned_to_retrain_abs_gap_mean,metrics.exact_retrain_alignment.unlearned_to_retrain_js_mean,metrics.exact_retrain_alignment.improvement_over_original_js,metrics.exact_retrain_alignment.unlearned_to_retrain_tv_mean,metrics.exact_retrain_alignment.prediction_disagreement_rate,metrics.efficiency.unlearn_time_seconds,metrics.efficiency.online_wall_clock_seconds,metrics.efficiency.offline_preprocessing_seconds

  "${PY[@]}" -c '
import json
from pathlib import Path

root = Path("results/mia_v2_primekg-full-nosource_eval")
files = [
    path for path in root.rglob("*.json")
    if "artifacts" not in path.parts
    and path.name != "aggregate_summary.json"
    and "primekg-full-nosource_" in path.name
]
payloads = [json.loads(path.read_text(encoding="utf-8")) for path in files]
assert len(payloads) == 114, len(payloads)
transfer = [item for item in payloads if item.get("method") == "hasi_transfer_primekg_dg_small_tuned"]
assert len(transfer) == 18, len(transfer)
assert all(item["metrics"]["evaluation_protocol"]["version"] == "paper_eval_20260715_v1" for item in transfer)
by_type = {kind: sum(item["metrics"]["unlearning_type"] == kind for item in transfer) for kind in ("node", "edge", "feature")}
assert by_type == {"node": 6, "edge": 6, "feature": 6}, by_type
print({"formal_results": len(payloads), "transfer_results": len(transfer), "transfer_by_type": by_type})
'

  touch "$output_root/FULL_MATRIX_WITH_TRANSFER_COMPLETE"
  echo "[complete] PrimeKG full no-source matrix with transferred small no-source config: 114 results"
fi
