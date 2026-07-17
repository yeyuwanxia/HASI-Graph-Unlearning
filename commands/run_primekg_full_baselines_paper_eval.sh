#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
CONDA_ENV="${CONDA_ENV:-base}"
RESULT_ROOT="${RESULT_ROOT:-results/mia_v2_primekg-full-nosource_eval/baselines}"
EXACT_RETRAIN_REFERENCE_ROOT="${EXACT_RETRAIN_REFERENCE_ROOT:-$RESULT_ROOT/retrain/edge/artifacts/exact_retrain}"
RUN_TYPES="${RUN_TYPES:-node,edge,feature}"
BASELINES="${BASELINES:-retrain,gif,gnndelete,grapheraser-bekm,grapheraser-blpa}"
OVERWRITE="${OVERWRITE:-0}"
REBUILD_EXACT_RETRAIN_REFERENCES="${REBUILD_EXACT_RETRAIN_REFERENCES:-0}"
REBUILD_GRAPHERASER_ARTIFACTS="${REBUILD_GRAPHERASER_ARTIFACTS:-0}"
DRY_RUN="${DRY_RUN:-0}"
QUIET_RESULT_JSON="${QUIET_RESULT_JSON:-1}"
SKIP_AGGREGATE="${SKIP_AGGREGATE:-0}"
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
  elif [[ "$QUIET_RESULT_JSON" == "1" ]]; then
    "$@" >/dev/null
  else
    "$@"
  fi
}

selection_for_type() {
  local kind=$1
  if [[ "$kind" == "node" ]]; then
    printf 'random_train\n'
  else
    printf 'random_all\n'
  fi
}

validate_forget_set() {
  local path=$1 kind=$2 selection=$3 base_path=$4 forget_seed=$5 ratio=$6
  jq -e \
    --arg kind "$kind" \
    --arg selection "$selection" \
    --arg base_path "$base_path" \
    --argjson forget_seed "$forget_seed" \
    --argjson ratio "$ratio" '
      .dataset == "primekg-full-nosource" and
      .unlearning_type == $kind and
      .selection == $selection and
      .seed == $forget_seed and
      .ratio == $ratio and
      .protocol.split_source == "shared_base" and
      .protocol.base_artifact_dir == $base_path and
      .protocol.base_training_graph == "train_subgraph" and
      (if $kind == "node" then
         .protocol.selection_scope == "train_mask_nodes"
       elif $kind == "feature" then
         .protocol.selection_scope == "feature_dimensions"
       else
         .protocol.selection_scope == "train_subgraph_unique_undirected_edges" and
         .protocol.sampling_unit == "unique_undirected_edge" and
         .protocol.ratio_denominator == "unique_candidate_undirected_edges" and
         .protocol.deletion_operator == "undirected_closure"
       end)
    ' "$path" >/dev/null
}

validate_result() {
  local path=$1 kind=$2 base_path=$3 forget_path=$4 expected_method=$5 require_offline=$6 reference_path=$7
  jq -e \
    --arg kind "$kind" \
    --arg base_path "$base_path" \
    --arg forget_path "$forget_path" \
    --arg reference_path "$reference_path" \
    --arg expected_method "$expected_method" \
    --argjson require_offline "$require_offline" '
      .dataset == "primekg-full-nosource" and
      (.baseline == $expected_method or .method == $expected_method) and
      .base_artifact.loaded == true and
      .base_artifact.path == $base_path and
      .forget_set.unlearning_type == $kind and
      .forget_set.path == $forget_path and
      .forget_set.protocol.split_source == "shared_base" and
      .forget_set.protocol.base_artifact_dir == $base_path and
      .forget_set.protocol.base_training_graph == "train_subgraph" and
      .metrics.unlearning_type == $kind and
      .metrics.evaluation_protocol.version == "paper_eval_20260715_v1" and
      .metrics.efficiency.online_wall_clock_seconds != null and
      (.metrics.efficiency.time_breakdown | type) == "object" and
      (if $require_offline then
         .metrics.efficiency.offline_preprocessing_seconds != null and
         .opengu_artifact.loaded == true and
         .opengu_artifact.artifact.offline_preprocessing_seconds != null
       else true end) and
      (if $kind == "feature" then
         .metrics.privacy.applicable == false and
         .metrics.privacy.status == "not_applicable_global_feature_dimension_request" and
         .metrics.privacy.strong_auc == null and
         .metrics.privacy.feature_proxy == false and
         .metrics.feature_compliance.status == "ok" and
         .metrics.feature_compliance.request_applied == true
       else
         .metrics.privacy.status == "ok" and
         .metrics.privacy.strong_auc != null and
         .metrics.privacy.strong_auc_null_mean != null and
         .metrics.privacy.strong_auc_null_std != null and
         .metrics.privacy.strong_auc_pvalue != null and
         .metrics.privacy.medium_evaluation == "held_out_target_split" and
         .metrics.privacy.medium_train_size > 0 and
         .metrics.privacy.medium_eval_size > 0
       end) and
      (if $expected_method == "opengu-gif" then
         .result.training.iteration == 100 and
         .result.training.scale == 1000000000 and
         .result.training.damp == 0 and .result.training.hops == 2
       else true end) and
      (if $kind == "node" then
         .metrics.structure.evaluation_scope == "retained_nodes"
       elif $kind == "edge" then
         .forget_set.protocol.sampling_unit == "unique_undirected_edge" and
         .forget_set.protocol.ratio_denominator == "unique_candidate_undirected_edges" and
         .forget_set.protocol.deletion_operator == "undirected_closure" and
         .metrics.edge_forgetting.status == "ok" and
         .metrics.edge_forgetting.request_applied == true and
         .metrics.edge_forgetting.exact_retrain_embedding_status == "ok" and
         .metrics.exact_retrain_alignment.status == "ok" and
         .exact_retrain_reference.loaded == true and
         .exact_retrain_reference.path == $reference_path and
         .metrics.exact_retrain_alignment.reference_path == $reference_path and
         .metrics.exact_retrain_alignment.reference_forget_set_sha256 != null
       else true end)
    ' "$path" >/dev/null
}

artifact_complete() {
  local artifact_dir=$1 metadata=$2
  [[ -s "$metadata" && -s "$artifact_dir/partition.json" && -d "$artifact_dir/shards" ]] || return 1
  jq -e '.offline_preprocessing_seconds != null and .num_shards > 0' "$metadata" >/dev/null || return 1

  local expected_shards actual_shards
  expected_shards=$(jq -r '.num_shards' "$metadata")
  actual_shards=$(find "$artifact_dir/shards" -maxdepth 1 -type f -name 'shard_*_model_state.pt' 2>/dev/null | wc -l)
  [[ "$actual_shards" -eq "$expected_shards" ]]
}

prepare_grapheraser_artifacts() {
  local kind=$1 baseline method_key base_seed artifact_dir metadata base_path
  for baseline in "${baselines[@]}"; do
    [[ "$baseline" == grapheraser-* ]] || continue
    method_key="opengu_${baseline//-/_}"
    for base_seed in 42 123 2024; do
      artifact_dir="$RESULT_ROOT/$method_key/$kind/artifacts/seed${base_seed}"
      metadata="$artifact_dir/metadata.json"
      base_path="results/shared_base/primekg-full-nosource/seed${base_seed}"

      if [[ "$DRY_RUN" == "1" || "$REBUILD_GRAPHERASER_ARTIFACTS" == "1" ]] || \
         ! artifact_complete "$artifact_dir" "$metadata"; then
        run_command "${PY[@]}" opengu_adapted_baselines/scripts/run_opengu_adapted_baseline.py \
          --baseline "$baseline" \
          --dataset_name primekg-full-nosource \
          --unlearning_type "$kind" \
          --seed "$base_seed" \
          --base_artifact_root results/shared_base \
          --device cuda:0 \
          --train_epochs 300 \
          --lr 0.01 \
          --artifact_dir "$artifact_dir" \
          --rebuild_artifact \
          --prepare_artifact_only
      fi

      if [[ "$DRY_RUN" != "1" ]]; then
        artifact_complete "$artifact_dir" "$metadata"
        jq -e \
          --arg kind "$kind" \
          --arg base_path "$base_path" \
          --argjson expected_seed "$base_seed" '
            .dataset == "primekg-full-nosource" and
            .seed == $expected_seed and
            .unlearning_type == $kind and
            .base_artifact.path == $base_path and
            .offline_preprocessing_seconds != null
          ' "$metadata" >/dev/null
        echo "[artifact validated] $artifact_dir"
      fi
    done
  done
}

run_retrain() {
  local kind=$1 forget_path=$2 base_seed=$3 output=$4 reference_path=${5:-}
  local -a args=(
    "${PY[@]}" experiments/run_baselines.py
    --baseline retrain
    --dataset_name primekg-full-nosource
    --unlearning_type "$kind"
    --seed "$base_seed"
    --base_artifact_root results/shared_base
    --forget_set_file "$forget_path"
    --device cuda:0
    --train_epochs 300
    --lr 0.01
    --output "$output"
  )
  if [[ "$kind" == "edge" ]]; then
    args+=(--save_exact_retrain_reference "$reference_path")
  fi
  run_command "${args[@]}"
}

run_opengu() {
  local baseline=$1 kind=$2 forget_path=$3 base_seed=$4 output=$5 reference_path=${6:-}
  local -a args=(
    "${PY[@]}" opengu_adapted_baselines/scripts/run_opengu_adapted_baseline.py
    --baseline "$baseline"
    --dataset_name primekg-full-nosource
    --unlearning_type "$kind"
    --seed "$base_seed"
    --base_artifact_root results/shared_base
    --forget_set_file "$forget_path"
    --device cuda:0
    --train_epochs 300
    --lr 0.01
    --experiment_name __root__
    --output "$output"
  )
  if [[ "$kind" == "edge" ]]; then
    args+=(--exact_retrain_reference "$reference_path")
  fi
  if [[ "$baseline" == grapheraser-* ]]; then
    args+=(--artifact_dir "$RESULT_ROOT/opengu_${baseline//-/_}/$kind/artifacts/seed${base_seed}")
  fi
  run_command "${args[@]}"
}

exact_retrain_reference_path() {
  local ratio=$1 base_seed=$2 forget_seed=$3
  printf '%s/%s/base%s_fseed%s.pt\n' \
    "$EXACT_RETRAIN_REFERENCE_ROOT" "$ratio" "$base_seed" "$forget_seed"
}

validate_exact_retrain_reference() {
  local path=$1 forget_path=$2 base_path=$3 base_seed=$4
  [[ -s "$path" && -s "$path.json" ]]
  jq -e \
    --arg forget_path "$forget_path" \
    --arg base_path "$base_path" \
    --argjson base_seed "$base_seed" '
      .schema_version == "exact_retrain_reference_v1" and
      .producer == "experiments/run_baselines.py" and
      .method == "retrain" and
      .dataset == "primekg-full-nosource" and
      .unlearning_type == "edge" and
      .seed == $base_seed and
      .forget_set_path == $forget_path and
      .base_artifact_path == $base_path and
      .forget_set_sha256 != null
    ' "$path.json" >/dev/null
}

aggregate_results() {
  local output_root
  output_root=$(dirname "$RESULT_ROOT")
  run_command "${PY[@]}" experiments/aggregate_results.py \
    --input_dir "$output_root" \
    --pattern '*primekg-full-nosource_*.json' \
    --output_json "$output_root/aggregate_summary.json" \
    --output_csv "$output_root/aggregate_summary.csv" \
    --group_by method,dataset,unlearning_type,ratio,selection \
    --metrics metrics.utility.accuracy_after,metrics.utility.accuracy_drop,metrics.utility.f1_macro_after,metrics.structure.degree_kl_divergence,metrics.structure.clustering_coefficient_change,metrics.structure.component_count_change,metrics.privacy.weak_auc,metrics.privacy.medium_auc,metrics.privacy.medium_train_size,metrics.privacy.medium_eval_size,metrics.privacy.strong_auc,metrics.privacy.strong_auc_null_mean,metrics.privacy.strong_auc_null_std,metrics.privacy.strong_auc_pvalue,metrics.privacy.privacy_score,metrics.representation.embedding_l2_mean,metrics.representation.member_embedding_l2_mean,metrics.representation.neighbor_drift_mean,metrics.edge_forgetting.forgotten_score_drop_mean,metrics.edge_forgetting.retained_control_score_drop_mean,metrics.edge_forgetting.targeted_drop_vs_control,metrics.edge_forgetting.forgotten_unlearned_to_retrain_abs_gap_mean,metrics.exact_retrain_alignment.unlearned_to_retrain_js_mean,metrics.exact_retrain_alignment.improvement_over_original_js,metrics.exact_retrain_alignment.unlearned_to_retrain_tv_mean,metrics.exact_retrain_alignment.prediction_disagreement_rate,metrics.efficiency.unlearn_time_seconds,metrics.efficiency.online_wall_clock_seconds,metrics.efficiency.offline_preprocessing_seconds
}

IFS=',' read -r -a kinds <<< "$RUN_TYPES"
IFS=',' read -r -a baselines <<< "$BASELINES"
validated_count=0

for kind in "${kinds[@]}"; do
  if [[ "$kind" != "node" && "$kind" != "edge" && "$kind" != "feature" ]]; then
    echo "Unsupported RUN_TYPES entry: $kind" >&2
    exit 2
  fi

  selection=$(selection_for_type "$kind")
  prepare_grapheraser_artifacts "$kind"

  for baseline in "${baselines[@]}"; do
    if [[ "$baseline" == "gif" && "$kind" != "edge" ]]; then
      echo "[unsupported by adapter, skipped] baseline=gif type=$kind"
      continue
    fi
    case "$baseline" in
      retrain)
        method_key=retrain
        expected_method=retrain
        require_offline=false
        ;;
      gif|gnndelete)
        method_key="opengu_${baseline}"
        expected_method="opengu-${baseline}"
        require_offline=false
        ;;
      grapheraser-bekm|grapheraser-blpa)
        method_key="opengu_${baseline//-/_}"
        expected_method="opengu-${baseline}"
        require_offline=true
        ;;
      *)
        echo "Unsupported BASELINES entry: $baseline" >&2
        exit 2
        ;;
    esac

    for ratio in r0p05 r0p1; do
      if [[ "$ratio" == "r0p05" ]]; then
        ratio_value=0.05
      else
        ratio_value=0.1
      fi

      for spec in "42 70042" "123 70123" "2024 72024"; do
        read -r base_seed forget_seed <<< "$spec"
        base_path="results/shared_base/primekg-full-nosource/seed${base_seed}"
        forget_path="experiments/forget_sets_eval/primekg-full-nosource/primekg-full-nosource_${kind}_${ratio}_${selection}_base${base_seed}_fseed${forget_seed}.json"
        output_dir="$RESULT_ROOT/$method_key/$kind"
        output="$output_dir/${method_key}_primekg-full-nosource_${kind}_${ratio}_${selection}_base${base_seed}_fseed${forget_seed}.json"
        reference_path=""
        if [[ "$kind" == "edge" ]]; then
          reference_path=$(exact_retrain_reference_path "$ratio" "$base_seed" "$forget_seed")
        fi

        validate_forget_set "$forget_path" "$kind" "$selection" "$base_path" "$forget_seed" "$ratio_value"
        mkdir -p "$output_dir"
        if [[ "$kind" == "edge" ]]; then
          mkdir -p "$(dirname "$reference_path")"
        fi

        needs_run=0
        if [[ ! -s "$output" || "$OVERWRITE" == "1" ]]; then
          needs_run=1
        fi
        if [[ "$baseline" == "retrain" && "$kind" == "edge" ]] && \
           { [[ "$REBUILD_EXACT_RETRAIN_REFERENCES" == "1" ]] || \
             [[ ! -s "$reference_path" ]] || [[ ! -s "$reference_path.json" ]]; }; then
          needs_run=1
        fi

        if [[ "$DRY_RUN" != "1" && "$kind" == "edge" && "$baseline" != "retrain" ]]; then
          if ! validate_exact_retrain_reference "$reference_path" "$forget_path" "$base_path" "$base_seed"; then
            echo "Missing or invalid exact-retrain reference: $reference_path" >&2
            echo "Run the matching retrain Edge job first (BASELINES=retrain RUN_TYPES=edge)." >&2
            exit 1
          fi
        fi

        if [[ "$needs_run" == "1" ]]; then
          if [[ "$baseline" == "retrain" ]]; then
            run_retrain "$kind" "$forget_path" "$base_seed" "$output" "$reference_path"
          else
            run_opengu "$baseline" "$kind" "$forget_path" "$base_seed" "$output" "$reference_path"
          fi
        else
          echo "[skip existing] $output"
        fi

        if [[ "$DRY_RUN" != "1" ]]; then
          if [[ "$kind" == "edge" ]]; then
            validate_exact_retrain_reference "$reference_path" "$forget_path" "$base_path" "$base_seed"
          fi
          validate_result "$output" "$kind" "$base_path" "$forget_path" "$expected_method" "$require_offline" "$reference_path"
          echo "[validated] $output"
          validated_count=$((validated_count + 1))
        fi
      done
    done
  done
done

if [[ "$SKIP_AGGREGATE" != "1" ]]; then
  aggregate_results
fi
if [[ "$DRY_RUN" != "1" ]]; then
  echo "[complete] validated baseline results: $validated_count"
fi
