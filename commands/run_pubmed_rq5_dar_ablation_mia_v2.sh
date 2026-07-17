#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/rq_runner_common.sh"

OUT="${OUT:-$RQ_ROOT/rq5_dar_ablation/hasi}"
for seed in 42 123 2024; do
  fs="experiments/rq_forget_sets/pubmed/rq1_node_selection/pubmed_node_r0p05_hub_train_seed${seed}.json"
  for variant in off on; do
    if [[ "$variant" == "off" ]]; then
      method="hasi_dar_off_rq5_mia_v2"
      dar_flag="--no-dar-enabled"
    else
      method="hasi_dar_on_rq5_mia_v2"
      dar_flag="--dar-enabled"
    fi
    output="$OUT/${method}_pubmed_node_r0p05_hub_train_seed${seed}.json"
    rq_run "$output" \
      --seed "$seed" \
      --forget_ratio 0.05 \
      --forget_set_file "$fs" \
      --method_name "$method" \
      --anchor_mode hierarchical \
      --anchor_lambda1 2.0 \
      --anchor_lambda2 0.5 \
      "$dar_flag" \
      --dar_lambda2 0.5 \
      --inpainting_mode full
  done
done

rq_aggregate "$OUT" '*rq5_mia_v2*.json'
rq_validate rq5 "$OUT"
