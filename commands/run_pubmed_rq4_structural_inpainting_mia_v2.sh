#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/rq_runner_common.sh"

OUT="${OUT:-$RQ_ROOT/rq4_structural_inpainting/hasi}"
for seed in 42 123 2024; do
  fs="experiments/rq_forget_sets/pubmed/rq3_anchor_ablation/pubmed_node_r0p05_hub_neighbor_train_seed${seed}.json"
  for mode in none full; do
    if [[ "$mode" == "none" ]]; then
      method="hasi_no_inpaint_rq4_mia_v2"
    else
      method="hasi_full_inpaint_rq4_mia_v2"
    fi
    output="$OUT/${method}_pubmed_node_r0p05_hub_neighbor_train_seed${seed}.json"
    rq_run "$output" \
      --seed "$seed" \
      --forget_ratio 0.05 \
      --forget_set_file "$fs" \
      --method_name "$method" \
      --anchor_mode hierarchical \
      --anchor_lambda1 2.0 \
      --anchor_lambda2 0.5 \
      --dar-enabled \
      --dar_lambda2 0.5 \
      --inpainting_mode "$mode"
  done
done

rq_aggregate "$OUT" '*rq4_mia_v2*.json'
rq_validate rq4 "$OUT"
