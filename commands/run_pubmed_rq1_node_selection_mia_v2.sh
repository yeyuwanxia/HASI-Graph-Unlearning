#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/rq_runner_common.sh"

OUT="${OUT:-$RQ_ROOT/rq1_node_selection/hasi_default}"
for seed in 42 123 2024; do
  for ratio in r0p05 r0p1; do
    [[ "$ratio" == "r0p05" ]] && ratio_value=0.05 || ratio_value=0.1
    for selection in random_train hub_train low_degree_train; do
      fs="experiments/rq_forget_sets/pubmed/rq1_node_selection/pubmed_node_${ratio}_${selection}_seed${seed}.json"
      method="hasi_default_rq1_mia_v2"
      output="$OUT/${method}_pubmed_node_${ratio}_${selection}_seed${seed}.json"
      rq_run "$output" \
        --seed "$seed" \
        --forget_ratio "$ratio_value" \
        --forget_set_file "$fs" \
        --method_name "$method" \
        --anchor_mode hierarchical \
        --anchor_lambda1 2.0 \
        --anchor_lambda2 0.5 \
        --dar-enabled \
        --dar_lambda2 0.5 \
        --inpainting_mode full
    done
  done
done

rq_aggregate "$OUT" '*rq1_mia_v2*.json'
rq_validate rq1 "$OUT"
