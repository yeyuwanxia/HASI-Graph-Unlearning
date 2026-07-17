#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/rq_runner_common.sh"
OUT="${OUT:-$RQ_ROOT/rq3_anchor_ablation/hasi}"

for seed in 42 123 2024; do
  fs="experiments/rq_forget_sets/pubmed/rq3_anchor_ablation/pubmed_node_r0p05_hub_neighbor_train_seed${seed}.json"

  for variant in no hier strong; do
    case "$variant" in
      no)
        method=hasi_no_anchor_rq3_mia_v2_daroff
        extra=(--anchor_mode none --anchor_lambda1 0 --anchor_lambda2 0)
        ;;
      hier)
        method=hasi_hier_anchor_rq3_mia_v2_daroff
        extra=(--anchor_mode hierarchical --anchor_lambda1 2.0 --anchor_lambda2 0.5)
        ;;
      strong)
        method=hasi_strong_anchor_rq3_mia_v2_daroff
        extra=(--anchor_mode hierarchical --anchor_lambda1 5.0 --anchor_lambda2 1.0)
        ;;
    esac

    output="$OUT/${method}_pubmed_node_r0p05_hub_neighbor_train_seed${seed}.json"
    rq_run "$output" \
      --seed "$seed" \
      --forget_ratio 0.05 \
      --forget_set_file "$fs" \
      --method_name "$method" \
      --no-dar-enabled \
      --dar_lambda2 0.5 \
      --inpainting_mode full \
      "${extra[@]}"
  done
done

rq_aggregate "$OUT" '*_daroff_*.json'
rq_validate rq3 "$OUT"
