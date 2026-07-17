#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CONDA_ENV="${CONDA_ENV:-base}"
ROOT="results/mia_v2_primekg-full-nosource_eval"
FORGET_ROOT="experiments/forget_sets_eval/primekg-full-nosource"
HUB_PPR_BATCH_SIZE="${HUB_PPR_BATCH_SIZE:-16}"
PY=(conda run --no-capture-output -n "$CONDA_ENV" python)

"${PY[@]}" -c 'import sys, torch; ok=torch.cuda.is_available(); print(f"CUDA available: {ok}, torch={torch.__version__}, runtime={torch.version.cuda}"); sys.exit(0 if ok else "CUDA is unavailable")'

"${PY[@]}" experiments/prepare_base_models.py \
  --datasets primekg-full-nosource \
  --seeds 42,123,2024 \
  --output_root results/shared_base \
  --model_type GCN \
  --hidden_channels 64 \
  --num_layers 2 \
  --dropout 0.5 \
  --train_epochs 300 \
  --lr 0.01 \
  --weight_decay 5e-4 \
  --split stratified_random \
  --train_ratio 0.6 \
  --val_ratio 0.2 \
  --test_ratio 0.2 \
  --training_graph train_subgraph \
  --device cuda:0

for seed in 42 123 2024; do
  base="results/shared_base/primekg-full-nosource/seed${seed}"
  test -s "$base/model_state.pt"
  test -s "$base/logits.pt"
  test -s "$base/embeddings.pt"
  jq -e \
    --argjson seed "$seed" '
      .dataset == "primekg-full-nosource" and
      .seed == $seed and
      .num_nodes == 129312 and
      .num_features == 3 and
      .num_classes == 10 and
      .model.type == "GCN" and
      .model.in_channels == 3 and
      .model.hidden_channels == 64 and
      .model.num_layers == 2 and
      .training.epochs == 300 and
      .training.split == "stratified_random" and
      .training.train_ratio == 0.6 and
      .training.val_ratio == 0.2 and
      .training.test_ratio == 0.2 and
      .training.training_graph == "train_subgraph" and
      .mask_counts.train_mask == 77588 and
      .mask_counts.val_mask == 25862 and
      .mask_counts.test_mask == 25862
    ' "$base/metadata.json" >/dev/null
done

"${PY[@]}" experiments/generate_forget_protocols.py \
  --datasets primekg-full-nosource \
  --unlearning_types node,edge,feature \
  --ratios 0.05,0.1 \
  --shared_base_seeds 42,123,2024 \
  --forget_seeds 70042,70123,72024 \
  --seed_pairing zip \
  --node_selections random_train \
  --edge_feature_selection random_all \
  --split_source shared_base \
  --base_artifact_root results/shared_base \
  --edge_scope train_subgraph \
  --edge_sampling_unit unique_undirected \
  --output_dir experiments/forget_sets_eval \
  --layout flat \
  --flat_seed_label base_fseed \
  --manifest "$FORGET_ROOT/manifest.json" \
  --overwrite >/dev/null

for kind in node edge feature; do
  if [[ "$kind" == "node" ]]; then selection=random_train; else selection=random_all; fi
  for ratio in r0p05 r0p1; do
    if [[ "$ratio" == "r0p05" ]]; then ratio_value=0.05; else ratio_value=0.1; fi
    for spec in "42 70042" "123 70123" "2024 72024"; do
      read -r seed fseed <<< "$spec"
      file="$FORGET_ROOT/primekg-full-nosource_${kind}_${ratio}_${selection}_base${seed}_fseed${fseed}.json"
      base="results/shared_base/primekg-full-nosource/seed${seed}"
      jq -e \
        --arg kind "$kind" --arg selection "$selection" --arg base "$base" \
        --argjson ratio "$ratio_value" --argjson fseed "$fseed" '
          .dataset == "primekg-full-nosource" and
          .unlearning_type == $kind and
          .selection == $selection and
          .ratio == $ratio and
          .seed == $fseed and
          (.targets | length) > 0 and
          .protocol.split_source == "shared_base" and
          .protocol.base_artifact_dir == $base and
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
        ' "$file" >/dev/null
    done
  done
done

ARTIFACT_ROOT="$ROOT/hasi/artifacts/hub_scores" \
HUB_PPR_BATCH_SIZE="$HUB_PPR_BATCH_SIZE" \
CONDA_ENV="$CONDA_ENV" \
  bash commands/prepare_primekg_full_hub_score_artifacts.sh

mkdir -p "$ROOT/.matrix_state"
touch "$ROOT/PREPARE_COMPLETE"
echo "[complete] PrimeKG full no-source shared bases, forget sets, and HubScore artifacts are ready."
