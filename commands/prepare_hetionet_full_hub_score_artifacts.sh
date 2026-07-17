#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
CONDA_ENV="${CONDA_ENV:-base}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-results/mia_v2_hetionet-full-nosource_eval/hasi/artifacts/hub_scores}"
DRY_RUN="${DRY_RUN:-0}"
PY=(conda run --no-capture-output -n "$CONDA_ENV" python)

if [[ "$DRY_RUN" != "1" ]]; then
  "${PY[@]}" -c 'import sys, torch; ok=torch.cuda.is_available(); print(f"CUDA available: {ok}, torch={torch.__version__}, runtime={torch.version.cuda}"); sys.exit(0 if ok else "CUDA is unavailable")'
fi

for seed in 42 123 2024; do
  manifest_dir="$ARTIFACT_ROOT/hetionet-full-nosource/seed${seed}"
  manifest="$manifest_dir/prepare_manifest.json"

  cmd=(
    "${PY[@]}" experiments/run_hasi.py
    --mode prepare_hub_scores
    --dataset_name hetionet-full-nosource
    --config configs/hasi_default.yaml
    --seed "$seed"
    --unlearning_type node
    --base_artifact_root results/shared_base
    --device cuda:0
    --graph_compute_backend torch
    --hub_ppr_batch_size 64
    --gradient-hub-score
    --hub-score-cache
    --hub_score_cache_root "$ARTIFACT_ROOT"
    --method_name hasi_hub_score_prepare
    --output "$manifest"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] '
    printf '%q ' "${cmd[@]}"
    printf '\n'
    continue
  fi
  mkdir -p "$manifest_dir"
  "${cmd[@]}"

  "${PY[@]}" -c '
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
cache = payload.get("hub_score_cache", {})
assert cache.get("enabled") is True and cache.get("path"), cache
assert cache.get("offline_preprocessing_seconds") is not None, cache
print("HubScore artifact ready: hit={} path={}".format(cache.get("hit"), cache.get("path")))
' "$manifest"
done
