#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RESULT_ROOT="$REPO_ROOT/results/mia_v2_primekg-full-nosource_eval"

cd "$REPO_ROOT"
test -f "$RESULT_ROOT/PREPARE_COMPLETE"

export CUDA_VISIBLE_DEVICES=0
export CONDA_ENV="${CONDA_ENV:-base}"
export HUB_PPR_BATCH_SIZE="${HUB_PPR_BATCH_SIZE:-16}"
export STRUCTURAL_METRICS_WORKERS="${STRUCTURAL_METRICS_WORKERS:-8}"
export STRUCTURAL_METRICS_PARALLEL_MIN_EDGES="${STRUCTURAL_METRICS_PARALLEL_MIN_EDGES:-100000}"

RUN_TYPES=node,feature,edge OVERWRITE=0 \
  bash commands/run_primekg_full_transfer_small_tuned_paper_eval.sh
