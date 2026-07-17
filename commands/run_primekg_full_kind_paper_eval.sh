#!/usr/bin/env bash
set -euo pipefail

kind="${1:-}"
if [[ "$kind" != "node" && "$kind" != "edge" && "$kind" != "feature" ]]; then
  echo "Usage: $0 {node|edge|feature}" >&2
  exit 2
fi

ROOT="results/mia_v2_primekg-full-nosource_eval"
test -f "$ROOT/PREPARE_COMPLETE"

RUN_TYPES="$kind" SKIP_AGGREGATE=1 \
  bash commands/run_primekg_full_baselines_paper_eval.sh

RUN_TYPES="$kind" HUB_PPR_BATCH_SIZE="${HUB_PPR_BATCH_SIZE:-16}" \
  bash commands/run_primekg_full_hasi_paper_eval.sh

mkdir -p "$ROOT/.matrix_state"
touch "$ROOT/.matrix_state/${kind}.complete"
echo "[complete] PrimeKG full no-source ${kind} matrix"
