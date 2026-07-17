#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RESULT_ROOT="$REPO_ROOT/results/mia_v2_primekg-full-nosource_eval"
LOG_ROOT="$RESULT_ROOT/logs"
mkdir -p "$LOG_ROOT"

nohup taskset -c 0-15 bash -lc "
  set -euo pipefail
  cd '$REPO_ROOT'
  export CUDA_VISIBLE_DEVICES=0 CONDA_ENV=base HUB_PPR_BATCH_SIZE=16
  bash commands/prepare_primekg_full_paper_eval.sh
" > "$LOG_ROOT/prepare_gpu0.log" 2>&1 &
prepare_pid=$!

launch_kind() {
  local kind=$1 gpu=$2 cpus=$3
  nohup taskset -c "$cpus" bash -lc "
    set -euo pipefail
    cd '$REPO_ROOT'
    while [[ ! -f '$RESULT_ROOT/PREPARE_COMPLETE' ]]; do
      if ! kill -0 '$prepare_pid' 2>/dev/null; then
        echo 'Preparation failed before PREPARE_COMPLETE; see prepare_gpu0.log.' >&2
        exit 1
      fi
      sleep 60
    done
    export CUDA_VISIBLE_DEVICES='$gpu' CONDA_ENV=base HUB_PPR_BATCH_SIZE=16
    export STRUCTURAL_METRICS_WORKERS=8 STRUCTURAL_METRICS_PARALLEL_MIN_EDGES=100000
    bash commands/run_primekg_full_kind_paper_eval.sh '$kind'
  " > "$LOG_ROOT/${kind}_gpu${gpu}.log" 2>&1 &
  printf '%s\n' "$!"
}

node_pid=$(launch_kind node 1 16-31)
edge_pid=$(launch_kind edge 2 32-47)
feature_pid=$(launch_kind feature 3 48-63)

nohup taskset -c 0-15 bash -lc "
  set -euo pipefail
  cd '$REPO_ROOT'
  while true; do
    complete=1
    for spec in '$node_pid node' '$edge_pid edge' '$feature_pid feature'; do
      read -r pid kind <<< \"\$spec\"
      if [[ ! -f '$RESULT_ROOT/.matrix_state/'\"\$kind\"'.complete' ]]; then
        complete=0
        if ! kill -0 \"\$pid\" 2>/dev/null; then
          echo \"\$kind job ended without a completion marker\" >&2
          exit 1
        fi
      fi
    done
    [[ \"\$complete\" == 1 ]] && break
    sleep 60
  done
  CONDA_ENV=base bash commands/finalize_primekg_full_paper_eval.sh
" > "$LOG_ROOT/finalize.log" 2>&1 &
finalize_pid=$!

printf 'prepare PID=%s GPU=0 CPUs=0-15\n' "$prepare_pid"
printf 'node PID=%s GPU=1 CPUs=16-31\n' "$node_pid"
printf 'edge PID=%s GPU=2 CPUs=32-47\n' "$edge_pid"
printf 'feature PID=%s GPU=3 CPUs=48-63\n' "$feature_pid"
printf 'finalizer PID=%s\n' "$finalize_pid"
