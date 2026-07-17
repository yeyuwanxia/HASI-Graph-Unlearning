#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export RUN_TYPES=edge
exec bash "$SCRIPT_DIR/run_pubmed_baselines_mia_v2.sh" "$@"
