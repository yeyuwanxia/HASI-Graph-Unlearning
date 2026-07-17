#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-base}"
ROOT="results/mia_v2_primekg-full-nosource_eval"
PY=(conda run --no-capture-output -n "$CONDA_ENV" python)

for kind in node edge feature; do
  test -f "$ROOT/.matrix_state/${kind}.complete"
done

"${PY[@]}" experiments/aggregate_results.py \
  --input_dir "$ROOT" \
  --pattern '*primekg-full-nosource_*.json' \
  --output_json "$ROOT/aggregate_summary.json" \
  --output_csv "$ROOT/aggregate_summary.csv" \
  --group_by method,dataset,unlearning_type,ratio,selection \
  --metrics metrics.utility.accuracy_after,metrics.utility.accuracy_drop,metrics.utility.f1_macro_after,metrics.structure.degree_kl_divergence,metrics.structure.clustering_coefficient_change,metrics.structure.component_count_change,metrics.privacy.weak_auc,metrics.privacy.medium_auc,metrics.privacy.medium_train_size,metrics.privacy.medium_eval_size,metrics.privacy.strong_auc,metrics.privacy.strong_auc_null_mean,metrics.privacy.strong_auc_null_std,metrics.privacy.strong_auc_pvalue,metrics.privacy.privacy_score,metrics.representation.embedding_l2_mean,metrics.representation.member_embedding_l2_mean,metrics.representation.neighbor_drift_mean,metrics.edge_forgetting.forgotten_score_drop_mean,metrics.edge_forgetting.retained_control_score_drop_mean,metrics.edge_forgetting.targeted_drop_vs_control,metrics.edge_forgetting.forgotten_unlearned_to_retrain_abs_gap_mean,metrics.exact_retrain_alignment.unlearned_to_retrain_js_mean,metrics.exact_retrain_alignment.improvement_over_original_js,metrics.exact_retrain_alignment.unlearned_to_retrain_tv_mean,metrics.exact_retrain_alignment.prediction_disagreement_rate,metrics.efficiency.unlearn_time_seconds,metrics.efficiency.online_wall_clock_seconds,metrics.efficiency.offline_preprocessing_seconds

"${PY[@]}" -c '
import json
from pathlib import Path

root = Path("results/mia_v2_primekg-full-nosource_eval")
files = [
    path for path in root.rglob("*.json")
    if "artifacts" not in path.parts
    and path.name not in {"aggregate_summary.json"}
    and "primekg-full-nosource_" in path.name
]
payloads = [json.loads(path.read_text(encoding="utf-8")) for path in files]
default_payloads = [
    item for item in payloads
    if item.get("method") != "hasi_transfer_primekg_dg_small_tuned"
]
assert len(default_payloads) == 96, len(default_payloads)
assert all(item.get("dataset") == "primekg-full-nosource" for item in default_payloads)
assert all(item["metrics"]["evaluation_protocol"]["version"] == "paper_eval_20260715_v1" for item in default_payloads)
by_type = {kind: sum(item["metrics"]["unlearning_type"] == kind for item in default_payloads) for kind in ("node", "edge", "feature")}
assert by_type == {"node": 30, "edge": 36, "feature": 30}, by_type
print({"default_results": len(default_payloads), "all_results_present": len(payloads), "default_by_type": by_type})
'

touch "$ROOT/FULL_MATRIX_COMPLETE"
echo "[complete] PrimeKG full no-source formal matrix: 96 results"
