# Frozen Evaluation Protocol (2026-07-15)

Protocol id: `paper_eval_20260715_v1`.

## Shared inputs

All compared methods must use the same shared-base artifact and the same forget-set JSON. Edge requests are sampled from unique undirected training-subgraph edges; applying a request removes its undirected closure.

## Privacy

Node and Edge medium MIA uses a deterministic stratified half of the target samples for attacker training and the disjoint half for evaluation. A supplied shadow set remains the preferred training source. Report medium train/evaluation sizes with the AUC.

Strong MIA is reported together with its permutation-null mean, standard deviation, and p-value. Global feature-dimension requests use Scheme A: feature privacy MIA is not applicable and utility/compliance are reported instead.

## Edge forgetting

The forgotten-edge score is cosine affinity mapped to `[0, 1]`; it is an embedding-affinity proxy, not a calibrated link probability. Report forgotten-edge score drop, a seeded retained-edge control drop, and their difference. `request_applied=true` requires every requested unique undirected edge to be absent after unlearning.

## Exact retrain alignment

Each Edge request has one exact-retrain reference bound to the dataset, shared-base path, forget-set path, and forget-set SHA-256. All methods for that request load the same reference. Report test-mask JS/TV distance to exact retrain, improvement relative to the original model, prediction disagreement, and forgotten-edge affinity gap.

## Change control

Do not revise metric definitions after inspecting method rankings. Any semantic change requires a new protocol id and rerunning every affected method/request pair. Runner validation must reject missing or mismatched protocol ids and exact-retrain references.

## Exact structural evaluator execution

Average local clustering remains the exact NetworkX definition. Large graphs may evaluate node chunks with a deterministic degree-balanced fork pool; result JSONs record `clustering_backend` and `clustering_workers`. This changes execution only, not the metric. Evaluator runtime is excluded from method-level `unlearn_time_seconds` and must not be presented as unlearning speedup.
