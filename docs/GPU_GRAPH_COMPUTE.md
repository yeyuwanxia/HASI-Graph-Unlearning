# GPU Graph Compute Backend

The optional torch backend changes execution, not HASI's mathematical stages.

- ERF/PPR still performs the configured fixed number of restart-and-propagate
  steps with the same `alpha`, seed mass, dangling-node rule, and threshold.
- DAR still uses exact unweighted shortest-path distances. The torch backend
  implements level-synchronous BFS and caches each source distance map only for
  the active graph. On undirected graphs, diversity checks root BFS at the few
  selected anchors and reuse those exact distance maps for every candidate.
- HubScore uses the existing NetworkX implementation on a cache miss. Its final
  score map is content-addressed and reused for the same graph, data split,
  model state, HubScore parameters, and gradient-scoring configuration.

## Runtime Selection

`--graph_compute_backend auto` selects torch only when the resolved graph device
is CUDA-capable. `cpu` forces the legacy NetworkX path; `torch` is strict and
raises if the requested device is unavailable. Every affected-region result
records the requested/used backend and any fallback reason.

Disable cache reuse with `--no-hub-score-cache`. Ad hoc runs default to
`results/cache/hub_scores`. Formal PubMed runs use the provenance-scoped root:

```text
results/mia_v2_pubmed_eval/hasi/artifacts/hub_scores/pubmed/seed<seed>/
```

Each schema-v2 artifact records its build time and producer metadata. Prepare
the three shared-base artifacts before formal runs:

```bash
CUDA_VISIBLE_DEVICES=2 CONDA_ENV=base \
  bash commands/prepare_pubmed_hub_score_artifacts.sh
```

Formal commands must add `--require-hub-score-cache-hit`. A cache miss then
fails before unlearning instead of mixing cold and warm timings. Report the
artifact's `offline_preprocessing_seconds` separately from request-dependent
`unlearn_time_seconds`; do not compare mixed process wall-clock values as pure
online latency.

With host GPU 2 exposed, use the process-local ordinal:

```bash
CUDA_VISIBLE_DEVICES=2 python experiments/run_hasi.py \
  ... \
  --device cuda:0 \
  --graph_compute_backend auto
```

## Parity Gate

Run the standalone gate in any environment containing NetworkX and a
CUDA-capable PyTorch build:

```bash
CUDA_VISIBLE_DEVICES=2 python experiments/check_graph_compute_parity.py --device cuda:0
```

The gate checks PPR values and affected-region identity, exact BFS distances,
and deterministic DAR phase-1 selection. Formal experiments should not start
unless it reports `status: ok`.
