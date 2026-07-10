# OpenGU-Adapted Baselines

This directory is intentionally separate from the repository's existing
`src/`, `experiments/`, and `results/` trees.

It stores:

- selected original OpenGU source files under `vendor/opengu_selected/`
- adapted baseline wrappers under `src/opengu_adapted_baselines/`
- run entrypoints under `scripts/`
- future outputs under `results/`
- future GraphEraser shard artifacts and adapted optimizer artifacts beside result JSON files under `results/`

The adapted runners use this repository's standard protocol:

- shared base artifacts from `results/shared_base/{dataset}/seed{seed}`
- JSON forget sets from `experiments/forget_sets_eval/...`
- the same dataset loader, GNN backbone, and evaluation metrics as HASI

## Supported Baselines

- `grapheraser-bekm`
- `grapheraser-blpa`
- `gif` for edge unlearning only
- `gnndelete`
- `megu`

`feature` forgetting is supported for GraphEraser, GNNDelete, and MEGU as a
unified-protocol extension. GIF remains edge-only. The GNNDelete adapter keeps
OpenGU provenance and ports its retention/randomness objective shape onto this
repository's shared-base trainer because the original OpenGU implementation
requires OpenGU's model_zoo, deletion-layer GNN variants, trainer stack, and
checkpoint layout. The MEGU adapter keeps OpenGU provenance and ports its
retain/forget objective shape onto the same shared-base trainer because the
original OpenGU implementation requires OpenGU's Learning_based_pipeline,
trainer stack, text forget-set paths, and CorrectAndSmooth post-processing.

## Example

```bash
conda run -n graphunlearning python opengu_adapted_baselines/scripts/run_opengu_adapted_baseline.py \
  --baseline grapheraser-bekm \
  --dataset_name pubmed \
  --unlearning_type node \
  --seed 42 \
  --base_artifact_root results/shared_base \
  --forget_set_file experiments/forget_sets_eval/pubmed/pubmed_node_r0p05_random_train_base42_fseed70042.json
```

Outputs and artifacts default to:

```text
opengu_adapted_baselines/results/<dataset>_eval/<experiment>/baselines/<method>/<type>/
```

GraphEraser offline artifacts are stored under the same method/type folder:

```text
opengu_adapted_baselines/results/<dataset>_eval/<experiment>/baselines/<method>/<type>/artifacts/seed<seed>/
```

## Wording

Use wording like:

```text
Baselines are adapted from the OpenGU benchmark and integrated into our
unified evaluation protocol.
```

Do not describe these runs as unmodified official baseline code.
