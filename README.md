# HASI Graph Unlearning

This repository implements HASI: Hub-Anchored Structural Inpainting for graph
unlearning experiments. The mainline code is organized around configurable HASI
runs, fixed forget-set protocols, privacy/structure/utility metrics, and
result aggregation.

## Layout

- `docs/`: HASI project overview, method reference, and medical graph story line.
- `configs/`: YAML experiment configuration, including `hasi_default.yaml`.
- `src/hasi/`: HASI modules: GHI, AS, EDP, GSI, DAR, and the unlearner.
- `src/models/`: GCN, GAT, GraphSAGE, and training utilities.
- `src/evaluation/`: utility, structure, representation, efficiency, and MIA metrics.
- `src/baselines/`: retrain, GraphEraser-style adapters, and official-source guards.
- `experiments/`: runnable experiment, protocol generation, and aggregation entry points.

## Environment

Install the mainline dependencies on the remote experiment machine:

```bash
pip install -r requirements.txt
```

If the server needs a CUDA-specific PyTorch wheel, install the matching
`torch` package first, then install the remaining requirements.

## Data Download

Download and prepare datasets before generating protocols or running
experiments:

```bash
python experiments/download_datasets.py --datasets cora,citeseer,pubmed
```

To include large datasets:

```bash
python experiments/download_datasets.py --datasets cora,citeseer,pubmed,reddit
```

By default, experiment and protocol scripts expect datasets to already exist
under `data/raw/`. If you intentionally want a run command to download missing
data, pass `--allow_download`.

## HASI Mainline

Single run with the default YAML config:

```bash
python experiments/run_hasi.py --config configs/hasi_default.yaml --mode unlearn --dataset_name cora
```

Use a fixed forget-set protocol:

```bash
python experiments/run_hasi.py \
  --config configs/hasi_default.yaml \
  --mode unlearn \
  --forget_set_file experiments/forget_sets/cora_node_r0p1_random_all_seed42.json \
  --seed 42
```

Override key ablation parameters from the CLI:

```bash
python experiments/run_hasi.py \
  --mode unlearn \
  --dataset_name cora \
  --unlearning_type node \
  --forget_ratio 0.1 \
  --seed 42 \
  --inpainting_mode full \
  --forget_weight 0.1 \
  --dar_k 5 \
  --dar_min_distance 2 \
  --anchor_lambda1 2.0 \
  --anchor_lambda2 0.5 \
  --erf_threshold 0.01
```

Disable DAR for w/o-DAR ablation:

```bash
python experiments/run_hasi.py --mode unlearn --dataset_name cora --no-dar-enabled
```

## Forget-Set Protocols

Generate one protocol file:

```bash
python experiments/generate_forget_sets.py \
  --dataset_name cora \
  --unlearning_type node \
  --forget_ratio 0.05 \
  --seed 42 \
  --selection random_all
```

Generate the default protocol matrix:

```bash
python experiments/generate_forget_protocols.py
```

The default matrix is:

- datasets: `cora,citeseer,pubmed`
- unlearning types: `node,edge,feature`
- node selection: `random_all`
- ratios: `0.05,0.1`
- seeds: `42,123,2024`

It writes JSON files under `experiments/forget_sets/` and a `manifest.json`.

## Result Aggregation

After experiment JSON files have been written under `results/` (for example `results/hasi/` and `results/baselines/<baseline>/`), aggregate them:

```bash
python experiments/aggregate_results.py
```

Default outputs:

- `results/aggregate_summary.json`
- `results/aggregate_summary.csv`

The default aggregation reports mean/std/min/max for utility, structural,
privacy, and efficiency metrics, grouped by method, dataset, unlearning type,
ratio, and forget-set selection.

## Baselines

List registered baselines:

```bash
python experiments/run_baselines.py --list_baselines
```

Run a baseline with the same forget-set protocol:

```bash
python experiments/run_baselines.py \
  --baseline retrain \
  --dataset_name cora \
  --unlearning_type node \
  --forget_set_file experiments/forget_sets/cora_node_r0p1_random_all_seed42.json
```

Baseline names without a `-surrogate` suffix are reserved for official or
official-derived adapters. `gnndelete-surrogate`, `gif-surrogate`,
`sgu-surrogate`, and `agu-surrogate` are fine-tuning sanity checks and should
not be reported as official paper baselines.

Official source metadata:

```bash
python experiments/install_official_baselines.py --list
```

## Metrics

Mainline result JSON files include:

- utility: accuracy, F1, accuracy/F1 drops
- representation: embedding drift for all/member/neighbor nodes
- structure: degree KL divergence, clustering coefficient change, component count change
- privacy: weak/medium/strong MIA AUC, overall MIA AUC, privacy score
- efficiency: unlearn time, retrain time when available, speedup vs retrain
