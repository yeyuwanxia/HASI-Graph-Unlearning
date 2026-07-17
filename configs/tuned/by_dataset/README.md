# Tuned Configs By Dataset

This directory contains dataset-specific tuned HASI configurations.

Layout:

```text
configs/tuned/by_dataset/
  cora/
    node.yaml
    edge.yaml
    feature.yaml
  citeseer/
    node.yaml
    edge.yaml
    feature.yaml
  pubmed/
    node.yaml
    edge.yaml
    feature.yaml
  hetionet-full-nosource/
    node.yaml
    edge.yaml
    feature.yaml
  primekg-disease-gene-small/
    edge.yaml
    feature.yaml
```

Selection notes:

- `cora/*` uses the tuned Cora configurations from `configs/tuned/cora_*`.
- `citeseer/*` currently reuses the Cora tuned configurations because Citeseer was used as a transfer validation set, not separately tuned.
- `pubmed/node.yaml` reuses the node tuned configuration because PubMed node was not separately tuned.
- `pubmed/edge.yaml` uses the PubMed-selected edge configuration from `results/tuning/pubmed/edge/repair32_anchor1p0_0p2_forget000`.
- `pubmed/feature.yaml` uses the PubMed-selected feature configuration from `results/tuning/pubmed/feature/drift1e2_lowforget005`.

- `hetionet-full-nosource/*` contains configurations tuned on `hetionet-small-nosource` and transferred unchanged to the full homogeneous projection. The original small-dataset tuning paths remain recorded in each YAML; these are transferred configurations, not full-dataset retuning outputs.

- `primekg-disease-gene-small/edge.yaml` keeps the default HASI configuration because the edge privacy-refine sweep did not improve the validation privacy-utility trade-off over default.
- `primekg-disease-gene-small/feature.yaml` uses the selected feature configuration from `results/tuning/primekg-disease-gene-small/feature/best_config.yaml`.
- `primekg-disease-gene-small/node.yaml` is intentionally omitted until the node round-2 coarse sweep is finalized.
