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
  primekg-full-nosource/
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

- `hetionet-full-nosource/*` contains the official final HASI configurations for Hetionet full. `node.yaml` uses full-dataset all-components `cfg0037`; `feature.yaml` uses the full-dataset feature selection; `edge.yaml` retains the small-dataset transfer after repeated full-dataset searches found no hard-pass replacement. `PROVENANCE.md` records the retired transfer parameters and selection trade-offs.

- `primekg-full-nosource/node.yaml` uses the full-dataset true-original-KL balance search `cfg0001`, validated on all six formal ratio/seed combinations. `edge.yaml` uses the full-dataset privacy-guarded search `cfg0009`, and `feature.yaml` uses the full-dataset utility-priority `cfg0000`. The provenance file records retired transfer parameters and selection trade-offs.

- `primekg-disease-gene-small/edge.yaml` keeps the default HASI configuration because the edge privacy-refine sweep did not improve the validation privacy-utility trade-off over default.
- `primekg-disease-gene-small/feature.yaml` uses the selected feature configuration from `results/tuning/primekg-disease-gene-small/feature/best_config.yaml`.
- `primekg-disease-gene-small/node.yaml` is intentionally omitted until the node round-2 coarse sweep is finalized.
