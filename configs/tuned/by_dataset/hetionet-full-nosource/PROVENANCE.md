# Hetionet-Full-NoSource Tuned Configuration Provenance

This directory contains the active tuned HASI configurations used for `hetionet-full-nosource`.

- `node.yaml`: `results/tuning/hetionet_full_node_optimal_search_gpu3_20260717/search/node/configs/config_0037.json`
- `edge.yaml`: final Edge selection, originally from `results/tuning/hetionet-small-nosource/edge_round2_comprehensive_refine/configs/config_0014.json`
- `feature.yaml`: `results/tuning/hetionet_full_feature_optimal_search_gpu0_20260717/final/feature/best_config.yaml`

`node.yaml` uses the full-dataset `cfg0037` selection with every HASI component
enabled. `edge.yaml` retains the parameters originally selected on the small
dataset, but it is now the final full-dataset Edge selection: repeated
Hetionet-Full searches did not find a candidate that improved privacy without
unacceptable utility or exact-alignment regression. `feature.yaml` uses the
active full-dataset tuned feature config because its formal `hasi_tuned` feature
results improve Hetionet feature accuracy/drop over both `hasi_default` and the
previous transferred feature config. Feature MIA privacy remains not applicable
for global feature-dimension deletion, so the feature selection is
utility/compliance based rather than privacy-AUC based.

## Final Node Selection

The final Node config is full-dataset search `cfg0037`:

```yaml
anchor_stabilization:
  lambda1: 0.25
  lambda2: 0.02

inpainting:
  cc_drop_threshold: 0.3
  min_damage_ratio: 0.1
  edge_threshold: 0.5
  max_added_edges: 256
  repair_ratio: 0.05

unlearning:
  edge_forget_loss_mode: uniform
  node_forget_loss_mode: uniform
  forget_weight: 1.5
  finetune_epochs: 150
  finetune_lr: 0.003
```

Anchor stabilization has positive `lambda1/lambda2`, DAR is enabled, and full
inpainting is enabled. The candidate result payloads confirm that anchor
stabilization was active, inpainting ran, and Node training used `uniform`.
The explicit `node_forget_loss_mode` in the final YAML preserves that runtime
behavior after the Node loss-mode implementation was separated.

Formal evaluation covered both ratios and base seeds `42/123/2024`:

`results/archive/candidate_evals/hetionet-full-nosource/hetionet_node_full_components_cfg0037_from_tuning_20260721/`

Formal Node means:

| Config | Ratio | Accuracy | Accuracy drop | Macro-F1 | Weak AUC | Medium AUC | Strong AUC | Unlearn time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| transferred | 0.05 | 0.827079 | 0.001913 | 0.466700 | 0.852764 | 0.965297 | 0.841368 | 684.26s |
| full tuned cfg0037 | 0.05 | 0.827044 | 0.001949 | 0.465980 | 0.853495 | 0.965525 | 0.842418 | 670.75s |
| transferred | 0.10 | 0.826618 | 0.002374 | 0.470316 | 0.856514 | 0.958831 | 0.843671 | 915.84s |
| full tuned cfg0037 | 0.10 | 0.826548 | 0.002445 | 0.469872 | 0.857345 | 0.958909 | 0.844263 | 874.60s |

Compared with the transferred config, `cfg0037` keeps Accuracy within
`0.000071` and Macro-F1 within `0.000721`, while running faster at both ratios.
Its Accuracy and Macro-F1 are above GNNDelete and both GraphEraser variants and
below Retrain, so it beats `3/4` utility baselines. All three privacy AUCs are
slightly worse than the transferred config at both ratios; this selection must
therefore be described as the final all-components utility configuration, not
as a privacy or strict Pareto improvement.

Two later full-dataset searches did not find a replacement:

- `results/tuning/hetionet_node_true_original_kl_guarded_gpu0_20260725/`: 11 candidate configs, zero hard-pass candidates.
- `results/tuning/hetionet_node_transfer_repair_guarded_gpu3_20260725/`: 8 candidate configs, zero hard-pass candidates.

The six candidate JSON files were promoted without GPU recomputation because
they already used the formal forget sets, exact-retrain references, and
three-seed protocol. Promotion changed only the method labels and config-source
path; all metric payloads were verified unchanged before regenerating the
formal aggregate.

### Retired Node Transfer Config

The previous Node config was transferred from:

`results/tuning/hetionet-small-nosource/node_round2_privacy_wide/configs/config_0007.json`

Its relevant parameters are retained here:

```yaml
anchor_stabilization:
  lambda1: 0.0
  lambda2: 0.02

inpainting:
  cc_drop_threshold: 0.3
  min_damage_ratio: 0.1
  edge_threshold: 0.5
  max_added_edges: 256
  repair_ratio: 0.05

unlearning:
  edge_forget_loss_mode: uniform
  node_forget_loss_mode: uniform
  forget_weight: 0.8
  finetune_epochs: 120
  finetune_lr: 0.003
```

The result payloads confirm that the retired Node runs actually used `uniform`.
Its `lambda1=0.0` disabled the primary anchor-stabilization loss term, which is
why it was not retained as the final all-components configuration despite its
slightly better utility/privacy means.

The original transfer Node JSON files and pre-promotion aggregate are retained
at:

`results/archive/formal_snapshots/hetionet-full-nosource/mia_v2_hetionet-full-nosource_eval_node_pre_cfg0037_hetionet_node_cfg0037_promotion_20260726_20260726_103028/`

## Final Edge Selection

The final Edge parameters are:

```yaml
anchor_stabilization:
  lambda1: 2.0
  lambda2: 0.5

inpainting:
  cc_drop_threshold: 0.4
  min_damage_ratio: 0.1
  edge_threshold: 0.5
  max_added_edges: 96
  repair_ratio: 0.35

unlearning:
  edge_forget_loss_mode: original_kl
  forget_weight: 0.075
  finetune_epochs: 50
  finetune_lr: 0.005
```

All HASI components remain enabled. The final decision also considers:

- `results/tuning/hetionet_edge_privacy_reverse_guarded_gpu3_20260723/`: 16 candidates over two ratios and two seeds; zero privacy improvements and zero hard-pass candidates.
- `results/tuning/hetionet_edge_node_robust_pareto_gpu3_20260721/`: no hard-pass replacement.
- Formal evaluation of the retained config over two ratios and three seeds.

The formal Edge results are labeled `hasi_tuned`. Their metrics are identical to
the previously labeled `hasi_transfer_hetionet_small_tuned` Edge results; only
the final method label was unified.

The original transfer-labeled Edge JSON files and the pre-unification aggregate
are retained at:

`results/archive/formal_snapshots/hetionet-full-nosource/mia_v2_hetionet-full-nosource_eval_edge_pre_final_unification_hetionet_edge_transfer_final_promotion_20260725_20260725_150506/`

## Rejected Edge Full-Dataset Candidate

Hetionet-full Edge search `config_0001` (`forget_weight=0.03`,
`finetune_lr=0.005`, `inpainting_max_added_edges=128`) was evaluated on the six
formal ratio/seed combinations, but it was not adopted because its formal
multi-seed privacy result regressed relative to `hasi_default`. Its result files
are retained as rejected-candidate evidence under:

`results/archive/candidate_evals/hetionet-full-nosource/mia_v2_hetionet-full-nosource_eval_rejected_edge_cfg0001_repair_20260721_221911/rejected_hasi_tuned_edge_cfg0001/`

The formal tree therefore continues to use the transferred Edge configuration
recorded in `edge.yaml`. The pre-candidate Edge results were restored from their
promotion archive and the formal aggregate was regenerated.

## Retired Feature Transfer Config

The previous `feature.yaml` came from `results/tuning/hetionet-small-nosource/feature_round3_utility_privacy_refine/configs/config_0032.json` and produced the formal method label `hasi_transfer_hetionet_small_tuned`.

Retained parameters from that transferred config:

```yaml
anchor_stabilization:
  lambda1: 1.0
  lambda2: 0.03

inpainting:
  cc_drop_threshold: 0.3
  min_damage_ratio: 0.1
  edge_threshold: 0.5
  max_added_edges: 256
  repair_ratio: 0.25

unlearning:
  edge_forget_loss_mode: uniform
  forget_weight: 0.3
  finetune_epochs: 80
  finetune_lr: 0.005
```

Formal feature summary over `r0.05/r0.1` and seeds `42/123/2024`:

- `hasi_default`: accuracy after `0.814766`, accuracy drop `0.014227`, F1 macro `0.400008`, unlearn time `6.302058s`.
- `hasi_transfer_hetionet_small_tuned`: accuracy after `0.815297`, accuracy drop `0.013695`, F1 macro `0.404178`, unlearn time `7.783408s`.
- `hasi_tuned` full feature tuned result: accuracy after `0.823022`, accuracy drop `0.005971`, F1 macro `0.404266`, unlearn time `9.765913s`.
