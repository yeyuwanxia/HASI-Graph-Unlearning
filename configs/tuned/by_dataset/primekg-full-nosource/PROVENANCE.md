# PrimeKG-Full-NoSource Tuned Configuration Provenance

This directory contains the active tuned HASI configurations for
`primekg-full-nosource`.

- `edge.yaml`: `results/tuning/primekg_full_edge_privacy_guarded_gpu0_20260721/search/edge/configs/config_0009.json`
- `feature.yaml`: `results/tuning/primekg_full_feature_utility_guarded_gpu2_20260721/final/feature/configs/config_0000.json`
- `node.yaml`: `results/tuning/primekg_node_true_original_kl_balance_gpu1_20260724/search/node/configs/config_0001.json`

## Final Node Selection

The selected Node configuration is `cfg0001` from the true-original-KL balance
search. It keeps anchor stabilization, DAR, and full inpainting enabled, and
uses these tuned parameters:

```yaml
anchor_stabilization:
  lambda1: 2.0
  lambda2: 0.2

inpainting:
  cc_drop_threshold: 0.3
  min_damage_ratio: 0.1
  edge_threshold: 0.5
  max_added_edges: 256
  repair_ratio: 0.2

unlearning:
  edge_forget_loss_mode: original_kl
  node_forget_loss_mode: original_kl
  forget_weight: 0.2
  finetune_epochs: 50
  finetune_lr: 0.01
```

Formal candidate evaluation covered both ratios and base seeds `42/123/2024`:

`results/archive/candidate_evals/primekg-full-nosource/primekg_node_true_original_kl_cfg0001_cfg0003_20260725/`

The result payloads confirm that Node training actually used `original_kl`
against `original_graph_logits`. Formal Node means are:

| Config | Ratio | Accuracy | Accuracy drop | Macro-F1 | Weak AUC | Medium AUC | Strong AUC | Unlearn time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| transferred | 0.05 | 0.491377 | 0.013237 | 0.265330 | 0.808086 | 0.992690 | 0.825158 | 2457.69s |
| full tuned cfg0001 | 0.05 | 0.494496 | 0.010118 | 0.271216 | 0.806627 | 0.998081 | 0.825033 | 2352.12s |
| transferred | 0.10 | 0.495798 | 0.008816 | 0.273309 | 0.801469 | 0.997102 | 0.815537 | 4137.36s |
| full tuned cfg0001 | 0.10 | 0.496056 | 0.008558 | 0.273949 | 0.797929 | 0.997599 | 0.814221 | 3914.82s |

Compared with the transferred config, `cfg0001` improves mean Accuracy,
Accuracy drop, Macro-F1, Weak AUC, and Strong AUC at both ratios, and all six
runs are faster. Across the six paired runs, utility and Weak AUC improve in
`4/6`, while Strong AUC improves in `5/6`. Medium AUC is worse in all six runs,
so this is the balanced final selection rather than a strict Pareto improvement
or evidence that HASI privacy is better than most baselines.

The six candidate JSON files were promoted without GPU recomputation because
they already used the formal forget sets, exact-retrain references, and
three-seed protocol. Promotion changed only the method labels and config-source
path; metric payloads were verified unchanged before regenerating the formal
aggregate.

### Retired Node Transfer Config

The previous Node config was transferred from:

`configs/tuned/by_dataset/primekg-disease-gene-small-nosource/node.yaml`

Its relevant parameters are retained here:

```yaml
anchor_stabilization:
  lambda1: 2.0
  lambda2: 0.05

inpainting:
  cc_drop_threshold: 0.3
  min_damage_ratio: 0.1
  edge_threshold: 0.5
  max_added_edges: 256
  repair_ratio: 0.2

unlearning:
  edge_forget_loss_mode: original_kl
  forget_weight: 0.2
  finetune_epochs: 80
  finetune_lr: 0.01
```

These retired results predate the separate `node_forget_loss_mode` option.
Although the config named `edge_forget_loss_mode: original_kl`, the result
payloads confirm that Node training actually used `uniform`; they must not be
described as original-KL runs.

The original transfer Node JSON files and pre-promotion aggregate are retained
at:

`results/archive/formal_snapshots/primekg-full-nosource/mia_v2_primekg-full-nosource_eval_node_pre_cfg0001_primekg_node_cfg0001_promotion_20260726_20260726_101034/`

`cfg0003` remains in the candidate result directory as supporting evidence but
was not selected.

## Final Edge Selection

The selected Edge configuration is search `cfg0009`, also evaluated as final
stage `cfg0000`. It keeps anchor stabilization, DAR, and full inpainting
enabled. Formal evaluation used both deletion ratios and base seeds
`42/123/2024`:

`results/archive/candidate_evals/primekg-full-nosource/primekg_edge_cfg0009_from_tuning_20260725/`

Compared with the previous transferred Edge configuration, the selected config
improves Accuracy, Macro-F1, Weak AUC, and Strong AUC at both ratios. All six
individual runs improve Accuracy, Macro-F1, Weak AUC, and Strong AUC. It is not
a strict Pareto improvement: Medium AUC is worse at `r=0.05`, and exact-retrain
alignment is worse on average. Claims should therefore describe it as a
utility plus Weak/Strong privacy improvement, not as uniformly better on every
privacy or alignment metric.

Formal Edge means:

| Config | Ratio | Accuracy | Macro-F1 | Weak AUC | Medium AUC | Strong AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| transferred | 0.05 | 0.503802 | 0.285514 | 0.752670 | 0.873217 | 0.778063 |
| full tuned cfg0009 | 0.05 | 0.510260 | 0.299315 | 0.585050 | 0.876747 | 0.719389 |
| transferred | 0.10 | 0.500361 | 0.279609 | 0.735402 | 0.882663 | 0.783952 |
| full tuned cfg0009 | 0.10 | 0.511445 | 0.300807 | 0.625735 | 0.881724 | 0.748111 |

## Retired Edge Transfer Config

The previous Edge config was transferred from:

`configs/tuned/by_dataset/primekg-disease-gene-small-nosource/edge.yaml`

Its relevant parameters are retained here:

```yaml
anchor_stabilization:
  lambda1: 2.0
  lambda2: 0.5

inpainting:
  cc_drop_threshold: 0.3
  min_damage_ratio: 0.1
  edge_threshold: 0.5
  max_added_edges: 256
  repair_ratio: 0.35

unlearning:
  edge_forget_loss_mode: original_kl
  forget_weight: 0.1
  finetune_epochs: 50
  finetune_lr: 0.01
```

The six candidate results were promoted without GPU recomputation because they
already used the formal forget sets, exact-retrain references, evaluation
protocol, and three base seeds. Promotion changed only the method label and
config-source path; all metric payloads were verified unchanged before the
formal aggregate was regenerated.

The original transfer Edge JSON results and pre-promotion aggregate are retained
at:

`results/archive/formal_snapshots/primekg-full-nosource/mia_v2_primekg-full-nosource_eval_edge_pre_cfg0009_primekg_edge_cfg0009_promotion_20260725_20260725_135238/`

## Final Feature Selection

The final Feature configuration keeps the default parameters and changes only:

```yaml
unlearning:
  finetune_epochs: 80
```

It was selected by the three-seed utility-priority stage and then evaluated on
all six formal ratio/seed combinations:

`results/archive/candidate_evals/primekg-full-nosource/primekg_feature_cfg0000_from_tuning_20260725/`

Formal Feature means:

| Config | Accuracy | Accuracy drop | Macro-F1 | Unlearn time |
| --- | ---: | ---: | ---: | ---: |
| default | 0.485487 | 0.019127 | 0.255830 | 11.002s |
| transferred | 0.477567 | 0.027047 | 0.246074 | 15.533s |
| full tuned cfg0000 | 0.493594 | 0.011020 | 0.265286 | 15.243s |

All three formal base seeds improve Accuracy and Macro-F1 over both default and
the transferred config. Feature compliance is `ok` in all six runs. Node-level
MIA is not applicable because this protocol deletes global feature dimensions.
PrimeKG exposes three candidate feature dimensions, so both evaluated ratios
round to one deleted dimension and produce identical utility means.

### Retired Feature Transfer Config

The previous Feature config was transferred from:

`configs/tuned/by_dataset/primekg-disease-gene-small-nosource/feature.yaml`

Its parameters are retained here:

```yaml
anchor_stabilization:
  lambda1: 1.0
  lambda2: 0.05

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

The six formal candidate results were promoted without GPU recomputation.
Promotion changed only the method label and config-source path; all metric
payloads were verified unchanged before regenerating the formal aggregate.

The original transfer Feature JSON files and pre-promotion aggregate are
retained at:

`results/archive/formal_snapshots/primekg-full-nosource/mia_v2_primekg-full-nosource_eval_feature_pre_cfg0000_primekg_feature_cfg0000_promotion_20260725_20260725_214822/`
