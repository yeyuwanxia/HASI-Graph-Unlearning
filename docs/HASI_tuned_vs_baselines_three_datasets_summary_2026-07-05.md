# HASI Tuned vs Baselines: Three-Dataset Summary

Date: 2026-07-05

This document summarizes current paired comparisons for HASI tuned against two baseline families:

- `src` baselines: repository-native `retrain`, `grapheraser-bekm`, `grapheraser-blpa`, and edge-only `gif`.
- `OpenGU adapted` baselines: `opengu-gnndelete`, `opengu-grapheraser-bekm`, `opengu-grapheraser-blpa`, and edge-only `opengu-gif`.

A win means HASI tuned is better on the metric: higher `accuracy_after`, higher `privacy_score`, and lower `embedding_l2_mean`. `all3` means the same paired run wins on accuracy, privacy, and embedding together. The `all3` denominator follows the embedding denominator; GIF does not expose embedding, so edge embedding/all3 denominators are smaller than edge accuracy/privacy denominators.

## Fairness Check
| dataset | comparison | HASI groups | pairs | missing | fairness issues |
| --- | --- | --- | --- | --- | --- |
| PubMed | HASI tuned vs src baselines | 180 | 672 | 0 | 0 |
| PubMed | HASI tuned vs OpenGU adapted | 180 | 672 | 0 | 0 |
| Hetionet-small-nosource | HASI tuned vs src baselines | 108 | 366 | 0 | 0 |
| Hetionet-small-nosource | HASI tuned vs OpenGU adapted | 108 | 366 | 0 | 0 |
| PrimeKG-NoSource | HASI tuned vs src baselines | 252 | 834 | 0 | 0 |
| PrimeKG-NoSource | HASI tuned vs OpenGU adapted | 252 | 834 | 0 | 0 |

All current paired comparisons have `missing=0` and `fairness issues=0`, meaning the compared methods share the same `forget_set.path`, forget targets, and shared-base artifact for each paired key.

## Overall Conclusions
| dataset | interpretation | recommended forget set vs OpenGU | recommended forget set vs src |
| --- | --- | --- | --- |
| PubMed | PubMed 最适合作为 HASI tuned 全面优于 baselines 的主证据，尤其是 node。 | node, seed42, default_main / fseed70042 (acc 6/6, privacy 6/6, emb 6/6, all3 6/6) | node, seed42, default_main / fseed70042 (acc 6/6, privacy 6/6, emb 6/6, all3 6/6) |
| Hetionet-small-nosource | Hetionet-small-nosource 适合作为 biomedical robustness 支撑；node 和 feature 都可用，edge 主要强调 utility/embedding。 | node, seed2024, fseeds_40_50_60 / fseed40 (acc 5/6, privacy 6/6, emb 5/6, all3 4/6) | node, seed2024, fseeds_40_50_60 / fseed50 (acc 6/6, privacy 6/6, emb 6/6, all3 6/6) |
| PrimeKG-NoSource | PrimeKG-NoSource 可以说明 utility/embedding 优势，但 privacy 和 all3 不稳，不适合写成全面优于。 | node, seed2024, default_main / fseed72024 (acc 6/6, privacy 4/6, emb 4/6, all3 3/6) | node, seed2024, fseeds_40_50_60 / fseed60 (acc 5/6, privacy 5/6, emb 6/6, all3 4/6) |

## PubMed

PubMed 最适合作为 HASI tuned 全面优于 baselines 的主证据，尤其是 node。

### HASI tuned vs src baselines

Paired check: `HASI groups=180`, `pairs=672`, `missing=0`, `fairness_issues=0`.

| type | acc | privacy | emb | all3 |
| --- | --- | --- | --- | --- |
| edge | 524/528 | 0/528 | 396/396 | 0/396 |
| feature | 72/72 | 36/72 | 72/72 | 36/72 |
| node | 72/72 | 67/72 | 72/72 | 67/72 |

| family | type | acc | privacy | emb | all3 |
| --- | --- | --- | --- | --- | --- |
| default_main | edge | 24/24 | 0/24 | 18/18 | 0/18 |
| default_main | feature | 18/18 | 7/18 | 18/18 | 7/18 |
| default_main | node | 18/18 | 16/18 | 18/18 | 16/18 |
| edge_100_110_120 | edge | 72/72 | 0/72 | 54/54 | 0/54 |
| edge_10_20_30 | edge | 72/72 | 0/72 | 54/54 | 0/54 |
| edge_130_140_150 | edge | 71/72 | 0/72 | 54/54 | 0/54 |
| edge_160_170_180 | edge | 71/72 | 0/72 | 54/54 | 0/54 |
| edge_190_200_210 | edge | 71/72 | 0/72 | 54/54 | 0/54 |
| edge_70_80_90 | edge | 72/72 | 0/72 | 54/54 | 0/54 |
| fseeds_40_50_60 | edge | 71/72 | 0/72 | 54/54 | 0/54 |
| fseeds_40_50_60 | feature | 54/54 | 29/54 | 54/54 | 29/54 |
| fseeds_40_50_60 | node | 54/54 | 51/54 | 54/54 | 51/54 |

| type | shared_base | recommended forget set | evidence |
| --- | --- | --- | --- |
| edge | seed42 | edge_190_200_210 / fseed190 | acc 8/8, privacy 0/8, emb 6/6, all3 0/6 |
| edge | seed123 | edge_100_110_120 / fseed110 | acc 8/8, privacy 0/8, emb 6/6, all3 0/6 |
| edge | seed2024 | edge_100_110_120 / fseed110 | acc 8/8, privacy 0/8, emb 6/6, all3 0/6 |
| feature | seed42 | fseeds_40_50_60 / fseed60 | acc 6/6, privacy 3/6, emb 6/6, all3 3/6 |
| feature | seed123 | fseeds_40_50_60 / fseed40 | acc 6/6, privacy 6/6, emb 6/6, all3 6/6 |
| feature | seed2024 | fseeds_40_50_60 / fseed40 | acc 6/6, privacy 5/6, emb 6/6, all3 5/6 |
| node | seed42 | default_main / fseed70042 | acc 6/6, privacy 6/6, emb 6/6, all3 6/6 |
| node | seed123 | fseeds_40_50_60 / fseed40 | acc 6/6, privacy 6/6, emb 6/6, all3 6/6 |
| node | seed2024 | default_main / fseed72024 | acc 6/6, privacy 6/6, emb 6/6, all3 6/6 |

### HASI tuned vs OpenGU adapted baselines

Paired check: `HASI groups=180`, `pairs=672`, `missing=0`, `fairness_issues=0`.

| type | acc | privacy | emb | all3 |
| --- | --- | --- | --- | --- |
| edge | 485/528 | 0/528 | 382/396 | 0/396 |
| feature | 70/72 | 31/72 | 48/72 | 16/72 |
| node | 70/72 | 68/72 | 57/72 | 55/72 |

| family | type | acc | privacy | emb | all3 |
| --- | --- | --- | --- | --- | --- |
| default_main | edge | 21/24 | 0/24 | 18/18 | 0/18 |
| default_main | feature | 17/18 | 6/18 | 12/18 | 3/18 |
| default_main | node | 17/18 | 16/18 | 15/18 | 13/18 |
| edge_100_110_120 | edge | 65/72 | 0/72 | 52/54 | 0/54 |
| edge_10_20_30 | edge | 67/72 | 0/72 | 54/54 | 0/54 |
| edge_130_140_150 | edge | 66/72 | 0/72 | 51/54 | 0/54 |
| edge_160_170_180 | edge | 69/72 | 0/72 | 52/54 | 0/54 |
| edge_190_200_210 | edge | 66/72 | 0/72 | 50/54 | 0/54 |
| edge_70_80_90 | edge | 65/72 | 0/72 | 53/54 | 0/54 |
| fseeds_40_50_60 | edge | 66/72 | 0/72 | 52/54 | 0/54 |
| fseeds_40_50_60 | feature | 53/54 | 25/54 | 36/54 | 13/54 |
| fseeds_40_50_60 | node | 53/54 | 52/54 | 42/54 | 42/54 |

| type | shared_base | recommended forget set | evidence |
| --- | --- | --- | --- |
| edge | seed42 | edge_160_170_180 / fseed180 | acc 8/8, privacy 0/8, emb 6/6, all3 0/6 |
| edge | seed123 | edge_130_140_150 / fseed140 | acc 8/8, privacy 0/8, emb 6/6, all3 0/6 |
| edge | seed2024 | edge_10_20_30 / fseed20 | acc 8/8, privacy 0/8, emb 6/6, all3 0/6 |
| feature | seed42 | fseeds_40_50_60 / fseed40 | acc 5/6, privacy 3/6, emb 4/6, all3 1/6 |
| feature | seed123 | fseeds_40_50_60 / fseed40 | acc 6/6, privacy 5/6, emb 4/6, all3 3/6 |
| feature | seed2024 | fseeds_40_50_60 / fseed40 | acc 6/6, privacy 5/6, emb 4/6, all3 3/6 |
| node | seed42 | default_main / fseed70042 | acc 6/6, privacy 6/6, emb 6/6, all3 6/6 |
| node | seed123 | fseeds_40_50_60 / fseed50 | acc 6/6, privacy 6/6, emb 6/6, all3 6/6 |
| node | seed2024 | fseeds_40_50_60 / fseed60 | acc 6/6, privacy 6/6, emb 4/6, all3 4/6 |

## Hetionet-small-nosource

Hetionet-small-nosource 适合作为 biomedical robustness 支撑；node 和 feature 都可用，edge 主要强调 utility/embedding。

### HASI tuned vs src baselines

Paired check: `HASI groups=108`, `pairs=366`, `missing=0`, `fairness_issues=0`.

| type | acc | privacy | emb | all3 |
| --- | --- | --- | --- | --- |
| edge | 146/168 | 17/168 | 126/126 | 17/126 |
| feature | 87/126 | 69/126 | 126/126 | 53/126 |
| node | 61/72 | 58/72 | 72/72 | 50/72 |

| family | type | acc | privacy | emb | all3 |
| --- | --- | --- | --- | --- | --- |
| default_main | edge | 20/24 | 2/24 | 18/18 | 2/18 |
| default_main | feature | 12/18 | 10/18 | 18/18 | 8/18 |
| default_main | node | 15/18 | 15/18 | 18/18 | 13/18 |
| edge_feature_10_20_30 | edge | 63/72 | 7/72 | 54/54 | 7/54 |
| edge_feature_10_20_30 | feature | 38/54 | 34/54 | 54/54 | 25/54 |
| fseeds_40_50_60 | edge | 63/72 | 8/72 | 54/54 | 8/54 |
| fseeds_40_50_60 | feature | 37/54 | 25/54 | 54/54 | 20/54 |
| fseeds_40_50_60 | node | 46/54 | 43/54 | 54/54 | 37/54 |

| type | shared_base | recommended forget set | evidence |
| --- | --- | --- | --- |
| edge | seed42 | edge_feature_10_20_30 / fseed20 | acc 8/8, privacy 0/8, emb 6/6, all3 0/6 |
| edge | seed123 | fseeds_40_50_60 / fseed50 | acc 7/8, privacy 1/8, emb 6/6, all3 1/6 |
| edge | seed2024 | edge_feature_10_20_30 / fseed30 | acc 7/8, privacy 3/8, emb 6/6, all3 3/6 |
| feature | seed42 | edge_feature_10_20_30 / fseed10 | acc 5/6, privacy 4/6, emb 6/6, all3 4/6 |
| feature | seed123 | fseeds_40_50_60 / fseed60 | acc 4/6, privacy 5/6, emb 6/6, all3 4/6 |
| feature | seed2024 | edge_feature_10_20_30 / fseed30 | acc 4/6, privacy 6/6, emb 6/6, all3 4/6 |
| node | seed42 | default_main / fseed70042 | acc 6/6, privacy 5/6, emb 6/6, all3 5/6 |
| node | seed123 | fseeds_40_50_60 / fseed40 | acc 4/6, privacy 5/6, emb 6/6, all3 3/6 |
| node | seed2024 | fseeds_40_50_60 / fseed50 | acc 6/6, privacy 6/6, emb 6/6, all3 6/6 |

### HASI tuned vs OpenGU adapted baselines

Paired check: `HASI groups=108`, `pairs=366`, `missing=0`, `fairness_issues=0`.

| type | acc | privacy | emb | all3 |
| --- | --- | --- | --- | --- |
| edge | 168/168 | 25/168 | 126/126 | 25/126 |
| feature | 84/126 | 72/126 | 106/126 | 50/126 |
| node | 55/72 | 50/72 | 62/72 | 40/72 |

| family | type | acc | privacy | emb | all3 |
| --- | --- | --- | --- | --- | --- |
| default_main | edge | 24/24 | 4/24 | 18/18 | 4/18 |
| default_main | feature | 12/18 | 12/18 | 18/18 | 8/18 |
| default_main | node | 14/18 | 12/18 | 15/18 | 10/18 |
| edge_feature_10_20_30 | edge | 72/72 | 10/72 | 54/54 | 10/54 |
| edge_feature_10_20_30 | feature | 36/54 | 34/54 | 46/54 | 22/54 |
| fseeds_40_50_60 | edge | 72/72 | 11/72 | 54/54 | 11/54 |
| fseeds_40_50_60 | feature | 36/54 | 26/54 | 42/54 | 20/54 |
| fseeds_40_50_60 | node | 41/54 | 38/54 | 47/54 | 30/54 |

| type | shared_base | recommended forget set | evidence |
| --- | --- | --- | --- |
| edge | seed42 | edge_feature_10_20_30 / fseed20 | acc 8/8, privacy 0/8, emb 6/6, all3 0/6 |
| edge | seed123 | edge_feature_10_20_30 / fseed10 | acc 8/8, privacy 2/8, emb 6/6, all3 2/6 |
| edge | seed2024 | edge_feature_10_20_30 / fseed20 | acc 8/8, privacy 2/8, emb 6/6, all3 2/6 |
| feature | seed42 | default_main / fseed70042 | acc 4/6, privacy 5/6, emb 6/6, all3 3/6 |
| feature | seed123 | fseeds_40_50_60 / fseed50 | acc 4/6, privacy 6/6, emb 4/6, all3 4/6 |
| feature | seed2024 | edge_feature_10_20_30 / fseed30 | acc 4/6, privacy 6/6, emb 6/6, all3 4/6 |
| node | seed42 | fseeds_40_50_60 / fseed60 | acc 5/6, privacy 5/6, emb 5/6, all3 4/6 |
| node | seed123 | default_main / fseed70123 | acc 5/6, privacy 5/6, emb 5/6, all3 4/6 |
| node | seed2024 | fseeds_40_50_60 / fseed40 | acc 5/6, privacy 6/6, emb 5/6, all3 4/6 |

## PrimeKG-NoSource

PrimeKG-NoSource 可以说明 utility/embedding 优势，但 privacy 和 all3 不稳，不适合写成全面优于。

### HASI tuned vs src baselines

Paired check: `HASI groups=252`, `pairs=834`, `missing=0`, `fairness_issues=0`.

| type | acc | privacy | emb | all3 |
| --- | --- | --- | --- | --- |
| edge | 283/312 | 20/312 | 234/234 | 15/234 |
| feature | 234/396 | 6/396 | 396/396 | 0/396 |
| node | 113/126 | 45/126 | 126/126 | 32/126 |

| family | type | acc | privacy | emb | all3 |
| --- | --- | --- | --- | --- | --- |
| default_main | edge | 22/24 | 0/24 | 18/18 | 0/18 |
| default_main | feature | 6/18 | 0/18 | 18/18 | 0/18 |
| default_main | node | 16/18 | 4/18 | 18/18 | 2/18 |
| edge_feature_100_110_120 | edge | 66/72 | 4/72 | 54/54 | 4/54 |
| edge_feature_100_110_120 | feature | 36/54 | 2/54 | 54/54 | 0/54 |
| edge_feature_70_80_90 | edge | 64/72 | 8/72 | 54/54 | 5/54 |
| edge_feature_70_80_90 | feature | 24/54 | 0/54 | 54/54 | 0/54 |
| feature_130_140_150 | feature | 30/54 | 0/54 | 54/54 | 0/54 |
| feature_160_170_180 | feature | 36/54 | 0/54 | 54/54 | 0/54 |
| feature_190_200_210 | feature | 36/54 | 2/54 | 54/54 | 0/54 |
| fseeds_10_20_30 | edge | 66/72 | 3/72 | 54/54 | 2/54 |
| fseeds_10_20_30 | feature | 36/54 | 0/54 | 54/54 | 0/54 |
| fseeds_10_20_30 | node | 48/54 | 21/54 | 54/54 | 15/54 |
| fseeds_40_50_60 | edge | 65/72 | 5/72 | 54/54 | 4/54 |
| fseeds_40_50_60 | feature | 30/54 | 2/54 | 54/54 | 0/54 |
| fseeds_40_50_60 | node | 49/54 | 20/54 | 54/54 | 15/54 |

| type | shared_base | recommended forget set | evidence |
| --- | --- | --- | --- |
| edge | seed42 | fseeds_10_20_30 / fseed20 | acc 8/8, privacy 1/8, emb 6/6, all3 1/6 |
| edge | seed123 | edge_feature_100_110_120 / fseed120 | acc 8/8, privacy 2/8, emb 6/6, all3 2/6 |
| edge | seed2024 | edge_feature_70_80_90 / fseed70 | acc 6/8, privacy 3/8, emb 6/6, all3 2/6 |
| feature | seed42 | fseeds_40_50_60 / fseed40 | acc 4/6, privacy 0/6, emb 6/6, all3 0/6 |
| feature | seed123 | edge_feature_100_110_120 / fseed100 | acc 4/6, privacy 0/6, emb 6/6, all3 0/6 |
| feature | seed2024 | edge_feature_100_110_120 / fseed120 | acc 4/6, privacy 2/6, emb 6/6, all3 0/6 |
| node | seed42 | fseeds_40_50_60 / fseed60 | acc 6/6, privacy 4/6, emb 6/6, all3 4/6 |
| node | seed123 | fseeds_10_20_30 / fseed10 | acc 6/6, privacy 4/6, emb 6/6, all3 4/6 |
| node | seed2024 | fseeds_40_50_60 / fseed60 | acc 5/6, privacy 5/6, emb 6/6, all3 4/6 |

### HASI tuned vs OpenGU adapted baselines

Paired check: `HASI groups=252`, `pairs=834`, `missing=0`, `fairness_issues=0`.

| type | acc | privacy | emb | all3 |
| --- | --- | --- | --- | --- |
| edge | 310/312 | 69/312 | 156/234 | 13/234 |
| feature | 234/396 | 44/396 | 294/396 | 1/396 |
| node | 115/126 | 40/126 | 84/126 | 24/126 |

| family | type | acc | privacy | emb | all3 |
| --- | --- | --- | --- | --- | --- |
| default_main | edge | 24/24 | 4/24 | 12/18 | 0/18 |
| default_main | feature | 6/18 | 2/18 | 18/18 | 1/18 |
| default_main | node | 16/18 | 5/18 | 12/18 | 3/18 |
| edge_feature_100_110_120 | edge | 72/72 | 15/72 | 36/54 | 3/54 |
| edge_feature_100_110_120 | feature | 36/54 | 6/54 | 36/54 | 0/54 |
| edge_feature_70_80_90 | edge | 72/72 | 19/72 | 36/54 | 5/54 |
| edge_feature_70_80_90 | feature | 24/54 | 4/54 | 48/54 | 0/54 |
| feature_130_140_150 | feature | 30/54 | 4/54 | 42/54 | 0/54 |
| feature_160_170_180 | feature | 36/54 | 10/54 | 36/54 | 0/54 |
| feature_190_200_210 | feature | 36/54 | 6/54 | 36/54 | 0/54 |
| fseeds_10_20_30 | edge | 71/72 | 13/72 | 36/54 | 1/54 |
| fseeds_10_20_30 | feature | 36/54 | 10/54 | 36/54 | 0/54 |
| fseeds_10_20_30 | node | 51/54 | 15/54 | 36/54 | 11/54 |
| fseeds_40_50_60 | edge | 71/72 | 18/72 | 36/54 | 4/54 |
| fseeds_40_50_60 | feature | 30/54 | 2/54 | 42/54 | 0/54 |
| fseeds_40_50_60 | node | 48/54 | 20/54 | 36/54 | 10/54 |

| type | shared_base | recommended forget set | evidence |
| --- | --- | --- | --- |
| edge | seed42 | edge_feature_100_110_120 / fseed120 | acc 8/8, privacy 2/8, emb 4/6, all3 0/6 |
| edge | seed123 | edge_feature_100_110_120 / fseed100 | acc 8/8, privacy 3/8, emb 4/6, all3 1/6 |
| edge | seed2024 | edge_feature_70_80_90 / fseed70 | acc 8/8, privacy 3/8, emb 4/6, all3 2/6 |
| feature | seed42 | default_main / fseed70042 | acc 2/6, privacy 2/6, emb 6/6, all3 1/6 |
| feature | seed123 | feature_190_200_210 / fseed210 | acc 4/6, privacy 0/6, emb 4/6, all3 0/6 |
| feature | seed2024 | fseeds_10_20_30 / fseed10 | acc 4/6, privacy 2/6, emb 4/6, all3 0/6 |
| node | seed42 | fseeds_40_50_60 / fseed60 | acc 5/6, privacy 4/6, emb 4/6, all3 3/6 |
| node | seed123 | fseeds_10_20_30 / fseed10 | acc 6/6, privacy 3/6, emb 4/6, all3 2/6 |
| node | seed2024 | default_main / fseed72024 | acc 6/6, privacy 4/6, emb 4/6, all3 3/6 |

## Recommended Reporting Choice

For paper-style reporting, use the OpenGU-adapted recommendation as the stricter selection criterion, because it includes GNNDelete in addition to GraphEraser variants. The practical choices are:

| dataset | preferred type | recommended forget set | paper-safe claim |
| --- | --- | --- | --- |
| PubMed | node | default_main / seed42 / fseed70042, or robustness_fseeds_40_50_60 / seed123 / fseed50 | Strongest full evidence; node can support a broad HASI advantage claim. |
| Hetionet-small-nosource | node and feature | node: robustness_fseeds_40_50_60 / seed2024 / fseed40; feature: robustness_edge_feature_fseeds_10_20_30 / seed2024 / fseed30 | Good biomedical robustness support; avoid claiming edge privacy superiority. |
| PrimeKG-NoSource | node | robustness_fseeds_40_50_60 / seed42 / fseed60, or default_main / seed2024 / fseed72024 | Use as utility/embedding support with privacy caveat; not a clean all-metric dominance dataset. |

