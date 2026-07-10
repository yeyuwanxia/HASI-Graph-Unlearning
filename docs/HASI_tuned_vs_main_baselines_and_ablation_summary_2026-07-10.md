# HASI Tuned vs Main Baselines and Ablations

Date: 2026-07-10

This document fixes the paper-facing baseline scope to:

- `HASI tuned` as the compared method.
- OpenGU-adapted core baselines: `opengu-gnndelete`, `opengu-grapheraser-bekm`, `opengu-grapheraser-blpa`, and edge-only `opengu-gif`.
- Repository-native `retrain` from `src` as the retraining reference.

MEGU is intentionally not mixed into the main baseline table. It should stay as a separate strong-baseline appendix because it changes the story, especially on representation drift.

A win means higher `accuracy_after`, higher `privacy_score`, and lower `embedding_l2_mean`. `all3` means a paired comparison wins on all available accuracy, privacy, and embedding metrics together.

## Fairness Check
| dataset | HASI groups | baseline records used | pairs | missing among used | extra unmatched baselines excluded | fairness issues |
| --- | --- | --- | --- | --- | --- | --- |
| PubMed | 180 | 852 | 852 | 0 | 180 | 0 |
| Hetionet-small-nosource | 108 | 474 | 474 | 0 | 0 | 0 |
| PrimeKG-NoSource | 252 | 1086 | 1086 | 0 | 0 | 0 |

All paired rows above are checked by `forget_set.path`, forget target hash, and shared-base path. `missing among used=0` means the paper-facing comparison set is complete after excluding baseline-only extra runs. `fairness issues=0` means the main-table comparisons are paired under the same shared-base and forget-set protocol.

## Overall Main-Baseline Results
| dataset | type | acc | privacy | emb | all3 | d_acc | d_priv | d_emb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PubMed | edge | 613/660 | 0/660 | 514/528 | 0/528 | +0.1570 | -0.1244 | -0.1097 |
| PubMed | feature | 94/96 | 51/96 | 72/96 | 36/96 | +0.0302 | +0.0246 | -0.2112 |
| PubMed | node | 94/96 | 88/96 | 81/96 | 75/96 | +0.0289 | +0.0859 | -0.2254 |
| Hetionet-small-nosource | edge | 188/210 | 26/210 | 168/168 | 26/168 | +0.2459 | -0.0815 | -6.1101 |
| Hetionet-small-nosource | feature | 87/168 | 90/168 | 148/168 | 52/168 | +0.1173 | +0.0070 | -4.0324 |
| Hetionet-small-nosource | node | 68/96 | 71/96 | 86/96 | 53/96 | +0.1521 | +0.0628 | -4.4093 |
| PrimeKG-NoSource | edge | 359/390 | 76/390 | 234/312 | 15/312 | +0.1507 | -0.0394 | -0.6633 |
| PrimeKG-NoSource | feature | 234/528 | 50/528 | 426/528 | 1/528 | +0.0229 | -0.1030 | -0.4236 |
| PrimeKG-NoSource | node | 144/168 | 63/168 | 126/168 | 34/168 | +0.0512 | -0.0214 | -0.5895 |

Interpretation: PubMed node is still the cleanest main evidence; Hetionet node/feature are strong biomedical robustness evidence; PrimeKG-NoSource mainly supports utility and representation, while privacy remains mixed.

## PubMed

Paired check: `HASI groups=180`, `baseline records used=852`, `pairs=852`, `missing among used=0`, `extra unmatched baselines excluded=180`, `fairness_issues=0`.

### Baseline Coverage
| type | baseline | paired runs |
| --- | --- | --- |
| edge | opengu-gif | 132 |
| edge | opengu-gnndelete | 132 |
| edge | opengu-grapheraser-bekm | 132 |
| edge | opengu-grapheraser-blpa | 132 |
| edge | retrain | 132 |
| feature | opengu-gnndelete | 24 |
| feature | opengu-grapheraser-bekm | 24 |
| feature | opengu-grapheraser-blpa | 24 |
| feature | retrain | 24 |
| node | opengu-gnndelete | 24 |
| node | opengu-grapheraser-bekm | 24 |
| node | opengu-grapheraser-blpa | 24 |
| node | retrain | 24 |

### Summary by Type
| type | acc | privacy | emb | all3 | d_acc | d_priv | d_emb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| edge | 613/660 | 0/660 | 514/528 | 0/528 | +0.1570 | -0.1244 | -0.1097 |
| feature | 94/96 | 51/96 | 72/96 | 36/96 | +0.0302 | +0.0246 | -0.2112 |
| node | 94/96 | 88/96 | 81/96 | 75/96 | +0.0289 | +0.0859 | -0.2254 |

### Summary by Family and Type
| family | type | acc | privacy | emb | all3 | d_acc | d_priv | d_emb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default_main | edge | 27/30 | 0/30 | 24/24 | 0/24 | +0.1562 | -0.1243 | -0.1134 |
| default_main | feature | 23/24 | 11/24 | 18/24 | 8/24 | +0.0285 | -0.0312 | -0.2075 |
| default_main | node | 23/24 | 21/24 | 21/24 | 18/24 | +0.0283 | +0.0839 | -0.2255 |
| edge_fseeds_100_110_120 | edge | 83/90 | 0/90 | 70/72 | 0/72 | +0.1570 | -0.1206 | -0.1109 |
| edge_fseeds_10_20_30 | edge | 85/90 | 0/90 | 72/72 | 0/72 | +0.1564 | -0.1266 | -0.1091 |
| edge_fseeds_130_140_150 | edge | 83/90 | 0/90 | 69/72 | 0/72 | +0.1563 | -0.1227 | -0.1083 |
| edge_fseeds_160_170_180 | edge | 86/90 | 0/90 | 70/72 | 0/72 | +0.1571 | -0.1232 | -0.1083 |
| edge_fseeds_190_200_210 | edge | 83/90 | 0/90 | 68/72 | 0/72 | +0.1577 | -0.1253 | -0.1089 |
| edge_fseeds_70_80_90 | edge | 83/90 | 0/90 | 71/72 | 0/72 | +0.1575 | -0.1298 | -0.1088 |
| fseeds_40_50_60 | edge | 83/90 | 0/90 | 70/72 | 0/72 | +0.1572 | -0.1226 | -0.1119 |
| fseeds_40_50_60 | feature | 71/72 | 40/72 | 54/72 | 28/72 | +0.0307 | +0.0433 | -0.2124 |
| fseeds_40_50_60 | node | 71/72 | 67/72 | 60/72 | 57/72 | +0.0290 | +0.0866 | -0.2254 |

### Recommended Forget Set per Shared Base
| type | shared_base | recommended forget set | evidence |
| --- | --- | --- | --- |
| edge | seed42 | edge_fseeds_160_170_180 / fseed180 | acc 10/10, privacy 0/10, emb 8/8, all3 0/8 |
| edge | seed123 | default_main / fseed70123 | acc 10/10, privacy 0/10, emb 8/8, all3 0/8 |
| edge | seed2024 | edge_fseeds_190_200_210 / fseed190 | acc 10/10, privacy 0/10, emb 8/8, all3 0/8 |
| feature | seed42 | fseeds_40_50_60 / fseed60 | acc 8/8, privacy 4/8, emb 6/8, all3 3/8 |
| feature | seed123 | fseeds_40_50_60 / fseed40 | acc 8/8, privacy 7/8, emb 6/8, all3 5/8 |
| feature | seed2024 | fseeds_40_50_60 / fseed40 | acc 8/8, privacy 7/8, emb 6/8, all3 5/8 |
| node | seed42 | default_main / fseed70042 | acc 8/8, privacy 8/8, emb 8/8, all3 8/8 |
| node | seed123 | fseeds_40_50_60 / fseed50 | acc 8/8, privacy 8/8, emb 8/8, all3 8/8 |
| node | seed2024 | fseeds_40_50_60 / fseed40 | acc 8/8, privacy 8/8, emb 6/8, all3 6/8 |

### All Seed Details
| type | family | shared_base | forget_seed | acc | privacy | emb | all3 | d_acc | d_priv | d_emb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edge | default_main | seed42 | fseed70042 | 8/10 | 0/10 | 8/8 | 0/8 | +0.1470 | -0.1132 | -0.1124 |
| edge | default_main | seed123 | fseed70123 | 10/10 | 0/10 | 8/8 | 0/8 | +0.1614 | -0.1314 | -0.1189 |
| edge | default_main | seed2024 | fseed72024 | 9/10 | 0/10 | 8/8 | 0/8 | +0.1603 | -0.1282 | -0.1089 |
| edge | edge_fseeds_100_110_120 | seed42 | fseed100 | 8/10 | 0/10 | 8/8 | 0/8 | +0.1475 | -0.1158 | -0.1153 |
| edge | edge_fseeds_100_110_120 | seed42 | fseed110 | 9/10 | 0/10 | 8/8 | 0/8 | +0.1466 | -0.1095 | -0.1134 |
| edge | edge_fseeds_100_110_120 | seed42 | fseed120 | 8/10 | 0/10 | 8/8 | 0/8 | +0.1487 | -0.1220 | -0.1101 |
| edge | edge_fseeds_100_110_120 | seed123 | fseed100 | 10/10 | 0/10 | 7/8 | 0/8 | +0.1599 | -0.1394 | -0.1082 |
| edge | edge_fseeds_100_110_120 | seed123 | fseed110 | 9/10 | 0/10 | 8/8 | 0/8 | +0.1584 | -0.0985 | -0.1058 |
| edge | edge_fseeds_100_110_120 | seed123 | fseed120 | 9/10 | 0/10 | 8/8 | 0/8 | +0.1591 | -0.1222 | -0.1094 |
| edge | edge_fseeds_100_110_120 | seed2024 | fseed100 | 10/10 | 0/10 | 8/8 | 0/8 | +0.1648 | -0.1315 | -0.1147 |
| edge | edge_fseeds_100_110_120 | seed2024 | fseed110 | 10/10 | 0/10 | 7/8 | 0/8 | +0.1627 | -0.1173 | -0.1095 |
| edge | edge_fseeds_100_110_120 | seed2024 | fseed120 | 10/10 | 0/10 | 8/8 | 0/8 | +0.1650 | -0.1294 | -0.1121 |
| edge | edge_fseeds_10_20_30 | seed42 | fseed10 | 9/10 | 0/10 | 8/8 | 0/8 | +0.1497 | -0.1176 | -0.1191 |
| edge | edge_fseeds_10_20_30 | seed42 | fseed20 | 9/10 | 0/10 | 8/8 | 0/8 | +0.1482 | -0.1198 | -0.1124 |
| edge | edge_fseeds_10_20_30 | seed42 | fseed30 | 9/10 | 0/10 | 8/8 | 0/8 | +0.1475 | -0.1239 | -0.1172 |
| edge | edge_fseeds_10_20_30 | seed123 | fseed10 | 10/10 | 0/10 | 8/8 | 0/8 | +0.1581 | -0.1299 | -0.1087 |
| edge | edge_fseeds_10_20_30 | seed123 | fseed20 | 9/10 | 0/10 | 8/8 | 0/8 | +0.1598 | -0.1277 | -0.1050 |
| edge | edge_fseeds_10_20_30 | seed123 | fseed30 | 10/10 | 0/10 | 8/8 | 0/8 | +0.1598 | -0.1309 | -0.1069 |
| edge | edge_fseeds_10_20_30 | seed2024 | fseed10 | 9/10 | 0/10 | 8/8 | 0/8 | +0.1606 | -0.1383 | -0.1121 |
| edge | edge_fseeds_10_20_30 | seed2024 | fseed20 | 10/10 | 0/10 | 8/8 | 0/8 | +0.1625 | -0.1246 | -0.1006 |
| edge | edge_fseeds_10_20_30 | seed2024 | fseed30 | 10/10 | 0/10 | 8/8 | 0/8 | +0.1614 | -0.1270 | -0.1004 |
| edge | edge_fseeds_130_140_150 | seed42 | fseed130 | 7/10 | 0/10 | 8/8 | 0/8 | +0.1426 | -0.1190 | -0.1062 |
| edge | edge_fseeds_130_140_150 | seed42 | fseed140 | 8/10 | 0/10 | 8/8 | 0/8 | +0.1480 | -0.1157 | -0.1102 |
| edge | edge_fseeds_130_140_150 | seed42 | fseed150 | 8/10 | 0/10 | 8/8 | 0/8 | +0.1448 | -0.1138 | -0.1120 |
| edge | edge_fseeds_130_140_150 | seed123 | fseed130 | 10/10 | 0/10 | 7/8 | 0/8 | +0.1605 | -0.1113 | -0.1043 |
| edge | edge_fseeds_130_140_150 | seed123 | fseed140 | 10/10 | 0/10 | 8/8 | 0/8 | +0.1593 | -0.1224 | -0.1106 |
| edge | edge_fseeds_130_140_150 | seed123 | fseed150 | 10/10 | 0/10 | 7/8 | 0/8 | +0.1591 | -0.1168 | -0.1001 |
| edge | edge_fseeds_130_140_150 | seed2024 | fseed130 | 10/10 | 0/10 | 8/8 | 0/8 | +0.1631 | -0.1412 | -0.1125 |
| edge | edge_fseeds_130_140_150 | seed2024 | fseed140 | 10/10 | 0/10 | 8/8 | 0/8 | +0.1656 | -0.1319 | -0.1087 |
| edge | edge_fseeds_130_140_150 | seed2024 | fseed150 | 10/10 | 0/10 | 7/8 | 0/8 | +0.1637 | -0.1317 | -0.1102 |
| edge | edge_fseeds_160_170_180 | seed42 | fseed160 | 8/10 | 0/10 | 8/8 | 0/8 | +0.1453 | -0.1093 | -0.1039 |
| edge | edge_fseeds_160_170_180 | seed42 | fseed170 | 9/10 | 0/10 | 8/8 | 0/8 | +0.1451 | -0.1144 | -0.1091 |
| edge | edge_fseeds_160_170_180 | seed42 | fseed180 | 10/10 | 0/10 | 8/8 | 0/8 | +0.1500 | -0.1290 | -0.1190 |
| edge | edge_fseeds_160_170_180 | seed123 | fseed160 | 10/10 | 0/10 | 8/8 | 0/8 | +0.1611 | -0.1350 | -0.1054 |
| edge | edge_fseeds_160_170_180 | seed123 | fseed170 | 10/10 | 0/10 | 7/8 | 0/8 | +0.1630 | -0.1155 | -0.1043 |
| edge | edge_fseeds_160_170_180 | seed123 | fseed180 | 10/10 | 0/10 | 8/8 | 0/8 | +0.1595 | -0.1266 | -0.1115 |
| edge | edge_fseeds_160_170_180 | seed2024 | fseed160 | 9/10 | 0/10 | 7/8 | 0/8 | +0.1618 | -0.1261 | -0.1021 |
| edge | edge_fseeds_160_170_180 | seed2024 | fseed170 | 10/10 | 0/10 | 8/8 | 0/8 | +0.1640 | -0.1253 | -0.1035 |
| edge | edge_fseeds_160_170_180 | seed2024 | fseed180 | 10/10 | 0/10 | 8/8 | 0/8 | +0.1641 | -0.1277 | -0.1158 |
| edge | edge_fseeds_190_200_210 | seed42 | fseed190 | 9/10 | 0/10 | 8/8 | 0/8 | +0.1489 | -0.1096 | -0.1089 |
| edge | edge_fseeds_190_200_210 | seed42 | fseed200 | 8/10 | 0/10 | 8/8 | 0/8 | +0.1464 | -0.1166 | -0.1149 |
| edge | edge_fseeds_190_200_210 | seed42 | fseed210 | 7/10 | 0/10 | 8/8 | 0/8 | +0.1444 | -0.1191 | -0.1036 |
| edge | edge_fseeds_190_200_210 | seed123 | fseed190 | 10/10 | 0/10 | 6/8 | 0/8 | +0.1583 | -0.1224 | -0.1093 |
| edge | edge_fseeds_190_200_210 | seed123 | fseed200 | 10/10 | 0/10 | 7/8 | 0/8 | +0.1617 | -0.1293 | -0.1021 |
| edge | edge_fseeds_190_200_210 | seed123 | fseed210 | 9/10 | 0/10 | 7/8 | 0/8 | +0.1617 | -0.1419 | -0.1013 |
| edge | edge_fseeds_190_200_210 | seed2024 | fseed190 | 10/10 | 0/10 | 8/8 | 0/8 | +0.1666 | -0.1269 | -0.1151 |
| edge | edge_fseeds_190_200_210 | seed2024 | fseed200 | 10/10 | 0/10 | 8/8 | 0/8 | +0.1655 | -0.1303 | -0.1147 |
| edge | edge_fseeds_190_200_210 | seed2024 | fseed210 | 10/10 | 0/10 | 8/8 | 0/8 | +0.1658 | -0.1320 | -0.1098 |
| edge | edge_fseeds_70_80_90 | seed42 | fseed70 | 9/10 | 0/10 | 8/8 | 0/8 | +0.1486 | -0.1239 | -0.1123 |
| edge | edge_fseeds_70_80_90 | seed42 | fseed80 | 9/10 | 0/10 | 8/8 | 0/8 | +0.1470 | -0.1094 | -0.1099 |
| edge | edge_fseeds_70_80_90 | seed42 | fseed90 | 9/10 | 0/10 | 8/8 | 0/8 | +0.1468 | -0.1330 | -0.1049 |
| edge | edge_fseeds_70_80_90 | seed123 | fseed70 | 9/10 | 0/10 | 8/8 | 0/8 | +0.1590 | -0.1395 | -0.1114 |
| edge | edge_fseeds_70_80_90 | seed123 | fseed80 | 10/10 | 0/10 | 8/8 | 0/8 | +0.1612 | -0.1428 | -0.1083 |
| edge | edge_fseeds_70_80_90 | seed123 | fseed90 | 10/10 | 0/10 | 7/8 | 0/8 | +0.1593 | -0.1226 | -0.1028 |
| edge | edge_fseeds_70_80_90 | seed2024 | fseed70 | 9/10 | 0/10 | 8/8 | 0/8 | +0.1654 | -0.1350 | -0.1082 |
| edge | edge_fseeds_70_80_90 | seed2024 | fseed80 | 9/10 | 0/10 | 8/8 | 0/8 | +0.1662 | -0.1381 | -0.1093 |
| edge | edge_fseeds_70_80_90 | seed2024 | fseed90 | 9/10 | 0/10 | 8/8 | 0/8 | +0.1639 | -0.1242 | -0.1118 |
| edge | fseeds_40_50_60 | seed42 | fseed40 | 9/10 | 0/10 | 8/8 | 0/8 | +0.1471 | -0.1129 | -0.1173 |
| edge | fseeds_40_50_60 | seed42 | fseed50 | 8/10 | 0/10 | 8/8 | 0/8 | +0.1480 | -0.1145 | -0.1156 |
| edge | fseeds_40_50_60 | seed42 | fseed60 | 7/10 | 0/10 | 8/8 | 0/8 | +0.1452 | -0.1157 | -0.1105 |
| edge | fseeds_40_50_60 | seed123 | fseed40 | 10/10 | 0/10 | 8/8 | 0/8 | +0.1614 | -0.1310 | -0.1079 |
| edge | fseeds_40_50_60 | seed123 | fseed50 | 10/10 | 0/10 | 7/8 | 0/8 | +0.1596 | -0.1212 | -0.1051 |
| edge | fseeds_40_50_60 | seed123 | fseed60 | 9/10 | 0/10 | 8/8 | 0/8 | +0.1575 | -0.1202 | -0.1110 |
| edge | fseeds_40_50_60 | seed2024 | fseed40 | 10/10 | 0/10 | 8/8 | 0/8 | +0.1640 | -0.1307 | -0.1155 |
| edge | fseeds_40_50_60 | seed2024 | fseed50 | 10/10 | 0/10 | 7/8 | 0/8 | +0.1671 | -0.1269 | -0.1113 |
| edge | fseeds_40_50_60 | seed2024 | fseed60 | 10/10 | 0/10 | 8/8 | 0/8 | +0.1645 | -0.1300 | -0.1133 |
| feature | default_main | seed42 | fseed70042 | 8/8 | 3/8 | 6/8 | 2/8 | +0.0227 | -0.0702 | -0.2072 |
| feature | default_main | seed123 | fseed70123 | 8/8 | 5/8 | 6/8 | 3/8 | +0.0334 | +0.0159 | -0.2052 |
| feature | default_main | seed2024 | fseed72024 | 7/8 | 3/8 | 6/8 | 3/8 | +0.0294 | -0.0392 | -0.2101 |
| feature | fseeds_40_50_60 | seed42 | fseed40 | 7/8 | 4/8 | 6/8 | 2/8 | +0.0232 | +0.0387 | -0.2021 |
| feature | fseeds_40_50_60 | seed42 | fseed50 | 8/8 | 3/8 | 6/8 | 2/8 | +0.0235 | -0.0324 | -0.2179 |
| feature | fseeds_40_50_60 | seed42 | fseed60 | 8/8 | 4/8 | 6/8 | 3/8 | +0.0227 | +0.0165 | -0.2135 |
| feature | fseeds_40_50_60 | seed123 | fseed40 | 8/8 | 7/8 | 6/8 | 5/8 | +0.0373 | +0.2457 | -0.2215 |
| feature | fseeds_40_50_60 | seed123 | fseed50 | 8/8 | 5/8 | 6/8 | 3/8 | +0.0360 | +0.0279 | -0.2243 |
| feature | fseeds_40_50_60 | seed123 | fseed60 | 8/8 | 6/8 | 6/8 | 4/8 | +0.0308 | +0.0478 | -0.1985 |
| feature | fseeds_40_50_60 | seed2024 | fseed40 | 8/8 | 7/8 | 6/8 | 5/8 | +0.0354 | +0.1434 | -0.2080 |
| feature | fseeds_40_50_60 | seed2024 | fseed50 | 8/8 | 2/8 | 6/8 | 2/8 | +0.0357 | -0.0523 | -0.2161 |
| feature | fseeds_40_50_60 | seed2024 | fseed60 | 8/8 | 2/8 | 6/8 | 2/8 | +0.0320 | -0.0461 | -0.2095 |
| node | default_main | seed42 | fseed70042 | 8/8 | 8/8 | 8/8 | 8/8 | +0.0214 | +0.1032 | -0.2326 |
| node | default_main | seed123 | fseed70123 | 8/8 | 5/8 | 6/8 | 4/8 | +0.0319 | +0.0237 | -0.2178 |
| node | default_main | seed2024 | fseed72024 | 7/8 | 8/8 | 7/8 | 6/8 | +0.0315 | +0.1247 | -0.2261 |
| node | fseeds_40_50_60 | seed42 | fseed40 | 8/8 | 8/8 | 6/8 | 6/8 | +0.0224 | +0.0904 | -0.2179 |
| node | fseeds_40_50_60 | seed42 | fseed50 | 8/8 | 7/8 | 8/8 | 7/8 | +0.0227 | +0.0775 | -0.2309 |
| node | fseeds_40_50_60 | seed42 | fseed60 | 8/8 | 7/8 | 7/8 | 6/8 | +0.0228 | +0.0672 | -0.2265 |
| node | fseeds_40_50_60 | seed123 | fseed40 | 8/8 | 7/8 | 6/8 | 6/8 | +0.0311 | +0.1040 | -0.2169 |
| node | fseeds_40_50_60 | seed123 | fseed50 | 8/8 | 8/8 | 8/8 | 8/8 | +0.0313 | +0.0798 | -0.2468 |
| node | fseeds_40_50_60 | seed123 | fseed60 | 8/8 | 8/8 | 7/8 | 7/8 | +0.0333 | +0.1042 | -0.2355 |
| node | fseeds_40_50_60 | seed2024 | fseed40 | 8/8 | 8/8 | 6/8 | 6/8 | +0.0334 | +0.0949 | -0.2183 |
| node | fseeds_40_50_60 | seed2024 | fseed50 | 7/8 | 6/8 | 6/8 | 5/8 | +0.0322 | +0.0731 | -0.2146 |
| node | fseeds_40_50_60 | seed2024 | fseed60 | 8/8 | 8/8 | 6/8 | 6/8 | +0.0323 | +0.0886 | -0.2208 |

## Hetionet-small-nosource

Paired check: `HASI groups=108`, `baseline records used=474`, `pairs=474`, `missing among used=0`, `extra unmatched baselines excluded=0`, `fairness_issues=0`.

### Baseline Coverage
| type | baseline | paired runs |
| --- | --- | --- |
| edge | opengu-gif | 42 |
| edge | opengu-gnndelete | 42 |
| edge | opengu-grapheraser-bekm | 42 |
| edge | opengu-grapheraser-blpa | 42 |
| edge | retrain | 42 |
| feature | opengu-gnndelete | 42 |
| feature | opengu-grapheraser-bekm | 42 |
| feature | opengu-grapheraser-blpa | 42 |
| feature | retrain | 42 |
| node | opengu-gnndelete | 24 |
| node | opengu-grapheraser-bekm | 24 |
| node | opengu-grapheraser-blpa | 24 |
| node | retrain | 24 |

### Summary by Type
| type | acc | privacy | emb | all3 | d_acc | d_priv | d_emb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| edge | 188/210 | 26/210 | 168/168 | 26/168 | +0.2459 | -0.0815 | -6.1101 |
| feature | 87/168 | 90/168 | 148/168 | 52/168 | +0.1173 | +0.0070 | -4.0324 |
| node | 68/96 | 71/96 | 86/96 | 53/96 | +0.1521 | +0.0628 | -4.4093 |

### Summary by Family and Type
| family | type | acc | privacy | emb | all3 | d_acc | d_priv | d_emb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default_main | edge | 26/30 | 4/30 | 24/24 | 4/24 | +0.2480 | -0.0899 | -6.0478 |
| default_main | feature | 12/24 | 14/24 | 24/24 | 8/24 | +0.1392 | +0.0124 | -5.1473 |
| default_main | node | 17/24 | 17/24 | 21/24 | 13/24 | +0.1501 | +0.0648 | -4.5481 |
| edge_feature_fseeds_10_20_30 | edge | 81/90 | 11/90 | 72/72 | 11/72 | +0.2454 | -0.0755 | -6.1294 |
| edge_feature_fseeds_10_20_30 | feature | 38/72 | 45/72 | 64/72 | 24/72 | +0.1312 | +0.0513 | -4.1132 |
| fseeds_40_50_60 | edge | 81/90 | 11/90 | 72/72 | 11/72 | +0.2457 | -0.0846 | -6.1115 |
| fseeds_40_50_60 | feature | 37/72 | 31/72 | 60/72 | 20/72 | +0.0962 | -0.0390 | -3.5801 |
| fseeds_40_50_60 | node | 51/72 | 54/72 | 65/72 | 40/72 | +0.1527 | +0.0621 | -4.3631 |

### Recommended Forget Set per Shared Base
| type | shared_base | recommended forget set | evidence |
| --- | --- | --- | --- |
| edge | seed42 | default_main / fseed70042 | acc 10/10, privacy 0/10, emb 8/8, all3 0/8 |
| edge | seed123 | fseeds_40_50_60 / fseed50 | acc 9/10, privacy 2/10, emb 8/8, all3 2/8 |
| edge | seed2024 | edge_feature_fseeds_10_20_30 / fseed30 | acc 9/10, privacy 3/10, emb 8/8, all3 3/8 |
| feature | seed42 | edge_feature_fseeds_10_20_30 / fseed30 | acc 5/8, privacy 5/8, emb 8/8, all3 3/8 |
| feature | seed123 | fseeds_40_50_60 / fseed50 | acc 4/8, privacy 6/8, emb 6/8, all3 4/8 |
| feature | seed2024 | edge_feature_fseeds_10_20_30 / fseed30 | acc 4/8, privacy 8/8, emb 8/8, all3 4/8 |
| node | seed42 | fseeds_40_50_60 / fseed60 | acc 7/8, privacy 7/8, emb 7/8, all3 6/8 |
| node | seed123 | fseeds_40_50_60 / fseed50 | acc 5/8, privacy 7/8, emb 7/8, all3 4/8 |
| node | seed2024 | fseeds_40_50_60 / fseed50 | acc 7/8, privacy 8/8, emb 6/8, all3 6/8 |

### All Seed Details
| type | family | shared_base | forget_seed | acc | privacy | emb | all3 | d_acc | d_priv | d_emb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edge | default_main | seed42 | fseed70042 | 10/10 | 0/10 | 8/8 | 0/8 | +0.2484 | -0.1422 | -6.1144 |
| edge | default_main | seed123 | fseed70123 | 8/10 | 2/10 | 8/8 | 2/8 | +0.2455 | -0.0828 | -6.4655 |
| edge | default_main | seed2024 | fseed72024 | 8/10 | 2/10 | 8/8 | 2/8 | +0.2500 | -0.0447 | -5.5636 |
| edge | edge_feature_fseeds_10_20_30 | seed42 | fseed10 | 10/10 | 0/10 | 8/8 | 0/8 | +0.2472 | -0.1441 | -6.2529 |
| edge | edge_feature_fseeds_10_20_30 | seed42 | fseed20 | 10/10 | 0/10 | 8/8 | 0/8 | +0.2369 | -0.1315 | -6.1763 |
| edge | edge_feature_fseeds_10_20_30 | seed42 | fseed30 | 10/10 | 0/10 | 8/8 | 0/8 | +0.2448 | -0.1399 | -6.2900 |
| edge | edge_feature_fseeds_10_20_30 | seed123 | fseed10 | 8/10 | 2/10 | 8/8 | 2/8 | +0.2477 | -0.0446 | -6.5589 |
| edge | edge_feature_fseeds_10_20_30 | seed123 | fseed20 | 8/10 | 1/10 | 8/8 | 1/8 | +0.2466 | -0.0493 | -6.5521 |
| edge | edge_feature_fseeds_10_20_30 | seed123 | fseed30 | 8/10 | 1/10 | 8/8 | 1/8 | +0.2472 | -0.0499 | -6.5171 |
| edge | edge_feature_fseeds_10_20_30 | seed2024 | fseed10 | 10/10 | 2/10 | 8/8 | 2/8 | +0.2498 | -0.0471 | -5.6640 |
| edge | edge_feature_fseeds_10_20_30 | seed2024 | fseed20 | 8/10 | 2/10 | 8/8 | 2/8 | +0.2461 | -0.0329 | -5.5509 |
| edge | edge_feature_fseeds_10_20_30 | seed2024 | fseed30 | 9/10 | 3/10 | 8/8 | 3/8 | +0.2428 | -0.0401 | -5.6025 |
| edge | fseeds_40_50_60 | seed42 | fseed40 | 10/10 | 0/10 | 8/8 | 0/8 | +0.2405 | -0.1479 | -6.2171 |
| edge | fseeds_40_50_60 | seed42 | fseed50 | 10/10 | 0/10 | 8/8 | 0/8 | +0.2406 | -0.1481 | -6.2227 |
| edge | fseeds_40_50_60 | seed42 | fseed60 | 10/10 | 0/10 | 8/8 | 0/8 | +0.2371 | -0.1560 | -6.2105 |
| edge | fseeds_40_50_60 | seed123 | fseed40 | 8/10 | 1/10 | 8/8 | 1/8 | +0.2469 | -0.0648 | -6.5494 |
| edge | fseeds_40_50_60 | seed123 | fseed50 | 9/10 | 2/10 | 8/8 | 2/8 | +0.2457 | -0.0579 | -6.5572 |
| edge | fseeds_40_50_60 | seed123 | fseed60 | 8/10 | 2/10 | 8/8 | 2/8 | +0.2500 | -0.0549 | -6.5366 |
| edge | fseeds_40_50_60 | seed2024 | fseed40 | 8/10 | 2/10 | 8/8 | 2/8 | +0.2542 | -0.0376 | -5.5669 |
| edge | fseeds_40_50_60 | seed2024 | fseed50 | 10/10 | 2/10 | 8/8 | 2/8 | +0.2489 | -0.0518 | -5.5780 |
| edge | fseeds_40_50_60 | seed2024 | fseed60 | 8/10 | 2/10 | 8/8 | 2/8 | +0.2475 | -0.0427 | -5.5651 |
| feature | default_main | seed42 | fseed70042 | 4/8 | 7/8 | 8/8 | 3/8 | +0.1373 | +0.0866 | -6.0649 |
| feature | default_main | seed123 | fseed70123 | 4/8 | 1/8 | 8/8 | 1/8 | +0.1448 | -0.1483 | -4.8362 |
| feature | default_main | seed2024 | fseed72024 | 4/8 | 6/8 | 8/8 | 4/8 | +0.1355 | +0.0989 | -4.5408 |
| feature | edge_feature_fseeds_10_20_30 | seed42 | fseed10 | 5/8 | 4/8 | 7/8 | 3/8 | +0.1295 | -0.0036 | -4.5133 |
| feature | edge_feature_fseeds_10_20_30 | seed42 | fseed20 | 4/8 | 6/8 | 7/8 | 2/8 | +0.1100 | +0.1353 | -4.0381 |
| feature | edge_feature_fseeds_10_20_30 | seed42 | fseed30 | 5/8 | 5/8 | 8/8 | 3/8 | +0.1361 | -0.0013 | -5.0184 |
| feature | edge_feature_fseeds_10_20_30 | seed123 | fseed10 | 4/8 | 3/8 | 7/8 | 2/8 | +0.1377 | -0.0027 | -4.3396 |
| feature | edge_feature_fseeds_10_20_30 | seed123 | fseed20 | 4/8 | 2/8 | 6/8 | 1/8 | +0.1289 | -0.0621 | -3.1416 |
| feature | edge_feature_fseeds_10_20_30 | seed123 | fseed30 | 4/8 | 2/8 | 8/8 | 1/8 | +0.1485 | -0.1746 | -5.4011 |
| feature | edge_feature_fseeds_10_20_30 | seed2024 | fseed10 | 4/8 | 8/8 | 7/8 | 4/8 | +0.1306 | +0.1741 | -3.7050 |
| feature | edge_feature_fseeds_10_20_30 | seed2024 | fseed20 | 4/8 | 7/8 | 6/8 | 4/8 | +0.1250 | +0.1690 | -2.5279 |
| feature | edge_feature_fseeds_10_20_30 | seed2024 | fseed30 | 4/8 | 8/8 | 8/8 | 4/8 | +0.1346 | +0.2274 | -4.3334 |
| feature | fseeds_40_50_60 | seed42 | fseed40 | 5/8 | 3/8 | 8/8 | 1/8 | +0.1336 | -0.0664 | -6.2114 |
| feature | fseeds_40_50_60 | seed42 | fseed50 | 4/8 | 3/8 | 6/8 | 1/8 | +0.0704 | -0.0738 | -2.5460 |
| feature | fseeds_40_50_60 | seed42 | fseed60 | 4/8 | 4/8 | 6/8 | 2/8 | +0.0284 | -0.0456 | -2.4074 |
| feature | fseeds_40_50_60 | seed123 | fseed40 | 4/8 | 0/8 | 8/8 | 0/8 | +0.1550 | -0.2406 | -6.7323 |
| feature | fseeds_40_50_60 | seed123 | fseed50 | 4/8 | 6/8 | 6/8 | 4/8 | +0.0792 | +0.1168 | -2.4317 |
| feature | fseeds_40_50_60 | seed123 | fseed60 | 4/8 | 5/8 | 6/8 | 4/8 | +0.0908 | +0.1477 | -2.1520 |
| feature | fseeds_40_50_60 | seed2024 | fseed40 | 4/8 | 5/8 | 8/8 | 4/8 | +0.1439 | +0.0482 | -6.0904 |
| feature | fseeds_40_50_60 | seed2024 | fseed50 | 4/8 | 5/8 | 6/8 | 4/8 | +0.0995 | -0.0153 | -1.8787 |
| feature | fseeds_40_50_60 | seed2024 | fseed60 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0649 | -0.2217 | -1.7707 |
| node | default_main | seed42 | fseed70042 | 6/8 | 4/8 | 7/8 | 4/8 | +0.1564 | +0.0798 | -5.1046 |
| node | default_main | seed123 | fseed70123 | 5/8 | 6/8 | 7/8 | 4/8 | +0.1348 | +0.0226 | -3.7649 |
| node | default_main | seed2024 | fseed72024 | 6/8 | 7/8 | 7/8 | 5/8 | +0.1592 | +0.0919 | -4.7749 |
| node | fseeds_40_50_60 | seed42 | fseed40 | 7/8 | 6/8 | 7/8 | 5/8 | +0.1546 | +0.0887 | -3.6249 |
| node | fseeds_40_50_60 | seed42 | fseed50 | 6/8 | 5/8 | 8/8 | 5/8 | +0.1408 | +0.0526 | -6.8032 |
| node | fseeds_40_50_60 | seed42 | fseed60 | 7/8 | 7/8 | 7/8 | 6/8 | +0.1475 | +0.0813 | -3.6869 |
| node | fseeds_40_50_60 | seed123 | fseed40 | 4/8 | 5/8 | 8/8 | 3/8 | +0.1599 | +0.0628 | -3.8431 |
| node | fseeds_40_50_60 | seed123 | fseed50 | 5/8 | 7/8 | 7/8 | 4/8 | +0.1533 | +0.0360 | -3.8170 |
| node | fseeds_40_50_60 | seed123 | fseed60 | 4/8 | 1/8 | 8/8 | 1/8 | +0.1540 | -0.0633 | -5.1553 |
| node | fseeds_40_50_60 | seed2024 | fseed40 | 6/8 | 8/8 | 7/8 | 5/8 | +0.1608 | +0.1222 | -4.7817 |
| node | fseeds_40_50_60 | seed2024 | fseed50 | 7/8 | 8/8 | 6/8 | 6/8 | +0.1513 | +0.0680 | -2.9658 |
| node | fseeds_40_50_60 | seed2024 | fseed60 | 5/8 | 7/8 | 7/8 | 5/8 | +0.1522 | +0.1106 | -4.5899 |

## PrimeKG-NoSource

Paired check: `HASI groups=252`, `baseline records used=1086`, `pairs=1086`, `missing among used=0`, `extra unmatched baselines excluded=0`, `fairness_issues=0`.

### Baseline Coverage
| type | baseline | paired runs |
| --- | --- | --- |
| edge | opengu-gif | 78 |
| edge | opengu-gnndelete | 78 |
| edge | opengu-grapheraser-bekm | 78 |
| edge | opengu-grapheraser-blpa | 78 |
| edge | retrain | 78 |
| feature | opengu-gnndelete | 132 |
| feature | opengu-grapheraser-bekm | 132 |
| feature | opengu-grapheraser-blpa | 132 |
| feature | retrain | 132 |
| node | opengu-gnndelete | 42 |
| node | opengu-grapheraser-bekm | 42 |
| node | opengu-grapheraser-blpa | 42 |
| node | retrain | 42 |

### Summary by Type
| type | acc | privacy | emb | all3 | d_acc | d_priv | d_emb |
| --- | --- | --- | --- | --- | --- | --- | --- |
| edge | 359/390 | 76/390 | 234/312 | 15/312 | +0.1507 | -0.0394 | -0.6633 |
| feature | 234/528 | 50/528 | 426/528 | 1/528 | +0.0229 | -0.1030 | -0.4236 |
| node | 144/168 | 63/168 | 126/168 | 34/168 | +0.0512 | -0.0214 | -0.5895 |

### Summary by Family and Type
| family | type | acc | privacy | emb | all3 | d_acc | d_priv | d_emb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| default_main | edge | 28/30 | 4/30 | 18/24 | 0/24 | +0.1480 | -0.0419 | -0.6544 |
| default_main | feature | 6/24 | 2/24 | 24/24 | 1/24 | -0.0146 | -0.0680 | -0.6014 |
| default_main | node | 20/24 | 7/24 | 18/24 | 3/24 | +0.0490 | -0.0335 | -0.6077 |
| edge_feature_fseeds_100_110_120 | edge | 84/90 | 16/90 | 54/72 | 4/72 | +0.1518 | -0.0419 | -0.6597 |
| edge_feature_fseeds_100_110_120 | feature | 36/72 | 8/72 | 54/72 | 0/72 | +0.0334 | -0.1035 | -0.3999 |
| edge_feature_fseeds_70_80_90 | edge | 82/90 | 22/90 | 54/72 | 5/72 | +0.1502 | -0.0355 | -0.6629 |
| edge_feature_fseeds_70_80_90 | feature | 24/72 | 4/72 | 66/72 | 0/72 | -0.0001 | -0.0795 | -0.4961 |
| feature_fseeds_130_140_150 | feature | 30/72 | 4/72 | 60/72 | 0/72 | +0.0209 | -0.1278 | -0.4583 |
| feature_fseeds_160_170_180 | feature | 36/72 | 10/72 | 54/72 | 0/72 | +0.0310 | -0.1034 | -0.3233 |
| feature_fseeds_190_200_210 | feature | 36/72 | 8/72 | 54/72 | 0/72 | +0.0355 | -0.1026 | -0.4062 |
| fseeds_10_20_30 | edge | 83/90 | 15/90 | 54/72 | 2/72 | +0.1508 | -0.0385 | -0.6600 |
| fseeds_10_20_30 | feature | 36/72 | 10/72 | 54/72 | 0/72 | +0.0319 | -0.1105 | -0.3226 |
| fseeds_10_20_30 | node | 63/72 | 25/72 | 54/72 | 15/72 | +0.0521 | -0.0348 | -0.5823 |
| fseeds_40_50_60 | edge | 82/90 | 19/90 | 54/72 | 4/72 | +0.1508 | -0.0407 | -0.6734 |
| fseeds_40_50_60 | feature | 30/72 | 4/72 | 60/72 | 0/72 | +0.0198 | -0.1053 | -0.4995 |
| fseeds_40_50_60 | node | 61/72 | 31/72 | 54/72 | 16/72 | +0.0510 | -0.0040 | -0.5907 |

### Recommended Forget Set per Shared Base
| type | shared_base | recommended forget set | evidence |
| --- | --- | --- | --- |
| edge | seed42 | fseeds_10_20_30 / fseed20 | acc 10/10, privacy 3/10, emb 6/8, all3 1/8 |
| edge | seed123 | edge_feature_fseeds_100_110_120 / fseed120 | acc 10/10, privacy 4/10, emb 6/8, all3 2/8 |
| edge | seed2024 | edge_feature_fseeds_70_80_90 / fseed70 | acc 8/10, privacy 4/10, emb 6/8, all3 2/8 |
| feature | seed42 | default_main / fseed70042 | acc 2/8, privacy 2/8, emb 8/8, all3 1/8 |
| feature | seed123 | feature_fseeds_190_200_210 / fseed190 | acc 4/8, privacy 0/8, emb 6/8, all3 0/8 |
| feature | seed2024 | feature_fseeds_190_200_210 / fseed200 | acc 4/8, privacy 2/8, emb 6/8, all3 0/8 |
| node | seed42 | fseeds_40_50_60 / fseed60 | acc 7/8, privacy 5/8, emb 6/8, all3 4/8 |
| node | seed123 | fseeds_10_20_30 / fseed10 | acc 8/8, privacy 5/8, emb 6/8, all3 4/8 |
| node | seed2024 | fseeds_40_50_60 / fseed60 | acc 7/8, privacy 5/8, emb 6/8, all3 4/8 |

### All Seed Details
| type | family | shared_base | forget_seed | acc | privacy | emb | all3 | d_acc | d_priv | d_emb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edge | default_main | seed42 | fseed70042 | 10/10 | 2/10 | 6/8 | 0/8 | +0.1569 | -0.0393 | -0.7252 |
| edge | default_main | seed123 | fseed70123 | 10/10 | 2/10 | 6/8 | 0/8 | +0.1454 | -0.0302 | -0.5667 |
| edge | default_main | seed2024 | fseed72024 | 8/10 | 0/10 | 6/8 | 0/8 | +0.1417 | -0.0562 | -0.6714 |
| edge | edge_feature_fseeds_100_110_120 | seed42 | fseed100 | 10/10 | 2/10 | 6/8 | 0/8 | +0.1609 | -0.0455 | -0.7602 |
| edge | edge_feature_fseeds_100_110_120 | seed42 | fseed110 | 10/10 | 2/10 | 6/8 | 0/8 | +0.1564 | -0.0528 | -0.7826 |
| edge | edge_feature_fseeds_100_110_120 | seed42 | fseed120 | 10/10 | 2/10 | 6/8 | 0/8 | +0.1589 | -0.0401 | -0.7600 |
| edge | edge_feature_fseeds_100_110_120 | seed123 | fseed100 | 10/10 | 3/10 | 6/8 | 1/8 | +0.1472 | -0.0222 | -0.5428 |
| edge | edge_feature_fseeds_100_110_120 | seed123 | fseed110 | 10/10 | 3/10 | 6/8 | 1/8 | +0.1484 | -0.0323 | -0.5246 |
| edge | edge_feature_fseeds_100_110_120 | seed123 | fseed120 | 10/10 | 4/10 | 6/8 | 2/8 | +0.1477 | -0.0255 | -0.5245 |
| edge | edge_feature_fseeds_100_110_120 | seed2024 | fseed100 | 8/10 | 0/10 | 6/8 | 0/8 | +0.1496 | -0.0592 | -0.7002 |
| edge | edge_feature_fseeds_100_110_120 | seed2024 | fseed110 | 8/10 | 0/10 | 6/8 | 0/8 | +0.1486 | -0.0504 | -0.6747 |
| edge | edge_feature_fseeds_100_110_120 | seed2024 | fseed120 | 8/10 | 0/10 | 6/8 | 0/8 | +0.1486 | -0.0491 | -0.6676 |
| edge | edge_feature_fseeds_70_80_90 | seed42 | fseed70 | 10/10 | 2/10 | 6/8 | 0/8 | +0.1583 | -0.0435 | -0.7490 |
| edge | edge_feature_fseeds_70_80_90 | seed42 | fseed80 | 10/10 | 2/10 | 6/8 | 0/8 | +0.1559 | -0.0413 | -0.7349 |
| edge | edge_feature_fseeds_70_80_90 | seed42 | fseed90 | 10/10 | 2/10 | 6/8 | 0/8 | +0.1580 | -0.0469 | -0.7719 |
| edge | edge_feature_fseeds_70_80_90 | seed123 | fseed70 | 9/10 | 3/10 | 6/8 | 1/8 | +0.1451 | -0.0311 | -0.5064 |
| edge | edge_feature_fseeds_70_80_90 | seed123 | fseed80 | 9/10 | 3/10 | 6/8 | 1/8 | +0.1440 | -0.0253 | -0.5597 |
| edge | edge_feature_fseeds_70_80_90 | seed123 | fseed90 | 10/10 | 2/10 | 6/8 | 0/8 | +0.1444 | -0.0346 | -0.5776 |
| edge | edge_feature_fseeds_70_80_90 | seed2024 | fseed70 | 8/10 | 4/10 | 6/8 | 2/8 | +0.1462 | -0.0196 | -0.7017 |
| edge | edge_feature_fseeds_70_80_90 | seed2024 | fseed80 | 8/10 | 3/10 | 6/8 | 1/8 | +0.1509 | -0.0305 | -0.6965 |
| edge | edge_feature_fseeds_70_80_90 | seed2024 | fseed90 | 8/10 | 1/10 | 6/8 | 0/8 | +0.1488 | -0.0471 | -0.6686 |
| edge | fseeds_10_20_30 | seed42 | fseed10 | 10/10 | 2/10 | 6/8 | 0/8 | +0.1619 | -0.0480 | -0.7273 |
| edge | fseeds_10_20_30 | seed42 | fseed20 | 10/10 | 3/10 | 6/8 | 1/8 | +0.1592 | -0.0418 | -0.7504 |
| edge | fseeds_10_20_30 | seed42 | fseed30 | 9/10 | 2/10 | 6/8 | 0/8 | +0.1602 | -0.0430 | -0.7472 |
| edge | fseeds_10_20_30 | seed123 | fseed10 | 10/10 | 2/10 | 6/8 | 0/8 | +0.1496 | -0.0320 | -0.5633 |
| edge | fseeds_10_20_30 | seed123 | fseed20 | 10/10 | 2/10 | 6/8 | 0/8 | +0.1447 | -0.0301 | -0.5189 |
| edge | fseeds_10_20_30 | seed123 | fseed30 | 10/10 | 2/10 | 6/8 | 0/8 | +0.1447 | -0.0293 | -0.5675 |
| edge | fseeds_10_20_30 | seed2024 | fseed10 | 8/10 | 0/10 | 6/8 | 0/8 | +0.1422 | -0.0366 | -0.7099 |
| edge | fseeds_10_20_30 | seed2024 | fseed20 | 8/10 | 0/10 | 6/8 | 0/8 | +0.1469 | -0.0501 | -0.6802 |
| edge | fseeds_10_20_30 | seed2024 | fseed30 | 8/10 | 2/10 | 6/8 | 1/8 | +0.1479 | -0.0353 | -0.6754 |
| edge | fseeds_40_50_60 | seed42 | fseed40 | 9/10 | 2/10 | 6/8 | 0/8 | +0.1580 | -0.0508 | -0.7905 |
| edge | fseeds_40_50_60 | seed42 | fseed50 | 10/10 | 2/10 | 6/8 | 0/8 | +0.1604 | -0.0489 | -0.7498 |
| edge | fseeds_40_50_60 | seed42 | fseed60 | 10/10 | 2/10 | 6/8 | 0/8 | +0.1585 | -0.0469 | -0.7686 |
| edge | fseeds_40_50_60 | seed123 | fseed40 | 10/10 | 2/10 | 6/8 | 0/8 | +0.1501 | -0.0318 | -0.5020 |
| edge | fseeds_40_50_60 | seed123 | fseed50 | 9/10 | 2/10 | 6/8 | 0/8 | +0.1488 | -0.0321 | -0.5945 |
| edge | fseeds_40_50_60 | seed123 | fseed60 | 10/10 | 3/10 | 6/8 | 1/8 | +0.1467 | -0.0305 | -0.5968 |
| edge | fseeds_40_50_60 | seed2024 | fseed40 | 8/10 | 3/10 | 6/8 | 1/8 | +0.1430 | -0.0379 | -0.6919 |
| edge | fseeds_40_50_60 | seed2024 | fseed50 | 8/10 | 3/10 | 6/8 | 2/8 | +0.1439 | -0.0416 | -0.6921 |
| edge | fseeds_40_50_60 | seed2024 | fseed60 | 8/10 | 0/10 | 6/8 | 0/8 | +0.1477 | -0.0461 | -0.6743 |
| feature | default_main | seed42 | fseed70042 | 2/8 | 2/8 | 8/8 | 1/8 | -0.0227 | -0.0629 | -0.6502 |
| feature | default_main | seed123 | fseed70123 | 2/8 | 0/8 | 8/8 | 0/8 | -0.0129 | -0.0844 | -0.5603 |
| feature | default_main | seed2024 | fseed72024 | 2/8 | 0/8 | 8/8 | 0/8 | -0.0082 | -0.0566 | -0.5937 |
| feature | edge_feature_fseeds_100_110_120 | seed42 | fseed100 | 4/8 | 2/8 | 6/8 | 0/8 | +0.0412 | -0.0601 | -0.5869 |
| feature | edge_feature_fseeds_100_110_120 | seed42 | fseed110 | 4/8 | 2/8 | 6/8 | 0/8 | +0.0221 | -0.0399 | -0.2840 |
| feature | edge_feature_fseeds_100_110_120 | seed42 | fseed120 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0367 | -0.0507 | -0.6046 |
| feature | edge_feature_fseeds_100_110_120 | seed123 | fseed100 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0312 | -0.1253 | -0.2053 |
| feature | edge_feature_fseeds_100_110_120 | seed123 | fseed110 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0286 | -0.1610 | -0.2274 |
| feature | edge_feature_fseeds_100_110_120 | seed123 | fseed120 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0468 | -0.1454 | -0.5104 |
| feature | edge_feature_fseeds_100_110_120 | seed2024 | fseed100 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0367 | -0.2073 | -0.4071 |
| feature | edge_feature_fseeds_100_110_120 | seed2024 | fseed110 | 4/8 | 2/8 | 6/8 | 0/8 | +0.0347 | -0.0693 | -0.3434 |
| feature | edge_feature_fseeds_100_110_120 | seed2024 | fseed120 | 4/8 | 2/8 | 6/8 | 0/8 | +0.0226 | -0.0726 | -0.4304 |
| feature | edge_feature_fseeds_70_80_90 | seed42 | fseed70 | 2/8 | 0/8 | 8/8 | 0/8 | -0.0236 | -0.0828 | -0.6663 |
| feature | edge_feature_fseeds_70_80_90 | seed42 | fseed80 | 4/8 | 2/8 | 6/8 | 0/8 | +0.0221 | -0.0405 | -0.2840 |
| feature | edge_feature_fseeds_70_80_90 | seed42 | fseed90 | 2/8 | 0/8 | 8/8 | 0/8 | -0.0233 | -0.0842 | -0.6655 |
| feature | edge_feature_fseeds_70_80_90 | seed123 | fseed70 | 2/8 | 0/8 | 8/8 | 0/8 | -0.0101 | -0.0886 | -0.5609 |
| feature | edge_feature_fseeds_70_80_90 | seed123 | fseed80 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0288 | -0.1611 | -0.2274 |
| feature | edge_feature_fseeds_70_80_90 | seed123 | fseed90 | 2/8 | 0/8 | 8/8 | 0/8 | -0.0101 | -0.0886 | -0.5610 |
| feature | edge_feature_fseeds_70_80_90 | seed2024 | fseed70 | 2/8 | 0/8 | 8/8 | 0/8 | -0.0096 | -0.0487 | -0.5798 |
| feature | edge_feature_fseeds_70_80_90 | seed2024 | fseed80 | 4/8 | 2/8 | 6/8 | 0/8 | +0.0347 | -0.0702 | -0.3432 |
| feature | edge_feature_fseeds_70_80_90 | seed2024 | fseed90 | 2/8 | 0/8 | 8/8 | 0/8 | -0.0098 | -0.0507 | -0.5765 |
| feature | feature_fseeds_130_140_150 | seed42 | fseed130 | 2/8 | 0/8 | 8/8 | 0/8 | -0.0233 | -0.0842 | -0.6655 |
| feature | feature_fseeds_130_140_150 | seed42 | fseed140 | 4/8 | 2/8 | 6/8 | 0/8 | +0.0399 | -0.0658 | -0.5387 |
| feature | feature_fseeds_130_140_150 | seed42 | fseed150 | 4/8 | 2/8 | 6/8 | 0/8 | +0.0394 | -0.0627 | -0.5406 |
| feature | feature_fseeds_130_140_150 | seed123 | fseed130 | 2/8 | 0/8 | 8/8 | 0/8 | -0.0100 | -0.0887 | -0.5611 |
| feature | feature_fseeds_130_140_150 | seed123 | fseed140 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0362 | -0.1765 | -0.3511 |
| feature | feature_fseeds_130_140_150 | seed123 | fseed150 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0361 | -0.1775 | -0.3544 |
| feature | feature_fseeds_130_140_150 | seed2024 | fseed130 | 2/8 | 0/8 | 8/8 | 0/8 | -0.0095 | -0.0502 | -0.5728 |
| feature | feature_fseeds_130_140_150 | seed2024 | fseed140 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0399 | -0.2236 | -0.2704 |
| feature | feature_fseeds_130_140_150 | seed2024 | fseed150 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0395 | -0.2208 | -0.2699 |
| feature | feature_fseeds_160_170_180 | seed42 | fseed160 | 4/8 | 2/8 | 6/8 | 0/8 | +0.0408 | -0.0593 | -0.5879 |
| feature | feature_fseeds_160_170_180 | seed42 | fseed170 | 4/8 | 2/8 | 6/8 | 0/8 | +0.0221 | -0.0400 | -0.2840 |
| feature | feature_fseeds_160_170_180 | seed42 | fseed180 | 4/8 | 2/8 | 6/8 | 0/8 | +0.0221 | -0.0404 | -0.2841 |
| feature | feature_fseeds_160_170_180 | seed123 | fseed160 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0313 | -0.1252 | -0.2052 |
| feature | feature_fseeds_160_170_180 | seed123 | fseed170 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0289 | -0.1615 | -0.2355 |
| feature | feature_fseeds_160_170_180 | seed123 | fseed180 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0284 | -0.1608 | -0.2229 |
| feature | feature_fseeds_160_170_180 | seed2024 | fseed160 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0367 | -0.2102 | -0.4090 |
| feature | feature_fseeds_160_170_180 | seed2024 | fseed170 | 4/8 | 2/8 | 6/8 | 0/8 | +0.0344 | -0.0641 | -0.3380 |
| feature | feature_fseeds_160_170_180 | seed2024 | fseed180 | 4/8 | 2/8 | 6/8 | 0/8 | +0.0348 | -0.0692 | -0.3432 |
| feature | feature_fseeds_190_200_210 | seed42 | fseed190 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0480 | -0.0382 | -0.6256 |
| feature | feature_fseeds_190_200_210 | seed42 | fseed200 | 4/8 | 2/8 | 6/8 | 0/8 | +0.0221 | -0.0401 | -0.2841 |
| feature | feature_fseeds_190_200_210 | seed42 | fseed210 | 4/8 | 2/8 | 6/8 | 0/8 | +0.0412 | -0.0602 | -0.5870 |
| feature | feature_fseeds_190_200_210 | seed123 | fseed190 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0480 | -0.1498 | -0.5288 |
| feature | feature_fseeds_190_200_210 | seed123 | fseed200 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0288 | -0.1611 | -0.2274 |
| feature | feature_fseeds_190_200_210 | seed123 | fseed210 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0316 | -0.1249 | -0.2053 |
| feature | feature_fseeds_190_200_210 | seed2024 | fseed190 | 4/8 | 2/8 | 6/8 | 0/8 | +0.0282 | -0.0686 | -0.4454 |
| feature | feature_fseeds_190_200_210 | seed2024 | fseed200 | 4/8 | 2/8 | 6/8 | 0/8 | +0.0348 | -0.0706 | -0.3434 |
| feature | feature_fseeds_190_200_210 | seed2024 | fseed210 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0367 | -0.2102 | -0.4092 |
| feature | fseeds_10_20_30 | seed42 | fseed10 | 4/8 | 2/8 | 6/8 | 0/8 | +0.0247 | -0.0281 | -0.2884 |
| feature | fseeds_10_20_30 | seed42 | fseed20 | 4/8 | 2/8 | 6/8 | 0/8 | +0.0400 | -0.0675 | -0.5393 |
| feature | fseeds_10_20_30 | seed42 | fseed30 | 4/8 | 2/8 | 6/8 | 0/8 | +0.0221 | -0.0408 | -0.2842 |
| feature | fseeds_10_20_30 | seed123 | fseed10 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0288 | -0.1620 | -0.2451 |
| feature | fseeds_10_20_30 | seed123 | fseed20 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0363 | -0.1780 | -0.3507 |
| feature | fseeds_10_20_30 | seed123 | fseed30 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0286 | -0.1607 | -0.2198 |
| feature | fseeds_10_20_30 | seed2024 | fseed10 | 4/8 | 2/8 | 6/8 | 0/8 | +0.0319 | -0.0670 | -0.3638 |
| feature | fseeds_10_20_30 | seed2024 | fseed20 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0397 | -0.2211 | -0.2688 |
| feature | fseeds_10_20_30 | seed2024 | fseed30 | 4/8 | 2/8 | 6/8 | 0/8 | +0.0347 | -0.0697 | -0.3434 |
| feature | fseeds_40_50_60 | seed42 | fseed40 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0369 | -0.0498 | -0.6036 |
| feature | fseeds_40_50_60 | seed42 | fseed50 | 2/8 | 0/8 | 8/8 | 0/8 | -0.0233 | -0.0842 | -0.6655 |
| feature | fseeds_40_50_60 | seed42 | fseed60 | 4/8 | 2/8 | 6/8 | 0/8 | +0.0394 | -0.0632 | -0.5395 |
| feature | fseeds_40_50_60 | seed123 | fseed40 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0466 | -0.1423 | -0.4938 |
| feature | fseeds_40_50_60 | seed123 | fseed50 | 2/8 | 0/8 | 8/8 | 0/8 | -0.0101 | -0.0885 | -0.5606 |
| feature | fseeds_40_50_60 | seed123 | fseed60 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0362 | -0.1778 | -0.3539 |
| feature | fseeds_40_50_60 | seed2024 | fseed40 | 4/8 | 2/8 | 6/8 | 0/8 | +0.0226 | -0.0713 | -0.4306 |
| feature | fseeds_40_50_60 | seed2024 | fseed50 | 2/8 | 0/8 | 8/8 | 0/8 | -0.0095 | -0.0502 | -0.5778 |
| feature | fseeds_40_50_60 | seed2024 | fseed60 | 4/8 | 0/8 | 6/8 | 0/8 | +0.0398 | -0.2205 | -0.2703 |
| node | default_main | seed42 | fseed70042 | 8/8 | 0/8 | 6/8 | 0/8 | +0.0614 | -0.1110 | -0.6853 |
| node | default_main | seed123 | fseed70123 | 6/8 | 1/8 | 6/8 | 0/8 | +0.0406 | -0.0934 | -0.4823 |
| node | default_main | seed2024 | fseed72024 | 6/8 | 6/8 | 6/8 | 3/8 | +0.0451 | +0.1040 | -0.6556 |
| node | fseeds_10_20_30 | seed42 | fseed10 | 8/8 | 1/8 | 6/8 | 1/8 | +0.0538 | -0.0762 | -0.6515 |
| node | fseeds_10_20_30 | seed42 | fseed20 | 8/8 | 0/8 | 6/8 | 0/8 | +0.0569 | -0.0994 | -0.7229 |
| node | fseeds_10_20_30 | seed42 | fseed30 | 8/8 | 1/8 | 6/8 | 1/8 | +0.0648 | -0.0406 | -0.7134 |
| node | fseeds_10_20_30 | seed123 | fseed10 | 8/8 | 5/8 | 6/8 | 4/8 | +0.0438 | +0.0184 | -0.5134 |
| node | fseeds_10_20_30 | seed123 | fseed20 | 8/8 | 1/8 | 6/8 | 0/8 | +0.0477 | -0.1050 | -0.4807 |
| node | fseeds_10_20_30 | seed123 | fseed30 | 6/8 | 3/8 | 6/8 | 2/8 | +0.0412 | -0.0708 | -0.5484 |
| node | fseeds_10_20_30 | seed2024 | fseed10 | 6/8 | 4/8 | 6/8 | 2/8 | +0.0503 | +0.0133 | -0.5622 |
| node | fseeds_10_20_30 | seed2024 | fseed20 | 5/8 | 5/8 | 6/8 | 2/8 | +0.0531 | +0.0167 | -0.5632 |
| node | fseeds_10_20_30 | seed2024 | fseed30 | 6/8 | 5/8 | 6/8 | 3/8 | +0.0569 | +0.0302 | -0.4847 |
| node | fseeds_40_50_60 | seed42 | fseed40 | 8/8 | 1/8 | 6/8 | 0/8 | +0.0574 | -0.0679 | -0.6738 |
| node | fseeds_40_50_60 | seed42 | fseed50 | 8/8 | 1/8 | 6/8 | 1/8 | +0.0627 | -0.0482 | -0.7605 |
| node | fseeds_40_50_60 | seed42 | fseed60 | 7/8 | 5/8 | 6/8 | 4/8 | +0.0526 | +0.0237 | -0.6395 |
| node | fseeds_40_50_60 | seed123 | fseed40 | 8/8 | 4/8 | 6/8 | 2/8 | +0.0400 | -0.0036 | -0.5205 |
| node | fseeds_40_50_60 | seed123 | fseed50 | 6/8 | 1/8 | 6/8 | 0/8 | +0.0477 | -0.1116 | -0.5005 |
| node | fseeds_40_50_60 | seed123 | fseed60 | 7/8 | 4/8 | 6/8 | 2/8 | +0.0422 | +0.0183 | -0.5227 |
| node | fseeds_40_50_60 | seed2024 | fseed40 | 5/8 | 6/8 | 6/8 | 2/8 | +0.0471 | +0.0863 | -0.6195 |
| node | fseeds_40_50_60 | seed2024 | fseed50 | 5/8 | 4/8 | 6/8 | 1/8 | +0.0510 | -0.0060 | -0.5736 |
| node | fseeds_40_50_60 | seed2024 | fseed60 | 7/8 | 5/8 | 6/8 | 4/8 | +0.0581 | +0.0727 | -0.5061 |

## Ablation Results

The ablation section uses the canonical PubMed RQ result roots under `results/rq/pubmed`. These runs are not mixed with the three-dataset main table; they explain internal HASI design choices.

### RQ3 Anchor Ablation

Root: `results/rq/pubmed/rq3_anchor_ablation/hasi`

| method | runs | mean_acc | mean_privacy | mean_emb | mean_time_s | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| hasi-hier-anchor-rq3-repairfix | 3 | +0.8665 | +0.6473 | +0.1381 | +1521.9493 | seed42, seed123, seed2024 |
| hasi-no-anchor-rq3-repairfix | 3 | +0.8671 | +0.8136 | +0.0711 | +4.0325 | seed42, seed123, seed2024 |
| hasi-strong-anchor-rq3-repairfix | 3 | +0.8687 | +0.6148 | +0.2131 | +1558.0991 | seed42, seed123, seed2024 |

Seed-level details:
| method | shared_base | ratio | accuracy_after | privacy_score | embedding_l2_mean | unlearn_time_s |
| --- | --- | --- | --- | --- | --- | --- |
| hasi-hier-anchor-rq3-repairfix | seed42 | r0p05 | 0.8646 | 0.8404 | 0.0965 | 1371.3 |
| hasi-hier-anchor-rq3-repairfix | seed123 | r0p05 | 0.8618 | 0.5191 | 0.1867 | 1905.9 |
| hasi-hier-anchor-rq3-repairfix | seed2024 | r0p05 | 0.8730 | 0.5825 | 0.1311 | 1288.6 |
| hasi-no-anchor-rq3-repairfix | seed42 | r0p05 | 0.8600 | 0.8802 | 0.0794 | 4.0 |
| hasi-no-anchor-rq3-repairfix | seed123 | r0p05 | 0.8725 | 0.7527 | 0.0718 | 4.1 |
| hasi-no-anchor-rq3-repairfix | seed2024 | r0p05 | 0.8689 | 0.8079 | 0.0622 | 4.0 |
| hasi-strong-anchor-rq3-repairfix | seed42 | r0p05 | 0.8631 | 0.8391 | 0.2264 | 1402.3 |
| hasi-strong-anchor-rq3-repairfix | seed123 | r0p05 | 0.8687 | 0.4676 | 0.2206 | 1973.2 |
| hasi-strong-anchor-rq3-repairfix | seed2024 | r0p05 | 0.8745 | 0.5377 | 0.1925 | 1298.8 |

### RQ4 Structural Inpainting Ablation

Root: `results/rq/pubmed/rq4_structural_inpainting/hasi`

| method | runs | mean_acc | mean_privacy | mean_emb | mean_time_s | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| hasi-full-inpaint | 3 | +0.8665 | +0.6473 | +0.1381 | +1453.5656 | seed42, seed123, seed2024 |
| hasi-no-inpaint | 3 | +0.8590 | +0.7306 | +0.1318 | +1443.2197 | seed42, seed123, seed2024 |

Seed-level details:
| method | shared_base | ratio | accuracy_after | privacy_score | embedding_l2_mean | unlearn_time_s |
| --- | --- | --- | --- | --- | --- | --- |
| hasi-full-inpaint | seed42 | r0p05 | 0.8646 | 0.8404 | 0.0965 | 1310.5 |
| hasi-full-inpaint | seed123 | r0p05 | 0.8618 | 0.5191 | 0.1867 | 1830.0 |
| hasi-full-inpaint | seed2024 | r0p05 | 0.8730 | 0.5825 | 0.1311 | 1220.1 |
| hasi-no-inpaint | seed42 | r0p05 | 0.8486 | 0.8784 | 0.1562 | 1285.2 |
| hasi-no-inpaint | seed123 | r0p05 | 0.8722 | 0.6966 | 0.0950 | 1826.8 |
| hasi-no-inpaint | seed2024 | r0p05 | 0.8562 | 0.6169 | 0.1443 | 1217.6 |

### RQ5 DAR Ablation

Root: `results/rq/pubmed/rq5_dar_ablation/hasi`

| method | runs | mean_acc | mean_privacy | mean_emb | mean_time_s | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| hasi-dar-off | 3 | +0.8712 | +0.9701 | +0.0895 | +3.2424 | seed42, seed123, seed2024 |
| hasi-dar-on | 3 | +0.8779 | +0.9812 | +0.1473 | +10914.1262 | seed42, seed123, seed2024 |

Seed-level details:
| method | shared_base | ratio | accuracy_after | privacy_score | embedding_l2_mean | unlearn_time_s |
| --- | --- | --- | --- | --- | --- | --- |
| hasi-dar-off | seed42 | r0p05 | 0.8722 | 0.9883 | 0.1024 | 3.4 |
| hasi-dar-off | seed123 | r0p05 | 0.8674 | 0.9670 | 0.0773 | 3.1 |
| hasi-dar-off | seed2024 | r0p05 | 0.8740 | 0.9548 | 0.0887 | 3.2 |
| hasi-dar-on | seed42 | r0p05 | 0.8775 | 0.9703 | 0.1544 | 11648.0 |
| hasi-dar-on | seed123 | r0p05 | 0.8793 | 0.9797 | 0.1482 | 10513.4 |
| hasi-dar-on | seed2024 | r0p05 | 0.8768 | 0.9935 | 0.1392 | 10580.9 |

## Paper-Facing Recommendation

- Main baseline table: use OpenGU core baselines plus src `retrain`.
- Supplementary consistency table: optionally keep the older src-baseline comparison.
- Strong-baseline appendix: report MEGU separately, not inside the main table.
- Main evidence: emphasize PubMed node, Hetionet node, and Hetionet feature.
- Careful wording: edge unlearning supports utility/representation much more than privacy; PrimeKG-NoSource feature is not a clean privacy win.
