# PubMed Default Main 与 Shared Base 协议记录

更新时间：2026-06-09

本文档整理当前 `results/pubmed/default_main` 中 fixed forget sets 对比结果，以及 `results/shared_base` 使用的数据划分、训练图构造方式和 forget sets 来源。这里的结论基于当前已读取的 30 个 PubMed 结果：

```text
2 ratios × 3 seeds × 5 methods = 30
```

其中每个对比组固定相同的：

```text
ratio + seed + forget_set
```

评价指标解释：

- `acc_drop`：越小越好。
- `f1_drop`：越小越好。
- `MIA AUC`：越接近 0.5 越好。
- `privacy_score`：越高越好。
- `time`：使用 `online_wall_clock_seconds`。

---

## 1. PubMed Default Main 对比结果

### 1.1 Group 1：ratio=0.05, seed=42

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | -0.0061 | -0.0069 | 0.5590 | 0.8821 | 1400.52s |
| hasi_default | -0.0079 | -0.0084 | 0.5588 | 0.8824 | 1397.87s |
| retrain | 0.0089 | 0.0068 | 0.6106 | 0.7788 | 2.60s |
| grapheraser_bekm | 0.0461 | 0.0616 | 0.6145 | 0.7709 | 6.10s |
| grapheraser_blpa | 0.0155 | 0.0132 | 0.6599 | 0.6801 | 6.27s |

小结：`hasi_default` 在 utility 和 privacy 上都最好。

### 1.2 Group 2：ratio=0.05, seed=123

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | 0.0030 | 0.0024 | 0.5969 | 0.8061 | 503.90s |
| hasi_default | -0.0028 | -0.0027 | 0.5922 | 0.8157 | 504.31s |
| retrain | 0.0043 | 0.0036 | 0.6045 | 0.7910 | 2.74s |
| grapheraser_bekm | 0.0996 | 0.1735 | 0.6140 | 0.7720 | 5.98s |
| grapheraser_blpa | 0.0162 | 0.0173 | 0.6801 | 0.6397 | 6.06s |

小结：`hasi_default` 最好；`hasi_tuned` 这一组没有超过 default。

### 1.3 Group 3：ratio=0.05, seed=2024

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | 0.0013 | 0.0005 | 0.5862 | 0.8276 | 862.91s |
| hasi_default | 0.0003 | -0.0013 | 0.5804 | 0.8392 | 859.14s |
| retrain | 0.0025 | 0.0008 | 0.6099 | 0.7802 | 2.74s |
| grapheraser_bekm | 0.1060 | 0.1834 | 0.6240 | 0.7520 | 6.16s |
| grapheraser_blpa | 0.0157 | 0.0140 | 0.6856 | 0.6289 | 5.84s |

小结：`hasi_default` 最好；`hasi_tuned` 比 baselines 大多好，但不如 default。

### 1.4 Group 4：ratio=0.1, seed=42

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | -0.0023 | -0.0032 | 0.5831 | 0.8339 | 2531.38s |
| hasi_default | -0.0066 | -0.0070 | 0.5590 | 0.8820 | 2511.54s |
| retrain | 0.0063 | 0.0048 | 0.5804 | 0.8391 | 2.77s |
| grapheraser_bekm | 0.0454 | 0.0614 | 0.5950 | 0.8100 | 6.09s |
| grapheraser_blpa | 0.0193 | 0.0163 | 0.6340 | 0.7321 | 5.87s |

小结：`hasi_default` 明显优于 `hasi_tuned`。

### 1.5 Group 5：ratio=0.1, seed=123

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | -0.0020 | -0.0025 | 0.5732 | 0.8536 | 1030.65s |
| hasi_default | -0.0018 | -0.0026 | 0.5651 | 0.8698 | 1023.84s |
| retrain | 0.0038 | 0.0026 | 0.5762 | 0.8477 | 2.80s |
| grapheraser_bekm | 0.0925 | 0.1554 | 0.6051 | 0.7898 | 6.07s |
| grapheraser_blpa | 0.0155 | 0.0162 | 0.6577 | 0.6845 | 6.09s |

小结：`hasi_tuned` 的 `accuracy_drop` 略好，但 `hasi_default` 的 F1 和 privacy 更好。

### 1.6 Group 6：ratio=0.1, seed=2024

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | 0.0046 | 0.0022 | 0.5756 | 0.8488 | 1461.59s |
| hasi_default | -0.0015 | -0.0034 | 0.5522 | 0.8955 | 1471.03s |
| retrain | 0.0005 | -0.0014 | 0.5561 | 0.8878 | 2.81s |
| grapheraser_bekm | 0.1034 | 0.1782 | 0.5979 | 0.8042 | 6.16s |
| grapheraser_blpa | 0.0147 | 0.0125 | 0.6589 | 0.6822 | 5.86s |

小结：`hasi_default` 最好，尤其 privacy 明显优于 tuned。

---

## 2. 胜出统计

### 2.1 与所有方法一起比较

| 指标 | hasi_tuned | hasi_default |
|---|---:|---:|
| accuracy_drop 最小 | 1/6 | 5/6 |
| f1_drop 最小 | 0/6 | 6/6 |
| MIA AUC 最接近 0.5 | 0/6 | 6/6 |
| privacy_score 最高 | 0/6 | 6/6 |

### 2.2 是否击败全部 baselines

| 指标 | hasi_tuned | hasi_default |
|---|---:|---:|
| accuracy_drop 优于全部 baselines | 5/6 | 6/6 |
| f1_drop 优于全部 baselines | 5/6 | 6/6 |
| MIA AUC 优于全部 baselines | 4/6 | 6/6 |
| privacy_score 优于全部 baselines | 4/6 | 6/6 |

### 2.3 当前结论

当前 `results/pubmed/default_main` 这批 fixed forget sets 结果中，`hasi_default` 比 `hasi_tuned` 更稳。

`hasi_tuned` 并不是完全差：它在大多数情况下仍然优于 baselines。但在这批 fixed forget sets 协议下，它没有保持调参结果里“优于 default”的表现。主要原因是：当前 tuned 配置并不是在这套 fixed forget sets 协议下重新选出来的。

---

## 3. Shared Base 使用的数据

当前 `results/shared_base` 使用的数据协议为：

```text
split = stratified_random
train / val / test = 60% / 20% / 20%
training_graph = train_subgraph
```

也就是先划分节点，再只用训练节点诱导出的训练子图训练 base GNN。

| dataset | full nodes | train | val | test | feature dim | train-subgraph edges |
|---|---:|---:|---:|---:|---:|---|
| Cora | 2708 | 1626 | 543 | 539 | 1433 | seed42: 3542, seed123: 3712, seed2024: 3838 |
| CiteSeer | 3327 | 1997 | 666 | 664 | 3703 | seed42: 3252, seed123: 3142, seed2024: 3272 |
| PubMed | 19717 | 11830 | 3943 | 3944 | 500 | seed42: 31100, seed123: 30952, seed2024: 31920 |

以 PubMed 为例：

- 完整图：19717 个节点。
- 训练节点：11830 个。
- 验证节点：3943 个。
- 测试节点：3944 个。
- base GNN 训练时只使用 11830 个训练节点、训练节点之间的边，以及训练节点对应的 feature。
- 验证集和测试集节点没有参与 base GNN 训练。

---

## 4. Forget Sets 来源

### 4.1 Node Forget Sets

node forget sets 来自训练节点。

以 PubMed 为例：

| forget set | forget nodes | candidate_count |
|---|---:|---:|
| `pubmed_node_r0p05_random_train_seed42.json` | 592 | 11830 |
| `pubmed_node_r0p1_random_train_seed42.json` | 1183 | 11830 |

这里的候选范围是 PubMed 的 `train_mask`。

### 4.2 Edge Forget Sets

edge forget sets 来自训练子图中的边。

虽然文件名里是 `random_all`，但这里的 `all` 指的是：

```text
所有候选训练边
```

不是完整图中的所有边。

以 PubMed seed42 为例：

```text
shared_base train-subgraph edges = 31100
edge forget candidate_count = 31100
```

因此 edge forget set 是从训练节点之间的边里抽取出来的。

### 4.3 Feature Forget Sets

feature forget sets 不是从训练节点中抽取，而是从 feature dimensions 中抽取。

以 PubMed 为例：

| ratio | feature dim | forgotten feature dims |
|---|---:|---:|
| 0.05 | 500 | 25 |
| 0.1 | 500 | 50 |

feature 遗忘的对象是“特征维度”，不是某些节点。因此它不能简单说“来自训练集节点”。不过，base model 训练时只使用过训练节点上的这些 feature，测试节点的 feature 没有参与 base 训练。

---

## 5. 一句话总结

`results/shared_base` 使用 60% 训练节点诱导出的训练子图训练 base GNN；node forget 从训练节点抽取，edge forget 从训练节点之间的边抽取，feature forget 从全部 feature 维度抽取。

在当前 `results/pubmed/default_main` 的 fixed forget sets 结果中，`hasi_default` 比 `hasi_tuned` 更稳；`hasi_tuned` 多数情况下仍优于 baselines，但没有在该协议下超过 default。
