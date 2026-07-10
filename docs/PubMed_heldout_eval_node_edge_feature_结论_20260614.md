# PubMed RQ1/RQ3 与 Held-out Evaluation 结论汇总（2026-06-15）

## 1. 实验设置

本文件现在整合三类 PubMed 结果：

```text
RQ1 clean node-selection experiment
RQ3 anchor stabilization ablation
Held-out node / feature / edge final evaluation
```

其中，held-out final evaluation 使用的是最终评估专用 forget sets，而不是调参时使用的 `experiments/forget_sets`。

```text
shared_base: results/shared_base/pubmed/seed42, seed123, seed2024
held-out forget sets: experiments/forget_sets_eval/pubmed
results: results/pubmed_eval
dataset: PubMed
ratios: 0.05, 0.1
base_seed: 42, 123, 2024
forget_seed: 70042, 70123, 72024
```

seed 对应关系：

```text
base_seed=42    forget_seed=70042
base_seed=123   forget_seed=70123
base_seed=2024  forget_seed=72024
```

评价口径：

```text
acc_drop / f1_drop 越小越好
MIA AUC 越接近 0.5 越好
privacy_score 越高越好
time 使用 online_wall_clock_seconds
```

---


## 2. RQ1: Node Selection Difficulty

RQ1 目标是验证不同训练节点删除类型是否带来不同 unlearning 难度。这个实验只改变 forget node selection，其他条件保持固定。

结果路径：

```text
forget sets: experiments/rq_forget_sets/pubmed/rq1_node_selection/
results: results/rq/pubmed/rq1_node_selection/hasi_default/
aggregate: results/rq/pubmed/rq1_node_selection/hasi_default/aggregate_summary.*
pre-repair archive: results/rq/pubmed/rq1_node_selection/hasi_default/pre_inpainting_repair_fix/
```

本节使用的是 node inpainting repair target 修复后的 RQ1 结果。修复后，`random_train` 和 `hub_train` 会真正触发 full inpainting；`low_degree_train` 因 affected region 为空，不触发补边。

补边状态：

| selection | ratio | triggered | edges_added | repair_budget | reason |
|---|---:|---:|---:|---:|---|
| low_degree_train | 0.05 / 0.1 | 0/3 | 0 | 0 | empty affected region |
| random_train | 0.05 / 0.1 | 3/3 | 256 | 256 | hub-to-hub deletion |
| hub_train | 0.05 / 0.1 | 3/3 | 256 | 256 | hub-to-hub deletion |

固定设置：

```text
dataset = pubmed
unlearning_type = node
method = hasi_default
config = configs/hasi_default.yaml
base_model = results/shared_base/pubmed/seed{seed}
candidate_scope = train_mask
score_graph = train_subgraph
ratios = 0.05, 0.1
seeds = 42, 123, 2024
```

selection 定义：

```text
random_train:
  从 train_mask 中随机采样

low_degree_train:
  在 train_subgraph 中计算 degree，选择 degree 最低的 train nodes

hub_train:
  在 train_subgraph 中计算 HASI hub score，选择 hub score 最高的 train nodes
```

这个 RQ1 是干净的：forget targets 全部来自对应 shared_base seed 的 `train_mask`，`low_degree_train` 和 `hub_train` 的排序只基于 `train_subgraph`，不使用 val/test 节点作为候选，也不使用 val/test 相关边计算排序。

### 2.1 Ratio = 0.05

三 seed 平均结果：

| selection | acc_drop | f1_drop | degree_kl | component_change | emb_l2 | member_l2 | neighbor_drift | MIA AUC | privacy | online time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| low_degree_train | -0.0029 | -0.0036 | 0.0358 | 599 | 0.0890 | 0.2608 | 0.1065 | 0.6371 | 0.7257 | 137.4s |
| random_train | -0.0035 | -0.0041 | 0.0650 | 850 | 0.0993 | 0.2896 | 0.1209 | 0.5762 | 0.8476 | 992.4s |
| hub_train | -0.0051 | -0.0064 | 0.1626 | 1616 | 0.1473 | 0.4662 | 0.1796 | 0.5094 | 0.9812 | 11023.5s |

`ratio=0.05` 下，结构破坏和表示漂移仍然呈现非常清楚的顺序：

```text
hub_train > random_train > low_degree_train
```

### 2.2 Ratio = 0.1

三 seed 平均结果：

| selection | acc_drop | f1_drop | degree_kl | component_change | emb_l2 | member_l2 | neighbor_drift | MIA AUC | privacy | online time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| low_degree_train | -0.0025 | -0.0031 | 0.0723 | 1197 | 0.0903 | 0.2557 | 0.1133 | 0.6174 | 0.7652 | 140.2s |
| random_train | -0.0034 | -0.0045 | 0.1178 | 1696 | 0.1198 | 0.2930 | 0.1449 | 0.5575 | 0.8850 | 1789.1s |
| hub_train | -0.0009 | -0.0036 | 0.4606 | 5815 | 0.1523 | 0.5128 | 0.1606 | 0.5775 | 0.8450 | 14240.1s |

`ratio=0.1` 下，hub 删除带来的结构破坏进一步放大，尤其是：

```text
hub_train component_change = 5815
random_train component_change = 1696
low_degree_train component_change = 1197
```

### 2.3 固定条件胜出次数

固定条件为同一组：

```text
ratio + seed
```

跨 6 组固定条件统计，结构破坏、表示漂移和运行成本仍然非常稳定：

```text
degree_kl:              hub > random > low, 6/6
clustering coefficient: hub > random > low, 6/6
component_change:       hub > random > low, 6/6
member_embedding_l2:    hub > random > low, 6/6
primary_anchor_drift:   hub > random > low, 6/6
secondary_anchor_drift: hub > random > low, 6/6
online_time:            hub > random > low, 6/6
```

`embedding_l2` 和 `neighbor_drift` 也基本符合这个顺序：

```text
embedding_l2:
  hub > random: 6/6
  random > low: 5/6
  full order hub > random > low: 5/6

neighbor_drift:
  hub > random: 6/6
  random > low: 5/6
  full order hub > random > low: 5/6
```

这说明 RQ1 在结构破坏、表示漂移和运行成本上有非常稳定的证据。

privacy 的方向仍然和结构破坏不同：

```text
MIA AUC 最高:       low_degree_train 5/6
privacy_score 最低: low_degree_train 5/6
```

也就是说，hub 删除虽然结构破坏最强，但强扰动和补边修复可能把 membership signal 打散，使 MIA AUC 反而更接近 0.5。low-degree 删除结构扰动小，反而保留了更强 residual membership signal。

### 2.4 Utility / 性能结论

从 downstream prediction utility 看，`accuracy_drop` 和 `f1_drop` 都很小，且没有稳定单调趋势。

跨 6 组固定条件统计：

```text
accuracy_drop 最大:
low_degree_train 2/6
random_train     1/6
hub_train        3/6

f1_drop 最大:
low_degree_train 2/6
random_train     1/6
hub_train        3/6
```

因此不能写成 hub 删除一定造成最大测试性能下降。更准确的说法是：

```text
HASI 在不同训练节点删除类型下都保持了稳定的测试集分类性能。
节点类型显著影响结构破坏、表示漂移和运行时间，
但对 accuracy/F1 的影响较弱，没有稳定的单调规律。
```

### 2.5 修复前后变化

node inpainting repair 后，RQ1 的核心结论没有根本变化，变化主要体现在 `random_train` 和 `hub_train` 的结构修复真正生效。

```text
low_degree_train:
  基本不变，因为没有触发 inpainting。

random_train:
  补边 256 条，degree KL 小幅下降，privacy 小幅变好。

hub_train:
  补边 256 条，degree KL 明显下降；
  ratio=0.05 下 privacy 明显变好；
  ratio=0.1 下 privacy 略变差。
```

运行成本会增加，尤其是 hub 删除：

```text
hub_train r0.05: 约 +89s
hub_train r0.1:  约 +228s
random_train:    约 +66s 到 +129s
```

### 2.6 RQ1 结论

RQ1 可以支持以下结论：

```text
不同训练节点删除类型会显著影响 graph unlearning 的难度。
Hub-node deletion 在结构和表示层面最难处理，random deletion 居中，low-degree deletion 最轻。
这个顺序在 node inpainting repair 后仍然稳定成立。
但是 membership privacy risk 与结构破坏程度不是简单同向关系；
low-degree deletion 造成的结构扰动较小，却可能留下更强的 membership signal。
```

适合论文中的英文表述：

```text
Hub-node deletion remains the most structurally and representationally
challenging unlearning setting after node inpainting repair, leading to the
largest degree distribution shift, component changes, embedding drift,
anchor drift, and online unlearning cost. However, privacy risk does not
follow the same monotonic order: low-degree deletion often leaves stronger
residual membership signals, while hub deletion can disrupt or repair away
membership separability. Across all deletion types, HASI maintains stable
downstream predictive utility.
```

---

## 3. RQ3: Anchor Stabilization Ablation

RQ3 目标是解释 HASI 内部 anchor 机制对 hub-neighbor 删除场景下的影响。这个实验不比较 baselines，而是固定 HASI 的其他模块，只改变 anchor 设置。

结果路径：

```text
forget sets: experiments/rq_forget_sets/pubmed/rq3_anchor_ablation/
results: results/rq/pubmed/rq3_anchor_ablation/hasi/
aggregate: results/rq/pubmed/rq3_anchor_ablation/hasi/aggregate_summary.*
pre-repair archive: results/rq/pubmed/rq3_anchor_ablation/hasi/pre_inpainting_repair_fix/
```

本节使用的是 node inpainting repair target 修复后的 RQ3 结果。修复后，三种 anchor 变体在三个 seed 上都真正触发了 full inpainting：

```text
triggered = true
status = ok
edges_added = 256
repair_budget = 256
```

因此本节结果比旧的 pre-repair RQ3 更适合作为最终 HASI 实现下的机制分析。

固定设置：

```text
dataset = pubmed
unlearning_type = node
selection = hub_neighbor_train
ratio = 0.05
seeds = 42, 123, 2024
base_model = results/shared_base/pubmed/seed{seed}
candidate_scope = train_mask_hub_neighbors
score_graph = train_subgraph
```

三个 anchor 变体：

```text
no_anchor:
  anchor_mode = none
  lambda1 = 0
  lambda2 = 0
  禁用整个 anchor manager / anchor loss / DAR anchor replacement

hier_anchor:
  anchor_mode = hierarchical
  lambda1 = 2.0
  lambda2 = 0.5
  主方法默认层级 anchor

strong_anchor:
  anchor_mode = hierarchical
  lambda1 = 5.0
  lambda2 = 1.0
  测试过强 anchor 的副作用
```

这个 RQ3 是干净的：三种 anchor 变体使用同一批 `hub_neighbor_train` forget sets，只改变 anchor 相关设置；forget nodes 来自训练子图中 hub 周围的训练节点，并且保留 hub/anchor 本身，因此更适合观察 anchor 对 hub-neighborhood 表示稳定性和隐私风险的影响。

### 3.1 平均结果

| variant | acc_drop | f1_drop | emb_l2 | member_l2 | neighbor_drift | primary_anchor_drift | secondary_anchor_drift | MIA AUC | privacy | online time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| no_anchor | 0.0057 | 0.0049 | 0.0711 | 0.3495 | 0.0901 | 0.1221 | 0.0944 | 0.5932 | 0.8136 | 144.4s |
| hier_anchor | 0.0063 | 0.0052 | 0.1381 | 0.3651 | 0.1499 | 0.2559 | 0.2060 | 0.6763 | 0.6473 | 1657.7s |
| strong_anchor | 0.0041 | 0.0034 | 0.2131 | 0.3975 | 0.2216 | 0.3677 | 0.3024 | 0.6926 | 0.6148 | 1698.0s |

### 3.2 固定 Seed 胜出次数

固定条件为同一个 seed 下的三种 anchor 变体比较：

```text
accuracy_drop 最小:
no_anchor     1/3
hier_anchor   1/3
strong_anchor 1/3

f1_drop 最小:
no_anchor     1/3
hier_anchor   1/3
strong_anchor 1/3

embedding_l2_mean 最小:
no_anchor 3/3

member_embedding_l2_mean 最小:
no_anchor 3/3

neighbor_drift_mean 最小:
no_anchor 3/3

primary_anchor_drift_mean 最小:
no_anchor 3/3

secondary_anchor_drift_mean 最小:
no_anchor 3/3

MIA AUC 最接近 0.5:
no_anchor 3/3

privacy_score 最高:
no_anchor 3/3

online runtime 最短:
no_anchor 3/3
```

### 3.3 RQ3 结论

RQ3 的修复后结果说明 anchor 机制确实影响 HASI 的 utility-drift-privacy trade-off，但它不是简单地“降低 drift”。

更准确的结论是：

```text
no_anchor:
  drift 最小，privacy 最好，runtime 最短。
  说明在 hub_neighbor_train 删除下，禁用 anchor manager 反而减少了表示变化和 MIA 风险。

hier_anchor:
  处于 no_anchor 和 strong_anchor 之间。
  但修复 node inpainting 后，它不再是稳定的 utility 最优配置。

strong_anchor:
  平均 utility 最好，但 drift 最大，MIA AUC 最高，privacy 最差。
  说明过强 anchor 可能换取部分 utility，但会放大表示漂移和 privacy 风险。
```

结构指标在 RQ3 中不适合作为主要解释变量，因为三种 anchor 变体删除的是同一批节点，并且使用同样的 full inpainting：

```text
degree_kl / component_change / clustering coefficient change 主要由相同 forget set 和相同 full inpainting 决定。
```

因此 RQ3 应重点讨论：

```text
utility
embedding drift
primary / secondary anchor drift
MIA privacy
runtime
```

论文中建议避免写成：

```text
Anchor stabilization reduces representation drift.
```

更合适的英文表述是：

```text
Anchor stabilization controls a utility-stability-privacy trade-off.
Disabling the anchor manager yields the lowest representation drift and
strongest privacy under hub-neighbor deletion. Increasing anchor strength
can improve downstream utility in some cases, but it amplifies anchor drift
and MIA risk. These results suggest that anchor strength should be calibrated
rather than maximized.
```

一句话总结：

```text
no_anchor 是 drift/privacy 最优配置；
strong_anchor 平均 utility 最好，但 drift/privacy 副作用最大；
hier_anchor 处于中间，但不是修复后 RQ3 的全面最优配置。
```

---

## 4. Node Unlearning

结果路径：

```text
HASI repairfix: results/pubmed_eval/default_main/hasi/node/inpainting_repair_fix
Baselines:      results/pubmed_eval/default_main/baselines/*/node
```

说明：本节已经更新为 node inpainting repair target 修复后的 HASI node 结果。旧 HASI node 结果仍保留在 `results/pubmed_eval/default_main/hasi/node/`，但不再作为本节主表口径。

### 4.1 固定条件逐组结果

逐组结果如下。每一组固定：`unlearning_type=node`、`ratio`、`base_seed`、`forget_seed`、`shared_base` 和 `forget_set`。

| ratio | base_seed | forget_seed | method | acc_drop | f1_drop | MIA AUC | privacy | degree KL | CC change | time |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 42 | 70042 | hasi_tuned_repairfix | -0.0053 | -0.0066 | 0.5750 | 0.8499 | 0.0572 | -0.0030 | 749.1s |
| 0.05 | 42 | 70042 | hasi_default_repairfix | -0.0051 | -0.0067 | 0.5678 | 0.8644 | 0.0572 | -0.0030 | 744.6s |
| 0.05 | 42 | 70042 | retrain | 0.0035 | 0.0015 | 0.5902 | 0.8196 | 0.0649 | -0.0033 | 3.0s |
| 0.05 | 42 | 70042 | grapheraser_bekm | 0.0494 | 0.0669 | 0.6242 | 0.7515 | 0.0649 | -0.0033 | 6.6s |
| 0.05 | 42 | 70042 | grapheraser_blpa | 0.0162 | 0.0138 | 0.6662 | 0.6676 | 0.0649 | -0.0033 | 6.6s |
| 0.05 | 123 | 70123 | hasi_tuned_repairfix | -0.0063 | -0.0073 | 0.5717 | 0.8567 | 0.0709 | -0.0031 | 1330.0s |
| 0.05 | 123 | 70123 | hasi_default_repairfix | -0.0051 | -0.0058 | 0.5669 | 0.8662 | 0.0709 | -0.0031 | 1307.3s |
| 0.05 | 123 | 70123 | retrain | 0.0010 | -0.0001 | 0.5792 | 0.8415 | 0.0729 | -0.0035 | 3.0s |
| 0.05 | 123 | 70123 | grapheraser_bekm | 0.0956 | 0.1592 | 0.6038 | 0.7924 | 0.0729 | -0.0035 | 6.5s |
| 0.05 | 123 | 70123 | grapheraser_blpa | 0.0170 | 0.0176 | 0.6650 | 0.6700 | 0.0729 | -0.0035 | 6.5s |
| 0.05 | 2024 | 72024 | hasi_tuned_repairfix | -0.0023 | -0.0036 | 0.5438 | 0.9124 | 0.0529 | -0.0031 | 633.8s |
| 0.05 | 2024 | 72024 | hasi_default_repairfix | -0.0013 | -0.0027 | 0.5359 | 0.9282 | 0.0529 | -0.0031 | 627.9s |
| 0.05 | 2024 | 72024 | retrain | 0.0013 | -0.0008 | 0.5807 | 0.8387 | 0.0607 | -0.0037 | 2.8s |
| 0.05 | 2024 | 72024 | grapheraser_bekm | 0.1075 | 0.1885 | 0.6131 | 0.7738 | 0.0607 | -0.0037 | 6.1s |
| 0.05 | 2024 | 72024 | grapheraser_blpa | 0.0122 | 0.0107 | 0.6700 | 0.6601 | 0.0607 | -0.0037 | 5.8s |
| 0.1 | 42 | 70042 | hasi_tuned_repairfix | -0.0061 | -0.0075 | 0.5604 | 0.8793 | 0.1173 | -0.0062 | 1105.8s |
| 0.1 | 42 | 70042 | hasi_default_repairfix | -0.0053 | -0.0066 | 0.5586 | 0.8828 | 0.1173 | -0.0062 | 1103.6s |
| 0.1 | 42 | 70042 | retrain | 0.0005 | -0.0013 | 0.5952 | 0.8096 | 0.1075 | -0.0066 | 2.8s |
| 0.1 | 42 | 70042 | grapheraser_bekm | 0.0441 | 0.0589 | 0.6220 | 0.7561 | 0.1075 | -0.0066 | 6.2s |
| 0.1 | 42 | 70042 | grapheraser_blpa | 0.0160 | 0.0134 | 0.6632 | 0.6737 | 0.1075 | -0.0066 | 6.1s |
| 0.1 | 123 | 70123 | hasi_tuned_repairfix | -0.0051 | -0.0058 | 0.6206 | 0.7587 | 0.1255 | -0.0067 | 1776.9s |
| 0.1 | 123 | 70123 | hasi_default_repairfix | -0.0035 | -0.0045 | 0.5561 | 0.8879 | 0.1255 | -0.0067 | 1781.5s |
| 0.1 | 123 | 70123 | retrain | 0.0033 | 0.0021 | 0.5872 | 0.8256 | 0.1289 | -0.0070 | 2.8s |
| 0.1 | 123 | 70123 | grapheraser_bekm | 0.0829 | 0.1323 | 0.5997 | 0.8006 | 0.1289 | -0.0070 | 6.1s |
| 0.1 | 123 | 70123 | grapheraser_blpa | 0.0122 | 0.0121 | 0.6532 | 0.6937 | 0.1289 | -0.0070 | 6.5s |
| 0.1 | 2024 | 72024 | hasi_tuned_repairfix | -0.0010 | -0.0026 | 0.5495 | 0.9011 | 0.1223 | -0.0062 | 1038.2s |
| 0.1 | 2024 | 72024 | hasi_default_repairfix | 0.0000 | -0.0012 | 0.5448 | 0.9103 | 0.1223 | -0.0062 | 1028.7s |
| 0.1 | 2024 | 72024 | retrain | 0.0018 | 0.0001 | 0.5999 | 0.8001 | 0.1176 | -0.0066 | 2.8s |
| 0.1 | 2024 | 72024 | grapheraser_bekm | 0.1184 | 0.2154 | 0.5957 | 0.8087 | 0.1176 | -0.0066 | 6.1s |
| 0.1 | 2024 | 72024 | grapheraser_blpa | 0.0139 | 0.0125 | 0.6497 | 0.7006 | 0.1176 | -0.0066 | 6.1s |

### 4.2 平均结果

| method | acc_drop | f1_drop | MIA AUC | privacy | degree KL | CC change | online time |
|---|---:|---:|---:|---:|---:|---:|---:|
| hasi_tuned_repairfix | -0.0044 | -0.0055 | 0.5702 | 0.8597 | 0.0910 | -0.0047 | 1105.6s |
| hasi_default_repairfix | -0.0034 | -0.0046 | 0.5550 | 0.8900 | 0.0910 | -0.0047 | 1098.9s |
| retrain | 0.0019 | 0.0002 | 0.5887 | 0.8225 | 0.0921 | -0.0051 | 2.9s |
| grapheraser_bekm | 0.0830 | 0.1369 | 0.6097 | 0.7805 | 0.0921 | -0.0051 | 6.3s |
| grapheraser_blpa | 0.0146 | 0.0133 | 0.6612 | 0.6776 | 0.0921 | -0.0051 | 6.3s |

### 4.3 固定条件胜出次数

固定条件为同一组：

```text
ratio + base_seed + forget_seed
```

胜出次数：

```text
test accuracy_drop 最小:
hasi_tuned_repairfix 6/6

test f1_drop 最小:
hasi_tuned_repairfix 5/6
hasi_default_repairfix 1/6

MIA AUC 最接近 0.5:
hasi_default_repairfix 6/6

privacy_score 最高:
hasi_default_repairfix 6/6

online runtime 最短:
retrain 6/6
```

### 4.4 结论

修复 node inpainting 后，PubMed node 的结论更稳：

```text
utility 最好: hasi_tuned_repairfix
privacy 最好: hasi_default_repairfix
```

相较旧 HASI node 结果：

```text
hasi_tuned_repairfix 的 utility、privacy、degree KL、CC change 基本全面优于旧 tuned；
hasi_default_repairfix 的 privacy 和结构指标小幅优于旧 default，utility 只有极小幅下降。
```

因此 PubMed node 最终主表建议使用：

```text
hasi_tuned_repairfix 作为 tuned HASI node 结果；
hasi_default_repairfix 用于说明 default 在 MIA/privacy 上更强。
```

## 5. Feature Unlearning

结果路径：

```text
results/pubmed_eval/default_main/hasi/feature
results/pubmed_eval/default_main/baselines/*/feature
```

### 5.1 平均结果

| method | acc_drop | f1_drop | MIA AUC | privacy | online time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | -0.0019 | -0.0025 | 0.6164 | 0.7672 | 137.4s |
| hasi_default | -0.0018 | -0.0024 | 0.6330 | 0.7340 | 134.5s |
| retrain | 0.0045 | 0.0035 | 0.6455 | 0.7090 | 2.8s |
| grapheraser_bekm | 0.0864 | 0.1434 | 0.5764 | 0.8473 | 6.4s |
| grapheraser_blpa | 0.0178 | 0.0169 | 0.5558 | 0.8885 | 6.3s |

### 5.2 固定条件胜出次数

```text
test accuracy_drop 最小:
hasi_tuned 4/6
hasi_default 2/6

test f1_drop 最小:
hasi_tuned 4/6
hasi_default 2/6

MIA AUC 最接近 0.5:
grapheraser_blpa 4/6
grapheraser_bekm 2/6

privacy_score 最高:
grapheraser_blpa 4/6
grapheraser_bekm 2/6

online runtime 最短:
retrain 6/6
```

### 5.3 结论

Feature 上，HASI 的 utility 保持最好，尤其是：

```text
hasi_tuned
```

`hasi_tuned` 相比 `hasi_default`：

```text
accuracy: 4/6
f1:       4/6
privacy:  3/6
time:     1/6
```

因此 feature 建议保留：

```text
hasi_tuned
```

但需要说明：

```text
GraphEraser 的 privacy 更强，但 utility drop 明显更大。
```

适合论文中的表述：

```text
On PubMed feature unlearning, HASI tuned achieves the best utility preservation,
while GraphEraser variants provide stronger MIA privacy at the cost of much larger utility degradation.
```

---

## 6. Edge Unlearning

结果路径：

```text
results/pubmed_eval/default_main/hasi/edge
results/pubmed_eval/default_main/baselines/*/edge
```

本轮 edge 中 HASI 同时比较了三种配置：

```text
hasi_edge_latest_tuned
hasi_edge_old_tuned
hasi_default
```

### 6.1 平均结果

| method | acc_drop | f1_drop | MIA AUC | privacy | online time |
|---|---:|---:|---:|---:|---:|
| hasi_edge_latest_tuned | -0.0035 | -0.0043 | 0.7017 | 0.5966 | 139.8s |
| hasi_edge_old_tuned | -0.0018 | -0.0026 | 0.7012 | 0.5976 | 139.9s |
| hasi_default | -0.0011 | -0.0019 | 0.7043 | 0.5913 | 140.8s |
| retrain | 0.0038 | 0.0028 | 0.6821 | 0.6358 | 3.0s |
| grapheraser_bekm | 0.0854 | 0.1435 | 0.6261 | 0.7479 | 6.7s |
| grapheraser_blpa | 0.0147 | 0.0140 | 0.6741 | 0.6519 | 6.7s |
| gif | 0.6646 | 0.7501 | 0.5311 | 0.9378 | 2.7s |

### 6.2 固定条件胜出次数

```text
test accuracy_drop 最小:
hasi_edge_latest_tuned 4/6
hasi_edge_old_tuned    2/6

test f1_drop 最小:
hasi_edge_latest_tuned 4/6
hasi_edge_old_tuned    2/6

MIA AUC 最接近 0.5:
gif 6/6

privacy_score 最高:
gif 6/6

online runtime 最短:
gif 6/6
```

HASI 内部比较：

```text
latest_tuned vs old_tuned:
accuracy 4/6
f1       4/6
privacy  3/6
time     3/6

latest_tuned vs default:
accuracy 5/6
f1       4/6
privacy  4/6
time     2/6

old_tuned vs default:
accuracy 4/6
f1       4/6
privacy  3/6
time     3/6
```

### 6.3 结论

Edge 上，如果只在 HASI 三个配置里选一个，建议暂时保留：

```text
hasi_edge_latest_tuned
```

原因是它的平均 utility 最好：

```text
latest_tuned acc_drop=-0.0035, f1_drop=-0.0043
old_tuned    acc_drop=-0.0018, f1_drop=-0.0026
default      acc_drop=-0.0011, f1_drop=-0.0019
```

privacy 方面，`old_tuned` 只比 `latest_tuned` 略好：

```text
latest_tuned MIA=0.7017, privacy=0.5966
old_tuned    MIA=0.7012, privacy=0.5976
default      MIA=0.7043, privacy=0.5913
```

这个差距不足以抵消 `latest_tuned` 的 utility 优势。

但 edge 仍然是当前最不理想的部分：

```text
HASI edge 的 MIA AUC 仍然约 0.70
GIF 的 privacy 很好，但 utility 几乎崩掉
GraphEraser privacy 比 HASI edge 更好，但 utility drop 明显更大
```

论文中可以写成：

```text
On edge unlearning, HASI preserves utility most effectively, but privacy remains the main limitation.
GIF achieves strong privacy but causes severe utility degradation.
```

---

## 7. 总体建议

### 7.1 暂时保留的 HASI 配置

| unlearning type | 建议保留配置 | 主要理由 |
|---|---|---|
| node | hasi_tuned | utility 最好，但 default privacy 更强 |
| feature | hasi_tuned | utility 最好，平均 privacy 也优于 default |
| edge | hasi_edge_latest_tuned | HASI 内部 utility 最好，privacy 与 old_tuned 差距很小 |

### 7.2 写论文时的核心说法

可以概括为：

```text
HASI 在 node / feature / edge 三类 unlearning 中整体表现出最强的 utility preservation。
在 node 上，default 配置的 privacy 更强；
在 feature 上，tuned 配置综合表现最好；
在 edge 上，HASI 的 utility 优势明显，但 privacy 仍是主要限制。
GraphEraser 在部分场景下具有更强 privacy，但通常伴随明显 utility degradation。
GIF 在 edge 上隐私最好，但 utility collapse 明显，不适合作为综合最优方法。
```

### 7.3 结果使用建议

最终论文主结果建议基于：

```text
results/pubmed_eval
```

而不是旧的：

```text
results/pubmed
```

原因是 `results/pubmed_eval` 使用的是 held-out forget requests：

```text
experiments/forget_sets_eval/pubmed
```

这能更清楚地区分：

```text
tuning forget requests
final evaluation forget requests
```

---

## 8. Cora Held-out Evaluation

本节整理 Cora 数据集上的 held-out final evaluation。当前已经删除 PubMed tuned transfer 到 Cora 的 HASI 结果，因此 Cora 主结果只比较：

```text
HASI:
  hasi_cora_tuned
  hasi_default

Baselines:
  retrain
  grapheraser_bekm
  grapheraser_blpa
  gif 仅用于 edge
```

实验口径：

```text
shared_base: results/shared_base/cora/seed42, seed123, seed2024
held-out forget sets: experiments/forget_sets_eval/cora
results: results/cora_eval
dataset: Cora
ratios: 0.05, 0.1
base_seed: 42, 123, 2024
forget_seed: 70042, 70123, 72024
```

评价口径：

```text
acc_drop / f1_drop 越小越好
MIA AUC 越接近 0.5 越好
privacy_score 越高越好
time 使用 online_wall_clock_seconds
```

### 8.1 Cora Node Unlearning

结果路径：

```text
HASI repairfix: results/cora_eval/default_main/hasi/node/inpainting_repair_fix
Baselines:      results/cora_eval/default_main/baselines/*/node
```

说明：本节已经更新为 node inpainting repair target 修复后的 HASI node 结果。旧 HASI node 结果仍保留在 `results/cora_eval/default_main/hasi/node/`，但不再作为本节主表口径。

平均结果：

| method | acc_drop | f1_drop | MIA AUC | privacy | degree KL | CC change | online time |
|---|---:|---:|---:|---:|---:|---:|---:|
| hasi_tuned_repairfix | -0.0065 | -0.0072 | 0.5715 | 0.8570 | 0.1657 | -0.0134 | 7.6s |
| hasi_default_repairfix | -0.0009 | -0.0013 | 0.5732 | 0.8536 | 0.1749 | -0.0150 | 8.0s |
| retrain | -0.0022 | -0.0035 | 0.5765 | 0.8469 | 0.1609 | -0.0157 | 2.2s |
| grapheraser_bekm | 0.2384 | 0.3298 | 0.5738 | 0.8524 | 0.1609 | -0.0157 | 5.1s |
| grapheraser_blpa | 0.4879 | 0.6492 | 0.5361 | 0.9279 | 0.1609 | -0.0157 | 5.6s |

逐组结果如下。每一组固定：`unlearning_type=node`、`ratio`、`base_seed`、`forget_seed`、`shared_base` 和 `forget_set`。

| ratio | base_seed | forget_seed | method | acc_drop | f1_drop | MIA AUC | privacy | degree KL | CC change | time |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 42 | 70042 | hasi_tuned_repairfix | -0.0130 | -0.0154 | 0.5943 | 0.8115 | 0.1436 | -0.0104 | 7.5s |
| 0.05 | 42 | 70042 | hasi_default_repairfix | -0.0074 | -0.0097 | 0.6002 | 0.7996 | 0.1921 | -0.0122 | 7.9s |
| 0.05 | 42 | 70042 | retrain | -0.0111 | -0.0147 | 0.5921 | 0.8157 | 0.1508 | -0.0123 | 1.8s |
| 0.05 | 42 | 70042 | grapheraser_bekm | 0.3043 | 0.4105 | 0.5868 | 0.8264 | 0.1508 | -0.0123 | 4.0s |
| 0.05 | 42 | 70042 | grapheraser_blpa | 0.4564 | 0.6033 | 0.5559 | 0.8883 | 0.1508 | -0.0123 | 4.5s |
| 0.05 | 123 | 70123 | hasi_tuned_repairfix | 0.0130 | 0.0176 | 0.5866 | 0.8267 | 0.1313 | -0.0086 | 3.5s |
| 0.05 | 123 | 70123 | hasi_default_repairfix | 0.0093 | 0.0141 | 0.5831 | 0.8337 | 0.1424 | -0.0101 | 3.9s |
| 0.05 | 123 | 70123 | retrain | 0.0130 | 0.0164 | 0.5379 | 0.9242 | 0.0980 | -0.0104 | 2.1s |
| 0.05 | 123 | 70123 | grapheraser_bekm | 0.2282 | 0.2823 | 0.5385 | 0.9230 | 0.0980 | -0.0104 | 5.0s |
| 0.05 | 123 | 70123 | grapheraser_blpa | 0.5288 | 0.7005 | 0.5104 | 0.9791 | 0.0980 | -0.0104 | 5.1s |
| 0.05 | 2024 | 72024 | hasi_tuned_repairfix | -0.0186 | -0.0203 | 0.5601 | 0.8797 | 0.1343 | -0.0059 | 7.3s |
| 0.05 | 2024 | 72024 | hasi_default_repairfix | -0.0167 | -0.0188 | 0.5556 | 0.8889 | 0.1424 | -0.0070 | 7.8s |
| 0.05 | 2024 | 72024 | retrain | -0.0056 | -0.0085 | 0.6002 | 0.7996 | 0.1322 | -0.0085 | 2.3s |
| 0.05 | 2024 | 72024 | grapheraser_bekm | 0.1280 | 0.1970 | 0.6156 | 0.7688 | 0.1322 | -0.0085 | 6.0s |
| 0.05 | 2024 | 72024 | grapheraser_blpa | 0.4935 | 0.6601 | 0.5542 | 0.8916 | 0.1322 | -0.0085 | 5.9s |
| 0.1 | 42 | 70042 | hasi_tuned_repairfix | -0.0148 | -0.0199 | 0.5785 | 0.8431 | 0.2085 | -0.0176 | 7.6s |
| 0.1 | 42 | 70042 | hasi_default_repairfix | -0.0074 | -0.0139 | 0.5813 | 0.8374 | 0.1954 | -0.0186 | 7.9s |
| 0.1 | 42 | 70042 | retrain | -0.0167 | -0.0217 | 0.6174 | 0.7652 | 0.2068 | -0.0193 | 2.3s |
| 0.1 | 42 | 70042 | grapheraser_bekm | 0.3302 | 0.4776 | 0.6014 | 0.7972 | 0.2068 | -0.0193 | 4.3s |
| 0.1 | 42 | 70042 | grapheraser_blpa | 0.4564 | 0.6011 | 0.5731 | 0.8539 | 0.2068 | -0.0193 | 4.9s |
| 0.1 | 123 | 70123 | hasi_tuned_repairfix | 0.0148 | 0.0191 | 0.5527 | 0.8946 | 0.2113 | -0.0247 | 7.9s |
| 0.1 | 123 | 70123 | hasi_default_repairfix | 0.0241 | 0.0262 | 0.5637 | 0.8725 | 0.1832 | -0.0287 | 8.3s |
| 0.1 | 123 | 70123 | retrain | 0.0186 | 0.0213 | 0.5257 | 0.9486 | 0.2068 | -0.0278 | 2.3s |
| 0.1 | 123 | 70123 | grapheraser_bekm | 0.2597 | 0.3246 | 0.5305 | 0.9390 | 0.2068 | -0.0278 | 5.6s |
| 0.1 | 123 | 70123 | grapheraser_blpa | 0.5250 | 0.6931 | 0.5036 | 0.9928 | 0.2068 | -0.0278 | 6.1s |
| 0.1 | 2024 | 72024 | hasi_tuned_repairfix | -0.0204 | -0.0246 | 0.5568 | 0.8864 | 0.1655 | -0.0133 | 11.5s |
| 0.1 | 2024 | 72024 | hasi_default_repairfix | -0.0074 | -0.0061 | 0.5553 | 0.8895 | 0.1939 | -0.0134 | 12.0s |
| 0.1 | 2024 | 72024 | retrain | -0.0111 | -0.0138 | 0.5858 | 0.8283 | 0.1707 | -0.0159 | 2.3s |
| 0.1 | 2024 | 72024 | grapheraser_bekm | 0.1800 | 0.2869 | 0.5701 | 0.8599 | 0.1707 | -0.0159 | 5.9s |
| 0.1 | 2024 | 72024 | grapheraser_blpa | 0.4675 | 0.6371 | 0.5193 | 0.9614 | 0.1707 | -0.0159 | 7.3s |

固定条件胜出次数：

```text
test accuracy_drop 最小:
hasi_tuned_repairfix 4/6
hasi_default_repairfix 1/6
retrain 1/6

test f1_drop 最小:
hasi_tuned_repairfix 4/6
hasi_default_repairfix 1/6
retrain 1/6

MIA AUC 最接近 0.5:
grapheraser_blpa 6/6

privacy_score 最高:
grapheraser_blpa 6/6

online runtime 最短:
retrain 6/6
```

结论：

```text
Cora node 上，hasi_tuned_repairfix 的 utility 最强；
hasi_default_repairfix 和 hasi_tuned_repairfix 的 privacy 接近；
GraphEraser-BLPA 的 privacy 最强，但 utility 损失非常大。
```

相较旧 HASI node 结果，repairfix 版本表现为：

```text
privacy 更好，CC change 更好；
utility 略有下降，degree KL 略有上升；
整体仍建议使用 hasi_tuned_repairfix 作为 Cora node 主 HASI 配置。
```

### 8.2 Cora Edge Unlearning

结果路径：

```text
results/cora_eval/default_main/hasi/edge
results/cora_eval/default_main/baselines/*/edge
```

平均结果：

| method | acc_drop | f1_drop | MIA AUC | privacy | online time |
|---|---:|---:|---:|---:|---:|
| hasi_cora_tuned | -0.0037 | -0.0042 | 0.5855 | 0.8289 | 3.97s |
| hasi_default | -0.0046 | -0.0054 | 0.6126 | 0.7747 | 4.48s |
| retrain | 0.0015 | 0.0013 | 0.5908 | 0.8184 | 2.21s |
| grapheraser_bekm | 0.2434 | 0.3405 | 0.5636 | 0.8728 | 4.88s |
| grapheraser_blpa | 0.4694 | 0.6211 | 0.5610 | 0.8779 | 5.28s |
| gif | 0.7464 | 0.8313 | 0.5817 | 0.8365 | 1.76s |

逐组结果如下。每一组固定：`unlearning_type=edge`、`ratio`、`base_seed`、`forget_seed`、`shared_base` 和 `forget_set`。

| ratio | base_seed | forget_seed | method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| 0.05 | 42 | 70042 | hasi_cora_tuned | -0.0056 | -0.0082 | 0.5554 | 0.8892 | 4.18s |
| 0.05 | 42 | 70042 | hasi_default | -0.0037 | -0.0060 | 0.5883 | 0.8234 | 4.80s |
| 0.05 | 42 | 70042 | retrain | -0.0037 | -0.0065 | 0.5741 | 0.8517 | 2.43s |
| 0.05 | 42 | 70042 | grapheraser_bekm | 0.3117 | 0.4463 | 0.5393 | 0.9213 | 4.88s |
| 0.05 | 42 | 70042 | grapheraser_blpa | 0.4471 | 0.5843 | 0.5397 | 0.9206 | 5.68s |
| 0.05 | 42 | 70042 | gif | 0.7273 | 0.8059 | 0.6192 | 0.7616 | 1.87s |
| 0.05 | 123 | 70123 | hasi_cora_tuned | 0.0093 | 0.0142 | 0.5952 | 0.8096 | 4.05s |
| 0.05 | 123 | 70123 | hasi_default | 0.0056 | 0.0087 | 0.5916 | 0.8169 | 4.71s |
| 0.05 | 123 | 70123 | retrain | 0.0148 | 0.0161 | 0.5705 | 0.8590 | 2.29s |
| 0.05 | 123 | 70123 | grapheraser_bekm | 0.2393 | 0.3068 | 0.5562 | 0.8877 | 5.65s |
| 0.05 | 123 | 70123 | grapheraser_blpa | 0.5083 | 0.6695 | 0.5635 | 0.8729 | 6.21s |
| 0.05 | 123 | 70123 | gif | 0.7681 | 0.8614 | 0.6020 | 0.7961 | 1.86s |
| 0.05 | 2024 | 72024 | hasi_cora_tuned | -0.0148 | -0.0174 | 0.5603 | 0.8795 | 3.65s |
| 0.05 | 2024 | 72024 | hasi_default | -0.0130 | -0.0133 | 0.6136 | 0.7729 | 4.06s |
| 0.05 | 2024 | 72024 | retrain | -0.0056 | -0.0047 | 0.5865 | 0.8270 | 1.98s |
| 0.05 | 2024 | 72024 | grapheraser_bekm | 0.1336 | 0.2035 | 0.5930 | 0.8141 | 4.86s |
| 0.05 | 2024 | 72024 | grapheraser_blpa | 0.4712 | 0.6214 | 0.5787 | 0.8425 | 4.70s |
| 0.05 | 2024 | 72024 | gif | 0.7440 | 0.8266 | 0.6309 | 0.7382 | 1.64s |
| 0.1 | 42 | 70042 | hasi_cora_tuned | -0.0056 | -0.0091 | 0.5885 | 0.8231 | 4.23s |
| 0.1 | 42 | 70042 | hasi_default | -0.0074 | -0.0121 | 0.6224 | 0.7551 | 4.57s |
| 0.1 | 42 | 70042 | retrain | -0.0056 | -0.0080 | 0.5974 | 0.8053 | 2.58s |
| 0.1 | 42 | 70042 | grapheraser_bekm | 0.3135 | 0.4496 | 0.5326 | 0.9347 | 4.85s |
| 0.1 | 42 | 70042 | grapheraser_blpa | 0.4267 | 0.5589 | 0.5348 | 0.9303 | 5.33s |
| 0.1 | 42 | 70042 | gif | 0.7273 | 0.8059 | 0.5398 | 0.9204 | 1.88s |
| 0.1 | 123 | 70123 | hasi_cora_tuned | 0.0111 | 0.0145 | 0.6192 | 0.7615 | 4.09s |
| 0.1 | 123 | 70123 | hasi_default | 0.0037 | 0.0041 | 0.6160 | 0.7680 | 4.72s |
| 0.1 | 123 | 70123 | retrain | 0.0130 | 0.0148 | 0.6050 | 0.7900 | 1.99s |
| 0.1 | 123 | 70123 | grapheraser_bekm | 0.2041 | 0.2580 | 0.5655 | 0.8691 | 4.32s |
| 0.1 | 123 | 70123 | grapheraser_blpa | 0.4935 | 0.6485 | 0.5686 | 0.8628 | 4.89s |
| 0.1 | 123 | 70123 | gif | 0.7681 | 0.8614 | 0.5361 | 0.9278 | 1.67s |
| 0.1 | 2024 | 72024 | hasi_cora_tuned | -0.0167 | -0.0189 | 0.5947 | 0.8107 | 3.60s |
| 0.1 | 2024 | 72024 | hasi_default | -0.0130 | -0.0138 | 0.6440 | 0.7120 | 4.03s |
| 0.1 | 2024 | 72024 | retrain | -0.0037 | -0.0036 | 0.6114 | 0.7773 | 1.98s |
| 0.1 | 2024 | 72024 | grapheraser_bekm | 0.2579 | 0.3791 | 0.5950 | 0.8099 | 4.70s |
| 0.1 | 2024 | 72024 | grapheraser_blpa | 0.4694 | 0.6438 | 0.5809 | 0.8383 | 4.87s |
| 0.1 | 2024 | 72024 | gif | 0.7440 | 0.8266 | 0.5624 | 0.8752 | 1.63s |

固定条件胜出次数：

```text
test accuracy_drop 最小:
hasi_cora_tuned 3/6
hasi_default    2/6
retrain         1/6

test f1_drop 最小:
hasi_cora_tuned 3/6
hasi_default    2/6
retrain         1/6

MIA AUC 最接近 0.5:
grapheraser_bekm 3/6
gif              2/6
hasi_cora_tuned  1/6

privacy_score 最高:
grapheraser_bekm 3/6
gif              2/6
hasi_cora_tuned  1/6

online runtime 最短:
gif 6/6
```

Edge 结论：

```text
Cora edge 上，hasi_cora_tuned 是 HASI 内部最均衡的配置。
HASI 的 utility 明显优于 GIF 和 GraphEraser。
GIF 和 GraphEraser-BEKM 在 privacy 上更强，但 GIF 出现严重 utility collapse，GraphEraser 的 utility drop 也明显偏大。
因此 Cora edge 的 HASI 主配置建议保留 hasi_cora_tuned。
```

### 8.3 Cora Feature Unlearning

结果路径：

```text
results/cora_eval/default_main/hasi/feature
results/cora_eval/default_main/baselines/*/feature
```

平均结果：

| method | acc_drop | f1_drop | MIA AUC | privacy | online time |
|---|---:|---:|---:|---:|---:|
| hasi_cora_tuned | -0.0056 | -0.0071 | 0.5423 | 0.9155 | 4.10s |
| hasi_default | -0.0043 | -0.0048 | 0.5531 | 0.8937 | 4.40s |
| retrain | -0.0028 | -0.0034 | 0.5743 | 0.8514 | 2.51s |
| grapheraser_bekm | 0.2415 | 0.3503 | 0.5239 | 0.9521 | 5.62s |
| grapheraser_blpa | 0.4784 | 0.6355 | 0.5571 | 0.8858 | 5.84s |

逐组结果如下。每一组固定：`unlearning_type=feature`、`ratio`、`base_seed`、`forget_seed`、`shared_base` 和 `forget_set`。

| ratio | base_seed | forget_seed | method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| 0.05 | 42 | 70042 | hasi_cora_tuned | -0.0074 | -0.0132 | 0.5436 | 0.9128 | 3.80s |
| 0.05 | 42 | 70042 | hasi_default | -0.0111 | -0.0172 | 0.5714 | 0.8573 | 3.86s |
| 0.05 | 42 | 70042 | retrain | -0.0093 | -0.0129 | 0.5614 | 0.8771 | 2.11s |
| 0.05 | 42 | 70042 | grapheraser_bekm | 0.3377 | 0.4938 | 0.5157 | 0.9687 | 4.34s |
| 0.05 | 42 | 70042 | grapheraser_blpa | 0.4286 | 0.5597 | 0.5571 | 0.8857 | 5.36s |
| 0.05 | 123 | 70123 | hasi_cora_tuned | 0.0074 | 0.0070 | 0.5836 | 0.8327 | 4.17s |
| 0.05 | 123 | 70123 | hasi_default | 0.0111 | 0.0162 | 0.5527 | 0.8946 | 5.61s |
| 0.05 | 123 | 70123 | retrain | 0.0111 | 0.0125 | 0.5959 | 0.8081 | 2.25s |
| 0.05 | 123 | 70123 | grapheraser_bekm | 0.2449 | 0.3521 | 0.5144 | 0.9711 | 5.77s |
| 0.05 | 123 | 70123 | grapheraser_blpa | 0.5195 | 0.6895 | 0.5721 | 0.8559 | 5.43s |
| 0.05 | 2024 | 72024 | hasi_cora_tuned | -0.0148 | -0.0163 | 0.5383 | 0.9235 | 4.17s |
| 0.05 | 2024 | 72024 | hasi_default | -0.0148 | -0.0165 | 0.5325 | 0.9351 | 4.34s |
| 0.05 | 2024 | 72024 | retrain | -0.0130 | -0.0148 | 0.5813 | 0.8373 | 2.31s |
| 0.05 | 2024 | 72024 | grapheraser_bekm | 0.1503 | 0.2321 | 0.5335 | 0.9330 | 5.56s |
| 0.05 | 2024 | 72024 | grapheraser_blpa | 0.5009 | 0.6775 | 0.5515 | 0.8970 | 5.72s |
| 0.1 | 42 | 70042 | hasi_cora_tuned | -0.0111 | -0.0182 | 0.5439 | 0.9121 | 4.01s |
| 0.1 | 42 | 70042 | hasi_default | -0.0148 | -0.0203 | 0.5936 | 0.8128 | 4.09s |
| 0.1 | 42 | 70042 | retrain | -0.0111 | -0.0154 | 0.5640 | 0.8720 | 3.60s |
| 0.1 | 42 | 70042 | grapheraser_bekm | 0.3154 | 0.4500 | 0.5119 | 0.9761 | 5.79s |
| 0.1 | 42 | 70042 | grapheraser_blpa | 0.4267 | 0.5590 | 0.5638 | 0.8723 | 6.94s |
| 0.1 | 123 | 70123 | hasi_cora_tuned | 0.0111 | 0.0159 | 0.5293 | 0.9413 | 4.21s |
| 0.1 | 123 | 70123 | hasi_default | 0.0223 | 0.0270 | 0.5168 | 0.9664 | 4.29s |
| 0.1 | 123 | 70123 | retrain | 0.0186 | 0.0228 | 0.5798 | 0.8404 | 2.27s |
| 0.1 | 123 | 70123 | grapheraser_bekm | 0.2338 | 0.3130 | 0.5021 | 0.9957 | 5.44s |
| 0.1 | 123 | 70123 | grapheraser_blpa | 0.5065 | 0.6693 | 0.5708 | 0.8583 | 5.90s |
| 0.1 | 2024 | 72024 | hasi_cora_tuned | -0.0186 | -0.0177 | 0.5148 | 0.9705 | 4.22s |
| 0.1 | 2024 | 72024 | hasi_default | -0.0186 | -0.0178 | 0.5519 | 0.8962 | 4.21s |
| 0.1 | 2024 | 72024 | retrain | -0.0130 | -0.0127 | 0.5633 | 0.8733 | 2.54s |
| 0.1 | 2024 | 72024 | grapheraser_bekm | 0.1670 | 0.2606 | 0.5660 | 0.8680 | 6.81s |
| 0.1 | 2024 | 72024 | grapheraser_blpa | 0.4879 | 0.6582 | 0.5273 | 0.9453 | 5.71s |

固定条件胜出次数：

```text
test accuracy_drop 最小:
hasi_default    3/6
hasi_cora_tuned 2/6
retrain         1/6

test f1_drop 最小:
hasi_default    3/6
hasi_cora_tuned 2/6
retrain         1/6

MIA AUC 最接近 0.5:
grapheraser_bekm 5/6
hasi_cora_tuned  1/6

privacy_score 最高:
grapheraser_bekm 5/6
hasi_cora_tuned  1/6

online runtime 最短:
retrain 6/6
```

Feature 结论：

```text
Cora feature 上，hasi_default 的 utility 胜出次数略多，但 hasi_cora_tuned 的 privacy 更稳定。
GraphEraser-BEKM 的 privacy 最强，但 utility drop 明显偏大。
如果只选择一个 Cora feature 的 HASI 主配置，可以保留 hasi_cora_tuned，理由是它在 utility 接近 default 的同时 privacy 更稳。
```

### 8.4 Cora 总体结论

Cora 上的结论可以概括为：

```text
Node:
  hasi_cora_tuned 是最合适的 HASI 主配置，utility 明显优于 baselines。

Edge:
  hasi_cora_tuned 是最合适的 HASI 主配置，HASI 保持 utility 的优势最明显。
  GIF / GraphEraser 在 privacy 上可能更强，但 utility 代价很大。

Feature:
  hasi_default 的 utility 胜出次数略多，hasi_cora_tuned 的 privacy 更稳。
  为保持 Cora 三类实验配置一致性，feature 也可以暂时保留 hasi_cora_tuned。
```

最终建议：

| dataset | unlearning type | 建议 HASI 配置 | 说明 |
|---|---|---|---|
| Cora | node | hasi_cora_tuned | utility 最稳定 |
| Cora | edge | hasi_cora_tuned | HASI 内部最均衡 |
| Cora | feature | hasi_cora_tuned | utility 接近 default，privacy 更稳 |

论文表述可以写成：

```text
On Cora, HASI consistently preserves downstream utility across node, edge,
and feature unlearning. GraphEraser and GIF occasionally achieve stronger
membership privacy, but this is accompanied by substantial utility degradation,
especially for edge and feature unlearning. Therefore, the Cora-tuned HASI
configuration is used as the main Cora configuration in the held-out evaluation.
```

