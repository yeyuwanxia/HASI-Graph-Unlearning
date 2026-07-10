# HASI vs Baselines 对比实验方法说明（2026-06-19）

本文档整理当前 HASI 与 baselines 对比实验的统一流程，包括数据集划分、shared base model 训练、forget sets 生成、最终评估方式和结果目录组织。

## 1. 实验目标

HASI vs baselines 的主实验目标是：在同一个数据集、同一个 split、同一个 base model、同一批 forget requests 下，比较不同 unlearning 方法的效果。

固定条件包括：

```text
dataset
unlearning_type: node / edge / feature
ratio: 0.05 / 0.1
base_seed: 42 / 123 / 2024
forget_seed: 70042 / 70123 / 72024
shared_base artifact
forget_set file
```

在每个固定条件下比较：

```text
HASI variants
retrain
grapheraser_bekm
grapheraser_blpa
gif, only for edge experiments
```

主要结果目录：

```text
results/pubmed_eval/default_main/
results/primekg-disease-gene-small_eval/default_main/
```

最终评估使用 held-out forget sets：

```text
experiments/forget_sets_eval/pubmed/
experiments/forget_sets_eval/primekg-disease-gene-small/
```

这些 forget sets 只用于最终评估，不用于 HASI 调参。

## 2. 数据集与划分

当前主对比实验使用两个医学相关数据集：

```text
PubMed
PrimeKG-DiseaseGene-Small
```

其中 PubMed 是 Planetoid biomedical citation graph，PrimeKG-DiseaseGene-Small 是从 PrimeKG 构造的 disease-gene/protein 医学知识图谱子图。两个数据集的原始划分来源不同：

| dataset | 原始划分 | 本文实验划分 |
|---|---|---|
| PubMed | Planetoid 默认节点划分：train=60，val=500，test=1000；其中 train 每类 20 个节点。 | 重新生成 60/20/20 class-stratified random split。 |
| PrimeKG-DiseaseGene-Small | PrimeKG 原始知识图谱没有官方节点分类 train/val/test 划分。 | 构造节点分类任务后生成 60/20/20 class-stratified random split。 |

因此，本文主对比实验不直接使用 PubMed 的原始小训练集划分；PrimeKG-DiseaseGene-Small 也不存在可直接沿用的官方节点分类划分。为了保证 shared base、forget sets 和 held-out evaluation 在两个数据集之间具有一致实验口径，本文对每个数据集重新做 class-stratified random split：

```text
train_ratio = 0.6
val_ratio   = 0.2
test_ratio  = 0.2
split seeds = 42, 123, 2024
```

每个 seed 对应一套独立的 train/val/test mask，也对应一个 shared base model。

| dataset | total nodes | features | classes | train | val | test |
|---|---:|---:|---:|---:|---:|---:|
| PubMed | 19717 | 500 | 3 | 11830 | 3943 | 3944 |
| PrimeKG-DiseaseGene-Small | 14673 | 8 | 2 | 8804 | 2934 | 2935 |

划分原则：

```text
train_mask: 用于训练 shared base GNN，也作为 node/edge forget target 的候选范围来源。
val_mask: 用于 tuning 阶段选择 HASI 超参数。
test_mask: 用于最终 HASI vs baselines 结果报告。
```

最终对比实验报告 test_mask 上的结果，不在 test_mask 上调参。

## 3. Shared Base Model 统一训练

所有 HASI 和 baselines 都从同一批 shared base artifacts 出发。

shared base 目录：

```text
results/shared_base/{dataset}/seed{base_seed}/
```

例如：

```text
results/shared_base/pubmed/seed42/
results/shared_base/primekg-disease-gene-small/seed2024/
```

每个目录包含：

```text
model_state.pt
embeddings.pt
logits.pt
metadata.json
```

### 3.1 Base GNN 配置

当前 shared base 使用统一 GCN 配置：

```text
model_type = GCN
hidden_channels = 64
num_layers = 2
dropout = 0.5
epochs = 300
lr = 0.01
weight_decay = 5e-4
split = stratified_random
training_graph = train_subgraph
```

### 3.2 训练图构造

shared base 不是在完整图上训练，而是在 train_mask 诱导出的训练子图上训练。

给定完整图：

```text
G_full = (V, E, X, y)
```

先划分节点：

```text
V_train, V_val, V_test
```

再构造训练子图：

```text
G_train = (V_train, E_train, X_train, y_train)
E_train = {(u, v) in E | u in V_train and v in V_train}
X_train = X[V_train]
y_train = y[V_train]
```

也就是说，base GNN 训练时只使用：

```text
train nodes
train-train edges
train node features
train node labels
```

val/test 节点和与 val/test 相关的边不参与 base model 训练。

### 3.3 Shared Base 训练图规模

| dataset | seed | train nodes | train-subgraph edges |
|---|---:|---:|---:|
| PubMed | 42 | 11830 | 31100 |
| PubMed | 123 | 11830 | 30952 |
| PubMed | 2024 | 11830 | 31920 |
| PrimeKG-DiseaseGene-Small | 42 | 8804 | 145514 |
| PrimeKG-DiseaseGene-Small | 123 | 8804 | 134364 |
| PrimeKG-DiseaseGene-Small | 2024 | 8804 | 136578 |

对应脚本：

```text
experiments/prepare_base_models.py
```

核心命令形式：

```bash
conda run -n graphunlearning python experiments/prepare_base_models.py \
  --datasets pubmed,primekg-disease-gene-small \
  --seeds 42,123,2024 \
  --data_root data/raw \
  --output_root results/shared_base \
  --model_type GCN \
  --hidden_channels 64 \
  --num_layers 2 \
  --dropout 0.5 \
  --train_epochs 300 \
  --lr 0.01 \
  --weight_decay 5e-4 \
  --split stratified_random \
  --train_ratio 0.6 \
  --val_ratio 0.2 \
  --test_ratio 0.2 \
  --training_graph train_subgraph
```

## 4. HASI 调参与最终评估分离

当前实验区分 tuning forget requests 和 final evaluation forget requests。

### 4.1 Tuning 阶段

HASI 超参数在 tuning 阶段选择。调参时：

```text
forget targets 从训练集 / 训练子图中生成
score 在 val_mask 上计算
不使用 final held-out forget sets
不在 test_mask 上选择超参数
```

调参结果保存到：

```text
results/tuning/{dataset}/...
```

最终被采用的 tuned 配置保存到：

```text
configs/tuned/by_dataset/{dataset}/{unlearning_type}.yaml
```

例如：

```text
configs/tuned/by_dataset/pubmed/node.yaml
configs/tuned/by_dataset/pubmed/edge.yaml
configs/tuned/by_dataset/pubmed/feature.yaml
configs/tuned/by_dataset/primekg-disease-gene-small/edge.yaml
configs/tuned/by_dataset/primekg-disease-gene-small/feature.yaml
```

### 4.2 Final Evaluation 阶段

最终 HASI vs baselines 对比使用独立 held-out forget sets：

```text
experiments/forget_sets_eval/{dataset}/
```

这些 forget sets 使用新的 forget_seed：

| base_seed | forget_seed |
|---:|---:|
| 42 | 70042 |
| 123 | 70123 |
| 2024 | 72024 |

这样可以避免在同一批 forget requests 上既调参又报告最终结果。

论文口径可以写成：

```text
Hyperparameters are selected on validation nodes using tuning forget requests.
Final comparisons are reported on test nodes using held-out forget requests.
```

## 5. Forget Sets 如何生成

最终评估 forget sets 由 shared_base split 派生。每个 forget set 都明确绑定：

```text
dataset
unlearning_type
ratio
base_seed
forget_seed
base_artifact_dir
selection
selection_scope
candidate_count
forget_count
```

manifest 路径：

```text
experiments/forget_sets_eval/{dataset}/manifest.json
```

### 5.1 Node Forget Sets

Node unlearning 的 final evaluation 使用：

```text
selection = random_train
selection_scope = shared_base train_mask
```

即：从对应 `results/shared_base/{dataset}/seed{base_seed}` 的 `train_mask` 中随机抽取节点。

文件命名：

```text
{dataset}_node_r{ratio_label}_random_train_base{base_seed}_fseed{forget_seed}.json
```

例如：

```text
experiments/forget_sets_eval/pubmed/pubmed_node_r0p05_random_train_base42_fseed70042.json
```

含义：

```text
dataset = pubmed
unlearning_type = node
ratio = 0.05
base_seed = 42
forget_seed = 70042
candidate_scope = train_mask
```

这样可以保证最终评估中被遗忘节点来自 base model 训练过的训练节点，而不是 val/test 节点。

### 5.2 Edge Forget Sets

Edge unlearning 的 final evaluation 使用：

```text
selection = random_all
selection_scope = train_subgraph_edges
```

这里的 `random_all` 是历史命名，实际候选范围不是完整图所有边，而是训练子图中的边：

```text
train_subgraph_edges = {(u, v) in E | u in train_mask and v in train_mask}
```

文件命名：

```text
{dataset}_edge_r{ratio_label}_random_all_base{base_seed}_fseed{forget_seed}.json
```

例如：

```text
experiments/forget_sets_eval/primekg-disease-gene-small/primekg-disease-gene-small_edge_r0p1_random_all_base123_fseed70123.json
```

这样可以保证被遗忘边来自 base model 训练图，而不是 val/test 相关边。

### 5.3 Feature Forget Sets

Feature unlearning 的 final evaluation 使用：

```text
selection = random_all
selection_scope = feature_dimensions
```

也就是从全部 feature dimensions 中随机选择需要遗忘的维度。

文件命名：

```text
{dataset}_feature_r{ratio_label}_random_all_base{base_seed}_fseed{forget_seed}.json
```

例如：

```text
experiments/forget_sets_eval/primekg-disease-gene-small/primekg-disease-gene-small_feature_r0p05_random_all_base2024_fseed72024.json
```

feature forget set 不是从节点中抽样，而是从特征维度中抽样。base model 训练阶段只使用训练节点上的这些特征；最终评估仍在 test_mask 上报告模型效果。

## 6. HASI vs Baselines 如何对比

每个固定条件下，所有方法使用同一个：

```text
shared_base artifact
forget_set file
ratio
base_seed
forget_seed
unlearning_type
```

然后分别运行 HASI 和 baselines，并保存 JSON 结果。

### 6.1 HASI

HASI 运行脚本：

```text
experiments/run_hasi.py
```

HASI 配置包括：

```text
configs/hasi_default.yaml
configs/tuned/by_dataset/{dataset}/{unlearning_type}.yaml
```

结果保存到：

```text
results/{dataset}_eval/default_main/hasi/{unlearning_type}/
```

当前 PubMed node inpainting repair 后的 HASI 新结果曾保存在：

```text
results/pubmed_eval/default_main/hasi/node/inpainting_repair_fix/
```

PrimeKG-DiseaseGene-Small 当前 edge/feature 已按 `default_main/hasi/{edge,feature}` 组织；node 仍以 round2 coarse 调参完成后的配置作为后续最终口径。

### 6.2 Retrain Baseline

Retrain baseline 表示收到删除请求后，在删除后的训练图上从头重新训练完整 GNN。

脚本：

```text
experiments/run_baselines.py --baseline retrain
```

结果目录：

```text
results/{dataset}_eval/default_main/baselines/retrain/{unlearning_type}/
```

### 6.3 GraphEraser Baselines

GraphEraser 当前包括两个分区版本：

```text
grapheraser_bekm
grapheraser_blpa
```

为了公平计时，GraphEraser 被拆成两阶段：

```text
offline preprocessing:
  partition graph
  train shard models
  save shard artifacts

online unlearning:
  load artifacts
  locate affected shards
  retrain affected shard models
  aggregate / predict / evaluate
```

GraphEraser artifacts 保存到：

```text
results/{dataset}_eval/default_main/baselines/grapheraser_bekm/{unlearning_type}/artifacts/seed{base_seed}/
results/{dataset}_eval/default_main/baselines/grapheraser_blpa/{unlearning_type}/artifacts/seed{base_seed}/
```

在线结果保存到：

```text
results/{dataset}_eval/default_main/baselines/grapheraser_bekm/{unlearning_type}/
results/{dataset}_eval/default_main/baselines/grapheraser_blpa/{unlearning_type}/
```

GraphEraser artifact 按以下维度区分：

```text
dataset
baseline method: bekm / blpa
unlearning_type
base_seed
```

不按 ratio 区分，因为 partition 和 shard models 与某次具体 ratio/forget request 无关。

### 6.4 GIF Baseline

GIF 当前主要用于 edge unlearning 对比。

结果目录：

```text
results/{dataset}_eval/default_main/baselines/gif/edge/
```

## 7. 评价指标

最终结果主要报告 test_mask 上的效用指标：

```text
accuracy_after
accuracy_drop
f1_macro_after
f1_macro_drop
```

隐私指标：

```text
weak_auc
medium_auc
strong_auc
overall_mia_auc
privacy_score
```

结构指标：

```text
degree_kl_divergence
clustering_coefficient_change
component_count_change
```

表示漂移指标：

```text
embedding_l2_mean
member_embedding_l2_mean
neighbor_drift_mean
primary_anchor_drift_mean
secondary_anchor_drift_mean
```

时间指标：

```text
unlearn_time_seconds
online_wall_clock_seconds
time_breakdown
```

论文主表建议使用：

```text
online_wall_clock_seconds
```

`unlearn_time_seconds` 保留为内部核心操作时间，用于 debug 或附录分析。

## 8. 公平性控制

当前对比实验的公平性来自以下控制：

1. 同一数据集、同一 seed 使用同一个 shared_base model。
2. shared_base 统一由 60/20/20 stratified split 生成。
3. shared_base 训练只使用 train_subgraph，val/test 不参与 base training。
4. 最终评估 forget sets 只从 shared_base 对应训练范围中生成：

```text
node: train_mask
edge: train_subgraph_edges
feature: feature_dimensions
```

5. HASI 和所有 baselines 使用同一个 forget_set_file。
6. HASI 调参和最终评估使用不同 forget requests。
7. 最终性能指标在 test_mask 上报告。
8. GraphEraser offline artifacts 与 online unlearning runtime 分开处理。
9. runtime 主口径使用脚本内部统一记录的 `online_wall_clock_seconds`。

## 9. 当前结果目录结构

PubMed 和 PrimeKG-DiseaseGene-Small 使用相同的主结果结构。以 PubMed 为例：

```text
results/pubmed_eval/default_main/
├── hasi/
│   ├── node/
│   ├── edge/
│   └── feature/
└── baselines/
    ├── retrain/
    │   ├── node/
    │   ├── edge/
    │   └── feature/
    ├── grapheraser_bekm/
    │   ├── node/
    │   │   └── artifacts/
    │   ├── edge/
    │   │   └── artifacts/
    │   └── feature/
    │       └── artifacts/
    ├── grapheraser_blpa/
    │   ├── node/
    │   │   └── artifacts/
    │   ├── edge/
    │   │   └── artifacts/
    │   └── feature/
    │       └── artifacts/
    └── gif/
        └── edge/
```

PrimeKG-DiseaseGene-Small 对应结构：

```text
results/primekg-disease-gene-small_eval/default_main/
```

## 10. 方法总结

当前 HASI vs baselines 的实验方法可以概括为：

```text
1. 对每个数据集按 seed 进行 60/20/20 stratified split。
2. 用 train_mask 诱导训练子图 G_train。
3. 在 G_train 上训练统一 GCN shared_base。
4. 基于 shared_base split 生成 held-out forget sets。
5. HASI 和 baselines 使用同一个 shared_base 和同一个 forget set 进行 unlearning。
6. 最终只在 test_mask 上报告结果。
7. tuning forget requests 和 final evaluation forget requests 分离。
```

一句话：

```text
The comparison is conducted under fixed shared base models and held-out forget requests, ensuring that HASI and all baselines are evaluated on the same training split, same trained model state, same unlearning targets, and same test-mask metrics.
```

## 11. PubMed Held-out Evaluation

本节补充 PubMed 在 held-out final evaluation 上的 node / edge / feature 结果。格式与 PrimeKG-DiseaseGene-Small 章节保持一致：按遗忘类型、ratio、base_seed 和 forget_seed 固定分组，组内比较 HASI 与 baselines。

结果路径：

```text
results/pubmed_eval/default_main/hasi/{node,edge,feature}
results/pubmed_eval/default_main/hasi/node/inpainting_repair_fix
results/pubmed_eval/default_main/baselines/*/{node,edge,feature}
```

固定设置：

```text
dataset = pubmed
ratios = 0.05, 0.1
base_seed = 42, 123, 2024
forget_seed = 70042, 70123, 72024
node selection_scope = shared_base train_mask
edge selection_scope = train_subgraph_edges
feature selection_scope = feature_dimensions
```

评价口径：

```text
acc_drop / f1_drop 越小越好
MIA AUC 越接近 0.5 越好
privacy_score 越高越好
time 使用 online_wall_clock_seconds
```

### 11.1 Node Unlearning

说明：node 使用 `results/pubmed_eval/default_main/hasi/node/inpainting_repair_fix/` 中的 repairfix 结果；旧的 `results/pubmed_eval/default_main/hasi/node/` 结果不作为本节主表口径。

逐组结果如下。每一组固定：`unlearning_type`、`ratio`、`base_seed`、`forget_seed`、`shared_base` 和 `forget_set`。

#### ratio=0.05, base_seed=42, forget_seed=70042

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned_repairfix | -0.0053 | -0.0066 | 0.5750 | 0.8499 | 749.1s |
| hasi_default_repairfix | -0.0051 | -0.0067 | 0.5678 | 0.8644 | 744.6s |
| retrain | 0.0035 | 0.0015 | 0.5902 | 0.8196 | 3.0s |
| grapheraser_bekm | 0.0494 | 0.0669 | 0.6242 | 0.7515 | 6.6s |
| grapheraser_blpa | 0.0162 | 0.0138 | 0.6662 | 0.6676 | 6.6s |

本组细化解读：<br>
utility：`hasi_tuned_repairfix` 在 accuracy_drop 上为本组最优；F1 最优为 `hasi_default_repairfix`。<br>
privacy：`hasi_tuned_repairfix` 不占优；MIA AUC 最接近 0.5 的是 `hasi_default_repairfix`，privacy_score 最高的是 `hasi_default_repairfix`。<br>
runtime：`hasi_tuned_repairfix` 在线时间为 749.1s，最短的是 `retrain`。<br>
HASI 内部：`hasi_tuned_repairfix` 相比 `hasi_default_repairfix` 改善了 acc_drop。<br>
组内取舍：PubMed node 的 tuned repairfix 主要提升 utility；privacy 往往需要和 `hasi_default_repairfix` 或 GraphEraser 对照解释。

#### ratio=0.05, base_seed=123, forget_seed=70123

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned_repairfix | -0.0063 | -0.0073 | 0.5717 | 0.8567 | 1330.0s |
| hasi_default_repairfix | -0.0051 | -0.0058 | 0.5669 | 0.8662 | 1307.3s |
| retrain | 0.0010 | -0.0001 | 0.5792 | 0.8415 | 3.0s |
| grapheraser_bekm | 0.0956 | 0.1592 | 0.6038 | 0.7924 | 6.5s |
| grapheraser_blpa | 0.0170 | 0.0176 | 0.6650 | 0.6700 | 6.5s |

本组细化解读：<br>
utility：`hasi_tuned_repairfix` 在 accuracy_drop, f1_drop 上为本组最优。<br>
privacy：`hasi_tuned_repairfix` 不占优；MIA AUC 最接近 0.5 的是 `hasi_default_repairfix`，privacy_score 最高的是 `hasi_default_repairfix`。<br>
runtime：`hasi_tuned_repairfix` 在线时间为 1330.0s，最短的是 `retrain`。<br>
HASI 内部：`hasi_tuned_repairfix` 相比 `hasi_default_repairfix` 改善了 acc_drop, f1_drop。<br>
组内取舍：PubMed node 的 tuned repairfix 主要提升 utility；privacy 往往需要和 `hasi_default_repairfix` 或 GraphEraser 对照解释。

#### ratio=0.05, base_seed=2024, forget_seed=72024

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned_repairfix | -0.0023 | -0.0036 | 0.5438 | 0.9124 | 633.8s |
| hasi_default_repairfix | -0.0013 | -0.0027 | 0.5359 | 0.9282 | 627.9s |
| retrain | 0.0013 | -0.0008 | 0.5807 | 0.8387 | 2.8s |
| grapheraser_bekm | 0.1075 | 0.1885 | 0.6131 | 0.7738 | 6.1s |
| grapheraser_blpa | 0.0122 | 0.0107 | 0.6700 | 0.6601 | 5.8s |

本组细化解读：<br>
utility：`hasi_tuned_repairfix` 在 accuracy_drop, f1_drop 上为本组最优。<br>
privacy：`hasi_tuned_repairfix` 不占优；MIA AUC 最接近 0.5 的是 `hasi_default_repairfix`，privacy_score 最高的是 `hasi_default_repairfix`。<br>
runtime：`hasi_tuned_repairfix` 在线时间为 633.8s，最短的是 `retrain`。<br>
HASI 内部：`hasi_tuned_repairfix` 相比 `hasi_default_repairfix` 改善了 acc_drop, f1_drop。<br>
组内取舍：PubMed node 的 tuned repairfix 主要提升 utility；privacy 往往需要和 `hasi_default_repairfix` 或 GraphEraser 对照解释。

#### ratio=0.1, base_seed=42, forget_seed=70042

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned_repairfix | -0.0061 | -0.0075 | 0.5604 | 0.8793 | 1105.8s |
| hasi_default_repairfix | -0.0053 | -0.0066 | 0.5586 | 0.8828 | 1103.6s |
| retrain | 0.0005 | -0.0013 | 0.5952 | 0.8096 | 2.8s |
| grapheraser_bekm | 0.0441 | 0.0589 | 0.6220 | 0.7561 | 6.2s |
| grapheraser_blpa | 0.0160 | 0.0134 | 0.6632 | 0.6737 | 6.1s |

本组细化解读：<br>
utility：`hasi_tuned_repairfix` 在 accuracy_drop, f1_drop 上为本组最优。<br>
privacy：`hasi_tuned_repairfix` 不占优；MIA AUC 最接近 0.5 的是 `hasi_default_repairfix`，privacy_score 最高的是 `hasi_default_repairfix`。<br>
runtime：`hasi_tuned_repairfix` 在线时间为 1105.8s，最短的是 `retrain`。<br>
HASI 内部：`hasi_tuned_repairfix` 相比 `hasi_default_repairfix` 改善了 acc_drop, f1_drop。<br>
组内取舍：PubMed node 的 tuned repairfix 主要提升 utility；privacy 往往需要和 `hasi_default_repairfix` 或 GraphEraser 对照解释。

#### ratio=0.1, base_seed=123, forget_seed=70123

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned_repairfix | -0.0051 | -0.0058 | 0.6206 | 0.7587 | 1776.9s |
| hasi_default_repairfix | -0.0035 | -0.0045 | 0.5561 | 0.8879 | 1781.5s |
| retrain | 0.0033 | 0.0021 | 0.5872 | 0.8256 | 2.8s |
| grapheraser_bekm | 0.0829 | 0.1323 | 0.5997 | 0.8006 | 6.1s |
| grapheraser_blpa | 0.0122 | 0.0121 | 0.6532 | 0.6937 | 6.5s |

本组细化解读：<br>
utility：`hasi_tuned_repairfix` 在 accuracy_drop, f1_drop 上为本组最优。<br>
privacy：`hasi_tuned_repairfix` 不占优；MIA AUC 最接近 0.5 的是 `hasi_default_repairfix`，privacy_score 最高的是 `hasi_default_repairfix`。<br>
runtime：`hasi_tuned_repairfix` 在线时间为 1776.9s，最短的是 `retrain`。<br>
HASI 内部：`hasi_tuned_repairfix` 相比 `hasi_default_repairfix` 改善了 acc_drop, f1_drop, time。<br>
组内取舍：PubMed node 的 tuned repairfix 主要提升 utility；privacy 往往需要和 `hasi_default_repairfix` 或 GraphEraser 对照解释。

#### ratio=0.1, base_seed=2024, forget_seed=72024

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned_repairfix | -0.0010 | -0.0026 | 0.5495 | 0.9011 | 1038.2s |
| hasi_default_repairfix | 0.0000 | -0.0012 | 0.5448 | 0.9103 | 1028.7s |
| retrain | 0.0018 | 0.0001 | 0.5999 | 0.8001 | 2.8s |
| grapheraser_bekm | 0.1184 | 0.2154 | 0.5957 | 0.8087 | 6.1s |
| grapheraser_blpa | 0.0139 | 0.0125 | 0.6497 | 0.7006 | 6.1s |

本组细化解读：<br>
utility：`hasi_tuned_repairfix` 在 accuracy_drop, f1_drop 上为本组最优。<br>
privacy：`hasi_tuned_repairfix` 不占优；MIA AUC 最接近 0.5 的是 `hasi_default_repairfix`，privacy_score 最高的是 `hasi_default_repairfix`。<br>
runtime：`hasi_tuned_repairfix` 在线时间为 1038.2s，最短的是 `retrain`。<br>
HASI 内部：`hasi_tuned_repairfix` 相比 `hasi_default_repairfix` 改善了 acc_drop, f1_drop。<br>
组内取舍：PubMed node 的 tuned repairfix 主要提升 utility；privacy 往往需要和 `hasi_default_repairfix` 或 GraphEraser 对照解释。

#### Node Unlearning 平均结果

| method | acc_drop | f1_drop | MIA AUC | privacy | online time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned_repairfix | -0.0044 | -0.0055 | 0.5702 | 0.8597 | 1105.6s |
| hasi_default_repairfix | -0.0034 | -0.0046 | 0.5550 | 0.8900 | 1098.9s |
| retrain | 0.0019 | 0.0002 | 0.5887 | 0.8225 | 2.9s |
| grapheraser_bekm | 0.0830 | 0.1369 | 0.6097 | 0.7805 | 6.3s |
| grapheraser_blpa | 0.0146 | 0.0133 | 0.6612 | 0.6776 | 6.3s |

#### Node Unlearning 固定条件胜出次数

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

hasi_tuned_repairfix 每组胜出指标数: 1/5, 2/5, 2/5, 2/5, 2/5, 2/5
hasi_tuned_repairfix 平均每组胜出指标数: 1.83/5
```

### 11.2 Edge Unlearning

说明：edge 使用 `hasi_edge_latest_tuned` 作为 PubMed edge 的 tuned HASI 主配置，同时保留 `hasi_default` 作为 HASI 内部对照。

逐组结果如下。每一组固定：`unlearning_type`、`ratio`、`base_seed`、`forget_seed`、`shared_base` 和 `forget_set`。

#### ratio=0.05, base_seed=42, forget_seed=70042

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_edge_latest_tuned | -0.0025 | -0.0026 | 0.6933 | 0.6134 | 141.6s |
| hasi_default | -0.0020 | -0.0028 | 0.7154 | 0.5692 | 139.4s |
| retrain | 0.0051 | 0.0036 | 0.6645 | 0.6711 | 3.0s |
| grapheraser_bekm | 0.0454 | 0.0612 | 0.6283 | 0.7434 | 6.8s |
| grapheraser_blpa | 0.0145 | 0.0125 | 0.6781 | 0.6438 | 6.8s |
| gif | 0.6585 | 0.7452 | 0.5199 | 0.9602 | 2.6s |

本组细化解读：<br>
utility：`hasi_edge_latest_tuned` 在 accuracy_drop 上为本组最优；F1 最优为 `hasi_default`。<br>
privacy：`hasi_edge_latest_tuned` 不占优；MIA AUC 最接近 0.5 的是 `gif`，privacy_score 最高的是 `gif`。<br>
runtime：`hasi_edge_latest_tuned` 在线时间为 141.6s，最短的是 `gif`。<br>
HASI 内部：`hasi_edge_latest_tuned` 相比 `hasi_default` 改善了 acc_drop, MIA, privacy。<br>
组内取舍：`gif` privacy 很强，但 acc_drop=0.6585, f1_drop=0.7452，utility 损失明显；`hasi_edge_latest_tuned` 的优势主要是保留 utility。

#### ratio=0.05, base_seed=123, forget_seed=70123

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_edge_latest_tuned | -0.0023 | -0.0029 | 0.6948 | 0.6103 | 141.1s |
| hasi_default | -0.0008 | -0.0016 | 0.7045 | 0.5910 | 139.1s |
| retrain | 0.0048 | 0.0042 | 0.6653 | 0.6694 | 2.9s |
| grapheraser_bekm | 0.0971 | 0.1676 | 0.6162 | 0.7675 | 6.5s |
| grapheraser_blpa | 0.0175 | 0.0182 | 0.6564 | 0.6872 | 6.3s |
| gif | 0.6691 | 0.7536 | 0.5028 | 0.9944 | 2.6s |

本组细化解读：<br>
utility：`hasi_edge_latest_tuned` 在 accuracy_drop, f1_drop 上为本组最优。<br>
privacy：`hasi_edge_latest_tuned` 不占优；MIA AUC 最接近 0.5 的是 `gif`，privacy_score 最高的是 `gif`。<br>
runtime：`hasi_edge_latest_tuned` 在线时间为 141.1s，最短的是 `gif`。<br>
HASI 内部：`hasi_edge_latest_tuned` 相比 `hasi_default` 改善了 acc_drop, f1_drop, MIA, privacy。<br>
组内取舍：`gif` privacy 很强，但 acc_drop=0.6691, f1_drop=0.7536，utility 损失明显；`hasi_edge_latest_tuned` 的优势主要是保留 utility。

#### ratio=0.05, base_seed=2024, forget_seed=72024

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_edge_latest_tuned | -0.0048 | -0.0062 | 0.6917 | 0.6166 | 139.1s |
| hasi_default | 0.0013 | 0.0005 | 0.6927 | 0.6145 | 138.6s |
| retrain | 0.0030 | 0.0022 | 0.6830 | 0.6340 | 3.0s |
| grapheraser_bekm | 0.1217 | 0.2230 | 0.6146 | 0.7708 | 6.4s |
| grapheraser_blpa | 0.0134 | 0.0123 | 0.6599 | 0.6802 | 6.3s |
| gif | 0.6663 | 0.7515 | 0.5231 | 0.9538 | 2.7s |

本组细化解读：<br>
utility：`hasi_edge_latest_tuned` 在 accuracy_drop, f1_drop 上为本组最优。<br>
privacy：`hasi_edge_latest_tuned` 不占优；MIA AUC 最接近 0.5 的是 `gif`，privacy_score 最高的是 `gif`。<br>
runtime：`hasi_edge_latest_tuned` 在线时间为 139.1s，最短的是 `gif`。<br>
HASI 内部：`hasi_edge_latest_tuned` 相比 `hasi_default` 改善了 acc_drop, f1_drop, MIA, privacy。<br>
组内取舍：`gif` privacy 很强，但 acc_drop=0.6663, f1_drop=0.7515，utility 损失明显；`hasi_edge_latest_tuned` 的优势主要是保留 utility。

#### ratio=0.1, base_seed=42, forget_seed=70042

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_edge_latest_tuned | -0.0041 | -0.0042 | 0.7066 | 0.5868 | 138.7s |
| hasi_default | -0.0048 | -0.0051 | 0.7028 | 0.5944 | 147.7s |
| retrain | 0.0023 | 0.0010 | 0.6904 | 0.6192 | 3.1s |
| grapheraser_bekm | 0.0436 | 0.0590 | 0.6446 | 0.7108 | 7.1s |
| grapheraser_blpa | 0.0142 | 0.0122 | 0.6927 | 0.6146 | 7.3s |
| gif | 0.6585 | 0.7452 | 0.5388 | 0.9224 | 2.9s |

本组细化解读：<br>
utility：`hasi_edge_latest_tuned` 不是本组最优；accuracy 最优为 `hasi_default`，F1 最优为 `hasi_default`。<br>
privacy：`hasi_edge_latest_tuned` 不占优；MIA AUC 最接近 0.5 的是 `gif`，privacy_score 最高的是 `gif`。<br>
runtime：`hasi_edge_latest_tuned` 在线时间为 138.7s，最短的是 `gif`。<br>
HASI 内部：`hasi_edge_latest_tuned` 相比 `hasi_default` 改善了 time。<br>
组内取舍：`gif` privacy 很强，但 acc_drop=0.6585, f1_drop=0.7452，utility 损失明显；`hasi_edge_latest_tuned` 的优势主要是保留 utility。

#### ratio=0.1, base_seed=123, forget_seed=70123

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_edge_latest_tuned | -0.0056 | -0.0063 | 0.7031 | 0.5937 | 138.8s |
| hasi_default | -0.0015 | -0.0025 | 0.7083 | 0.5834 | 138.7s |
| retrain | 0.0053 | 0.0050 | 0.6846 | 0.6308 | 3.0s |
| grapheraser_bekm | 0.0976 | 0.1654 | 0.6225 | 0.7549 | 6.4s |
| grapheraser_blpa | 0.0177 | 0.0185 | 0.6698 | 0.6605 | 6.4s |
| gif | 0.6691 | 0.7536 | 0.5427 | 0.9147 | 2.7s |

本组细化解读：<br>
utility：`hasi_edge_latest_tuned` 在 accuracy_drop, f1_drop 上为本组最优。<br>
privacy：`hasi_edge_latest_tuned` 不占优；MIA AUC 最接近 0.5 的是 `gif`，privacy_score 最高的是 `gif`。<br>
runtime：`hasi_edge_latest_tuned` 在线时间为 138.8s，最短的是 `gif`。<br>
HASI 内部：`hasi_edge_latest_tuned` 相比 `hasi_default` 改善了 acc_drop, f1_drop, MIA, privacy。<br>
组内取舍：`gif` privacy 很强，但 acc_drop=0.6691, f1_drop=0.7536，utility 损失明显；`hasi_edge_latest_tuned` 的优势主要是保留 utility。

#### ratio=0.1, base_seed=2024, forget_seed=72024

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_edge_latest_tuned | -0.0020 | -0.0033 | 0.7205 | 0.5591 | 139.3s |
| hasi_default | 0.0010 | 0.0004 | 0.7024 | 0.5952 | 141.1s |
| retrain | 0.0020 | 0.0006 | 0.7049 | 0.5902 | 3.1s |
| grapheraser_bekm | 0.1070 | 0.1849 | 0.6301 | 0.7398 | 6.9s |
| grapheraser_blpa | 0.0112 | 0.0101 | 0.6876 | 0.6248 | 6.8s |
| gif | 0.6663 | 0.7515 | 0.5594 | 0.8812 | 2.8s |

本组细化解读：<br>
utility：`hasi_edge_latest_tuned` 在 accuracy_drop, f1_drop 上为本组最优。<br>
privacy：`hasi_edge_latest_tuned` 不占优；MIA AUC 最接近 0.5 的是 `gif`，privacy_score 最高的是 `gif`。<br>
runtime：`hasi_edge_latest_tuned` 在线时间为 139.3s，最短的是 `gif`。<br>
HASI 内部：`hasi_edge_latest_tuned` 相比 `hasi_default` 改善了 acc_drop, f1_drop, time。<br>
组内取舍：`gif` privacy 很强，但 acc_drop=0.6663, f1_drop=0.7515，utility 损失明显；`hasi_edge_latest_tuned` 的优势主要是保留 utility。

#### Edge Unlearning 平均结果

| method | acc_drop | f1_drop | MIA AUC | privacy | online time |
|---|---:|---:|---:|---:|---:|
| hasi_edge_latest_tuned | -0.0035 | -0.0043 | 0.7017 | 0.5966 | 139.8s |
| hasi_default | -0.0011 | -0.0019 | 0.7043 | 0.5913 | 140.8s |
| retrain | 0.0038 | 0.0028 | 0.6821 | 0.6358 | 3.0s |
| grapheraser_bekm | 0.0854 | 0.1435 | 0.6261 | 0.7479 | 6.7s |
| grapheraser_blpa | 0.0147 | 0.0140 | 0.6741 | 0.6519 | 6.7s |
| gif | 0.6646 | 0.7501 | 0.5311 | 0.9378 | 2.7s |

#### Edge Unlearning 固定条件胜出次数

固定条件为同一组：

```text
ratio + base_seed + forget_seed
```

胜出次数：

```text
test accuracy_drop 最小:
hasi_edge_latest_tuned 5/6
hasi_default 1/6

test f1_drop 最小:
hasi_edge_latest_tuned 4/6
hasi_default 2/6

MIA AUC 最接近 0.5:
gif 6/6

privacy_score 最高:
gif 6/6

online runtime 最短:
gif 6/6

hasi_edge_latest_tuned 每组胜出指标数: 1/5, 2/5, 2/5, 0/5, 2/5, 2/5
hasi_edge_latest_tuned 平均每组胜出指标数: 1.50/5
```

### 11.3 Feature Unlearning

说明：feature 使用 `configs/tuned/by_dataset/pubmed/feature.yaml` 对应的 `hasi_tuned` 结果，同时保留 `hasi_default` 作为 HASI 内部对照。

逐组结果如下。每一组固定：`unlearning_type`、`ratio`、`base_seed`、`forget_seed`、`shared_base` 和 `forget_set`。

#### ratio=0.05, base_seed=42, forget_seed=70042

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | -0.0048 | -0.0053 | 0.6029 | 0.7943 | 145.4s |
| hasi_default | -0.0043 | -0.0052 | 0.6058 | 0.7885 | 135.9s |
| retrain | 0.0048 | 0.0038 | 0.6426 | 0.7149 | 2.8s |
| grapheraser_bekm | 0.0487 | 0.0656 | 0.5978 | 0.8044 | 6.5s |
| grapheraser_blpa | 0.0167 | 0.0151 | 0.5215 | 0.9570 | 6.5s |

本组细化解读：<br>
utility：`hasi_tuned` 在 accuracy_drop, f1_drop 上为本组最优。<br>
privacy：`hasi_tuned` 不占优；MIA AUC 最接近 0.5 的是 `grapheraser_blpa`，privacy_score 最高的是 `grapheraser_blpa`。<br>
runtime：`hasi_tuned` 在线时间为 145.4s，最短的是 `retrain`。<br>
HASI 内部：`hasi_tuned` 相比 `hasi_default` 改善了 acc_drop, f1_drop, MIA, privacy。<br>
组内取舍：PubMed feature 上 `hasi_tuned` 主要保持 utility；GraphEraser 往往 privacy 更强，但 utility 损失更明显。

#### ratio=0.05, base_seed=123, forget_seed=70123

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | -0.0033 | -0.0034 | 0.6338 | 0.7325 | 129.2s |
| hasi_default | -0.0030 | -0.0031 | 0.6292 | 0.7416 | 128.3s |
| retrain | 0.0030 | 0.0022 | 0.6427 | 0.7146 | 2.9s |
| grapheraser_bekm | 0.1040 | 0.1831 | 0.5993 | 0.8013 | 6.3s |
| grapheraser_blpa | 0.0170 | 0.0180 | 0.5477 | 0.9046 | 6.5s |

本组细化解读：<br>
utility：`hasi_tuned` 在 accuracy_drop, f1_drop 上为本组最优。<br>
privacy：`hasi_tuned` 不占优；MIA AUC 最接近 0.5 的是 `grapheraser_blpa`，privacy_score 最高的是 `grapheraser_blpa`。<br>
runtime：`hasi_tuned` 在线时间为 129.2s，最短的是 `retrain`。<br>
HASI 内部：`hasi_tuned` 相比 `hasi_default` 改善了 acc_drop, f1_drop。<br>
组内取舍：PubMed feature 上 `hasi_tuned` 主要保持 utility；GraphEraser 往往 privacy 更强，但 utility 损失更明显。

#### ratio=0.05, base_seed=2024, forget_seed=72024

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | 0.0010 | 0.0002 | 0.6158 | 0.7683 | 136.0s |
| hasi_default | -0.0025 | -0.0047 | 0.6919 | 0.6162 | 135.8s |
| retrain | 0.0020 | 0.0010 | 0.6496 | 0.7009 | 2.8s |
| grapheraser_bekm | 0.1113 | 0.1940 | 0.5280 | 0.9440 | 6.3s |
| grapheraser_blpa | 0.0122 | 0.0106 | 0.6190 | 0.7621 | 6.2s |

本组细化解读：<br>
utility：`hasi_tuned` 不是本组最优；accuracy 最优为 `hasi_default`，F1 最优为 `hasi_default`。<br>
privacy：`hasi_tuned` 不占优；MIA AUC 最接近 0.5 的是 `grapheraser_bekm`，privacy_score 最高的是 `grapheraser_bekm`。<br>
runtime：`hasi_tuned` 在线时间为 136.0s，最短的是 `retrain`。<br>
HASI 内部：`hasi_tuned` 相比 `hasi_default` 改善了 MIA, privacy。<br>
组内取舍：PubMed feature 上 `hasi_tuned` 主要保持 utility；GraphEraser 往往 privacy 更强，但 utility 损失更明显。

#### ratio=0.1, base_seed=42, forget_seed=70042

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | 0.0033 | 0.0031 | 0.6430 | 0.7141 | 135.0s |
| hasi_default | 0.0038 | 0.0043 | 0.6121 | 0.7758 | 131.9s |
| retrain | 0.0112 | 0.0099 | 0.6390 | 0.7219 | 2.7s |
| grapheraser_bekm | 0.0548 | 0.0741 | 0.5941 | 0.8118 | 6.2s |
| grapheraser_blpa | 0.0304 | 0.0282 | 0.5027 | 0.9947 | 6.0s |

本组细化解读：<br>
utility：`hasi_tuned` 在 accuracy_drop, f1_drop 上为本组最优。<br>
privacy：`hasi_tuned` 不占优；MIA AUC 最接近 0.5 的是 `grapheraser_blpa`，privacy_score 最高的是 `grapheraser_blpa`。<br>
runtime：`hasi_tuned` 在线时间为 135.0s，最短的是 `retrain`。<br>
HASI 内部：`hasi_tuned` 相比 `hasi_default` 改善了 acc_drop, f1_drop。<br>
组内取舍：PubMed feature 上 `hasi_tuned` 主要保持 utility；GraphEraser 往往 privacy 更强，但 utility 损失更明显。

#### ratio=0.1, base_seed=123, forget_seed=70123

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | -0.0053 | -0.0062 | 0.5757 | 0.8487 | 143.0s |
| hasi_default | -0.0018 | -0.0023 | 0.6431 | 0.7139 | 134.8s |
| retrain | 0.0035 | 0.0026 | 0.6199 | 0.7602 | 2.8s |
| grapheraser_bekm | 0.0958 | 0.1648 | 0.5925 | 0.8150 | 6.6s |
| grapheraser_blpa | 0.0165 | 0.0176 | 0.5357 | 0.9286 | 6.5s |

本组细化解读：<br>
utility：`hasi_tuned` 在 accuracy_drop, f1_drop 上为本组最优。<br>
privacy：`hasi_tuned` 不占优；MIA AUC 最接近 0.5 的是 `grapheraser_blpa`，privacy_score 最高的是 `grapheraser_blpa`。<br>
runtime：`hasi_tuned` 在线时间为 143.0s，最短的是 `retrain`。<br>
HASI 内部：`hasi_tuned` 相比 `hasi_default` 改善了 acc_drop, f1_drop, MIA, privacy。<br>
组内取舍：PubMed feature 上 `hasi_tuned` 主要保持 utility；GraphEraser 往往 privacy 更强，但 utility 损失更明显。

#### ratio=0.1, base_seed=2024, forget_seed=72024

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | -0.0023 | -0.0031 | 0.6272 | 0.7456 | 135.9s |
| hasi_default | -0.0030 | -0.0033 | 0.6159 | 0.7681 | 140.5s |
| retrain | 0.0025 | 0.0014 | 0.6794 | 0.6413 | 2.8s |
| grapheraser_bekm | 0.1040 | 0.1789 | 0.5464 | 0.9072 | 6.4s |
| grapheraser_blpa | 0.0139 | 0.0122 | 0.6080 | 0.7841 | 6.1s |

本组细化解读：<br>
utility：`hasi_tuned` 不是本组最优；accuracy 最优为 `hasi_default`，F1 最优为 `hasi_default`。<br>
privacy：`hasi_tuned` 不占优；MIA AUC 最接近 0.5 的是 `grapheraser_bekm`，privacy_score 最高的是 `grapheraser_bekm`。<br>
runtime：`hasi_tuned` 在线时间为 135.9s，最短的是 `retrain`。<br>
HASI 内部：`hasi_tuned` 相比 `hasi_default` 改善了 time。<br>
组内取舍：PubMed feature 上 `hasi_tuned` 主要保持 utility；GraphEraser 往往 privacy 更强，但 utility 损失更明显。

#### Feature Unlearning 平均结果

| method | acc_drop | f1_drop | MIA AUC | privacy | online time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | -0.0019 | -0.0025 | 0.6164 | 0.7672 | 137.4s |
| hasi_default | -0.0018 | -0.0024 | 0.6330 | 0.7340 | 134.5s |
| retrain | 0.0045 | 0.0035 | 0.6455 | 0.7090 | 2.8s |
| grapheraser_bekm | 0.0864 | 0.1434 | 0.5764 | 0.8473 | 6.4s |
| grapheraser_blpa | 0.0178 | 0.0169 | 0.5558 | 0.8885 | 6.3s |

#### Feature Unlearning 固定条件胜出次数

固定条件为同一组：

```text
ratio + base_seed + forget_seed
```

胜出次数：

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

hasi_tuned 每组胜出指标数: 2/5, 2/5, 0/5, 2/5, 2/5, 0/5
hasi_tuned 平均每组胜出指标数: 1.33/5
```

## 12. PrimeKG-DiseaseGene-Small Held-out Evaluation

本节补充 PrimeKG-DiseaseGene-Small 在 held-out final evaluation 上的 edge / feature 结果。node 仍在进行 round2 coarse 调参，因此本节暂不整理 node 主结果。

结果路径：

```text
results/primekg-disease-gene-small_eval/default_main/hasi/{edge,feature}
results/primekg-disease-gene-small_eval/default_main/baselines/*/{edge,feature}
```

固定设置：

```text
dataset = primekg-disease-gene-small
ratios = 0.05, 0.1
base_seed = 42, 123, 2024
forget_seed = 70042, 70123, 72024
edge selection_scope = train_subgraph_edges
feature selection_scope = feature_dimensions
```

评价口径：

```text
acc_drop / f1_drop 越小越好
MIA AUC 越接近 0.5 越好
privacy_score 越高越好
time 使用 online_wall_clock_seconds
```

### 12.1 Edge Unlearning

说明：edge 的 tuning 结果显示 default 已经是最好的 HASI 配置，因此这里将 `hasi_default` 作为当前选定的 `hasi_selected_default` 报告。

逐组结果如下。每一组固定：`unlearning_type`、`ratio`、`base_seed`、`forget_seed`、`shared_base` 和 `forget_set`。

#### ratio=0.05, base_seed=42, forget_seed=70042

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_selected_default | 0.0007 | 0.0007 | 0.5759 | 0.8482 | 506.6s |
| retrain | 0.0014 | 0.0015 | 0.5686 | 0.8628 | 12.2s |
| grapheraser_bekm | 0.0767 | 0.0863 | 0.6035 | 0.7930 | 17.5s |
| grapheraser_blpa | 0.0221 | 0.0240 | 0.5946 | 0.8108 | 19.2s |
| gif | 0.5993 | 0.7034 | 0.5265 | 0.9469 | 12.5s |

本组细化解读：<br>
utility：`hasi_selected_default` 在 accuracy_drop, f1_drop 上为本组最优。<br>
privacy：`hasi_selected_default` 不占优；MIA AUC 最接近 0.5 的是 `gif`，privacy_score 最高的是 `gif`。<br>
runtime：`hasi_selected_default` 在线时间为 506.6s，最短的是 `retrain`。<br>
组内取舍：`gif` privacy 很强，但 acc_drop=0.5993, f1_drop=0.7034，utility 明显崩坏；`hasi_selected_default` 的优势主要是保留 utility。

#### ratio=0.05, base_seed=123, forget_seed=70123

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_selected_default | -0.0031 | -0.0032 | 0.5850 | 0.8301 | 502.5s |
| retrain | 0.0003 | 0.0004 | 0.5633 | 0.8734 | 13.1s |
| grapheraser_bekm | 0.1608 | 0.1960 | 0.5894 | 0.8212 | 16.3s |
| grapheraser_blpa | 0.0239 | 0.0260 | 0.5954 | 0.8092 | 18.9s |
| gif | 0.6034 | 0.7078 | 0.5270 | 0.9460 | 11.5s |

本组细化解读：<br>
utility：`hasi_selected_default` 在 accuracy_drop, f1_drop 上为本组最优。<br>
privacy：`hasi_selected_default` 不占优；MIA AUC 最接近 0.5 的是 `gif`，privacy_score 最高的是 `gif`。<br>
runtime：`hasi_selected_default` 在线时间为 502.5s，最短的是 `gif`。<br>
组内取舍：`gif` privacy 很强，但 acc_drop=0.6034, f1_drop=0.7078，utility 明显崩坏；`hasi_selected_default` 的优势主要是保留 utility。

#### ratio=0.05, base_seed=2024, forget_seed=72024

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_selected_default | 0.0020 | 0.0022 | 0.5733 | 0.8534 | 500.5s |
| retrain | 0.0020 | 0.0022 | 0.5550 | 0.8901 | 13.0s |
| grapheraser_bekm | 0.1022 | 0.1168 | 0.5926 | 0.8149 | 16.6s |
| grapheraser_blpa | 0.0194 | 0.0210 | 0.5936 | 0.8128 | 18.7s |
| gif | 0.6061 | 0.7106 | 0.5250 | 0.9501 | 11.8s |

本组细化解读：<br>
utility：`hasi_selected_default` 在 accuracy_drop 上为本组最优。<br>
privacy：`hasi_selected_default` 不占优；MIA AUC 最接近 0.5 的是 `gif`，privacy_score 最高的是 `gif`。<br>
runtime：`hasi_selected_default` 在线时间为 500.5s，最短的是 `gif`。<br>
组内取舍：`gif` privacy 很强，但 acc_drop=0.6061, f1_drop=0.7106，utility 明显崩坏；`hasi_selected_default` 的优势主要是保留 utility。

#### ratio=0.1, base_seed=42, forget_seed=70042

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_selected_default | 0.0017 | 0.0018 | 0.5558 | 0.8883 | 487.6s |
| retrain | 0.0014 | 0.0014 | 0.5456 | 0.9089 | 11.6s |
| grapheraser_bekm | 0.0739 | 0.0830 | 0.5421 | 0.9159 | 17.6s |
| grapheraser_blpa | 0.0279 | 0.0304 | 0.5456 | 0.9089 | 18.9s |
| gif | 0.5993 | 0.7034 | 0.5389 | 0.9223 | 13.1s |

本组细化解读：<br>
utility：`hasi_selected_default` 不是本组最优；accuracy 最优为 `retrain`，F1 最优为 `retrain`。<br>
privacy：`hasi_selected_default` 不占优；MIA AUC 最接近 0.5 的是 `gif`，privacy_score 最高的是 `gif`。<br>
runtime：`hasi_selected_default` 在线时间为 487.6s，最短的是 `retrain`。<br>
组内取舍：`gif` privacy 很强，但 acc_drop=0.5993, f1_drop=0.7034，utility 明显崩坏；`hasi_selected_default` 的优势主要是保留 utility。

#### ratio=0.1, base_seed=123, forget_seed=70123

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_selected_default | -0.0007 | -0.0007 | 0.5651 | 0.8697 | 503.6s |
| retrain | 0.0010 | 0.0011 | 0.5401 | 0.9197 | 12.3s |
| grapheraser_bekm | 0.1806 | 0.2247 | 0.5437 | 0.9125 | 17.2s |
| grapheraser_blpa | 0.0198 | 0.0215 | 0.5559 | 0.8882 | 18.8s |
| gif | 0.6034 | 0.7078 | 0.5203 | 0.9593 | 12.4s |

本组细化解读：<br>
utility：`hasi_selected_default` 在 accuracy_drop, f1_drop 上为本组最优。<br>
privacy：`hasi_selected_default` 不占优；MIA AUC 最接近 0.5 的是 `gif`，privacy_score 最高的是 `gif`。<br>
runtime：`hasi_selected_default` 在线时间为 503.6s，最短的是 `retrain`。<br>
组内取舍：`gif` privacy 很强，但 acc_drop=0.6034, f1_drop=0.7078，utility 明显崩坏；`hasi_selected_default` 的优势主要是保留 utility。

#### ratio=0.1, base_seed=2024, forget_seed=72024

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_selected_default | 0.0024 | 0.0026 | 0.5390 | 0.9221 | 496.5s |
| retrain | 0.0034 | 0.0037 | 0.5281 | 0.9438 | 12.2s |
| grapheraser_bekm | 0.1131 | 0.1304 | 0.5498 | 0.9004 | 17.5s |
| grapheraser_blpa | 0.0198 | 0.0214 | 0.5511 | 0.8978 | 19.3s |
| gif | 0.6061 | 0.7106 | 0.5486 | 0.9029 | 12.3s |

本组细化解读：<br>
utility：`hasi_selected_default` 在 accuracy_drop, f1_drop 上为本组最优。<br>
privacy：`hasi_selected_default` 不占优；MIA AUC 最接近 0.5 的是 `retrain`，privacy_score 最高的是 `retrain`。<br>
runtime：`hasi_selected_default` 在线时间为 496.5s，最短的是 `retrain`。<br>
组内取舍：`gif` privacy 很强，但 acc_drop=0.6061, f1_drop=0.7106，utility 明显崩坏；`hasi_selected_default` 的优势主要是保留 utility。

#### Edge Unlearning 平均结果

| method | acc_drop | f1_drop | MIA AUC | privacy | online time |
|---|---:|---:|---:|---:|---:|
| hasi_selected_default | 0.0005 | 0.0006 | 0.5657 | 0.8686 | 499.5s |
| retrain | 0.0016 | 0.0017 | 0.5501 | 0.8998 | 12.4s |
| grapheraser_bekm | 0.1179 | 0.1395 | 0.5702 | 0.8596 | 17.1s |
| grapheraser_blpa | 0.0221 | 0.0241 | 0.5727 | 0.8546 | 19.0s |
| gif | 0.6030 | 0.7073 | 0.5310 | 0.9379 | 12.3s |

#### Edge Unlearning 固定条件胜出次数

固定条件为同一组：

```text
ratio + base_seed + forget_seed
```

胜出次数：

```text
test accuracy_drop 最小:
hasi_selected_default 5/6
retrain 2/6

test f1_drop 最小:
hasi_selected_default 4/6
retrain 2/6

MIA AUC 最接近 0.5:
gif 5/6
retrain 1/6

privacy_score 最高:
gif 5/6
retrain 1/6

online runtime 最短:
retrain 4/6
gif 2/6

tuned 每组胜出指标数: 2/5, 2/5, 1/5, 0/5, 2/5, 2/5
tuned 平均每组胜出指标数: 1.50/5
```

### 12.2 Feature Unlearning

说明：feature 使用 `configs/tuned/by_dataset/primekg-disease-gene-small/feature.yaml` 对应的 round2 `hasi_tuned` 结果，同时保留 `hasi_default` 作为 HASI 内部对照。

由于 PrimeKG-DiseaseGene-Small 只有 8 维 feature，`ratio=0.05` 和 `ratio=0.1` 在 held-out eval 中都只遗忘 1 个 feature 维度；因此同一个 seed 下两个 ratio 的结果高度接近。

逐组结果如下。每一组固定：`unlearning_type`、`ratio`、`base_seed`、`forget_seed`、`shared_base` 和 `forget_set`。

#### ratio=0.05, base_seed=42, forget_seed=70042

本组 feature target: `[5]`。

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | 0.0075 | 0.0077 | 0.5881 | 0.8239 | 406.1s |
| hasi_default | 0.0164 | 0.0171 | 0.6471 | 0.7059 | 487.3s |
| retrain | 0.0010 | 0.0011 | 0.5330 | 0.9340 | 11.3s |
| grapheraser_bekm | 0.0514 | 0.0558 | 0.7270 | 0.5461 | 18.5s |
| grapheraser_blpa | 0.0266 | 0.0288 | 0.6025 | 0.7950 | 19.6s |

本组细化解读：<br>
utility：`hasi_tuned` 不是本组最优；accuracy 最优为 `retrain`，F1 最优为 `retrain`。<br>
privacy：`hasi_tuned` 不占优；MIA AUC 最接近 0.5 的是 `retrain`，privacy_score 最高的是 `retrain`。<br>
runtime：`hasi_tuned` 在线时间为 406.1s，最短的是 `retrain`。<br>
HASI 内部：`hasi_tuned` 相比 `hasi_default` 改善了 acc_drop, f1_drop, MIA, privacy, time。<br>
组内取舍：`hasi_tuned` 主要体现为 HASI 内部对 default 的改善；但完整方法对比中，`retrain` 在该小 feature 任务上仍然更强。

#### ratio=0.05, base_seed=123, forget_seed=70123

本组 feature target: `[2]`。

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | 0.0007 | 0.0007 | 0.5480 | 0.9040 | 404.4s |
| hasi_default | -0.0003 | -0.0004 | 0.5376 | 0.9248 | 486.9s |
| retrain | 0.0017 | 0.0018 | 0.5316 | 0.9369 | 11.5s |
| grapheraser_bekm | 0.1489 | 0.1792 | 0.5582 | 0.8836 | 17.2s |
| grapheraser_blpa | 0.0150 | 0.0163 | 0.6238 | 0.7524 | 20.1s |

本组细化解读：<br>
utility：`hasi_tuned` 不是本组最优；accuracy 最优为 `hasi_default`，F1 最优为 `hasi_default`。<br>
privacy：`hasi_tuned` 不占优；MIA AUC 最接近 0.5 的是 `retrain`，privacy_score 最高的是 `retrain`。<br>
runtime：`hasi_tuned` 在线时间为 404.4s，最短的是 `retrain`。<br>
HASI 内部：`hasi_tuned` 相比 `hasi_default` 改善了 time。<br>
组内取舍：`hasi_tuned` 主要体现为 HASI 内部对 default 的改善；但完整方法对比中，`retrain` 在该小 feature 任务上仍然更强。

#### ratio=0.05, base_seed=2024, forget_seed=72024

本组 feature target: `[3]`。

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | 0.0051 | 0.0054 | 0.5824 | 0.8352 | 409.4s |
| hasi_default | 0.0072 | 0.0075 | 0.5846 | 0.8307 | 486.6s |
| retrain | 0.0027 | 0.0029 | 0.5155 | 0.9690 | 12.0s |
| grapheraser_bekm | 0.1101 | 0.1273 | 0.5952 | 0.8096 | 17.6s |
| grapheraser_blpa | 0.0194 | 0.0209 | 0.6088 | 0.7824 | 18.8s |

本组细化解读：<br>
utility：`hasi_tuned` 不是本组最优；accuracy 最优为 `retrain`，F1 最优为 `retrain`。<br>
privacy：`hasi_tuned` 不占优；MIA AUC 最接近 0.5 的是 `retrain`，privacy_score 最高的是 `retrain`。<br>
runtime：`hasi_tuned` 在线时间为 409.4s，最短的是 `retrain`。<br>
HASI 内部：`hasi_tuned` 相比 `hasi_default` 改善了 acc_drop, f1_drop, MIA, privacy, time。<br>
组内取舍：`hasi_tuned` 主要体现为 HASI 内部对 default 的改善；但完整方法对比中，`retrain` 在该小 feature 任务上仍然更强。

#### ratio=0.1, base_seed=42, forget_seed=70042

本组 feature target: `[5]`。

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | 0.0075 | 0.0077 | 0.5881 | 0.8238 | 399.4s |
| hasi_default | 0.0164 | 0.0171 | 0.6475 | 0.7050 | 483.3s |
| retrain | 0.0010 | 0.0011 | 0.5340 | 0.9320 | 11.6s |
| grapheraser_bekm | 0.0514 | 0.0558 | 0.7270 | 0.5461 | 18.8s |
| grapheraser_blpa | 0.0266 | 0.0288 | 0.6025 | 0.7950 | 17.5s |

本组细化解读：<br>
utility：`hasi_tuned` 不是本组最优；accuracy 最优为 `retrain`，F1 最优为 `retrain`。<br>
privacy：`hasi_tuned` 不占优；MIA AUC 最接近 0.5 的是 `retrain`，privacy_score 最高的是 `retrain`。<br>
runtime：`hasi_tuned` 在线时间为 399.4s，最短的是 `retrain`。<br>
HASI 内部：`hasi_tuned` 相比 `hasi_default` 改善了 acc_drop, f1_drop, MIA, privacy, time。<br>
组内取舍：`hasi_tuned` 主要体现为 HASI 内部对 default 的改善；但完整方法对比中，`retrain` 在该小 feature 任务上仍然更强。

#### ratio=0.1, base_seed=123, forget_seed=70123

本组 feature target: `[2]`。

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | 0.0007 | 0.0007 | 0.5480 | 0.9040 | 404.8s |
| hasi_default | -0.0003 | -0.0004 | 0.5376 | 0.9248 | 486.8s |
| retrain | 0.0017 | 0.0018 | 0.5316 | 0.9369 | 12.2s |
| grapheraser_bekm | 0.1489 | 0.1792 | 0.5582 | 0.8836 | 17.4s |
| grapheraser_blpa | 0.0150 | 0.0163 | 0.6238 | 0.7523 | 19.2s |

本组细化解读：<br>
utility：`hasi_tuned` 不是本组最优；accuracy 最优为 `hasi_default`，F1 最优为 `hasi_default`。<br>
privacy：`hasi_tuned` 不占优；MIA AUC 最接近 0.5 的是 `retrain`，privacy_score 最高的是 `retrain`。<br>
runtime：`hasi_tuned` 在线时间为 404.8s，最短的是 `retrain`。<br>
HASI 内部：`hasi_tuned` 相比 `hasi_default` 改善了 time。<br>
组内取舍：`hasi_tuned` 主要体现为 HASI 内部对 default 的改善；但完整方法对比中，`retrain` 在该小 feature 任务上仍然更强。

#### ratio=0.1, base_seed=2024, forget_seed=72024

本组 feature target: `[3]`。

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | 0.0051 | 0.0054 | 0.5824 | 0.8352 | 401.1s |
| hasi_default | 0.0072 | 0.0075 | 0.5846 | 0.8307 | 482.0s |
| retrain | 0.0027 | 0.0029 | 0.5138 | 0.9724 | 12.0s |
| grapheraser_bekm | 0.1101 | 0.1273 | 0.5952 | 0.8096 | 17.9s |
| grapheraser_blpa | 0.0194 | 0.0209 | 0.6088 | 0.7824 | 17.8s |

本组细化解读：<br>
utility：`hasi_tuned` 不是本组最优；accuracy 最优为 `retrain`，F1 最优为 `retrain`。<br>
privacy：`hasi_tuned` 不占优；MIA AUC 最接近 0.5 的是 `retrain`，privacy_score 最高的是 `retrain`。<br>
runtime：`hasi_tuned` 在线时间为 401.1s，最短的是 `retrain`。<br>
HASI 内部：`hasi_tuned` 相比 `hasi_default` 改善了 acc_drop, f1_drop, MIA, privacy, time。<br>
组内取舍：`hasi_tuned` 主要体现为 HASI 内部对 default 的改善；但完整方法对比中，`retrain` 在该小 feature 任务上仍然更强。

#### Feature Unlearning 平均结果

| method | acc_drop | f1_drop | MIA AUC | privacy | online time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | 0.0044 | 0.0046 | 0.5728 | 0.8544 | 404.2s |
| hasi_default | 0.0077 | 0.0081 | 0.5898 | 0.8203 | 485.5s |
| retrain | 0.0018 | 0.0019 | 0.5266 | 0.9469 | 11.8s |
| grapheraser_bekm | 0.1035 | 0.1208 | 0.6268 | 0.7464 | 17.9s |
| grapheraser_blpa | 0.0203 | 0.0220 | 0.6117 | 0.7766 | 18.8s |

#### Feature Unlearning 固定条件胜出次数

固定条件为同一组：

```text
ratio + base_seed + forget_seed
```

胜出次数：

```text
test accuracy_drop 最小:
retrain 4/6
hasi_default 2/6

test f1_drop 最小:
retrain 4/6
hasi_default 2/6

MIA AUC 最接近 0.5:
retrain 6/6

privacy_score 最高:
retrain 6/6

online runtime 最短:
retrain 6/6

hasi_tuned 每组胜出指标数: 0/5, 0/5, 0/5, 0/5, 0/5, 0/5
hasi_tuned 平均每组胜出指标数: 0.00/5
```

HASI 内部 tuned vs default 固定条件胜出次数：

```text
accuracy_drop: hasi_tuned 4/6
f1_drop:       hasi_tuned 4/6
MIA/privacy:   hasi_tuned 4/6
online time:   hasi_tuned 6/6
```

### 12.3 PrimeKG-DiseaseGene-Small edge / feature 结论

```text
edge:
  当前选定 HASI 配置为 default，也即 hasi_selected_default。
  它在 utility 上最稳，accuracy_drop 5/6 组胜出，f1_drop 4/6 组胜出。
  但 privacy 最强的是 GIF 或 retrain；GIF 的 utility 明显崩坏。
  因此 edge 不需要继续扩大 HASI 调参，正式评估保留 default-selected HASI。

feature:
  round2 tuned 在 held-out eval 上相比 HASI default 有平均改善：acc_drop 0.0077 -> 0.0044，f1_drop 0.0081 -> 0.0046，MIA AUC 0.5898 -> 0.5728，privacy 0.8203 -> 0.8544，online time 485.5s -> 404.2s。
  但相比所有方法，hasi_tuned 在 6 个固定条件组中没有拿到单项最优；retrain 在 utility、privacy 和 runtime 上都更强。
  HASI 内部看，tuned 相比 default 在 4/6 组上改善 utility / privacy，在 6/6 组上 runtime 更短。
  由于 PrimeKG-DiseaseGene-Small 只有 8 维 feature，ratio=0.05 和 ratio=0.1 都只遗忘 1 个 feature 维度，因此 feature forgetting 的 ratio 敏感性和解释力弱于 edge/node。
```

当前建议：

```text
edge:    使用 configs/tuned/by_dataset/primekg-disease-gene-small/edge.yaml，即 default-selected 配置。
feature: 可以保留 configs/tuned/by_dataset/primekg-disease-gene-small/feature.yaml，但论文中应说明 held-out 上提升不稳定，且 feature 维度太低。
node:    等 node round2 coarse 调参完成后再确定最终配置。
```
