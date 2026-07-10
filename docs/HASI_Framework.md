# HASI Method Reference

> 方法细节参考文档, 仅保留六大组件 · 三种遗忘类型流程 · 损失函数。
> Motivation, RQ, 实验设计, 创新点, 现状缺口请见 `PROJECT_OVERVIEW.md`。
> 最近一次同步: 2026-05-14 (对齐 commit `8a9f799`)。

---

## 目录

- [1. Method Overview](#1-method-overview)
- [2. 六大核心组件](#2-六大核心组件)
  - [2.1 Hub Identification (GHI)](#21-hub-identification-ghi)
  - [2.2 Anchor Stabilization (AS)](#22-anchor-stabilization-as)
  - [2.3 ERF-based Partitioning (EDP)](#23-erf-based-partitioning-edp)
  - [2.4 Generative Structural Inpainting (GSI)](#24-generative-structural-inpainting-gsi)
  - [2.5 Distributed Anchor Replacement (DAR)](#25-distributed-anchor-replacement-dar)
  - [2.6 MIA Privacy Evaluation](#26-mia-privacy-evaluation)
- [3. 三种遗忘类型流程](#3-三种遗忘类型流程)
  - [3.1 Node Unlearning](#31-node-unlearning)
  - [3.2 Edge Unlearning](#32-edge-unlearning)
  - [3.3 Feature Unlearning](#33-feature-unlearning)
- [4. 损失函数设计](#4-损失函数设计)
- [5. 默认超参数](#5-默认超参数)
- [6. 与代码模块的对应](#6-与代码模块的对应)

---

## 1. Method Overview

HASI 沿六个动词组织整个流水线:

```
识别 (Identify)    →  GHI: 找出结构关键 Hub
锚定 (Anchor)      →  AS:  分层约束 Hub 嵌入, 阻断误差级联
定位 (Localize)    →  EDP: PPR 派生功能性影响区域
修复 (Inpaint)     →  GSI: 条件触发生成式重建, 抹平结构异常
转移 (Relocate)    →  DAR: Primary 必须删除时, 把锚定职责分散到 k 个替代节点
验证 (Verify)      →  MIA: 三级攻击者闭环检查隐私
```

每个动词对应一个独立组件, 六个组件之间无循环依赖。
`HASIUnlearner` (`hasi/unlearner.py`) 是统一编排入口。

---

## 2. 六大核心组件

### 2.1 Hub Identification (GHI)

#### 目标

识别对网络结构与信息流至关重要的"枢纽节点"。

#### HubScore

```
HubScore(v) = α · Norm(GradientSensitivity(v))
            + β · Norm(CentralityScore(v))
            + γ · Norm(ERFInfluence(v))
```

默认权重: α=0.4, β=0.3, γ=0.3。三个子指标:

| 子指标 | 计算 | 优势 | 代价 |
|--------|------|------|------|
| GradientSensitivity | ‖∂L / ∂h_v‖, 多次前向取平均 | 任务相关, 最准确 | 需反向传播, 贵 |
| CentralityScore | 0.6 · PageRank + 0.3 · Degree + 0.1 · Eigenvector | 快速, 纯拓扑 | 忽略任务语义 |
| ERFInfluence | PPR 矩阵列求和, 节点被多少人依赖 | 捕获消息传递影响 | 需要 PPR 计算 |

#### Filter-and-Refine 加速

GradientSensitivity 对每个节点需要一次反向传播, 在 |V| > 100K 时不可行。两阶段优化:

```
Stage 1 (Filter): 只算 CentralityScore (O(|E|))
                  → 保留 top filter_ratio 的候选, 默认 10%
                  → 过滤掉 90% 的"显然不是 Hub"的节点

Stage 2 (Refine): 仅对候选算 GradientSensitivity 与 ERFInfluence
                  → 最终 HubScore 按上述加权
```

效率: 原始 O(n · backprop), 优化后 O(0.1n · backprop), 约 10x 加速。
Top 1% / 5% 的排序基本不受影响, 因为高 HubScore 节点的 Centrality 几乎必然在 Top 10% 内。

配置建议:

- `filter_ratio = 1.0`: 小图 (|V| < 5K, Cora / CiteSeer), 不过滤
- `filter_ratio = 0.1`: 中等规模 (PubMed)
- `filter_ratio = 0.05`: 大图 (Reddit, Ogbn-Arxiv)

#### 锚点分类

```
所有节点按 HubScore 降序排列:
  Top 0%-1%   → Primary Anchors   (一级锚点, 强约束)
  Top 1%-5%   → Secondary Anchors (二级锚点, 弱约束)
  其余         → Regular Nodes    (普通节点, 无约束)
```

- Primary: 受 λ₁ 强约束保护, 如需删除则触发 DAR (见 §2.5)
- Secondary: 受 λ₂ 弱约束, 可删除, 从邻居中选单点替代
- Regular: 自由删除

---

### 2.2 Anchor Stabilization (AS)

#### 目标

在遗忘微调阶段防止 Hub 节点的 embedding 漂移, 阻断误差级联传播。

#### 理论基础

灵感来自持续学习中的 Elastic Weight Consolidation (EWC)。EWC 约束参数, HASI 约束**节点表示**:

```
L_anchor = λ₁ · (1/|H₁|) · Σ_{v∈H₁} ‖h_v - h_v^orig‖²
         + λ₂ · (1/|H₂|) · Σ_{v∈H₂} ‖h_v - h_v^orig‖²
```

- H₁: Primary 集合, λ₁ = 2.0 (默认, 见 §5)
- H₂: Secondary 集合, λ₂ = 0.5 (默认)
- h_v^orig: 遗忘操作前存储的节点表示快照

#### 分层策略的意义

```
Primary    (λ₁ = 2.0)  embedding 几乎不动, 作为骨架
Secondary  (λ₂ = 0.5)  embedding 可以小幅调整, 作为缓冲
Regular    (λ  = 0)    embedding 自由更新, 适应新结构
```

从核心到边缘形成稳定性梯度, 避免"全图僵硬"和"全图混乱"两个极端。

注: 默认值由 commit `8a9f799` 从 (10.0, 1.0) 放松到 (2.0, 0.5),
目的是改善 acc-privacy 平衡。在效用敏感场景 (例如要求 acc drop < 2%) 可适当增大 λ₁ 至 5.0。

---

### 2.3 ERF-based Partitioning (EDP)

#### 目标

精确界定遗忘操作的影响范围, 只更新受影响节点。

#### 为什么不用 METIS / BEKM / BLPA

```
METIS / BEKM:  基于拓扑边切割, 语义相关节点被切到不同分片, Hub 被强制割裂
BLPA:          按标签传播切社区, Hub 跨社区桥梁被强制归入单一社区
ERF:           基于功能性依赖 (梯度流), 语义相关节点自然聚合, Hub 成为聚合中心
```

在 degree 方差极大的幂律图上, 任何依赖"平衡切割"的方案都会破坏 Hub 的完整性。
ERF 不切割, 而是按消息传递的影响半径动态聚合。

#### ERF 的定义与近似

节点 v 的有效感受野:

```
ERF_δ(v) = { u ∈ V : ‖∂h_v^(L) / ∂x_u‖ > δ }
```

直接计算需要逐节点反向传播。用 Personalized PageRank (PPR) 近似:

```
PPR(u; v) = α · Σ_{t=0}^∞ (1-α)^t · P^t(u, v)
```

其中 α = 0.15 是重启概率, P 是归一化邻接矩阵。

#### 影响范围计算

给定遗忘目标 D_f = {v₁, ..., v_k}:

```
R(D_f) = ∪_{v_i ∈ D_f} { u : PPR(u; v_i) > δ_PPR }
```

只在 R(D_f) 上微调, V \ R(D_f) 完全不动。

#### 复杂度

PPR Push 算法 O(m / δ), 远优于 METIS 的 O(m log n)。

#### Adaptive 子图微调

代码 (`hasi/unlearner.py`) 中实际执行的策略:

- `data.num_nodes > 5000` 且 `subgraph_finetune = true`: 在 R(D_f) 上微调
- 其余情况: 全图微调, 通过 anchor loss + ERF mask 控制更新范围

理由: 小图上子图微调的 overhead 比直接全图微调还大, 且失去 anchor loss 的全局信号。

---

### 2.4 Generative Structural Inpainting (GSI)

#### 目标

修复删除操作留下的局部拓扑空洞, 恢复图统计特性, 消除 MIA 可识别信号。

#### 为什么需要修复

```
删除节点 v 前:           删除后:
  A --- v --- B             A         B    ← 断裂
  |     |     |             |         |
  C --- D --- E             C --- D --- E

后果:
  1. A-B 失去连接, 信息流中断
  2. D 度数 4 → 2, 局部度分布偏移
  3. 聚类系数下降, 社区结构破损
  4. 结构异常成为 MIA 攻击者的可识别信号
```

#### 两种实现方案

**方案 A: Masked Graph Autoencoder (MGAE), 低成本**

```
编码器 (GCN):  G⁻ → Z (潜在表示)
解码器 (内积): Z → 预测邻居间缺失边的概率
阈值过滤:      prob > 0.5 的边被添加

训练: 随机 mask 15% 节点的边, 重建
推理: mask 被删节点的邻域, 重建连接
```

优点: 快速, 适合中小规模修复。
缺点: 生成质量有限, 难以捕获复杂分布。

**方案 B: Latent Graph Diffusion, 高精度**

```
1. 提取被删节点的 k 跳局部子图
2. 编码到潜在空间 (VQ-VAE)
3. 前向扩散, 加噪 T 步
4. 反向扩散, 以残缺图 + 锚点为条件, 逐步去噪
5. 解码回图空间
6. 合并回全图
```

优点: 生成质量高, 支持条件生成。
缺点: 计算代价大, 需要预训练扩散模型。

#### 触发条件 (不是每次都修复)

```
触发任一即修复:
  1. Hub-to-Hub 连接被删除 (两端皆为 Primary 或 Secondary)
  2. 局部聚类系数下降 > cc_drop_threshold (默认 0.30)
  3. 连通分量数量增加 (碎裂)

跳过条件:
  1. 删除节点全为 Regular, 不在 Hub 邻域
  2. 局部结构变化 < min_damage_ratio (默认 0.10)
```

#### inpainting_mode (commit `8a9f799` 新增)

| 模式 | 行为 | 用途 |
|------|------|------|
| `none` | 完全跳过 inpainting | 纯锚定消融 (w/o GSI) |
| `local_only` | 触发位置局部修复, 不预训练 inpainter | 快速迭代, 节省冷启动 |
| `full` | 标准流程, 预训练 + 条件触发 | 主实验默认 |

`lazy_train_inpainter = true` 时, inpainter 在首次触发 inpainting 时才训练, 进一步降低预处理开销。

---

### 2.5 Distributed Anchor Replacement (DAR)

> DAR 处理"Primary Anchor 必须删除" (例如 GDPR 合规) 的特殊场景。

#### 问题

Primary Anchor 被强约束保护, 但隐私法规可能强制要求删除。
若选择结构上靠近被删节点的单点替代 (single replace), 替代节点会突然受到强约束。
属性突变 (degree, embedding) 成为 MIA 攻击者的明显信号, 反而泄露"哪里删过东西"。

#### 核心思路

```
单点替代:     删除 Primary v → 选 1 个邻居 v' 强约束 (λ ≈ λ₁) → MIA 可识别突变
分散替代:     删除 Primary v → 选 k 个分散节点 → 每个弱约束 (λ₂ / k) → MIA 难以定位
```

总约束力 Σ (λ₂ / k) = λ₂, 等效于一个 Secondary 节点, 但每个替代节点不显眼。

#### 四种选择策略

**策略 A: `hubscore` (默认, 见 §5)**

```
直接选全图 HubScore 最高的 k 个候选
选择与被删节点位置无关, 隐私中等, 效用最佳, 复杂度 O(n log n)
```

**策略 B: `proximity_weighted` (效用优先)**

```
score(u) = HubScore(u) × (1 / distance(u, v))
选出节点聚集在 v 附近, 效用好, 隐私差, 用作对比基线
```

**策略 C: `privacy_constrained` (隐私优先)**

```
1. 过滤 candidates: 只保留 distance(u, v) ≥ min_distance 的节点
2. 在过滤集合中选 HubScore 最高的 k 个
简单有效, 无多样性保证
```

**策略 D: `distributed` (隐私-效用平衡)**

完整算法含三个机制。

##### 机制 1: 连通分量感知分配 (Component Coverage)

```
模拟删除 v 后的图 G'
检测连通分量 {C₁, C₂, ..., C_m}

若 m == 1 (无断裂): 正常走后续流程

若 m > 1 (断裂):
  → 过滤微小碎片: 忽略 |C_i| < θ (默认 θ=10) 的分量
    理由: 微小碎片节点极少, 在其内部分配锚点会让 MIA 攻击者轻松推断删除位置
    碎片内节点不施加锚定, 任其自由漂移
  → 在剩余有效分量中, 按节点数比例分配 k 个名额, 每个分量至少 1 个
  → 在每个分量内部独立执行后续选择

示例: k=5, C₁(800), C₂(150), C₃(50), C₄(3)
  → C₄ < θ, 忽略
  → 有效分量总计 1000 节点
  → C₁ 占 80% → 分 4 个, C₂ 占 15% → 分 1 个, C₃ 占 5% → 分 1 个 (向上取整)
  → 若有效分量数 > k, 优先分配给最大分量

θ 自适应建议: θ = max(10, 0.1% × |V|)
```

##### 机制 2: 贪心多样性 + Gumbel 噪声

```
候选集 C = {u : u ∉ Anchors ∪ Deleted, HubScore(u) > P70}
过滤   C = {u ∈ C : distance(u, v) ≥ min_distance}

归一化:
  hub_norm(u) = (HubScore(u) - min) / (max - min) ∈ [0, 1]
  div_norm(u) = (min_dist_to_selected(u) - min) / (max - min) ∈ [0, 1]

贪心循环 r = 1, ..., k:
  对每个候选 u:
    score(u) = α · hub_norm(u) + β · div_norm(u) + Gumbel(0, τ)
  选择 v'_r = argmax(score)
  从 C 中移除 distance(u, v'_r) < min_distance 的所有 u

默认: α=0.6, β=0.4, τ=0.1
```

Gumbel 噪声: argmax(score + Gumbel) 等价于 Gumbel-Softmax 采样, 提供 ε-DP 保证。
τ 控制随机性, τ↑ 更随机 (隐私好), τ↓ 更确定 (效用好)。

##### 机制 3: 最小距离约束 + Adaptive Decay

```
选完一个节点后, 将其 min_distance 跳内的所有候选移除
保证选出的 k 个两两分散, 不聚集

Adaptive Decay (候选枯竭时):
  current_min_dist = min_distance
  while |selected| < k:
    正常贪心选择 + min_distance 排除
  if |selected| < k:
    while |selected| < k and current_min_dist >= 1:
      current_min_dist -= 1
      恢复距离 > current_min_dist 的候选
      继续贪心

示例: min_distance=3, k=5, 选了 3 个后候选枯竭
  → 放宽到 2, 恢复部分候选, 继续选
  → 还不够 → 放宽到 1, 至少能选到非直接邻居

日志记录 (original_min_dist, final_min_dist), 用于分析:
频繁衰减说明 min_distance 默认对该图偏大。
```

#### 两阶段执行 (解决 Inpainting 时序悖论)

DAR 必须在删除前缓存距离 (删除后节点不存在, 无法 BFS), 同时希望基于最终拓扑做精选。两阶段:

```
Phase 1 (v 仍在图中):
  1. BFS 缓存 v 到所有候选的距离: deletion_context['candidate_distances']
  2. 缓存 v 的邻居集合: deletion_context['neighbors']
  3. 模拟删除, 检测连通分量, 应用机制 1 分配名额
  4. 按机制 2 + 3 在每个分量内预选 2k 候选

= = = = 物理删除 + Inpainting = = = =

Phase 2 (v 已删除, 修复完成):
  5. h_new ← model(G_repaired).detach().clone()   # 关键: detach + clone
  6. 从 2k 候选中精选 k 个 (按 Phase 1 的分量名额独立精选)
  7. 计算动态锚定目标:
     cached_dist(v'_i, v) == 1  → 锚定到 h_new[v'_i]   (邻居适应新结构, 隐私优先)
     cached_dist(v'_i, v) >= 2  → 锚定到 h_orig[v'_i]  (远端保持稳定, 效用优先)
```

#### 关键实现细节

**Tip 1: 距离缓存数据结构**

```python
deletion_context = {
    'deleted_node_id': v,
    'neighbors': set(G.neighbors(v)),
    'candidate_distances': nx.single_source_shortest_path_length(
        G, v, cutoff=max_search_radius
    ),
}
# 查询: deletion_context['candidate_distances'].get(u, float('inf'))
```

**Tip 2: h_new 快照时机**

物理删除 + Inpainting 之后, 微调开始之前, 一次性前向传播后立即 detach + clone:

```python
model.eval()
with torch.no_grad():
    full_embeddings = model(G_repaired.x, G_repaired.edge_index)
h_new_snapshot = full_embeddings.detach().clone()
register_distributed_anchors(anchor_nodes, h_new_snapshot, deletion_context)
model.train()
```

若不 detach, h_new 随参数更新而变动, ‖h - h_new‖² ≡ 0, 锚定信号失效 (移动靶问题)。

**Tip 3: 分量感知候选采样**

预选 2k 候选时, 必须按 Phase 1 的分量名额分别采样, 否则全局采样可能让某分量名额落空:

```python
candidates = []
for comp_id, quota in component_quotas.items():
    if quota > 0:
        comp_candidates = select_candidates_in_component(
            component=comp_id, k=2 * quota, strategy='distributed'
        )
        candidates.extend(comp_candidates)

# Phase 2 精选也按分量独立, 名额不跨分量挪用
```

---

### 2.6 MIA Privacy Evaluation

#### 目标

量化遗忘操作的隐私保护效果: 攻击者能否识别哪些节点被遗忘?

#### 三级攻击者

| 层级 | 知识假设 | 使用特征 | 现实威胁度 |
|------|---------|---------|-----------|
| Weak | 黑盒查询 | 预测置信度, 预测熵变化 | 最常见 |
| Medium | 持有同分布影子图, 可模拟遗忘 | Shadow Model 训练的迁移特征 | 较现实 |
| Strong | 拥有遗忘前后的模型参数 (白盒) | 10 维全特征 | 理论上界 |

#### Strong Attacker 的 10 维攻击特征

给定节点 u:

```
1.  confidence_change       = max(softmax_before) - max(softmax_after)
2.  abs_confidence_change   = |confidence_change|
3.  entropy_change          = entropy_after - entropy_before
4.  embedding_l2_drift      = ‖h_before - h_after‖₂
5.  embedding_cosine_dist   = 1 - cos(h_before, h_after)
6.  degree_change_abs       = deg_before - deg_after
7.  degree_change_relative  = (deg_before - deg_after) / deg_before
8.  neighbor_avg_drift      = mean(‖h_n^before - h_n^after‖ for n ∈ N(u))
9.  neighbor_max_drift      = max(‖h_n^before - h_n^after‖ for n ∈ N(u))
10. homophily_change        = |local_homophily_before - local_homophily_after|
```

#### Shadow Model 技术 (Medium Attacker)

```
1. 持有同分布图 G_shadow (从同数据集采样 50% 节点)
2. 在 G_shadow 上训练 Shadow GNN
3. 随机选节点, 模拟遗忘操作
4. 记录遗忘前后的特征变化, 构建 (特征, 标签) 数据集
5. 训练 MIA 分类器 (Logistic Regression / Random Forest)
6. 将分类器迁移攻击 Target Model
```

#### 评估指标

```
Per-tier accuracy:
  ≈ 0.50     完美隐私 (随机猜测)
  > 0.60     有隐私泄露
  > 0.80     严重泄露

overall_mia_auc  (主指标, 文献标准, 见 evaluation/mia/privacy_score.py):
  Strong 攻击者在 shuffled member / non-member 划分上的 AUC
  ≈ 0.5  → 完美防御
  → 1.0  → 完全泄露

privacy_score = 1.0 - 2 · |accuracy - 0.5|
  1.0 = 完美隐私
  0.0 = 零隐私
```

---

## 3. 三种遗忘类型流程

### 3.1 Node Unlearning

#### 情况 A: 删除 Regular 节点

```
1. 接收请求 delete(v), v ∈ Regular
2. EDP 计算 R(v) = {u : PPR(u; v) > δ}
3. 物理删除 v 及关联边
4. 条件触发 GSI 修复 v 的邻域
5. 微调  L = L_task(R(v)) + L_anchor(H₁, H₂) + forget_weight · L_forget({v})
6. 评估
```

#### 情况 B: 删除 Secondary 节点

```
1. 接收请求 delete(v), v ∈ Secondary
2. 从 v 的邻居中选 HubScore 最高的 u 作为单点替代
3. 更新 H₂ ← (H₂ \ {v}) ∪ {u}
4. 后续同情况 A
```

#### 情况 C: 删除 Primary 节点 (触发 DAR)

```
=== Phase 1: 预删除 (v 仍在图中, 唯一能算 distance(·, v) 的时机) ===
 1. 接收请求 delete(v), v ∈ Primary
 2. 缓存 neighbors(v) 和 v 到所有候选的距离 (DAR §2.5 Tip 1)
 3. 模拟删除, 检测连通分量 {C₁, ..., C_m}
 4. 按分量分配 k 个锚点名额 (DAR 机制 1):
    → 忽略 |C_i| < θ 的微小碎片
    → 剩余分量按节点数比例分配, 每个 >= 1
    → 微小碎片中节点不施加锚定
 5. 按 distributed 策略预选 2k 候选, 同时记录每个候选的缓存距离

=== Phase 2: 删除 + 修复 + 精选 ===
 6. 物理删除 v 及关联边
 7. 条件触发 GSI 修复拓扑
 8. h_new ← model(G_repaired).detach().clone()
 9. 基于 G_repaired, 在每个分量内独立从 2k 中精选最终 k 个

=== Phase 3: 锚定与微调 ===
10. 将 v 从 H₁ 移除, 将 k 个替代节点加入 H₂ (各 λ₂ / k 权重)
11. 注册分散锚点到 anchor_loss, 使用动态锚定目标:
    cached_dist == 1 → 锚定到 h_new (邻居, 隐私优先)
    cached_dist >= 2 → 锚定到 h_orig (远端, 效用优先)
12. 微调  L = L_task + L_anchor(H₁, H₂) + L_distributed + forget_weight · L_forget
13. MIA 评估 (三级攻击者)
```

关键不变量: Step 2 的距离缓存不可省略。Step 6 删除后 v 不在图中, 任何 distance(·, v) 调用会返回 ∞。

---

### 3.2 Edge Unlearning

```
1. 接收请求 delete(edge(u, w))
2. 计算受影响节点 {u, w} 及其 ERF
3. 物理删除指定边 (保留节点)
4. 条件触发 GSI (local 策略, 仅 Hub-Hub 边触发)
5. 微调  L = L_task + L_anchor + forget_weight · L_forget({u, w})
   forget_weight 通过 KL(logits[u] || logits_orig[u]) 推开端点表征
6. 评估
```

注: 设计文档曾包含一条 "若某 Primary Anchor 因删边失去 > 50% 连接则增加 k 个补充锚点"
的分支, 该机制目前未在代码实现 (`hasi/unlearner.py` 中无对应分支), 暂记为 future work。

---

### 3.3 Feature Unlearning

```
1. 接收请求 delete(features=[d₁, d₂, ...])
2. 在原始特征上算 logits_orig (forget target, 时序关键)
3. 对受影响节点的指定维度做 mask (置零或重采样均值)
4. 检查受影响节点是否含锚点:
   若是: 锚定目标改为 h_new (post-mask), 不能用 h_old
5. EDP 计算 affected_nodes (特征改变后表征漂移 > τ 的节点)
6. (可选) 训练特征 inpainter 补回被删维度的近似
7. 微调  L = L_task + L_anchor (锚定 h_new) + forget_weight · L_forget(affected_nodes)
8. 评估
```

#### 锚定目标选择 (关键设计决策)

```
锚定 h_old: 模型被迫通过邻居聚合"恢复"已删特征, Model Inversion 风险, 隐私泄露
锚定 h_new: 接受锚点自身性能下降, 防止下降级联扩散, 隐私安全
```

特征遗忘必须锚定 h_new。这与 Node Unlearning 情况 C 中"邻居锚定 h_new, 远端锚定 h_orig"
的策略相同 (邻居受拓扑直接冲击, 需适应新状态)。

时序不变量: logits_orig 必须在 mask 之前算。
若在 mask 后算, forget target 已包含被删信息, L_forget 信号失效。

---

## 4. 损失函数设计

### 4.1 完整损失函数

```
L_total = L_task + L_anchor + forget_weight · L_forget
        = L_task + (L_primary + L_secondary + L_distributed) + λ_f · L_forget
```

#### L_task (任务损失)

```
L_task = CrossEntropy(f_θ(x_R), y_R)
```

仅在 ERF 受影响区域 R 上计算 (大图 subgraph_finetune 模式) 或全图计算 (小图模式)。
执行模式由 `hasi/unlearner.py` 中 `adaptive_subgraph = subgraph_finetune AND num_nodes > 5000` 控制。

#### L_primary (一级锚点损失)

```
L_primary = λ₁ · (1 / |H₁|) · Σ_{v ∈ H₁} ‖h_v - h_v^orig‖²
```

强约束 (λ₁ = 2.0 默认), 维持网络骨架。

#### L_secondary (二级锚点损失)

```
L_secondary = λ₂ · (1 / |H₂|) · Σ_{v ∈ H₂} ‖h_v - h_v^orig‖²
```

弱约束 (λ₂ = 0.5 默认), 允许微小适应。

#### L_distributed (DAR 分散锚定损失)

```
L_distributed = Σ_{i=1}^{k} (λ₂ / k) · ‖h_{v'_i} - h_{v'_i}^{target}‖²
```

权重分摊: Σ (λ₂ / k) = λ₂, 无双重稀释, k 个分散节点的总约束力等效于一个 Secondary 锚点。

动态锚定目标:

```
h_{v'_i}^{target} =
  ├─ Snapshot(h_new)[v'_i]    if cached_dist(v'_i, v_deleted) == 1   (邻居)
  └─ h_{v'_i}^{orig}          if cached_dist(v'_i, v_deleted) >= 2   (远端)
```

Snapshot(h_new) 精确定义: 物理删除 + Inpainting 完成后, 参数 θ 更新前,
一次性前向传播产生的 detached tensor (常量, 不参与梯度计算)。

#### L_forget (遗忘损失, commit `8a9f799` 新增)

三种类型统一形式, 由 `forget_weight = λ_f` 控制 (默认 0.0, > 0 时启用):

```
node:    L_forget = KL ( softmax(f_θ(G_minus, V_forget)) || uniform )
edge:    L_forget = KL ( logits[u] || logits_orig[u] ) on endpoints
feature: L_forget = KL ( logits || logits_uniform ) on affected_nodes
```

设计理由: anchor loss 是"保住保留区域", forget loss 是"主动推开遗忘区域",
两者共同构成 push-pull 平衡。当 anchor loss 单独不足以让模型遗忘时 (例如 acc drop 大但 MIA 仍高), 启用 forget_weight 可提升隐私维度。

### 4.2 损失设计原则

| 决策 | 理由 |
|------|------|
| 锚定 embedding 而非参数 | 参数空间维度太高, embedding 直接表达节点状态 |
| 分层约束 λ₁ > λ₂ > λ₂/k | 从核心到边缘的稳定性梯度 |
| DAR 锚定到动态目标 (邻居 h_new, 远端 h_orig) | 邻居需适应新结构 (隐私), 远端保持稳定 (效用) |
| k 个分散节点总权重 = λ₂ | 集体等效于一个 Secondary, 但每个不显眼 |
| h_new 为 detached snapshot | 避免移动靶问题, 保证锚定信号有梯度 |
| L_task 在 R 上计算 (大图) 或全图 (小图) | 子图 overhead 与全图 anchor 信号的 trade-off |
| forget_weight 默认 0 | 主实验保持 push-pull 平衡, 隐私强化场景才开启 |

---

## 5. 默认超参数

| 类别 | 参数 | 范围 | 默认 | 作用 |
|------|------|------|------|------|
| Hub 识别 | `primary_ratio` | 0.005-0.02 | 0.01 | Hub 保护范围 |
| Hub 识别 | `secondary_ratio` | 0.03-0.10 | 0.05 | 缓冲带大小 |
| Hub 识别 | `filter_ratio` | 0.05-1.0 | 0.1 (大图) / 1.0 (小图) | Filter-and-Refine 粗筛比例 |
| 锚点 | `lambda1` (λ₁) | 1.0-10.0 | **2.0** | Primary 约束强度 |
| 锚点 | `lambda2` (λ₂) | 0.1-2.0 | **0.5** | Secondary 约束强度 |
| ERF | `alpha` | 0.1-0.25 | 0.15 | PPR 重启概率 |
| ERF | `threshold` (δ) | 0.005-0.03 | 0.01 | 影响范围阈值 |
| ERF | `k_steps` | 2-5 | 3 | PPR Power Iteration 步数 |
| Inpainting | `mode` | none / local_only / full | full (主实验) / none (消融) | inpainter 启用档位 |
| Inpainting | `method` | mgae / diffusion | mgae | 后端选择 |
| Inpainting | `cc_drop_threshold` | 0.2-0.5 | 0.30 | 触发阈值 |
| Inpainting | `min_damage_ratio` | 0.05-0.2 | 0.10 | 跳过阈值 |
| DAR | `enabled` | true / false | true | 是否启用 DAR |
| DAR | `k` | 1-10 | 5 | 分散锚点数 |
| DAR | `strategy` | hubscore / proximity_weighted / privacy_constrained / distributed | **hubscore** | 选择策略 |
| DAR | `min_distance` | 1-5 | 2 | 空间分散约束 |
| DAR | `gumbel_tau` (τ) | 0.01-0.5 | 0.1 | 选择随机性 |
| DAR | `small_component_threshold` (θ) | 5-20 | 10 | 微小碎片过滤 |
| DAR | `alpha_score` | 0.4-0.8 | 0.6 | 贪心策略中 hub_norm 权重 |
| DAR | `beta_score` | 0.2-0.6 | 0.4 | 贪心策略中 div_norm 权重 |
| 遗忘 | `forget_weight` (λ_f) | 0.0-1.0 | **0.0** | L_forget 权重, 默认关闭 |
| 微调 | `finetune_epochs` | 30-100 | 50 | 微调轮数 |
| 微调 | `finetune_lr` | 0.001-0.05 | 0.01 | 微调学习率 |
| 优化 | `subgraph_finetune` | true / false | true | 大图启用子图微调 |
| 优化 | `lazy_train_inpainter` | true / false | true | 延迟训练 inpainter |

加粗项为 commit `8a9f799` 引入或修改的默认值。

---

## 6. 与代码模块的对应

| 组件 | 代码位置 | 关键类 |
|------|---------|-------|
| GHI | `hasi/hub_identification/` | `HubScorer`, `CentralityHubIdentifier`, `GradientHubIdentifier`, `ERFHubIdentifier` |
| AS | `hasi/anchor_stabilization/` | `AnchorManager`, `AnchorStabilizationLoss`, `KnowledgeDistillationLoss` |
| EDP | `hasi/erf_partitioning/` | `PPRComputer`, `ERFCalculator`, `ERFPartitioner` |
| GSI | `hasi/structural_inpainting/` | `MaskedGraphAutoencoder`, `GraphDiffusionModel`, `StructuralInpainter` |
| DAR | `hasi/dar/` | `DARPipeline`, `DeletionContext`, `ComponentDetector`, `GumbelSelector`, `AdaptiveDecay`, `create_strategy()` |
| MIA | `evaluation/mia/` | `PrivacyEvaluator`, `WeakAttacker`, `MediumAttacker`, `StrongAttacker` |
| 主流水线 | `hasi/unlearner.py` | `HASIUnlearner` |
| 损失函数 | `hasi/anchor_stabilization/anchor_loss.py` + `hasi/unlearner.py` | `AnchorStabilizationLoss`, `_compute_forget_loss` |

---

## 附: Critical Invariants 速查

以下五条若违反会导致静默 bug (代码运行正常但结果错误):

1. **DAR 距离缓存必须在删除前** Phase 1 的 `nx.single_source_shortest_path_length(G, v)` 必须在 v 仍在图中时调用, 否则后续 cached_dist 全部为 ∞。
2. **h_new 必须 detach + clone** 否则 ‖h - h_new‖² 在反向传播时退化为 0, DAR 锚定信号失效。
3. **Feature unlearning 的 forget target 在 mask 前算** 否则 logits_orig 已包含被删特征信息, L_forget 信号失效。
4. **DAR 权重不双重稀释** k 个替代节点总权重 = λ₂, 每个 λ₂ / k。不要再除一次。
5. **Baseline MIA 评估传入 b.model 而非原始模型** `PrivacyEvaluator` 的 `model_after` 参数必须是 baseline 跑完的模型, 不是预训练模型。

更多上下文 (数据集 / Baselines / RQ / 实验进度 / 已知问题) 见 `PROJECT_OVERVIEW.md`。
