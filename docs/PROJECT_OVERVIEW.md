# HASI 项目整理 (Motivation · Method · Architecture)

> 投稿前内部参考文档。涵盖项目动机、方法细节、代码架构、实验设计与当前进度。
> 编写日期: 2026-05-14。最近一次代码 commit: `8a9f799` (2026-03-20)。

---

## 0. 项目一句话定位

**HASI (Hub-Anchored Structural Inpainting)** 把图遗忘从"破坏性擦除"重构为"结构感知的外科手术",
在删除图数据 (节点 / 边 / 特征) 时,
先锚定枢纽防止误差级联, 再对结构空洞做生成式修复, 并通过分散锚定替代 (DAR) 解决"必须删除枢纽"
的隐私-效用平衡难题, 配合三级 MIA 攻击者评估闭环验证遗忘质量。

主卖点 (按优先级):

1. **结构感知**: 通过 Hub 识别 + 分层锚定阻断幂律图上的误差级联 (回应 RQ1, RQ2, RQ3)。
2. **隐私机制**: DAR + 生成式修复将 Primary Hub 删除场景下的 MIA 攻击成功率压回随机猜测 (回应 RQ4, RQ5)。
3. **范围 (sub-claim)**: 同一个流程支持 node / edge / feature 三种遗忘类型 (回应 RQ6)。

> 之前的 framing 把"Universal"放在 title 主词位置, 导致和"隐私核心贡献"两个卖点互相稀释。
> 本文档建议把主卖点收敛到 1 + 2, "Universal" 降为子结论而非招牌词。

---

## 1. Motivation

### 1.1 任务定义

输入

- 原始图 G = (V, E, X), 其中 |V| = n, |E| = m, X ∈ R^{n × d}
- 训练好的 GNN: f_θ : G → Y
- 遗忘请求 D_f, 三选一:
  - 节点遗忘: D_f ⊆ V
  - 边遗忘: D_f ⊆ E
  - 特征遗忘: D_f ⊆ {1, ..., d}

输出

更新模型 f_{θ'} 需同时满足:

1. **遗忘性** f_{θ'} ≈ f_{θ*}, 其中 θ* 是在 G \ D_f 上从头重训的参数。
2. **效用性** 在 D_r = G \ D_f 上保持下游任务性能。
3. **效率性** Time(f_{θ'}) ≪ Time(Retrain)。
4. **隐私性** 三级 MIA 攻击成功率 ≈ 50% (随机猜测)。

ε-Unlearning 形式: 对任意输出 o, P(o | f_{θ'}) / P(o | f_{θ*}) ≤ e^ε。

### 1.2 核心挑战 (为什么图遗忘不能复用表格 / 图像遗忘)

图数据的特殊性: **节点之间存在结构依赖**。删除一个节点不仅影响它自己,
还通过 GNN 的消息传递机制影响所有 k 跳邻居。

幂律图 (社交网络 / 引文网络) 进一步加剧问题:

- Hub 节点 (度数极高) 删除后, 邻接矩阵 A 局部塌陷 → 表示空间整体偏移
- 现有方法忽视 Hub 的结构特殊性, 误差按 Hub 邻居半径扩散
- 删除后留下的"结构空洞"在度分布 / 聚类系数上留下可识别痕迹 → MIA 易攻

### 1.3 现有方法的统一缺陷

| 方法 | 来源 | 失效路径 |
|------|------|---------|
| GraphEraser-BEKM | CCS 2022 | 按边特征 K-Means 切割, Hub 边被分散到多簇, 跨簇全局上下文丢失 |
| GraphEraser-BLPA | CCS 2022 | 平衡约束迫使 Hub 归入单一社区, 其他社区失去跨社区桥梁 |
| GIF | WWW 2023 | 影响函数泰勒近似在非凸 GNN 损失上误差累积; 删除改变 A 后 H⁻¹ 失效 |
| GNNDelete | ICLR 2023 | 梯度参数更新忽略邻接矩阵变化, 梯度方向不准 |
| SGU / AGU | 2025 | 节点影响力 / 邻居优先级, 但无结构修复, 无 Hub 保护 |

共同盲区 (统一刻画)

```
删除 → 参数更新 → 完成
              ↓
        结构空洞 + Hub 受损
              ↓
   分布偏移 + 局部异质性 + MIA 漏洞
```

**关键观察 (Insight)**:

> 现有图遗忘工作把图当成"独立样本的集合"处理。
> 在幂律图上, **Hub 不可以被当作普通节点对待**,
> **删除留下的结构空洞必须主动修复**, 否则误差级联和隐私痕迹都无法消除。

HASI 围绕这两条观察展开。

---

## 2. Method

### 2.1 设计原则: 四字箴言 + Primary 删除特例

```
识别 (Identify)    →  GHI: 找出结构关键 Hub
锚定 (Anchor)      →  AS:  分层约束 Hub 嵌入, 阻断误差级联
定位 (Localize)    →  EDP: PPR 派生功能性影响区域, 替代拓扑切割
修复 (Inpaint)     →  GSI: 条件触发生成式重建, 抹平结构异常
转移 (Relocate)    →  DAR: Primary 必须删除时, 把"锚的功能"分散到 k 个替代节点
验证 (Verify)      →  MIA: 三级攻击者闭环检查隐私
```

注: 原 framework 文档用"识别-锚定-修复-验证"四字, 但 EDP 和 DAR 都没装进去。
本文档显式列出 6 个动词与组件一一对应, 消除"6 大组件 vs 4 字箴言"的映射歧义。

### 2.2 整体流程 (Node Unlearning)

```
Input: trained model f_θ + graph G + forget nodes D_f
  ─── Preprocessing (一次性) ──────────────────────
  1. GHI: Filter-and-Refine 计算 HubScore
  2. AnchorManager: 分类 Primary (top 1%) / Secondary (1-5%) / Regular
  3. (lazy) 预热 inpainter

  ─── For each unlearn request ────────────────────
  4. Classify D_f 中每个节点的 anchor 层级
  5. 若有 Primary ∈ D_f:
       DAR Phase 1: 节点尚在图中, 缓存 BFS 距离, 按连通分量分配候选名额
  6. EDP: PPR 计算 ERF 影响区域
  7. 物理删除节点 / 边 / 特征
  8. GSI: 触发条件成立则用 MGAE 或扩散模型修复局部拓扑
  9. 若有 Primary ∈ D_f:
       计算 h_new = current_embeddings.detach().clone()
       DAR Phase 2: 从 2k 候选中精选 k 个, 计算动态锚定目标
  10. Fine-tune: 最小化 L_task + L_anchor + forget_weight · L_forget
  11. 返回 f_{θ'} 与 metrics
  ─── Evaluation ──────────────────────────────────
  12. Utility (Acc / F1 / drop)
  13. Structural (degree KL / CC change / component count)
  14. Privacy (MIA Weak / Medium / Strong + overall_mia_auc)
  15. Efficiency (unlearn_time / speedup_vs_retrain)
Output: f_{θ'} + metrics
```

### 2.3 六大组件

#### 2.3.1 Hub Identification (GHI)

**目标** 找出对网络结构与信息流至关重要的"枢纽节点"。

**HubScore**

```
HubScore(v) = α · Norm(GradientSensitivity(v))
            + β · Norm(CentralityScore(v))
            + γ · Norm(ERFInfluence(v))
```

- GradientSensitivity: ‖∂L / ∂h_v‖_2, 表达模型对该节点表征的依赖度
- CentralityScore: degree / betweenness / eigenvector 加权混合
- ERFInfluence: 从 v 出发的 PPR 质量在 k 步内的累积影响

**Filter-and-Refine 加速**

| Stage | 操作 | 复杂度 |
|-------|------|--------|
| Stage 1 | 仅算 CentralityScore, 保留 top filter_ratio 候选 | O(\|E\|) |
| Stage 2 | 仅对候选算 GradientSensitivity 与 ERFInfluence | O(filter_ratio · \|V\| · cost) |

经验值: 小图 (< 5K 节点) 设 filter_ratio = 1.0; 大图 (> 200K) 设 filter_ratio = 0.05。

**锚点分类**

- Primary: 排名 top 1% (primary_ratio = 0.01), λ₁ 强约束
- Secondary: 排名 top 1%-5% (secondary_ratio = 0.05), λ₂ 弱约束
- Regular: 其余

#### 2.3.2 Anchor Stabilization (AS)

**目标** 在微调期间冻结 Hub 嵌入, 把误差控制在 ERF 影响范围内。

**理论基础** 误差级联可建模为 GNN 在删除扰动 ΔA 下的表征漂移:

```
‖h_v(A + ΔA) - h_v(A)‖ ≤ L · ‖ΔA‖ · ρ(v)
```

其中 ρ(v) 为节点 v 的影响半径。锚定约束 ‖h_v - h_v^orig‖² ≤ ε 把 Hub 处的 ρ(v) 钳制住,
等价于在 GNN 计算图上插入 stop-gradient 屏障。

**分层意义**

- Primary (λ₁ = 2.0, 默认): 强刚性, 防止结构骨架漂移
- Secondary (λ₂ = 0.5, 默认): 弱弹性缓冲, 避免锚定过死导致 acc 下降
- 默认值由 commit 8a9f799 从 (10.0, 1.0) 放松到 (2.0, 0.5), 用于平衡 acc-privacy

#### 2.3.3 ERF-based Partitioning (EDP)

**目标** 用功能性影响区域代替 METIS / BEKM / BLPA 的拓扑切割。

**ERF 定义** 给定遗忘节点集 D_f, ERF(D_f) = {v ∈ V : PPR(v ← D_f) > δ}。

**PPR 近似** Power Iteration k 步逼近, 复杂度 O(m / δ)。

**为什么不用 METIS**

- METIS 追求最小切边, 在 degree 方差极大的幂律图上必然把 Hub 的邻居切散
- ERF 基于功能依赖 (信息流远近) 聚合, Hub 与其影响半径内的节点自然聚合在一起

**输出** 一个 affected_region 子图, 后续微调仅在子图上做 (subgraph_finetune = true 时)。

#### 2.3.4 Generative Structural Inpainting (GSI)

**目标** 修复删除后留下的局部拓扑空洞, 抹平结构异常信号。

**两种后端**

| 后端 | 优势 | 适用 |
|------|------|-----|
| MGAE (Masked Graph Autoencoder) | 训练快, 推理快 | 中小规模, 默认 |
| Graph Diffusion | 表达力强, 多步去噪 | 大规模 / 需要更高保真 |

**触发条件** (满足任一即触发)

1. Hub-to-Hub 边被删除 (两端皆为 Primary 或 Secondary)
2. 局部聚类系数下降 > cc_drop_threshold (默认 0.30)
3. 删除导致连通分量碎裂

**跳过条件**

1. 删除节点全为 Regular 且不在 Hub 邻域
2. 局部结构变化 < min_damage_ratio (默认 0.10)

**inpainting_mode (commit 8a9f799 新增)**

- `none`: 完全跳过 (用于纯锚定消融)
- `local_only`: 仅在触发位置局部修复, 不预训练 inpainter
- `full`: 标准流程, 预训练 + 条件触发

#### 2.3.5 Distributed Anchor Replacement (DAR)

> DAR 是 HASI 在 Primary Hub 必须删除场景下 (GDPR 合规等) 的核心隐私机制。

**问题** 单点替代 (用一个新 Hub 接管被删除 Hub 的锚定功能) 会被 MIA 攻击者识别为"焊点":
新 Hub 的属性 (度数 / embedding) 突变, 反而泄露了"这里删过东西"的信号。

**核心思路** 把"锚定职责"分散到 k 个空间分布良好的替代节点, 每个节点只承担 λ₂ / k 的约束。

**两阶段执行 (解决 Inpainting 时序悖论)**

```
Phase 1 (删除前, 节点 v 仍在图中):
  1. BFS 缓存 v 到所有候选的距离 (删除后无法计算)
  2. 模拟删除, 检测连通分量变化
  3. 按分量分配候选名额, 过滤 < theta 的微小碎片
  4. 每个分量预选 2k 个候选

Phase 2 (删除 + 修复完成后):
  5. h_new ← current_embeddings.detach().clone()
  6. 从 2k 中精化为 k 个
  7. 计算动态锚定目标:
       cached_dist[u] == 1  →  锚定到 h_new[u]   (隐私优先, 直接邻居要适应新结构)
       cached_dist[u] >= 2  →  锚定到 h_orig[u]  (效用优先, 远端节点保持稳定)
```

**四种策略**

| 策略 | 机制 | 效用 | 隐私 |
|------|------|------|------|
| `hubscore` | 全局 top-k HubScore 节点 | 最高 | 最低 |
| `proximity_weighted` | 偏向删除节点附近 | 高 | 低 |
| `privacy_constrained` | 先满足最小距离约束再选 | 中 | 高 |
| `distributed` | 分量感知 + 贪心多样性 + Gumbel 噪声 | 平衡 | 平衡 |

**Gumbel 噪声** 对每个候选 score 加 -log(-log(U(0,1))) / τ, 引入随机性等价于 ε-DP 选择。

**Adaptive Decay** 候选不足时, 把 min_distance 从 3 自动放松到 2 再到 1, 避免硬约束失败。

#### 2.3.6 MIA Privacy Evaluation

**三级攻击者模型**

| 层级 | 能力 | 攻击特征 |
|------|------|---------|
| Weak | 黑盒查询 | 仅置信度 / 熵的变化 |
| Medium | Shadow Model | 在同分布图上训练影子模型 (最现实威胁) |
| Strong | 白盒 | 10 维全特征, 含 embedding drift |

**10 维 Strong 攻击特征**

```
1.  confidence_change           5.  embedding_l2_drift
2.  abs_confidence_change       6.  embedding_cosine_dist
3.  entropy_change              7.  degree_change_abs
4.  forget_confidence_ratio     8.  degree_change_relative
                                9.  neighbor_avg_drift
                                10. neighbor_max_drift  (+ homophily_change)
```

**指标**

- Per-tier accuracy: 期望 ≈ 0.5
- `overall_mia_auc`: 主指标, AUC of Strong attacker on shuffled member/non-member split
- `privacy_score = 1 - 2 · |acc - 0.5|`: 1.0 = 完美隐私, 0.0 = 零隐私

### 2.4 损失函数

```
L_total = L_task + L_anchor + forget_weight · L_forget
        = L_task + (L_primary + L_secondary + L_distributed) + λ_f · L_forget

L_primary     = λ₁ · mean_{v ∈ S_P} ‖h_v - h_v^orig‖²
L_secondary   = λ₂ · mean_{v ∈ S_S} ‖h_v - h_v^orig‖²
L_distributed = sum_{i=1}^{k} (λ₂ / k) · ‖h_{v_i} - target_i‖²

  where target_i =
    h_new[v_i]   if cached_dist(v_i, v_deleted) == 1   (直接邻居 → 适应新结构)
    h_orig[v_i]  if cached_dist(v_i, v_deleted) >= 2   (远端节点 → 保持稳定)

L_forget (commit 8a9f799 新增, 三种类型统一形式):
  node:    KL ( softmax(f_{θ'}(G_minus, v_forget)) || uniform )      # 让 forget 节点输出退化
  edge:    KL ( logits[u] || logits_orig[u] ) on endpoints           # 端点表征推离 forget
  feature: KL ( logits || logits_uniform ) on affected nodes
```

**实现关键点 (Critical Invariants)**

1. **Distance caching 在删除前** DAR Phase 1 必须在 v 仍在图中时跑完 BFS。删除后 NetworkX 中 v 不存在, 距离无法计算。
2. **h_new 必须 detach + clone** 否则 ‖h - h_new‖² 在反向传播时变成 0, 锚定信号失效。
3. **特征遗忘的 forget target 在 mask 前算** 先在原始特征上算 logits 当 forget 目标, 再 mask 特征。否则 forget 信号会包含被 mask 的内容。
4. **DAR 权重不双重稀释** k 个替代节点总权重 = λ₂, 每个 λ₂ / k。不要再除一次。
5. **Baseline MIA 评估用 b.model** PrivacyEvaluator 接受的应是 baseline 跑完的模型, 不是原始预训练模型。

### 2.5 三种遗忘类型

#### Node Unlearning

```
情况 A (Regular):    EDP → 物理删除 → (可选) Inpainting → 微调 (L_task + L_anchor)
情况 B (Secondary):  + 该节点从锚点集移除, 邻接 Regular 节点提升缓冲
情况 C (Primary):    + DAR 两阶段 + Inpainting 强制触发 + 动态锚定目标
```

#### Edge Unlearning

```
1. 标记 (u, v) ∈ D_f 为待删边
2. EDP 算 ERF(u) ∪ ERF(v)
3. 物理删除边
4. (可选) Inpainting: 仅 Hub-Hub 边触发
5. 微调: anchor 仍约束 Hub 嵌入, forget loss 推开 (u, v) 端点的 logits
```

#### Feature Unlearning

```
1. 先在原始 X 上算 logits_orig (用于 forget target, 关键时序)
2. mask 指定特征维度 (X[:, d_f] ← 0 或重采样均值)
3. EDP 算 affected_nodes = {v : 特征改变后表征漂移 > τ}
4. (可选) 训练特征 inpainter 补回被删维度的近似
5. 微调: anchor 锚到 h_new (mask 后的表征), forget loss 在 affected_nodes 上推开
```

> 之前的 framework 文档把 feature unlearning 的 anchor target 写成 h_old, 这是 Model Inversion 风险。
> 必须锚到 h_new, 即 mask 后的表征。

---

## 3. Architecture

### 3.1 代码模块 ↔ 论文组件映射

| 论文组件 | 代码位置 | 关键类 / 函数 |
|---------|---------|--------------|
| GHI | `hasi/hub_identification/` | `HubScorer`, `CentralityHubIdentifier`, `GradientHubIdentifier`, `ERFHubIdentifier` |
| AS | `hasi/anchor_stabilization/` | `AnchorManager`, `AnchorStabilizationLoss`, `KnowledgeDistillationLoss` |
| EDP | `hasi/erf_partitioning/` | `PPRComputer`, `ERFCalculator`, `ERFPartitioner` |
| GSI | `hasi/structural_inpainting/` | `MaskedGraphAutoencoder`, `GraphDiffusionModel`, `StructuralInpainter` |
| DAR | `hasi/dar/` | `DARPipeline`, `DeletionContext`, `ComponentDetector`, `GumbelSelector`, `AdaptiveDecay`, `create_strategy()` |
| MIA | `evaluation/mia/` | `PrivacyEvaluator`, `WeakAttacker`, `MediumAttacker`, `StrongAttacker` |
| 主流水线 | `hasi/unlearner.py` | `HASIUnlearner` (统一入口) |
| GNN 模型 | `models/` | `GCN`, `GAT`, `GraphSAGE`, `UnlearnableGNN`, `GNNTrainer` |
| Baselines | `baselines/` | `RetrainBaseline`, `GraphEraserBEKM`, `GraphEraserBLPA`, `GNNDelete`, `GIF`, `SGU`, `AGU` |
| 配置 | `configs/` | `hasi_default.yaml`, `hasi_fast.yaml`, `dataset_configs.yaml`, `baseline_configs.yaml` |
| 实验入口 | `experiments/` | `run_hasi.py`, `run_baselines.py`, `hyperparam_sweep.py`, `parallel_sweep.py` |

### 3.2 关键调用链

`experiments/run_hasi.py` 7 步流程:

```
[1] Load dataset                  →  data.load_dataset(name)
[2] Build model                   →  models.{GCN,GAT,GraphSAGE} + UnlearnableGNN
[3] Train base model              →  GNNTrainer.train_full_batch
[4] Init HASI unlearner           →  HASIUnlearner(model, data, **cfg)
[5] Preprocess                    →  unlearner.preprocess()
                                       ├── hub_scorer.compute_hub_scores
                                       ├── anchor_manager.classify_anchors
                                       └── inpainter.train_inpainter (lazy 模式下推迟)
[6] Run unlearning                →  unlearner.unlearn_{nodes,edges,features}(forget_set, ...)
                                       ├── if Primary ∈ forget_set:
                                       │     dar.run_phase1
                                       ├── partitioner.partition_for_unlearning
                                       ├── physical_delete
                                       ├── if trigger: inpainter.apply
                                       ├── if Primary ∈ forget_set:
                                       │     compute h_new, dar.run_phase2
                                       └── fine_tune (L_task + L_anchor + forget_weight · L_forget)
[7] Evaluate                      →  UtilityEvaluator + StructuralMetrics
                                       + PrivacyEvaluator + EfficiencyMetrics
```

### 3.3 配置驱动 (合并优先级: CLI > dataset_configs > hasi_default)

完整字段在 `configs/hasi_default.yaml`, 关键项:

```yaml
model:                  type / hidden_channels / num_layers / dropout
hub_identification:     method / filter_ratio / primary_ratio / secondary_ratio
anchor_stabilization:   lambda1 (2.0) / lambda2 (0.5)            # 8a9f799 放松
erf_partitioning:       alpha (0.15) / k_steps (3) / threshold (0.01)
inpainting:             mode (none|local_only|full) / method (mgae|diffusion)
dar:                    enabled / k (5) / strategy (hubscore) / min_distance / gumbel_tau
unlearning:             type / ratio / forget_weight (默认 0.0) / finetune_epochs / finetune_lr
optimization:           affected_anchor_scope / subgraph_finetune (true) / lazy_train_inpainter
evaluation:             mia_levels / run_structural_metrics
```

### 3.4 评估栈

| 维度 | 指标 | 实现 |
|------|------|------|
| 效用 | test_accuracy / f1_macro / accuracy_drop | `evaluation/utility_metrics.py` |
| 结构 | degree_kl_divergence / clustering_coefficient_change / component_count_change | `evaluation/structural_metrics.py` |
| 效率 | unlearn_time / speedup_vs_retrain / memory_overhead | `evaluation/efficiency_metrics.py` |
| 隐私 | weak / medium / strong accuracy + **overall_mia_auc** + privacy_score | `evaluation/mia/` |
| 认证 | ε-δ unlearning bound (理论上界) | `evaluation/certified_deletion.py` |

主指标: `overall_mia_auc` (Strong 攻击者 AUC, 文献标准)。

### 3.5 自适应执行 (commit 8a9f799 引入)

- **subgraph_finetune**: 大图 (num_nodes > 5000) 走 ERF 子图微调, 小图走全图微调
- **lazy_train_inpainter**: 预处理时不训练 inpainter, 第一次触发 inpainting 时再训, 节省冷启动
- **inpainting_mode**: 提供 `none` / `local_only` / `full` 三档, 便于消融
- **forget_weight**: 三种遗忘类型统一接口, 默认 0.0, > 0 时启用 L_forget

---

## 4. Experimental Setup

### 4.1 数据集

| 数据集 | 节点数 | 边数 | 特征 | 类别 | 图特性 | 模型 |
|--------|-------:|-----:|----:|----:|--------|------|
| Cora | 2,708 | 10,556 | 1,433 | 7 | 小图, 高同质性 | GCN |
| CiteSeer | 3,327 | 9,104 | 3,703 | 6 | 稀疏, 低同质性 | GCN |
| PubMed | 19,717 | 88,648 | 500 | 3 | 中等规模 | GCN |
| Reddit | 232,965 | ~11.6M | 602 | 41 | 大图, 社区结构 | GraphSAGE |
| Ogbn-Arxiv | 169,343 | ~1.17M | 128 | 40 | 大规模, 时序 | GraphSAGE |

### 4.2 Baselines

| 方法 | 来源 | 类型支持 |
|------|------|---------|
| Retrain | 黄金标准 | 全部 |
| GraphEraser-BEKM | CCS 2022 | node |
| GraphEraser-BLPA | CCS 2022 | node |
| GNNDelete | ICLR 2023 | node / edge |
| GIF | WWW 2023 | node |
| SGU | 2025 | node |
| AGU | IJCAI 2025 | node |

> GraphEraser 的 BEKM 与 BLPA 必须独立报告为两行 (不可合并为 "GraphEraser")。

### 4.3 RQ 重新分组 (建议)

将 6 个并列 RQ 改为 3 层:

**Track A: 效用 + 结构感知 (主线)**

- RQ1 整体效用: HASI vs 7 baselines × 5 数据集
- RQ2 锚定贡献: w/ vs w/o Anchor Stabilization, 不同 λ₁
- RQ3 分区策略: ERF vs BEKM vs BLPA

**Track B: 隐私 (核心新贡献)**

- RQ4 DAR 隐私-效用平衡: 4 策略 × 3 级 MIA, k 与 min_distance 消融
- RQ5 修复对隐私的贡献: w/ vs w/o Inpainting, MIA AUC 对比

**Track C: 范围 (sub-claim)**

- RQ6 三种遗忘类型 × 不同 ratio × 不同图规模

### 4.4 实验矩阵 (按 Track 整理)

| Track | 表 / 图 | 关键变量 | 关键指标 |
|------:|--------|---------|---------|
| A | 主表 | method × dataset | test_accuracy, accuracy_drop, speedup |
| A | 消融表 | HASI variants | acc + delta |
| B | DAR 策略表 | strategy × MIA tier | acc + per-tier MIA + overall_mia_auc |
| B | Pareto 曲线 | k × min_distance | (MIA, acc) 散点 |
| C | Universal 表 | method × type | test_accuracy |
| C | 可扩展性折线 | dataset_size / ratio | unlearn_time |

---

## 5. Current Status & Gaps

### 5.1 已完成 (代码层)

- 六大组件全部实现, 各组件单元测试通过 (228 项)
- 7 个 baselines 全部实现
- 三种遗忘类型 (node / edge / feature) 流水线打通
- MIA 三级攻击者实现
- 配置驱动 + CLI override + 多 seed 接口

### 5.2 当前实验结果覆盖 (results/, 截至 2026-03-20)

| 维度 | 实际 | 设计目标 | 缺口 |
|------|------|---------|------|
| 数据集 | Cora | Cora / CiteSeer / PubMed / Reddit / Ogbn-Arxiv | 缺 4/5 |
| 遗忘类型 | node | node / edge / feature | 缺 2/3 |
| 遗忘比例 | 0.0004, 0.05, 0.1 | 1% / 5% / 10% / 20% / 30% | 缺 0.2 / 0.3 |
| Baseline MIA | **完全缺失** (所有 baseline JSON `overall_mia_auc = None`) | 必须有 | 致命 |
| 多 seed | 单 seed=42 | 至少 3 seed (42, 123, 2024) mean±std | 缺 |
| inpainting 启用 | mode=none | mode=full (RQ5 要求) | 当前跑的是阉割版 |
| forget_weight 启用 | 0.0 | > 0 | 当前未使用 |

### 5.3 最新一批 (Cora, node, r=0.1, 2026-03-20) 关键数字

| 方法 | test_acc | drop | MIA AUC | Privacy | speedup |
|------|---------:|-----:|--------:|--------:|--------:|
| original | 0.809 | — | — | — | — |
| retrain | 0.789 | 0.020 | N/A | N/A | 1.0x |
| gnndelete | 0.792 | 0.017 | N/A | N/A | — |
| gif | 0.792 | 0.017 | N/A | N/A | — |
| sgu | 0.795 | 0.014 | N/A | N/A | — |
| agu | 0.794 | 0.015 | N/A | N/A | — |
| bekm | 0.647 | 0.162 | N/A | N/A | — |
| blpa | 0.528 | 0.281 | N/A | N/A | — |
| **HASI (current)** | **0.762** | **0.047** | **0.520** | **0.968** | **2.61x** |

观察

- HASI 在 acc 维度暂时**落后 gnndelete / gif / sgu / agu 约 3 pp**, 仅胜 GraphEraser 两种策略
- HASI MIA AUC ≈ 0.52 接近随机猜测, 隐私维度好, 但缺 baseline 对照无法证明优势
- DAR 在该批次触发 2 次 (270 个删点中), 占比 < 1%, "core contribution" 在该实验里未充分发挥
- inpainting 关闭, forget_weight = 0, 即"阉割版 HASI"

### 5.4 已知问题清单 (按严重度)

**🔴 致命 (投稿前必须修复)**

1. Baselines 全无 MIA 数据 → 隐私优势无法对照证明
2. Commit message 自称 "9 实验 #1", 但 results/ 只有 1/9 → 数据与叙事不一致
3. 大图 (Reddit / Ogbn-Arxiv) 完全空白 → RQ6 + RQ1 大图部分均无数据

**🟠 严重**

4. Cora r=0.1 上 HASI acc 落后 baselines, "DAR core contribution" 需要专门 stress test (forget set 全为 Primary Hub) 验证
5. inpainting_mode 在最新一批被关闭, RQ5 完全没数据
6. forget_weight 新机制未启用, 不知道开启后能否拉回 acc
7. edge / feature 流水线无 results/JSON 产物

**🟡 中等**

8. 多 seed 缺失, 无法报告 mean ± std, 无显著性检验
9. 设备配置: 最新一批日志显示在 CPU 跑 (与 cfg device='cuda' 不符)
10. results/ 内混入失败 JSON (GitHub 连接错误 / TypeError), 未清理
11. reviewer.md 是 2026-02-11 旧 audit, 与当前代码状态严重脱节, 需要重做

**🔵 写作 / framing**

12. Title 中 "Universal" 与第 4.5 节 "DAR core contribution" 与第 3 章 "结构感知" 三个卖点并列, 需要收敛主线
13. 第 2 章对 GraphEraser 用 "结构性溃败" 等强词, 顶刊审稿人会反弹, 改中性化
14. 6 大组件 vs 4 字箴言映射不对齐 (EDP, DAR 找不到位置), 已在本文档 2.1 节修正

### 5.5 建议优先级 (投稿前)

| 优先 | 动作 | 成本 | 解决的问题 |
|------|------|------|-----------|
| P0 | 给 baselines 接 `PrivacyEvaluator`, 补 MIA 列 | 改 1 文件 | 隐私对照表 |
| P0 | 重跑最新 r=0.1 with `inpainting.mode=full` + `forget_weight=0.1~0.3` | 改配置 | 验证完整版 HASI 能否追上 baseline |
| P0 | 跑齐 3 数据集 × 3 类型 × r=0.05/0.1, 多 seed | 中 (需要 GPU) | 兑现 "9 实验" claim |
| P1 | 构造 Primary-heavy forget set, 让 DAR 触发率 ≥ 30% 的 stress test | 改 forget set 选择 | DAR core contribution 证据 |
| P1 | Reddit / Ogbn-Arxiv 至少跑一次, 哪怕 r=0.01 | 高 (大图 + GPU) | RQ6 大图证据 |
| P2 | 改 title / abstract 主线, 收敛到 "结构感知 + 隐私" | 写作 | 反 selling-point 稀释 |
| P2 | 清理 results/ 失败文件, 重做 reviewer.md audit | 低 | 仓库整洁 + checklist |

---

## 6. 附: 与 README / HASI_Framework 的区别

- `README.md`: 使用导向, 含 install / API / quickstart
- `HASI_Framework.md`: 早期 (2026-02) 完整设计文档, 1053 行, 但 framing 已过期
- `CLAUDE.md`: AI 协作约定 + 关键 invariant
- 本文档 (`PROJECT_OVERVIEW.md`): **投稿前的统一参考**, 整合 motivation + method + architecture, 并显式列出当前数据 vs 设计目标的缺口

下次 framework 更新时, 建议把 HASI_Framework.md 收敛到只保留方法细节 (第 4 节六大组件 + 第 5 节流程 + 第 6 节损失函数), 把 motivation / RQ / 实验设计移到本文档统一维护。
