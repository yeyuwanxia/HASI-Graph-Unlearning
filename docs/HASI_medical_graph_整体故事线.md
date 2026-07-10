# HASI 医疗图数据故事线与实验组织方案

> 目的：将 HASI（Hub-Anchored Structural Inpainting）从通用图遗忘工作，重新组织为一个更有现实背景和审稿说服力的“医疗图数据遗忘”故事线。本文档可用于继续撰写论文 Introduction、设计 RQ、补充实验矩阵和整理实验结果。

---

## 0. 一句话定位

**HASI 面向医疗图数据中的隐私保护图遗忘问题，将删除请求从“破坏性擦除”重新定义为“结构感知的外科式编辑”：在删除患者、疾病、药物、蛋白、医学关系或敏感特征时，先识别并稳定医学图中的关键 Hub，再定位受影响医学邻域，修复删除造成的结构空洞，并在关键 Hub 必须删除时通过分散锚定替代降低隐私泄露风险。**

更简洁的论文定位可以写成：

> Medical graph unlearning is not merely about removing a sample from a model; it must remove sensitive medical information while preserving the clinical or biological structure that supports reliable prediction and reasoning.

---

## 1. 为什么医疗图场景更适合这个工作

### 1.1 医疗数据天然是图结构

医疗 AI 中很多数据不是独立样本，而是关系网络。例如：

- 患者 — 疾病 — 症状 — 药物 — 检查指标图；
- 疾病 — 基因 — 蛋白 — 通路图；
- 药物 — 靶点 — 适应症 — 禁忌症图；
- 医学文献引用图；
- 多中心医疗机构之间的患者、诊疗、检查和知识关联图。

因此，医疗模型中的一个“样本”经常不是独立记录，而是整个医学关系网络中的一个节点、边或特征。

### 1.2 医疗图中的删除请求更真实、更敏感

医疗图遗忘可以来自多种真实场景：

1. **患者撤回授权**  
   某个患者节点及其关联诊疗边需要删除。

2. **机构退出多中心建模**  
   某个医院或机构子图需要从联合建模结果中移除。

3. **医学证据更新**  
   旧的药物—疾病关系、疾病—基因关系或蛋白互作关系被新证据证伪，需要删除错误边。

4. **敏感特征不可继续使用**  
   某些检查指标、遗传特征、人口学变量或隐私敏感特征需要做 feature unlearning。

5. **罕见病或小群体隐私风险**  
   删除一个罕见病患者或小群体样本后，局部结构变化可能暴露“谁被删除了”。

这使得医疗图遗忘不仅是效率问题，更是隐私、合规和模型可靠性问题。

### 1.3 医疗 Hub 不能被当成普通节点

在通用图里，Hub 可能只是高连接节点；但在医疗图里，Hub 往往有明确医学语义，例如：

- 高连接疾病：diabetes、hypertension、cancer；
- 高频症状：fever、pain、fatigue；
- 关键药物：aspirin、metformin；
- 关键基因或蛋白；
- 医学知识图谱中的核心疾病、通路或药物节点；
- 多中心医疗图中的大型医院或关键队列节点。

这些节点连接大量患者、症状、药物、检查、基因和疾病。删除它们或其邻域会影响大范围医学关系链条。

因此，医疗图遗忘的关键难点不是“删掉一个样本”，而是：

> 如何在删除敏感医学实体或关系的同时，避免破坏医学图中的知识骨架和下游预测能力。

### 1.4 删除会留下可被攻击者观察的结构疤痕

医疗图删除可能留下以下结构异常：

- 局部度数突然下降；
- 聚类系数下降；
- 连通分量碎裂；
- Hub 邻域 embedding 漂移；
- 邻居节点预测置信度异常变化；
- 药物、疾病、蛋白或患者子图的局部结构不自然。

这些结构疤痕不仅会降低效用，也可能成为 membership inference attack（MIA）的可观察信号。

因此，HASI 的核心问题可以定义为：

> How can we unlearn sensitive medical graph information while preserving clinically meaningful graph structure and suppressing deletion-induced privacy signals?

---

## 2. Introduction 建议故事线

Introduction 可以组织为 6 个段落。

---

### Paragraph 1: 医疗 AI 越来越依赖图结构

**功能：** 建立医疗图背景，不要一开始就讲 unlearning。

**核心逻辑：**

现代医疗 AI 不只处理独立样本，还处理大量关系型医学数据。患者、疾病、症状、药物、检查指标、基因、蛋白和文献证据之间形成复杂图结构。这些图支持疾病预测、药物重定位、蛋白功能预测、临床知识推理和辅助诊疗。

**可写成：**

> Modern medical AI increasingly relies on graph-structured data, including patient-disease-symptom graphs, drug-disease knowledge graphs, protein-protein interaction networks, and biomedical literature graphs. These graphs provide relational context for clinical prediction, therapeutic discovery, and biomedical knowledge reasoning. In such systems, model predictions are shaped not only by individual entities but also by their surrounding medical neighborhoods.

---

### Paragraph 2: 医疗图需要遗忘机制

**功能：** 引出 graph unlearning 的必要性。

**核心逻辑：**

医疗图是动态、敏感和受监管的。患者可能撤回授权，医院可能退出多中心建模，医学证据可能更新，敏感特征可能不再允许使用。完全重训成本高，因此需要高效图遗忘。

**可写成：**

> However, medical graphs are continuously revised and regulated. Patients may withdraw consent, institutions may request data removal, biomedical relations may be corrected as evidence evolves, and sensitive clinical features may become unavailable. These scenarios require graph unlearning: removing the influence of requested nodes, edges, or features from a trained GNN without retraining from scratch.

---

### Paragraph 3: 医疗图遗忘比普通数据遗忘更难

**功能：** 说明图结构依赖和 Hub 风险。

**核心逻辑：**

不同于表格样本，医疗图实体不是独立的。一个疾病节点连接症状、药物、患者、基因和文献证据；一个蛋白 Hub 连接多个生物通路。删除一个节点、边或特征会影响邻居和多跳区域。

**可写成：**

> Unlike tabular records, medical graph entities are not independent. A disease node may connect symptoms, drugs, genes, patients, and literature evidence, while a protein hub may participate in multiple biological pathways. Removing such entities perturbs not only the deleted item but also the surrounding clinical or biological neighborhood through message passing.

---

### Paragraph 4: 现有方法忽略了医疗图里的两个关键后果

**功能：** 温和批判现有方法，避免过度攻击。

**核心逻辑：**

现有方法通常聚焦分片、影响函数或梯度修正，能够提升效率，但它们往往把删除看成局部参数更新，没有显式处理：

1. 医学 Hub 受损；
2. 删除后结构空洞留下的隐私痕迹。

**可写成：**

> Existing graph unlearning methods mainly focus on partitioning, influence approximation, or gradient-based model correction. While these strategies can improve efficiency, they often treat deletion as a local update problem. In medical graphs, this overlooks two important effects: damage to clinically meaningful hubs and structural holes left by deletion. Both effects can degrade model utility and create privacy-observable traces.

---

### Paragraph 5: 提出 HASI

**功能：** 引出方法，不要堆太多细节。

**核心逻辑：**

HASI 将医疗图遗忘视为结构感知的外科式编辑。它做六件事：

1. 识别医学 Hub；
2. 锚定 Hub 表示；
3. 定位受影响医学邻域；
4. 修复结构空洞；
5. 当 Primary Hub 必须删除时，分散转移锚定职责；
6. 用三级 MIA 验证隐私。

**可写成：**

> We propose HASI, a hub-anchored structural inpainting framework for privacy-preserving medical graph unlearning. HASI first identifies structurally and semantically important medical hubs, stabilizes them as anchors to prevent error cascades, localizes the affected region through effective receptive fields, repairs deletion-induced structural holes through generative inpainting, and uses distributed anchor replacement when a primary hub itself must be removed. The resulting model is evaluated under utility, structural fidelity, efficiency, and multi-level membership inference attacks.

---

### Paragraph 6: 贡献总结

**功能：** 收束贡献，避免把 Universal 放成最大卖点。

建议三条贡献：

1. **Medical graph unlearning problem framing**  
   提出医疗图遗忘中的两个关键风险：Hub damage 和 structural scars。

2. **HASI framework**  
   提出 Hub anchoring + ERF localization + structural inpainting + DAR 的统一框架。

3. **Medical graph evaluation**  
   在医学文献图、生物网络、医学知识图谱或患者图上评估 utility、structure、privacy、efficiency。

**可写成：**

> Our contributions are threefold. First, we identify hub damage and deletion-induced structural scars as two overlooked challenges in medical graph unlearning. Second, we propose HASI, a structure-aware unlearning framework that combines hub identification, hierarchical anchor stabilization, ERF-based localization, structural inpainting, and distributed anchor replacement. Third, we provide a multi-dimensional evaluation on medical graph datasets, covering utility, structural fidelity, privacy, efficiency, and different unlearning requests.

---

## 3. HASI 组件的医疗化解释

| HASI 组件 | 技术作用 | 医疗图解释 | 应该支撑的实验 |
|---|---|---|---|
| GHI: Hub Identification | 找出结构关键 Hub | 找出关键疾病、药物、蛋白、症状、机构或文献节点 | Hub deletion vs random deletion |
| AS: Anchor Stabilization | 锚定 Hub embedding，阻断误差级联 | 稳定医学知识骨架，避免保留医学关系漂移 | w/o anchor、flat anchor、hierarchical anchor |
| EDP: ERF-based Partitioning | 定位受影响区域 | 找到真正受删除影响的医学邻域 | ERF vs BEKM/BLPA/random region |
| GSI: Structural Inpainting | 修复结构空洞 | 修复删除后不自然的医学关系断裂 | none/local/full inpainting |
| DAR: Distributed Anchor Replacement | Primary Hub 必须删除时分散替代 | 避免单个替代医学节点突变为隐私“焊点” | Primary-heavy stress test |
| MIA Evaluation | 验证删除痕迹是否可被攻击者识别 | 检查患者、疾病、蛋白或关系删除是否可被推断 | weak/medium/strong MIA |

---

## 4. 推荐 Research Questions

### RQ1: 医疗图遗忘是否存在 Hub-induced fragility？

**问题：**

> Are medical graph unlearning requests involving hubs more damaging than random deletion requests?

**实验设计：**

比较三类删除：

1. random deletion；
2. low-degree deletion；
3. medical hub deletion。

**数据集：**

- PubMed citation graph；
- PPI / ogbn-proteins；
- PrimeKG / TDC-style medical KG。

**指标：**

- Accuracy / F1 drop；
- embedding drift；
- neighbor drift；
- degree KL divergence；
- clustering coefficient change；
- MIA AUC。

**预期结论：**

Hub deletion 会导致更大的结构漂移、性能下降和隐私暴露风险。这一实验负责证明论文问题真实存在。

---

### RQ2: HASI 是否能在医疗图上保持效用和结构稳定性？

**问题：**

> Does HASI preserve utility and medical graph structure better than existing graph unlearning methods?

**比较方法：**

- Retrain；
- GraphEraser-BEKM；
- GraphEraser-BLPA；
- GIF；
- GNNDelete；
- SGU；
- AGU；
- HASI。

**指标：**

- Accuracy / F1；
- accuracy drop；
- degree KL divergence；
- clustering coefficient change；
- component count change；
- unlearn time；
- speedup vs retrain。

**呈现方式：**

主表建议同时报告：

| Dataset | Method | F1/Acc | Drop | Degree KL | CC Change | MIA AUC | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|

**预期结论：**

HASI 不一定在所有 accuracy 上绝对最高，但应在“效用 + 结构 + 隐私 + 效率”的综合权衡上最好。

---

### RQ3: Anchor Stabilization 是否能减少医疗 Hub 周围的误差级联？

**问题：**

> Does hierarchical hub anchoring prevent representation drift around medical hubs?

**比较变体：**

1. HASI-full；
2. w/o Anchor；
3. flat anchor；
4. hierarchical anchor；
5. overly strong anchor, e.g., large lambda1；
6. relaxed anchor, e.g., small lambda1。

**指标：**

- Hub embedding drift；
- 1-hop neighbor drift；
- 2-hop neighbor drift；
- F1 / accuracy；
- MIA AUC。

**关键图：**

可以画一张 drift heatmap：

- 横轴：删除类型；
- 纵轴：节点层级，Primary / Secondary / Regular / Neighbor；
- 颜色：embedding drift。

**预期结论：**

分层锚定比无锚定和统一锚定更稳，能够在保持 Hub 骨架的同时避免全图僵硬。

---

### RQ4: Structural Inpainting 是否能减少结构疤痕和隐私泄露？

**问题：**

> Does structural inpainting reduce deletion-induced structural scars and membership inference risks?

**比较变体：**

1. inpainting_mode = none；
2. inpainting_mode = local_only；
3. inpainting_mode = full。

**指标：**

- degree KL divergence；
- clustering coefficient change；
- component count change；
- local homophily change；
- MIA AUC；
- weak / medium / strong MIA accuracy。

**关键图：**

- 结构指标柱状图；
- MIA AUC 对比图；
- deletion neighborhood before/after 可视化。

**预期结论：**

inpainting 能降低结构异常，从而降低 MIA 攻击成功率。

---

### RQ5: 当医疗 Primary Hub 必须删除时，DAR 是否优于单点替代？

**问题：**

> When a primary medical hub must be deleted, does distributed anchor replacement provide a better privacy-utility trade-off than single replacement?

**删除设置：**

构造 Primary-heavy forget set：

- top 1% HubScore disease / protein / drug / paper nodes；
- 或在 PubMed / PPI / PrimeKG 中选择高中心性节点；
- 让 DAR 触发率至少达到 30%，否则无法支撑核心贡献。

**比较策略：**

1. no replacement；
2. single replacement；
3. hubscore replacement；
4. proximity_weighted；
5. privacy_constrained；
6. distributed DAR。

**指标：**

- F1 / accuracy；
- MIA AUC；
- replacement node degree change；
- replacement embedding drift；
- local structural change；
- runtime。

**关键图：**

Pareto curve：

- x-axis: utility drop；
- y-axis: MIA AUC；
- 点越靠近左下越好。

**预期结论：**

单点替代会产生明显突变，distributed DAR 可以降低替代节点的可识别性，在隐私和效用之间更平衡。

---

### RQ6: HASI 是否支持多种医疗删除请求？

**问题：**

> Can HASI handle node, edge, and feature unlearning requests in medical graphs?

**删除类型解释：**

| 删除类型 | 医疗含义 | 示例 |
|---|---|---|
| Node unlearning | 删除患者、疾病、药物、蛋白、文献节点 | patient withdrawal, disease node removal |
| Edge unlearning | 删除医学关系 | drug-disease edge, protein-protein interaction, citation edge |
| Feature unlearning | 删除敏感特征 | lab test, gene signature, demographic feature |

**指标：**

- F1 / accuracy；
- MIA AUC；
- runtime；
- structural metrics。

**定位：**

这是 scope / generality sub-claim，不建议作为主卖点。

---

## 5. 医疗图数据集组织建议

### 5.1 第一层：PubMed citation graph

**定位：** 最低成本、最容易接入的医疗相关图。

**任务：**

- node classification。

**删除场景：**

- random paper node deletion；
- high-degree / high-PageRank paper node deletion；
- citation edge deletion；
- feature dimension deletion。

**优点：**

- 实现成本低；
- 可快速验证代码；
- 和现有 Cora 结果自然衔接。

**缺点：**

- 医疗语义较弱，更像 biomedical literature graph；
- 不能单独支撑强医疗场景叙事。

**适合实验：**

- RQ1；
- RQ2；
- RQ3；
- RQ4；
- 初步 RQ6。

---

### 5.2 第二层：PPI / ogbn-proteins

**定位：** 生物医学结构图，适合支撑 Hub 的生物学意义。

**任务：**

- protein function prediction；
- multi-label classification。

**删除场景：**

- protein hub node deletion；
- protein-protein interaction edge deletion；
- gene-set / biological signature feature deletion。

**优点：**

- Hub 有明确生物学意义；
- 更适合说明“不能把 Hub 当普通节点”；
- 适合验证结构稳定性和生物网络泛化。

**缺点：**

- 多标签任务可能需要单独适配 evaluation；
- 训练和 unlearning 成本高于 PubMed。

**适合实验：**

- RQ1；
- RQ2；
- RQ5；
- RQ6。

---

### 5.3 第三层：PrimeKG / TDC-style medical knowledge graph

**定位：** 最能体现医疗知识图谱遗忘的真实场景。

**任务：**

- link prediction；
- drug-disease prediction；
- relation prediction。

**删除场景：**

- drug-disease edge deletion；
- disease node deletion；
- drug node deletion；
- high-degree gene / disease / drug hub deletion；
- sensitive relation type deletion。

**优点：**

- 医疗故事最强；
- 可以自然解释 node、edge、feature 三类遗忘；
- 适合顶会论文背景。

**缺点：**

- 数据预处理和任务适配成本较高；
- 可能需要把 HASI 从同构图扩展到异构图或先做同构化处理。

**适合实验：**

- 强化版 RQ2；
- RQ5；
- RQ6；
- case study。

---

## 6. 最小可投稿实验闭环

如果时间有限，建议先完成下面 6 个实验。

| 优先级 | 实验 | 数据集 | 删除类型 | 目的 |
|---|---|---|---|---|
| P0 | E1: Hub-induced fragility | PubMed | random vs hub node | 证明问题真实存在 |
| P0 | E2: Main comparison | PubMed | node | HASI vs baselines |
| P0 | E3: Baseline MIA | PubMed | node | 补齐隐私对照 |
| P1 | E4: HASI ablation | PubMed | node | Anchor / ERF / GSI 消融 |
| P1 | E5: DAR stress test | PubMed or PPI | Primary hub node | 支撑 DAR 核心贡献 |
| P1 | E6: Biological graph validation | PPI / ogbn-proteins | hub node / edge | 医疗泛化 |
| P2 | E7: Multi-request validation | PubMed / PPI | node / edge / feature | 支撑 scope sub-claim |
| P2 | E8: Medical KG case | PrimeKG / TDC | edge / node | 强化医疗叙事 |

最小主论文版本可以先做：

- PubMed + PPI；
- node unlearning 为主；
- edge/feature 作为补充；
- PrimeKG 作为 case study 或 future extension。

---

## 7. 结果应该如何支撑 Introduction

| Introduction claim | 必须有的证据 | 推荐图表 |
|---|---|---|
| 医疗图删除不是局部问题 | Hub deletion 比 random deletion 更大幅影响结构和效用 | Motivation figure |
| 医疗 Hub 不能当普通节点处理 | Hub embedding drift 和 neighbor drift 更明显 | Drift heatmap |
| 现有方法忽略结构空洞 | Baselines 有更高 degree KL / CC change / MIA AUC | Main comparison table |
| Anchor 能稳定医学图骨架 | w/o anchor 漂移更大 | Ablation table |
| Inpainting 能减少结构疤痕 | full GSI 的结构指标和 MIA 最好 | Structural + privacy bar plots |
| DAR 能处理 Primary Hub 删除 | distributed DAR 在 utility-MIA Pareto 上更优 | Pareto curve |
| HASI 支持多类医疗删除 | node/edge/feature 都能运行 | Scope table |

---

## 8. 推荐图表设计

### Figure 1: Motivation Figure

**主题：** Why medical graph unlearning is structurally hard?

图中展示：

1. 医疗图局部结构：patient / disease / symptom / drug / protein；
2. 删除一个 Hub disease 或 protein；
3. 邻域断裂、embedding drift、MIA signal 出现；
4. HASI 通过 anchoring + inpainting + DAR 修复。

---

### Figure 2: HASI Framework

横向 pipeline：

1. Medical graph input；
2. Hub identification；
3. Anchor stabilization；
4. ERF localization；
5. Structural inpainting；
6. DAR if primary hub is deleted；
7. Utility + structure + privacy evaluation。

---

### Figure 3: Hub Deletion Fragility

柱状图或折线图：

- random deletion；
- low-degree deletion；
- secondary hub deletion；
- primary hub deletion。

指标：

- accuracy drop；
- degree KL；
- MIA AUC。

---

### Figure 4: Utility-Privacy Pareto for DAR

散点图：

- no replacement；
- single replacement；
- proximity；
- privacy-constrained；
- distributed DAR。

x-axis: utility drop。  
y-axis: MIA AUC。  
越靠左下越好。

---

### Table 1: Main Results

| Dataset | Method | Acc/F1 | Drop | Degree KL | CC Change | MIA AUC | Speedup |
|---|---|---:|---:|---:|---:|---:|---:|

---

### Table 2: Ablation Results

| Variant | Acc/F1 | Hub Drift | Neighbor Drift | Degree KL | MIA AUC |
|---|---:|---:|---:|---:|---:|
| HASI-full | | | | | |
| w/o Anchor | | | | | |
| w/o ERF | | | | | |
| w/o GSI | | | | | |
| w/o DAR | | | | | |

---

## 9. 当前最需要补的实验

根据现有项目状态，优先补下面几件事。

### P0: Baseline MIA 必须补齐

如果只有 HASI 的 MIA AUC，而 baseline 的 MIA 为空，就无法证明隐私优势。需要给所有 baseline 接入同一个 PrivacyEvaluator，并报告 weak / medium / strong MIA 或至少 overall_mia_auc。

### P0: 重跑完整版 HASI

当前如果使用的是 `inpainting_mode=none` 和 `forget_weight=0`，则不能支撑 Introduction 中 structural inpainting 和 push-pull unlearning 的叙事。

建议至少跑：

```yaml
inpainting:
  mode: full

unlearning:
  forget_weight: 0.1
```

并做 `forget_weight = 0.0 / 0.1 / 0.2 / 0.3` 的敏感性分析。

### P1: 构造 Primary-heavy stress test

随机删除中 Primary Hub 触发率太低，不足以证明 DAR。必须构造专门的 stress test：

```text
forget_set = top-k Primary Hub nodes by HubScore
```

目标：

```text
DAR trigger rate >= 30%
```

否则 DAR 很难成为核心贡献。

### P1: 补结构指标

故事线强调 structural scars，因此不能只报 accuracy。至少需要：

- degree KL divergence；
- clustering coefficient change；
- component count change；
- hub embedding drift；
- neighbor embedding drift。

### P2: 补 edge / feature unlearning

edge / feature 不要作为主卖点，但需要支持 scope claim。可以先在 PubMed 上低成本跑，再在 PPI 或 PrimeKG 上增强。

---

## 10. 推荐标题

### 标题 1：最稳妥

**HASI: Hub-Anchored Structural Inpainting for Privacy-Preserving Medical Graph Unlearning**

优点：  
清楚、完整，保留方法名和医疗场景。

---

### 标题 2：更强调问题

**Structure-Aware Medical Graph Unlearning via Hub Anchoring and Structural Inpainting**

优点：  
强调 structure-aware，适合方法论文。

---

### 标题 3：更强调隐私

**Privacy-Preserving Medical Graph Unlearning with Hub Anchoring and Distributed Structural Repair**

优点：  
适合投安全、隐私或医疗 AI 交叉方向。

---

## 11. 摘要雏形

> Medical graph learning has become increasingly important for clinical prediction, therapeutic discovery, and biomedical knowledge reasoning. However, medical graphs are sensitive, dynamic, and regulated: patients may withdraw consent, institutions may request data removal, biomedical relations may be corrected, and sensitive features may become unavailable. These scenarios require graph unlearning, which removes the influence of requested nodes, edges, or features without retraining from scratch. Existing graph unlearning methods mainly focus on partitioning, influence approximation, or gradient-based correction, but often overlook two medical-graph-specific challenges: damage to clinically meaningful hubs and structural holes left by deletion. These effects can degrade utility and create privacy-observable structural scars.  
>
> This paper proposes HASI, a hub-anchored structural inpainting framework for privacy-preserving medical graph unlearning. HASI identifies structurally important medical hubs, stabilizes them as hierarchical anchors, localizes the affected region through effective receptive fields, repairs deletion-induced structural holes through generative inpainting, and uses distributed anchor replacement when a primary hub itself must be removed. We evaluate HASI on medical graph datasets under node, edge, and feature unlearning requests, using utility, structural fidelity, efficiency, and membership inference risk as evaluation dimensions. The results aim to show that structure-aware unlearning provides a more reliable privacy-utility trade-off for medical graph learning systems.

---

## 12. 需要避免的写法

### 不建议 1：把 Universal 放在主标题

不建议：

> Universal Graph Unlearning with HASI

问题：  
会稀释“医疗图 + 结构感知 + 隐私”的主线。

建议：

> Privacy-Preserving Medical Graph Unlearning with Hub-Anchored Structural Inpainting

---

### 不建议 2：过度攻击现有方法

不建议：

> Existing methods fail completely.

建议：

> Existing methods can improve efficiency, but they often overlook hub damage and deletion-induced structural scars in medical graphs.

---

### 不建议 3：只报告 accuracy

如果故事线强调结构和隐私，只报告 accuracy 会让方法动机不成立。至少要同时报告：

- utility；
- structural fidelity；
- privacy；
- efficiency。

---

### 不建议 4：DAR 只放在随机删除实验里

随机删除中 DAR 触发率可能太低。DAR 必须通过 Primary-heavy stress test 单独验证。

---

## 13. 最终故事线总结

最终可以把论文主线压缩成下面这句话：

> Medical graph unlearning must remove sensitive entities, relations, or features without damaging the clinical or biological structure that supports prediction. Existing update-centric methods often overlook clinically meaningful hubs and deletion-induced structural scars. HASI treats medical graph unlearning as structure-aware surgical editing: it stabilizes medical hubs, localizes affected neighborhoods, repairs structural holes, and distributes anchor replacement when a primary hub must be deleted, thereby improving the trade-off among utility, structural fidelity, efficiency, and privacy.

---

## 14. 下一步执行清单

### 写作层面

- [ ] 将 Introduction 改成医疗图背景；
- [ ] 将 Universal 降为 scope sub-claim；
- [ ] 将主要贡献改成 structure-aware + privacy；
- [ ] 增加一个 motivation figure；
- [ ] 增加一个 HASI framework figure；
- [ ] 把 RQ 改成医疗场景驱动。

### 实验层面

- [ ] PubMed 上跑 random vs hub deletion；
- [ ] 给所有 baseline 补 MIA；
- [ ] 重跑 HASI-full；
- [ ] 做 anchor / ERF / GSI 消融；
- [ ] 做 Primary-heavy DAR stress test；
- [ ] 增加 PPI 或 ogbn-proteins；
- [ ] 尝试 PrimeKG / TDC-style medical KG case study；
- [ ] 补 3 seeds mean ± std。

### 结果组织层面

- [ ] 主表同时放 utility / structure / privacy / efficiency；
- [ ] 单独画 DAR Pareto curve；
- [ ] 单独画 inpainting 对 structural scars 的影响；
- [ ] 单独画 Hub deletion fragility；
- [ ] 把 edge / feature 作为 scope 表，而不是最大卖点。

---

## 15. 参考资料与依据

本文档主要基于以下内容整理：

1. 当前 HASI 项目文档：`PROJECT_OVERVIEW.md`
2. 当前 HASI 方法文档：`HASI_Framework.md`
3. PyTorch Geometric Planetoid / PubMed dataset documentation
4. PyTorch Geometric PPI dataset documentation
5. OGB ogbn-proteins dataset documentation
6. Therapeutics Data Commons and PrimeKG resources

