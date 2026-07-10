# PubMed Feature Tuned 结果分析

更新时间：2026-06-10

本文档整理当前 `results/pubmed/default_main` 下 PubMed feature unlearning 的 tuned 结果分析。当前读取结果共 30 个 JSON：

```text
2 ratios × 3 seeds × 5 methods = 30
```

方法包括：

- `hasi_tuned`
- `hasi_default`
- `retrain`
- `grapheraser_bekm`
- `grapheraser_blpa`

评价指标解释：

- `acc_drop`：越小越好。
- `f1_drop`：越小越好。
- `MIA AUC`：越接近 0.5 越好。
- `privacy_score`：越高越好。
- `time`：使用 `online_wall_clock_seconds`。

---

## 1. Fixed Forget Sets 分组结果

### 1.1 Group 1：ratio=0.05, seed=42

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | -0.0028 | -0.0030 | 0.5625 | 0.8750 | 143.70s |
| hasi_default | -0.0061 | -0.0067 | 0.5986 | 0.8028 | 143.24s |
| retrain | 0.0068 | 0.0059 | 0.6667 | 0.6665 | 2.94s |
| grapheraser_bekm | 0.0439 | 0.0590 | 0.5784 | 0.8431 | 7.09s |
| grapheraser_blpa | 0.0165 | 0.0152 | 0.5278 | 0.9444 | 7.01s |

小结：`hasi_default` utility 最好，`grapheraser_blpa` privacy 最好。

### 1.2 Group 2：ratio=0.05, seed=123

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | -0.0058 | -0.0071 | 0.6188 | 0.7624 | 144.74s |
| hasi_default | -0.0051 | -0.0050 | 0.6337 | 0.7325 | 143.91s |
| retrain | 0.0033 | 0.0022 | 0.6429 | 0.7143 | 2.95s |
| grapheraser_bekm | 0.0928 | 0.1588 | 0.5906 | 0.8188 | 6.82s |
| grapheraser_blpa | 0.0193 | 0.0199 | 0.5531 | 0.8938 | 6.74s |

小结：`hasi_tuned` utility 优于 default，但 privacy 最好的是 `grapheraser_blpa`。

### 1.3 Group 3：ratio=0.05, seed=2024

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | -0.0013 | -0.0020 | 0.6238 | 0.7524 | 144.40s |
| hasi_default | -0.0015 | -0.0026 | 0.6751 | 0.6498 | 144.15s |
| retrain | 0.0018 | 0.0002 | 0.7033 | 0.5933 | 2.99s |
| grapheraser_bekm | 0.1476 | 0.2954 | 0.5371 | 0.9259 | 6.99s |
| grapheraser_blpa | 0.0160 | 0.0135 | 0.6062 | 0.7877 | 6.48s |

小结：`hasi_default` utility 略好，`hasi_tuned` privacy 比 default 好；全局 privacy 最好的是 `grapheraser_bekm`。

### 1.4 Group 4：ratio=0.1, seed=42

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | -0.0013 | -0.0018 | 0.5284 | 0.9432 | 144.38s |
| hasi_default | 0.0005 | -0.0000 | 0.5116 | 0.9768 | 146.04s |
| retrain | 0.0084 | 0.0074 | 0.6392 | 0.7216 | 2.91s |
| grapheraser_bekm | 0.0548 | 0.0750 | 0.5804 | 0.8391 | 6.78s |
| grapheraser_blpa | 0.0213 | 0.0192 | 0.5263 | 0.9475 | 6.96s |

小结：`hasi_tuned` utility 最好，`hasi_default` privacy 最好。

### 1.5 Group 5：ratio=0.1, seed=123

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | -0.0018 | -0.0033 | 0.5222 | 0.9555 | 145.85s |
| hasi_default | 0.0048 | 0.0028 | 0.5912 | 0.8175 | 144.72s |
| retrain | 0.0068 | 0.0062 | 0.6387 | 0.7226 | 3.01s |
| grapheraser_bekm | 0.0971 | 0.1604 | 0.5923 | 0.8154 | 6.67s |
| grapheraser_blpa | 0.0223 | 0.0226 | 0.5538 | 0.8925 | 7.01s |

小结：这一组 `hasi_tuned` 同时拿到最好的 utility 和 privacy。

### 1.6 Group 6：ratio=0.1, seed=2024

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | 0.0013 | 0.0007 | 0.6267 | 0.7467 | 157.38s |
| hasi_default | -0.0008 | -0.0010 | 0.6818 | 0.6364 | 146.16s |
| retrain | 0.0036 | 0.0025 | 0.7001 | 0.5997 | 2.98s |
| grapheraser_bekm | 0.1473 | 0.3026 | 0.5416 | 0.9168 | 7.02s |
| grapheraser_blpa | 0.0155 | 0.0140 | 0.5842 | 0.8316 | 6.65s |

小结：`hasi_default` utility 更好；`hasi_tuned` privacy 比 default 好，但全局 privacy 最好的是 `grapheraser_bekm`。

---

## 2. 全方法胜出统计

| 指标 | 胜出情况 |
|---|---|
| accuracy_drop 最小 | `hasi_tuned` 3/6, `hasi_default` 3/6 |
| f1_drop 最小 | `hasi_tuned` 3/6, `hasi_default` 3/6 |
| MIA AUC 最接近 0.5 | `grapheraser_blpa` 2/6, `grapheraser_bekm` 2/6, `hasi_default` 1/6, `hasi_tuned` 1/6 |
| privacy_score 最高 | `grapheraser_blpa` 2/6, `grapheraser_bekm` 2/6, `hasi_default` 1/6, `hasi_tuned` 1/6 |
| online time 最短 | `retrain` 6/6 |

---

## 3. Tuned 对比 Default

| 指标 | tuned 胜出 |
|---|---:|
| accuracy_drop | 3/6 |
| f1_drop | 3/6 |
| MIA AUC | 5/6 |
| privacy_score | 5/6 |
| time | 1/6 |

### 3.1 平均结果

| method | acc_drop | f1_drop | MIA AUC | privacy | time |
|---|---:|---:|---:|---:|---:|
| hasi_tuned | -0.0019 | -0.0027 | 0.5804 | 0.8392 | 146.74s |
| hasi_default | -0.0014 | -0.0021 | 0.6154 | 0.7693 | 144.70s |
| retrain | 0.0051 | 0.0041 | 0.6652 | 0.6697 | 2.96s |
| grapheraser_bekm | 0.0972 | 0.1752 | 0.5701 | 0.8599 | 6.89s |
| grapheraser_blpa | 0.0185 | 0.0174 | 0.5586 | 0.8829 | 6.81s |

### 3.2 Tuned vs Default 结论

feature tuned 是有价值的。它相比 default 平均 utility 略好，privacy 明显更好：

- 平均 `acc_drop` 从 -0.0014 改善到 -0.0019。
- 平均 `f1_drop` 从 -0.0021 改善到 -0.0027。
- 平均 `MIA AUC` 从 0.6154 降到 0.5804。
- 平均 `privacy_score` 从 0.7693 提升到 0.8392。

但 tuned 不是每一组都赢 default。utility 上 tuned 与 default 是 3/6 对 3/6；privacy 上 tuned 更稳定，MIA AUC 和 privacy_score 都是 5/6 优于 default。

---

## 4. Tuned 对比其他 Baselines

这里的 baselines 指：

- `retrain`
- `grapheraser_bekm`
- `grapheraser_blpa`

### 4.1 Tuned 是否同时赢过所有 baselines

| 指标 | tuned 胜出 |
|---|---:|
| accuracy_drop | 6/6 |
| f1_drop | 6/6 |
| MIA AUC | 1/6 |
| privacy_score | 1/6 |
| time | 0/6 |

也就是说，`hasi_tuned` 在 utility 上 6/6 全胜所有 baselines，但 privacy 只在 1/6 组同时超过所有 baselines，时间没有赢。

### 4.2 Tuned 分别对比每个 baseline

| 对比对象 | accuracy_drop | f1_drop | MIA AUC | privacy_score | time |
|---|---:|---:|---:|---:|---:|
| tuned vs retrain | 6/6 | 6/6 | 6/6 | 6/6 | 0/6 |
| tuned vs grapheraser_bekm | 6/6 | 6/6 | 3/6 | 3/6 | 0/6 |
| tuned vs grapheraser_blpa | 6/6 | 6/6 | 1/6 | 1/6 | 0/6 |

### 4.3 Tuned vs Baselines 结论

`hasi_tuned` 的 utility 明显最好；相比 `retrain`，utility 和 privacy 都更好。相比 GraphEraser variants，`hasi_tuned` 的 utility 明显更好，但 privacy 不稳定。

GraphEraser-BLPA 和 GraphEraser-BEKM 在若干 fixed conditions 下能取得更强 privacy，尤其 BLPA 在多组中 MIA AUC 更接近 0.5 或 privacy_score 更高；但它们的 utility 损失明显更大。runtime 上 `hasi_tuned` 不占优势。

---

## 5. 总体结论

PubMed feature unlearning 中，`hasi_tuned` 相比 `hasi_default` 提升了平均 utility-privacy trade-off。它在 utility 上稳定超过 `retrain`、`grapheraser_bekm` 和 `grapheraser_blpa`，但在 privacy 上并不总是超过 GraphEraser variants。

因此这批结果适合支持如下表述：

```text
For PubMed feature unlearning, the tuned HASI configuration improves the average utility-privacy trade-off over the default HASI configuration. It consistently outperforms retrain and GraphEraser variants in utility, while GraphEraser-BLPA/BEKM still achieve stronger privacy in several fixed conditions at the cost of much larger utility degradation.
```
