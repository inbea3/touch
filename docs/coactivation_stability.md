# 协同关系标签稳定性与估计方法比较实验

## 1. 目的与研究边界

本实验只使用 STAG pressure-only 数据及其 metadata.mat，不使用同步视频或图像。研究对象是基于压力观测构造的传感器统计关系，不将相关性解释为因果或已验证的物理协作。

本轮检查事件窗口重复帧是否改变估计，比较 Pearson、Spearman 和 Ledoit-Wolf 收缩相关，并通过时间块 Bootstrap 与事件级 Bootstrap 评估关系估计的敏感性。

统计汇总单位为 Recording（78 个），物体（26 个）是外层泛化单位。每个 Recording 内的 149,878 条传感器边彼此相关，不能当作独立样本。

## 2. 数据与共同协议

- 26 个非空物体类别、78 个非空手 Recording；
- 548 个有效传感器，每个 Recording 有 149,878 条无向候选边；
- 压力按 (raw - 500) / 150 映射到 0–1 并截断；
- 主预测任务沿用最早 5 个 hasValidLabel=True 帧作为早期输入；
- 目标为同一 Recording 中严格晚于第 5 个输入帧的有效接触帧；
- 事件分解仅保留峰值晚于输入边界的可靠官方事件，且将事件窗口裁剪到输入边界之后，避免目标窗口回看早期输入之前的帧；
- Recording-relative 0.5 阈值作为事件窗口主分析候选。

## 3. 方法

### 3.1 窗口重叠与重复计数

对事件窗口分别构造 expanded（按事件拼接，允许同一帧重复出现）和 unique（按原始帧索引取并集）两种数据。对原始压力比较两种相关矩阵。

基线相减后的物体压力中，同一帧可能对应不同事件基线，不能仅按帧索引直接去重。因此另比较 expanded pooled 估计与 event-balanced 估计：每个事件先计算相关矩阵，再对 Fisher-z 相关作等事件权重平均。

### 3.2 相关性估计

- Pearson：主估计；
- Spearman：逐传感器秩变换后计算 Pearson 相关；
- Ledoit-Wolf：估计收缩协方差，再标准化为相关矩阵。

常数传感器及未定义边记为 NaN。比较边相关 Spearman、平均绝对差及 r≥0.7 正边 Jaccard。

### 3.3 时间块 Bootstrap

对每个 Recording 的未来有效接触序列使用 circular moving-block Bootstrap，块长 10 帧、20 次重复；为控制计算量，使用固定随机种子抽取 96 个传感器节点。每次重抽样比较 Pearson 相关矩阵。

该结果是低重复次数、节点子样本上的探索性稳定性诊断，不是对全部 149,878 条边的正式区间推断。

### 3.4 事件级 Bootstrap

对至少含 5 帧的基线相减事件窗口，分别计算事件级 Pearson 相关矩阵；将相关系数转换为 Fisher-z 后，对事件有放回重抽样 100 次，再转换回相关系数，估计 95% 百分位区间。该步骤使用固定随机种子抽取 64 个传感器节点。

每个 Recording 的事件数较少，且同一 Recording 内事件可能相关；因此事件级 Bootstrap 仅作为事件构成敏感性分析。

## 4. 实验结果

### 4.1 窗口重叠

70 个至少构造出一个合格未来事件窗口的 Recording 中，只有 1 个出现窗口重叠，即 Recording 049（tape）：

- expanded 事件帧出现次数：634；
- unique 帧数：296；
- 重复出现比例：53.3%。

其余 69 个 Recording 没有重复帧。按 70 个 Recording 汇总，重复出现比例均值为 0.76%，中位数为 0。

对原始压力，expanded 与 unique 相关矩阵的 Recording 级边 Spearman 中位数为 1.000，r≥0.7 正边 Jaccard 中位数为 1.000。重叠对绝大多数 Recording 的总体关系矩阵影响很小，但 tape 这一 Recording 必须去重或单独做敏感性报告。

对基线相减后的压力，expanded pooled 与 event-balanced 矩阵差异明显：边 Spearman 中位数为 0.213，r≥0.7 正边 Jaccard 中位数为 0.018。主要原因不只是重复帧，而是 pooled 方法按事件窗口长度加权，长事件贡献更多；event-balanced 方法让每个事件等权。两者对应不同统计目标，不能互相替代。

### 4.2 相关性估计方法

基于 78 个 Recording 的未来有效接触压力：

| 指标 | Recording 级中位数 |
|---|---:|
| Pearson 与 Spearman 的边相关 Spearman | 0.847 |
| Pearson 与 Ledoit-Wolf 的边相关 Spearman | 0.9997 |
| Pearson 与 Ledoit-Wolf 的平均绝对相关差 | 0.0083 |
| Ledoit-Wolf 收缩系数 | 0.0254 |

Pearson 与 Ledoit-Wolf 在当前数据规模下几乎一致；Spearman 与 Pearson 存在更明显但仍较高的一致性。当前证据不支持将 Ledoit-Wolf 作为主标签估计器；Pearson 作为主估计、Spearman 作为稳健性对照更简洁，也更容易解释。

### 4.3 时间块 Bootstrap

20 次重复、10 帧块长、96 个固定随机节点子样本的 Recording 级中位数：

| 指标 | 中位数 |
|---|---:|
| 重抽样相关矩阵边 Spearman | 0.925 |
| 平均绝对相关差 | 0.0459 |
| r≥0.7 正边 Jaccard | 0.539 |

连续相关强度的整体排序具有一定稳定性，但二元阈值边集合的稳定性明显较弱。20 次重复和 96 个节点仅用于先导诊断，后续正式分析需增加重复次数，并比较不同块长。

### 4.4 事件级 Bootstrap

事件级分析对每个 Recording 至少需要两个合格事件级矩阵，因此 78 个 Recording 中有 60 个可计算区间。每个 Recording 可用事件数均值为 3.91，中位数为 4。

在 64 个固定随机传感器节点上，事件级相关矩阵 95% 区间宽度的 Recording 级中位数为 0.440；区间不包含 0 的边比例中位数为 25.6%。

这说明单个短事件窗口的相关矩阵估计不稳定，事件间差异较大。该结果不能被解释为真实物理边的显著性检验，因为事件级窗口很短、事件数量少，且事件之间并不保证独立。

## 5. 研究判断与当前决策

1. 事件窗口重叠不是全数据集普遍问题，但 tape Recording 存在严重重叠，代码必须显式报告 expanded 与 unique 两种结果。
2. Pearson 与 Ledoit-Wolf 在当前任务下几乎等价，没有足够证据增加收缩估计作为主标签；保留为方法对照即可。
3. Spearman 可作为稳健性对照，但不应与 Pearson 的数值含义混淆。
4. 时间块 Bootstrap 表明连续相关矩阵整体排序尚可，但 r≥0.7 的二元边标签稳定性一般；主任务应继续使用连续相关强度，二元边只作为辅助分析。
5. 事件级等权平均与 pooled 事件帧估计明显不同，且事件级区间较宽。现阶段不应把 event-balanced 标签替换为主标签。
6. 当前更稳妥的主标签仍是未来 hasValidLabel 有效接触帧上的连续 Pearson 相关矩阵；基线相减后的物体压力相关矩阵保留为重要的候选替代标签与消融分析。

## 6. 局限与下一步

- 事件级 Bootstrap 使用 64 个固定随机传感器节点；时间块 Bootstrap 使用 96 个节点，均不是全部边的区间推断。
- 时间块 Bootstrap 仅 20 次，不能作为最终区间报告；应提高到至少 200 次，并比较块长 5、10、20 帧。
- 事件级 Bootstrap 的事件数少，且事件窗口最短 5 帧；区间宽度主要反映当前事件级估计的高不确定性。
- 本轮没有进行物体级 Bootstrap，也没有把 26 个物体视为大样本；最终跨物体性能不确定性必须在物体层面评估。
- 尚未验证不同 Recording 的采样时间间隔完全一致；块长 10 帧是帧尺度敏感性参数，不应直接等同于固定秒数。

下一步先完成 Bootstrap 块长敏感性（5/10/20 帧）与重复次数增加，再冻结主标签和物体级 Support/Query 划分。此之前不进入 GNN/MAML 主实验。

## 7. 复现命令

\`\`\`bash
python scripts/run_coactivation_stability.py --metadata data/interim/metadata.mat --output-dir results/coactivation_stability --k-early 5 --block-length 10 --bootstrap-reps 20 --bootstrap-nodes 96

python scripts/run_event_cluster_bootstrap.py --metadata data/interim/metadata.mat --output-dir results/coactivation_stability --k-early 5 --min-event-frames 5 --bootstrap-reps 100 --bootstrap-nodes 64
\`\`\`

原始压力矩阵不写入结果文件，也不提交仓库。
