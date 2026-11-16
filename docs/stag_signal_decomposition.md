# STAG 信号分解复现与协同结构对照

## 1. 研究目的

本阶段在原始触觉压力序列中分离手部姿态信号与物体相关压力信号，并比较分解前后的传感器相关结构。研究问题是：此前基于 hasValidLabel 的完整接触帧计算得到的相关矩阵，是否同时包含显著的手部姿态共同变化？

## 2. 原论文方法

Sundaram et al. (2019), *Nature*, “Learning the signatures of the human grasp using a scalable tactile glove”, DOI: 10.1038/s41586-019-1234-z，Methods 的 “Decomposing signal and sensor correlations” 描述：

1. 每个 Recording 计算逐帧平均压力响应；
2. 使用标准差为 3 帧的对称 Gaussian 核平滑；
3. 对平滑信号计算前向差分梯度，以五帧对称窗口定位梯度局部极大值；
4. 在候选接触前最多 20 帧内寻找最近的压力局部极小值；
5. 接触前最小值和后续局部最大值均需满足相对于 Recording 动态范围的 20% 阈值；
6. 以局部最大值为中心，向前后各扩展最多 80 帧，直到信号不再满足 0.5 检测阈值；
7. 从物体操作帧中减去接触前的空手姿态帧，得到物体相关压力。

论文将手部姿态近似视为在接触前达到充分弯曲后短时间内饱和，并假设物体接触产生的附加法向压力可以叠加在该姿态信号上。因此，基线相减是有物理假设的近似分解，不是无条件成立的去噪操作。

## 3. 官方事件标记与实现边界

本次重新解压了 pressure-only classification_lite.zip，读取其 metadata.mat。原始 MAT 字段为 isGrasp；官方 HTML README 将其拼写为 isGrap。官方 README 说明：

- isTransition 标记接触前的空手姿态帧；
- isGrap 标记用于信号分离的可靠物体接触事件；
- transitionDir 在 MAT 中与事件锚点配套存储。

本实现将 isTransition 作为接触前锚点，并将 transitionDir 解释为从该锚点向前到关联接触峰值的帧偏移；isGrasp 用于选取可靠事件。该偏移解释与逐段压力曲线相符，但公开 README 未给出 transitionDir 的正式字段定义，因此在代码中应视为依据数据结构和曲线检查得到的操作性解释。

**重要边界：** 原论文没有公开其事件检测实现的完整源代码。我们使用数据集提供的官方事件标记恢复事件锚点，而不是声称仅凭论文文字完整复刻了内部局部极值检测器。论文对 0.5 帧有效性阈值的归一化分母也没有完全说明，因此同时报告两种解释。

## 4. 实现协议

### 4.1 压力预处理

- 只保留 548 个物理传感器像素；
- 按论文分类预处理将原始值 500–650 映射到 0–1，并截断到该范围；
- 每帧对 548 个传感器取平均；
- 使用 scipy.ndimage.gaussian_filter1d，sigma=3，reflect 对称边界。

### 4.2 事件与基线

每个可靠事件包含：接触前基准索引（isTransition）、接触峰索引（基准索引加 transitionDir）和可靠性（同一事件锚点处 isGrasp=True）。手部姿态样本为接触前基准帧的 548 维压力向量。

### 4.3 0.5 阈值的两种可复现解释

论文写明了“normalized scale”上的 0.5 阈值，但没有用公式明确给出其分母。结合前文两项事件筛选阈值均以 Recording dynamic range 定义，当前将 **recording-relative** 作为更贴近原文的主分析解释：

- **recording-relative（主分析候选）：** 对每个 Recording 的平滑均值信号使用全局最小值和最大值归一化，\(z(t)=(s(t)-\min s)/(\max s-\min s)\)，以 0.5 为阈值；
- **event-relative（敏感性分析）：** 对每个事件使用接触前基线至接触峰值的上升量归一化，\(z_e(t)=(s(t)-s(t_0))/(s(q)-s(t_0))\)，以 0.5 为阈值。

这是有依据的操作性选择，不应写成论文明确给出的公式。两种模式均从接触峰值向前后扩展，最多各 80 帧，遇到低于阈值的帧即停止。

### 4.4 基线相减

对事件窗口内每帧压力向量 x(t)，以及接触前基线帧 x(t0)，定义：

x_object(t) = x(t) - x(t0)

不对负值截断；负值表示相对于基线该传感器压力读数下降。

## 5. 真实数据结果

审查范围：26 个非空物体类别、78 个非空手 Recording、548 个传感器。

| 指标 | recording-relative（主分析） | event-relative（敏感性） |
|---|---:|---:|
| 可靠事件数 | 448 | 448 |
| 成功构造窗口 | 317 | 446 |
| 展开窗口帧数 | 5,337 | 6,669 |
| 去重窗口帧数 | 4,999 | 6,652 |
| 窗口长度中位数 | 12 | 11 |
| 窗口长度均值 | 16.84 | 14.95 |
| 窗口长度范围 | 2–132 | 3–95 |

event-relative 模式的 6,652 个唯一窗口帧中，有 6,514 帧属于 hasValidLabel，即窗口内 97.9% 的帧也被原始接触筛选标记为有效；这些窗口只覆盖全部 hasValidLabel 帧的 8.23%。该比例只描述 event-relative 版本，不应移用到 recording-relative 主分析。

### 5.1 相关结构对照

以每个 Recording 为统计单位。每个相关矩阵包含 548 个传感器节点和 149,878 个无向传感器对。边内不视为独立样本。下表报告两种阈值归一化规则下的 Recording 级中位数：

| 对照 | Spearman：recording / event | Jaccard（tau=0.7）：recording / event | 平均绝对相关差：recording / event |
|---|---:|---:|---:|
| 全部 hasValidLabel 原始帧 vs 事件窗口原始帧 | 0.309 / 0.414 | 0.057 / 0.072 | 0.238 / 0.221 |
| 事件窗口原始帧 vs 基线相减后物体压力 | 0.590 / 0.591 | 0.461 / 0.470 | 0.215 / 0.198 |
| 全部 hasValidLabel 原始帧 vs 基线相减后物体压力 | 0.217 / 0.274 | 0.048 / 0.063 | 0.271 / 0.250 |
| 手部姿态基线 vs 物体压力 | 0.184 / 0.196 | 0.030 / 0.033 | 0.403 / 0.396 |

Spearman 使用每个 Recording 中最多 20,000 条共同有效边的固定随机种子子样本；平均绝对差、边密度和 Jaccard 使用全部共同有效边。Recording 是汇总单位，不能把全部边数当作独立样本量。上表中 recording-relative 对照覆盖 70 个可比较 Recording；event-relative 对照覆盖 74 个。手部姿态与物体压力的比较分别覆盖 60 和 63 个 Recording。

### 5.2 正边密度

下表为先对每个 Recording 计算正边比例、再对 Recording 求均值：

| 数据 | recording-relative：r>=0.5 / 0.7 / 0.9 | event-relative：r>=0.5 / 0.7 / 0.9 |
|---|---:|---:|
| 全部 hasValidLabel 原始帧 | 3.04% / 0.573% / 0.0235% | 同左 |
| 事件窗口原始帧 | 9.53% / 3.53% / 0.507% | 9.31% / 3.24% / 0.401% |
| 基线相减后物体压力 | 10.75% / 3.62% / 0.447% | 9.69% / 2.99% / 0.318% |
| 手部姿态基线帧 | 16.94% / 7.61% / 1.75% | 同左 |

手部姿态相关矩阵分别覆盖 60/63 个 Recording；物体压力相关矩阵分别覆盖 70/74 个 Recording（recording-relative / event-relative）。


### 5.3 事件窗口可视化抽查

对 448 个可靠事件逐一计算两种归一化窗口，并对代表性样本绘制平滑均值曲线、基线锚点、接触峰值锚点和 0.5 阈值。可视化结果见本轮生成的 `event_window_visual_audit.png`。

关键审查结果：

- recording-relative 成功构造 317/448 个窗口；其余 131 个接触峰值在 Recording 全局归一化尺度上的值低于 0.5。
- 这 131 个被全局阈值排除的峰值帧，全部仍被数据集的 `hasValidLabel=True` 标记。这说明阈值会排除一部分数据集已标记为物体持有的接触事件，必须在论文中明确其保守筛选效果。
- event-relative 成功构造 446/448 个窗口；另 2 个事件的接触峰值没有高于接触前基线。
- 两种窗口只有 54 个完全相同。对两种规则均成功构造的事件，区间 IoU 中位数为 0.846，但存在明显离群：RID 49（tape）一个事件的 recording-relative 窗口长 132 帧，而 event-relative 仅 7 帧；RID 22（screw_driver）一个事件分别为 4 帧和 88 帧。
- RID 33（cat）的一个事件在全局归一化峰值仅 0.138，因此 recording-relative 不构造窗口，而 event-relative 构造 8 帧；RID 20（bracket）的一个事件全局峰值为 0.498，处于 0.5 阈值边缘。

这些例子表明，归一化规则不只是小幅调整窗口边界，而会改变事件纳入与持续接触时长。当前仍以 recording-relative 作为主分析候选，因为它与论文前文的 Recording dynamic range 叙述更一致；event-relative 只作为敏感性分析。由于论文未明确给出 0.5 的归一化分母，不能声称这一选择已被原作者代码证实。

### 5.4 缺少官方事件标记的 Recording

78 个非空手 Recording 中，4 个没有任何 `isTransition` 或 `isGrasp` 标记：

| Recording | 物体 | hasValidLabel 帧数 |
|---|---|---:|
| 026-chain | chain | 597 |
| 046-chain | chain | 785 |
| 067-chain | chain | 1,927 |
| 080-kiwano | kiwano | 1,059 |

四个 Recording 合计仍有 4,368 个 `hasValidLabel` 帧，但无法按官方锚点构造接触前基线。缺失并非“没有接触”，而是缺少本分解方案所需的官方事件定位信息。可视化见 `no_marker_recordings_audit.png`。

对这些 Recording 试运行基于论文文字的梯度/局部极值候选检测，分别产生 24、41、107、47 个候选。由于没有视频或人工事件标注验证，候选数不能视为真实抓取次数，也不能直接补入主分析。当前处理原则是：**不以 hasValidLabel 基线或未经核验的自动候选替代官方事件；4 个 Recording 从分解主分析中排除，但保留在原始 hasValidLabel 对照分析中。** 这会使分解主分析覆盖 74 个 Recording、25 个物体类别；chain 类别没有可用的分解 Recording，kiwano 保留 2 个。

## 6. 结果解释

1. **事件窗口筛选本身已经改变关系结构。** 全部 hasValidLabel 帧与事件窗口原始帧的正边集合 Jaccard 中位数只有 0.072。这说明持续接触期的全部数据和接触事件附近的数据不能直接混为同一种标签来源。
2. **基线相减进一步改变边结构。** 在完全相同的事件窗口上，原始压力与基线相减压力的边 Spearman 中位数为 0.591，tau=0.7 的 Jaccard 中位数为 0.470。分解不是简单的幅值平移；不同传感器的基线响应不同，因而相关结构会改变。
3. **手部姿态与物体压力的关系结构不同。** 手部姿态基线与物体压力的边相关 Spearman 中位数为 0.196，Jaccard 中位数为 0.033。这与将两种信号分开分析的研究动机一致，但仍是描述性证据。
4. **不能据此断言分解后标签更准确。** 目前没有独立的物体压力真值，也没有人工逐帧标注全部事件。结果证明分解显著改变了目标，而不是证明改变后的矩阵一定更接近真实物理协同。
5. **0.5 阈值归一化方式会改变纳入事件数量。** recording-relative 317 个窗口，event-relative 446 个窗口；131 个被全局阈值排除的峰值仍属于 hasValidLabel 帧。当前以 recording-relative 为主分析候选，但仍须保留阈值歧义和敏感性结果。\n6. **事件标记缺失不是无接触。** 4 个 Recording 有 4,368 个有效接触帧却没有官方基线锚点；不能将其直接视为无抓取，也不能用未验证检测器自动补标。

## 7. 对论文主任务的建议

下一版协同关系主标签建议从“所有 hasValidLabel 原始接触帧上的 Pearson 相关”调整为“可靠事件窗口内，基线相减后的物体相关压力 Pearson 相关”。原始 hasValidLabel 相关矩阵保留为对照/消融条件。

但在正式锁定主标签前，仍需：
- 尽可能获取同步视频或人工标注，核验 131 个全局阈值未通过事件及两种窗口的离群案例；
- 研究 4 个缺少官方事件标记的 Recording 是否能通过人工复核补标；在此之前不混用自动候选和官方标记；
- 评估排除 chain 类别及 kiwano 仅剩 2 个 Recording 对 Support/Query 任务设计的影响；
- 检查每个物体仅 3 个 Recording 带来的 Support/Query 任务数量限制；
- 在物体级划分下重新估计标签稳定性，避免 Recording 级官方 split 被误用为跨物体泛化评估。

## 8. 复现命令

```bash
python scripts/run_stag_signal_decomposition.py \
  --metadata data/interim/metadata.mat \
  --output-dir results/stag_decomposition \
  --sigma 3 --radius 80 --threshold 0.5
```

输出 JSON 汇总、Recording 级 CSV、事件窗口级 CSV。脚本不保存原始压力矩阵或逐边标签。
