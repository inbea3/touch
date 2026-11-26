# TouchScale 跨类别少样本结构适应 Pilot：结果复核与泄漏审计（2026-11-26）

## 1. 目的与数据来源

本报告复核 GitHub Actions 成功运行 `45770119803` 的原始产物，文件来自 Artifact `15094653126`（`touchscale-cross-object-fewshot-pilot.zip`）。分析在下载后的 CSV 上完成，不重新训练模型。

本轮成功运行耗时 1,330.28 秒（约 22 分 10 秒）。实际纳入 25 条有效 Episode、7 个归一化任务类别；另有 1 条无法在官方 metadata 索引中核验的候选记录被排除并留痕。原始压力数组未上传到 Artifact。

任务定义：
- 输入：Episode 前 40% 时间窗的 15 个 Pad 压力摘要（45 维）。
- 目标：Episode 后 40% 时间窗内 15 个 Pad 的 Pearson 相关矩阵上三角（105 维）。
- 外层：Leave-One-Normalized-Category-Out（LOCO）。
- Support/Query：按完整 Episode 隔离。
- 对照：Reptile 初始化与 pooled MLP；名义梯度更新预算相同。
- 随机种子：17、29、43。

## 2. 总体指标复核

| Shot | 方法 | 可评估 held-out 类别 | 配对评估记录 | 平均 Query Episode 数 | 零样本 MAE | 适应后 MAE | 适应后 RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| 1-shot | pooled MLP | 7 | 21 | 2.57 | 0.31412 | 0.30261 | 0.43153 |
| 1-shot | Reptile | 7 | 21 | 2.57 | 0.22558 | 0.22072 | 0.29704 |
| 2-shot | pooled MLP | 5 | 15 | 2.20 | 0.40539 | 0.38887 | 0.58752 |
| 2-shot | Reptile | 5 | 15 | 2.20 | 0.22656 | 0.21972 | 0.30233 |

以汇总均值计算，Reptile 适应后 MAE 相对 pooled MLP：
- 1-shot：降低约 27.1%（绝对差 0.08189）。
- 2-shot：降低约 43.5%（绝对差 0.16914）。

注意：Reptile 在零样本阶段已经有明显较低的 MAE（1-shot 0.22558 vs 0.31412；2-shot 0.22656 vs 0.40539）。因此当前观察到的优势不能全部归因于“少量 Support 带来的适应增益”；它同时反映元训练后初始化本身的预测优势。两种方法从零样本到适应后的 MAE 变化都较小，论文必须分别报告初始化能力与适应增量。

## 3. 类别与随机种子敏感性

以同一 held-out category × seed × shot 配对 Reptile 与 pooled MLP：
- 1-shot：21 组中 Reptile MAE 更低 18 组；按类别先平均三个 seed 后，7 个类别中 6 个更低、1 个更高。
- 2-shot：15 组中 Reptile MAE 更低 15 组；按类别先平均三个 seed 后，5 个可评估类别全部更低。
- 1-shot 的 lever switch 是明确反例：pooled MLP 平均 MAE 0.22315，Reptile 0.23398，Reptile 高 0.01084。
- 1-shot 收益较大的类别包括 container closure（差值 Reptile−MLP = −0.18544）与 hex nut（−0.13230）；2-shot container closure 的差值为 −0.47601，对总体平均贡献较大。

类别平均差值（Reptile MAE − pooled MLP MAE）：

| Shot | 类别 | pooled MLP MAE | Reptile MAE | 差值 |
|---|---|---:|---:|---:|
| 1 | A4 paper sheet set | 0.22186 | 0.16680 | −0.05506 |
| 1 | container closure (lid/cap) | 0.53649 | 0.35105 | −0.18544 |
| 1 | hex nut | 0.33676 | 0.20447 | −0.13230 |
| 1 | lever switch | 0.22315 | 0.23398 | +0.01084 |
| 1 | lid / latch component | 0.32688 | 0.21366 | −0.11322 |
| 1 | shipping label | 0.19646 | 0.15620 | −0.04026 |
| 1 | zipper / zipper pull | 0.27667 | 0.21890 | −0.05777 |
| 2 | A4 paper sheet set | 0.21635 | 0.17956 | −0.03679 |
| 2 | container closure (lid/cap) | 0.83494 | 0.35894 | −0.47601 |
| 2 | hex nut | 0.33420 | 0.20050 | −0.13370 |
| 2 | lid / latch component | 0.35206 | 0.21094 | −0.14112 |
| 2 | shipping label | 0.20678 | 0.14867 | −0.05811 |

本报告不把 seed×category 的 21/15 条记录视为完全独立样本，也不把逐传感器边作为独立统计单位。可计算的常规配对检验仅作探索性参考，不作为论文确认性显著性证据：同一 Episode 会在不同 seed 组合中重复出现，类别数仅 7/5，且类别任务并非随机抽取的独立总体样本。后续统计推断应以类别/物体/参与者等真正独立单位为基础，并在扩大任务池后重做。

## 4. 参与者重叠审计：发现关键泛化边界

将成功运行的 `episode_signal_audit.csv` 与 `usable_task_pool.csv` 按 Episode ID 合并，得到 25 条记录、13 个 manifest actor 标识。类别层面的参与者数与 LOCO 留出类别在训练类别中出现过的 actor 数如下：

| Held-out 类别 | Episode 数 | Actor 数 | 同时出现在其他训练类别的 Actor 数 |
|---|---:|---:|---:|
| A4 paper sheet set | 3 | 2 | 2 |
| container closure (lid/cap) | 4 | 2 | 0 |
| hex nut | 5 | 3 | 1 |
| lever switch | 2 | 2 | 1 |
| lid / latch component | 5 | 5 | 2 |
| shipping label | 4 | 2 | 1 |
| zipper / zipper pull | 2 | 2 | 2 |

13 个 actor 中，9 个只出现在一个类别；3 个出现在两个类别；1 个出现在三个类别。也就是说，LOCO 确实留出了类别，但并没有保证参与者身份完全隔离：7 个 held-out 类别中有 6 个存在 actor 与训练类别重叠。

这不证明模型一定发生了参与者泄漏，但意味着当前结果只能支持“在当前采样协议下对未见归一化类别的探索性迁移”，不能支持“对未见参与者的泛化”。如果压力模式存在个体差异，模型可能利用了参与者相关特征。

Actor/date 信息来自本轮人工整理的候选 manifest/audit 字段；当前 authoritative metadata 自动核验的字段为 Episode ID 与 shard。物理对象实例 ID 尚未建立，因此 normalized category 不能等同于独立物理对象身份。

## 5. 数据质量与任务规模限制

- 25 条 Episode 分布为：hex nut 5、lid / latch component 5、shipping label 4、container closure 4、A4 paper sheet set 3、lever switch 2、zipper / zipper pull 2。
- 1-shot 可评估 7 个 held-out 类别；2-shot 可评估 5 个；5-shot 不可评估。没有重复抽样或复制 Query 来凑 5-shot。
- 信号审计 CSV 中 25/25 状态为 ok，但日志出现 NumPy 相关矩阵除以零的 RuntimeWarning。特征函数把非有限相关值替换为 0，因此流程完成不代表每条 Episode 的每个 Pad 对都具有可估计相关系数。下一轮应显式记录零方差 Pad 数、NaN 相关边比例，并对全零/近常数通道做质量阈值审查。
- 当前任务来自 150 行分层 metadata 候选审查，不是全数据集的任务总体普查。
- 目标是 Episode 内未来时间窗的 Pad-Pad Pearson 关联，不是因果接触图，也不是物理结构真值。
- 当前 15 Pad 子集的空间选择和代表性需要另行说明；结果不能外推到完整触觉阵列。

## 6. 结论边界

目前可说：
1. 在本轮 25 Episode、7 类别的探索性任务池中，Reptile 相较 pooled MLP 显示较低的 held-out MAE。
2. 优势在多数类别出现，2-shot 可评估类别方向一致；1-shot 的 lever switch 是反例。
3. Reptile 零样本 MAE 已明显较低，需区分元初始化的预测优势与 Support 适应本身的增量。

目前不可说：
1. 已证明 Reptile 对所有物体或机器人操作任务有效；
2. 已证明参与者无关的泛化；
3. 已证明 5-shot 有效；
4. 已证明预测到真实物理/因果接触结构；
5. 当前 p 值可作为确认性显著性证据。

## 7. 下一步执行决策

**暂停扩展模型或增加 GNN。先扩大和重构任务池，并把参与者独立性纳入数据设计。**

建议按以下顺序执行：
1. 只扩展/扫描官方 metadata 索引，不立即下载全量压力分片；确认全数据的 Episode、actor、日期、类别字段与映射质量。
2. 在 metadata 层面建立候选任务池，要求每个目标类别有多个独立 Episode，并尽可能覆盖多个 actor；区分同一 actor 的重复操作与真正独立参与者。
3. 预先设计两种评估：A. leave-one-category-out（类别迁移）；B. actor-disjoint split（参与者隔离）。若数据支持，再设计 category-and-actor-disjoint 的严格评估。
4. 对可行任务仅下载小规模必要 shard，先复现一轮最小 pilot；增加显式零方差/NaN 边质量审计。
5. 在更大且更独立的任务池上重做 Reptile 与 Support-matched MLP 配对比较；按类别/参与者汇总，不按传感器边做显著性检验。
6. 只有结构任务与元学习优势经独立性审计后仍成立，再考虑 5-shot、更多随机种子或 GNN 结构增量。

本报告结论为探索性审计，不替代完整数据集的独立确认性评估。

## 8. 复现信息

- Repository: `inbea3/touch`
- Branch: `thesis-few-shot-structure-20261125`
- Successful workflow: `45770119803`
- Artifact ID: `15094653126`
- Artifact: `touchscale-cross-object-fewshot-pilot.zip`
- Script: `scripts/run_touchscale_cross_object_fewshot_pilot.py`
- Workflow: `.github/workflows/touchscale-cross-object-fewshot-pilot.yml`
