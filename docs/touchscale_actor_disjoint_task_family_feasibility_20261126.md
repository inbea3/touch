# TouchScale 跨参与者任务家族元学习可行性审计（2026-11-26）

## 1. 审计目的

本审计把当前 25-Episode LOCO pilot 的参与者重叠问题放回完整 TouchScale metadata 中评估，判断公开数据是否支持更严格的少样本任务构造。仅使用 metadata，不下载额外 HDF5/tactile shards，不访问视频。

输入产物：
- Full metadata identity audit：workflow `45655312000`，Artifact `15042611555`。
- Meta-task family audit：workflow `45760834326`，Artifact `15088954998`。
- Full task-intent classification：`docs/touchscale_full_task_intent_classification_20261125.md`。
- Metadata geometry audit：`docs/touchscale_metadata_geometry_audit_20261125.md`。

## 2. 全量 metadata 事实

| 项目 | 数量 |
|---|---:|
| Episode | 15,324 |
| 唯一 Episode ID | 15,324 |
| task_id | 953 |
| task_name | 920 |
| scene | 22 |
| actor | 24 |
| objects mesh catalogue 行 | 193 |
| items 非空 Episode | 11,279 |
| items 为空 Episode | 4,045 |
| items 文本提及数 | 38,901 |

注意：task_id 在数据中存在跨 task_name / scene 的碰撞。沿用已有 task-intent 审计规则，任务模板应以 `(scene, task_name)` 作为分组键，不可仅使用 task_id。全量审计发现 663 个 task_id 对应多个 task_name，542 个 task_id 出现在多个 scene；因此任何按 task_id 划分的任务池都必须先解决此问题。

对象身份仍然是限制：
- Episode 的 `items` 是自由文本的 involved-object 列表，不是 canonical primary-object ID。
- objects.parquet 的 193 条记录是 mesh catalogue，不代表每条 Episode 的物理对象实例。
- 38,901 个 item mentions 中，与 mesh catalogue 的 exact-name match 只有 2,565 条；即使精确匹配，也只能作为候选关联，不能证明触摸的是同一个物理实例。
- 因此当前不能把本数据集的结果称为严格的 unseen physical-object-instance adaptation。

## 3. 跨参与者重复任务模板

根据完整的 `task_actor_coverage.csv`，按 `(scene, task_name)` 任务模板汇总（本数据中 920 个组合，与 920 个 distinct task_name 数量一致）：

| 指标 | 结果 |
|---|---:|
| 仅由 1 个 actor 执行的任务模板 | 862 |
| 由 2 个 actor 执行的任务模板 | 56 |
| 由 3 个 actor 执行的任务模板 | 2 |
| 至少 2 个 actor 的任务模板 | 58 |
| 至少 2 个 actor 且每个 actor 至少 2 条 Episode | 50 |
| 至少 2 个 actor 且每个 actor 至少 3 条 Episode | 44 |
| 至少 2 个 actor 且每个 actor 至少 6 条 Episode | 28 |
| 至少 3 个 actor 且每个 actor 至少 2 条 Episode | 1 |
| 至少 3 个 actor 且每个 actor 至少 3 条 Episode | 1 |
| 至少 3 个 actor 且每个 actor 至少 6 条 Episode | 0 |

最稳定的三参与者任务模板为 `Operate the lever switch once`：actor_01 有 8 条、actor_02 有 8 条、actor_23 有 5 条 Episode；它满足每个 actor 至少 3 条，但不满足 5-shot + 1 Query 的三参与者条件。

另有 28 个任务模板在至少两个 actor 中各有 6 条或以上 Episode，理论上可构造 5-shot + 至少 1 Query 的 actor-held-out 小样本测试，但这 28 个模板大多只有两个 actor，且 actor 覆盖高度不均。

## 4. Actor 覆盖不均衡

在跨参与者且每个 actor 至少 6 条 Episode 的任务模板中：
- actor_01 覆盖 22 个模板；
- actor_02 覆盖 15 个；
- actor_03 覆盖 14 个；
- actor_04 覆盖 11 个；
- actor_07 覆盖 5 个；
- actor_08 覆盖 6 个；
- actor_15、actor_19 各覆盖 2 个。

这意味着 5-shot actor-held-out 评估主要由少数参与者贡献。若采用 leave-one-actor-out，必须逐 actor 报告可评估任务数，不能把所有 Episode 或 task×actor 组合当成独立参与者样本。建议先以 1-shot / 2-shot 为主，5-shot 只作可行子集上的次要分析。

## 5. 对现有 25-Episode pilot 的影响

现有 pilot 的 LOCO 是按归一化类别留出，而非 actor-disjoint。当前 7 个 held-out 类别中有 6 个在训练类别中出现过相同 actor。因此它不能作为 participant-independent generalization 证据。

本轮结果仍保留为探索性结果：Reptile 的 held-out MAE 在 1-shot/2-shot 多数类别优于 pooled MLP；但它不能证明对新参与者或新物理对象实例的泛化。现有 pilot 的类别标签、actor 字段及 Episode 子集也不能直接视为完整数据集的任务总体抽样。

## 6. 推荐的研究任务重定义

**建议将主要实验单位从“物理对象类别”改为“触觉交互任务模板 × 参与者”，预测目标仍保持为触觉动态结构。**

建议论文核心问题表述为：

> 在公开触觉交互数据中，元学习能否使模型利用少量新参与者的 Support Episode，快速适应其执行已知交互任务时的触觉结构表征？

这是 participant-adaptive interaction-structure learning，不应写成 unseen physical-object-instance learning。

建议把评估拆成两个互补、不可混报的轴：

### A. 未见任务模板适应（LOTO）

- 分组键：`(scene, task_name)`。
- 外层留出完整任务模板，保证该模板所有 actor / Episode 都不进入训练。
- Support/Query 在 held-out task 内按完整 Episode 切分。
- 测试参与者可能在其他任务中出现，因此这项评估不代表 unseen-actor generalization。
- 适用于测试模型能否适应新的交互任务/操作结构。

### B. 未见参与者适应（LOAO）

- 外层按 actor 留出，held-out actor 的所有 Episode 均不得进入 meta-training。
- 仅评估 held-out actor 执行、且至少一个训练 actor 也执行过的共享任务模板。
- Support/Query 仅从 held-out actor 的该任务模板 Episode 中按完整 Episode 切分。
- 按 actor 分层汇总；对每个 actor 报告可评估任务模板数量。
- 这项评估测试新参与者对已知任务模板的少样本适应，不等于新物理对象实例适应。

两个协议都必须确保 scaler、特征筛选、任务采样和超参数选择不接触外层 held-out actor/task Query。Reptile 与 Support-matched MLP 必须共享相同 outer split、Support IDs、Query IDs、网络结构和适应梯度预算。

## 7. Go / No-Go 决策

- **NO-GO：** 以当前 metadata 直接宣称对未见物理对象实例的少样本适应。
- **GO（有条件）：** 继续触觉交互结构适应；完整 metadata 为 LOTO 提供大量任务模板，也为 LOAO 提供 58 个跨 actor 模板，其中 50 个满足至少两个 actor 各有 2 条 Episode，28 个满足各有 6 条 Episode。
- **NO-GO：** 立即全量下载 15,324 条 Episode 的压力 shard 或直接重训。
- **GO：** 先实现 metadata-only 的 split feasibility enumerator，输出每个 actor / task 的 Support/Query 可行性；随后仅下载小规模代表性 shard 做 signal QC，再启动 1-shot / 2-shot actor-held-out pilot。

## 8. 复现与来源

- Repository: `inbea3/touch`
- Branch: `thesis-few-shot-structure-20261125`
- Full metadata identity audit: [workflow 45655312000](https://github.com/inbea3/touch/actions/runs/45655312000), Artifact `15042611555`
- Task-family actor coverage audit: [workflow 45760834326](https://github.com/inbea3/touch/actions/runs/45760834326), Artifact `15088954998`
- Full task-intent classification: `docs/touchscale_full_task_intent_classification_20261125.md`
- Primary-object codebook: `docs/touchscale_primary_object_annotation_codebook_v1.md`
- Metadata identity audit: `docs/touchscale_identity_audit_20261126.md`

所有结论均来自 metadata；候选 eligibility 不代表原始压力信号质量通过。后续必须对选定 Episodes 进行 HDF5/tactile signal QC。
