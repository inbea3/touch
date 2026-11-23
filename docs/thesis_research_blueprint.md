# Thesis Research Blueprint（阶段 A 冻结版）

- 版本：A-1.0
- 冻结日期：2026-11-23
- 仓库：`inbea3/touch`
- 分支：`integration/old-research-assets`
- 研究数据边界：STAG/HumanGrasp pressure-only
- 状态：研究蓝图冻结；具体实验仍按矩阵状态推进

## 1. 论文暂定定位

**暂定题目：基于元学习与图神经网络的跨物体触觉传感器协同激活预测研究**

英文：*Cross-Object Tactile Sensor Co-Activation Prediction from Tactile Pressure Data: Statistical Modeling, Graph Structure, and Meta-Learning.*

题目中的“图神经网络”和“元学习”是待验证的方法方向，不是预设成功的结论。若 GNN 或元学习未显示可靠增量，最终题目、摘要和贡献必须据实收窄。

## 2. 核心研究问题

- **RQ1（统计目标）**：早期压力观测能否预测同一 recording 后续有效接触帧上的传感器对统计相关关系？相关标签在时间窗口、重复抓取和物体层级上有多稳定？
- **RQ2（跨物体预测）**：简单统计特征与 Node-MLP 能否在严格物体隔离条件下泛化？空间拓扑是否在 Node-MLP 之上提供独立预测增量？
- **RQ3（少样本适应）**：以物体为任务时，Reptile 或其他元学习初始化能否优于同预算普通 MLP 微调？
- **RQ4（结构与适应的关系）**：若图模型有增量，该增量是否能在合理初始化、随机拓扑对照和物体级不确定性分析下保持？元学习与图模型是否值得组合，必须由分阶段证据决定。

## 3. 研究范围与排除项

### 纳入
- STAG/HumanGrasp pressure-only 压力阵列及完成任务定义所需的元数据。
- 官方事件标记、`hasValidLabel`、时间戳等已审计字段；Gaussian smoothing、baseline subtraction 仅作为候选预处理，必须通过预注册敏感性实验决定是否纳入。
- 统计相关目标、Node-MLP、空间图模型、Reptile 及经论证的后续元学习对照。
- Weight 重复抓取数据仅作为重复抓取变异/目标可行性证据，除非另行定义并验证新 estimand，不并入主训练任务。

### 排除
- 同步视频、视觉标注、姿态估计和多模态融合。
- 将相关边解释为因果关系、物理接触真值或无误差 ground truth。
- 将传感器边、帧、episode、fold 或重复 grasp 当作独立物体样本。
- 因暂定论文题目而强行保留没有增量证据的 GNN/元学习模块。

## 4. 数据与任务定义

### 4.1 数据层级
已审计的 STAG/HumanGrasp pressure-only 数据记录：135,187 帧、32×32 压力阵列、82 条源 recording；排除 4 条 empty-hand 后，建模集合为 26 个非空物体、78 条 recording，每个物体 3 条 recording。有效物理传感器为 548 个。

### 4.2 主任务（within-recording future relation prediction）
对每条 recording，按时间排序有效帧。主输入为最早 5 个 `hasValidLabel=True` 有效接触帧构造的早期压力表征；主目标为其后有效接触帧中每一无向传感器对的 Pearson 相关系数。辅助标签为 (I(r_{ij}\geq0.7))。这是窗口条件下的统计关系预测，不是原始压力流端到端接触检测，也不是因果协同识别。

已有主基线和 Reptile 结果固定使用 96/548 传感器子集；不能将其结果写成完整 548 传感器性能。另有 G2 完整 548 节点图建模实验流，须单独标注其 estimand、输入、切分与结果，不能把不同实验流的数值直接合并。

### 4.3 Support/Query 与外层隔离
- 固定历史任务协议：18/4/4 个物体分别用于 Meta-train/validation/test；每物体 3 条 recording；Support 为 1 或 2 条 recording，Query 为其余 recording。
- Support/Query 必须按完整 recording 隔离，禁止在边级随机拆分。
- 历史 18/4/4 测试对象已被阶段七探索使用，全部 26 个物体也已进入阶段八重复交叉验证。因此该划分不能再称为 untouched confirmatory test。
- 阶段八 5-fold × 5-repeat 结果可作为当前最稳健的探索性物体级证据；重复折训练集重叠，不把 25 折视为独立数据集。
- 确认性泛化需要新增独立物体/数据；若无法获得，论文必须明确限定结论范围。

## 5. 已有证据基线（冻结时可引用）

### 5.1 数据和标签审查
- 96 节点、5/10/20 帧 circular moving-block Bootstrap（每组 200 次）：物体级边排序 Spearman 中位数 0.9424/0.9193/0.8964；(r\geq0.7) 正边 Jaccard 中位数 0.6047/0.5365/0.5000；相关系数 MAD 中位数 0.0390/0.0478/0.0539。
- 结果说明相关关系有一定排序稳定性，但阈值边集合和数值估计仍有不确定性。

### 5.2 96 节点基线
- 阶段七固定 18/4/4 划分中，MLP 零样本 1-shot MAE 0.1100，训练集常数 0.1314；该四物体测试已用于探索，不能作最终确认。
- 阶段八 5-fold × 5-repeat：MLP 零样本 MAE 0.11716（1-shot 95% CI 0.11188–0.12304），常数 0.13305（0.12674–0.13985）；配对物体级改善 0.01589（0.01306–0.01872），26/26 物体同方向。
- Support 微调没有稳定增益；1-shot 明显恶化，2-shot 改善不确定。
- (r\geq0.7) 正边约 0.567%，分类高度不平衡。

### 5.3 Reptile
- 200 meta-iterations × 3 seeds；每种方法每 seed 8,000 次梯度更新。
- 四个历史 validation 物体：Reptile Support-adapted MAE 为 1-shot 0.12587、2-shot 0.12454；Support-matched MLP 为 0.12332、0.12273；普通 pooled MLP 为 0.12443、0.12356。
- Reptile 没有显示 MAE 优势；适应步数诊断亦未显示稳定改善。n=4 物体，仅作探索性描述。
- Reptile 不应被包装为已证明有效的元学习方案。

### 5.4 Weight 重复抓取审计
- 26 个物体、每物体 6 个 grasp interval；Fisher 聚合的 split-half 边排序一致性和正边 Jaccard 均低于 pooled-frame Pearson。
- 当前不支持将 Weight 聚合目标并入主任务或将 grasp 数当成新增独立物体任务。

### 5.5 G2 完整 548 节点空间图实验流（与 96 节点任务分开管理）
- 已有项目记录显示 G2-R2 完成 15/15 folds；k6 是候选主图，k4/k8 为邻域规模消融。
- 阶段 A 将 G2-R3 冻结为初始化鲁棒性与拓扑随机控制：Node-MLP/k6 多随机初始化；degree-preserving random rewired k6 图；100–150 epochs，使用内部 validation；报告物体级配对效应和不确定性。
- 具体 G2-R2 指标、划分清单、特征定义、脚本参数、R3 精确执行参数必须从对应报告/运行产物复核后再运行；本蓝图不补造缺失数值。G2 与 Reptile 不在 R3 阶段合并训练。

## 6. 模型比较层级

1. 常数/统计基线。
2. Node-MLP（不使用图边消息传递）。
3. 空间图模型（k4/k6/k8 等预定义邻接）。
4. G2-R3 初始化与拓扑随机控制，验证空间拓扑增量不是初始化偶然或任意稀疏图效应。
5. 元学习流：普通 MLP fine-tuning vs Reptile；是否开展 MAML/ANIL 由公平对照动机和结果决定。
6. 只有前述两条证据链各自站稳后，才考虑图模型与元学习组合。组合模型不是阶段 A 的默认下一步。

## 7. 统计分析冻结原则

- 主要推断单位：物体。
- 主要回归指标：MAE；次要指标：RMSE、Spearman。
- 分类仅作辅助：AUROC、Average Precision/PR-AUC、Precision、Recall、F1、正类率；阈值只在 validation 选择并报告校准。
- 报告逐物体结果、配对差值、改善物体数、95% 物体级 Bootstrap 区间和效应量。
- 对物体重采样，不对边/帧/episode 重采样冒充独立样本。
- 明确训练集重叠、重复测量、标签估计误差和模型选择带来的限制。
- 所有预处理、特征、超参数和模型选择仅使用训练/验证数据；测试 Query 不参与选择。

## 8. 论文结构冻结（八章）

1. 绪论：研究背景、问题、意义、研究问题与贡献边界。
2. 数据基础与协同关系统计建模：数据审计、Pearson estimand、稳定性和层级结构。
3. 跨物体预测任务与基础模型：任务构造、特征、基线和验证协议。
4. 时序/图结构模型：Node-MLP、空间图、模型比较与消融（结果未完成前标注方案）。
5. 元学习方法：物体级任务、Reptile/候选方法及公平对照。
6. 实验设计与统计评价：数据隔离、指标、物体级不确定性、复现。
7. 实验结果与讨论：按真实实验结果组织，负结果同样报告。
8. 结论与局限：适用边界、独立确认需求和后续研究。

## 9. 贡献声明的证据门槛

可暂时主张：建立 pressure-only 的物体级未来相关关系预测任务；给出相关标签稳定性分析；在 26 个物体的重复分组验证中观察到早期压力特征相对常数基线的预测增量；对普通适应、Reptile 与重复抓取聚合的边界进行了实证审查。

只有在 G2-R3 通过随机初始化和 degree-preserving rewiring 对照后，才可主张所用空间拓扑具有超越 Node-MLP/随机拓扑的增量。只有元学习在公平、物体级对照中稳定优于普通 fine-tuning，才可主张元学习有增量。当前不主张因果协同、完整数据域普遍泛化或元学习已有效。

## 10. 阶段 A 结束条件

- [x] 研究问题、数据边界、主要 estimand 和证据限制形成统一文档。
- [x] 论文八章结构冻结。
- [x] 已有结果与未完成研究分开登记。
- [x] G2-R3 初始化/拓扑控制方向纳入实验矩阵。
- [ ] 对照 GitHub 中 G2-R2 的精确报告、配置和 15-fold 清单，补齐 G2-R3 的可执行参数。
- [ ] 三份冻结文件提交后进行一致性复核。
- [ ] 确认无现有文档/实验状态冲突后，进入下一阶段执行。

## 11. 权威来源
- `docs/project_status_and_next_steps.md`
- `docs/repeated_object_cv_results.md`
- `docs/meta_learning_protocol.md`
- `docs/weight_object_level_aggregation_results_20261122.md`
- `docs/weight_object_level_aggregation_protocol_20261122.md`
- G2-R2/R3 当前研究记录与对应运行产物（需在具体实验启动前重新核对）。
