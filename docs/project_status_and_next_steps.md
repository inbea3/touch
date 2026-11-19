# 项目进度总览与后续实施路线

> 更新日期：2026-11-19  
> 当前研究范围：STAG/HumanGrasp pressure-only  
> 当前代码分支：integration/old-research-assets  
> 本文严格区分已完成真实实验、已提交代码/协议和待执行计划。

## 一、论文定位与研究问题

暂定题目：**基于元学习与图神经网络的触觉传感器协同激活预测研究**。

核心问题：在少样本条件下，如何利用早期触觉观测预测未见物体的传感器协同关系，并检验元学习能否改善跨物体迁移与快速适应。

论文以统计学硕士研究要求为导向，重点包括协同关系的统计定义与稳定性、物体级独立验证、模型增量比较、物体异质性、不确定性和效应量。协同关系是统计操作定义，不解释为因果关系或无误差物理真值。

研究范围锁定为 STAG/HumanGrasp pressure-only，不引入同步视频、视觉标注或多模态数据。GNN 是待验证的结构建模候选，不是必须堆叠的组件。

## 二、已完成工作

### 阶段一至三：数据核验与早期探索
- 已读取用户提供的 classification_lite.zip 中的 metadata.mat。
- 数据包含 135,187 帧、32×32 压力阵列、82 条原始 Recording。
- 排除 4 条 empty-hand Recording 后，纳入 26 个非空物体、78 条 Recording，每个物体 3 条。
- 数据审查、旧仓库 inbea3/old 可复用探索迁移、早期分类和压力 CNN 内部探索已记录在仓库对应文档。
- 官方 test 未作为本研究的训练或调参数据；内部验证结果不得写成官方测试性能。

### 阶段四：特征消融与模型机制探索
- 已建立早期压力特征和模型比较探索。
- 材料见 docs/feature_ablation.md、docs/classification_exploration.md、docs/pressure_cnn_experiment.md。
- 该分类探索属于前期可复用工作，不替代协同关系预测主任务。

### 阶段五：协同关系定义、标签审查与稳定性
- 主目标为最早 5 个有效接触帧之后、后续有效接触帧上的连续 Pearson 相关系数。
- 辅助二分类标签为 r≥0.7，表示指定观测窗口上的统计正相关，不是因果关系或无误差物理真值。
- 已开展相关估计器、窗口和标签稳定性分析。
- 已完成 5/10/20 帧 circular moving-block Bootstrap，每组 200 次，固定 96 节点。物体级边 Spearman 中位数分别为 0.9424/0.9193/0.8964；r≥0.7 边 Jaccard 中位数分别为 0.6047/0.5365/0.5000；MAD 中位数分别为 0.0390/0.0478/0.0539。
- 标签审查支持连续相关强度作为主目标，阈值标签作为辅助任务；阈值、窗口和时间依赖的局限已记录。
- 不使用同步视频核验事件；没有官方事件锚点的 Recording 不做未经验证的自动补标。

### 阶段六：物体级划分与 Support/Query 任务构造
固定划分 seed=20261118：
- Meta-train：18 个物体、54 个 Recording；
- Meta-validation：4 个物体、12 个 Recording；
- Meta-test：4 个物体、12 个 Recording。
每个物体 3 条 Recording，穷举 1-shot 和 2-shot Support/Query 组合，每个物体每个 shot 有 3 个 Episode；总计 156 个 Episode。Support 和 Query 按 Recording 隔离，同一物体的 Recording 不跨外层集合，数据泄漏检查通过。
注意：156 个 Episode 不是 156 个独立物体任务；Meta-train 独立任务仍只有 18 个物体。

### 阶段七：首轮预测基线
- 固定使用同一组 96/548 有效传感器，每条 Recording 4,560 条无向边。
- 输入为最早 5 个 hasValidLabel=True 帧；目标为其后的有效接触帧 Pearson 相关。
- 特征包括传感器级早期统计量、时间斜率、首末值、传感器对称组合、空间几何和早期相关。
- 标准化和全局模型只在 Meta-train 拟合。
- 比较训练集常数、Support 常数、历史相关、MLP 零样本/Support 适应，以及 Logistic 辅助分类。
- 单次固定四物体 Meta-test 中，MLP 零样本 MAE=0.1100，常数基线 MAE=0.1314；MLP 1-shot/2-shot 适应没有一致收益。
- 详见 docs/coactivation_baseline_results.md。

### 阶段八：跨物体稳健性验证

阶段八完成了两种互补的重复物体级评估，需区分统计用途。

**A. 20 次随机 18/4/4 重复留出**
- 划分种子 20261118–20261137；固定相同 96 传感器子集。
- MLP 零样本 MAE=0.11555±0.00471；RMSE=0.15178±0.00579；Spearman=0.36096±0.04062。
- 常数预测 MAE=0.13118±0.00631。
- MLP 零样本 MAE 平均相对下降约 11.91%；20/20 次划分均优于常数基线。
- 1-shot Support 微调 MAE=0.12101，2-shot=0.11615，未显示稳定增益。
- 20 次重复重用同一批 26 个物体，划分间 SD 描述物体组成敏感性，不是 20 个独立数据集的标准误。
- 本地详细结果位于 stag_stage8_run/results/repeated_object_holdout/。

**B. 5-fold × 5-repeat 重复物体级分组交叉验证（阶段八主要统计验证）**
- 共 25 次外层拟合；每次 26 个物体在 5 折中各测试一次；每个物体累计 5 次 held-out 评估。
- 同一物体的 3 个 Recording 始终处于同一折。
- 跨重复先对同一物体的 held-out 结果求平均，再以 26 个物体为统计单位；以物体为单位进行 5,000 次 percentile bootstrap，给出 95% CI。
- 1-shot MLP 零样本 MAE=0.11716（95% CI 0.11188–0.12304）；常数 MAE=0.13305（0.12674–0.13985）。
- 2-shot MLP 零样本 MAE=0.11716（0.11192–0.12316）；常数 MAE=0.13305（0.12658–0.13972）。
- 常数减 MLP 零样本 MAE 改善：1-shot +0.01589（95% CI 0.01306–0.01872），2-shot +0.01589（0.01297–0.01875）；26/26 物体方向一致。
- MLP 零样本相对常数 MAE 改善约 11.73%。
- MLP Support 适应没有显示稳定增益：1-shot 适应明显变差；2-shot 差异较小，配对物体级 Bootstrap 区间跨 0。
- 辅助 r≥0.7 正边比例约 0.567%，类别严重不平衡；Logistic 有较高召回但 Precision 低，回归分数排序表现不能等同于校准分类概率。
- 正式报告：docs/repeated_object_cv_results.md；协议：docs/repeated_object_cv_protocol.md。

**阶段八综合判断：** 早期压力特征对未来连续协同关系有跨物体预测增量；普通 Support 微调未证明能稳定改善预测。下一步应研究更适合新物体快速适应的任务级初始化，而不是直接宣称 MAML 必要或有效。

## 三、尚未完成的内容

1. Reptile 初版已实现并完成首轮真实数据 pilot；ANIL/MAML 尚未实现。
2. 首轮 Reptile 配置未显示相较普通 MLP 微调的增量；元学习是否有收益仍未得到证明。
3. 尚未完成输入帧连续性和真实时间跨度敏感性分析。
4. 当前主预测实验只使用 96/548 传感器，尚未完成 192 或完整 548 规模对照。
5. 尚未完成 GNN 的结构增量实验。
6. 尚未完成针对最终模型的独立冻结测试流程及论文最终统计推断。
7. 只有 26 个物体任务，结论必须限定在当前对象范围和数据协议内。

## 四、下一阶段：元学习研究与实施

### 阶段九：Reptile 元学习实现（下一项工作）

目标：检验跨物体任务训练得到的初始化，是否比普通 MLP 初始化更适合 1-shot/2-shot 新物体适应。

先实现一阶 Reptile，不将 MAML 设为必须算法。任务定义：
- 一个物体是一个 meta-task；
- 一个物体的 3 条 Recording 构成任务内数据；
- Support 由 1 或 2 条 Recording 构成；
- Query 为该物体剩余 Recording；
- 任务采样以物体为单位；评估和统计汇总以物体为单位；
- Support 用于内循环适应，Query 用于独立任务评估。标准 Reptile 外循环通过任务内训练后的参数位移更新初始化，不应误称为 MAML 的 Query-loss 二阶优化。

核心更新：
\[
\phi_i^{(0)}=\theta,\qquad
\phi_i^{(k+1)}=\phi_i^{(k)}-\alpha\nabla_{\phi_i^{(k)}}\mathcal L_{\mathcal T_i}^{S}(\phi_i^{(k)})
\]
\[
\theta\leftarrow\theta+\epsilon(\phi_i^{(K)}-\theta)
\]

实现前必须处理：
1. 4,560 条边并非独立样本；任务采样以物体为单位，并对 Recording/边级损失合理归一化。
2. Support/Query 必须按 Recording 分开，不能随机拆边。
3. 每个物体只有 3 条 Recording，需明确任务采样与 1-shot/2-shot 组合。
4. Reptile 外循环以任务为单位聚合参数位移，并记录任务异质性。
5. 对照模型使用相同特征、基础网络、任务划分和适应预算。
6. 旧 18/4/4 测试集和阶段八所有物体已参与研究决策/交叉验证，不能再称为 untouched final test。必须设计新的冻结评估策略，或明确将结果定位为探索性验证；若要做最终确认，优先考虑新增独立物体/数据，现有数据不足时需在论文中如实说明。
7. 记录随机种子、内循环步数、内外学习率、任务采样、训练曲线及逐物体适应前后结果。

### 阶段十：公平对照实验
至少比较：
- 常数预测；
- 普通 MLP 零样本；
- 普通 MLP + Support 微调；
- Reptile + Support 适应；
- ANIL + Support 适应（在相同预算下可公平实现时）。

MAML 是可选扩展，不预设其必然胜出。核心增量：
\[
\Delta_{\mathrm{meta}}=\mathrm{MAE}_{\mathrm{ordinary\ fine\text{-}tuning}}-\mathrm{MAE}_{\mathrm{meta\ initialization}}
\]
正值表示元初始化误差更低。同步报告适应前后变化、MAE/RMSE/Spearman、物体级差值和置信区间。

### 阶段十一：元学习消融与统计分析
- 1-shot/2-shot；
- 内循环步数、内学习率 α、外循环步长 ε；
- meta-batch size 和任务采样策略；
- 元初始化训练随机种子；
- 训练任务数量敏感性；
- 适应前后逐物体配对误差；
- 以物体为单位的 Bootstrap 区间和效应量。

不得把传感器边或 Episode 当独立样本做显著性检验。

### 阶段十二：GNN 结构增量评估（暂缓）
只有在 Reptile/ANIL 与普通 MLP 对照完成后，才决定是否引入 GNN。若引入，需检验空间邻接、距离核或可学习图结构是否提供独立增量；不能仅因暂定题目包含 GNN 就强行加入。

## 五、论文组织建议
1. 绪论：触觉协同关系预测与跨物体少样本迁移问题；
2. 数据与统计关系定义：pressure-only 数据审查、Pearson 目标、标签稳定性和时间依赖；
3. 基线模型与跨物体泛化：特征构造、物体级验证、重复分组 CV；
4. 元学习方法：Reptile/ANIL 任务定义、算法、优化与复杂度；
5. 实验结果：与普通微调公平对照、少样本适应、消融和物体异质性；
6. 统计讨论：不确定性、效应量、类别不平衡、适用边界与局限。

题目中的 GNN 可在结构增量证据出现后保留；若没有证据支持，应考虑收窄题目，不应为题目强加实验。

## 六、下一次会话启动清单
1. 检查分支中是否已有 Reptile/ANIL 代码，避免重复实现。
2. 固定任务采样、内外循环定义，编写 docs/meta_learning_protocol.md。
3. 设计 Reptile 与普通 MLP fine-tuning 的严格配对实验协议。
4. 明确旧测试集已被探索性使用后的评估边界。
5. 实现 Reptile 最小可运行版本和单元测试。
6. 先做少量任务/少量迭代 smoke test，再开展正式多随机种子实验。
7. 元学习结果按物体汇总，保存任务、随机种子、模型参数和日志。
8. 若 Reptile 无稳定增量，先分析任务异质性和原因，再决定 ANIL/MAML，不自动增加模型复杂度。

## 七、结论表达边界

目前可以表述：
- 早期 pressure-only 统计特征对未来连续 Pearson 协同关系具有一定跨物体预测增量；
- 该增量在重复物体级验证中方向一致；
- 普通 Support 微调没有显示稳定增益；
- 因此有必要进一步研究任务级元初始化是否能改善新物体少样本适应。

目前不可表述：
- 已证明元学习有效；
- 已证明 MAML 优于普通模型；
- 已证明 GNN 能改善预测；
- 已验证完整 548 传感器阵列性能；
- 已验证真实物理协同机制或因果关系；
- 已获得对所有日常物体的普遍泛化保证。


## 阶段九最新进度（2026-11-19）

- 已审查 scripts/ 与 tests/：当前分支没有 Reptile、ANIL、MAML 实现；现有预测基线为 sklearn MLPRegressor，使用 64/32 隐藏层和 Support partial_fit。
- 已新增 docs/meta_learning_protocol.md，明确 Reptile 一阶参数位移、任务/Recording/边的层级、任务等权损失、同结构普通 MLP 微调对照、验证边界、评价指标和停止规则。
- 真实元学习代码、单元测试、smoke test 与正式实验尚未完成。当前连接可读写 GitHub 文件，但本仓库未提供可直接读取的 metadata.mat；因此不能在此声称已完成真实数据训练。
- 下一步：按该协议实现 PyTorch Reptile 与匹配的 pooled MLP fine-tuning 对照，并增加 synthetic unit tests；随后需要本地 metadata.mat 执行真实数据 smoke test。


## 阶段九最新进度（2026-11-19：Reptile 初版与首轮真实数据 pilot）

- 已实现 `scripts/run_reptile_meta_learning.py`，使用 PyTorch 64/32 MLP 和一阶 Reptile 参数位移更新。
- 已新增 `tests/test_reptile_meta_learning.py`，11 项单元测试全部通过，覆盖 Support/Query Recording 隔离、Recording 等权损失、任务等权参数位移、普通 MLP 梯度预算匹配、适应过程和物体级汇总。
- 已恢复此前上传的 `classification_lite.zip`，SHA-256 与数据审查记录一致；ZIP 完整性检查通过，解压得到真实 `metadata.mat`。
- 首轮真实 pilot：固定 96/548 传感器、18 个 meta-train 物体、4 个 meta-validation 物体；50 个 meta-iterations、meta-batch=4、inner steps=5、inner lr=0.0005、outer step=0.05；3 个随机种子。普通 MLP 与 Reptile 每个种子各执行 1,000 次梯度更新，更新预算匹配。
- 验证集 1-shot MAE：普通 MLP 微调 0.12448，Reptile 微调 0.14500；2-shot MAE：普通 MLP 微调 0.12392，Reptile 微调 0.14438。4/4 验证物体均未显示 Reptile 优势。
- 本轮未运行 Meta-test，也未进行超参数搜索。上述结果仅是首轮探索性 pilot，不代表 Reptile 方法在所有设置下无效。
- GitHub 的 `results/reptile_meta_learning/` 保存 pilot summary、配置、provenance 和物体级指标；完整 Episode/任务采样/训练曲线/scaler/checkpoint 已打包为独立运行产物。
- 下一步先分析初始化漂移和内循环适应行为，并在 Meta-validation 上做有限超参数敏感性分析；若无稳定增量，不自动升级为 MAML。
