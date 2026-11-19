# 阶段九：Reptile 元学习协议（预注册草案）

> 状态：协议与实现已提交；17 项单元测试、真实数据 smoke test 和 27 组敏感性网格均已通过；最优配置的长程验证正在执行。
> 数据范围：STAG/HumanGrasp pressure-only。  
> 代码分支：integration/old-research-assets。

## 1. 研究目的

检验跨物体任务训练得到的初始化，能否改善新物体少样本适应。核心比较：

1. Reptile 初始化 + Support 适应，是否优于同结构普通 MLP 初始化 + 相同 Support 适应预算；
2. 两种初始化在适应前后的误差变化是否不同；
3. 差异是否在 1-shot、2-shot 和不同物体上保持一致。

不预设元学习有效。MAML 不作为本阶段必做项；GNN 暂缓。

## 2. 任务与数据单位

- Meta-task：一个物体。
- Task samples：该物体的 3 条 Recording。
- Support：从 3 条 Recording 中选 1 条或 2 条。
- Query：该物体剩余 Recording。
- 预测样本：Recording 内有效传感器无向边。
- 推断与汇总单位：物体，而不是边或枚举 Episode。

固定主输入/目标沿用阶段七/八：最早 5 个 hasValidLabel=True 帧构造输入特征，目标为其后的有效接触帧 Pearson 相关。固定传感器子集为原实验相同的 96/548，传感器种子 20261116。连续相关强度为回归目标；r >= 0.7 仅作辅助分析。

## 3. 外层数据边界

沿用固定 18/4/4 物体清单：Meta-train 18 个物体、Meta-validation 4 个物体、Meta-test 4 个物体。

训练阶段只从 Meta-train 任务采样。Meta-validation 只用于选择超参数、内循环预算和早停；其 Query 不参与梯度更新。Meta-test 在所有选择冻结后才执行 Support 适应，Query 只作评估。

**重要限制：** 该 18/4/4 划分的测试对象已在阶段七报告中用于探索性比较；全体 26 个对象也已参与阶段八重复分组交叉验证。因此本阶段即使遵循上述梯度隔离，也不能把 Meta-test 结果称为完全未触碰的独立确认性测试。结果应标为后续方法开发的探索性评估。若论文需要确认性泛化结论，必须新增独立物体/数据，或明确承认现有数据无法提供独立确认集。

## 4. Reptile 算法定义

令共享回归网络参数为 θ。每轮从 Meta-train 的物体任务中均匀抽取一个 meta-batch，避免某一物体因边数/有效边数较多而获得更大任务抽样概率。

对每个任务 Ti，从其 Recording 中构造 Support/Query Episode。Reptile 内循环仅使用 Support：

φ_i(0) = θ

φ_i(k+1) = φ_i(k) - α ∇ L_Si(φ_i(k))

任务级参数位移为 Δ_i = φ_i(K) - θ。对 meta-batch 的任务位移先等权平均，再进行外循环：

θ ← θ + ε (1/B) Σ_i Δ_i

其中 α 为内循环学习率，K 为内循环更新步数，ε 为 Reptile 外循环步长，B 为 meta-batch 任务数。Query 不参与标准 Reptile 的参数位移更新；Query 用于 Meta-validation 选择与独立任务评估。不得将该算法描述为 MAML 的二阶 Query-loss 优化。

## 5. 损失与样本权重

每个任务先按 Recording 计算均方误差，再对该任务 Support Recording 等权平均；meta-batch 再对任务等权平均。不能把所有任务的边简单拼接后按边数平均，否则长 Recording 或有效边较多的物体会主导优化。

每条 Recording 内部的边损失仍是该 Recording 的预测误差摘要。由于边之间相关，边数只影响该 Recording 损失的数值精度，不作为独立样本量用于置信区间或显著性检验。

## 6. 公平对照

使用同一 PyTorch MLP 架构、相同输入特征、目标、标准化器、训练物体、Episode 生成规则和 Support 适应预算，比较：

1. 常数预测；
2. 普通 MLP pooled training，零样本；
3. 普通 MLP pooled training + Support fine-tuning；
4. Reptile 初始化，零样本；
5. Reptile 初始化 + Support fine-tuning。

普通 MLP 与 Reptile 必须共享同一模型定义和相同优化器/批次策略。Reptile 训练的计算预算、总梯度步数和任务访问次数要完整记录；如不能完全匹配，应报告差异，不得宣称严格等预算。

ANIL 作为后续扩展：只有在明确共享特征提取器/任务头结构，并确保比较预算公平后再实现。MAML 不列入第一轮。

## 7. 超参数与验证策略

首轮建议采用小网格：
- inner steps K：1、5、10；
- inner learning rate α：1e-4、5e-4、1e-3；
- outer step ε：0.01、0.05、0.1；
- meta-batch tasks B：4；
- Support shots：1、2；
- 固定随机种子至少 3 个，用于检查训练随机性。

上述数值是待验证的初始搜索范围，不是已经确定的最优参数。首轮按完整 3×3×3 网格执行 27 组配置：K ∈ {1,5,10}、α ∈ {1e-4,5e-4,1e-3}、ε ∈ {0.01,0.05,0.1}；每组 meta-batch=4、3 个固定随机种子、50 个 meta-iterations。每组普通 pooled MLP 的梯度更新次数与对应 Reptile 配置严格相同。

超参数选择只看 Meta-validation 的物体级 Query MAE；不得使用 Meta-test Query 选参。预先规定选择指标为 1-shot 与 2-shot 的 Reptile Support-adapted MAE 等权平均；先在每个物体内平均三个随机种子，再对四个验证物体等权平均。选择该指标最低的配置；不以 Meta-test 或逐边指标调参。随后可对冻结配置运行较长训练，但结果仍属探索性验证。

先运行 smoke test 检查数值稳定、参数确实更新、Support/Query 无交叉，再启动完整网格。

## 8. 评价指标与统计分析

主要指标：连续目标 MAE；辅助报告 RMSE、Spearman。二分类 r >= 0.7 仅报告 AUROC、Average Precision、Precision、Recall、F1 和正类比例，并明确类别极不平衡。

主要配对差值：

D_i = MAE(i, ordinary fine-tuning) - MAE(i, Reptile fine-tuning)

正值表示 Reptile 适应误差更低。按物体形成配对差值，再计算物体级 Bootstrap 区间和改善物体数。对 4 个 Meta-test 物体的区间只能视为高度有限的探索性描述；不得以数千条边或 24 个 Episode 扩大自由度。

必须分别报告零样本、Support 适应后的指标和适应增量，避免只展示 Reptile 的最终分数而掩盖适应前差异。

## 9. 必须记录的实验产物

- 完整配置与所有随机种子；
- 每轮任务采样记录、Support/Query Recording ID；
- 每个 epoch/iteration 的训练损失和 Meta-validation Query 指标；
- Reptile 初始化权重与普通 MLP 权重；
- 每个物体、shot、模型的适应前后 MAE/RMSE/Spearman；
- 物体级配对差值及 Bootstrap 区间；
- 训练时间、总梯度更新次数和失败/早停记录；
- 数据版本、输入特征版本、传感器索引与代码 commit。

不得输出原始压力阵列或不必要的逐边标签。

## 10. 解释边界与停止规则

- 若 Reptile 适应未优于普通 MLP 微调，不能据此宣称元学习有效；先检查任务间异质性、内外循环尺度、Support 数量和训练稳定性。
- 若结果只在一个 shot 或少数物体上改善，应报告条件性收益，不作普遍结论。
- 若 Reptile 没有稳定增量，不自动升级为 MAML；可分析 ANIL 是否有清晰动机，否则将论文重点收敛为跨物体结构化预测与少样本适应的负面/边界结果。
- 本阶段结果不能证明因果协同或完整 548 传感器阵列性能。

## 11. 实现状态与执行门槛

- 已提交 `scripts/run_reptile_meta_learning.py` 与 `tests/test_reptile_meta_learning.py`；17 项单元测试已在 GitHub Actions run `44493081143` 中执行通过。
- 官方 pressure-only 数据已在临时执行环境下载并完成 smoke test：元数据加载、特征提取、Reptile 与 pooled MLP 训练、验证评估和输出文件完整性检查均成功。Smoke 配置只有 1 个种子和 2 次 meta-iterations，不是正式性能实验。
- 代码记录同种子初始化 SHA-256、梯度更新次数、Reptile task Episode / Support Recording 访问次数及 pooled baseline Recording 抽样次数。两方法梯度步数、初始化和训练物体池匹配，但任务/Recording 访问序列并非严格相同；报告中不得声称严格数据暴露等价。
- 27 组预先规定的 Meta-validation 超参数网格已在 GitHub Actions run `44493070529` 完成；每组 3 个随机种子。选择配置为 K=10、α=0.001、ε=0.1。27 组配置的 Reptile 1-shot 与 2-shot MAE 均高于对应普通 pooled MLP。当前追加 200 meta-iterations 的长程验证。Meta-test 不参与调参。
- Smoke test 的短程指标不用于论文性能结论。此前未核验的 pilot MAE 数值和对象级结果已从仓库移除并作废。
- 原始压力数据、提取特征、scaler、模型权重、逐 Episode 和逐对象指标仅留在受控临时执行环境，不上传公开仓库或 Actions artifacts。对外报告需引用 STAG 原始论文。


## 12. 数据集引用

Subramanian Sundaram, Petr Kellnhofer, Yunzhu Li, Jun-Yan Zhu, Antonio Torralba, and Wojciech Matusik. “Learning the signatures of the human grasp using a scalable tactile glove.” *Nature*, 2019. DOI: 10.1038/s41586-019-1234-z.
