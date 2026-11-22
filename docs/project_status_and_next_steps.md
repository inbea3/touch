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

1. Reptile 代码、12 项单元测试、pressure-only 真实数据 smoke test，以及 200-iteration × 3 seeds 的选定配置验证均已完成；Reptile 未在当前四个 Meta-validation 物体上显示 MAE 优势。详细结果见 `results/reptile_meta_learning/selected_validation_200x3_20261119.md`。
2. 27 组、三随机种子的敏感性网格已完成；所有配置的 Reptile 1-shot/2-shot MAE 均高于普通 MLP。选定配置的长程验证没有改变这一总体判断。
3. 尚未完成输入帧连续性和真实时间跨度敏感性分析。
4. 当前主预测实验只使用 96/548 传感器，尚未完成 192 或完整 548 规模对照。
5. 尚未完成 GNN 的结构增量实验。
6. 尚未完成针对最终模型的独立冻结测试流程及论文最终统计推断。
7. 只有 26 个物体任务，结论必须限定在当前对象范围和数据协议内。

## 四、下一阶段：元学习研究与实施

### 阶段九：Reptile 元学习实现（当前阶段）

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


## 阶段九最新进度（2026-11-19：200 iteration × 3 seeds 验证已完成）

- 已新增 `scripts/run_reptile_meta_learning.py`：PyTorch 64/32 MLP、一阶 Reptile 参数位移、1-shot/2-shot Support/Query Episode、普通 pooled MLP 对照和物体级评估代码。
- `tests/test_reptile_meta_learning.py` 现包含 15 项测试。GitHub Actions run `44506987171` 成功，Reptile 测试组 15/15 通过；同一回归工作流中的其他阶段测试也全部通过。
- 真实数据 smoke test：GitHub Actions run `44491723400` 成功；官方 pressure-only `classification_lite.zip` 下载、解压、元数据审查、特征提取、Reptile/普通 MLP 训练、验证评估和产物检查均通过。配置为 1 个随机种子、2 个 meta-iterations、meta-batch=2、inner-steps=2、1 个适应步，仅验证管线，不用于模型优劣结论。
- 数据版本核验：ZIP SHA-256 为 `ce0f27ae87e8ce2ec405f756dfe278b0b8011ef12e9de93e468b7ef31ad76c58`；`metadata.mat` SHA-256 为 `19062ed3415df6e21ae055c5149315a5350652a27d59ab55fed74e395f2f1378`；元数据包含 135,187 帧，pressure-only 字段完整。
- 公平性审计：两种方法使用同一架构、同一随机种子初始化、同一 Meta-train 物体池和相同梯度更新次数；但 Reptile 按任务 Episode 使用 1/2 条 Support Recording 连续更新，普通 pooled MLP 每步从全部训练 Recording 抽样，因此任务/Recording 访问序列并非严格相同。代码现记录初始化 SHA-256、访问次数和这一差异，不宣称严格数据暴露等价。
- Smoke test 输出只作为执行诊断；4 个 Meta-validation 物体和单次极短训练不足以支持性能推断。此前未核验的 pilot 数值与对象级结果已删除，不得引用。
- 27 组超参数敏感性网格已在 GitHub Actions run `44493070529` 成功完成。预先规定的选择规则选出 K=10、inner_lr=0.001、outer_step=0.1；1-shot/2-shot 等权 Reptile MAE 为 0.128356，普通 MLP 为 0.124438。27/27 组中 Reptile 的两种 shot MAE 均高于普通 MLP。
- 聚合结果已保存至 `results/reptile_meta_learning/sensitivity_grid_summary.json` 和 `sensitivity_grid_summary.csv`。逐对象/逐 Episode 指标和模型权重仍未上传。
- 选定配置的长程验证已完成：GitHub Actions run `44502406385` 成功，源代码提交 `e3fad2171241cb91409268dc014376ce59d7922d`。200 iterations × 3 seeds；每种方法每个 seed 8,000 次梯度更新；初始化哈希一致；Support-matched MLP 回放相同任务顺序和 Support Recording ID；Meta-test 未使用。
- Artifact `reptile-selected-validation-44502406385-1` 已成功上传，SHA-256 `68602f212260b93ef7b4d1fcc2026aaee010455c3d8484de813a2ca765a72f36`，保留至 2026-12-19。包含配置、summary、provenance、Episode/物体指标、任务采样、三条训练 trace、scaler、九个 checkpoint 和数据审计输出；未包含原始压力帧。
- 四个 Meta-validation 物体上，Reptile Support-adapted MAE：1-shot 0.12587、2-shot 0.12454；Support-matched MLP：1-shot 0.12332、2-shot 0.12273；普通 pooled MLP：1-shot 0.12443、2-shot 0.12356。Reptile 未显示 MAE 优势；其平均 Spearman 略高，但 MAE/RMSE 较差。对象级 bootstrap 的独立单位 n=4，结果仅作探索性描述，不能作为确认性显著性结论。
- 训练公平性：三种方法每个 seed 均为 8,000 次梯度更新、同一模型架构和同一 seed 初始化哈希。Support-matched MLP 精确复用 Reptile 的任务与 Support Recording 调度；ordinary pooled MLP 的 Recording 访问策略不同，因此仅作为辅助基准。
- 详细数值、区间、配置、限制和复现文件清单见 `results/reptile_meta_learning/selected_validation_200x3_20261119.md`。
- 已完成 A/B/C 三组诊断：GitHub Actions run `44506987275` 成功，复用原 200×3 checkpoint 与 train-only scaler，未重新训练。Artifact `reptile-diagnostics-44506987275-1`（SHA-256 `3a0709587348ddd83cd1a8fa222c9bba6bc1a89fd022e57bf37009001e6849bb`）包含适应曲线、物体级结果、任务位移余弦及汇总文件，保留至 2026-12-19。
- 适应步数诊断：Reptile 1-shot MAE 随 full-batch steps 从 0 到 20 单调上升（0.12323→0.12795）；2-shot 在 0 steps 为 0.12323，所有测试步数均未优于零样本。Support-matched MLP 在 1–2 步略有改善，之后恶化；Ordinary MLP 在 0–1 步最好。结果支持适应过度/Support 特异拟合风险，但四个验证物体不足以作确认性推断。
- 适应算子诊断：10 步 training-style minibatch 相比 10 步 full-batch，Reptile 1-shot MAE 仅改善 0.00065，2-shot 几乎不变（差异 +0.00004）。因此原训练/验证算子不一致确实存在，但不是解释当前 MAE 劣势的充分原因。
- 任务位移方向诊断：18 个 Meta-train 物体的平均 pairwise cosine，1-shot 为 0.2544、2-shot 为 0.2085；负余弦比例分别为 8.9% 和 13.9%。任务更新整体偏正向且仅中等对齐，不支持“广泛严重梯度冲突”的说法；物体异质性仍可能存在，但尚未证明是主因。
- 修订后的完整诊断结论见 `results/reptile_meta_learning/diagnostic_analysis_200x3_20261119.md`。随后已完成预先规定的保守适应策略比较（见阶段十最新进度）；小学习率仅使 Reptile 适应接近零样本，没有带来实质增益。当前不再建议继续在同一验证集上搜索 Reptile 步数/学习率。若继续元学习，下一步应转向 Query-aware 外循环目标的 ANIL/MAML 风格小规模探索，并继续严格限制为探索性验证。
- 研究范围继续保持 pressure-only；沿用固定 18/4/4 划分。Meta-validation/Meta-test 均属于探索性评估，不能称为独立确认性测试。公开仓库和 Actions Artifact 不包含原始 STAG 压力帧；Artifact 中包含可复现诊断所需的模型 checkpoint、train-only scaler、配置、逐 Episode/物体诊断指标及数据审计摘要。


## 阶段十补充进度（2026-11-19：保守适应策略比较已完成）

- 已新增 scripts/compare_reptile_adaptation_policies.py、tests/test_reptile_adaptation_policy_comparison.py 和 .github/workflows/reptile-adaptation-policy-comparison.yml。
- GitHub Actions run 44509113507 成功；脚本编译和适应策略单元测试通过，准确下载并复用 run 44502406385 的三种方法 × 三个随机种子 checkpoint，以及同一 train-only scaler。
- 预先规定五种策略：zero-shot；1/2 步、lr=0.001；1/2 步、lr=0.0001。三种模型（Reptile、Support-matched MLP、Ordinary MLP）均在同一固定 Support/Query Episode 上评估。
- 本轮完成 1,620 条 Episode-query 记录、360 条 object-seed 汇总、120 条 object-level 汇总；4 个 Meta-validation 物体为独立分析单位。Meta-test 未加载、未评估、未用于策略选择。
- Reptile 的 0.001 学习率适应使 MAE 变差：1-shot 一步 +0.000530、两步 +0.001477；2-shot 一步 +0.000122、两步 +0.000784（相对 Reptile zero-shot）。将学习率降为 0.0001 后，MAE 差异仅约 -0.00001 至 -0.00007，属于近乎无变化，不能称为有效适应增益。
- Ordinary MLP 在全部策略下 MAE 最低或接近最低，zero-shot MAE 为 0.122340；低学习率适应后的 MAE 约 0.12230，变化极小。Support-matched MLP 在 0.001、两步适应时改善：1-shot MAE 0.122853，2-shot 0.122384；但 Reptile 在相同策略下分别为 0.124702 和 0.124009。
- Reptile Spearman 通常略高于 Ordinary MLP，但 MAE/RMSE 不占优。排序相关与绝对误差需分别报告。
- 完整结果见 results/reptile_meta_learning/adaptation_policy_comparison_20261119.md。Artifact reptile-adaptation-policy-comparison-44509113507-1，SHA-256 41910cb0e841553b77f6b507f79429a71611e2bc0bf7da09b5b1647fb3198a97，保留至 2026-12-19；包含五个 CSV、summary JSON 和数据审计摘要，不含原始 STAG 压力帧。
- 研究判断：保守适应未能揭示 Reptile 的实质优势；不再在同一四个验证物体上进行更广泛的 Reptile 超参数搜索。若继续元学习，建议研究 Query-aware 外循环目标（ANIL/MAML 风格），使元训练目标更贴近跨 Recording Query 泛化；先制定与普通 MLP 公平匹配的梯度预算和验证协议，再进行小规模探索。当前 18/4/4 数据划分不能提供 untouched confirmatory test，最终论文必须明确这一限制。


## 阶段十二最新进度（2026-11-19：First-order ANIL 三随机种子实验已完成）

- 已冻结协议 `docs/anil_meta_learning_protocol.md`，实现 `scripts/run_anil_meta_learning.py`，增加 `tests/test_anil_meta_learning.py`，并配置 `.github/workflows/anil-meta-learning.yml`。
- GitHub Actions run `44511962054` 成功；ANIL 编译、五项核心单元测试、reference checkpoint/scaler 核验、官方 pressure-only 数据下载与审计、正式训练、输出断言和 artifact 上传全部通过。
- 算法为 first-order ANIL：每个任务只在 Support Recording 上更新最后一层回归头 9 步；随后使用完整 Query Recording 的等 Recording 权重 MSE 计算 Query 梯度，将一阶梯度转移至共享初始化；每个 meta-batch 更新一次外循环参数。未计算二阶导数。
- 固定配置：200 meta-iterations × meta-batch 4 × 3 seeds；inner/outer lr=0.001；inner head steps=9；batch size=1024；weight decay=0.001；评估使用 5 步 head-only adaptation。每 seed 为 7,200 次 Support head 更新 + 800 次 Query gradient evaluation = 8,000 次 backward pass。
- 对照包括：ANIL；在 ANIL 相同对象/Support Recording episode 调度上训练的 Support-only MLP（7,200 次全网络更新）；复用原 selected-validation 200×3 Reptile checkpoint；复用普通 pooled MLP checkpoint。四种初始化在相同穷举 Validation Episode 上，以 zero-shot 和相同 5-step head-only 算子评估。
- 结果共 864 条 episode-query-method-policy 记录、192 条 object-seed 汇总、64 条 object-level 汇总、16 条方法/策略/shot 汇总、56 条配对物体差值；独立分析单位始终是 4 个 Meta-validation 物体。Meta-test 没有加载或评估。
- 1-shot MAE：ANIL zero-shot 0.123058、5-step head-only 0.123608；Support-only schedule 0.123304 / 0.123081；Reptile 0.123225 / 0.123674；ordinary pooled MLP 0.122340 / 0.122418。
- 2-shot MAE：ANIL zero-shot 0.123058、5-step head-only 0.123342；Support-only schedule 0.123304 / 0.123182；Reptile 0.123225 / 0.123441；ordinary pooled MLP 0.122340 / 0.122327。
- ANIL 相对 Support-only schedule 的 zero-shot MAE 优势仅约 0.000246，四物体 bootstrap 区间跨 0；ANIL 未超过 ordinary pooled MLP。5-step head-only 适应使 ANIL 1-shot MAE 平均增加 0.000550，2-shot 增加 0.000284，未显示稳定适应收益。Reptile 与 ANIL MAE 接近，Reptile Spearman 较高；ordinary pooled MLP 的 MAE/RMSE 整体最佳。
- ANIL 的训练 Query loss 在三种子中均下降至约 0.025 左右，说明 Query-aware 优化过程执行并收敛；但训练 Query loss 下降没有转化为优于 pooled MLP 的验证泛化表现。
- 完整报告：`results/reptile_meta_learning/anil_first_order_validation_20261119.md`。Artifact `anil-meta-learning-44511962054-1`，SHA-256 `635bec289f39fb2986795e31b07a104407e2ab23dd18b5129d3f55c014151b78`，保留至 2026-12-19；含模型权重、训练 trace、任务调度、Episode/物体级指标、配对差值、summary 和数据审计，不含原始 STAG 压力帧。
- 研究决策：ANIL 是有意义的 Query-aware 元学习对照，但当前结果不支持其优于普通 MLP。不要立即开展二阶 MAML，也不要在相同四个验证物体上继续广泛调参。若要形成确认性元学习结论，下一步优先获取新增独立物体/Recording 或外部 pressure-only 数据；若无法获取，应将论文定位为探索性跨物体少样本适应研究，诚实报告 Reptile/ANIL 未带来可靠的预测增益。
- GNN 结构增量实验继续暂缓，除非新的独立数据或后续证据显示图结构建模具有清晰的额外研究价值。

## 阶段十三：公开 pressure-only 数据扩展调研（2026-11-19）

- 已完成系统性数据资源调研，报告见 `docs/data_expansion_research_20261119.md`。
- 现有 STAG classification 数据为 26 个物体、78 个物体 Recording、548 个传感器。官方还提供 blindfolded（pressure-only 1.58 MiB）、weight-prediction（2.85 MiB）和 hand-pose（5.63 MiB）数据包。前两者可能增加同一物体的跨条件/跨 session Recording，但不能直接增加独立物体任务数；hand-pose 的标签是 G1–G7 抓取类型，不是物体身份，只能用于姿态混杂因素分析。
- 首选外部人手压力数据为 EgoTactile-OXT：63 个 object-level tasks、767 条 trajectories、314,100 帧，其中 692 条 gloved-hand 直接测压轨迹；162 个布局 cell 对应 137 个唯一物理 sensor indices；约 3.75 GB，CC BY-NC 4.0。先只使用 gloved-hand 轨迹，bare-hand weakly-paired 子集仅作敏感性分析。
- 第二首选为官方 TUM-ICS Hide-and-Seek：33 个物理物体、61 个 object-weight 类别、约 4.3k trajectories、1,120,116 帧、44 维单手 skin-force 通道、约 624 MB、CC BY 4.0。按物理物体分组，同一物体的所有重量变体必须留在同一外层 split；只提取 skin force，不使用 proximity、distance、pose、wrist F/T 或视频。
- TouchScale 是高潜力大规模数据：当前公开 100 小时、15,324 episodes、929 tasks、24 participants、880 taxels/hand、905+ object records / 193 object meshes；约 1.08 TB、430 shards、CC BY-NC 4.0 且需要接受访问条件。先下载 metadata 和 geometry，再选择 10–20 个对象的少量 shards，绝不下载全量；items 与 mesh object_name 并非总能一一对应，必须先审计 object identity。
- OpenTouch（约 5 小时、3 小时 contact-rich、21.1 GB、CC BY 4.0）和 TouchAnything/EgoTouch（约 88.5 GB）保留为候选；需先核实 object-instance ID、数据许可、压力通道 schema。TouchAnything 文档有 16×16 与 21×21 压力网格描述差异。
- 其他候选 ActionSense、ADL tactile signatures、EgoPressure 主要是活动/参与者/平板手势数据，不适合作为当前 object-level grasp 协同预测的首批主数据。
- 推荐执行顺序：A) 审计 STAG blindfolded / weight / hand-pose 三个 pressure-only 小包；B) 下载 EgoTactile-OXT 与 HAS（合计约 4.4 GB），先做小样本 pressure-only Smoke Test；C) 再按 metadata 选择性下载 TouchScale shards；D) 最后评估 OpenTouch / TouchAnything。
- 跨数据集不得按通道索引拼接不同硬件。各数据集单独建图、单独 train-only scaler、独立计算未来 Pearson 相关标签；统一任务单位为物理物体，Recording/trajectory 为 Support/Query 分割单位，sensor edges / frames 不是独立统计单位。
- EgoTactile 建议按物体类别分层做 5-fold object-level outer CV；HAS 按物理 object ID 做 5-fold，不能按 61 个 object-weight labels 随机拆分。当前 STAG 18/4/4 Meta-test 已在此前阶段使用，不能再称 untouched confirmatory test。
- 数据扩展后开展 object-count learning curve（例如 10/20/30/全部可用训练物体），保持测试物体、模型、随机种子、训练预算和适应策略不变，比较 Ordinary MLP、Reptile、ANIL 和 Support-matched controls。目标是检验增加独立物体任务是否使元学习的边际收益增长，而不是把更多帧/窗口误当成更多任务。
- 当前尚未下载/审计新增数据包，尚未运行外部数据实验；本阶段是公开资源调研和预注册式实验规划，不得把候选数据规模当作已验证可用样本数。下一项实际执行任务为 STAG 三个 pressure-only 小包的下载与 metadata inventory。


## 阶段十四：STAG 三个额外 pressure-only 数据包审计（2026-11-19）

- 用户上传的 blindfolded_lite、weights_lite、handposes_lite 均已成功读取 metadata.mat 和 README/note。重新读取原有 classification_lite.zip 作为逐项对照。仅读取压力矩阵和元数据；没有读取图像/视频，也没有将原始 ZIP 上传仓库。
- 四套数据均为 int16、N×32×32 压力矩阵；逐数据集检查得到 548 个随时间变化的物理传感器位置，另有 476 个恒定位置；四套数据的 548 位置 mask 完全一致。每条 recording 内时间戳均非递减。
- Classification：135,187 帧、82 recordings、26 个物理物体（每物体 3 条）和 4 条 empty_hand recordings；hasValidLabel 共 88,269 帧。
- Blind-folded：6,686 帧、105 recordings，即 26 个物体各 4 条（104 条）+ 1 条 empty_hand；525 个 hasValidLabel 帧，恰好每条 recording 5 帧；全部 6,686 帧 splitId=1，属于官方 test split。对象名称与 classification 的 26 个非空对象完全一致。决策：不得加入元训练；保留为 held-out condition-shift 描述性资源，但 5 帧不足以稳定估计当前未来 sensor-pair correlation 标签。
- Weight：11,682 帧、27 recordings，即每个 26 个物理对象各 1 条 + 1 条 empty_hand；每条记录含 graspIndex 0–5 六个主要抓取区间，graspIndex=6 是尾部不完整片段（共 128 帧且 graspValid=0）。graspValid 共 2,301 帧，其中物理对象 2,214、empty_hand 87；六个主要 graspIndex 的有效帧总数分别为 375、389、389、376、388、384，对应每个对象每次抓取中位数约 14–15 帧。决策：这是最有希望的同硬件重复抓取数据，可按完整 grasp interval 做 Support/Query，但先检验 14–15 帧相关标签稳定性；必要时按预先冻结的多抓取聚合策略评估，不得将帧随机拆分。
- Hand poses：24,037 帧、7 条连续 recording、G1–G7 抓取姿态标签，isGrasp=1 共 7,697 帧；官方 split 17,072 train / 6,965 test。README note 明确每条 recording 含多种不同抓取，不存在 recording 与姿态类别一一对应。决策：不是物体任务数据，不增加元学习任务数；仅可在确有需要时用于压力域抓取姿态混杂因素分析。
- 四套数据的非空 object names 中，Blind-folded 和 Weight 与 Classification 的 26 个物体名称精确一致。三份新增数据包并未增加任何独立物理物体，因此独立 object task 数仍为 26。
- 观测压力范围：classification 446–1000；blindfolded 509–991；hand poses 510–992；weight 510–654。尽管传感器 mask 相同，数值分布仍有差异，需保留 train-only scaler 并审查 baseline/calibration，不可直接混合原始压力帧。
- 完整审计报告：`docs/stag_extension_audit_20261119.md`；机器可读 inventory：`results/data_expansion/stag_extension_inventory_20261119.json`。四个 ZIP 的 SHA-256 和结构统计已写入报告/inventory。
- 当前未运行新模型实验、未将 blind-folded/weight/hand-pose 数据纳入训练，也未声称新增独立任务。下一步：优先对 Weight 的 6 个抓取区间开展 pressure-only future sensor-pair correlation 标签稳定性审计；同时评估 blind-folded 仅作为描述性 condition-shift 数据是否有可用统计目标。审计通过前不重跑 Reptile/ANIL。


### 阶段十四补充：Weight 重复抓取标签稳定性审查（2026-11-19）

- 已复用 Reptile selected-validation run 44502406385 的固定 96 传感器索引（seed 20261116），并验证其与 Weight 数据 548 个有效传感器位置匹配；使用原管线变换 clip((pressure-500)/150, 0, 1)。
- 对 26 个物理物体，每个物体的 graspIndex 0–5 六个完整抓取区间，按时间排序取前 5 个 graspValid 帧为早期输入，其余 6–10 帧估计未来 Pearson 相关；每次抓取有效帧 11–15 个。graspIndex=6 不含有效抓取帧，排除。
- 每物体穷举 10 个唯一 3-grasp vs 3-grasp 划分，共 260 个嵌套比较；先在物体内平均，再以 26 个物体为单位做 5,000 次 Bootstrap。
- 三抓取对三抓取的 edge-correlation Spearman 均值 0.0504（物体级均值中位数 0.0425，95% object bootstrap CI [0.0366, 0.0660]）；r≥0.7 正边 Jaccard 均值 0.0406（中位数 0.0361，CI [0.0283, 0.0541]）。r≥0.5 Jaccard 均值 0.0690；r≥0.9 Jaccard 均值 0.0401（44 个分割因双方均无正边而未定义）。
- 单次抓取间比较更低：390 个嵌套抓取对的 Spearman 均值约 0.0026，r≥0.7 Jaccard 均值约 0.0112。使用完整 graspValid 窗口（包含最早五帧，作为非因果敏感性诊断）时，三对三 Spearman 均值仍约 0.0559、Jaccard 约 0.0411。
- 解释：重复抓取确实增加了同一物体内的 episode 数，但当前未来传感器相关矩阵在抓取组之间表现出很低的一致性。原因可能包括短窗口估计噪声、抓取间接触状态变化或真实关系变化；本审计不能区分这些原因。
- 决策：暂不将 Weight 的六个 grasp interval 当成可交换 Recording 直接加入 Reptile/ANIL 训练，也不声称其解决独立任务不足。下一步仅做压力通道方差/有效覆盖诊断和收缩协方差估计器敏感性；如果没有可解释且预先规定的聚合目标，则将 Weight 保留为重复抓取变异性分析/负向可行性结果。
- 报告：`docs/weight_grasp_label_stability_20261119.md`；机器可读结果：`results/data_expansion/weight_grasp_label_stability_20261119.json`。未训练模型、未使用视频或其他模态。


---

# 最新整合补充（2026-11-19）：GNN 进展与外部触觉研究启发

> 本节是当前实验分支 `experiment/g0-spatial-topology-audit-20261119` 的最新整合状态。原有阶段九至十四的 Reptile、ANIL 和数据审计记录继续保留；G0–G2 的空间图实验记录在本实验分支，尚需在后续代码审查后再决定是否合并回 `integration/old-research-assets`。本节不覆盖原有历史记录。

## A. 当前论文统一定位

暂定研究主线进一步表述为：

**基于空间结构化压力表征的跨物体泛化与少样本快速适应研究。**

英文工作标题：*Spatially Structured Pressure Representation for Cross-Object Generalization and Few-Shot Adaptation*。

该表述是研究工作标题，不是最终论文题目。研究仍以 STAG/HumanGrasp pressure-only 为核心，不引入视频、图像、视觉标注或机器人仿真数据。

统一研究问题：

- **RQ1 空间结构：** 显式建模压力传感器的物理空间组织，能否在匹配的非图节点模型之上，为未来传感器对相关预测提供增量信息？
- **RQ2 跨物体适应：** 相比普通训练与匹配的 Support 微调，任务级元初始化能否改善未见物体的少样本适应？
- **综合问题：** 具有空间结构的压力表征能否改善跨物体泛化与少样本适应？这是待检验问题，不是既定结论。

## B. 当前证据状态：必须区分已完成和待验证

### B1. 已完成的元学习研究（来自 integration 分支既有记录）

- Reptile 实现、单元测试、pressure-only smoke test、27 组敏感性网格、200 iterations × 3 seeds 长程验证及适应诊断均已记录。
- Reptile 在四个 Meta-validation 物体上没有显示 MAE 优势；增加适应步数通常使 MAE 变差，降低学习率仅使结果接近 zero-shot，不能称为有效适应增益。
- First-order ANIL 的三随机种子实验已完成。Query-aware 外循环确实执行并降低训练 Query loss，但验证 MAE 没有超过普通 pooled MLP；head-only 适应也未显示稳定收益。
- 以上结果均受独立任务数限制：Meta-validation 只有4个物体，且现有18/4/4划分与全体26物体重复CV已经参与研究决策。不能将当前 Meta-test 描述为 untouched confirmatory test。
- 研究要求保留元学习作为研究问题，但必须允许得出“当前数据未显示可靠增益”的科学结论；不得为了满足方法名目而宣称 Reptile/ANIL 有效。

### B2. GNN 空间图研究（当前实验分支已完成的 G0–G2）

**G0：拓扑审计**
- 压力矩阵为 135,187 × 32 × 32；以全档案时间标准差大于0定义的活动位置得到548个有效传感器。
- 全548节点的 k=4/6/8 图均连通；k=6 有2,226条无向边。
- 原随机96节点子集的空间邻域明显更粗；几何最远点抽样改善覆盖，但仍不能恢复全548局部邻域尺度。
- 坐标是32×32网格像素，不是经过物理标定的毫米距离；548节点 mask 使用全档案无监督方差规则，严格归纳式评估时应披露这一点。

**G1：96节点探索性对照**
- Edge-MLP、Node-MLP、Spatial-GNN 在原随机96节点上完成一次5折物体分组实验。
- Node-MLP 指标略优；96节点稀疏空间图没有显示 GNN 增量。
- 该结果不能用于否定完整548节点的物理空间图。

**G2：完整548节点 k=6 主对照**
- 26个物体、78条非空 Recording、每条 Recording 149,878个无序传感器对；训练每条 Recording/epoch 抽样4,096对，测试分块评估全部无序对。
- Node-MLP：MAE 0.122504、RMSE 0.159556、Spearman 0.324799。
- Spatial-GNN k=6：MAE 0.122150、RMSE 0.159044、Spearman 0.324883。
- 度数保持 Rewired-GNN：MAE 0.123158、RMSE 0.160570、Spearman 0.317289。
- 三个 split seeds、共15个外层折；真实空间 GNN 相对 Node-MLP 的 MAE 差值为 -0.000353（物体 bootstrap 描述性95%区间 -0.000525 至 -0.000168），22/26物体更好。
- GNN MAE 增益约0.29%，Spearman 几乎不变。训练固定5 epochs，末期训练损失仍下降；k=4/k=8敏感性、多次独立重连图和收敛性尚未完成。故当前结论是“有小幅初步增量信号，仍需稳健性验证”，不是“已证明 GNN 优越”。

G0–G2详细报告与结果：
- `docs/g0_spatial_topology_audit.md`
- `docs/g1_spatial_gnn_96_pilot.md`
- `docs/g2_full548_spatial_gnn_experiment.md`
- `results/g2_full548_summary.json`

### B3. 外部视频研究启发（仅作为概念背景）

已阅读用户提供的 Bilibili 视频 BV1SLME6WE38 自动字幕，形成 `docs/external_research_note_01_tactile_perception.md`。视频主要讨论触觉重要性、传感器空间分辨率、触觉反馈、多个传感器集成、机器人操作，以及视觉-触觉融合和少量示教迁移。

吸收的概念启发：
1. 触觉传感器具有空间组织结构，支持提出 RQ1，但不证明 GNN 必然有效。
2. 多传感器协同和物体差异，支持开展物体级异质性与跨物体验证。
3. 少量示教后的迁移，支持提出快速适应研究动机，但视频没有提供 Reptile/MAML/ANIL 的直接算法证据。
4. 视觉-触觉融合、机器人控制、仿真器和视频模态全部排除在当前实验范围之外。

视频字幕由 AI 自动生成，专业名词可能有识别错误。正式论文应优先引用经核实的原始同行评审论文，不应仅凭字幕引用具体技术成果或数量。

## C. 融合后的执行顺序

### 优先级 1：完成 G2-R 稳健性验证
1. 完整548节点 k=4、k=8 敏感性，沿用相同三组物体划分。
2. 增加训练预算，并使用训练物体内部验证划分选择训练轮数，检查是否收敛。
3. 每个划分生成多个独立度数保持重连图，估计随机拓扑本身的变异。
4. 继续使用 Node-MLP 作为匹配对照；统计单位保持物体。
5. 根据结果判断真实空间图增量是否稳定、是否足以支持后续融合。

### 优先级 2：处理元学习证据与独立任务数量
1. 暂停在同一四个 Meta-validation 物体上继续扩大 Reptile/ANIL 超参数搜索。
2. 先审计是否存在可增加独立物体任务的 pressure-only 数据。已审计的 blindfolded、weight、hand-pose 小包均没有增加新的物理物体身份；Weight 可提供同物体重复抓取条件，但不能算作新增独立物体任务。
3. 外部数据仅在 object identity、压力通道 schema、许可和任务兼容性核实后纳入；不同硬件通道不可按索引直接拼接。
4. 若无法获得新的独立物体任务，元学习结论应明确为当前样本规模下未观察到可靠增益，整体研究定位为探索性跨物体少样本适应研究。

### 优先级 3：有条件地开展 GNN × 元学习
只有当优先级1显示图结构贡献在 k、训练预算和重连实现下具有稳定性，才开展小规模组合比较：
- Node-MLP + ordinary adaptation；
- Spatial-GNN + ordinary adaptation；
- Node-MLP + meta-learning；
- Spatial-GNN + meta-learning。

必须匹配对象划分、Support/Query Recording、初始化、梯度更新预算和适应算子；避免在仅26个独立物体上进行过大的因子网格。

## D. 当前暂停点与下次启动清单

**当前暂停点：** G0、G1、G2(k=6) 已完成；G2-R 尚未执行。Reptile 与 first-order ANIL 的现有探索性结果不支持其优于普通 MLP。视频研究启发已整理为独立备忘录，不改变数据与实验边界。

**下一次启动顺序：**
1. 检查实验分支的 G0–G2 脚本、测试、原始产物清单和提交记录，确认可复现性及与 integration 分支的差异。
2. 执行 G2-R1：k=4/k=8。
3. 执行 G2-R2：训练收敛性与训练内验证。
4. 执行 G2-R3：多次度数保持随机重连。
5. 完成 G2-R 后再决定是否进行 GNN × 元学习组合实验。
6. 同步评估新增独立 pressure-only 物体任务的可行性；不得把同一物体的额外 Recording 误计为新任务。

**不变的研究边界：** pressure-only；不引入视频/视觉；对象是统计推断单位；边、帧和 Episode 不是独立物体；所有结论必须来自真实运行并报告局限；不预设 GNN 或元学习必然优于普通模型。


---

# 最新进度（2026-11-22）：G2-R1 完整548节点 k=4/k=8 敏感性实验

> 本节更新当前实验分支 `experiment/g0-spatial-topology-audit-20261119`。本轮沿用 G2 数据版本、传感器映射、模型架构和外层物体划分；不改变 pressure-only 研究边界。

## 1. 实验完成情况

- 已从先前的 G0–G2 实验归档中恢复 G2 可执行脚本、完整548节点映射和 G2 结果包。
- 已重新解压官方 `classification_lite.zip` 并核验 `metadata.mat` SHA-256：`19062ed3415df6e21ae055c5149315a5350652a27d59ab55fed74e395f2f1378`，与 G2 完全一致。
- 完成预运行：一个外层fold中 Node-MLP、Spatial-GNN k4、Spatial-GNN k8 均完成训练与全量测试边预测。
- 正式实验完成：3个 split seeds（20261119、20261120、20261121）× 5折物体分组交叉验证，共15个外层fold；每个seed下26个物体各作为测试物体一次。
- 三个模型均使用相同 NodeRelNet 可训练结构、相同初始化、相同训练Recording顺序和相同抽样pair索引；仅邻接矩阵不同。
- 固定沿用 G2 的5 epochs、每条训练Recording/epoch抽样4,096个有效传感器对、AdamW lr=0.0005、weight decay=0.001；测试全部149,878个无序传感器对。
- 每折 StandardScaler 仅在该折训练Recording上拟合；先对Recording在物体内求平均，再对26个物体等权汇总。
- 全部78条Recording的未来Pearson目标均有限。未使用视频、图像、视觉标签或物体类别作为模型输入。

## 2. 总体结果

| 模型 | MAE ↓ | RMSE ↓ | Spearman ↑ |
|---|---:|---:|---:|
| Node-MLP | 0.122496 | 0.159533 | 0.324919 |
| Spatial-GNN k4 | 0.122388 | 0.159310 | 0.324097 |
| Spatial-GNN k8 | 0.122227 | 0.159157 | 0.323847 |

物体级配对MAE差值定义为 GNN − Node-MLP，负值有利于GNN。先对每个物体跨3个split seed的差值取平均，再以26个物体为统计单位做50,000次bootstrap：

- **k4：** 平均差值 -0.000108；描述性95% object-bootstrap区间 [-0.000259, +0.000053]；16/26个物体MAE更低；相对MAE改善约0.088%。区间跨0，且 seed 20261120 的平均差值为正，尚不支持稳定增量。
- **k8：** 平均差值 -0.000269；描述性95% object-bootstrap区间 [-0.000447, -0.000071]；21/26个物体MAE更低；相对MAE改善约0.220%。三个split seed的平均差值均为负，当前显示较一致的小幅MAE增量。
- k4、k8的平均Spearman均略低于Node-MLP；因此不能称为所有指标全面改善。

## 3. 训练收敛边界

15个fold的平均训练MSE（Epoch 1→5）：
- Node-MLP：0.032812 → 0.026612；
- Spatial-GNN k4：0.032692 → 0.026507；
- Spatial-GNN k8：0.032664 → 0.026471。

第4到第5轮训练MSE仍分别下降约2.38%、2.27%、2.26%。本轮为了严格检验邻接尺度，沿用G2的5-epoch预算；因此尚未证明收敛，不能将当前k4/k8结果当作最终模型选择。G2此前的k6结果仍作为历史参考，未在本轮同一执行批次中重新训练；跨阶段k值排序只作参考。

## 4. 当前研究判断

1. k4没有显示足够稳定的增量证据。
2. k8相对同轮Node-MLP呈现较一致的小幅MAE改善，但绝对增益有限，Spearman没有同步提高。
3. 目前不能据此认定k8是最终最优邻接尺度，也不能宣称GNN全面优于Node-MLP。
4. 三次重复使用同一26个物体，不能把78个“物体-重复”当作78个独立物体；bootstrap区间仅作物体级描述性不确定性摘要。
5. 548节点mask仍沿用G0根据全压力档案非零时间变化得到的无监督mask；不使用标签，但使用了全档案压力变化，论文中必须披露。
6. 当前结果没有引入视频、多模态或因果解释。

## 5. 结果归档

- 完整报告：`docs/g2r1_k4_k8_sensitivity.md`
- 机器可读摘要：`results/g2r1_k4_k8_summary.json`
- 本地完整复现包：`G2R1_k4_k8_results_package.zip`，包含原始G2脚本、548节点映射、完整fold/object metrics、训练曲线、报告和MAE图；不包含原始压力阵列。
- 结果包 SHA-256：`0b949a66602a8c17696658e35b730c586a03efdc4cfa66701664b18db0482ec2`

## 6. 下一阶段：G2-R2 训练预算与收敛性

1. 预先冻结训练物体内部验证方案；外层测试物体不得参与epoch选择。
2. 在相同外层split seeds与对象折上增加训练预算，记录训练与内部验证曲线。
3. 比较Node-MLP、Spatial-GNN k4/k6/k8，确保模型初始化、训练样本抽样、优化器预算和验证流程匹配。
4. 根据训练物体内部验证选择epoch，再对外层留出物体评估；报告逐物体配对MAE、RMSE、Spearman和描述性区间。
5. 在收敛性检查完成前，不开展大规模GNN×元学习组合搜索。

**当前暂停点：** G2-R1 已完成；k8显示小幅且较一致的MAE增量，k4证据不稳定；5 epochs不足以确认收敛。下一步执行G2-R2，而不是立即将k8定为最终图结构。
