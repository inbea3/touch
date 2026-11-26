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


## 阶段十五：Weight 压力方差、传感器覆盖与协方差估计器敏感性（2026-11-22）

- 已重新上传并实际读取 `weights_lite`；ZIP 完整性通过。输入 SHA-256：`37b1c5fcb246383cdaed552ce07e35690883545cdfdbfc118b48c1cbd2ee3cf3`。压力矩阵 11,682 × 32 × 32、int16、范围 510–654，无 NaN/Inf；27 recordings（含 empty_hand）、26 个物理物体、2,301 个 graspValid 帧。
- 仅使用 pressure-only 数据及 object/grasp/timestamp 元数据；未访问视频、图像或其他模态；未训练模型。
- 复现固定 96/548 传感器子集，seed=20261116；精确坐标、索引及哈希保留在运行产物。
- 26 × 6 = 156 个完整 object/grasp 区间，每段 11–15 个有效帧；剔除前 5 个早期输入帧后，未来目标仅 6–10 帧。graspIndex=6 不完整尾段排除。
- 传感器方差审计：未来窗口中非零方差传感器比例中位数 1.00，方差 >1e-6 比例中位数 1.00；有效传感器参与率（participation ratio）中位数 74.49/96，物体级 bootstrap 95% CI [67.84, 74.55]。但各 grasp 的有效传感器数范围 12.23–82.46，仍有显著异质性。
- 预先规定的 transformed pressure >0.01/0.02/0.05 三个阈值在所选通道未来窗口中均得到 100% 比例，因为该数据变换后的数值全部高于 0.05；这些阈值在 Weight 数据中不具备接触覆盖区分能力，不能作为 contact label。
- 对每物体 10 个唯一 3-grasp vs 3-grasp 划分，分别比较 sample Pearson、Ledoit–Wolf、OAS；未来窗口与 full-valid-window（非未来敏感性）两种协议，共 1,560 条 partition 记录。每个相关矩阵 4,560 条无向边；无估计器警告或失败。
- 未来窗口 edge Spearman：Pearson 0.05043（object bootstrap 95% CI [0.03647, 0.06671]）；Ledoit–Wolf 0.05122 [0.03729, 0.06748]；OAS 0.05142 [0.03699, 0.06817]。收缩估计器只带来极小的排序一致性变化。
- r≥0.7 正边 Jaccard：Pearson 0.04059 [0.02821, 0.05399]；Ledoit–Wolf 0.04051 [0.02600, 0.05603]；OAS 0.04100 [0.02594, 0.05701]。没有实质改善。收缩方法降低平均绝对边差异，但会把相关估计向零收缩，且 r≥0.5 Jaccard 反而下降；不能把较低绝对差异直接解释为真实物理目标更稳定。
- full-valid-window 敏感性结果仍然很弱：edge Spearman Pearson 0.05592、Ledoit–Wolf 0.05647、OAS 0.05667。
- 决策：Ledoit–Wolf/OAS 改善数值正则化，但没有解决跨 grasp 的边排序与高阈值正边一致性问题。Weight 的六个 grasp interval 仍不得作为可交换 Recording 加入 Reptile/ANIL 主训练集；保留为重复抓取变异性/标签可行性分析，不宣称增加独立 object task 数。
- 完整运行摘要：`results/data_expansion/weight_variance_shrinkage_audit_20261122.md`；机器摘要：`results/data_expansion/weight_variance_shrinkage_audit_20261122.json`。本地运行包包含 config、environment、input manifest、sensor selection、逐 grasp 方差、逐 partition 指标、物体级汇总、bootstrap、paired estimator differences、run.log 和 manifest SHA-256；不包含原始压力帧。
- 后续：暂不重训 Reptile/ANIL，不进行同一验证集上的额外搜索。若未来构造跨 grasp pooled object-level target，必须另立协议并明确其 estimand 与当前 within-recording future-correlation target 不同。


## 阶段十六：Weight 跨抓取物体级关系聚合可行性（2026-11-22）

- 已冻结新 estimand 的分析协议：`docs/weight_object_level_aggregation_protocol_20261122.md`。研究问题是六次重复 grasp 是否能聚合成可复现的物体级传感器关系目标；该目标与原来的 within-recording future-correlation target 不同。
- 使用阶段十五相同的实际 `weights_lite` 输入，SHA-256：`37b1c5fcb246383cdaed552ce07e35690883545cdfdbfc118b48c1cbd2ee3cf3`。仍只使用 pressure-only 数据，不访问图像、视频、姿态等其他模态；不训练模型。
- 固定 96/548 传感器子集（seed=20261116）、压力变换 `clip((pressure-500)/150,0,1)`、每次 grasp 前 5 个有效帧剔除、未来窗口剩余帧；每物体六个完整 grasp，26 个物体，156 个 grasp intervals。
- 比较四种聚合：等 grasp Fisher-z 均值、等 grasp Fisher-z 中位数、按 future frame 数加权 Fisher-z 均值、合并 future frames 后 Pearson。每物体穷举 10 个唯一 3-vs-3 划分；未来窗口和 all-valid non-future 敏感性两套协议，共 2,080 条 partition 记录。另完成 156 条 leave-one-grasp-out 影响诊断；统计推断先在物体内汇总，再以 26 个物体为单位做 5,000 次 bootstrap（seed=20261122）。
- 未来窗口 edge Spearman：等权 Fisher 均值 0.00688（95% CI [0.00322, 0.01076]）；Fisher 中位数 0.00394 [-0.00033, 0.00800]；帧数加权 Fisher 均值 0.00696 [0.00332, 0.01071]；合并帧 Pearson 0.05043 [0.03603, 0.06646]。
- r≥0.7 正边 Jaccard：三种 Fisher 聚合分别为 0.00254、0.00257、0.00404；合并帧 Pearson 为 0.04059。Fisher 聚合虽然平均绝对边差异较低，但边排序和正边重合更差；较低绝对差异不能解释为更可靠的结构。
- Leave-one-grasp-out 与 all-six 聚合的 Spearman 均值 0.90581，但每个估计共享 5/6 个 grasp，这是重叠样本影响诊断，不是独立验证。不能将其作为目标稳定性证据。
- 六次 grasp 的 edge Fisher-z 离散程度也保留在 `heterogeneity_by_object.csv`；其跨物体中位数统计显示 grasp-level 关系存在明显变异。
- **决策：** 当前 Weight 六次 grasp 不支持可复现的跨抓取物体级关系目标。等 grasp Fisher 聚合没有改善 split-half 排序/边重合；不把 Weight 加入 Reptile/ANIL 主训练，不宣称其增加独立 object task。保留为重复抓取变异性和目标可行性结果。
- 报告：`docs/weight_object_level_aggregation_results_20261122.md`；机器摘要：`results/data_expansion/weight_object_level_aggregation_20261122.json`；复现脚本：`scripts/analyze_weight_object_aggregation.py`。本地运行包 `weight_object_aggregation_20261122_run02.zip` 包含配置、数据哈希、传感器选择、相关矩阵派生文件、逐划分/物体/LOO 结果、bootstrap、日志和 SHA-256 manifest，不含原始压力帧。
- 当前研究方向不变：以原 classification pressure-only 数据上的跨物体预测为主线；Weight 暂作负向可行性/重复抓取变异性分析。任何未来新目标均需单独定义 estimand 和独立物体级评估，不能与原任务混报。


## 阶段十七：外部 pressure-only 数据集接入审计准备（2026-11-22）

- 阶段十六确认 STAG Weight 六次 grasp 的简单物体级 Fisher 聚合不能提供可靠的跨抓取关系排序；不继续围绕同一批 Weight 标签调参。
- 下一步转向独立外部对象数据，先审计 TUM-ICS Hide-and-Seek（HAS），不立即训练模型。公开数据卡标注约 1,120,116 帧、约 4.3k trajectories、33 个物理物体、61 个 object-weight 类别、CC BY 4.0；官方 schema 中有 44 维 `obs_skin_force_left`。官方仓库说明 temporal windows 按 `episode_index` 分组且不跨 episode；标签映射把 61 个 object-weight labels 映射为 34 个 object labels（含 negative）和 4 个 weight labels。
- 已冻结 `docs/has_pressure_only_external_data_audit_protocol_20261122.md`。允许读取的字段仅限 `obs_skin_force_left`、`timestamp`、`episode_index`、`classification_label` 与官方 label mapping；禁止使用 point-cloud、pose、proximity、distance、DAMI、wrist F/T 等字段。该数据不含本研究需要使用的视频模态；本阶段不接入任何视觉信息。
- HAS 为机器人触觉皮肤，与 STAG 人手压力阵列硬件不同。只作为独立外部域研究；44 通道不可与 STAG 的 96/548 通道拼接，独立建立通道处理、标准化和目标矩阵。
- 首轮只做 dataset revision/license/file manifest、episode label consistency、物理对象与 weight variant 映射、官方 split 是否 object-disjoint、episode 时长/时间戳、44 通道质量和重复 episode 数量审计。只有足够物理对象具有多个完整 episode、且仅用允许字段可定义早期/未来窗口后，才另立 HAS-specific target protocol。
- 对象级划分必须把同一物理对象的所有 weight variants 放在同一 outer fold；不能把 61 个 object-weight labels 当成 61 个独立物体。若对象级样本量不足或 episode 标签/目标定义不清，停止模型训练。
- 该数据候选来自官方数据卡与官方代码仓库：Hugging Face `TUM-ICS/Hide-and-Seek`；GitHub `tactile-hide-and-seek/Tactile-Hide-and-Seek`。许可为 CC BY 4.0，使用时需按要求署名。外部结果与 STAG 结果分开报告，不混合原始通道或宣称同硬件验证。


## 阶段十七实施补充（2026-11-23：HAS pressure-only 审计工具链已提交）

- 已新增 `scripts/audit_has_pressure_only.py`，严格只保留 `obs_skin_force_left`、`timestamp`、`episode_index`、`classification_label` 四个字段；采用 streaming 逐行读取，不写出原始压力向量。
- 已新增 `data/has_label_mapping.json`，映射内容取自官方 `tactile-hide-and-seek/Tactile-Hide-and-Seek` 仓库的 `data/label_mapping.json`。
- 已新增手动触发 workflow：`.github/workflows/has-pressure-only-audit.yml`。workflow 上传 90 天 artifact，包含 config、run.log、episode inventory、object split audit、channel quality、label consistency、summary 和 SHA-256 manifest。
- 本次提交只建立可复现审计工具链，**尚未运行 workflow，也未下载/实测完整数据**。因此 episode 数、物体级 split overlap、时间戳质量、通道统计及 Go/No-Go 仍全部待实测。
- GitHub commits：script `93d67833ff53139c9c7d0e32d4e305b0d8f432aa`；mapping `abde74f90af2d7708ee4d8d2e67b6ab8ffefcf7a`；workflow `f170f1c80c91ef6f444e3af66eb946dcd476c15b`。
- 执行方式：在 GitHub Actions 选择 “HAS pressure-only external audit”，输入 `max_rows_per_split=0` 运行完整审计；首次可先用小样本检查工具链，但小样本结果不得用于 Go/No-Go。
