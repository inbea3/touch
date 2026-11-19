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