# 第七阶段：统计预测基线实验

## 1. 研究目标

评估最早五个有效接触压力帧能否预测其后的传感器协同关系。研究范围为 STAG pressure-only；不使用同步视频或多模态信息。预测单位为 Recording 中的无向传感器对，外层泛化单位为物体。

## 2. 数据与协议

- 26 个非空物体、78 个 Recording；固定外层划分为 Meta-train 18/54、Meta-validation 4/12、Meta-test 4/12（物体/Recording）。
- 每个 Recording 选取最早 5 个 hasValidLabel=True 帧作为输入；目标为其后所有有效接触帧上的 Pearson 相关。
- 为控制计算量，沿用此前稳定性分析的固定 96/548 传感器子集，每个 Recording 4,560 条无向边。
- 压力变换沿用前一阶段块 Bootstrap 设置：clip((pressure-500)/150, 0, 1)。
- 输入特征只由最早五帧构造：每个传感器的均值、标准差、最小值、最大值、极差、按真实时间戳计算的线性斜率、首末值；对无向边构造对称的和、绝对差、乘积特征，并加入传感器对的空间几何特征及早期 Pearson 相关。
- 标准化器和全局模型只在 Meta-train 上拟合。Meta-validation / Meta-test 的 Query 不参与训练或调参。
- 对每个验证/测试物体穷举 1-shot 与 2-shot Support / Query 组合。Support 用于任务内适应；Query 只用于评估。先对 Episode 汇总，再按物体汇总，避免将重复组合误认为独立任务。

## 3. 比较方法

1. 训练集常数预测：使用 Meta-train 的 Recording 加权平均未来相关系数。
2. Support 常数预测：使用当前 Episode Support Recording 的平均未来相关系数。
3. 历史相关：直接用最早五帧的 Pearson 相关预测未来相关。
4. Logistic Regression：辅助任务，预测未来相关是否达到 r≥0.7；报告全局模型和以 Meta-train 系数为二次先验的 Support 适应模型。
5. MLP：主连续预测模型，隐藏层为 64、32；报告零样本模型和 Support 上继续训练 5 个 epoch 的适应模型。

固定配置：随机种子 20261116；MLP 50 个 epoch、学习率 0.0005、L2 系数 0.001；Logistic Regression C=1、balanced class weights；Support Logistic 二次先验惩罚系数 1。此次为首轮基线实验，不用 Meta-validation 选择超参数，避免把初步结果包装成调优后的最终性能。

## 4. Meta-test 连续目标结果

以下先在每个物体内部对三个 Episode 组合取平均，再以四个 Meta-test 物体汇总。

| 方法 | 1-shot MAE | 1-shot RMSE | 1-shot Spearman | 2-shot MAE |
|---|---:|---:|---:|---:|
| 训练集常数 | 0.1314 | 0.1718 | — | 0.1314 |
| Support 常数 | 0.1296 | 0.1730 | — | 0.1290 |
| 历史相关 | 0.4470 | 0.5334 | 0.0156 | 0.4470 |
| MLP 零样本 | **0.1100** | **0.1457** | **0.4003** | 0.1100 |
| MLP Support 适应 | 0.1130 | 0.1505 | 0.3802 | **0.1099** |

MLP 零样本相较训练集常数预测的 MAE 相对下降约 16.3%。但 Support 适应并未呈现一致收益：1-shot 平均误差反而增加，2-shot 仅有轻微改善。当前结果支持早期特征含有跨物体预测信息，但尚不能证明少样本适应或元学习有增量价值。

## 5. Meta-test 逐物体结果

| 物体 | 常数 MAE | 历史相关 MAE | MLP 零样本 MAE | MLP 2-shot 适应 MAE |
|---|---:|---:|---:|---:|
| bracket | 0.1185 | 0.4289 | 0.1011 | 0.0992 |
| clip | 0.1149 | 0.4547 | 0.0975 | 0.0969 |
| full_can | 0.1572 | 0.4624 | 0.1298 | 0.1323 |
| stapler | 0.1349 | 0.4418 | 0.1116 | 0.1110 |

四个物体上 MLP 零样本 MAE 均低于常数基线，但误差水平存在明显物体差异。full_can 是当前四个测试物体中相对困难的对象。

## 6. 辅助二分类结果与类别不平衡

Meta-train、Meta-validation、Meta-test 的 r≥0.7 边比例分别约为 0.53%、0.59%、0.71%。正类极少，因此 Accuracy 不适合作为主要指标。

Meta-test 1-shot：
- Logistic Regression 零样本：AUROC 0.8907、Average Precision 0.1078、F1 0.0498、Precision 0.0258、Recall 0.8639。
- Logistic Support 适应：AUROC 0.8867、Average Precision 0.1076、F1 0.0517、Precision 0.0268、Recall 0.8229。
- MLP 回归分数用于排序时：AUROC 0.9107、Average Precision 0.1736；但若直接以预测相关系数 0.7 为分类阈值，F1 为 0。这说明 MLP 回归输出尚未校准为二分类概率，不能将其高 AUROC 直接解释为可用的阈值分类器。

Logistic 能找到较多正边，但 Precision 仅约 2.6%。二分类仍需进一步研究阈值选择、概率校准及不同阈值敏感性。

## 7. 统计解释与局限

1. 独立样本量有限：Meta-test 仅四个物体。边数虽多，但同一 Recording 内边之间高度依赖，不能把数万条边当作独立样本进行常规显著性检验。
2. 传感器覆盖有限：本轮只使用固定 96/548 个传感器，结论不代表完整 548 传感器阵列。
3. 输入时长不完全一致：78 个 Recording 中有 18 个最早五个有效帧并不连续；这是基于有效接触帧的条件预测，不是固定连续时间窗预测。
4. 任务定义有条件性：目标在 hasValidLabel=True 的有效接触帧上计算，不等价于原始压力流上的接触检测。
5. Support 适应尚无明确增益：1-shot / 2-shot 的适应结果变化较小，当前不支持直接引入 MAML。
6. 二分类极不平衡：r≥0.7 正边比例低于 1%，需以 PR-AUC、Precision、Recall 和校准分析为主。
7. 本轮没有开展多随机种子外层划分、完整传感器阵列实验或显著性推断，不能据此宣称方法已达到最终泛化性能。

## 8. 当前结论与下一步

阶段性结论：MLP 是目前连续 Pearson 预测中表现最好的已测试基线；早期五帧直接相关不是有效预测器。Logistic 对稀有正边具有较好的排序和召回能力，但精确率较低。Support 适应没有稳定改善，因此当前证据不足以支持 MAML 的必要性。

下一步优先级：
1. 在 Meta-validation 上开展 MLP 训练轮数、Support 适应轮数和二分类阈值校准，不触碰 Meta-test；
2. 做输入连续性与输入时间跨度敏感性分析；
3. 将传感器子集扩大或比较多组固定子集；
4. 对物体级外层划分开展重复分组验证，量化跨物体波动；
5. 只有在简单模型局限明确后，再决定是否开展 GNN；MAML 必须以相对普通微调的增量收益为前提。

## 9. 复现

命令：

python scripts/run_object_coactivation_baselines.py --metadata data/interim/metadata.mat --split-manifest results/object_episode_splits/object_split_manifest.json --output-dir results/coactivation_baselines --seed 20261116 --nodes 96 --mlp-epochs 50 --adapt-epochs 5

python -m unittest discover -s tests -p 'test_object_coactivation_baselines.py' -v

输出：baseline_results.json、baseline_object_metrics.csv、baseline_recording_coverage.csv。原始压力数组不写入结果文件。
