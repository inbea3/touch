# 第八阶段：重复物体级外层交叉验证协议

## 1. 目的

第七阶段固定划分只有 4 个 Meta-test 物体。为检验 MLP 相对常数基线的增量是否依赖这一次随机划分，补充重复物体级分组交叉验证。该实验是稳健性分析，不替换第七阶段固定 18/4/4 结果，也不把多个折或 Episode 当成独立物体。

## 2. 预先固定的设计

- 数据范围：STAG/HumanGrasp pressure-only；不使用视频或多模态信息。
- 全部样本：26 个非空物体、78 个 Recording，每个物体 3 个 Recording。
- 外层划分：5-fold grouped CV，重复 5 次，共 25 次外层拟合。
- 每次重复内，26 个物体随机排列后均分为 5 折，测试折大小为 6/5/5/5/5；其余 20 或 21 个物体用于训练。
- 同一物体的三个 Recording 必须始终在同一折；每个物体在每次重复中恰好作为测试物体一次，因此每个物体累计有 5 次 held-out 评估。
- 固定传感器子集为 96/548，传感器抽样种子 20261116；模型随机种子固定为 20261116。划分种子从 20261119 起，每次重复递增 1。
- 模型超参数沿用第七阶段：MLP 隐藏层 64/32、L2 系数 0.001、学习率 0.0005、训练 50 epochs；Support 适应 5 epochs。Logistic Regression 使用 C=1、balanced class weights；Support 二次先验惩罚系数 1。
- 不在本轮外层折上调参。所有折分别在本折训练对象上拟合 StandardScaler 和全局模型。

## 3. Episode 与统计单位

每个 held-out 物体仍有 3 个 Recording。对每个物体穷举 1-shot 和 2-shot 的 Support/Query 组合，每个 shot 数 3 个 Episode。Support 标签只用于任务内适应，Query 只用于评估。

每次重复中，每个物体只在一个测试折出现。跨 5 次重复，先对同一物体的 5 次 held-out 结果求平均，得到每个物体、每个 shot、每种模型的一个统计量；随后以 26 个物体为单位汇总均值、标准差和 95% percentile bootstrap CI。Bootstrap 重采样单位是物体，绝不是传感器边。

Fold-level 和 repeat-level 汇总仅用于描述划分波动。不同 Fold 的训练集高度重叠，不把 25 个 Fold 当作 25 个独立重复样本。

## 4. 比较方法与指标

沿用第七阶段方法：训练集常数、Support 常数、早期五帧 Pearson 历史相关、MLP 零样本、MLP Support 适应，以及辅助 Logistic Regression 零样本和 Support 适应。

连续 Pearson 预测的主要指标为 MAE、RMSE；Spearman 为排序补充指标。r≥0.7 是辅助二分类目标，报告 AUROC、Average Precision、F1、Precision、Recall 和正类比例。由于正类比例低于 1%，不能以 Accuracy 作为主要结论。

另做配对物体级 MAE 差值：
- 常数预测 MAE − MLP 零样本 MAE；
- MLP 零样本 MAE − MLP Support 适应 MAE；
- 常数预测 MAE − MLP Support 适应 MAE。

正差值表示右侧模型误差更低。差值的 95% CI 通过 26 个物体级差值 Bootstrap 得到。

## 5. 解释边界

1. 这是对 26 个现有物体的重复分组交叉验证，不是新增独立数据集。
2. 固定 18/4/4 测试集此前已用于第七阶段报告；本阶段所有物体都会在不同重复中进入训练集。因此，不能再把原固定四物体测试结果描述为未触碰的独立最终测试集。
3. 25 个外层拟合的训练对象有大量重叠，Fold 间标准差不是独立样本置信区间。
4. 只有 26 个独立物体，Bootstrap 区间应解释为当前物体集合内的有限样本不确定性，不能外推为对所有日常物体的保证。
5. 96/548 固定传感器子集限制仍然存在；本轮不回答完整 548 传感器阵列性能。
6. 超参数固定于第七阶段首轮基线，本实验不是 Nested CV。若根据本轮结果进一步选择模型或调参，需要另设独立外层评估或采用嵌套分组交叉验证。

## 6. 复现

\`\`\`bash
python scripts/run_repeated_object_cv.py --metadata data/interim/metadata.mat --split-manifest results/object_episode_splits/object_split_manifest.json --output-dir results/repeated_object_cv --repeats 5 --folds 5 --split-seed 20261119 --sensor-seed 20261116 --model-seed 20261116 --nodes 96 --mlp-epochs 50 --adapt-epochs 5 --bootstrap-reps 5000

python -m unittest discover -s tests -p 'test_repeated_object_cv.py' -v
\`\`\`

输出包括 repeated_cv_results.json、repeated_cv_split_manifest.json、物体×重复长表、物体级聚合表、Fold/Repeat 描述表和配对 MAE 差值表。
