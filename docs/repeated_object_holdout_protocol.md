# 第八阶段：重复物体级划分稳健性验证

## 1. 本阶段目的

第七阶段在固定 18/4/4 划分上完成首轮基线，Meta-test 只有 4 个独立物体。为检验结论是否依赖某一次随机划分，本阶段对 26 个非空物体重复执行随机 18/4/4 外层划分，并在每次划分上重新拟合相同基线。

本阶段不更换预测任务，不引入视频或多模态数据，不增加模型复杂度，也不根据测试集结果调参。

## 2. 固定与变化因素

| 因素 | 处理 |
|---|---|
| 物体外层划分 | 每次改变，18/4/4 |
| 划分随机种子 | 20261118 起连续递增 |
| 重复次数 | 默认 20 |
| 传感器子集 | 固定为同一组 96/548 个有效传感器 |
| 传感器子集种子 | 固定 20261116 |
| 输入 | 最早 5 个有效接触帧 |
| 连续预测标签 | 未来有效接触帧上的 Pearson 相关 |
| 辅助分类标签 | r≥0.7 |
| MLP 结构 | 64、32 隐藏单元 |
| MLP 全局训练 | 50 epoch，学习率 0.0005，alpha=0.001 |
| Support 适应 | MLP 继续训练 5 epoch |
| Logistic Regression | C=1，balanced class weights |
| Support Logistic | 相对 Meta-train 参数的二次先验惩罚系数 1 |

每次重复均只使用该次 Meta-train 的物体拟合标准化器和全局模型。Meta-validation 和 Meta-test 的 Query 不参与拟合。Meta-validation 仅用于描述验证表现；本轮不执行超参数选择。Meta-test 只用于该次冻结配置下的评估。

## 3. 统计汇总规则

每次划分的 Meta-test 包含 4 个物体。对每个物体，先对其 1-shot 或 2-shot 的三个 Support/Query 组合进行物体内汇总。随后计算该次划分的 4 个测试物体平均表现。

跨重复划分报告：
- 每次划分的物体平均指标；
- 重复划分层面的均值和标准差；
- 每个物体在其作为 Meta-test 出现时的表现均值和标准差；
- 每个重复划分的完整 26 物体分配清单。

**重要统计解释：** 20 次重复不是 20 组独立数据集。它们反复使用相同的 26 个物体，且不同划分的训练/验证/测试集合会重叠。因此重复划分标准差用于描述划分敏感性，不应直接当作独立重复抽样的标准误，不应据此进行普通独立样本 t 检验，也不应将所有测试 Episode 或传感器边当作独立样本。

## 4. 输出文件

脚本生成：
- repeated_split_summary.json：固定配置、种子、跨重复汇总和限制；
- repeated_split_assignments.csv：每次重复中 26 个物体的完整 train/validation/test 归属；
- repeated_split_run_metrics.csv：每次划分、shot、模型和指标的物体级汇总；
- repeated_split_object_metrics.csv：每次划分内各测试/验证物体的详细指标；
- repeated_split_object_stability.csv：同一物体多次成为测试物体时的表现波动。

## 5. 复现命令

    python scripts/run_repeated_object_holdout.py --metadata data/interim/metadata.mat --output-dir results/repeated_object_holdout --n-repeats 20 --split-seed 20261118 --sensor-seed 20261116 --nodes 96 --mlp-epochs 50 --adapt-epochs 5

    python -m unittest discover -s tests -p 'test_repeated_object_holdout.py' -v

## 6. 当前状态

**实验协议与自动化脚本已提交，真实数据重复运行尚未完成。** 当前执行环境中没有可直接读取的 data/interim/metadata.mat，因此尚未生成真实重复划分的指标。不能把脚本提交或单元测试通过表述成 20 次真实数据实验已经完成。

拿到与第七阶段相同版本的 metadata.mat 后，运行上述命令即可生成完整结果。运行完成后，再依据真实输出判断 MLP 是否在不同物体组合下稳定优于常数基线，以及 Support 适应是否有一致增量。

## 7. 结果判读标准

1. 若 MLP 在大多数重复划分中优于常数预测，且物体级表现没有被少数对象主导，可认为第七阶段的增量具有一定划分稳健性。
2. 若性能方差较大，或部分划分中 MLP 不优于常数预测，则应把结论限定为对物体组成敏感，继续研究数据异质性。
3. 若 1-shot/2-shot Support 适应仍无一致改善，不应仅因课题原计划包含 MAML 就继续堆叠元学习模块。
4. 只有在简单模型的跨物体增量稳定、且存在明确的结构性误差后，才进入 GNN 增量实验。
