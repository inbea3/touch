# 第四阶段：特征消融与模型机制分析

## 研究问题

在严格按 Recording 划分的内部验证中，识别性能主要来自单帧压力状态、短时窗口统计、特征尺度处理，还是模型结构复杂度？本阶段是旧仓库迁移的早期物体分类探索，不替代论文主线的传感器协同激活预测任务。

## 数据与验证协议

- STAG pressure-only 数据，排除 empty_hand，26 类。
- 从 5 帧连续窗口（K=5、stride=10）构建输入；所有窗口不跨 Recording、类别、官方 split 或无效帧间隔。
- 两折 Recording 级交叉验证；每折 26 个 Recording 训练、26 个 Recording 验证。
- 官方测试 Recording 不参与训练、标准化、早停或模型选择。
- K=1 条件取同一 K=5 窗口的末帧，确保窗口起点和 Recording 划分一致；它不是独立采集的一帧任务。
- KNN 与 MLP 的 StandardScaler 均在每折训练数据内拟合。

## 消融条件

1. K=1 last frame：只用窗口末帧的 548 维传感器向量。
2. K=5 mean：5 帧逐传感器均值。
3. K=5 mean+SD：逐传感器均值与标准差拼接。
4. K=5 spatial mean+SD：32×32 网格上的均值与标准差展平。
5. 特征尺度：在相同 K=5 mean+SD 特征下比较 StandardScaler 与不标准化的 KNN。
6. 模型：KNN（k=5，distance weights）与固定结构 MLP（64 hidden units）。

所有超参数固定，没有根据官方测试集调参。脚本：`scripts/run_feature_ablation.py`。

## 真实数据结果

下表为两折指标的算术平均；折间 SD 仅作描述性波动。

| 表征 | 模型 | Accuracy | Macro-F1 |
|---|---|---:|---:|
| K=1 last frame | KNN | 23.01% | 20.96% |
| K=5 mean | KNN | 22.99% | 21.05% |
| K=5 mean+SD | KNN | 23.59% | 21.58% |
| K=5 mean+SD | KNN（不标准化） | 24.79% | 23.06% |
| K=1 last frame | MLP-64 | 19.60% | 16.90% |
| K=5 mean+SD | MLP-64 | 18.42% | 15.49% |
| K=5 spatial mean+SD | MLP-64 | 19.75% | 17.22% |

补充：KNN 在 K=5 mean+SD 特征下，标准化版本 Accuracy=23.59%、Macro-F1=21.58%；不标准化版本 Accuracy=24.79%、Macro-F1=23.06%。当前固定参数下，StandardScaler 并未改善 KNN。此发现只适用于当前数据、特征和距离度量，不能推广为“标准化普遍有害”。

## 结果解释

### 1. 短时多帧信息增益有限

K=1 与 K=5 mean 的 KNN Accuracy 几乎相同（23.01% vs 22.99%）。加入窗口标准差后，Accuracy 增至 23.59%，Macro-F1 增至 21.58%，但增幅较小，且两折波动明显。当前结果不支持“5 帧统计特征带来明确、稳定的大幅提升”。

### 2. MLP 未显示超过 KNN 的优势

在 K=5 mean+SD 条件下，MLP Accuracy=18.42%、Macro-F1=15.49%，低于 KNN 的 23.59% 和 21.58%。目前证据不支持仅靠增加一个全连接隐藏层即可改善分类。

### 3. 空间统计展平不等于空间结构建模

32×32 网格统计特征中，非传感器位置为常数零。将其展平与仅保留 548 个有效传感器位置，在逐像素均值/标准差表示下只是补零位置差异，数值信息等价。因此 MLP 的空间展平结果不能作为独立的“空间结构增益”证据。CNN 的局部邻域卷积才显式引入网格邻接关系，但当前 Pressure CNN 内部验证没有超过 KNN。

### 4. 特征尺度需要谨慎处理

本次 KNN 不标准化时略好，但目前只有固定 k=5、distance weights 和两折验证。不同距离度量、特征权重与尺度处理可能交互；不应在本数据上进行大规模搜索并把最好的一次验证结果当作确定结论。

## 与既有模型的对照

Pressure CNN 两折 Accuracy=18.57%、Macro-F1=15.35%；KNN 参考基线为 23.59%、21.58%。本阶段新运行的 KNN/MLP 消融结果与既有 KNN 基线接近，说明目前最有竞争力的仍是简单邻近方法，而不是复杂神经网络。

## 结论与研究决策

当前可支持的结论：
- 当前任务下，K=5 的均值/标准差统计没有带来很大的稳定增益；
- 固定 MLP 和当前 Spatial CNN 均未超过 KNN；
- KNN 对特征尺度有一定敏感性，需在预先限定的验证协议下研究；
- 目前样本的 Recording 数量不足以支持强统计显著性结论。

当前不能支持的结论：
- 不能断言时间信息完全无用；
- 不能断言空间结构无用；
- 不能断言 KNN 在总体上显著优于深度学习；
- 不能将本分类探索结果替代论文主线的协同激活预测证据。

建议暂停增加分类模型复杂度。下一步应回到论文主线，正式定义协同关系标签、预测时点、独立统计单位及跨物体任务划分；分类模块保留为独立探索与方法比较附录。

## 复现

```bash
python -m pip install -r requirements-models.txt
python scripts/run_feature_ablation.py \
  --metadata data/interim/metadata.mat \
  --output results/early_classification/feature_ablation.json
```

原始数据及派生窗口不提交 GitHub。真实运行结果保存在本地 JSON，仓库保留脚本和方法报告。
