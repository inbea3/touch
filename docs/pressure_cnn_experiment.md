# Pressure CNN：两折 Recording 级验证实验

## 研究问题

在与 Logistic Regression、KNN 相同的两折 Recording 级交叉验证协议下，保留 32×32 压力图的空间位置是否能带来分类增益？

## 数据与划分

- STAG pressure-only 数据；排除 empty_hand。
- K=5 连续帧，stride=10，26 个物体类别。
- 7,918 个窗口由原始 metadata 重新构建；压力图张量为 (7918, 5, 32, 32)。
- 每折 26 个官方训练 Recording 用于拟合，另 26 个官方训练 Recording 用于验证。
- 官方 test Recording 不进入训练、标准化、早停或模型选择。
- 两折交叉验证的验证单位是 Recording；同一 Recording 的窗口始终留在同一折。

## 模型与训练设置

采用轻量级 SpatialPressureCNN：
- 输入：5 帧 × 32 × 32 压力图。
- 卷积通道：8、16、16；两次 2×2 最大池化。
- 保留空间位置：卷积特征图展平后接 64 维全连接层，不使用全局平均池化。
- Dropout=0.5。
- 优化器 AdamW，学习率 0.0005，weight decay=0.001。
- batch size=256，最多 15 epochs，验证 Macro-F1 早停，patience=6。
- 损失使用训练折类别频数平方根倒数权重。
- 每折分别计算像素均值和标准差，仅使用该折训练窗口拟合；标准化后截断到 [-5,5]。验证窗口只应用训练折参数。

固定随机种子为 2027、2028。未使用官方测试集。

## 真实数据结果

| 模型 | 两折 Accuracy 均值 | 两折 Accuracy SD | 两折 Macro-F1 均值 | 两折 Macro-F1 SD |
|---|---:|---:|---:|---:|
| Logistic Regression | 18.94% | 6.21% | 16.28% | 4.55% |
| KNN (k=5) | 23.59% | 17.51% | 21.58% | 16.23% |
| SpatialPressureCNN | 18.57% | 4.06% | 15.35% | 2.70% |

CNN 分折结果：

| 折 | Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 | 最佳 Epoch |
|---|---:|---:|---:|---:|---:|
| 1 | 21.44% | 20.93% | 17.26% | 17.98% | 13 |
| 2 | 15.70% | 14.86% | 13.44% | 14.47% | 15 |

## 结果解释

1. 当前 CNN 的平均 Accuracy 与 Logistic Regression 接近，Macro-F1 略低。
2. CNN 没有超过 KNN；KNN 在两折平均 Accuracy 和 Macro-F1 上均更高。
3. CNN 的折间标准差较小，但只有两个折，不能据此声称其稳定性显著更好。
4. 结果不支持“增加空间卷积复杂度必然提升性能”的假设。当前模型没有表现出相对传统基线的明确优势。
5. 每类只有两个官方训练 Recording，且同一 Recording 内窗口相关。两折结果只作为探索性模型诊断，不构成充分的统计显著性证据。

## 研究限制与后续

- 当前只有两折，不能用折间 SD 构造可靠的总体置信区间。
- CNN 早停依赖每类单个验证 Recording，验证指标可能受 Recording 特异性影响。
- 需要检查每折训练/验证损失曲线、逐类别 Recall 和 Recording 级误差，判断是否存在训练不足或类别表征混淆。
- 下一步不应直接扩大 CNN 或反复试网络直到验证指标变好。应先固定模型比较协议，报告负结果；若要调参，应采用预先限定的小型候选集，并明确当前数据的选择不确定性。
- 官方 test 仍保持封存，待模型与流程锁定后进行一次最终评估。

## 复现

```bash
python scripts/train_pressure_cnn_cv.py \
  --metadata data/interim/metadata.mat \
  --output results/early_classification/pressure_cnn_cv.json \
  --epochs 15
```

脚本会从 metadata 重新生成压力图窗口，因此无需把原始压力数据或派生 NPZ 上传 GitHub。运行环境需安装 `requirements-models.txt`。
