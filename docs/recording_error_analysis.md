# Recording 级误差分析与类别难度

## 1. 目的与评估边界

本报告分析两折 Recording 级交叉验证在官方训练集上的 pooled out-of-fold（OOF）预测。每个官方训练 Recording 恰好作为验证集一次；官方 `splitId=1` 测试 Recording 完全不参与本分析。该报告用于发现类别难点和采集过程敏感性，不是最终测试性能报告。

数据设置：K=5 帧、stride=10、排除 empty_hand、仅保留 hasValidLabel 帧、548 个传感器像素、26 个物体类别。

## 2. OOF 汇总指标

| 模型 | OOF Accuracy | Balanced Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.1792 | 0.1729 | 0.1675 | 0.1729 |
| KNN (k=5) | 0.2070 | 0.2087 | 0.2048 | 0.2116 |

这里的 OOF 汇总是将两折所有验证窗口合并后计算，因而与此前“先算每折指标、再对两折求均值”的结果不同。两种统计口径不可混为一谈。本报告使用 pooled OOF 口径分析类别。

## 3. 最难识别类别

以下按 pooled OOF Recall 从低到高排列：

### Logistic Regression

| 类别 | Recall | F1 |
|---|---:|---:|
| empty_can | 0.007 | 0.009 |
| multimeter | 0.027 | 0.042 |
| full_can | 0.033 | 0.052 |
| bracket | 0.036 | 0.046 |
| spray_can | 0.037 | 0.065 |

相对容易的类别：cat（Recall 0.409）、clip（0.363）、chain（0.338）、stapler（0.268）、lotion（0.268）。

### KNN

| 类别 | Recall | F1 |
|---|---:|---:|
| multimeter | 0.018 | 0.033 |
| safety_glasses | 0.037 | 0.047 |
| spoon | 0.068 | 0.088 |
| allen_key_set | 0.071 | 0.117 |
| bracket | 0.096 | 0.120 |

相对容易的类别：ball（Recall 0.534）、clip（0.349）、stapler（0.336）、kiwano（0.299）、cat（0.293）。

## 4. Recording 级稳定性

每个 Recording 只包含一个真实物体类别，因此这里的 Recording Accuracy 表示该 Recording 内窗口分类正确比例。它不是一个具有多类别支持的单独 Macro-F1，也不能把 Recording 内窗口视为独立重复。

| 模型 | 26 个训练 Recording 的 Accuracy 均值 | Recording 间 SD |
|---|---:|---:|
| Logistic Regression | 0.1850 | 0.1721 |
| KNN | 0.2314 | 0.1905 |

两种方法的 Recording 间离散程度都很大，KNN 略高。这与两折结果的波动相一致，提示模型性能对具体抓取过程/Recording 身份较敏感。由于每类只有两个官方训练 Recording，该结果只能作为描述性证据。

## 5. 混淆矩阵和逐项结果

运行脚本会生成：

- `per_class_metrics.csv`：每类 Precision、Recall、F1、窗口数和两条训练 Recording 的平均准确率及标准差；
- `per_recording_metrics.csv`：每个训练 Recording 的窗口数、正确数和 Accuracy；
- `logistic_regression_confusion_matrix.csv`；
- `knn_confusion_matrix.csv`；
- `error_analysis.json`：协议、OOF 汇总、逐折指标、类别难度排序及限制说明。

命令：

```bash
python scripts/analyze_recording_errors.py \
  --data data/processed/early_classification.npz \
  --metadata data/interim/metadata.mat \
  --output-dir results/early_classification/error_analysis
```

这些派生报告默认保存在本地 results 目录，不上传原始数据或窗口级数组到 GitHub。

## 6. 解释与下一步

1. KNN 在 pooled OOF Accuracy、Balanced Accuracy 和 Macro-F1 上略优于 Logistic Regression，但绝对表现仍然较低，尚不能据此证明复杂模型有优势。
2. multimeter 在两种模型下都处于最难类别；bracket 也在两种模型的困难类别列表中。empty_can 对 Logistic Regression 特别困难，而 safety_glasses、spoon 对 KNN 特别困难。
3. 混淆矩阵应与原始物体外观/接触区域和压力图一起检查，不能只凭低 Recall 推断物理原因。
4. 下一阶段应先检查类别相似性、每个 Recording 的标签帧数与有效窗口分布、不同 Recording 的压力幅值差异，再决定 CNN 的输入与增强方式。
5. 所有类别难度结论来自训练集内部 OOF；官方测试 Recording 仍须保持封存，最后仅做一次预先确定的最终评估。

## 7. 统计限制

- 每类只有两个官方训练 Recording，OOF 中每类仅有两个独立采集单元。
- 同一 Recording 的滑动窗口在时间上相关；窗口级 Accuracy/Recall 是描述性指标，不可用于普通独立样本标准误、t 检验或窗口级 Bootstrap。
- 两折之间共享训练 Recording，折结果并非独立重复；折间 SD 不能解释为总体抽样不确定性的置信区间。
- 若需要强推断或广泛外推，需要增加独立 Recording / 新采集批次，或获取独立外部数据。
