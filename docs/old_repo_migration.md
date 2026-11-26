# 旧仓库研究资产迁移记录

来源：inbea3/old；目标：inbea3/touch。

本次把旧仓库 v4 的早期触觉物体分类代码作为**独立探索模块**接入，不覆盖本仓库现有协同激活预测研究方案。两条路线预测对象、标签和统计单位不同，不能混用实验结果。

## 新增内容
- src/touch_research/classification/data.py：连续有效帧窗口、Recording 分组划分、NPZ/JSON 保存。
- src/touch_research/classification/models.py：空间位置保留型 Pressure CNN、传感器统计特征编码器、Prototypical episode 分类器。\n- scripts/train_pressure_cnn_cv.py：Pressure CNN 两折 Recording 级训练与验证。\n- docs/pressure_cnn_experiment.md：CNN 实验协议、真实数据结果和限制。
- scripts/build_early_classification_data.py：从 metadata.mat 构建窗口。
- scripts/evaluate_early_classification_baselines.py：Logistic Regression 和 KNN 固定参数基线。
- tests/test_classification_pipeline.py：合成数据测试。
- requirements-models.txt：可选模型依赖。

## 相对旧实现的修正
1. 一个 Recording 的所有窗口只能进入一个集合，并显式检查集合交集。
2. 只在原始帧连续且有效的片段内生成窗口，不跨 Recording、物体、官方 split 或无效帧缺口。
3. 基线使用固定参数，不根据测试结果选模型；当前数据每类只有 2 个官方训练 Recording 和 1 个测试 Recording，调参需在训练 Recording 内采用分组交叉验证。
4. 使用 NPZ 数值数组和 JSON 元数据，不用 pickle。
5. 明确当前划分检验同类别下未见 Recording 的泛化，不代表未见类别泛化。

## 旧代码审查
- 旧 official_classification.py 在训练期间反复查看 test accuracy 并据此保存 best checkpoint，测试集因此参与模型选择；本次不迁移该训练循环。
- 旧 trial_builder.py 分类窗口以 isTransition 为锚点；本次使用连续 hasValidLabel 片段和固定 stride。
- 旧仓库记录的 AUC/accuracy 是历史报告，未在本次重新运行，不视为本仓库复现结果。

## 运行
```bash
python -m pip install -r requirements-models.txt
python scripts/build_early_classification_data.py --metadata data/interim/metadata.mat --output data/processed/early_classification.npz --k-frames 5 --stride 10
python scripts/evaluate_early_classification_baselines.py --data data/processed/early_classification.npz --seed 42
PYTHONPATH=src python -m unittest discover -s tests -v
```

用户已提供 pressure-only 压缩包并完成首轮本地审查。实际官方 split 为 55 个训练 Recording、27 个测试 Recording；每个非空物体类别为 2 个训练 Recording、1 个测试 Recording。首轮 K=5、stride=10 窗口基线结果见 docs/data_audit.md。结果仅为单一官方划分上的初步窗口级试运行，不是最终论文结论。原始数据和派生数组不提交 GitHub。

## 追加：两折 Recording 级交叉验证

新增 `two_fold_recording_cv` 和 `scripts/cross_validate_early_classification.py`。

协议：
- 仅使用官方 `splitId=0` 的 52 个可形成有效窗口的训练 Recording；
- 每个类别恰好有 2 个训练 Recording，每折按类别留出其中 1 个 Recording；
- 两折轮换验证 Recording；官方 `splitId=1` 的 26 个测试 Recording 不进入训练、验证或模型选择；
- 标准化参数仅在当前折训练窗口上拟合；
- 同时报告窗口级指标和逐 Recording Accuracy。单个 Recording 只包含一个真实类别，因此不报告单 Recording Macro-F1，避免该指标产生误导。

真实数据试运行（K=5，stride=10，26 类）：
| 模型 | 两折窗口 Accuracy 均值 ± SD | 两折窗口 Macro-F1 均值 ± SD | 验证 Recording Accuracy 均值 ± SD |
|---|---:|---:|---:|
| Logistic Regression | 0.1894 ± 0.0621 | 0.1628 ± 0.0455 | 0.1850 ± 0.1721 |
| KNN (k=5) | 0.2359 ± 0.1751 | 0.2158 ± 0.1623 | 0.2314 ± 0.1905 |

每折各包含 26 个训练 Recording、26 个验证 Recording。折间窗口数不相等（第 1 折训练/验证 3472/2160；第 2 折 2160/3472），因此折间均值和标准差仅作描述性稳定性信息，不是独立重复抽样置信区间。Recording 级 Accuracy 的离散程度较大，说明类别表现对具体 Recording 敏感。

这些内部交叉验证指标用于训练集内模型诊断与方案比较，不能替代官方测试集最终评估，也不能把窗口数当成独立样本量。每类只有两个官方训练 Recording，数据不足以支持可靠的高维超参数搜索；建议保持模型复杂度克制。

## 追加：Pressure CNN 两折实验

新增 `SpatialPressureCNN` 与 `scripts/train_pressure_cnn_cv.py`。为保留传感器空间位置，CNN 使用卷积特征图展平，而不是全局平均池化。每折的像素标准化参数只从该折训练 Recording 估计；官方 test 不参与训练、标准化、早停或模型选择。

真实数据结果（K=5，stride=10，26 类）：

| 模型 | Accuracy 均值 ± 两折 SD | Macro-F1 均值 ± 两折 SD |
|---|---:|---:|
| Logistic Regression | 18.94% ± 6.21% | 16.28% ± 4.55% |
| KNN (k=5) | 23.59% ± 17.51% | 21.58% ± 16.23% |
| SpatialPressureCNN | 18.57% ± 4.06% | 15.35% ± 2.70% |

CNN 两折 Accuracy 分别为 21.44%、15.70%；Macro-F1 分别为 17.26%、13.44%。CNN 没有超过 KNN，且平均 Macro-F1 略低于 Logistic Regression。该结果不支持“空间 CNN 在当前数据条件下优于传统基线”的主张。

两折 SD 仅是描述性折间波动，不是置信区间；每类只有两个训练 Recording，CNN 早停也依赖每类一个验证 Recording，因此不宜作强显著性推断。详细设置、结果和限制见 `docs/pressure_cnn_experiment.md`。官方测试集仍保持封存，待比较协议锁定后再进行最终评估。

## 追加：第四阶段特征消融

新增 `scripts/run_feature_ablation.py` 与 `docs/feature_ablation.md`。真实 STAG 数据两折 Recording 级内部验证结果显示：

- KNN 的 K=1 last-frame Accuracy / Macro-F1 为 23.01% / 20.96%；
- KNN 的 K=5 mean 为 22.99% / 21.05%；
- KNN 的 K=5 mean+SD 为 23.59% / 21.58%；
- KNN 在 K=5 mean+SD 不标准化时为 24.79% / 23.06%；
- MLP-64 在 K=5 mean+SD 下为 18.42% / 15.49%；
- MLP-64 在 K=5 spatial mean+SD 下为 19.75% / 17.22%。

解释边界：
1. K=1 取 K=5 窗口末帧，保持相同窗口起点与折分；不能解释为另行采集的单帧实验。
2. 当前 K=5 均值/标准差增益较小，且两折波动大，不能声称时序统计具有稳定显著增益。
3. 32×32 统计图的非传感器位置为常数零；展平网格特征与 548 个有效传感器特征在数值上等价（仅补零位置不同），不能把 MLP 的该比较当成独立空间结构检验。
4. 当前固定 KNN 在不标准化时表现略高，但仅支持当前设定下的描述性发现，不代表标准化普遍有害。
5. 分类探索的 KNN/CNN/MLP 结果不能替代论文主线的传感器协同激活预测结论。建议暂停堆叠分类模型，回到主线定义协同标签和跨物体预测任务。
