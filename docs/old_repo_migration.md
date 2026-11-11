# 旧仓库研究资产迁移记录

来源：inbea3/old；目标：inbea3/touch。

本次把旧仓库 v4 的早期触觉物体分类代码作为**独立探索模块**接入，不覆盖本仓库现有协同激活预测研究方案。两条路线预测对象、标签和统计单位不同，不能混用实验结果。

## 新增内容
- src/touch_research/classification/data.py：连续有效帧窗口、Recording 分组划分、NPZ/JSON 保存。
- src/touch_research/classification/models.py：压力图 CNN、传感器统计特征编码器、Prototypical episode 分类器。
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
