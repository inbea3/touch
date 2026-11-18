# 触觉传感器协同激活预测研究

本仓库用于开展硕士学位论文研究：

**基于元学习与图神经网络的触觉传感器协同激活预测研究**

## 数据范围

后续研究统一使用 STAG pressure-only 数据及其配套元数据，不引入同步视频、图像标注或视觉—触觉多模态任务。事件有效性通过压力信号、官方元数据和统计敏感性分析进行内部审查；对缺失官方事件锚点的 Recording 不进行未经验证的自动补标。

## 研究目标

研究在少样本条件下，如何利用物体早期触觉观测及传感器间的统计关系与空间结构信息，预测未见物体上的传感器协同关系，并评估元学习方法对跨物体泛化和快速适应的实际贡献。

本研究将重点关注：

1. 短时间观测下协同关系标签及相关性估计的稳定性；
2. 时序统计特征、传感器空间结构和关系特征对预测性能的增量贡献；
3. 元学习在未见物体少样本预测任务中的收益、适用条件与局限；
4. 预测性能的不确定性、跨物体差异及模型复杂度与收益之间的关系。

## 当前实施阶段

- 数据阶段：已读取用户提供的 pressure-only 压缩包并完成首轮审查，真实数据审查见 `docs/data_audit.md`。
- [阶段一：数据集核验与获取](docs/data_audit.md)
- [数据目录与下载说明](data/README.md)

## 项目进度与下一步

- [当前全部研究进度、阶段结论与后续元学习实施路线](docs/project_status_and_next_steps.md)

## 研究方案

论文预期流程、方法设计、实验方案、统计分析重点和风险控制见：

- [论文预期流程与研究方案](docs/research_plan.md)

## 旧仓库成果迁移

已将 `inbea3/old` 中早期触觉物体分类的部分可复用组件接入独立探索模块。该路线不替代当前协同激活预测主线。

- [迁移记录与代码审查](docs/old_repo_migration.md)
- [早期触觉物体分类探索](docs/classification_exploration.md)
- [Recording 级误差分析与类别难度](docs/recording_error_analysis.md)
- [Pressure CNN 两折实验](docs/pressure_cnn_experiment.md)
- [第四阶段：特征消融与模型机制分析](docs/feature_ablation.md)
- [第五阶段：协同关系定义与标签审查](docs/coactivation_definition.md)
- [第五阶段补充：STAG 信号分解复现](docs/stag_signal_decomposition.md)
- [第五阶段第三项：协同关系标签稳定性实验](docs/coactivation_stability.md)
- [第五阶段补充：标签块 Bootstrap 敏感性审查](docs/coactivation_block_sensitivity.md)
- [第六阶段：物体级划分与 Support / Query Episode 构建](docs/object_episode_protocol.md)
- [第七阶段：统计预测基线实验](docs/coactivation_baseline_results.md)
- [第八阶段：重复物体级划分稳健性协议](docs/repeated_object_holdout_protocol.md)
- [第八阶段：重复物体级交叉验证协议](docs/repeated_object_cv_protocol.md)
- [第八阶段：重复物体级交叉验证结果](docs/repeated_object_cv_results.md)

可选安装与运行：

```bash
python -m pip install -r requirements-models.txt
python scripts/build_early_classification_data.py --metadata data/interim/metadata.mat
python scripts/evaluate_early_classification_baselines.py
python scripts/cross_validate_early_classification.py
python scripts/analyze_recording_errors.py
python scripts/run_feature_ablation.py
python scripts/audit_coactivation_labels.py --metadata data/interim/metadata.mat
python scripts/run_stag_signal_decomposition.py --metadata data/interim/metadata.mat --output-dir results/stag_decomposition
python scripts/audit_stag_event_windows.py --metadata data/interim/metadata.mat --output-dir results/stag_decomposition
python scripts/run_coactivation_stability.py --metadata data/interim/metadata.mat --output-dir results/coactivation_stability --k-early 5 --block-length 10 --bootstrap-reps 20 --bootstrap-nodes 96
python scripts/run_event_cluster_bootstrap.py --metadata data/interim/metadata.mat --output-dir results/coactivation_stability --k-early 5 --min-event-frames 5 --bootstrap-reps 100 --bootstrap-nodes 64
python scripts/run_coactivation_block_sensitivity.py --metadata data/interim/metadata.mat --output-dir results/coactivation_block_sensitivity --block-lengths 5 10 20 --bootstrap-reps 200 --bootstrap-nodes 96 --k-early 5 --seed 20261116
python -m unittest discover -s scripts -p 'test_coactivation_block_sensitivity.py' -v
python scripts/build_object_episode_splits.py --metadata data/interim/metadata.mat --output-dir results/object_episode_splits --seed 20261118 --k-early 5
python -m unittest discover -s tests -p 'test_object_episode_splits.py' -v
python scripts/run_object_coactivation_baselines.py --metadata data/interim/metadata.mat --split-manifest results/object_episode_splits/object_split_manifest.json --output-dir results/coactivation_baselines --seed 20261116 --nodes 96 --mlp-epochs 50 --adapt-epochs 5
python -m unittest discover -s tests -p 'test_object_coactivation_baselines.py' -v
python scripts/run_repeated_object_holdout.py --metadata data/interim/metadata.mat --output-dir results/repeated_object_holdout --n-repeats 20 --split-seed 20261118 --sensor-seed 20261116 --nodes 96 --mlp-epochs 50 --adapt-epochs 5
python -m unittest discover -s tests -p 'test_repeated_object_holdout.py' -v
python scripts/run_repeated_object_cv.py --metadata data/interim/metadata.mat --split-manifest results/object_episode_splits/object_split_manifest.json --output-dir results/repeated_object_cv --repeats 5 --folds 5 --split-seed 20261119 --sensor-seed 20261116 --model-seed 20261116 --nodes 96 --mlp-epochs 50 --adapt-epochs 5 --bootstrap-reps 5000
python -m unittest discover -s tests -p 'test_repeated_object_cv.py' -v
PYTHONPATH=src python -m unittest discover -s tests -v
```

真实数据已用于传统基线、Recording 级误差分析和 Pressure CNN 两折内部验证。官方 test 仍保持封存，相关指标均为训练集内部探索结果，不是最终测试结论。

## 研究状态说明

仓库中的研究方案属于**预先制定的研究计划**。文档中提出的模型性能、元学习收益和应用效果均为待验证假设，不代表已经完成实验或取得结果。数据集规模、数据字段、采样方式、许可条件及标签构造细节，均需在正式实验前通过原始数据和数据说明文件核验。

## 研究原则

- 按物体或独立抓取片段划分数据，避免相邻帧和滑动窗口造成信息泄露。
- 所有预测输入只能使用预测时点已经获得的信息。
- 不将同一物体内大量帧、窗口或传感器边对误认为独立物体级样本。
- 使用统计基线和机器学习基线检验复杂模型的必要性。
- 如实报告负面结果、模型局限和不确定性，不预设复杂模型必然优于简单方法。


第八阶段重复物体级交叉验证的汇总结果位于 `results/repeated_object_cv/`，完整实验包可从阶段报告获取。
