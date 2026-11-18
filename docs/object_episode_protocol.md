# 第六阶段：物体级划分与 Support / Query Episode 构建

## 1. 目标与范围

本阶段把已审查的 pressure-only 统计标签定义转化为可复现的跨物体任务划分和少样本 Episode 清单。只使用 STAG/HumanGrasp pressure-only 压力序列与 metadata.mat，不使用同步视频、图像标注或多模态数据。

本阶段只构建数据协议和索引清单，不训练 GNN、MAML 或其他预测模型。输出文件不包含原始压力数组，也不包含逐边相关矩阵。

## 2. 数据核验

从本地 classification_lite.zip 中读取 metadata.mat，实际核验结果：

- 135,187 帧，82 个 Recording；
- 27 个 object ID，其中 empty_hand 为 ID 0；
- 排除 4 个 empty_hand Recording 后，剩余 26 个物体、78 个 Recording；
- 每个非空物体恰好对应 3 个 Recording；
- 每个 Recording 均有足够的 hasValidLabel=True 帧构造最早 5 个输入观测及其后的未来目标。

按时间戳排序后，输入取最早 5 个有效接触帧，边界为第 5 个有效帧；目标为该边界之后的全部有效接触帧。78 个 Recording 中，60 个输入帧组在原始 frame 编号上连续，18 个不连续。输入时间跨度中位数为 0.5460 秒，最大为 5.3193 秒。因此 K=5 是“最早 5 个有效帧”，不是“连续 5 帧”。

每个 Recording 的未来有效目标帧数均值为 1,009.54，中位数为 910，范围为 410–1,922。

## 3. 固定物体级外层划分

使用固定随机种子 20261118，对排序后的 26 个非空 object ID 进行一次随机排列，再分配为 18/4/4：

| 集合 | 物体数 | Recording 数 | 物体级平均未来有效帧数 |
|---|---:|---:|---:|
| Meta-train | 18 | 54 | 1,002.89 |
| Meta-validation | 4 | 12 | 982.67 |
| Meta-test | 4 | 12 | 1,066.33 |

具体物体：

- **Meta-train：** allen_key_set, ball, battery, board_eraser, cat, empty_can, gel, kiwano, lotion, mug, multimeter, safety_glasses, scissors, screw_driver, spoon, spray_can, tape, tea_box。
- **Meta-validation：** brain, chain, coin, pen。
- **Meta-test：** bracket, clip, full_can, stapler。

同一物体的三个 Recording 全部进入同一个外层集合。官方 splitId 仅写入 Recording 清单供追溯，不参与本研究的分组决策。

该 18/4/4 是固定主划分，不是经过测试集性能调优的最优划分。由于只有 4 个 Meta-test 物体，单次留出结果的物体级不确定性较高；后续应考虑补充重复物体级分组验证或外层交叉验证，不能把 Episode 数量当成独立物体数。

## 4. Episode 定义

一个物体定义为一个任务（task），一个 Recording 定义为该任务中的一个样本（sample）。对每个物体的三个 Recording 穷举：

- **1-shot：** 从三个 Recording 中选一个作为 Support，另外两个作为 Query，共 C(3,1)=3 个 Episode。
- **2-shot：** 从三个 Recording 中选两个作为 Support，剩余一个作为 Query，共 C(3,2)=3 个 Episode。

每个物体有 6 个 Episode：3 个 1-shot、3 个 2-shot。

| 集合 | 1-shot Episode | 2-shot Episode | 合计 |
|---|---:|---:|---:|
| Meta-train（18 个物体） | 54 | 54 | 108 |
| Meta-validation（4 个物体） | 12 | 12 | 24 |
| Meta-test（4 个物体） | 12 | 12 | 24 |
| **合计** | **78** | **78** | **156** |

穷举 Episode 仅提供可复现的 Support / Query 组合池，不意味着 156 个 Episode 是 156 个独立任务。真正的独立外层任务单位仍是 26 个物体，其中 Meta-train 只有 18 个物体任务。

## 5. 信息边界与防泄露规则

1. Meta-validation 和 Meta-test 物体的任何 Recording 均不得参与 Meta-train 的拟合、特征选择、标准化参数估计或超参数搜索。
2. 同一 Episode 内 Support 与 Query 不包含相同 Recording，不通过帧级随机切分构造。
3. Meta-train：Support 标签可用于任务内适应；Query 标签可用于元训练外循环目标。
4. Meta-validation：仅用 Support 标签适应；Query 结果用于模型选择，不用于梯度训练。
5. Meta-test：模型结构、超参数和决策规则冻结后，才允许用 Support 标签适应；Query 标签只用于最终报告。
6. 输入只取每个 Recording 最早 5 个 hasValidLabel=True 帧；目标只取其后的有效接触帧。未来压力值不得进入输入特征、图结构、标准化或特征选择。
7. 固定物理量程映射可以按预先声明的常数执行；任何从数据估计的均值、方差、阈值、特征选择或降维参数必须仅在 Meta-train 上拟合。
8. 同一物体不同 Support / Query 组合会复用 Recording。评估时先在物体内汇总，再在物体间汇总；不可把 24 个 Meta-test Episode 当成 24 个独立样本。

## 6. 自动化审查结果

脚本对 78 个 Recording 和 156 个 Episode 执行检查：

- 26 个物体恰好分配一次，三个集合两两不重叠；
- 78 个非空物体 Recording 全部且仅分配一次；
- 每个物体的三个 Recording 均属于同一外层集合；
- 每个 Episode 的 Support 和 Query 均非空且 Recording ID 不重叠；
- 每个 Episode 中所有 Recording 均属于同一个物体和同一个外层集合。

以上检查全部通过。单元测试 5 项全部通过。

## 7. 当前研究判断与风险

**可执行性：** 现有数据结构足以支持基于 Recording 的 1-shot / 2-shot Support / Query 任务构建；任务清单可重复生成，且不需要视频数据。

**统计局限：** 只有 18 个 Meta-train 物体任务和 4 个 Meta-test 物体任务。156 个枚举 Episode 是重复组合，不会增加独立物体数。因此目前不能据此声称 MAML 具有足够统计功效，也不能仅凭单次 4 物体测试集给出强泛化结论。

**输入时间跨度：** 18/78 个 Recording 的最早 5 个有效帧并不连续，最大时间跨度达 5.319 秒。后续预测基线必须报告该事实，并将连续 5 帧或最大输入跨度约束作为敏感性协议，不能把当前 K=5 描述成固定连续时间窗口。

**任务语义：** 当前是接触有效帧已由 hasValidLabel 预先筛选的条件预测任务，不是原始压力流上的端到端接触检测。

## 8. 下一步

1. 在固定物体划分上建立常数预测和历史相关基线；
2. 建立 Logistic Regression 与 MLP 的边预测基线，严格使用相同 Episode 和边集合；
3. 分别报告 1-shot 与 2-shot，先按物体汇总；
4. 通过基线结果评估复杂图模型是否有增量价值；
5. 再判断 MAML 是否值得实现，并补充重复物体级划分或外层交叉验证作为稳健性分析。

## 9. 复现命令

```bash
python scripts/build_object_episode_splits.py \\
  --metadata data/interim/metadata.mat \\
  --output-dir results/object_episode_splits \\
  --seed 20261118 \\
  --k-early 5

python -m unittest discover -s tests -p 'test_object_episode_splits.py' -v
```

输出包括 object_split_manifest.json、recording_manifest.csv、episode_manifest.csv 和 split_episode_audit.json。清单仅包含对象/Recording 索引、时间边界、帧数统计和 Episode 分配，不包含原始压力数组或逐边标签。
