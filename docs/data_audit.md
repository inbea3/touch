# 阶段一：数据集核验与获取

## 1. 阶段目标

获取合法可用的 MIT STAG / HumanGrasp 分类压力数据，核对官方数据说明与本地文件是否一致，建立可复现的数据审查流程。本阶段不进行建模、不构造训练窗口，也不生成协同标签。

## 2. 官方来源核验

| 项目 | 官方信息 |
|---|---|
| 数据集 | Scalable Tactile Glove (STAG) / HumanGrasp |
| 项目主页 | https://stag.csail.mit.edu/ |
| Classification 说明 | https://stag.csail.mit.edu/datasets/readme_classification.html |
| 许可 | https://stag.csail.mit.edu/license.html |
| 压力专用下载 | https://stag.csail.mit.edu/datasets/classification_lite.zip |
| 论文 | Sundaram et al., Nature, 2019, DOI: 10.1038/s41586-019-1234-z |

官方项目页描述 548 个传感器、26 种物体和约 135,000 帧。这些是项目级总体描述，不等同于当前下载包内的有效帧数；本地实际数量必须由数据审查脚本核验。

## 3. 官方数据结构

官方 Classification 数据说明给出的压缩包结构：

```text
classification/
├── metadata.mat
└── [batch]/
    └── [recording]/
        ├── [frame].jpg
        ├── viz/
        │   └── [frame].jpg
        └── pressuredata.mat
```

主要字段说明：

| 字段 | 含义 |
|---|---|
| `batches` | batch 名称列表 |
| `batchId` | 每帧对应的 batch 索引（零基） |
| `recordings` | recording 名称列表 |
| `recordingId` | 每帧对应的 recording 索引（零基） |
| `objects` | 物体名称列表 |
| `objectId` | 每帧对应的物体索引（零基） |
| `frame` | 对应图像的帧号（零基） |
| `ts` | 相对首帧的时间戳（秒） |
| `pressure` | 32×32 压力图；548 个像素为有效物理传感器 |
| `splits` / `splitId` | 官方实验 train/test 划分及其索引 |
| `hasValidLabel` | 官方预处理的持握有效帧标记 |
| `isBalanced` | 是否属于均衡类别采样帧 |
| `isTransition` | 接触前最后一个空帧标记 |
| `isGrasp` | 可靠物体接触标记 |

官方说明还指出，逐 recording 的 `pressuredata.mat` 包含时间戳 `tStamp`、帧号和形状为 (N\times32\times32) 的 int16 压力数组，数值范围描述为 500–1024。压力专用包是否包含全部逐 recording 文件，以实际 ZIP 内容为准。

## 4. 许可与数据管理

官方 Research License 将数据和模型限定为研究用途，禁止商业使用，并限制除备份外的复制和分发。基于数据的公开成果需要引用 Sundaram 等人的 Nature 论文。请在使用前完整阅读许可，并遵守学校/实验室的数据管理要求。

本仓库只保存下载和审查脚本、数据说明及不含原始数据的审查摘要。原始 ZIP、解压 MAT 文件及受许可约束的派生数据均不提交 Git。

## 5. 执行方法

### 5.1 安装依赖

```bash
python -m pip install -r requirements-data.txt
```

### 5.2 下载压力专用数据

```bash
python scripts/download_dataset.py
```

默认下载到 `data/raw/classification_lite.zip`。脚本会计算 SHA-256，记录文件完整性。若文件已存在，默认不会重复下载；使用 `--force` 可重新下载。

### 5.3 解压并审查

```bash
python scripts/audit_dataset.py --archive data/raw/classification_lite.zip
```

也可以手动解压后指定目录：

```bash
python scripts/audit_dataset.py --data-dir /path/to/extracted_dataset
```

审查输出默认位于 `results/data_audit/`：

- `data_audit.json`：MAT 文件变量、形状、数据类型和识别出的统计数量；
- `object_frame_counts.csv`：若对象索引字段可用，输出各对象帧数。

脚本对 MATLAB v5 MAT 使用 SciPy，对 v7.3/HDF5 MAT 尝试使用 h5py。若字段结构不符合预期，脚本保留变量清单供人工检查；不会猜测字段含义。

## 6. 本阶段核验清单

- [ ] ZIP 下载成功，记录文件大小和 SHA-256；
- [ ] 找到并读取 `metadata.mat`；
- [ ] 确认本地 MAT 格式和主要变量；
- [ ] 确认物体类别数量及名称；
- [ ] 确认 recording 数量及每个 recording 的帧数；
- [ ] 核验 `pressure` 的实际维度、数据类型和数值范围；
- [ ] 核验时间戳字段及采样间隔分布；
- [ ] 核验 548 个有效传感器位置是否可从官方数据或代码中取得；
- [ ] 核验官方 split、有效帧和接触标记的分布；
- [ ] 记录缺失值、异常值和重复帧情况。

## 7. 本地核验结果（已完成首轮审查）

**状态：已取得用户提供的官方 pressure-only 压缩包，并成功读取本地 MAT 文件。**

### 7.1 文件完整性与内容

| 项目 | 实测结果 |
|---|---:|
| ZIP 文件大小 | 38,397,353 bytes（约 36.6 MiB） |
| ZIP 条目 | 2 |
| ZIP CRC 完整性检查 | 通过 |
| 文件 1 | `metadata.mat`，38,985,951 bytes |
| 文件 2 | `readme.html`，9,754 bytes |
| MAT 读取方式 | SciPy `loadmat`，MATLAB v5 |
| SHA-256 | `ce0f27ae87e8ce2ec405f756dfe278b0b8011ef12e9de93e468b7ef31ad76c58` |

压缩包没有逐 Recording 的图片及独立 `pressuredata.mat` 文件；但 `metadata.mat` 本身包含完整的 135,187 帧压力矩阵，因此足以进行本阶段压力序列与元数据审查。

### 7.2 实际数据规模

| 指标 | 本地实测 |
|---|---:|
| 总帧数 | 135,187 |
| Recording | 82 |
| Batch | 3 |
| 物体类别（含 empty_hand） | 27 |
| 实际物体类别（不含 empty_hand） | 26 |
| `hasValidLabel` 为真的帧 | 88,269 |
| `isBalanced` 为真的帧 | 52,650 |
| `isGrasp` 为真的帧 | 448 |
| `isTransition` 为真的帧 | 1,740 |
| 压力矩阵形状 | (135187, 32, 32) |
| 压力数据类型 | int16 |
| 实际压力最小值 / 最大值 | 446 / 1000 |
| NaN / Inf | 0 / 0 |
| 非恒定压力像素数 | 548 |

注意：压力实测范围 446–1000 与官方 readme 的概括范围 500–1024 不完全一致。预处理不得将观测值强行截断到官方概括范围；当前代码改为按 1024 基准线性缩放且不裁剪，并保留该差异供论文讨论。

### 7.3 Recording 与官方划分

- 82 个 Recording 均只属于一个 `splitId`，未发现同一 Recording 同时进入官方 train 和 test 的情况。
- 官方划分共有 55 个 train Recording 和 27 个 test Recording。
- 每个非空物体类别均有 2 个 train Recording 和 1 个 test Recording；empty_hand 有 3 个 train Recording 和 1 个 test Recording。
- 因此，正式基线应使用官方 `splitId`，并按 Recording 检查独立性。不能再用随机窗口划分替代官方划分。
- 每个类别只有一个官方 test Recording，测试结果对该 Recording 的个体差异可能敏感；后续必须报告类别级指标，并在研究局限中说明，不能把数千个窗口当作数千个独立物体样本。

### 7.4 时间戳与连续性

在每个 Recording 内，时间戳严格递增。帧号在各 Recording 内连续递增。相邻时间差中位数约为 0.1360 秒；最小值约 0.1170 秒，最大值约 1.6321 秒。存在少量较大的时间间隔，窗口生成必须检查原始帧索引连续性，不能跨越缺口。

### 7.5 第一轮窗口与基线试运行

采用以下固定、可复核的初步设置：

- 排除 `empty_hand`；
- 仅保留 `hasValidLabel=True` 的帧；
- 每个窗口取连续 5 帧；
- 窗口步长为 10 帧；
- 不跨 Recording、物体类别、官方 split 或无效帧间隔；
- 548 个非恒定压力像素作为传感器输入；
- 按官方 `splitId` 划分训练集和测试集；
- Logistic Regression 与 KNN 使用固定参数，不根据测试集选模型。

得到 7,918 个窗口，其中官方 train 5,632 个、test 2,286 个；训练集覆盖 52 个 Recording，测试集覆盖 26 个 Recording，两个集合没有 Recording 重叠。两侧均覆盖全部 26 个物体类别。

| 固定基线 | Accuracy | Macro-F1 |
|---|---:|---:|
| Logistic Regression | 0.2187 | 0.2015 |
| KNN（k=5） | 0.3513 | 0.3300 |

**这些数值是本次单一官方划分上的初步窗口级试运行，不是最终论文结果。** 同一 Recording 内的窗口相关，不能据此构造普通独立同分布置信区间；后续应补充按 Recording / 类别层级的稳定性分析，并将 CNN、Prototypical Networks 与简单基线在完全一致协议下比较。当前试运行未进行超参数搜索，也未用 test 指标选择模型。

### 7.6 可复核性与许可

原始 ZIP 和 MAT 文件不上传 GitHub。数据审查 JSON、逐类别统计 CSV 和本地实验结果保留在本地运行环境；仓库仅维护审查脚本、协议和研究解释。请在许可允许范围内使用和保存派生结果。

首轮审查结论：该压缩包足以支持压力序列数据审查和早期物体分类基线；但是否足以支持主研究中的传感器协同预测，还需单独核验 `isGrasp` / `isTransition` 的定义、协同标签构造及有效任务数量。

## 8. 参考资料

1. Sundaram, S., Kellnhofer, P., Li, Y., Zhu, J.-Y., Torralba, A., & Matusik, W. (2019). Learning the signatures of the human grasp using a scalable tactile glove. *Nature*, 569, 698–702. https://doi.org/10.1038/s41586-019-1234-z
2. STAG Project. Classification dataset documentation. https://stag.csail.mit.edu/datasets/readme_classification.html
3. STAG Project. License for use of STAG data and models. https://stag.csail.mit.edu/license.html
