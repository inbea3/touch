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
| `isGrap` | 可靠物体接触标记 |

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

## 7. 本地核验结果

**状态：待运行本地下载与审查。**

目前已核验官方网页和数据说明，但尚未在研究运行环境中取得并检查本地压缩包。因此本文件不把官方网页概述写成本地实测结果，也不提前宣称阶段一的数据检查已经完成。

运行脚本后，应将实测数字和异常情况补充于此，并保留审查 JSON/CSV 作为审计记录。

## 8. 参考资料

1. Sundaram, S., Kellnhofer, P., Li, Y., Zhu, J.-Y., Torralba, A., & Matusik, W. (2019). Learning the signatures of the human grasp using a scalable tactile glove. *Nature*, 569, 698–702. https://doi.org/10.1038/s41586-019-1234-z
2. STAG Project. Classification dataset documentation. https://stag.csail.mit.edu/datasets/readme_classification.html
3. STAG Project. License for use of STAG data and models. https://stag.csail.mit.edu/license.html
