# STAG 数据目录说明

本目录只保存本地研究数据，不将原始数据提交到 Git 仓库。

## 官方数据来源

- 项目主页：https://stag.csail.mit.edu/
- Classification 数据说明：https://stag.csail.mit.edu/datasets/readme_classification.html
- Research License：https://stag.csail.mit.edu/license.html
- 压力数据专用压缩包：https://stag.csail.mit.edu/datasets/classification_lite.zip

官方项目页列出的 classification 完整包约 13.6 GiB，压力数据专用包约 36.6 MiB。建议先使用压力数据专用包进行数据审查。

## 许可提醒

STAG 数据采用研究用途许可。许可条款限制商业用途，并限制除备份外的数据复制和再分发。请在下载和使用前自行阅读完整许可，并遵守所在机构的数据管理要求。不要把 ZIP、解压后的 MAT 文件、提取特征、训练权重或其他受许可约束的派生数据提交到本仓库或公开分发。

## 推荐目录

```text
data/
├── raw/
│   └── classification_lite.zip
└── interim/
    └── metadata.mat
```

上述路径均由仓库根目录的 `.gitignore` 排除。

## 下载与审查

在仓库根目录执行：

```bash
python -m pip install -r requirements-data.txt
python scripts/download_dataset.py
python scripts/audit_dataset.py --archive data/raw/classification_lite.zip
```

若已手动解压到其他目录：

```bash
python scripts/audit_dataset.py --data-dir /path/to/extracted_dataset
```

审查结果默认写入 `results/data_audit/`。结果文件不包含原始压力数组，但可能包含数据集类别名称及统计信息；分享前仍应检查是否符合许可和机构要求。

## 官方说明中已确认的数据结构

Classification 数据说明指出：

- ZIP 中包含 `metadata.mat`；
- 每个 batch 下包含多个 recording；
- recording 可包含图像、可视化图像和 `pressuredata.mat`；
- `pressuredata.mat` 包含时间戳 `tStamp`、帧号及形状为 (N\times32\times32) 的 int16 压力数组，压力值范围描述为 500–1024；
- `metadata.mat` 包含 `batches`、`batchId`、`recordings`、`recordingId`、`objects`、`objectId`、`frame`、`ts`、`pressure`、`splits`、`splitId`、`hasValidLabel`、`isBalanced`、`isTransition` 和 `isGrap` 等字段；
- 压力图为 32×32，其中只有 548 个像素对应手套上的物理传感器；
- 元数据中的 batch、recording、object 索引采用从 0 开始的索引。

这些是官方文档描述，不代表本地文件已经验证。实际数量、字段形状和数据质量以审查脚本对下载文件的输出为准。
