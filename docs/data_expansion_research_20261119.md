# 数据扩展系统调研与实验方案（2026-11-19）

## 1. 调研目标与结论

本报告针对论文主线“pressure-only 触觉传感器协同关系预测与跨物体少样本适应”，评估现有 STAG 数据的可扩展部分，以及可公开获取的外部压力/触觉数据。

**核心结论：应同时扩展 Recording 数量和独立物体任务数量，但两者必须分开报告。**

1. STAG 官方另有 blindfolded、weight-estimation、hand-pose 三套 pressure-only 小包。它们使用相同 STAG 548-sensor 硬件，下载成本极低，适合增加条件变化和同物体 Recording；但它们不能被直接计作新增独立物体任务。
2. 外部数据中，EgoTactile-OXT 最适合作为第一优先级的人手 pressure-only 外部数据：63 个 object-level tasks、767 条 trajectories，其中 692 条为直接 gloved-hand 记录，且保留 162 通道压力布局映射；约 3.75 GB。
3. Tactile Hide-and-Seek（HAS）最适合作为低成本的跨硬件独立复现：官方数据约 624 MB，33 个物理物体、约 4.3k trajectories、1.12M 帧，可从 44 维 skin-force 通道中单独提取触觉压力信号。
4. TouchScale 具有更大潜力（100 小时、15,324 episodes、929 tasks、24 participants、880 taxels/hand），但当前公开版本约 1.08 TB 且需要接受 Hugging Face 数据访问条件。应先利用元数据和少量 shard 做可行性抽样，不应直接下载全量。
5. OpenTouch、TouchAnything 等可作为后续候选，但其 object-instance 任务定义、数据许可或标注可靠性仍需核验。

所有数据扩展仍坚持 pressure-only：视频、RGB、深度、IMU、手部姿态、腕部 F/T、语言描述均不作为模型输入。若原始数据包包含这些文件，只读取压力通道，其他模态不进入训练、验证、特征工程或标签构造。

## 2. 当前 STAG 数据边界

当前论文实验实际使用 STAG classification pressure-only 数据：
- 26 个非空物体；
- 78 个物体 Recording（每个物体 3 条）；
- 548 个物理压力传感器；
- 原始约 135,187 帧；
- 当前模型为控制计算量使用固定 96/548 传感器子集；
- 当前历史外层划分为 18/4/4 个物体。

本研究的独立元学习任务单位是物体，不是帧、窗口、边或 Recording。当前 18 个 Meta-train 物体仍然只是 18 个独立任务；重复抽样和把连续帧切成窗口不会增加独立任务数。

STAG 官方项目页另提供以下 pressure-only 包（页面明确说明压力数据单独足以复现其相关实验）：

| 官方数据集 | Pressure-only 包大小 | 与当前任务的关系 | 初步决策 |
|---|---:|---|---|
| Classification | 36.6 MiB | 当前已使用 | 已纳入 |
| Blind-folded classification | 1.58 MiB | 同类物体在无视觉意识条件下的压力记录；可扩展条件变化/Recording | 立即审计并纳入候选 |
| Weight prediction | 2.85 MiB | 同一物体集的标准化抓取/重复称重记录；原论文报告 11,682 帧、多个 recording sessions | 立即审计并纳入候选 |
| Hand pose | 5.63 MiB | 标签是 G1–G7 抓取姿态及 open hand，不是物体身份；原论文报告 24,037 帧、7 个 recording sessions | 仅作姿态/混杂因素分析，不作为物体任务扩充 |

### 2.1 STAG 扩展的具体价值与限制

**Blind-folded：** 官方 metadata 仍包含 objectId、recordingId、pressure、timestamps 和 actionStartTS/actionEndTS 等字段。其价值是同一物体在不同视觉意识条件下的压力分布变化，可作为 condition-shift / cross-session 评估。不能把相同 objectId 的 blindfolded Recording 分配到不同外层集合。

**Weight prediction：** 官方数据记录同一物体集的重复抓取和重量估计过程，具有 weights、graspIndex、graspValid 等字段。原论文说明对象在多个 recording sessions 中重复采集，合计 11,682 帧。它最有希望增加同一物体的有效 Recording 数和 Support/Query 组合数，但必须确认每个对象的 recording 数、批次、重复序列和与 classification 数据的重叠情况。

**Hand pose：** metadata 中的 object 名称实际为 G1–G7 抓取类型（另含 open hand），不是日常物体 ID。不能把它的 7 个抓取类别当作 7 个新增物体任务。可用于检验协同关系是否受抓取姿态强烈影响，或作为独立的压力姿态描述性分析。

### 2.2 STAG 数据审计任务

在正式训练前，分别审计三个额外 pressure-only ZIP，并生成不可变数据清单：
- 文件 SHA-256、下载来源和许可说明；
- objectId / object name / recordingId / batchId / session；
- 每个 object 的 Recording 数、帧数、有效接触帧数；
- 548 个传感器的有效索引和压力基线；
- 时间戳单调性、采样间隔、重复帧和空 Recording；
- 与当前 classification 数据是否有相同 Recording 或同一抓取事件；
- blindfolded / weight 数据是否具有可与当前任务对齐的对象身份；
- 每个 Recording 的输入 5 帧和未来帧是否足够构造协同标签。

**严禁将同一物体的 classification、blindfolded、weight Recording 分到不同外层物体集合。** 新数据只会增加原 26 个物体内部的观测条件和 Recording 数，不会自动增加独立物体任务数。

## 3. 外部公开数据集候选与适配性

### 3.1 第一优先级：EgoTactile / EgoTactile-OXT

官方数据：
- Project: https://egotactile.github.io/
- Hugging Face 原始数据: https://huggingface.co/datasets/HustleHard/EgoTactile
- OXT 压力布局转换版: https://huggingface.co/datasets/HustleHard/EgoTactile-OXT
- Paper: https://arxiv.org/abs/2606.09243

已公开资料显示：
- 63 个日常物体、7 个物体类别、12 位参与者；
- 原始数据包含 162 个 pressure sensing locations，约 15 Hz，压力范围 0–350 N；
- OXT 转换版有 63 个 object-level tasks、767 条 trajectories、314,100 帧；
- 其中 692 条为 gloved_hand 直接采集，75 条为 bare_hand weakly-paired；
- OXT 版本约 3.75 GB，CC BY-NC 4.0；
- OXT 文档说明 162 个布局位置映射到 137 个唯一物理 sensor indices（部分 bend sensor 被展开到多个空间 cell），不能把 162 个 cell 误认为 162 个独立物理传感器；
- OXT 保留压力值（N）、对象/轨迹元数据和传感器布局映射；可以完全不读取 RGB。

**适配性：高。** 这是最接近 STAG 的人手全手压力数据，并且有更多物体、参与者和重复轨迹。它可建立独立的跨物体任务池，适合检验 Reptile / ANIL 是否在更丰富的任务分布中获益。

**主要风险：**
1. 传感器布局与 STAG 不同，不能将 162/137 个通道直接映射到 STAG 的 548 个通道。
2. OXT 的 162 个空间 cell 中存在重复物理 sensor 映射；建图前应还原唯一物理通道或在明确的 cell-level 口径下进行分析。
3. Bare-hand 子集是同步配对动作，不保证可见手与实际测压手接触严格一致；主分析应先只使用 692 条 gloved_hand 轨迹。75 条 weakly-paired bare_hand 只作为敏感性分析。
4. OXT 记录有 QC 警告/失败标记。应按官方 QC 字段做预先规定的纳入规则，并对警告/失败对象做纳入与排除敏感性分析，不可仅凭自动 QC 分数宣称数据无噪声。
5. 原始 Hugging Face Dataset Viewer 存在 Arrow schema 错误；OXT 转换版提供了对象级 Zarr 归档，但需要独立核验文件完整性和数据映射。

**获取方式：** 使用 Hugging Face datasets / huggingface_hub 获取 EgoTactile-OXT 的对象级 ZIP/Zarr；先下载 2–3 个对象做 schema、压力单位、传感器映射和时间戳 Smoke Test，再决定是否下载全部 63 个对象。只提取压力数组和必要 metadata，删除/忽略 RGB 文件，不将视频上传仓库。

**许可：** CC BY-NC 4.0，仅限非商业研究，论文需正确署名引用。

### 3.2 第二优先级：Tactile Hide-and-Seek（HAS）

官方资源：
- Dataset: https://huggingface.co/datasets/TUM-ICS/Hide-and-Seek
- Project: https://tactile-hide-and-seek.github.io/
- Code: https://github.com/tactile-hide-and-seek/Tactile-Hide-and-Seek
- Paper: https://doi.org/10.1109/ICRA57385.2026.11696735

官方 Hugging Face 卡片显示：
- 1,120,116 帧；
- 约 4.3k trajectories；
- 33 个物理物体、61 个 object-weight categories；
- 44 维左手 tactile skin force、44 维 proximity、44 维 distance；
- 88×6 tactile point cloud；
- 数据约 624 MB；
- CC BY 4.0；
- 机器人在 vision-denied 条件下进行单手 push 和双手 grasp，含不同重量变体及不同日期采集。

**适配性：中高（作为独立硬件域复现），不是 STAG 的直接合并数据。**

主分析仅允许使用 obs_skin_force_left（44 维）或预先指定的单手 force 通道。不得使用 proximity、distance、pose、DAMI wrench、wrist F/T、point-cloud 中非压力几何信息或视频。若使用点云坐标，仅可用于构造传感器固定空间位置，且需证明该坐标是传感器布局元数据而不是随时间变化的额外状态。

**任务定义：** 物理物体作为 meta-task；同一物体不同重量类别应被归入同一个外层物体组，不能将同一物体的不同重量类别分到 train/test 两边。每条 episode/trajectory 是 Support 或 Query 的最小分割单位。官方 frame-level split 不可替代 object-level split。

**风险：** 机器人触觉皮肤与人手 STAG 不同；每个对象可能有多重量版本；44 通道传感器拓扑与 STAG 不同；官方标签是 object-weight 分类，不是协同相关真值。因此应从 pressure-only 序列重新计算未来传感器 Pearson 相关标签，在 HAS 内部独立训练/评估，不能把现成分类准确率当作协同预测结果。

**获取方式：** 使用 datasets.load_dataset("TUM-ICS/Hide-and-Seek")，或官方 GitHub loader。数据约 624 MB，适合优先下载。先确认官方 split、label mapping、episode_index 和 object identity 对齐。

### 3.3 第三优先级：TouchScale

资源：
- Dataset: https://huggingface.co/datasets/2077AIDataFoundation/TouchScale
- Project / Paper: 见 Hugging Face 数据卡片

已公开数据卡片显示：
- 当前仓库公开 100 小时版本；
- 15,324 episodes、929 tasks、22 scene types、24 participants；
- 905+ objects，另含 193 个 object meshes；
- 每只手 880 个 taxels，15 个传感器 pad，逐 taxel normal force（N）；
- 压力采样率约 35–90 Hz，必须按原始 timestamp 处理；
- 430 个 WebDataset shards，完整数据约 1.08 TB；
- 每个 shard 约 2.5 GB；
- CC BY-NC 4.0，并需先接受 Hugging Face 数据访问条件；
- metadata/episodes.parquet 有 episode_id、task_id、items、actor_id、collection_date、quality_score、shard 等字段；items 自由文本与 mesh object_name 不总是一致。

**适配性：非常高，但获取成本高。** 它提供更丰富的人手 normal-force pressure、多个参与者和真实操作任务，是未来最有希望的大规模数据源。

**获取策略：绝不下载 1.08 TB 全量。**
1. 先接受数据访问条件并仅下载 metadata/episodes.parquet、左右手 glove geometry JSON、object metadata。
2. 仅用 metadata 确认稳定 object identity、重复 episode 数、任务/物体对应关系和 shard 分布。
3. 选择 10–20 个对象、每对象至少 3–5 个 episode 的小型样本；只下载对应 shard（每个约 2.5 GB），提取 left_hand_data.npz / right_hand_data.npz 的 normal-force tactile arrays。
4. 丢弃视频、深度、IMU、剪切力通道和其他模态；只保留 normal force、timestamp、object/episode ID 和固定 taxel geometry。
5. 若 schema 与对象分组可靠，再考虑扩展到 30–50 个物体；始终按 object_id 做外层划分。

**主要风险：** 905+ 是对象记录规模，193 是 mesh catalog，不能直接把两者都当成独立 object classes；items 与 mesh 名称并非一一对应。任务 ID 也不等同于物体 ID。必须先建立可审计的 object identity mapping。不同采样率不可直接按帧序号对齐。

### 3.4 第四优先级：OpenTouch

资源：
- Project: https://opentouch-tactile.github.io/
- Dataset: https://huggingface.co/datasets/rayxsong/opentouch
- Code: https://github.com/OpenTouch-MIT/opentouch
- Paper: https://arxiv.org/abs/2512.16842

项目页报告约 5 小时记录，其中 3 小时为密集标注的 contact-rich interactions；覆盖 14 个日常环境、超过 8,000 个 objects；使用 FPC-based full-hand tactile sensor，压力/触觉与姿态同步约 30 Hz。Hugging Face 版本约 21.1 GB，标注数据集 license 为 CC BY 4.0。标注由视频和触觉图经 LLM 自动生成。

**适配性：中等。** 适合做 pressure-only 跨环境、跨任务压力统计的外部验证；但“8,000 objects”是否对应可稳定追踪的 8,000 个独立 object instances、是否存在同一物体的重复 trajectories，必须从 final_annotation.zip 和 HDF5 metadata 验证。若仅有 object category 而无稳定 instance ID，就不能构造严格的 object-level meta-task。

**获取方式：** 先下载 annotations 和 1–2 个 HDF5 场景文件，验证 tactile pressure 字段、传感器索引、episode boundaries 和 object IDs；只提取 tactile pressure，不读 RGB/pose。

### 3.5 暂缓：TouchAnything / EgoTouch

资源：
- Project: https://jianyi2004.github.io/TouchAnything-Website/
- Dataset: https://huggingface.co/datasets/zhouzhoujy/EgoTouch
- Code: https://github.com/Jianyi2004/TouchAnything

项目页报告 208 manipulation tasks、1,891 episodes、1000+ objects；Hugging Face 数据目录约 88.5 GB。HDF5 文档包含 left/right pressure grids（21×21，归一化 [0,1]），但项目页其他位置描述 16×16 palm grids，存在需要核实的 schema/版本差异。GitHub 仓库 MIT license 只明确代码许可，不能据此推断数据许可。

**适配性：潜在较高，但当前不进入主数据扩展。** 需要先确认实际 pressure grid 维度、原始传感器含义、稳定 object-instance 标识、数据许可和完整下载情况，再决定是否纳入。

### 3.6 其他候选：暂不作为主扩展数据

| 数据集 | 已知规模与传感器 | 不作为主任务的原因 |
|---|---|---|
| ActionSense | 10 位受试者、约 778 分钟、20 项厨房活动；手套上的电阻式压力网格 | 主要是活动/参与者数据，公开描述中仅涉及少量（约 5）独立物体；更适合跨参与者稳健性分析 |
| ADL tactile signatures | 22 位受试者、21 项日常活动、Tekscan Grip 4256E；Zenodo 约 49.5 MB 文件 | 活动/人体功能任务，不是明确的重复 object-level grasp 任务；可作为压力域迁移补充 |
| EgoPressure | 21 位参与者、1,344 条手势序列、31 种手势、Sensel Morph touchpad | 触摸平板手势，不是对多物体的全手抓握；约 1.96 TB 且需选择性下载，任务定义不匹配 |

## 4. 推荐的分阶段数据获取顺序

### 阶段 A：零/低成本 STAG 扩展审计（先做）

下载三个额外 pressure-only 小包：
- blindfolded_lite.zip（1.58 MiB）
- weights_lite.zip（2.85 MiB）
- handposes_lite.zip（5.63 MiB）

审计后将结果写入：
- results/data_expansion/stag_extension_inventory.json
- results/data_expansion/stag_recording_manifest.csv
- docs/stag_extension_audit.md

Go/No-Go：
- Blindfolded / weight 若可识别 objectId、Recording ID、同物体新增 Recording，纳入同物体多条件分析；
- 若发现与现有 classification 重复的同一抓取事件，去重；
- Hand pose 只用于姿态/混杂因素分析，不增加物体任务计数。

### 阶段 B：获取 EgoTactile-OXT + HAS（首批外部数据）

首批预计下载量约 4.4 GB：
- EgoTactile-OXT：约 3.75 GB；
- HAS：约 624 MB。

先做小样本 Smoke Test，再决定全量下载。两套数据分别建立独立数据适配器和独立标准化器，不合并原始传感器数组，不跨数据集混合计算 Pearson 标签。

### 阶段 C：TouchScale 元数据优先、shard 选择性下载

在 Hugging Face 接受访问条件后，先只取 metadata 和 sensor geometry。确认 10–20 个物体有重复 episodes 后，选择少量 shard 做压力数据验证。若对象身份映射不可靠或每个对象缺少足够 episode，则停止，不扩大下载。

### 阶段 D：OpenTouch / TouchAnything 仅在许可和对象 ID 核验通过后考虑

不因为数据集宣传中的 object 数量大就直接纳入。必须满足：pressure-only 字段可单独读取、object instance 可稳定识别、episode 边界清楚、许可允许学术使用。

## 5. 跨数据集统一的 pressure-only 协同关系任务

当前 STAG 标签协议可作为起点，但不能把 hasValidLabel 直接套用到外部数据集。各数据集独立确定压力接触有效帧规则，并在训练前冻结。

### 5.1 任务定义

- meta-task：一个物理 object identity；
- recording/sample：一次独立 grasp / push trajectory；
- 节点：该数据集的物理压力 sensor channels；
- 输入：该 trajectory 早期最初 K=5 个有效接触观测；
- 目标：输入时点之后有效接触压力序列的 sensor-pair Pearson correlation；
- 主目标：连续 Pearson correlation（必要时 Fisher z 变换）；
- 辅助目标：正相关阈值边（保留 0.5/0.7/0.9 敏感性分析）；
- 推断单位：物体；Recording、Episode、帧、传感器边均为物体内的嵌套观测，不得当成独立 n。

对外部数据集，接触阈值只能从 pressure-only no-contact baseline / calibration 估计，不使用 RGB、视频或人工观看视频来选择接触帧。若无可审计的无接触基线，应设计压力阈值敏感性分析并将其作为协议限制。

### 5.2 传感器数与特征对齐

绝不能把 STAG 548 通道与其他数据集的 162、44、880 通道按索引直接拼接。

推荐每个数据集单独建图：
- STAG：保留当前固定 96 传感器子集；
- EgoTactile：先还原 137 个唯一物理 sensor indices，再按固定空间覆盖规则选 96 个；映射 cell 不可重复计数；
- HAS：使用单手 44 个 skin-force channels（全部使用），不混入 F/T 和 proximity；
- TouchScale：从 880 taxels 中按预先固定的空间分层规则选 96 个代表性 taxels；左右手分别分析；
- OpenTouch / TouchAnything：先确定稳定传感器索引和几何布局，再制定规则。

每个数据集都使用自己的 train-only scaler。若保留当前 31 维 edge feature schema，应重新核对各数据集的空间坐标、通道单位、缺失值、传感器固定布局，并用单元测试证明输入特征维度和含义一致。跨平台的“96 节点”仅是计算控制，不表示传感器物理位置相同。

### 5.3 数据划分

**EgoTactile（63 objects）：**
- 首选按 7 个 object categories 分层的 5-fold object-level outer CV；
- 每个 outer fold 约 12–13 个 object 作测试，其余 object 内部再划分训练和验证；
- 每个 object 的 gloved-hand trajectories 全部跟随其 object 进入同一个 outer split；
- Support / Query 必须按完整 trajectory 划分；
- 可另做 participant-held-out 敏感性分析，但不得把同一 object 泄漏到不同 object split。

**HAS（33 physical objects）：**
- 以物理 object ID 分组，不能按 61 个 object-weight labels 随机划分；
- 推荐 5-fold object-level outer CV；同一物体的所有 weight variants、episodes、frames 必须进入同一个外层 fold；
- trajectory 级 Support / Query 分离；官方 frame-level train/validation/test 仅作原任务复现参考，不代替 object-level split。

**STAG 扩展数据：**
- 同一 object 的 classification / blindfolded / weight recordings 必须整体分组；
- 当前 18/4/4 划分的 Meta-test 已被既往阶段使用，不能再称为 untouched confirmatory test；
- 新增 STAG Recording 只做同物体跨条件 / Recording 稳健性分析，不冒充新增任务。

### 5.4 模型比较

每个数据集内部独立比较：
1. Ordinary pooled MLP；
2. Ordinary MLP + Support fine-tuning；
3. Reptile；
4. First-order ANIL；
5. Support-matched controls。

先确保普通 MLP 与元学习使用相同任务划分、输入特征、训练集 scaler、随机种子和可比梯度预算。GNN 暂不作为首批扩展的必要条件；只有在更大独立任务池上，MLP 已建立稳定基线且图结构有可检验增量时再考虑。

### 5.5 数据规模敏感性实验

在同一数据集的固定 outer folds 中，预先设置 Meta-train object 数量阶梯，例如：
- 10 个训练物体；
- 20 个训练物体；
- 30 个训练物体；
- 可用时使用全部训练物体。

每个规模使用嵌套子集（小规模是大规模的子集），测试对象始终保持不变；每个规模使用相同随机种子、模型结构、训练预算和适应策略。绘制 object-level test MAE / RMSE / Spearman 随训练物体数变化的学习曲线。

重点检验：
- Ordinary MLP 是否随数据量增加而稳定改善；
- Reptile / ANIL 是否比普通 MLP 获得更大的边际收益；
- Support adaptation 是否在更多 object tasks / recordings 下变得稳定；
- 性能变化来自独立 object 数量，还是仅来自每个 object 内 Recording 数量。

不得把增加帧数、切更多窗口、增加 edge rows 当作独立任务数增长。

## 6. 统计报告与论文写法

- 所有主指标先在 trajectory 内汇总，再在 object 内汇总；
- object 是外层独立统计单位；
- 报告每个 object 的结果、均值、标准差、配对 object-level bootstrap 区间；
- 对 n 很小的数据集明确将区间称为描述性不确定性，不用 edge/frame 级显著性检验；
- 不把不同硬件平台的 MAE 直接混合成一个总平均；
- 跨数据集结果分别报告，再讨论方向一致性和 domain shift；
- 明确区分“更多 Recording”与“更多独立物体任务”；
- 说明外部数据许可、版本、SHA-256、预处理、排除规则和数据获取日期。

## 7. 当前执行决策

按最低成本和研究价值排序：

1. **立即审计 STAG blindfolded + weight + hand-pose pressure-only 小包。** 这是零硬件成本、相同 548 sensor layout 的最直接扩展；重点判断前两者是否提供可用的同物体额外 Recording。
2. **首批下载 EgoTactile-OXT 和官方 HAS。** EgoTactile 用来扩大人手跨物体任务；HAS 用来检验同一统计任务在机器人触觉硬件域中的可迁移性。
3. **TouchScale 先做 metadata/shard 可行性，不下载全量。** 只有 object identity、episode grouping 和压力数组审查通过后才扩大。
4. OpenTouch、TouchAnything 保持候选，不进入主实验，直到对象身份、压力 schema 和数据许可都清楚。

**不建议现在继续训练 MAML/GNN。** 应先完成数据盘点、外部数据 pressure-only Smoke Test 和对象级任务数量审查；只有在确认有效独立任务数量确实增加后，再冻结新的模型实验协议。

## 8. 主要公开来源

- STAG project and pressure-only archives: https://stag.csail.mit.edu/
- STAG blindfolded metadata: https://stag.csail.mit.edu/datasets/readme_blindfolded.html
- STAG weight metadata: https://stag.csail.mit.edu/datasets/readme_weights.html
- STAG hand-pose metadata: https://stag.csail.mit.edu/datasets/readme_handposes.html
- Sundaram et al. (2019), Nature: https://doi.org/10.1038/s41586-019-1234-z
- EgoTactile project: https://egotactile.github.io/
- EgoTactile-OXT: https://huggingface.co/datasets/HustleHard/EgoTactile-OXT
- TUM-ICS Hide-and-Seek: https://huggingface.co/datasets/TUM-ICS/Hide-and-Seek
- TouchScale: https://huggingface.co/datasets/2077AIDataFoundation/TouchScale
- OpenTouch: https://opentouch-tactile.github.io/
- TouchAnything / EgoTouch: https://jianyi2004.github.io/TouchAnything-Website/
- ActionSense: https://action-net.csail.mit.edu/dataset_info.html
- ADL tactile signatures: https://doi.org/10.5281/zenodo.4607289
- EgoPressure: https://huggingface.co/datasets/eth-siplab/EgoPressure
