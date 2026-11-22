# G2-R3 冻结实验协议：初始化稳健性与随机拓扑对照

**状态：** 正式运行中；协议在完整外层测试结果汇总前冻结。  
**范围：** STAG pressure-only，完整548节点；不引入视频、图像、视觉标签或类别输入。

## 1. 研究问题
- R3-A：G2-R2观察到的真实Spatial-GNN k6相对Node-MLP的小幅MAE增量，是否对参数初始化稳健？
- R3-B：真实物理k6邻接是否优于保持节点度数序列与连通性的随机重连拓扑？

## 2. 固定数据与外层评估
- 数据文件：`metadata.mat`；SHA-256 `19062ed3415df6e21ae055c5149315a5350652a27d59ab55fed74e395f2f1378`。
- 548传感器、26个非空物体、78条Recording；每条Recording 149,878个无序传感器对。
- 输入：最早5帧有效pressure观测；目标：其后所有有效pressure帧计算的Pearson相关。
- 外层：split seeds 20261119、20261120、20261121；每个seed五折物体分组CV。同一物体的Recording始终同折。
- 内层验证：每个外层训练集合随机抽取至少4个物体（约20%）；仅用于选择共同训练轮数。外层测试物体不参与epoch选择。
- 评估指标按Recording先在物体内平均，再对物体等权汇总；物体是统计汇总单位。

## 3. 实验因子
- 初始化种子：20261123、20261124、20261125。
- 图条件：Node-MLP、真实Spatial-GNN k6、三个独立Rewired-GNN k6。
- 随机重连种子：2026112201、2026112202、2026112203。每张重连图必须保持k6真实图的精确度数序列、边数与连通性；保存邻接矩阵、图哈希、交换次数和与真实图的边Jaccard。
- 同一(split, fold, init)内，所有图条件共享完全相同的初始参数、训练Recording调度、pair索引、StandardScaler和训练轮数。
- 每个(init, fold)由Node-MLP的内部验证物体等权MAE选择共同epoch，所有图条件使用同一epoch；避免为GNN条件额外选择有利训练轮数。
- 优化器：AdamW，lr=0.0005，weight_decay=0.001；最多100 epochs，patience=15，min_delta=1e-5。
- 内部验证每条Recording固定抽样最多8,192个pair；外层测试评估全部有限pair。

## 4. 训练轨迹与复原要求
每次运行保存：
1. 完整stdout/stderr日志，按行flush；
2. 运行manifest：UTC起止时间、完整命令参数、Python/torch/numpy/scipy/sklearn/networkx版本、CPU/GPU信息、数据/节点映射/源代码SHA-256；
3. 外层和内层物体/Recording划分清单；
4. 每个epoch训练MSE、验证MAE、耗时、当前最佳epoch及early-stop计数；
5. 每个匹配条件的初始化状态与哈希；
6. 每个条件的refit逐epoch训练MSE；
7. 最终模型state_dict、scaler均值/尺度、optimizer state、torch RNG state、邻接矩阵哈希；
8. 每折每条件Recording级与物体级测试指标；
9. 原始邻接矩阵、图结构审计、原始CSV/JSON和原始日志；
10. 原子更新的进度checkpoint，允许从已完成的模型条件恢复。

## 5. 分析计划
- 报告每个初始化seed下真实k6与Node-MLP的配对MAE差值、方向一致性和逐物体差值。
- 报告每个随机重连图与真实k6、Node-MLP的配对差值；三个重连实例分别报告，不将其当作三个独立物体数据集。
- Bootstrap以26个物体为重采样单位；三个split和三个初始化属于重复测量，不得伪装成独立物体。
- 同步报告MAE、RMSE、Spearman；不以单一指标宣称全面优势。
- 所有结论限定于当前26个物体与本数据协议；不推断因果或普适物理协同机制。

## 6. 停止规则
若某个外层fold或模型条件失败，保存异常堆栈和此前已完成产物；修复后仅从未完成条件续跑。不得把smoke test数值作为科学结果，也不得在查看外层测试结果后改变超参数。
