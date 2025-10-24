## Stage 1 后续操作（以 train_joint_balanced_weights.sh 为主线）

本页整理在对比实验中确认 `train_joint_balanced_weights.sh` 效果最佳之后，推荐的后续工作流程与产出规范。目标是在不修改代码的前提下，稳健复现、微调验证、完成功能性评估与定版归档。

### 目标
- 复现实验稳定性：多次独立运行统计均值/方差，排除偶然性。
- 微调但不过拟合：围绕当前最优配置做小范围超参微调，验证收益与稳健性。
- 评估可解释：细化误差分析、鲁棒性评估与可视化，定位薄弱点。
- 结果可复现：固化命令、环境、依赖与数据切分，形成最终版记录。

### 立即执行（复现实验与选择 checkpoint）
- 多次独立训练（建议 3–5 次），记录每次运行的关键指标与随机种子：

```bash
for i in 1 2 3; do bash /home/shipeng/PhysioTouch/train_joint_balanced_weights.sh; done
```

- 基于「验证集」指标选择最佳 checkpoint，避免测试集信息泄漏。
- 按 `WANDB_METRICS_EXPLAINED.md` 中的主指标定义进行记录（例如宏/微平均 F1、AUROC、AUPRC 等），与现有指标口径保持一致。

### 精炼与验证（围绕已优配置的小范围微调）
- 学习率、权重衰减、warmup、梯度裁剪、损失项平衡系数的小范围搜索（先粗后细）。
- 遵循同一评估流程与早停准则；仅使用验证集进行调参。
- 数据与划分核验：
  - 确认患者/主体级划分无泄漏；如条件允许，做 K-fold 或多折重复验证，报告均值±标准差。
  - 再核对类不平衡的度量选择，与 `WANDB_METRICS_EXPLAINED.md` 保持一致。
- 消融实验：
  - 对比 equal vs balanced vs baseline_alternating 的最终复现均值与方差。
  - 分别去掉/调整关键损失项或数据增广，验证“平衡权重”带来的具体收益。

### 误差与鲁棒性评估
- 误差分析：
  - 分层统计（类别/动作/病人/场景），绘制混淆矩阵、PR 曲线、ROC 曲线与置信度校准图。
  - 抽样失败样例做定性分析，标注常见失败模式与可疑输入条件。
- 鲁棒性评估：
  - 不同随机种子下的稳定性；不同 batch size 的影响。
  - 推理时的输入扰动（裁剪、尺度变化、轻噪声）下的表现波动。

### 定版与发布
- 锁定“最终模型”：
  - 按验证集流程确定最终 checkpoint；仅在定版前对测试集评估一次，避免反复查看测试集。
  - 固化运行命令、环境与依赖版本，打上 git tag；把核心信息记录到 `README.md` 与 `Task.md` 的“最终设置与结果”区块。
- 产物与复现：
  - 归档：最佳权重、配置、日志与评估表格；在 W&B 上为最终 run 打标签，记录主指标、均值±标准差与关键图表链接。
  - 记录完整数据切分信息与任何外部资源版本号。

### 建议的记录规范
- 运行信息：脚本名（`/home/shipeng/PhysioTouch/train_joint_balanced_weights.sh`）、时间戳、随机种子（如有）、GPU/驱动/框架版本、数据版本标识。
- 指标口径：与 `WANDB_METRICS_EXPLAINED.md` 一致；统一使用宏/微平均的定义，避免口径混用。
- 可视化：混淆矩阵、PR/ROC、学习曲线（train/val），以及关键失败样例截图或链接。
- 复现实验需报告均值±标准差；对重要比较（如 baseline vs balanced）使用相同评估脚本与阈值策略。

### 模板：复现实验命令（示例）
（如脚本暂不支持自定义种子/输出目录，也可直接重复运行；若支持参数，请在此处替换为对应参数。）

```bash
# 简单重复运行 5 次
for i in 1 2 3 4 5; do bash /home/shipeng/PhysioTouch/train_joint_balanced_weights.sh; done
```

### 模板：结果记录表（示例）

| Run | 配置备注 | 主指标（Val） | 次指标（Val） | 训练时长 | Checkpoint 标识 |
|---|---|---|---|---|---|
| 1 | balanced 默认 | Macro F1= | AUROC= | hh:mm | ckpt_xx |
| 2 | balanced 默认 | Macro F1= | AUROC= | hh:mm | ckpt_xx |
| 3 | balanced 默认 | Macro F1= | AUROC= | hh:mm | ckpt_xx |
| 平均 |  |  |  |  |  |
| 标准差 |  |  |  |  |  |

### 风险与注意事项
- 避免测试集泄漏：测试集仅在最终定版前评估一次。
- 指标稳定性：短期波动可能来自数据采样或非确定性算子，需通过重复运行消解。
- 比较公平性：所有对比使用相同数据划分、评测脚本与阈值选择策略。

### 参考文件
- `WANDB_METRICS_EXPLAINED.md`
- `Task.md`
- `README.md`


