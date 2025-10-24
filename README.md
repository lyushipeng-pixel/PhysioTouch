# PhysioTouch - Stage1 自监督预训练

<div align="center">

**基于MAE的触觉传感器多模态自监督学习框架**

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

---

## 📖 项目简介

PhysioTouch是一个触觉传感器数据的自监督学习框架，采用Masked Autoencoder (MAE)架构，旨在学习触觉图像的通用表征。本项目实现了**Stage1阶段的预训练**，支持静态重建和动态时序建模两种任务。

### 🎯 核心特性

- 🔥 **双任务自监督学习**：同时学习静态触觉特征和动态时序关系
- 🚀 **联合训练创新**：提出联合训练方法，在单个batch内同时优化静态和动态任务
- 📊 **精细化监控**：Wandb全方位监控，包括损失分解、梯度分析、任务贡献度等
- 🎮 **灵活配置**：支持GPU指定、权重调整、训练模式切换等
- ⚡ **分布式训练**：基于PyTorch DDP的多GPU训练支持

---

## 🏗️ 项目架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        PhysioTouch Stage1                        │
│                     Self-Supervised Pre-training                 │
└─────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
            ┌───────▼────────┐       ┌───────▼────────┐
            │  静态重建任务   │       │  动态建模任务   │
            │   (1帧图像)    │       │   (4帧序列)    │
            └───────┬────────┘       └───────┬────────┘
                    │                         │
                    │    ┌──────────────┐    │
                    └───►│ CLIP-ViT-L-14│◄───┘
                         │   Encoder    │
                         └──────┬───────┘
                                │
                         ┌──────▼───────┐
                         │  MAE Decoder │
                         └──────────────┘
```

### 训练模式对比

#### 🎲 随机交替模式 (Baseline)
```python
for batch in data:
    if random() < 0.5:
        loss = static_reconstruction(batch[0])  # 只用第1帧
    else:
        loss = dynamic_modeling(batch)          # 使用4帧
    loss.backward()
```

#### 🎯 联合训练模式 (创新点)
```python
for batch in data:
    loss_static = static_reconstruction(batch[0])     # 第1帧重建
    loss_recon = dynamic_reconstruction(batch[:3])    # 前3帧重建
    loss_pred = frame_prediction(batch[3])            # 第4帧预测
    loss_dynamic = loss_recon + loss_pred
    
    loss = α × loss_static + β × loss_dynamic        # 联合优化
    loss.backward()
```

**优势**：
- ✅ 每个batch同时优化两个任务 → 梯度更稳定
- ✅ 避免"灾难性遗忘"
- ✅ 静态和动态特征互相促进
- ✅ 更精细的任务平衡控制（通过α、β调整）

---

## 📂 项目结构

```
PhysioTouch/
├── Main_stage1.py                      # 主训练脚本
├── Stage1_engine.py                    # 训练循环引擎
├── config.py                           # 配置参数定义
├── model/
│   ├── mae_model.py                    # MAE模型实现
│   └── ...
├── dataloader/
│   ├── tactile_dataset.py              # 触觉数据集加载器
│   └── ...
├── util/
│   ├── misc.py                         # 工具函数
│   ├── lr_sched.py                     # 学习率调度
│   └── ...
├── train_joint_test.sh                 # 快速测试脚本 (1 epoch)
├── train_joint_equal_weights.sh        # 等权重训练 (α=0.5, β=0.5)
├── train_joint_balanced_weights.sh     # 平衡权重训练 (α=0.75, β=0.25)
├── train_baseline_alternating.sh       # 基线对比训练
├── WANDB_METRICS_EXPLAINED.md          # Wandb指标详细说明
├── METRICS_QUICK_REFERENCE.txt         # 指标速查表
└── README.md                           # 本文档
```

---

## 🚀 快速开始

### 环境要求

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.0+ (推荐)
- Wandb账号 (用于实验追踪)

### 安装依赖

```bash
# 激活conda环境
conda activate physiotouch  # 或你的环境名

# 安装依赖（如果还未安装）
pip install torch torchvision
pip install transformers
pip install wandb
pip install timm
```

### 数据准备

确保触觉数据集路径正确：
- 数据集CSV文件：`Tacquad.csv`
- 数据应包含4帧连续的触觉图像

### 运行训练

#### 1️⃣ 快速测试（1 epoch）

```bash
bash train_joint_test.sh
```

**配置**：
- Epochs: 1
- Batch size: 8
- 训练模式: 联合训练（α=0.5, β=0.5）
- GPU: GPU 0

**用途**：验证代码正确性，检查是否有错误

---

#### 2️⃣ 基线对比实验（100 epochs）

```bash
bash train_baseline_alternating.sh
```

**配置**：
- 训练模式: 🎲 随机交替（原始AnyTouch方法）
- Static ratio: 0.5
- Epochs: 100

**用途**：建立baseline，作为联合训练的对比参照

---

#### 3️⃣ 等权重联合训练（100 epochs）

```bash
bash train_joint_equal_weights.sh
```

**配置**：
- 训练模式: 🎯 联合训练
- α (静态权重): 0.5
- β (动态权重): 0.5
- Epochs: 100

**特点**：最直观的联合训练，静态和动态任务权重相等

---

#### 4️⃣ 平衡权重联合训练（100 epochs，推荐）

```bash
bash train_joint_balanced_weights.sh
```

**配置**：
- 训练模式: 🎯 联合训练
- α (静态权重): 0.75
- β (动态权重): 0.25
- Epochs: 100

**特点**：根据损失尺度调整权重，使静态和动态任务对总loss的贡献接近1:1

**预期效果最好的配置** ⭐

---

## ⚙️ 配置说明

### GPU指定

编辑训练脚本中的`GPU_IDS`变量：

```bash
# 使用单个GPU
GPU_IDS="0"          # 使用GPU 0
GPU_IDS="1"          # 使用GPU 1

# 使用多个GPU
GPU_IDS="0,1"        # 使用GPU 0和1
GPU_IDS="0,1,2,3"    # 使用所有4个GPU

# 使用所有可用GPU
GPU_IDS=""           # 留空
```

### 权重调整

如果需要自定义α和β：

```bash
# 在训练脚本中修改
ALPHA=0.6    # 静态权重
BETA=0.4     # 动态权重

# 或在命令行中传递
torchrun Main_stage1.py \
    --use_joint_training \
    --alpha 0.6 \
    --beta 0.4 \
    ...
```

### 其他重要参数

```bash
--batch_size 8              # 每个GPU的批次大小
--epochs 100                # 训练轮数
--lr 1.5e-4                 # 基础学习率（会根据batch size自动调整）
--warmup_epochs 40          # 学习率warmup轮数
--mask_ratio 0.75           # MAE掩码比例
--accum_iter 2              # 梯度累积步数
```

---

## 📊 监控与分析

### Wandb指标体系

训练过程中，系统会自动上传以下指标到Wandb：

#### 🎯 核心三角指标（必看）

| 指标 | 含义 | 期望趋势 | 典型值 |
|------|------|----------|--------|
| `batch/loss_total` | 总损失 | ⬇️ 下降 | 0.8 → 0.3 |
| `batch/loss_ratio` | 静态/动态比值 | ➡️ 稳定 | 0.7-1.0 |
| `batch/static_contribution` | 静态任务占比 | ➡️ ~50% | 45%-55% |

#### 🔬 损失分解（调试用）

| 指标 | 含义 | 说明 |
|------|------|------|
| `batch/loss_static` | 静态重建损失 | 第1帧的MAE loss |
| `batch/loss_dynamic` | 动态总损失 | 前3帧重建 + 第4帧预测 |
| `batch/loss_recon` | 动态重建损失 | 前3帧的重建loss |
| `batch/loss_pred` | 预测损失 | 第4帧的预测loss |

**验证公式**: `loss_dynamic = loss_recon + loss_pred`

#### ⚖️ 权重与贡献

| 指标 | 含义 | 用途 |
|------|------|------|
| `batch/weighted_static` | α × loss_static | 静态对总loss的贡献值 |
| `batch/weighted_dynamic` | β × loss_dynamic | 动态对总loss的贡献值 |
| `batch/static_contribution` | 静态实际占比 | 调参依据 |
| `batch/dynamic_contribution` | 动态实际占比 | 调参依据 |

### 实验对比

登录Wandb查看实验对比：

```
https://wandb.ai/your-username/PhysioTouch-Stage1
```

**重点对比**：
1. 最终`epoch/train_loss`（越低越好）
2. 收敛速度和稳定性
3. `loss_ratio`的稳定性（balanced应该更稳定）
4. `static_contribution`是否接近50%

### 异常检测

#### 🔴 危险信号（立即停止）
- `loss = NaN` 或 `Inf` → 梯度爆炸
- `loss`突然大幅上升（>2倍）→ 检查学习率
- `loss`完全不下降（>20 epochs）→ 检查代码

#### 🟡 警告信号（需调整参数）
- `loss_ratio > 1.5` → 动态太容易，增加β
- `loss_ratio < 0.5` → 静态太容易，增加α
- `static_contribution > 65%` → 调整α↓ 或 β↑
- `static_contribution < 35%` → 调整α↑ 或 β↓

---

## 🧪 实验建议

### 推荐实验流程

```
Step 1: 快速验证
├─ bash train_joint_test.sh (1 epoch)
└─ 检查：代码是否正常运行，Wandb是否正常记录

Step 2: 建立Baseline
├─ bash train_baseline_alternating.sh (100 epochs)
└─ 目的：建立对比基准

Step 3: 联合训练实验
├─ bash train_joint_equal_weights.sh (100 epochs)
├─ bash train_joint_balanced_weights.sh (100 epochs)
└─ 对比：哪个配置效果最好

Step 4: 超参数调优（可选）
├─ 基于最佳配置，微调α和β
└─ 例如：α=0.70/0.80, β=0.30/0.20
```

### 预测最佳配置

基于理论分析，**预测**`train_joint_balanced_weights.sh`效果最好：

**理由**：
1. `loss_dynamic`通常是`loss_static`的1.5-2倍
2. 等权重会导致动态任务占比过高（~67%）
3. α=0.75, β=0.25可使两个任务贡献接近50:50
4. 联合训练比随机交替更稳定

**但实验才是真理！** 请以实际Wandb曲线为准。

---

## 📈 性能基准

### 预期性能指标

| 配置 | loss_static | loss_dynamic | loss_total | 备注 |
|------|-------------|--------------|------------|------|
| Baseline (交替) | ~0.3 | ~0.4 | ~0.35 | 可能有震荡 |
| 等权重联合 | ~0.3 | ~0.4 | ~0.35 | 动态占比偏高 |
| 平衡权重联合 | ~0.3 | ~0.4 | **~0.33** | **预期最佳** ⭐ |

> 注：实际结果取决于数据集和训练过程

### 收敛时间

- **快速测试**: ~5-10分钟（1 epoch）
- **完整训练**: ~10-15小时（100 epochs，单GPU）
- **分布式训练**: 时间与GPU数量成反比

---

## 🛠️ 故障排除

### 常见问题

#### 1. `torchrun: command not found`

**解决**：
```bash
conda activate physiotouch  # 激活正确的环境
which torchrun              # 验证torchrun存在
```

#### 2. Wandb连接失败

**解决**：
```bash
# 登录Wandb
wandb login

# 或设置离线模式
export WANDB_MODE=offline
```

#### 3. GPU内存不足

**解决**：
- 减小batch size：`--batch_size 4`
- 增加梯度累积：`--accum_iter 4`
- 使用更少的GPU

#### 4. Loss不下降

**检查**：
1. 数据是否正确加载（打印一个batch检查）
2. 学习率是否过大或过小
3. 查看Wandb的`loss_recon`和`loss_pred`，定位问题

#### 5. 训练速度慢

**优化**：
- 使用多GPU：`GPU_IDS="0,1,2,3"`
- 启用混合精度训练（代码已默认启用）
- 检查数据加载是否成为瓶颈

---

## 📚 文档参考

- **[WANDB_METRICS_EXPLAINED.md](WANDB_METRICS_EXPLAINED.md)** - Wandb指标完整说明（446行详细文档）
- **[METRICS_QUICK_REFERENCE.txt](METRICS_QUICK_REFERENCE.txt)** - 指标速查表（1页纸快速参考）
- **[verify_update.sh](verify_update.sh)** - 代码验证脚本

---

## 🔬 技术细节

### 模型架构

- **Encoder**: CLIP-ViT-L-14 (预训练权重)
- **Decoder**: Transformer Decoder (轻量级)
- **输入尺寸**: 224×224×3
- **Patch size**: 14×14
- **序列长度**: 256 patches

### 训练策略

- **优化器**: AdamW (β1=0.9, β2=0.95)
- **学习率**: 1.5e-4 × (batch_size × accum_iter) / 256
- **Warmup**: Cosine warmup (40 epochs)
- **学习率调度**: Cosine annealing
- **权重衰减**: 0.05
- **混合精度**: FP16 (自动启用)

### 数据增强

- Random resized crop
- Random horizontal flip
- Color jitter
- Normalization

---

## 🤝 贡献指南

欢迎贡献代码、报告bug或提出新功能建议！

### 开发流程

1. Fork本仓库
2. 创建feature分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -am 'Add some feature'`
4. 推送到分支：`git push origin feature/your-feature`
5. 提交Pull Request

---

## 📄 许可证

本项目采用 [Apache 2.0 License](LICENSE)

---

## 🙏 致谢

- **AnyTouch**: 本项目基于AnyTouch的Stage1代码框架
- **CLIP**: 使用OpenAI的CLIP-ViT-L-14作为预训练backbone
- **MAE**: 采用Masked Autoencoder的自监督学习思路

---

## 📧 联系方式

如有问题或建议，请通过以下方式联系：

- 提交Issue: [GitHub Issues](https://github.com/your-repo/issues)
- 邮箱: your-email@example.com

---

## 🔖 版本历史

### v1.1.0 (2025-10-24) - 当前版本
- ✨ **新增**：联合训练模式（同时优化静态和动态任务）
- ✨ **新增**：动态损失分解监控（loss_recon + loss_pred）
- ✨ **新增**：GPU指定功能
- ✨ **新增**：详细的Wandb监控体系
- 📝 **文档**：完善的使用文档和指标说明
- 🐛 **修复**：多个训练稳定性问题

### v1.0.0 (初始版本)
- 🎉 基础MAE训练框架
- 🎲 随机交替训练模式
- 📊 基础Wandb监控

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个Star！⭐**

Made with ❤️ by PhysioTouch Team

</div>

