# 📊 Wandb监控参数完整解析

## 概览

`train_joint_test.sh`（联合训练模式）共记录 **20个核心参数**，分为三个层级：

1. **Batch级别** (每个batch更新) - 14个参数
2. **Epoch级别** (每个epoch汇总) - 9个参数  
3. **Summary级别** (训练结束) - 3个参数

---

## 📈 一、Batch级别参数（实时监控）

每完成一个梯度更新步骤就记录一次，用于观察训练的**细粒度动态**。

### 1.1 基础训练信息

#### `batch/loss_total`
- **含义**: 当前batch的总损失
- **计算**: `loss_total = α × loss_static + β × loss_dynamic`
- **作用**: 反映模型整体学习状态
- **期望趋势**: ⬇️ **持续下降**
- **典型范围**: 
  - 初始: ~0.8-1.2
  - 收敛: ~0.3-0.5

#### `batch/lr`
- **含义**: 当前batch的学习率
- **计算**: 由lr_scheduler动态调整
- **作用**: 控制参数更新幅度
- **期望趋势**: 
  - 📈 Warmup阶段(前40 epochs): 逐渐上升
  - 📉 之后: 余弦退火，逐渐下降
- **典型范围**: 6.25e-05 → 1e-03 → 0

#### `batch/epoch`
- **含义**: 当前所在的epoch编号
- **作用**: 时间轴标记
- **期望趋势**: ➡️ 0 → 4 (快速测试)

#### `batch/iteration`
- **含义**: 全局迭代步数
- **计算**: `global_step = 当前batch // accum_iter + epoch × (总batches // accum_iter)`
- **作用**: 全局进度追踪
- **期望趋势**: ➡️ 线性递增

---

### 1.2 核心损失分解 ⭐

#### `batch/loss_static`
- **含义**: 静态重建损失（第1帧的MAE loss）
- **计算**: MSE(重建的第1帧, 原始第1帧)
- **作用**: 衡量模型对**单帧触觉图像**的重建能力
- **物理意义**: 学习静态触觉特征表示
- **期望趋势**: ⬇️ **下降**
- **典型范围**: 
  - 初始: ~0.6-0.8
  - 收敛: ~0.2-0.4

#### `batch/loss_dynamic`
- **含义**: 动态重建+预测损失（4帧的MAE loss）
- **计算**: MSE(重建的前3帧+预测的第4帧, 原始4帧)
- **公式**: `loss_dynamic = loss_recon + loss_pred`
- **作用**: 衡量模型对**触觉时序变化**的理解能力
- **物理意义**: 学习动态触觉和时序关系
- **期望趋势**: ⬇️ **下降**
- **典型范围**: 
  - 初始: ~0.8-1.0
  - 收敛: ~0.3-0.5

#### `batch/loss_recon` ⭐ (新增)
- **含义**: 前3帧的重建损失（动态损失的重建部分）
- **计算**: `mean(MSE(pred[:3], target[:3]) * mask)`
  - 对前3帧应用掩码（75%掩码比例）
  - 只计算被掩码区域的重建损失
- **作用**: 监控模型在视频序列中的**重建能力**
- **期望趋势**: ⬇️ **下降**
- **典型范围**: 
  - 初始: ~0.5-0.7
  - 收敛: ~0.2-0.3
- **⚠️ 调试用途**: 
  - 如果`loss_recon`不下降 → 重建任务有问题（检查掩码比例、模型容量）
  - 比较`loss_recon`和`loss_static` → 了解单帧vs多帧重建的差异

#### `batch/loss_pred` ⭐ (新增)
- **含义**: 第4帧的预测损失（动态损失的预测部分）
- **计算**: `mean(MSE(pred[3], target[3]))`
  - 基于前3帧预测第4帧
  - 计算**全帧**的预测损失（无掩码）
- **作用**: 监控模型的**时序预测能力**
- **期望趋势**: ⬇️ **下降**
- **典型范围**: 
  - 初始: ~0.3-0.4
  - 收敛: ~0.1-0.2
- **⚠️ 调试用途**: 
  - 如果`loss_pred`不下降 → 预测任务太难（可能需要更多时序建模能力）
  - 比较`loss_recon`和`loss_pred` → 了解重建vs预测的难度差异
- **验证公式**: `loss_dynamic ≈ loss_recon + loss_pred`（应**严格相等**）

#### `batch/loss_ratio` ⭐⭐⭐
- **含义**: 静态损失与动态损失的比值
- **计算**: `loss_ratio = loss_static / loss_dynamic`
- **作用**: **关键调试指标！**反映两种任务的难度平衡
- **期望趋势**: ➡️ **相对稳定** (在0.7-1.0之间)
- **典型值**: 
  - 理想: ~0.8-0.9 (静态稍容易)
  - 如果 > 1.2: 动态任务太容易，可能需要增加β
  - 如果 < 0.6: 静态任务太容易，可能需要增加α
- **⚠️ 重要**: 这个值决定如何调整α和β！

---

### 1.3 加权损失（实际优化目标）

#### `batch/weighted_static`
- **含义**: 加权后的静态损失
- **计算**: `weighted_static = α × loss_static`
- **作用**: 静态损失对总loss的实际贡献值
- **期望趋势**: ⬇️ **下降**
- **典型范围**: 
  - α=0.5时: 约为loss_static的一半

#### `batch/weighted_dynamic`
- **含义**: 加权后的动态损失
- **计算**: `weighted_dynamic = β × loss_dynamic`
- **作用**: 动态损失对总loss的实际贡献值
- **期望趋势**: ⬇️ **下降**
- **典型范围**: 
  - β=0.5时: 约为loss_dynamic的一半

---

### 1.4 权重与贡献度

#### `batch/alpha`
- **含义**: 静态损失权重系数
- **设置**: 在脚本中固定为0.5
- **作用**: 控制静态任务的重要性
- **期望趋势**: ➡️ **常量** (0.5)

#### `batch/beta`
- **含义**: 动态损失权重系数
- **设置**: 在脚本中固定为0.5
- **作用**: 控制动态任务的重要性
- **期望趋势**: ➡️ **常量** (0.5)

#### `batch/static_contribution` ⭐
- **含义**: 静态损失在总loss中的实际占比
- **计算**: `static_contribution = (α × loss_static) / loss_total`
- **作用**: 显示静态任务**实际**占据多少优化资源
- **期望趋势**: ➡️ **接近期望值** (~0.5)
- **⚠️ 如果偏差大**: 需要调整α和β
  - 如果实际 > 期望: 静态loss过大，增加β
  - 如果实际 < 期望: 动态loss过大，增加α

#### `batch/dynamic_contribution` ⭐
- **含义**: 动态损失在总loss中的实际占比
- **计算**: `dynamic_contribution = (β × loss_dynamic) / loss_total`
- **作用**: 显示动态任务**实际**占据多少优化资源
- **期望趋势**: ➡️ **接近期望值** (~0.5)
- **关系**: `static_contribution + dynamic_contribution = 1.0`

#### `batch/mode`
- **含义**: 训练模式标识
- **值**: 2 (联合训练模式)
- **作用**: 用于区分不同训练模式的数据
- **期望趋势**: ➡️ **常量** (2)

---

## 🎯 二、Epoch级别参数（汇总统计）

每个epoch结束后计算整个epoch的统计值，用于观察**宏观趋势**。

### 2.1 Epoch基础信息

#### `epoch`
- **含义**: 当前epoch编号
- **作用**: 时间轴标记
- **期望趋势**: ➡️ 0 → 4

#### `epoch/train_loss`
- **含义**: 整个epoch的平均总损失
- **作用**: Epoch级别的总体学习效果
- **期望趋势**: ⬇️ **稳定下降**

#### `epoch/lr`
- **含义**: Epoch结束时的学习率
- **作用**: 学习率调度监控
- **期望趋势**: 📈📉 先升后降

#### `epoch/best_loss_so_far`
- **含义**: 到目前为止的最佳损失
- **作用**: 追踪历史最优性能
- **期望趋势**: ⬇️ **阶梯式下降**

---

### 2.2 Epoch级别的损失分解

#### `epoch/loss_static`
- **含义**: 整个epoch的平均静态损失
- **作用**: Epoch级别的静态重建性能
- **期望趋势**: ⬇️ **下降**

#### `epoch/loss_dynamic`
- **含义**: 整个epoch的平均动态损失
- **作用**: Epoch级别的动态理解性能
- **期望趋势**: ⬇️ **下降**

#### `epoch/loss_ratio`
- **含义**: Epoch级别的平均loss比值
- **作用**: 整体任务平衡性监控
- **期望趋势**: ➡️ **稳定**

#### `epoch/weighted_static`
- **含义**: Epoch级别的平均加权静态损失
- **期望趋势**: ⬇️ **下降**

#### `epoch/weighted_dynamic`
- **含义**: Epoch级别的平均加权动态损失
- **期望趋势**: ⬇️ **下降**

---

### 2.3 实际贡献度（与期望对比）⭐⭐

#### `epoch/actual_static_contribution`
- **含义**: Epoch级别静态损失的实际平均占比
- **计算**: `(α × loss_static_avg) / (α × loss_static_avg + β × loss_dynamic_avg)`
- **作用**: 验证α和β设置是否合理
- **期望值**: 0.5 (因为α=β=0.5)
- **期望趋势**: ➡️ **围绕0.5波动**
- **⚠️ 调参依据**:
  - 如果实际 > 0.6: 静态占比过高 → 减小α或增大β
  - 如果实际 < 0.4: 动态占比过高 → 增大α或减小β

#### `epoch/actual_dynamic_contribution`
- **含义**: Epoch级别动态损失的实际平均占比
- **期望值**: 0.5
- **关系**: `actual_static + actual_dynamic = 1.0`

---

## 🏆 三、Summary级别参数（训练总结）

训练结束时记录一次，保存最终结果。

#### `summary/best_epoch`
- **含义**: 最佳模型所在的epoch
- **作用**: 记录何时达到最佳性能

#### `summary/best_loss`
- **含义**: 训练过程中的最低loss
- **作用**: 评估最终性能

#### `summary/total_time_seconds`
- **含义**: 总训练时间（秒）
- **作用**: 效率统计

---

## 🎨 四、参数分组与可视化建议

### 4.1 核心监控组（最重要）

```
优先级1: 必看指标
├─ batch/loss_total          (总体趋势)
├─ batch/loss_static         (静态性能)
├─ batch/loss_dynamic        (动态性能)
└─ batch/loss_ratio ⭐       (平衡性)
```

### 4.2 调试组（用于调参）

```
优先级2: 调参依据
├─ batch/static_contribution     (实际静态占比)
├─ batch/dynamic_contribution    (实际动态占比)
├─ epoch/actual_static_contribution
└─ epoch/actual_dynamic_contribution
```

### 4.3 详细分析组

```
优先级3: 深度分析
├─ batch/weighted_static
├─ batch/weighted_dynamic
├─ batch/lr
└─ epoch/best_loss_so_far
```

---

## 📉 五、典型趋势图示例

### 理想的训练曲线

```
Loss趋势:
loss_total   ██████▄▄▄▄▃▃▃▃▂▂▂▂▁▁▁  (平滑下降)
loss_static  ███████▄▄▄▃▃▃▂▂▂▁▁▁   (下降，略快)
loss_dynamic ████████▄▄▄▃▃▃▃▂▂▂▁  (下降，略慢)

Loss Ratio:
loss_ratio   ─────0.8──0.85──0.82──0.88───  (稳定在0.7-1.0)

Contribution:
static       ─────50%──48%──52%──49%───    (围绕50%波动)
dynamic      ─────50%──52%──48%──51%───    (围绕50%波动)

Learning Rate:
lr           ▁▂▃▄▅▆▇███████▇▆▅▄▃▂▁         (先升后降)
```

---

## ⚠️ 六、异常模式识别

### 6.1 Loss Ratio异常

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| loss_ratio > 1.5 | 动态任务太容易 | 增加β权重 |
| loss_ratio < 0.5 | 静态任务太容易 | 增加α权重 |
| loss_ratio剧烈波动 | 训练不稳定 | 降低学习率或增加batch size |

### 6.2 Contribution异常

| 现象 | 期望 | 实际 | 调整策略 |
|------|------|------|---------|
| 静态占比过高 | 50% | >65% | α: 0.5→0.3, β: 0.5→0.7 |
| 动态占比过高 | 50% | >65% | α: 0.5→0.7, β: 0.5→0.3 |

### 6.3 Loss异常

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| Loss不下降 | 学习率过低 | 增加lr |
| Loss震荡 | 学习率过高 | 降低lr |
| Loss=NaN | 梯度爆炸 | 降低lr或gradient clip |
| Loss突然上升 | 学习率过高/数据异常 | 检查数据和lr schedule |

---

## 🎯 七、5轮快速测试应该观察什么？

### Epoch 0 (初始状态)
```
✅ 检查项:
  - loss_total: ~0.8-1.2 (正常初始范围)
  - loss_static vs loss_dynamic: 接近
  - loss_ratio: 0.7-1.2
  - static_contribution: 40%-60% (合理)
```

### Epoch 1-2 (学习启动)
```
✅ 检查项:
  - loss_total: 开始下降 (降幅5-10%)
  - loss_ratio: 开始稳定
  - lr: 从6e-5逐渐上升
```

### Epoch 3-4 (趋势确认)
```
✅ 检查项:
  - loss_total: 持续下降
  - loss_static 和 loss_dynamic: 都在下降
  - contribution: 稳定在期望范围
  - 没有NaN/Inf
```

### 判断标准
```
✅ 测试通过:
  - 所有loss都在下降
  - loss_ratio相对稳定 (波动 < 30%)
  - contribution接近期望值 (误差 < 20%)
  - 无异常报错

❌ 需要调试:
  - 任何loss不下降或上升
  - loss_ratio > 2.0 或 < 0.3
  - contribution严重偏离 (>70% 或 <30%)
  - 出现NaN/Inf
```

---

## 🚀 八、使用这些参数的实战流程

### Step 1: 运行快速测试
```bash
./train_joint_test.sh
```

### Step 2: 观察核心指标
```
在Wandb中查看:
1. batch/loss_total 曲线 → 是否下降？
2. batch/loss_ratio 曲线 → 是否稳定？
3. batch/*_contribution → 是否平衡？
```

### Step 3: 决策
```
如果所有指标正常 ✅
  → 运行完整训练: ./train_joint_equal_weights.sh

如果loss_ratio偏离严重 ⚠️
  → 调整α和β，重新测试

如果contribution不平衡 ⚠️
  → 使用train_joint_balanced_weights.sh
```

---

## 📚 总结

### 核心三角指标 ⭐⭐⭐
```
1. batch/loss_total          → 整体性能
2. batch/loss_ratio          → 任务平衡
3. batch/static_contribution → 资源分配
```

只要这三个指标正常，训练就在正确轨道上！

### 记忆口诀
```
Loss要下降，Ratio要稳定，Contribution要平衡
有了这三点，训练准成功！
```

---

**创建日期**: 2025-10-24  
**适用脚本**: `train_joint_test.sh`, `train_joint_equal_weights.sh`, `train_joint_balanced_weights.sh`

