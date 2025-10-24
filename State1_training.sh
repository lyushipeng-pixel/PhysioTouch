#!/bin/bash

################################################################################
# Stage1 静态重建训练脚本 - 优化版（包含检查点清理策略）
################################################################################

echo "========================================="
echo "  PhysioTouch Stage1 静态重建训练"
echo "  优化版 - 自动检查点管理"
echo "========================================="
echo ""

# 创建输出目录
OUTPUT_DIR="./log/static_optimized"
mkdir -p ${OUTPUT_DIR}

echo "⚙️  训练配置:"
echo "  - 输出目录: ${OUTPUT_DIR}"
echo "  - Batch size: 8"
echo "  - Epochs: 100"
echo "  - 静态重建比例: 50%"
echo "  - 保存策略: best.pth (最佳) + last.pth (最后)"
echo "  - 检查点保存频率: 每10个epoch (用于恢复训练)"
echo "  - 保留编号检查点: 最近2个"
echo "  - 分布式端口: 29501"
echo ""

# 设置分布式训练端口（避免与其他训练冲突）
export MASTER_PORT=29501

echo "🚀 启动训练..."
echo "========================================="
echo ""

torchrun \
    --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    Main_stage1.py \
    --batch_size 8 \
    --epochs 100 \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${OUTPUT_DIR} \
    --use_video \
    --static_ratio 0.5 \
    --save_freq 10 \
    --keep_last_n 2 \
    --accum_iter 2

echo ""
echo "========================================="
echo "  训练完成！"
echo "  检查结果: ${OUTPUT_DIR}/"
echo "========================================="

