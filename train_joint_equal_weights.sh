#!/bin/bash

################################################################################
# PhysioTouch Stage1 联合训练脚本
# 实验配置: 等权重 (alpha=0.5, beta=0.5)
################################################################################

echo "========================================="
echo "  PhysioTouch Stage1 联合训练"
echo "  实验: 等权重配置"
echo "========================================="
echo ""

# 实验配置
EXPERIMENT_NAME="joint_alpha0.5_beta0.5"
OUTPUT_DIR="./log/${EXPERIMENT_NAME}"
ALPHA=0.5    # 静态loss权重
BETA=0.5     # 动态loss权重
GPU_IDS="0"  # ⭐ 指定使用的GPU，例如 "0" 或 "0,1,2,3"

mkdir -p ${OUTPUT_DIR}

# ⭐ 设置可见的GPU
if [ ! -z "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    echo "🎮 使用GPU: $GPU_IDS"
else
    echo "🎮 使用所有可用GPU"
fi

echo "⚙️  训练配置:"
echo "  - 实验名称: ${EXPERIMENT_NAME}"
echo "  - 输出目录: ${OUTPUT_DIR}"
echo "  - 训练模式: 🎯 联合训练 (Joint Training)"
echo "  - Alpha (静态权重): ${ALPHA}"
echo "  - Beta (动态权重): ${BETA}"
echo "  - Batch size: 8"
echo "  - Epochs: 20"
echo "  - 掩码比例: 75%"
echo "  - 梯度累积: 2"
echo "  - 分布式端口: 29501"
echo ""

export MASTER_PORT=29501

echo "🚀 启动联合训练..."
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
    --mask_ratio 0.75 \
    --use_joint_training \
    --alpha ${ALPHA} \
    --beta ${BETA} \
    --save_freq 10 \
    --keep_last_n 2 \
    --accum_iter 2

echo ""
echo "========================================="
echo "  训练完成！"
echo "  检查结果: ${OUTPUT_DIR}/"
echo "  - best.pth: 最佳模型"
echo "  - last.pth: 最后一个epoch"
echo "  - log.txt: 训练日志"
echo "========================================="

