#!/bin/bash

################################################################################
# PhysioTouch Stage1 Baseline训练脚本
# 训练模式: 随机交替 (用于对比联合训练)
################################################################################

echo "========================================="
echo "  PhysioTouch Stage1 Baseline训练"
echo "  模式: 🎲 随机交替 (对比实验)"
echo "========================================="
echo ""

# Baseline配置
OUTPUT_DIR="./log/baseline_alternating"
STATIC_RATIO=0.5
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
echo "  - 训练模式: 🎲 随机交替 (Random Alternating)"
echo "  - 静态比例: ${STATIC_RATIO}"
echo "  - Batch size: 8"
echo "  - Epochs: 20"
echo "  - 掩码比例: 75%"
echo ""

export MASTER_PORT=29502

echo "🚀 启动Baseline训练..."
echo "========================================="
echo ""

torchrun \
    --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    Main_stage1.py \
    --batch_size 8 \
    --epochs 20 \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${OUTPUT_DIR} \
    --use_video \
    --mask_ratio 0.75 \
    --static_ratio ${STATIC_RATIO} \
    --save_freq 10 \
    --keep_last_n 2 \
    --accum_iter 2

echo ""
echo "========================================="
echo "  Baseline训练完成！"
echo "  用途: 对比联合训练的性能"
echo "  结果: ${OUTPUT_DIR}/"
echo "========================================="



