#!/bin/bash

################################################################################
# PhysioTouch Stage1 联合训练 - 快速测试脚本
# 用途: 验证代码正确性 (5 epochs)
################################################################################

echo "========================================="
echo "  PhysioTouch 联合训练 - 快速测试"
echo "  目的: 验证代码正确性"
echo "========================================="
echo ""

# 测试配置
OUTPUT_DIR="./log/joint_test"
ALPHA=0.5
BETA=0.5
GPU_IDS="0"  # ⭐ 指定使用的GPU，例如 "0" 或 "0,1,2,3"

mkdir -p ${OUTPUT_DIR}

# ⭐ 设置可见的GPU
if [ ! -z "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    echo "🎮 使用GPU: $GPU_IDS"
else
    echo "🎮 使用所有可用GPU"
fi

echo "⚙️  测试配置:"
echo "  - 训练模式: 🎯 联合训练"
echo "  - Epochs: 5 (快速测试)"
echo "  - Batch size: 8"
echo "  - Alpha/Beta: ${ALPHA}/${BETA}"
echo ""

export MASTER_PORT=29501

echo "🚀 启动测试..."
echo "========================================="
echo ""

torchrun \
    --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    Main_stage1.py \
    --batch_size 8 \
    --epochs 1 \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${OUTPUT_DIR} \
    --use_video \
    --mask_ratio 0.75 \
    --use_joint_training \
    --alpha ${ALPHA} \
    --beta ${BETA} \
    --accum_iter 2

echo ""
echo "========================================="
echo "  测试完成！"
echo "  ✅ 检查点:"
echo "    1. 训练是否正常运行"
echo "    2. loss_static 和 loss_dynamic 是否都在下降"
echo "    3. 没有 NaN 或 Inf"
echo "    4. 检查 Wandb 日志"
echo "  📊 结果目录: ${OUTPUT_DIR}/"
echo "========================================="

