#!/bin/bash

# Stage 1 训练脚本
# 使用单 GPU 进行训练

torchrun \
    --nproc_per_node=1 \
    main_stage1.py \
    --batch_size 8 \
    --epochs 2 \
    --output_dir ./log/test \
    --use_video