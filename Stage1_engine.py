"""
stage1_engine_static.py - 支持静态重建损失的训练引擎

版本: v2.1 (Bug Fix)
修复:
- 修复 ZeroDivisionError: 延迟创建 meter，避免 count=0
- 只在实际使用时才创建和更新 loss_static 和 loss_dynamic

新增功能:
- 支持静态重建损失（data_type=0，单帧图像）
- 支持动态重建+预测损失（data_type=1，4帧视频）
- 随机选择静态或动态模式进行训练
- 分别记录和监控两种损失

参考: AnyTouch - Learning Unified Static-Dynamic Representation
"""

import math
import sys
import random

import torch
import wandb  # ⭐ 添加 Wandb
import util.misc as misc
import util.lr_sched as lr_sched


def train_one_epoch(
    model: torch.nn.Module,  
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,  
    device: torch.device,   
    epoch: int,   
    loss_scaler,  
    args=None
):  
    """  
    训练一个 epoch（支持静态+动态重建损失）
    
    参数:  
        model: 要训练的模型  
        data_loader: 数据加载器，加载连续四帧触觉图片
        optimizer: 优化器  
        device: 训练设备 (cuda/cpu)
        epoch: 当前 epoch 编号  
        loss_scaler: 损失缩放器，用于混合精度训练
        args: 训练参数配置（需包含 static_ratio 参数）
        
    返回:
        dict: 训练统计信息字典
    """  
    model.train(True)  
    
    # 初始化 metric logger
    metric_logger = misc.MetricLogger(delimiter="  ")  
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    
    # ⭐ 重要：loss_static 和 loss_dynamic 会在第一次使用时动态添加
    # 不在此处提前创建，避免 ZeroDivisionError (count=0)
    # 延迟创建策略：
    #   - 只在实际使用时才创建 meter
    #   - 确保每个 meter 的 count 至少为 1
    #   - 避免打印未使用的 meter 时除零错误
    
    header = 'Epoch: [{}]'.format(epoch)  
    print_freq = 20  

    accum_iter = args.accum_iter  
    optimizer.zero_grad()
    
    # 获取静态模式的使用比例（默认0.5）
    static_ratio = getattr(args, 'static_ratio', 0.5)
    print(f'Static reconstruction ratio: {static_ratio:.1%}')

    # 统计静态和动态模式的使用次数
    static_count = 0
    dynamic_count = 0

    # ====================================================================
    # 训练循环
    # ====================================================================
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):  

        # 学习率调度 (每个迭代调整，而不是每个 epoch)
        if data_iter_step % accum_iter == 0:  
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)  
          
        # 将数据移到设备上
        # samples shape: [batch_size, 4, 3, 224, 224]
        samples = samples.to(device, non_blocking=True)  

        # ================================================================
        # ⭐ 核心改动：随机选择静态或动态模式
        # ================================================================
        use_static = random.random() < static_ratio
        
        # 前向传播（使用混合精度）
        with torch.cuda.amp.autocast():  
            if use_static:
                # ============================================
                # 静态模式：只使用第1帧重建
                # ============================================
                # 输入: [batch_size, 3, 224, 224]
                # Loss: 单帧MAE重建损失（掩码位置）
                loss, _, _ = model(samples[:, 0], data_type=0)
                data_type_name = 'static'
                static_count += 1
            else:
                # ============================================
                # 动态模式：使用4帧重建+预测
                # ============================================
                # 输入: [batch_size, 4, 3, 224, 224]
                # Loss: 前3帧重建（掩码） + 第4帧预测（全部）
                loss, _, _ = model(samples, data_type=1)
                data_type_name = 'dynamic'
                dynamic_count += 1

        loss_value = loss.item()  

        # 检查损失是否有效（避免 NaN 或 Inf）
        if not math.isfinite(loss_value):  
            print(f"Loss is {loss_value} (mode: {data_type_name}), stopping training")
            sys.exit(1)  

        # 梯度累积
        loss /= accum_iter  
        loss_scaler(
            loss, optimizer, parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0
        )  
        
        if (data_iter_step + 1) % accum_iter == 0:  
            optimizer.zero_grad()  

        torch.cuda.synchronize()  

        # ================================================================
        # 更新指标（总损失）
        # ================================================================
        metric_logger.update(loss=loss_value)
        
        # ================================================================
        # ⭐ 延迟创建和更新静态/动态损失 meter
        # 修复 ZeroDivisionError: 只在实际使用时才创建 meter
        # ================================================================
        if use_static:
            # 静态模式：检查是否需要创建 meter
            if 'loss_static' not in metric_logger.meters:
                metric_logger.add_meter('loss_static', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
            metric_logger.update(loss_static=loss_value)
        else:
            # 动态模式：检查是否需要创建 meter
            if 'loss_dynamic' not in metric_logger.meters:
                metric_logger.add_meter('loss_dynamic', misc.SmoothedValue(window_size=20, fmt='{value:.4f}'))
            metric_logger.update(loss_dynamic=loss_value)
            
        # 更新学习率
        lr = optimizer.param_groups[0]["lr"]  
        metric_logger.update(lr=lr)  

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        
        # ================================================================
        # ⭐ Wandb 记录：每个 batch 上传一次（仅主进程）
        # ================================================================
        if misc.is_main_process() and (data_iter_step + 1) % accum_iter == 0:
            # 计算全局 iteration
            global_step = data_iter_step // accum_iter + epoch * len(data_loader) // accum_iter
            
            # 基础信息
            wandb_log = {
                "batch/loss": loss_value_reduce,
                "batch/lr": lr,
                "batch/epoch": epoch,
                "batch/iteration": global_step,
            }
            
            # ⭐ 记录分 loss：静态或动态
            if use_static:
                wandb_log["batch/loss_static"] = loss_value_reduce
                wandb_log["batch/mode"] = 0  # 静态模式
            else:
                wandb_log["batch/loss_dynamic"] = loss_value_reduce
                wandb_log["batch/mode"] = 1  # 动态模式
            
            # 上传到 Wandb
            wandb.log(wandb_log)

    # ====================================================================
    # Epoch 结束后的统计
    # ====================================================================
    
    # 同步所有进程的统计信息
    metric_logger.synchronize_between_processes()  
    
    # 打印统计信息
    print("Averaged stats:", metric_logger)
    
    # 打印静态/动态模式使用统计
    total_count = static_count + dynamic_count
    if total_count > 0:
        print(f"Static mode: {static_count} batches ({static_count/total_count:.1%})")
        print(f"Dynamic mode: {dynamic_count} batches ({dynamic_count/total_count:.1%})")
    else:
        print("Warning: No batches processed!")
    
    # 返回统计信息
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
