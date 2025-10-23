import math
import sys
from typing import Iterable

import torch
import numpy as np

import util.misc as misc
import util.lr_sched as lr_sched

def train_one_epoch(model: torch.nn.Module,  
                    data_loader: torch.utils.data.DataLoader,  # ✏️ 修改: 只接收一个 DataLoader  
                    optimizer: torch.optim.Optimizer,  
                    device: torch.device,   
                    epoch: int,   
                    loss_scaler,  
                    log_writer=None,  
                    args=None):  
    """  
    训练一个 epoch  
    参数:  
        model: 要训练的模型  
        data_loader: 数据加载器(只有一个,加载连续四帧图片)  
        optimizer: 优化器  
        device: 训练设备  
        epoch: 当前 epoch 编号  
        loss_scaler: 损失缩放器(用于混合精度训练)  
        log_writer: TensorBoard 日志记录器  
        args: 训练参数  
    """  
    model.train(True)  
    metric_logger = misc.MetricLogger(delimiter="  ")  
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))  
    header = 'Epoch: [{}]'.format(epoch)  
    print_freq = 20  
  
    accum_iter = args.accum_iter  
  
    optimizer.zero_grad()  
  
    if log_writer is not None:  
        print('log_dir: {}'.format(log_writer.log_dir))  
  
    # ✏️ 修改: 只使用一个 DataLoader  
    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):  
  
        # 学习率调度(每个迭代而不是每个 epoch)  
        if data_iter_step % accum_iter == 0:  
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)  
          
        # 数据加载器只返回图像张量 [batch_size, 4, 3, 224, 224]  
        samples = samples.to(device, non_blocking=True)  
  
        # 前向传播  
        with torch.cuda.amp.autocast():  
            # ✏️ 修改: 不传递 sensor_type,使用 data_type=0 表示视频数据  
            loss, _, _ = model(samples, data_type=1)  
  
        loss_value = loss.item()  
  
        # 检查损失是否有效  
        if not math.isfinite(loss_value):  
            print("Loss is {}, stopping training".format(loss_value))  
            sys.exit(1)  
  
        # 梯度累积  
        loss /= accum_iter  
        loss_scaler(loss, optimizer, parameters=model.parameters(),  
                    update_grad=(data_iter_step + 1) % accum_iter == 0)  
        if (data_iter_step + 1) % accum_iter == 0:  
            optimizer.zero_grad()  
  
        torch.cuda.synchronize()  
  
        # 更新指标  
        metric_logger.update(loss=loss_value)  
        lr = optimizer.param_groups[0]["lr"]  
        metric_logger.update(lr=lr)  
  
        # TensorBoard 日志记录  
        loss_value_reduce = misc.all_reduce_mean(loss_value)  
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:  
            """ 使用 epoch_1000x 作为 tensorboard 的 x 轴  
            这样可以在批次大小改变时校准不同的曲线  
            """  
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)  
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)  
            log_writer.add_scalar('lr', lr, epoch_1000x)  
  
    # 同步所有进程的统计信息  
    metric_logger.synchronize_between_processes()  
    print("Averaged stats:", metric_logger)  
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}    